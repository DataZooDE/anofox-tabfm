"""Which architectures take which fitted-values route, and why it matters.

`ExportWrapper._all_row_logits` produces the in-context fitted values the engine
surfaces as `is_training` rows. Two routes exist:

  * cheap    -- project the model's own `train_embeddings` with its head;
  * duplicate -- re-present the training rows as queries ([train; train; test]),
                 costing a sequence of N+T instead of T (~1.5x wall-clock).

The split is NOT a style choice. Measured on real weights, three separable
classes, where every architecture's QUERY rows score 1.00:

         cheap   duplicate
    v2    0.35     1.00
    v2.5  0.68     1.00
    v2.6  1.00     1.00
    v3    1.00     1.00

v2 and v2.5 must take the duplicate route or they emit confident nonsense --
worse than the zero pad this replaced, because it looks like an answer. v2.6 and
v3 classification must NOT, or every user pays 1.5x for values already correct.

REGRESSION always takes the duplicate route. Classification reads an argmax,
which absorbs small logit error (v2.6: the routes agree to 0.0018 of probability
and never disagree on the class). Regression decodes the bar distribution to a
mean, which does not absorb it -- v2.6 fitted values correlate 0.79 with the
target through the cheap route and 1.00 through the duplicate one.

These tests pin the routing itself rather than the accuracy: the numbers above
need real checkpoints, which CI does not have, but the DISPATCH is checkable on
random weights and is what silently rots when upstream adds an architecture.
"""
import pytest
import torch

from export_tabpfn import configs
from export_tabpfn.tabpfn_patched import (ExportWrapper, apply_module_patches,
                                          build_random_model, prepare_model_for_export)

# (arch, config, duplicate-route for classification?) -- regression is always
# duplicate, so it is parametrised separately below.
CASES = [("v2", "fixture", True), ("v2.5", "fixture25", True),
         ("v2.6", "fixture26", False), ("v3", "fixture3", False)]


def _model(arch, cfgname):
    cfg = configs.get(cfgname, task="classification")
    apply_module_patches(cfg.arch)
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    m = build_random_model("classification", kw, seed=0, arch=cfg.arch)
    return prepare_model_for_export(m, arch=cfg.arch, allow_synthetic_pos_base=True)


@pytest.mark.parametrize("arch,cfgname,wants_duplicate", CASES)
def test_route_dispatch(arch, cfgname, wants_duplicate):
    """The wrapper sends each architecture down the route measured correct for it."""
    m = _model(arch, cfgname)
    seen = {}
    original = m.forward

    def spy(x, y, *a, **kw):
        seen["rows"] = x.shape[0]
        return original(x, y, *a, **kw)

    m.forward = spy
    T, N, H = 20, 12, 5
    ExportWrapper(m, "classification").eval()(
        torch.randn(1, T, H), torch.randint(0, 2, (1, N)).float())

    # The duplicate route feeds N extra rows; the cheap one feeds exactly T.
    assert seen["rows"] == (T + N if wants_duplicate else T), (
        f"{arch} classification took the {'cheap' if seen['rows'] == T else 'duplicate'} "
        f"route; the measured-correct one is {'duplicate' if wants_duplicate else 'cheap'}")


@pytest.mark.parametrize("arch,cfgname,_c", CASES)
def test_regression_always_duplicates(arch, cfgname, _c):
    """Every architecture takes the duplicate route for regression."""
    cfg = configs.get(cfgname, task="regression")
    apply_module_patches(cfg.arch)
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    m = build_random_model("regression", kw, seed=0, arch=cfg.arch)
    m = prepare_model_for_export(m, arch=cfg.arch, allow_synthetic_pos_base=True)
    seen = {}
    original = m.forward

    def spy(x, y, *a, **kw):
        seen["rows"] = x.shape[0]
        return original(x, y, *a, **kw)

    m.forward = spy
    T, N, H = 20, 12, 5
    ExportWrapper(m, "regression").eval()(torch.randn(1, T, H), torch.randn(1, N))
    assert seen["rows"] == T + N, (
        f"{arch} regression took the cheap route; the bar-distribution mean does "
        f"not absorb its error (0.79 vs 1.00 correlation on real weights)")


@pytest.mark.parametrize("arch,cfgname,_wants", CASES)
def test_output_covers_every_row_and_context_is_not_a_pad(arch, cfgname, _wants):
    """Whatever the route, the output is [1,T,C] and context rows are not zeros.

    The defect being fixed was a zero pad over rows < N, which the engine
    surfaced as fitted values: argmax of zeros is one class for every row.

    Deliberately NOT asserting that the context rows differ from each other.
    That is the property that actually matters, but it needs REAL weights: a
    random-init model can legitimately answer one class for every row, and on
    the duplicate route (v2, v2.5) it does. Asserting it here would fail for
    reasons unrelated to the defect. The distinctness and accuracy checks live
    in test/sql/tabfm_real_models.test, which runs under TABFM_REAL_WEIGHTS.
    """
    m = _model(arch, cfgname)
    T, N, H = 20, 12, 5
    out = ExportWrapper(m, "classification").eval()(
        torch.randn(1, T, H), torch.randint(0, 2, (1, N)).float())
    assert out.shape[0] == 1 and out.shape[1] == T
    assert out[0, :N].abs().sum() > 0, f"{arch}: context rows are still a zero pad"
