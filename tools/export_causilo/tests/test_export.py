"""Does the Causilo export compute what upstream computes, at shapes it was not traced at?

Three independent questions, because each has a failure that the others cannot see:

  1. the IN-GRAPH normaliser vs upstream's numpy `Normalizer("none")`;
  2. the wrapper's network vs upstream's own `ModelRunner.predict`;
  3. the exported ONNX graph vs the wrapper, in ONNX Runtime, at shapes different from
     the trace, over EVERY row (context and query separately).

A graph can pass 3 and be wrong, if the wrapper is wrong (1, 2), or pass 1-2 and be wrong
once exported: the trace bakes a Python value in (the `math.log(key_count)` trap), which
a random-weight model only shows at a shape other than the one it was traced at.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
import torch

from causilo.data.normalization import Normalizer
from causilo.execution.runner import ModelRunner

from export_causilo import configs, export
from export_causilo.causilo_patches import (apply, build_model, normalise_features, predict,
                                            predict_all_rows)
from export_causilo.fixture import sensitive, seeded_model

TOL = 1e-4   # float32 vs upstream's float64 normaliser, and summation order


def _table(rng, t, h, s, *, nan=False, const=False, outliers=False):
    x = rng.standard_normal((t, h)) * rng.uniform(0.5, 5, h) + rng.uniform(-3, 3, h)
    if outliers:
        x[2, 0], x[5, min(1, h - 1)] = 80.0, -60.0
    if const:
        x[:, h - 1] = 3.5
    if nan:
        x[3, 0] = np.nan
        x[7, h // 2] = np.nan
        col = min(1, h - 1)
        x[:s, col] = np.where(np.arange(s) % 3 == 0, np.nan, x[:s, col])
    return x


# --- 1. the normaliser --------------------------------------------------------------------

NORMALISER_CASES = [
    (40, 7, 30, {}),
    (40, 7, 30, dict(nan=True)),
    (30, 6, 20, dict(outliers=True)),       # the 4-sigma second pass and the arcsinh tails
    (25, 5, 14, dict(const=True)),          # a constant column: spread at rounding level -> 1
    (12, 4, 2, {}),                         # two training rows
    (9, 3, 1, {}),                          # ONE training row: ddof 0, std floor
    (60, 9, 45, dict(nan=True, outliers=True)),
]


@pytest.mark.parametrize("t,h,s,kw", NORMALISER_CASES)
def test_normaliser_matches_upstream(t, h, s, kw):
    x = _table(np.random.default_rng(0), t, h, s, **kw)
    want = Normalizer.fit(x[:s].copy(), "none").transform(x)
    got = normalise_features(torch.from_numpy(x.astype(np.float32))[None], torch.zeros(1, s)).numpy()[0]
    assert (np.isnan(want) == np.isnan(got)).all(), "NaN must be restored where upstream restores it"
    finite = ~np.isnan(want)
    assert np.abs(want[finite] - got[finite]).max() < TOL


def test_normaliser_test_can_fail():
    """A normaliser that statistics over ALL rows instead of the training prefix is caught."""
    x = _table(np.random.default_rng(1), 40, 6, 12, outliers=True)
    want = Normalizer.fit(x[:12].copy(), "none").transform(x)
    wrong = normalise_features(torch.from_numpy(x.astype(np.float32))[None], torch.zeros(1, 40)).numpy()[0]
    assert np.abs(want - wrong).max() > 10 * TOL


# --- 2. the network ------------------------------------------------------------------------

@pytest.mark.parametrize("config", ["fixture", "real"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("t,h,s", [(20, 5, 12), (33, 8, 10), (18, 3, 17)])
def test_network_matches_upstream_runner(config, task, t, h, s):
    apply()
    # sensitive(): a default-initialised model ignores its input, so this comparison would pass
    # for a wrapper that got the network wrong
    model = sensitive(build_model(task, configs.get(config)))
    g = torch.Generator().manual_seed(7)
    table = torch.randn(1, t, h, generator=g)
    targets = torch.randint(0, 3, (1, s), generator=g).float() if task == "classification" \
        else torch.randn(1, s, generator=g)
    with torch.no_grad():
        ours = predict(model, table, targets)
        want = ModelRunner(model).predict(table, targets)
    assert ours.shape == want.shape == (1, t - s, model.config.outputs)
    assert (ours - want).abs().max().item() < TOL


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_every_row_is_a_prediction_and_query_rows_are_unchanged(task):
    """The duplicate route gives all T rows, and does not move a query row.

    Why it is the route: on real weights the cheap alternative (last layer over all rows)
    scored fitted R2 0.50 on a clean linear problem where the query R2 was 0.999. That
    accuracy claim needs real weights (parity_real.py); what random weights CAN pin is the
    structure: T outputs, and query rows equal to the single pass.
    """
    apply()
    model = seeded_model(task)
    g = torch.Generator().manual_seed(3)
    t, h, s = 26, 6, 15
    table = torch.randn(1, t, h, generator=g)
    targets = torch.randint(0, 3, (1, s), generator=g).float() if task == "classification" \
        else torch.randn(1, s, generator=g)
    with torch.no_grad():
        every = predict_all_rows(model, table, targets)
        single = predict(model, table, targets)
    assert every.shape == (1, t, model.config.outputs)
    assert (every[:, s:] - single).abs().max().item() < TOL
    # and the context rows are not the query-row pad: they differ from each other
    assert every[0, :s].std(dim=0).max().item() > 1e-4


# --- 3. the exported graph -----------------------------------------------------------------

@pytest.fixture(scope="module", params=["classification", "regression"])
def exported(request, tmp_path_factory):
    task = request.param
    cfg = configs.get("fixture")
    model = seeded_model(task)        # responsive AND well-conditioned; see fixture.py
    path = tmp_path_factory.mktemp(task) / "g.onnx"
    classes = cfg.classes if task == "classification" else 0
    wrapper = export.export_graph(model, path, example=cfg.example, classes=classes)
    return task, cfg, classes, path, wrapper, model


def test_graph_matches_wrapper_on_every_row_at_other_shapes(exported):
    task, cfg, classes, path, wrapper, _ = exported
    # H=7 and H=4 need feature padding (group of 3), H=9 does not; all differ from the trace
    r = export.check_parity(path, wrapper, cfg.parity_shapes, classes)
    assert r["ok"], r
    for shape in r["shapes"]:
        assert shape["max_abs_delta_context_rows"] < export.PARITY_TOL
        assert shape["max_abs_delta_query_rows"] < export.PARITY_TOL


@pytest.mark.parametrize("t,h,s", [(6, 1, 3), (8, 2, 8), (10, 3, 2), (5, 14, 4)])
def test_graph_edge_shapes(exported, t, h, s):
    """One feature, no query rows at all (T == S), a minimal context, a wide table."""
    task, cfg, classes, path, wrapper, _ = exported
    r = export.check_parity(path, wrapper, ((t, h, s),), classes)
    assert r["ok"], r


def test_graph_has_no_onehot(exported):
    """ONNX OneHot is not in the MLX interpreter's op table; equality against arange is."""
    _, _, _, path, _, _ = exported
    proto = onnx.load(str(path), load_external_data=False)
    assert "OneHot" not in {n.op_type for n in proto.graph.node}


def test_graph_is_weight_free_after_postprocess(exported):
    _, _, _, path, _, model = exported
    tensor_map = export.postprocess(path, dict(model.state_dict()))
    export.delete_weight_data(path)
    export.assert_weight_free(path, tensor_map)
    assert len(tensor_map["initializers"]) > 50


def test_parity_at_other_shapes_catches_a_baked_length(tmp_path):
    """Put upstream's `math.log(key_count)` back and parity at another shape must go red.

    This is the one guard for that trap, so it has to be shown to work. The trace evaluates
    `math.log` in Python and records a CONSTANT (there is no Log node for a structural check
    to find), and the eager wrapper, evaluated at the new shape, is right while the graph is
    not. So the graph is only checkable by running it at a shape other than the one it was
    traced at, which is what check_parity does. A parity test at the trace shape alone passes.
    """
    import math
    from causilo.nn.layers import scaling

    apply()
    patched = scaling.QueryScale.forward

    def upstream(self, query, key_count):
        log_count = query.new_tensor([[math.log(max(key_count, 1))]])
        length_gain = scaling.bounded_multiplier(self.length(log_count), 8.0)
        length_gain = length_gain * scaling.bounded_multiplier(self.correction(log_count), 2.0)
        return query * (length_gain.reshape(self.heads, 1, self.head_width)
                        * scaling.bounded_multiplier(self.content(query), 2.0))

    scaling.QueryScale.forward = upstream
    try:
        cfg = configs.get("fixture")
        model = seeded_model("classification")
        path = tmp_path / "bad.onnx"
        wrapper = export.export_graph(model, path, example=cfg.example, classes=cfg.classes)
        at_trace = export.check_parity(path, wrapper, (cfg.example,), cfg.classes)
        elsewhere = export.check_parity(path, wrapper, cfg.parity_shapes, cfg.classes)
        assert at_trace["ok"], "sanity: a baked length is invisible at the traced shape"
        assert not elsewhere["ok"] and elsewhere["worst"] > 10 * export.PARITY_TOL, elsewhere
    finally:
        scaling.QueryScale.forward = patched
