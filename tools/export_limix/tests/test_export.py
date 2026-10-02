"""LimiX export: is the traced graph right at shapes it was NOT traced at?

The spike's README documents exactly where the first export failed. Those three shapes are the
red test: each is a (T, H, S) at which ONNX Runtime must agree with PyTorch, traced at one
shape (T=20, H=6, S=12). A graph that only works at its trace shape bakes a Python value in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from export_limix import configs, export
from export_limix.limix_patches import build_model

TRACE = (20, 6, 12)                      # (T, H, S): the shape the README traces at
#: The README's three shapes that failed, verbatim: (T, H, S).
README_FAILING = [(20, 6, 15), (30, 6, 12), (40, 8, 30)]


def _export(tmp_path, task):
    cfg = configs.fixture()
    model = build_model(cfg.model_config, seed=0)
    path = tmp_path / f"g_{task}.onnx"
    wrapper = export.export_graph(model, path, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                                  dim_features=cfg.dim_features, example=TRACE,
                                  max_classes=cfg.max_classes, task=task)
    return cfg, path, wrapper


@pytest.fixture(scope="module", params=["classification", "regression"])
def exported(request, tmp_path_factory):
    return (request.param, *_export(tmp_path_factory.mktemp(request.param), request.param))


@pytest.mark.parametrize("t,h,s", README_FAILING)
def test_graph_runs_and_matches_pytorch_at_a_shape_it_was_not_traced_at(exported, t, h, s):
    task, cfg, path, wrapper = exported
    r = export.check_parity(path, wrapper, ((t, h, s),), max_classes=cfg.max_classes, task=task)
    assert r["ok"], r


#: The README's shapes all have an EVEN H. LimiX groups features in twos and pads with
#: `num_features % features_per_group`, a Python modulo on a symbolic size: traced at H=6 it can only
#: record "no padding needed". Odd widths are the common case in practice.
ODD_WIDTHS = [(20, 5, 12), (24, 7, 10), (33, 9, 21)]


@pytest.mark.parametrize("t,h,s", ODD_WIDTHS)
def test_odd_feature_widths_need_the_group_padding(exported, t, h, s):
    task, cfg, path, wrapper = exported
    r = export.check_parity(path, wrapper, ((t, h, s),), max_classes=cfg.max_classes, task=task)
    assert r["ok"], r


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_replacement_y_decoder_equals_upstreams_on_a_uniform_y_type(task):
    """Patch 5 claims the split is the identity when y_type is uniform. Prove it, don't assert it."""
    cfg = configs.fixture()
    model = build_model(cfg.model_config, seed=3)
    model._export_task = "cls" if task == "classification" else "reg"
    g = torch.Generator().manual_seed(0)
    out = torch.randn(1, 9, cfg.model_config["embed_dim"], generator=g)
    y_type = (torch.zeros if task == "classification" else torch.ones)(1, 9, 1)
    up_cls, up_reg = type(model)._upstream_y_decoder(model, out, y_type)
    new_cls, new_reg = model.y_decoder(out, y_type)
    want, got = (up_cls, new_cls) if task == "classification" else (up_reg, new_reg)
    assert torch.equal(want, got)


# --- the engine decodes EVERY row, so every row must be a real prediction -------------------------

def _wrapper(task, seed=0):
    from export_limix.limix_patches import ExportWrapper
    cfg = configs.fixture()
    return cfg, ExportWrapper(build_model(cfg.model_config, seed=seed), task=task).eval()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_context_rows_are_not_a_zero_pad(task):
    """The wrapper used to return zeros for rows < S. The engine surfaces those rows as is_training
    fitted values, so every one decoded to the same answer (the defect #51 fixed for other families)."""
    cfg, w = _wrapper(task)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 24, 6, generator=g)
    y = (torch.randint(0, 3, (1, 14), generator=g).float() if task == "classification"
         else torch.randn(1, 14, generator=g) * 3 + 7)
    with torch.no_grad():
        out = w(x, y)
    assert out.shape[1] == 24
    assert out[:, :14].abs().max().item() > 0.0, "context rows are an all-zero pad"


def test_regression_is_raw_in_raw_out():
    """The model never normalises its target: upstream's own examples z-score y before the call and
    invert after. The engine feeds RAW targets and reads the output as raw, so the graph must do both.
    Equivariance is the property: predict(a*y + b) == a*predict(y) + b."""
    cfg, w = _wrapper("regression")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 20, 6, generator=g)
    y = torch.randn(1, 12, generator=g)
    a, b = 40.0, 150.0
    with torch.no_grad():
        base = w(x, y)
        scaled = w(x, y * a + b)
    assert (scaled - (base * a + b)).abs().max().item() < 1e-3 * (abs(a) * base.abs().max().item() + abs(b))
