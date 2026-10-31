"""Mitra-v2 (`mitra-v2`): the 1,000-bin regressor exported raw-in / raw-out.

The v2 classifier needs no export (its safetensors header is byte-identical to v1's); these tests
cover the regressor's BinnedRegressionWrapper and the checkpoint-signature claims.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct

import numpy as np
import pytest
import torch

from export_mitra import configs, export

REPO = pathlib.Path(__file__).resolve().parents[3]
CACHE = pathlib.Path.home() / ".cache/anofox-tabfm"
CLF1 = CACHE / "autogluon__mitra-classifier@main/model.safetensors"
CLF2 = CACHE / "autogluon__mitra-classifier-2@main/model.safetensors"
REG2 = CACHE / "autogluon__mitra-regressor-2@main/model.safetensors"


def _wrapper(seed=0):
    cfg = configs.get("fixture2", "regression")
    model = export.build_model("regression", cfg.model_kwargs, seed=seed)
    return export.wrapper_for(model, cfg.arch)


def _feed(y_ctx, t=20, h=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g)
    y = torch.zeros(1, t)
    y[0, : len(y_ctx)] = torch.tensor(y_ctx, dtype=torch.float32)
    return x, y, torch.tensor([len(y_ctx)]), torch.tensor([h])


def test_fixture2_exports_with_parity(tmp_path):
    cfg = configs.get("fixture2", "regression")
    model = export.build_model("regression", cfg.model_kwargs, seed=0)
    gp = tmp_path / "g.onnx"
    wrapper = export.export_graph(model, gp, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features,
                                  example=cfg.example, arch=cfg.arch)
    assert isinstance(wrapper, export.BinnedRegressionWrapper)
    tmap = export.postprocess(gp, dict(model.state_dict()))
    # the bin grid is code-generated and stays inline, out of the map
    assert not any("bin_" in k for k in tmap["initializers"])
    parity = export.check_parity(gp, model, wrapper, cfg.parity_shapes)
    assert parity["ok"], parity


def test_predictions_stay_inside_the_bin_grid():
    w = _wrapper()
    y_ctx = [3.0, 7.0, 5.0, 4.0, 6.5, 3.5, 6.0, 4.5, 5.5, 3.2, 6.8, 5.1]
    with torch.no_grad():
        out = w(*_feed(y_ctx))[0, len(y_ctx):, 0]
    lo, hi = min(y_ctx), max(y_ctx)
    assert torch.all(out >= lo - 0.5 * (hi - lo)) and torch.all(out <= hi + 0.5 * (hi - lo))


def test_affine_equivariance_and_query_labels_ignored():
    w = _wrapper()
    y_ctx = [3.0, 7.0, 5.0, 4.0, 6.5, 3.5, 6.0, 4.5, 5.5, 3.2, 6.8, 5.1]
    x, y, ts, d = _feed(y_ctx)
    y2 = y.clone(); y2[0, ts:] = float("nan")  # query labels: never read
    with torch.no_grad():
        a = w(x, y, ts, d)[0, ts:, 0]
        b = w(x, y * 1000.0 + 5000.0, ts, d)[0, ts:, 0]
        c = w(x, y2, ts, d)[0, ts:, 0]
    assert torch.allclose(b, a * 1000.0 + 5000.0, rtol=1e-4, atol=1e-2)
    assert torch.equal(a, c)


def test_constant_target_predicts_the_constant_region():
    w = _wrapper()
    y_ctx = [4.2] * 12
    with torch.no_grad():
        out = w(*_feed(y_ctx))[0, 12:, 0]
    assert torch.isfinite(out).all()
    assert torch.all((out - 4.2).abs() <= 1.5)  # range guarded to 1: grid spans [3.7, 5.7]


def test_binning_matches_torch_bucketize():
    """count(edges < v) - 1, clamped, is upstream's bucketize(v, bins) - 1."""
    w = _wrapper()
    edges = w.get_buffer("bin_edges")
    v = torch.linspace(-0.7, 1.7, 997)
    mine = torch.clamp((edges[None, :] < v[:, None]).sum(-1) - 1, 0, edges.numel() - 2)
    theirs = torch.clamp(torch.bucketize(v, edges) - 1, 0, edges.numel() - 2)
    assert torch.equal(mine, theirs)


def _header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)


def test_classifier2_header_is_classifier1s():
    if not (CLF1.exists() and CLF2.exists()):
        pytest.skip("mitra classifier weights not cached")
    assert _header(CLF1) == _header(CLF2)
    assert hashlib.sha256(_header(CLF2)).hexdigest() == (
        "cb1a261bb3d9ca505e0db66e21df85bec6777f7e104247d4a71a8eb8d8b3b96a")


def test_regressor2_matches_real2_config_and_committed_map():
    if not REG2.exists():
        pytest.skip("mitra regressor-2 weights not cached")
    raw = _header(REG2)
    header = {k: v for k, v in json.loads(raw).items() if k != "__metadata__"}
    cfg = configs.get("real2", "regression")
    model = export.build_model("regression", cfg.model_kwargs)
    sd = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert sd == {k: tuple(v["shape"]) for k, v in header.items()}
    assert sd["final_layer.weight"] == (1000, 512) and sd["y_embedding.y_embedding.weight"] == (1000, 512)
    tmap = json.loads((REPO / "resources/tensor_map_mitra2_regression.json").read_text())
    assert set(tmap["initializers"].values()) == set(header)
    assert hashlib.sha256(raw).hexdigest() == (
        "ff7288fd5eda3910b21863813683614eaddcceb60ef9e60be4eb79bf9c69c7cb")
