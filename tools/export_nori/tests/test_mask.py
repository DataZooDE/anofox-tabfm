"""The masked (ROCm) contract: padded + masked == the positional CPU forward, and every control is caught.

Random weights at the fixture config (offline), woken up as for the fixture so masking errors are visible;
the same gate runs on the real weights in test_real_weights.py."""

from __future__ import annotations

import pytest
import torch

from export_nori import configs, export, fixture
from export_nori.mask_parity import NEGATIVE_CONTROLS, TOL, check_onnx, compare
from export_nori.nori_mask import encode_x


@pytest.fixture(scope="module")
def model():
    return fixture.seeded_model()


def test_padded_and_masked_equals_positional(model):
    assert compare(model) < TOL


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_every_negative_control_is_caught(model, control):
    with NEGATIVE_CONTROLS[control]():
        assert compare(model) > 10 * TOL, control


def test_the_migraphx_safe_tokenizer_is_upstreams(model):
    """No integer floor-division / modulo and no split: the same embedding, at decades, zero and NaN."""
    g = torch.Generator().manual_seed(0)
    d = torch.randn(1, 30, 4, 2, generator=g) * torch.tensor([1e-6, 1e-3, 1, 1e3, 0, 10, 100, 0.1]).view(1, 1, 4, 2)
    d[0, 3, 1, 0] = float("nan")
    d[0, 6, 2, 1] = 1000.0
    d[0, 7, 0, 0] = -0.1
    with torch.no_grad():
        assert torch.allclose(model.encoder_x({"data": d.clone()})["data"], encode_x(model.encoder_x[0], d.clone()),
                              atol=1e-5)


def test_the_masked_graph_in_ort_equals_positional(tmp_path, model):
    import onnx

    cfg = configs.fixture()
    graph = tmp_path / "mask.onnx"
    export.export_mask_graph(model, graph, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features,
                             example=cfg.example)
    res = check_onnx(graph, model)
    assert res["ok"], res
    proto = onnx.load(str(graph), load_external_data=False)
    assert [i.name for i in proto.graph.input] == ["x", "y", "train_size", "d", "n_rows"]
    ops = {n.op_type for n in proto.graph.node}
    assert not ops & {"SplitToSequence", "SequenceAt", "Einsum", "Unique", "NonZero"}, ops
