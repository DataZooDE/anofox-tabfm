"""The masked (ROCm) contract: padded + masked == the positional CPU forward, and every control is caught.

Random weights at the fixture config (offline), woken up as for the fixture so masking errors are visible;
the same gate runs on the real weights in test_real_weights.py."""

from __future__ import annotations

import pytest
import torch

from export_limix import configs, export, fixture
from export_limix.limix_mask import embed_y, encode_x
from export_limix.mask_parity import NEGATIVE_CONTROLS, TOL, check_onnx, compare

TASKS = ("classification", "regression")


@pytest.fixture(scope="module")
def model():
    return fixture.seeded_model()


@pytest.mark.parametrize("task", TASKS)
def test_padded_and_masked_equals_positional(model, task):
    assert compare(model, task) < TOL


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_every_negative_control_is_caught(model, task, control):
    with NEGATIVE_CONTROLS[control]():
        assert compare(model, task) > 10 * TOL, control


def test_the_migraphx_safe_tokenizer_is_upstreams(model):
    """No integer floor-division / modulo and no split: the same embedding, at decades, zero and NaN."""
    g = torch.Generator().manual_seed(0)
    d = torch.randn(1, 30, 4, 2, generator=g) * torch.tensor([1e-6, 1e-3, 1, 1e3, 0, 10, 100, 0.1]).view(1, 1, 4, 2)
    d[0, 3, 1, 0] = float("nan")
    d[0, 6, 2, 1] = 1000.0
    d[0, 7, 0, 0] = -0.1
    with torch.no_grad():
        ref = model.encoder_x({"data": d.clone(), "nan_encoding": torch.zeros_like(d)})["data"]
        assert torch.allclose(ref, encode_x(model.encoder_x[0], d.clone()), atol=1e-5)


@pytest.mark.parametrize("labels", [[0, 1, 2, 1, 0], [0, 2, 2, 0], [2, 2, 2], [1, 0, 2], [1, 1, 2]])
def test_the_class_rank_is_upstreams_unique_rank(model, labels):
    """The presence table over the class ids == upstream's MulticlassTargetEncoder (torch.unique) followed by
    EmbYEncoderStep, on gaps, a single class and out-of-order ids; padded rows get the mask embedding.

    The table covers ids 0..K-1 (K = max_num_classes: 3 here, 10 on the released model), which is the engine's
    contract: it feeds dense class ids and caps the class count at the model's max_classes."""
    s, pad = len(labels), 4
    y = torch.tensor(labels, dtype=torch.float32)
    with torch.no_grad():
        steps = model.cls_y_encoder
        y_up = torch.cat((y, torch.full((s + pad,), float("nan")))).view(1, -1, 1)   # [context ; all rows]
        state = {"data": y_up, "eval_pos": s}
        for step in steps:
            state = step(state)
        out = state["data"].float().reshape(2 * s + pad, -1)
        ref, mask_row = out[:s], out[s]                                               # context rows, a query row
        y_ctx = torch.cat((y, torch.full((pad,), float("nan"))))
        ctx = torch.arange(s + pad) < s
        got = embed_y(model, "cls", y_ctx, ctx)[0]                                    # [2(s+pad), E]
    assert torch.equal(got[:s], ref)
    assert torch.equal(got[s + pad], mask_row)


@pytest.mark.parametrize("task", TASKS)
def test_the_masked_graph_in_ort_equals_positional(tmp_path, model, task):
    import onnx

    cfg = configs.fixture()
    graph = tmp_path / "mask.onnx"
    export.export_mask_graph(model, graph, task=task, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features,
                             example=cfg.example, max_classes=cfg.max_classes)
    res = check_onnx(graph, model, task)
    assert res["ok"], res
    proto = onnx.load(str(graph), load_external_data=False)
    assert [i.name for i in proto.graph.input] == ["x", "y", "train_size", "d", "n_rows"]
    ops = {n.op_type for n in proto.graph.node}
    assert not ops & {"SplitToSequence", "SequenceAt", "Einsum", "Unique", "NonZero", "Mod"}, ops
