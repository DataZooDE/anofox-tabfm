"""The masked (ROCm) contract: padded + masked == the positional CPU forward, and every control is caught.

Random weights at the fixture config (offline); the same gate runs on the real weights in
test_real_weights.py. A pass only counts because each negative control (one mask broken) fails it.
"""

from __future__ import annotations

import pytest
import torch

from export_exaone import configs, export
from export_exaone.exaone_patches import build_model
from export_exaone.mask_parity import NEGATIVE_CONTROLS, TOL, check_onnx, compare


def _model(task):
    cfg = configs.get("fixture", task)
    m = build_model(task, cfg.model_config, cfg.regression_config, seed=5)
    with torch.no_grad():  # wake the attention so masking errors are visible on random weights
        for name, p in m.named_parameters():
            if name.endswith("output_weight"):
                p.mul_(4.0)
    return m.eval()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_padded_and_masked_equals_positional(task):
    assert compare(_model(task), task) < TOL[task]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_every_negative_control_is_caught(task, control):
    m = _model(task)
    with NEGATIVE_CONTROLS[control]():
        assert compare(m, task) > 10 * TOL[task], control


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_the_masked_graph_in_ort_equals_positional(tmp_path, task):
    import onnx

    cfg = configs.get("fixture", task)
    m = _model(task)
    graph = tmp_path / "mask.onnx"
    export.export_mask_graph(m, graph, task=task, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features,
                             example=cfg.example)
    res = check_onnx(graph, m, task)
    assert res["ok"], res
    proto = onnx.load(str(graph), load_external_data=False)
    assert [i.name for i in proto.graph.input] == ["x", "y", "train_size", "d", "n_rows"]
    ops = {n.op_type for n in proto.graph.node}
    assert not ops & {"SplitToSequence", "SequenceAt", "Einsum", "Unique", "NonZero"}, ops
