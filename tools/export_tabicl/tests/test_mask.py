"""The masked (ROCm) contract: padded + masked == the positional CPU forward, and every control is caught.

Random weights at the fixture config (offline), drawn so the output responds to the table
(mask_parity.sensitive_model); the same gate runs on the real weights in test_real_weights.py."""

from __future__ import annotations

import pytest
import torch

from export_tabicl import configs, export
from export_tabicl.mask_parity import NEGATIVE_CONTROLS, TOL, check_onnx, compare, sensitive_model
from export_tabicl.tabicl_mask import group_indices

TASKS = ("classification", "regression")


@pytest.fixture(scope="module", params=TASKS)
def model(request):
    return sensitive_model(request.param)


def test_padded_and_masked_equals_positional(model):
    assert compare(model) < TOL


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_every_negative_control_is_caught(model, control):
    with NEGATIVE_CONTROLS[control]():
        assert compare(model) > 10 * TOL, control


@pytest.mark.parametrize("d", range(1, 13))
def test_the_division_free_grouping_is_upstreams_modulo(d):
    """(j + 2**i) % d for every real column, at every width the wrap-around differs (d < 4 wraps twice)."""
    width = 16
    cols = group_indices(width, torch.tensor([d]), size=3)
    for i, c in enumerate(cols):
        j = torch.arange(d)
        assert torch.equal(c[:d], (j + 2 ** i) % d)
        assert torch.equal(c[d:], torch.arange(d, width))   # a padded group reads its own (zero) column


@pytest.mark.parametrize("task", TASKS)
def test_the_masked_graph_in_ort_equals_positional(tmp_path, task):
    import onnx

    cfg = configs.fixture()
    m = sensitive_model(task)
    graph = tmp_path / "mask.onnx"
    export.export_mask_graph(m, graph, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features, example=cfg.example)
    res = check_onnx(graph, m)
    assert res["ok"], res
    proto = onnx.load(str(graph), load_external_data=False)
    assert [i.name for i in proto.graph.input] == ["x", "y", "train_size", "d", "n_rows"]
    ops = {n.op_type for n in proto.graph.node}
    # MIGraphX's parser has no sequence ops; a runtime integer Div/Mod is miscompiled on its GPU target
    assert not ops & {"SplitToSequence", "SequenceAt", "Einsum", "Unique", "NonZero", "Mod"}, ops
