"""The split export: two graphs, chained, must equal one forward pass."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
import torch

from export_tabicl import configs, split
from export_tabicl.tabicl_patches import build_model

# Same budget the single-graph export is held to in test_export.py.
BUDGET = 1e-3


def _export(tmp_path, task="classification"):
    cfg = configs.fixture()
    model = build_model(task, cfg.model_kwargs, seed=0)
    prep_p = tmp_path / f"graph_prepare_{task}.onnx"
    qry_p = tmp_path / f"graph_query_{task}.onnx"
    split.export_split(model, prep_p, qry_p, dim_train=cfg.dim_train,
                       dim_features=cfg.dim_features, example=cfg.example)
    return model, prep_p, qry_p


def test_split_graphs_match_a_single_forward(tmp_path):
    """Shapes deliberately differ from the export example, so a baked dim fails rather than passes."""
    model, prep_p, qry_p = _export(tmp_path)
    worst = split.check_split_parity(prep_p, qry_p, model,
                                     shapes=((24, 9, 7), (31, 5, 7), (18, 13, 7)))
    assert worst < BUDGET, f"chained graphs drifted from the single pass: {worst:.2e}"


def test_prepare_output_is_reusable_across_query_batches(tmp_path):
    """One prepared context, two different query batches -- the point of the whole exercise."""
    model, prep_p, qry_p = _export(tmp_path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    s_prep = ort.InferenceSession(str(prep_p), opts, providers=["CPUExecutionProvider"])
    s_qry = ort.InferenceSession(str(qry_p), opts, providers=["CPUExecutionProvider"])
    names = [o.name for o in s_prep.get_outputs()]

    S, H = 26, 7
    xs = torch.randn(1, S, H)
    y = torch.randint(0, model.max_classes, (1, S)).float()
    prepared = s_prep.run(None, {"x": xs.numpy(), "y": y.numpy()})

    xq = torch.randn(1, 12, H)
    whole = s_qry.run(None, {"x": xq.numpy(), "y": y.numpy(), **dict(zip(names, prepared))})[0]
    a = s_qry.run(None, {"x": xq[:, :5].numpy(), "y": y.numpy(), **dict(zip(names, prepared))})[0]
    b = s_qry.run(None, {"x": xq[:, 5:].numpy(), "y": y.numpy(), **dict(zip(names, prepared))})[0]

    assert np.abs(whole - np.concatenate([a, b], axis=1)).max() < BUDGET, \
        "the same context gave different answers depending on how the queries were batched"


def test_context_tensor_count_matches_the_column_blocks(tmp_path):
    model, prep_p, _ = _export(tmp_path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(prep_p), opts, providers=["CPUExecutionProvider"])
    hidden = [o.name for o in sess.get_outputs() if o.name.startswith("hidden_")]
    assert len(hidden) == split.n_context_tensors(model)
    assert [o.name for o in sess.get_outputs()][-1] == "support_reps"


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_both_tasks_export(tmp_path, task):
    model, prep_p, qry_p = _export(tmp_path, task)
    worst = split.check_split_parity(prep_p, qry_p, model, shapes=((22, 7, 7),))
    assert worst < BUDGET, f"{task}: {worst:.2e}"
