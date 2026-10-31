"""EXAONE Tabular exporter: the re-expressed forward is upstream's forward, and the graph is the wrapper.

Random weights at the fixture config (every structural flag of the released models, tiny dims), offline.
The real-weight contracts live in test_real_weights.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from export_exaone import configs, export
from export_exaone.exaone_patches import ExportWrapper, build_model, label_encoder, model_forward, upstream_forward

TASKS = ("classification", "regression")


def _model(task, seed=3):
    cfg = configs.get("fixture", task)
    return build_model(task, cfg.model_config, cfg.regression_config, seed=seed)


def _labels(task, s, gen):
    if task == "classification":
        return torch.randint(0, 3, (1, s), generator=gen).float()
    return torch.randn(1, s, generator=gen)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("t,h,s", [(30, 7, 20), (12, 1, 5), (9, 4, 2)])
def test_the_export_forward_is_upstreams_forward(task, t, h, s):
    """model_forward (no validation, no cache, plain attention, unique-free label ranks) = upstream's
    model.forward with one member, at several shapes including one feature and S = 2."""
    m = _model(task)
    gen = torch.Generator().manual_seed(t * 100 + h * 10 + s)
    x, y = torch.randn(1, t, h, generator=gen), _labels(task, s, gen)
    with torch.no_grad():
        ours = model_forward(m, x, y, s, regression=task == "regression")
    ref = upstream_forward(m, x, y, s)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=2e-5, rtol=1e-5), (ours - ref).abs().max()


def test_the_unique_free_label_rank_equals_upstreams_label_encoder():
    """Duplicates, gaps, unsorted support labels and NaN query placeholders: same embedding as upstream's
    torch.unique-based LabelEncoder."""
    enc = _model("classification").label_encoder
    labels = torch.tensor([3.0, 7.0, 3.0, 0.0, 7.0, 9.0, float("nan"), float("nan")])
    ours = label_encoder(enc, labels, 6, regression=False)
    with torch.no_grad():
        ref = enc(labels.view(-1, 1, 1), 6)[:, 0]
    assert torch.allclose(ours, ref, atol=1e-6)


def test_a_constant_column_is_dropped_exactly_as_upstream_drops_it():
    """The feature encoder zeroes a column constant over ALL rows (support and query); the export keeps
    that transductive rule, so a constant column and its absence give the same encoding."""
    m = _model("regression")
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, 20, 3, generator=gen)
    x[0, :, 1] = 4.0
    y = torch.randn(1, 12, generator=gen)
    with torch.no_grad():
        ours = model_forward(m, x, y, 12, regression=True)
    assert torch.allclose(ours, upstream_forward(m, x, y, 12), atol=2e-5)


@pytest.mark.parametrize("task", TASKS)
def test_context_rows_are_the_context_presented_again_as_queries(task):
    m = _model(task)
    gen = torch.Generator().manual_seed(1)
    x, y = torch.randn(1, 20, 5, generator=gen), _labels(task, 12, gen)
    w = ExportWrapper(m, task)
    with torch.no_grad():
        full = w(x, y)[0]
        again = w(torch.cat([x[:, :12], x[:, :12]], dim=1), y)[0, 12:]
    assert torch.allclose(full[:12], again, atol=1e-5)
    assert full[:12].std(dim=0).max() > 0


def test_the_regression_wrapper_is_upstreams_regressor_point_for_one_member():
    """float32 population-variance standardisation, mean of the quantile bank, inverse."""
    m = _model("regression")
    rng = np.random.default_rng(0)
    xtr, xte = rng.standard_normal((25, 4)), rng.standard_normal((9, 4))
    ytr = rng.standard_normal(25) * 40 + 150
    x = torch.from_numpy(np.vstack([xtr, xte]).astype(np.float32))[None]
    with torch.no_grad():
        ours = ExportWrapper(m, "regression")(x, torch.from_numpy(ytr.astype(np.float32))[None])[0, 25:, 0]
    y32 = torch.as_tensor(ytr, dtype=torch.float32)
    c, s = y32.mean(), (y32 - y32.mean()).square().mean().sqrt()
    ref = upstream_forward(m, x, ((y32 - c) / s)[None], 25)[0].mean(-1) * s + c
    assert torch.allclose(ours, ref, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("task", TASKS)
def test_the_graph_is_dynamic_and_weight_free(tmp_path, task):
    cfg = configs.get("fixture", task)
    m = _model(task)
    graph = tmp_path / "g.onnx"
    w = export.export_graph(m, graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                            dim_features=cfg.dim_features, example=cfg.example, task=task)
    tensor_map = export.postprocess(graph, dict(m.state_dict()))
    parity = export.check_parity(graph, w, ((40, 7, 30), (9, 1, 5), (33, 10, 21)), task=task)
    assert parity["ok"], parity
    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tensor_map)
    assert set(tensor_map["initializers"].values()) == set(m.state_dict())


def test_the_graph_has_no_flash_attention_or_data_dependent_ops(tmp_path):
    """The plain attention path: the graph carries Softmax and MatMul and none of Unique / NonZero /
    data-dependent compress ops that a fixed-shape backend cannot compile."""
    import onnx

    cfg = configs.get("fixture", "classification")
    m = _model("classification")
    graph = tmp_path / "g.onnx"
    export.export_graph(m, graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                        dim_features=cfg.dim_features, example=cfg.example, task="classification")
    ops = {n.op_type for n in onnx.load(str(graph), load_external_data=False).graph.node}
    assert {"Softmax", "MatMul"} <= ops
    assert not ops & {"Unique", "NonZero", "Compress", "Loop", "If"}, ops & {"Unique", "NonZero", "Compress",
                                                                            "Loop", "If"}
