"""Nori exporter: the patches are upstream's function, and the graph is the wrapper's function.

Random weights at the fixture config (every structural flag of the released model, tiny dims), so this
runs offline in seconds. The real-weight contracts live in test_real_weights.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from export_nori import configs, export
from export_nori.nori_patches import ExportWrapper, build_model, quantile_mean, unpatched


@pytest.fixture(scope="module")
def model():
    return build_model(configs.fixture().model_config, seed=3)


@pytest.mark.parametrize("t,h,s", [(30, 7, 20), (12, 4, 5), (9, 1, 2)])
def test_the_patched_forward_is_upstreams_forward(model, t, h, s):
    """Patch 1-4 are each the identical function on the inference path: the patched model and upstream's
    own forward (every patch reverted) agree bit for bit, at odd and even widths and at S = 2."""
    torch.manual_seed(t + h + s)
    x, y = torch.randn(1, t, h), torch.randn(1, s)
    with torch.no_grad():
        ours = model(x, y, eval_pos=s, task_type="reg")
        with unpatched():
            theirs = model(x, y, eval_pos=s, task_type="reg")
    assert ours.shape == theirs.shape == (1, t - s, model.num_reg_quantiles)
    assert torch.equal(ours, theirs)


def test_the_closed_form_mean_is_upstreams_quantile_mean():
    """quantile_mean (no sort) equals upstream's sort + trapezoid on the uniform tau grid, including on a
    crossed (unsorted) bank, which is what the sort in upstream exists for."""
    from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

    for k in (19, 999):
        tau = torch.tensor([(i + 1.0) / (k + 1.0) for i in range(k)], dtype=torch.float64)
        q = torch.randn(7, k, dtype=torch.float64) * 3 + 1          # deliberately unsorted
        assert torch.allclose(quantile_mean(q), quantile_dist_mean_simple(q, tau), atol=1e-12, rtol=0)


def test_the_released_grid_is_the_uniform_one_the_closed_form_assumes(model):
    """The legacy evenly spaced grid is what the closed form needs; upstream reconstructs it for a pinball
    checkpoint that records only its width (Nori-30M records `num_reg_quantiles: 999` and no grid)."""
    real = build_model(configs.real().model_config, seed=0)
    k = real.num_reg_quantiles
    assert k == 999 and real.regression_loss == "pinball"
    assert real.regression_quantiles == tuple((i + 1.0) / (k + 1.0) for i in range(k))


def test_the_wrapper_is_upstreams_regressor_on_one_forward(model):
    """The wrapper = upstream's NoriRegressor point path for ONE member: z-score y (population std),
    forward, quantile mean, invert. Query rows only, since upstream predicts only those."""
    from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

    rng = np.random.default_rng(0)
    xtr, xte = rng.standard_normal((25, 5)), rng.standard_normal((9, 5))
    ytr = rng.standard_normal(25) * 40 + 150
    x = torch.from_numpy(np.vstack([xtr, xte]).astype(np.float32))[None]
    with torch.no_grad():
        ours = ExportWrapper(model)(x, torch.from_numpy(ytr.astype(np.float32))[None])[0, 25:, 0].double()
        mu, sd = ytr.mean(), ytr.std()
        with unpatched():
            q = model(x, torch.from_numpy(((ytr - mu) / sd).astype(np.float32))[None], eval_pos=25,
                      task_type="reg")
        tau = torch.tensor(model.regression_quantiles, dtype=torch.float64)
        theirs = quantile_dist_mean_simple(q[0].double(), tau) * sd + mu
    assert torch.allclose(ours, theirs, atol=1e-3, rtol=1e-5)


def test_context_rows_are_the_context_presented_again_as_queries(model):
    """Rows < S are in-context values: the prediction a row gets when it is presented as a query against
    the full context, not a pad. Equal to asking for exactly those rows as the test set."""
    torch.manual_seed(1)
    x, y = torch.randn(1, 20, 5), torch.randn(1, 12)
    w = ExportWrapper(model)
    with torch.no_grad():
        full = w(x, y)[0, :, 0]
        ctx_as_queries = w(torch.cat([x[:, :12], x[:, :12]], dim=1), y)[0, 12:, 0]
    assert torch.allclose(full[:12], ctx_as_queries, atol=1e-5)
    assert full[:12].std() > 0


def test_the_graph_is_dynamic_and_weight_free(tmp_path, model):
    """ORT on the exported graph = the wrapper, at shapes OTHER than the export example (odd and even H,
    a single feature), and every checkpoint tensor is an external stub that is deleted afterwards."""
    cfg = configs.fixture()
    graph = tmp_path / "g.onnx"
    w = export.export_graph(model, graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                            dim_features=cfg.dim_features, example=cfg.example)
    tensor_map = export.postprocess(graph, dict(model.state_dict()))
    parity = export.check_parity(graph, w, ((40, 7, 30), (9, 1, 5), (33, 10, 21)))
    assert parity["ok"], parity
    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tensor_map)
    # every checkpoint tensor is a graph initializer (the engine injects them all by name)
    assert set(tensor_map["initializers"].values()) == set(model.state_dict())
