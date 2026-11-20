"""Nori 1.5 (Model v3, two-phase encoder): the patches are upstream's function, the graph is the wrapper's.

Random weights at the 1.5 fixture config (every structural flag of the released model, tiny dims), so this
runs offline in seconds. The real-weight contracts live in test_real_weights_v15.py.

Written red first: ``configs.fixture15`` / ``real15`` and the two-phase path of the patched forward did not
exist, and ``assert_shipped_path`` refused ``two_phase_config`` outright (Nori 1.5's encoder, "a different
model").
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import onnx
import pytest
import torch

from export_nori import configs, export
from export_nori.nori_patches import ExportWrapper, build_model, quantile_mean, unpatched

MLX_OPS_HEADER = pathlib.Path(__file__).resolve().parents[3] / "src" / "include" / "tabfm_mlx_ops.hpp"


@pytest.fixture(scope="module")
def model():
    return build_model(configs.fixture15().model_config, seed=3)


# ---- the config ---------------------------------------------------------------------------------------


def test_the_real_config_is_the_released_model_v3():
    """Transcribed from nori.pt['model_config'] (Synthefy/Nori-1-5): 9 cell layers at width 128 and a
    two-phase encoder (2 target tokens, 12 row layers at width 2 * 128 over 8 heads)."""
    cfg = configs.real15().model_config
    assert cfg["nlayers"] == 9 and cfg["embed_dim"] == 128 and cfg["nhead"] == 4
    assert cfg["features_per_group"] == 1
    assert cfg["two_phase_config"] == {"target_tokens": 2, "row_layers": 12, "row_heads": 8,
                                       "row_hidden_dim": 768}
    assert cfg["use_prompt_quantile_ple"] and cfg["prompt_quantile_ple_bins"] == 64
    assert cfg["qass_mode"] == "log_only"
    dec = cfg["decoder_config"]
    k = dec["num_reg_quantiles"]
    assert k == 999 and dec["regression_loss"] == "pinball"
    assert dec["regression_quantiles"] == [(i + 1) / (k + 1) for i in range(k)]
    assert cfg["mask_prediction"] is False        # forced off for inference, as upstream's load_model does


def test_the_fixture_config_has_the_same_structure_at_tiny_dims():
    real, fix = configs.real15().model_config, configs.fixture15().model_config
    assert fix["two_phase_config"] and fix["use_prompt_quantile_ple"]
    assert fix["features_per_group"] == real["features_per_group"] == 1
    assert fix["embed_dim"] < real["embed_dim"] and fix["nlayers"] < real["nlayers"]
    k = fix["decoder_config"]["num_reg_quantiles"]
    assert fix["decoder_config"]["regression_quantiles"] == [(i + 1) / (k + 1) for i in range(k)]


def test_two_phase_is_accepted_and_the_rest_of_the_guard_still_holds():
    good = configs.fixture15().model_config
    configs.assert_shipped_path(good)
    for key, value, why in [
        ("mask_prediction", True, "imputation dict"),
        ("feature_positional_embedding_type", "learned", "feature_positional_embedding_type"),
        ("sequence_attention_mode", "inducing", "full sequence attention"),
        ("task_type", "cls", "regression-only"),
    ]:
        bad = dict(good, **{key: value})
        with pytest.raises(ValueError, match=why):
            configs.assert_shipped_path(bad)


def test_a_quantile_grid_that_is_not_the_uniform_one_is_refused():
    """The in-graph mean is a closed form only on tau_k = (k + 1) / (K + 1); an explicit grid is fine when
    it IS that grid (the released config records it explicitly), and refused when it is anything else."""
    cfg = configs.fixture15().model_config
    k = cfg["decoder_config"]["num_reg_quantiles"]
    skewed = [(i + 1) ** 2 / (k + 1) ** 2 for i in range(k)]
    bad = dict(cfg, decoder_config=dict(cfg["decoder_config"], regression_quantiles=skewed))
    with pytest.raises(ValueError, match="evenly spaced"):
        configs.assert_shipped_path(bad)
    assert configs.fixture15().name == "fixture15" and configs.real15().name == "real15"
    assert configs.get("fixture15").name == "fixture15" and configs.get("real15").name == "real15"


def test_the_released_architecture_builds_at_the_released_tensor_count():
    """Random weights, released dims: 381 tensors (what nori.pt's model_state_dict holds), ~20.76 M params."""
    real = build_model(configs.real15().model_config, seed=0)
    assert len(real.state_dict()) == 381
    assert 20_000_000 < sum(p.numel() for p in real.parameters()) < 21_500_000
    assert real.regression_quantiles == tuple((i + 1.0) / 1000.0 for i in range(999))


# ---- the patches --------------------------------------------------------------------------------------


@pytest.mark.parametrize("t,h,s", [(30, 7, 20), (12, 4, 5), (9, 1, 2), (17, 2, 10)])
def test_the_patched_forward_is_upstreams_forward(model, t, h, s):
    """Every patch is the identical function on the inference path: the patched model and upstream's own
    forward (every patch reverted) agree, at odd and even widths, one feature, and S = 2."""
    torch.manual_seed(t + h + s)
    x, y = torch.randn(1, t, h), torch.randn(1, s)
    with torch.no_grad():
        ours = model(x, y, eval_pos=s, task_type="reg")
        with unpatched():
            theirs = model(x, y, eval_pos=s, task_type="reg")
    assert ours.shape == theirs.shape == (1, t - s, model.num_reg_quantiles)
    assert torch.allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_the_patched_forward_handles_missing_values_like_upstream(model):
    torch.manual_seed(5)
    x, y = torch.randn(1, 24, 6), torch.randn(1, 15)
    x[0, 3, 2] = float("nan")
    x[0, 17, 0] = float("nan")
    x[0, 20, 5] = float("inf")
    x[0, :, 4] = 1.0                      # a constant column: the PLE boundaries are all equal
    with torch.no_grad():
        ours = model(x, y, eval_pos=15, task_type="reg")
        with unpatched():
            theirs = model(x, y, eval_pos=15, task_type="reg")
    assert torch.isfinite(ours).all()
    assert torch.allclose(ours, theirs, atol=1e-6, rtol=1e-6)


def test_the_closed_form_mean_is_upstreams_quantile_mean_on_the_released_grid():
    from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

    k = 999
    tau = torch.tensor([(i + 1.0) / (k + 1.0) for i in range(k)], dtype=torch.float64)
    q = torch.randn(7, k, dtype=torch.float64) * 3 + 1          # deliberately unsorted
    assert torch.allclose(quantile_mean(q), quantile_dist_mean_simple(q, tau), atol=1e-12, rtol=0)


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


def test_query_rows_do_not_see_each_other(model):
    """Row attention is context-only, so a query row's answer cannot depend on which other queries come with
    it (the property the engine's chunking and the ROCm padding both rely on)."""
    torch.manual_seed(2)
    x, y = torch.randn(1, 18, 4), torch.randn(1, 10)
    w = ExportWrapper(model)
    with torch.no_grad():
        both = w(x, y)[0, 10:, 0]
        alone = torch.stack([w(torch.cat([x[:, :10], x[:, 10 + i:11 + i]], dim=1), y)[0, 10, 0]
                             for i in range(8)])
    assert torch.allclose(both, alone, atol=1e-5)


def test_the_clamp_activation_equals_relu_including_nan_and_negative_zero(model):
    """Patch 6: the replaced ReLUs are the same function on every input an activation can see."""
    relus = [m for m in model.modules() if type(m) is torch.nn.ReLU]
    assert relus, "the 1.5 encoder has ReLUs (its fusion network): the patch has something to replace"
    probe = torch.tensor([-3.0, -0.0, 0.0, 0.5, 7.0, float("nan"), float("inf"), -float("inf")])
    for relu in relus:
        want = torch.nn.functional.relu(probe)
        got = relu(probe)
        assert torch.equal(torch.nan_to_num(got, nan=-1.0), torch.nan_to_num(want, nan=-1.0))
        assert torch.isnan(got[5]) == torch.isnan(want[5])


# ---- the graph ----------------------------------------------------------------------------------------


def _mlx_ops() -> set[str]:
    text = MLX_OPS_HEADER.read_text()
    body = text.split("MlxImplementedOps()", 1)[1].split("};", 1)[0]
    return set(re.findall(r'"([A-Za-z0-9_]+)"', body))


def test_the_graph_is_dynamic_weight_free_and_runs_on_every_backend_op_set(tmp_path, model):
    """ORT on the exported graph = the wrapper, at shapes OTHER than the export example (odd and even H,
    a single feature, S far from T), every checkpoint tensor is an external stub that is deleted afterwards,
    and every op is one the MLX interpreter runs (Unique and Relu cost LimiX its first MLX port)."""
    cfg = configs.fixture15()
    graph = tmp_path / "g.onnx"
    w = export.export_graph(model, graph, dim_rows=cfg.dim_rows, dim_train=cfg.dim_train,
                            dim_features=cfg.dim_features, example=cfg.example)
    tensor_map = export.postprocess(graph, dict(model.state_dict()))
    parity = export.check_parity(graph, w, ((40, 7, 30), (9, 1, 5), (33, 10, 21), (25, 2, 3)))
    assert parity["ok"], parity
    ops = {n.op_type for n in onnx.load(str(graph), load_external_data=False).graph.node}
    assert not ops & {"Unique", "Relu", "NonZero", "Einsum"}, ops
    assert ops <= _mlx_ops(), sorted(ops - _mlx_ops())
    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tensor_map)
    # every checkpoint tensor is a graph initializer (the engine injects them all by name)
    assert set(tensor_map["initializers"].values()) == set(model.state_dict())


# ---- the weights file ---------------------------------------------------------------------------------


def _write_ckpt(tmp_path, model, key):
    sd = {k: v.detach().contiguous().clone() for k, v in sorted(model.state_dict().items())}
    path = tmp_path / "nori.pt"
    torch.save({"model_config": configs.fixture15().model_config, key: sd,
                "inference_config": {"name": "model_v3_inference.json", "members": []}}, str(path))
    return path


def test_convert_reads_a_model_v3_checkpoint(tmp_path, model):
    """nori.pt of Nori 1.5 keeps its weights under ``model_state_dict`` (no EMA copy: the 30M file's
    ``ema_state_dict``). The converter writes the same bare, sorted, float32 safetensors either way."""
    from safetensors import safe_open

    from export_nori import convert

    src = _write_ckpt(tmp_path, model, "model_state_dict")
    out = tmp_path / "model.safetensors"
    assert convert.convert(src, out) == len(model.state_dict())
    with safe_open(str(out), "pt") as f:
        assert list(f.keys()) == sorted(model.state_dict())
        for name, value in model.state_dict().items():
            assert torch.equal(f.get_tensor(name), value.to(torch.float32)), name


def test_convert_still_prefers_the_ema_weights(tmp_path, model):
    from safetensors import safe_open

    from export_nori import convert

    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema = {k: v + 1.0 for k, v in sd.items()}
    path = tmp_path / "both.pt"
    torch.save({"model_config": {}, "model_state_dict": sd, "ema_state_dict": ema}, str(path))
    out = tmp_path / "both.safetensors"
    convert.convert(path, out)
    with safe_open(str(out), "pt") as f:
        key = sorted(sd)[0]
        assert torch.equal(f.get_tensor(key), ema[key].to(torch.float32))


def test_convert_refuses_a_checkpoint_with_no_weights(tmp_path):
    from export_nori import convert

    path = tmp_path / "empty.pt"
    torch.save({"model_config": {}}, str(path))
    with pytest.raises(ValueError, match="ema_state_dict.*model_state_dict"):
        convert.convert(path, tmp_path / "x.safetensors")
