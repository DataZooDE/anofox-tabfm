"""Export configurations: fixture (tiny) / real (the released Nori-30M architecture).

The exported ONNX graph is architecture-only and weight-free, so ``real`` builds the model with RANDOM
weights at the dims the released checkpoint declares; no Synthefy checkpoint bytes are involved.

``_REAL_CONFIG`` is transcribed verbatim from the ``model_config`` dict embedded in
``Synthefy/Nori-30M :: nori.pt`` (revision 63c9f7fa, 29,206,833 params over 773 tensors), which
``synthefy_nori.utils.loading.build_model`` consumes directly, exactly as upstream's ``load_model`` does.

Nori is a LimiX descendant (upstream NOTICE: "This codebase originated as a fork of LimiX") with a
deeper pre-norm stack: 28 ``smf`` layers, RMSNorm, SwiGLU, DeepNorm residual scaling, a query-adaptive
softmax temperature in the sample attention (QASSMax), a target-aware embedding of the context labels,
and a 999-quantile pinball regression head. The checkpoint config also carries the classification head
(``num_classes: 10``) but ``build_model`` refuses any ``task_type`` other than ``reg``: the released
model is regression-only, and so is this exporter.

``assert_shipped_path`` pins the flags that keep the traced path narrow:

* ``mask_prediction`` is forced False (upstream's ``load_model`` does the same for inference): the
  imputation head is off, so ``forward`` returns the quantile bank rather than a dict.
* ``feature_positional_embedding_type = "none"``: the other kinds draw ``torch.randn`` / ``randperm``
  at a symbolic column count on every call.
* ``two_phase_config`` is allowed: it selects Nori 1.5's encoder (see nori_patches.apply).
* the regression head is a PINBALL head on the legacy evenly spaced grid tau_k = (k + 1) / (K + 1),
  which is what makes the in-graph mean a closed form (see ``nori_patches.quantile_mean``).
"""

from __future__ import annotations

import copy
import dataclasses

OPSET = 18

# Verbatim from nori.pt["model_config"] (mask_prediction is overridden to False below, as upstream's
# load_model(mask_prediction=False) does for every inference call).
_REAL_CONFIG = {
    "preprocess_config_x": {
        "num_features": 2, "nan_handling_enabled": True, "normalize_on_train_only": True,
        "normalize_x": True, "remove_outliers": False, "normalize_by_used_features": True,
    },
    "encoder_config_x": {
        "num_features": 2, "embedding_size": 224, "mask_embedding_size": 224, "encoder_use_bias": True,
        "numeric_embed_type": "RBF",
        "RBF_config": {
            "token_embed_dim": 16, "n_kernels": 64, "sigma": 0.5, "use_learn_sigma": False,
            "use_learn_embeddings": False, "use_random_kernels": False, "use_original_features": False,
        },
    },
    "encoder_config_y": {
        "num_inputs": 1, "embedding_size": 224, "nan_handling_y_encoder": True, "max_num_classes": 10,
    },
    "decoder_config": {"num_classes": 10, "num_reg_quantiles": 999},
    "feature_positional_embedding_type": "none",
    "nlayers": 28, "nhead": 4, "embed_dim": 224, "hid_dim": 768, "features_per_group": 2,
    "dropout": 0, "pre_norm": True, "activation": "swiglu", "recompute_attn": False,
    "layer_arch": "smf", "self_share_all_kv_heads": False, "cross_share_all_kv_heads": False,
    "seq_attn_isolated": True, "seq_attn_serial": False,
    "mask_prediction": False,
    "use_qassmax": True, "qass_mode": "full", "qass_sdpa_scale": False,
    "use_target_aware_embedding": True, "use_column_specific_y_aware": True,
    "sequence_attention_mode": "full", "sequence_attention_inducing_points": 128,
    "flat_attention_params": False, "norm_type": "rmsnorm", "deepnorm_alpha": 0.18898223650461363,
}


# Nori 1.5 (Model v3), verbatim from ``Synthefy/Nori-1-5 :: nori.pt["model_config"]`` (revision 79afc5a6,
# 381 tensors, 20,757,832 params). Only two entries differ from the file: ``mask_prediction`` (forced False for
# inference, as upstream's ``load_model`` does) and ``regression_quantiles`` (the file lists the 999 levels
# 0.001 ... 0.999; they are generated here and the real-weight test proves them equal to the file's).
_K15 = 999
_REAL_CONFIG_15 = {
    "preprocess_config_x": {
        "num_features": 1, "nan_handling_enabled": True, "normalize_on_train_only": True,
        "normalize_x": True, "remove_outliers": False, "normalize_by_used_features": True,
        "prompt_quantile_ple_bins": 64,
    },
    "encoder_config_x": {
        "num_features": 1, "embedding_size": 128, "mask_embedding_size": 128, "encoder_use_bias": True,
        "numeric_embed_type": "RBF",
        "RBF_config": {
            "token_embed_dim": 16, "n_kernels": 64, "sigma": 0.5, "use_learn_sigma": False,
            "use_learn_embeddings": False, "use_original_features": False,
            "compute_dtype": "float64", "input_transform": "scientific", "use_raw_input_residual": False,
        },
        "prompt_quantile_ple_bins": 64,
    },
    "encoder_config_y": {"num_inputs": 1, "embedding_size": 128, "nan_handling_y_encoder": True},
    "decoder_config": {
        "num_reg_quantiles": _K15, "regression_loss": "pinball",
        "regression_quantiles": [(i + 1) / (_K15 + 1) for i in range(_K15)],
    },
    "feature_positional_embedding_type": "none",
    "nlayers": 9, "nhead": 4, "embed_dim": 128, "hid_dim": 384, "features_per_group": 1,
    "feature_group_circular": False, "dropout": 0, "pre_norm": True, "activation": "swiglu",
    "recompute_attn": False, "layer_arch": "smf", "self_share_all_kv_heads": False,
    "cross_share_all_kv_heads": False, "seq_attn_isolated": True, "seq_attn_serial": False,
    "mask_prediction": False,
    "use_qassmax": True, "use_target_aware_embedding": True, "use_column_specific_y_aware": True,
    "omit_feature_decoder": True, "feature_positional_embedding_num_slots": 1000,
    "use_column_deepset": False, "column_deepset_hidden_dim": None,
    "two_phase_config": {"target_tokens": 2, "row_layers": 12, "row_heads": 8, "row_hidden_dim": 768},
    "norm_type": "rmsnorm", "deepnorm_alpha": 0.3333333333333333,
    "use_prompt_quantile_ple": True, "prompt_quantile_ple_bins": 64,
    "use_conditional_landmarks": False, "conditional_landmark_embedding_dim": 32,
    "conditional_landmark_bins": 8, "conditional_landmark_shrinkage": 4.0,
    "conditional_landmark_dropout": 0.0,
    "qass_mode": "log_only", "use_logn_attention": False, "use_learnable_attn_temperature": False,
    "attn_n_ref": 1024.0,
}


def _fixture_config_15() -> dict:
    """Tiny CI-fixture dims, same structural shape (every flag) as the released 1.5 config."""
    cfg = copy.deepcopy(_REAL_CONFIG_15)
    cfg["preprocess_config_x"]["prompt_quantile_ple_bins"] = 8
    cfg["encoder_config_x"]["prompt_quantile_ple_bins"] = 8
    cfg["prompt_quantile_ple_bins"] = 8
    cfg["encoder_config_x"]["embedding_size"] = 24
    cfg["encoder_config_x"]["mask_embedding_size"] = 24
    cfg["encoder_config_x"]["RBF_config"]["n_kernels"] = 8
    cfg["encoder_config_x"]["RBF_config"]["token_embed_dim"] = 8
    cfg["encoder_config_y"]["embedding_size"] = 24
    k = 19
    cfg["decoder_config"] = {"num_reg_quantiles": k, "regression_loss": "pinball",
                             "regression_quantiles": [(i + 1) / (k + 1) for i in range(k)]}
    cfg["nlayers"] = 2
    cfg["nhead"] = 2
    cfg["embed_dim"] = 24
    cfg["hid_dim"] = 48
    cfg["two_phase_config"] = {"target_tokens": 2, "row_layers": 2, "row_heads": 2, "row_hidden_dim": 96}
    return cfg


def _fixture_config() -> dict:
    """Tiny CI-fixture dims, same structural shape (every flag) as the released config."""
    cfg = copy.deepcopy(_REAL_CONFIG)
    cfg["encoder_config_x"]["embedding_size"] = 24
    cfg["encoder_config_x"]["mask_embedding_size"] = 24
    cfg["encoder_config_x"]["RBF_config"]["n_kernels"] = 8
    cfg["encoder_config_x"]["RBF_config"]["token_embed_dim"] = 8
    cfg["encoder_config_y"]["embedding_size"] = 24
    cfg["decoder_config"]["num_reg_quantiles"] = 19
    cfg["nlayers"] = 2
    cfg["nhead"] = 2
    cfg["embed_dim"] = 24
    cfg["hid_dim"] = 48
    return cfg


DIM_ROWS = ("rows", 4, 100_000)
DIM_TRAIN = ("train", 2, 100_000)
DIM_FEATURES = ("features", 1, 512)
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_TRAIN_FIXTURE = ("train", 2, 4096)
DIM_FEATURES_FIXTURE = ("features", 1, 64)


def assert_shipped_path(cfg: dict) -> None:
    """Fail loudly if the config leaves the traced path."""
    if cfg.get("mask_prediction"):
        raise ValueError("mask_prediction=True returns the imputation dict, not the quantile bank; "
                         "upstream's load_model forces it False for inference and so does the exporter.")
    if cfg.get("feature_positional_embedding_type") != "none":
        raise ValueError("only feature_positional_embedding_type='none' is on the exported path: the others "
                         "draw torch.randn / torch.randperm at a symbolic column count on every call.")
    if cfg.get("sequence_attention_mode", "full") != "full" or cfg.get("qass_sdpa_scale"):
        raise ValueError("only full sequence attention with the QASS temperature folded into q is traced.")
    if cfg.get("task_type", "reg") != "reg":
        raise ValueError("Nori is regression-only (upstream build_model refuses any other task_type).")
    dec = cfg["decoder_config"]
    k = int(dec.get("num_reg_quantiles", 1))
    grid = dec.get("regression_quantiles")
    uniform = grid is None or (len(grid) == k and all(g == (i + 1) / (k + 1) for i, g in enumerate(grid)))
    if dec.get("regression_loss", "pinball") != "pinball" or not uniform:
        raise ValueError("the in-graph point estimate assumes a pinball head on the legacy evenly spaced "
                         "grid tau_k = (k+1)/(K+1) (listed explicitly or not at all); this checkpoint "
                         "declares something else.")
    if k < 2:
        raise ValueError("a single-output head has no quantile bank to collapse.")


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    model_config: dict
    dim_rows: tuple
    dim_train: tuple
    dim_features: tuple
    example: tuple           # (T, H, S) export example; parity shapes must differ
    parity_shapes: tuple     # ((T, H, S), ...)


def fixture() -> ExportConfig:
    cfg = _fixture_config()
    assert_shipped_path(cfg)
    return ExportConfig(
        name="fixture", model_config=cfg,
        dim_rows=DIM_ROWS_FIXTURE, dim_train=DIM_TRAIN_FIXTURE, dim_features=DIM_FEATURES_FIXTURE,
        example=(20, 6, 12), parity_shapes=((40, 7, 30), (9, 1, 5)),
    )


def real() -> ExportConfig:
    cfg = copy.deepcopy(_REAL_CONFIG)
    assert_shipped_path(cfg)
    return ExportConfig(
        name="real", model_config=cfg,
        dim_rows=DIM_ROWS, dim_train=DIM_TRAIN, dim_features=DIM_FEATURES,
        example=(20, 8, 12), parity_shapes=((16, 3, 10), (60, 20, 40)),
    )


def fixture15() -> ExportConfig:
    cfg = _fixture_config_15()
    assert_shipped_path(cfg)
    return ExportConfig(
        name="fixture15", model_config=cfg,
        dim_rows=DIM_ROWS_FIXTURE, dim_train=DIM_TRAIN_FIXTURE, dim_features=DIM_FEATURES_FIXTURE,
        example=(20, 6, 12), parity_shapes=((40, 7, 30), (9, 1, 5)),
    )


def real15() -> ExportConfig:
    cfg = copy.deepcopy(_REAL_CONFIG_15)
    assert_shipped_path(cfg)
    return ExportConfig(
        name="real15", model_config=cfg,
        dim_rows=DIM_ROWS, dim_train=DIM_TRAIN, dim_features=DIM_FEATURES,
        example=(20, 8, 12), parity_shapes=((16, 3, 10), (60, 20, 40)),
    )


def get(name: str) -> ExportConfig:
    if name == "fixture":
        return fixture()
    if name == "real":
        return real()
    if name == "fixture15":
        return fixture15()
    if name == "real15":
        return real15()
    raise ValueError(f"unknown config {name!r} (fixture|real|fixture15|real15)")
