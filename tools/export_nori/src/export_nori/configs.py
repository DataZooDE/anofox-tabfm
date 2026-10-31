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
* no ``two_phase_config`` (that is Nori 1.5's encoder, a different model).
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
    if cfg.get("two_phase_config"):
        raise ValueError("two_phase_config is Nori 1.5's encoder, a different model; not exported here.")
    if cfg.get("sequence_attention_mode", "full") != "full" or cfg.get("qass_sdpa_scale"):
        raise ValueError("only full sequence attention with the QASS temperature folded into q is traced.")
    if cfg.get("task_type", "reg") != "reg":
        raise ValueError("Nori is regression-only (upstream build_model refuses any other task_type).")
    dec = cfg["decoder_config"]
    if dec.get("regression_loss", "pinball") != "pinball" or dec.get("regression_quantiles") is not None:
        raise ValueError("the in-graph point estimate assumes a pinball head on the legacy evenly spaced "
                         "grid tau_k = (k+1)/(K+1); this checkpoint declares something else.")
    if int(dec.get("num_reg_quantiles", 1)) < 2:
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


def get(name: str) -> ExportConfig:
    if name == "fixture":
        return fixture()
    if name == "real":
        return real()
    raise ValueError(f"unknown config {name!r} (fixture|real)")
