"""Export configurations: fixture (tiny) / real (the released LimiX-2M architecture).

The exported ONNX graph is architecture-only and weight-free, so ``real`` builds
the model with RANDOM weights at the dims the released checkpoint declares — no
StableAI checkpoint bytes are ever involved.

``_REAL_CONFIG`` is transcribed verbatim from the ``config`` dict embedded in
``stable-ai/LimiX-2M :: LimiX-2M.ckpt`` (2,377,837 params over 137 tensors).
Unlike the other exporters LimiX's config is NESTED (per-encoder sub-dicts), and
``utils.loading.build_model`` consumes it directly.

Two flags keep the traced path narrow, pinned by ``assert_shipped_path``:

* ``mask_prediction = False`` — the imputation head is off, so ``forward``
  returns plain logits rather than the dict with ``feature_pred`` and a
  ``process_config`` of preprocessing statistics.
* ``feature_positional_embedding_type = "none"`` — avoids the runtime
  ``torch.randn`` positional draw at ``transformer.py:277``, which cannot be
  traced at a symbolic column count.
"""

from __future__ import annotations

import copy
import dataclasses

OPSET = 18

# Verbatim from LimiX-2M.ckpt["config"].
_REAL_CONFIG = {
    "preprocess_config_x": {
        "num_features": 2, "nan_handling_enabled": True,
        "normalize_on_train_only": True, "normalize_x": True,
        "remove_outliers": False, "normalize_by_used_features": True,
    },
    "encoder_config_x": {
        "num_features": 2, "embedding_size": 96, "mask_embedding_size": 96,
        "encoder_use_bias": True, "numeric_embed_type": "RBF",
        "RBF_config": {
            "token_embed_dim": 16, "n_kernels": 64, "sigma": 0.5,
            "use_learn_sigma": False, "use_learn_embeddings": False,
            "use_random_kernels": False, "use_original_features": False,
        },
    },
    "encoder_config_y": {
        "num_inputs": 1, "embedding_size": 96,
        "nan_handling_y_encoder": True, "max_num_classes": 10,
    },
    "decoder_config": {"num_classes": 10},
    "feature_positional_embedding_type": "none",
    "nlayers": 12, "nhead": 6, "embed_dim": 96, "hid_dim": 384,
    "features_per_group": 2, "dropout": 0, "pre_norm": False,
    "activation": "gelu", "recompute_attn": False, "layer_arch": "smf",
    "self_share_all_kv_heads": False, "cross_share_all_kv_heads": False,
    "seq_attn_isolated": True, "seq_attn_serial": False,
    "mask_prediction": False,
}


def _fixture_config() -> dict:
    """Tiny CI-fixture dims, same structural shape as the released config."""
    cfg = copy.deepcopy(_REAL_CONFIG)
    cfg["encoder_config_x"]["embedding_size"] = 24
    cfg["encoder_config_x"]["mask_embedding_size"] = 24
    cfg["encoder_config_x"]["RBF_config"]["n_kernels"] = 8
    cfg["encoder_config_x"]["RBF_config"]["token_embed_dim"] = 8
    cfg["encoder_config_y"]["embedding_size"] = 24
    cfg["encoder_config_y"]["max_num_classes"] = 3
    cfg["decoder_config"]["num_classes"] = 3
    cfg["nlayers"] = 2
    cfg["nhead"] = 2
    cfg["embed_dim"] = 24
    cfg["hid_dim"] = 48
    return cfg


DIM_ROWS = ("rows", 4, 100_000)
DIM_TRAIN = ("train", 2, 100_000)
DIM_FEATURES = ("features", 2, 512)
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_TRAIN_FIXTURE = ("train", 2, 4096)
DIM_FEATURES_FIXTURE = ("features", 2, 64)


def assert_shipped_path(cfg: dict) -> None:
    """Fail loudly if the config leaves the traced path."""
    if cfg.get("mask_prediction"):
        raise ValueError(
            "mask_prediction=True returns the imputation dict (feature_pred + "
            "preprocessing statistics), not logits; the engine contract is "
            "logits[1,T,C]. The released LimiX-2M checkpoint sets it False."
        )
    if cfg.get("feature_positional_embedding_type") != "none":
        raise ValueError(
            f"feature_positional_embedding_type="
            f"{cfg.get('feature_positional_embedding_type')!r}: only 'none' is on "
            "the exported path — the others draw torch.randn at a symbolic column "
            "count (transformer.py:277), which cannot be traced."
        )


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    model_config: dict
    max_classes: int
    dim_rows: tuple
    dim_train: tuple
    dim_features: tuple
    example: tuple           # (T, H, S) export example; parity shapes must differ
    parity_shapes: tuple     # ((T, H, S), ...)


def fixture() -> ExportConfig:
    cfg = _fixture_config()
    assert_shipped_path(cfg)
    return ExportConfig(
        name="fixture", model_config=cfg, max_classes=cfg["decoder_config"]["num_classes"],
        dim_rows=DIM_ROWS_FIXTURE, dim_train=DIM_TRAIN_FIXTURE,
        dim_features=DIM_FEATURES_FIXTURE,
        example=(20, 6, 12), parity_shapes=((40, 8, 30),),
    )


def real() -> ExportConfig:
    cfg = copy.deepcopy(_REAL_CONFIG)
    assert_shipped_path(cfg)
    return ExportConfig(
        name="real", model_config=cfg, max_classes=cfg["decoder_config"]["num_classes"],
        dim_rows=DIM_ROWS, dim_train=DIM_TRAIN, dim_features=DIM_FEATURES,
        example=(20, 8, 12), parity_shapes=((16, 4, 10), (60, 20, 40)),
    )


def get(name: str) -> ExportConfig:
    if name == "fixture":
        return fixture()
    if name == "real":
        return real()
    raise ValueError(f"unknown config {name!r} (fixture|real)")
