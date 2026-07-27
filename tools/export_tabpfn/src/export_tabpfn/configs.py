"""Export configurations: tiny / fixture / real / real25.

`tiny`/`fixture` use small dims (fast CI). `real` uses the published TabPFN v2
architecture dims (emsize=192, nlayers=12, nhead=6, features_per_group=2) — dims
only; the model is built with RANDOM weights so no Prior Labs weight bytes are
involved (the exported graph is architecture-only and weight-free).

`real25` / `fixture25` target the **TabPFN-2.5** architecture
(`tabpfn.architectures.tabpfn_v2_5`), whose dims are transcribed from the
`config` embedded in `Prior-Labs/tabpfn_2_5 :: tabpfn-v2.5-*-v2.5_default.ckpt`.
The 2.5 line is deeper (nlayers 24 vs 12) and adds `num_thinking_rows`; see
`tabpfn_patched.ARCHES`.
"""

from __future__ import annotations

import dataclasses

OPSET = 18


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    model_kwargs: dict  # architecture Config kwargs (+ num_buckets for regression)
    max_classes: int  # classification n_out
    num_buckets: int  # regression n_out
    example: tuple  # (T, H, N)
    parity_shapes: tuple  # ((T, H, N), ...) — all != example
    arch: str = "v2"  # key into tabpfn_patched.ARCHES


# small fast dims (probe-validated)
_FIXTURE_KWARGS = dict(emsize=32, nlayers=2, nhead=2, features_per_group=2, seed=0)
# published TabPFN v2 architecture dims (Prior-Labs/TabPFN-v2-*)
_REAL_KWARGS = dict(emsize=192, nlayers=12, nhead=6, features_per_group=2, seed=0)


def fixture() -> ExportConfig:
    return ExportConfig(
        name="fixture", model_kwargs=dict(_FIXTURE_KWARGS),
        max_classes=4, num_buckets=16,
        example=(12, 5, 8), parity_shapes=((40, 7, 30), (16, 9, 6)),
    )


def tiny() -> ExportConfig:
    return ExportConfig(
        name="tiny", model_kwargs=dict(_FIXTURE_KWARGS),
        max_classes=10, num_buckets=32,
        example=(16, 6, 10), parity_shapes=((40, 7, 30), (200, 25, 150)),
    )


def real() -> ExportConfig:
    return ExportConfig(
        name="real", model_kwargs=dict(_REAL_KWARGS),
        max_classes=10, num_buckets=5000,
        example=(16, 6, 10), parity_shapes=((32, 12, 20),),
    )


# --- TabPFN-2.5 ---------------------------------------------------------------
#
# Transcribed from each released checkpoint's own embedded `config`
# (Prior-Labs/tabpfn_2_5). The two heads are NOT the same architecture — the
# classifier is 24 layers with a linear encoder, the regressor 18 layers with an
# MLP encoder — so `real25` is task-aware, exactly as export_tabicl is for
# `bias_free_ln`. Getting this wrong leaves initializers unmapped at injection.
_REAL25_COMMON = dict(
    emsize=192, features_per_group=3, nhead=3, nhid_factor=2,
    feature_positional_embedding="subspace",
    num_thinking_rows=64, seed=42,
    encoder_use_bias=False, encoder_mlp_hidden_dim=1024, encoder_mlp_num_layers=2,
    item_attention_type="full", feature_attention_type="full",
    multiquery_item_attention_for_test_set=True,
    nan_handling_enabled=True, nan_handling_y_encoder=True,
    remove_empty_features=True, normalize_x=True,
    normalize_by_used_features=True, normalize_on_train_only=True,
    dropout=0.0, recompute_attn=False, recompute_layer=False,
)
_REAL25_CLF = dict(_REAL25_COMMON, nlayers=24, encoder_type="linear")
_REAL25_REG = dict(_REAL25_COMMON, nlayers=18, encoder_type="mlp")
# Tiny 2.5-shaped dims for the committed CI fixture (thinking rows kept small).
_FIXTURE25_KWARGS = dict(
    emsize=32, nlayers=2, nhead=2, features_per_group=3, seed=42,
    feature_positional_embedding="subspace", num_thinking_rows=4,
    encoder_type="linear", encoder_use_bias=False, dropout=0.0,
    item_attention_type="full", feature_attention_type="full",
)


def fixture25() -> ExportConfig:
    return ExportConfig(
        name="fixture25", model_kwargs=dict(_FIXTURE25_KWARGS), arch="v2.5",
        max_classes=4, num_buckets=16,
        example=(12, 5, 8), parity_shapes=((40, 7, 30), (16, 9, 6)),
    )


def real25(task: str = "classification") -> ExportConfig:
    kwargs = _REAL25_REG if task == "regression" else _REAL25_CLF
    return ExportConfig(
        name="real25", model_kwargs=dict(kwargs), arch="v2.5",
        max_classes=10, num_buckets=5000,
        example=(16, 6, 10), parity_shapes=((32, 12, 20),),
    )


def get(name: str, task: str = "classification") -> ExportConfig:
    if name == "real25":
        return real25(task)
    return {"tiny": tiny, "fixture": fixture, "real": real, "fixture25": fixture25}[name]()
