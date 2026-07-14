"""Export configurations: fixture (tiny) / real (v2 default architecture).

The exported ONNX graph is architecture-only and weight-free, so ``real`` builds
the model with the TabICL v2 DEFAULT __init__ dims and RANDOM weights — no
Google/soda-inria checkpoint bytes are ever involved. If onboarding the real
``tabicl-classifier-v2`` / ``tabicl-regressor-v2`` checkpoints, first verify the
checkpoint config matches these dims (download config from HF repo
``jingang/TabICL``); a shape mismatch would fail initializer injection.
"""

from __future__ import annotations

import dataclasses

OPSET = 18

# TabICL v2 __init__ defaults (== the released v2 architecture).
_REAL_KWARGS = dict(
    max_classes=10, num_quantiles=999, embed_dim=128,
    col_num_blocks=3, col_nhead=8, col_num_inds=128,
    row_num_blocks=3, row_nhead=8, row_num_cls=4,
    icl_num_blocks=12, icl_nhead=8, ff_factor=2,
)

# Tiny unit-test dims for the committed CI fixture (~25k params).
_FIXTURE_KWARGS = dict(
    max_classes=3, num_quantiles=16, embed_dim=16,
    col_num_blocks=1, col_nhead=2, col_num_inds=8,
    row_num_blocks=1, row_nhead=2, row_num_cls=2,
    icl_num_blocks=1, icl_nhead=2, ff_factor=2,
)

DIM_ROWS = ("rows", 4, 100_000)
DIM_TRAIN = ("train", 2, 100_000)
DIM_FEATURES = ("features", 2, 512)
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_TRAIN_FIXTURE = ("train", 2, 4096)
DIM_FEATURES_FIXTURE = ("features", 2, 64)


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    model_kwargs: dict
    dim_rows: tuple
    dim_train: tuple
    dim_features: tuple
    example: tuple           # (T, H, S) export example; parity shapes must differ
    parity_shapes: tuple     # ((T, H, S), ...)


def fixture() -> ExportConfig:
    return ExportConfig(
        name="fixture", model_kwargs=dict(_FIXTURE_KWARGS),
        dim_rows=DIM_ROWS_FIXTURE, dim_train=DIM_TRAIN_FIXTURE,
        dim_features=DIM_FEATURES_FIXTURE,
        example=(20, 5, 12), parity_shapes=((40, 7, 30),),
    )


def real() -> ExportConfig:
    return ExportConfig(
        name="real", model_kwargs=dict(_REAL_KWARGS),
        dim_rows=DIM_ROWS, dim_train=DIM_TRAIN, dim_features=DIM_FEATURES,
        example=(20, 8, 12), parity_shapes=((16, 4, 10), (60, 20, 40)),
    )


def get(name: str) -> ExportConfig:
    if name == "fixture":
        return fixture()
    if name == "real":
        return real()
    raise ValueError(f"unknown config {name!r} (fixture|real)")
