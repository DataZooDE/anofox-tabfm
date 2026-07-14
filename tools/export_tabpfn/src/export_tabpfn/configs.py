"""Export configurations: tiny / fixture / real.

`tiny`/`fixture` use small dims (fast CI). `real` uses the published TabPFN v2
architecture dims (emsize=192, nlayers=12, nhead=6, features_per_group=2) — dims
only; the model is built with RANDOM weights so no Prior Labs weight bytes are
involved (the exported graph is architecture-only and weight-free).
"""

from __future__ import annotations

import dataclasses

OPSET = 18


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    model_kwargs: dict  # TabPFNV2Config kwargs (+ num_buckets for regression)
    max_classes: int  # classification n_out
    num_buckets: int  # regression n_out
    example: tuple  # (T, H, N)
    parity_shapes: tuple  # ((T, H, N), ...) — all != example


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


def get(name: str) -> ExportConfig:
    return {"tiny": tiny, "fixture": fixture, "real": real}[name]()
