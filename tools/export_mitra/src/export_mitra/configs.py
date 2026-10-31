"""Export configs: 'real' (released Mitra dims) and 'fixture' (tiny random)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    model_kwargs: dict
    dim_rows: tuple = ("rows", 4, 100_000)
    dim_features: tuple = ("features", 2, 100)
    example: tuple = (40, 8, 8, 24)        # (T, H, d, train_size)
    parity_shapes: tuple = field(default_factory=lambda: (
        (32, 6, 6, 20), (50, 10, 7, 30), (24, 4, 4, 12)))
    # "v1" (autogluon/mitra-*, the `mitra` entry) or "v2" (autogluon/mitra-*-2,
    # the `mitra-v2` entry). Only the v2 REGRESSOR differs structurally: a
    # dim_output-bin distributional head whose support targets are embedded by
    # bin id, exported through export.BinnedRegressionWrapper.
    arch: str = "v1"


def get(config: str, task: str) -> Config:
    is_cls = task == "classification"
    if config == "real":
        # released Mitra: dim=512, n_layers=12, n_heads=4; dim_output 10 | 1
        return Config(model_kwargs=dict(
            dim=512, dim_output=(10 if is_cls else 1), n_layers=12, n_heads=4))
    if config == "fixture":
        # tiny random-init CI fixture (weight-free graph + tiny safetensors)
        return Config(
            model_kwargs=dict(
                dim=32, dim_output=(4 if is_cls else 1), n_layers=2, n_heads=2),
            dim_rows=("rows", 4, 5000),
            dim_features=("features", 2, 32),
            example=(16, 5, 5, 10),
            parity_shapes=((14, 4, 4, 8), (20, 6, 5, 12)))
    if config == "real2":
        # Mitra-v2 (arXiv 2609.04540). config.json of autogluon/mitra-classifier-2
        # is byte-identical to v1's; autogluon/mitra-regressor-2 has dim_output
        # 1000 (a 1,000-bin head). H is traced up to 256, the v2 feature budget.
        return Config(
            model_kwargs=dict(dim=512, dim_output=(10 if is_cls else 1000),
                              n_layers=12, n_heads=4),
            dim_features=("features", 2, 256), arch="v2")
    if config == "fixture2":
        return Config(
            model_kwargs=dict(
                dim=32, dim_output=(4 if is_cls else 32), n_layers=2, n_heads=2),
            dim_rows=("rows", 4, 5000),
            dim_features=("features", 2, 32),
            example=(16, 5, 5, 10),
            parity_shapes=((14, 4, 4, 8), (20, 6, 5, 12)), arch="v2")
    raise ValueError(f"unknown config {config!r} (real|fixture|real2|fixture2)")
