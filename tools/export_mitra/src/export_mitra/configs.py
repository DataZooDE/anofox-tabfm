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
    raise ValueError(f"unknown config {config!r} (real|fixture)")
