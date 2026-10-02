"""Architecture dimensions for the export.

`real` is read out of the published checkpoints' own `config.json`
(nums-ai/causilo @ 94f2bd91, classifier/ and regressor/): only the `outputs` field
differs between the two tasks (10 class logits / 999 regression channels). The
weights themselves are never committed.

`fixture` is a small random-init model for CI. It keeps every structural feature of
the real one that can change the graph: a feature group of 3 (so H not divisible
by 3 needs padding), more than one layer in the column/prediction stacks, two row
rounds, and more than one head.
"""

from __future__ import annotations

from dataclasses import dataclass

from causilo.model import ModelConfig


@dataclass(frozen=True)
class ExportConfig:
    name: str
    model: dict                    # ModelConfig kwargs minus task/outputs
    classes: int                   # classification head width (the engine's max_classes)
    channels: int                  # regression output channels
    example: tuple                 # (T, H, S) used for tracing
    parity_shapes: tuple           # (T, H, S), all different from `example`

    def model_config(self, task: str) -> ModelConfig:
        outputs = self.classes if task == "classification" else self.channels
        return ModelConfig(task=task, outputs=outputs, **self.model)


_REAL = dict(
    width=128, expansion=2, group_size=3, frequencies=16,
    column_latents=128, column_heads=4, column_depths=(3, 3),
    row_heads=8, row_latents=4, row_depths=(3, 3),
    prediction_heads=4, prediction_depth=12,
)

_FIXTURE = dict(
    width=16, expansion=2, group_size=3, frequencies=4,
    column_latents=8, column_heads=2, column_depths=(2, 1),
    row_heads=2, row_latents=2, row_depths=(2, 2),
    prediction_heads=2, prediction_depth=2,
)


def fixture() -> ExportConfig:
    return ExportConfig("fixture", dict(_FIXTURE), classes=3, channels=16,
                        example=(20, 5, 12), parity_shapes=((40, 7, 30), (16, 9, 6), (33, 4, 29)))


def real() -> ExportConfig:
    return ExportConfig("real", dict(_REAL), classes=10, channels=999,
                        example=(16, 6, 10), parity_shapes=((32, 12, 20), (24, 7, 15)))


def get(name: str) -> ExportConfig:
    return {"fixture": fixture, "real": real}[name]()
