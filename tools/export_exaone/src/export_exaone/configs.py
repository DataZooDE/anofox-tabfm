"""Export configurations: fixture (tiny) / real (the released EXAONE Tabular v1 checkpoints).

The exported ONNX graphs are architecture-only and weight-free, so ``real`` builds the model with RANDOM
weights at the dims the released checkpoints use; no LG AI Research weight bytes are involved.

The real configs are upstream's own frozen release manifests (``exaonetabular.presets``):

* classifier ``ModelConfig(class_capacity=10)``: width 192, 6 heads, 12 CAST blocks, feed-forward 768,
  ONE column per feature token, 3 item-summary and 32 feature-summary tokens, 2 feature-attention
  repeats; 20,807,866 parameters.
* regressor ``ModelConfig(feature_attention_repeats=2)`` + ``RegressionConfig(quantile_count=999,
  decoder_hidden_width=384)``: the same trunk with a 999-quantile head; 21,111,678 checkpoint values
  (21,110,679 parameters + the 999-entry ``quantile_levels`` buffer).

``assert_shipped_path`` pins the property the export relies on: one column per token
(``columns_per_group == 1``), the vectorised feature-encoder path upstream's inference takes for the
released models.
"""

from __future__ import annotations

import copy
import dataclasses

OPSET = 18

_REAL = {
    "classification": ({"class_capacity": 10}, None),
    "regression": ({"feature_attention_repeats": 2}, {"quantile_count": 999, "decoder_hidden_width": 384}),
}

_FIXTURE_MODEL = {
    "width": 24, "head_count": 2, "block_count": 2, "feedforward_widths": 48, "columns_per_group": 1,
    "item_summary_count": 3, "feature_summary_count": 4, "feature_attention_repeats": 2,
}


def _fixture(task: str):
    model = dict(_FIXTURE_MODEL)
    if task == "classification":
        model["class_capacity"] = 3
        return model, None
    return model, {"quantile_count": 19, "decoder_hidden_width": 32}


DIM_ROWS = ("rows", 4, 100_000)
DIM_TRAIN = ("train", 2, 100_000)
DIM_FEATURES = ("features", 1, 500)
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_TRAIN_FIXTURE = ("train", 2, 4096)
DIM_FEATURES_FIXTURE = ("features", 1, 64)


def assert_shipped_path(model_config: dict) -> None:
    if int(model_config.get("columns_per_group", 1)) != 1:
        raise ValueError("only columns_per_group == 1 (the released models) is on the exported path: the "
                         "grouped path packs varying coordinates with a data-dependent per-group loop.")


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    task: str
    model_config: dict
    regression_config: dict | None
    max_classes: int
    dim_rows: tuple
    dim_train: tuple
    dim_features: tuple
    example: tuple           # (T, H, S) export example; parity shapes must differ
    parity_shapes: tuple     # ((T, H, S), ...)


def get(name: str, task: str) -> ExportConfig:
    if task not in _REAL:
        raise ValueError(f"unknown task {task!r} (classification|regression)")
    if name == "fixture":
        model, reg = _fixture(task)
        assert_shipped_path(model)
        return ExportConfig(name, task, model, reg, model.get("class_capacity", 3), DIM_ROWS_FIXTURE,
                            DIM_TRAIN_FIXTURE, DIM_FEATURES_FIXTURE, example=(20, 6, 12),
                            parity_shapes=((40, 7, 30), (9, 1, 5)))
    if name == "real":
        model, reg = copy.deepcopy(_REAL[task])
        assert_shipped_path(model)
        return ExportConfig(name, task, model, reg, model.get("class_capacity", 10), DIM_ROWS, DIM_TRAIN,
                            DIM_FEATURES, example=(20, 8, 12), parity_shapes=((16, 3, 10), (60, 20, 40)))
    raise ValueError(f"unknown config {name!r} (fixture|real)")
