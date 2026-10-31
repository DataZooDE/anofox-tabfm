"""Architecture dimensions for the export.

The three published sizes take their dims from upstream's own `MODEL_KWARGS`
(sdm/models/kumo/tabular/model.py), which `KumoTabular(size=...)` uses to build the module the
`nvidia/Kumo-Tabular` checkpoints load into; nothing is transcribed by hand. Heads: 10 classes,
999 quantiles.

`fixture` is a small random-init model for CI. It keeps every structural feature of the published
ones that changes the graph: a feature group of 3 (so the circular group wrap matters), more than one
embedding and ICL layer, several readout tokens, a `row_project` Linear (cell_channels x readout !=
icl_channels, as in `medium`), and the Test-GQA branch (`num_icl_key_value_heads_for_query`, as in
`medium` / `large`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sdm.models.kumo.tabular.model import MODEL_KWARGS


@dataclass(frozen=True)
class ExportConfig:
    name: str
    model: dict                    # _KumoTabular kwargs minus num_classes/num_quantiles
    classes: int                   # classification head width (the engine's max_classes)
    quantiles: int                 # regression head width
    example: tuple = (24, 7, 15)   # (T, H, S) used for tracing
    # (T, H, S[, d, categorical columns]), all different from `example`
    parity_shapes: tuple = field(default=((40, 9, 25), (33, 12, 20, 10, (0, 4)), (17, 3, 9, 2)))


_FIXTURE = dict(
    cell_channels=16, num_embedding_layers=2, num_embedding_heads=2, num_inducing_points=8,
    group_size=3, num_frequencies=4, num_readout_tokens=2, icl_channels=24, num_icl_layers=2,
    num_icl_heads=2, num_icl_key_value_heads_for_query=1,
)


def get(name: str) -> ExportConfig:
    if name == "fixture":
        return ExportConfig("fixture", dict(_FIXTURE), classes=3, quantiles=16)
    return ExportConfig(name, dict(MODEL_KWARGS[name]), classes=10, quantiles=999)
