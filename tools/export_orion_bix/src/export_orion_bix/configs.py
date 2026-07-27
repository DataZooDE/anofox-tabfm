"""Export configurations: fixture (tiny) / real (the released v1.1 architecture).

The exported ONNX graph is architecture-only and weight-free, so ``real`` builds
the model with RANDOM weights at the dims the released checkpoint declares — no
Lexsi checkpoint bytes are ever involved.

``_REAL_KWARGS`` is transcribed from the ``config`` dict embedded in
``Lexsi/Orion-BiX::Orion-BiX-v1.1.ckpt`` (27,051,666 params over 277 tensors).
Note the three ``*_attention_type = "standard"`` flags: despite the model's
name, the released weights do NOT use the bi-axial or linear-attention branches.
``assert_shipped_path`` enforces that, so a future checkpoint that flips a flag
fails loudly rather than silently exporting a different architecture.
"""

from __future__ import annotations

import dataclasses

OPSET = 18

# Verbatim from Orion-BiX-v1.1.ckpt["config"].
_REAL_KWARGS = dict(
    max_classes=10, embed_dim=128,
    col_num_blocks=3, col_nhead=4, col_num_inds=128,
    row_num_blocks=3, row_nhead=8, row_num_cls=4, row_rope_base=100000.0,
    icl_num_blocks=12, icl_nhead=4, ff_factor=2,
    dropout=0.0, activation="gelu", norm_first=True,
    col_attention_type="standard", col_feature_map="identity",
    row_attention_type="standard", row_feature_map="identity",
    icl_attention_type="standard", icl_feature_map="identity",
)

# Tiny unit-test dims for the committed CI fixture.
_FIXTURE_KWARGS = dict(
    max_classes=3, embed_dim=16,
    col_num_blocks=1, col_nhead=2, col_num_inds=8,
    row_num_blocks=1, row_nhead=2, row_num_cls=2, row_rope_base=100000.0,
    icl_num_blocks=1, icl_nhead=2, ff_factor=2,
    dropout=0.0, activation="gelu", norm_first=True,
    col_attention_type="standard", col_feature_map="identity",
    row_attention_type="standard", row_feature_map="identity",
    icl_attention_type="standard", icl_feature_map="identity",
)

DIM_ROWS = ("rows", 4, 100_000)
DIM_TRAIN = ("train", 2, 100_000)
DIM_FEATURES = ("features", 2, 512)
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_TRAIN_FIXTURE = ("train", 2, 4096)
DIM_FEATURES_FIXTURE = ("features", 2, 64)


def assert_shipped_path(kwargs: dict) -> None:
    """Fail loudly if the config leaves the traced (standard-attention) path.

    The patch set in ``orion_bix_patches`` covers the modules the released
    weights instantiate. ``bi_axial`` / ``linear`` attention would pull in
    ``BiAxialAttention`` / ``LinearAttentionBlock``, whose own skip-branches are
    unpatched — the export would either fail or, worse, trace a subtly different
    graph.
    """
    for key in ("col_attention_type", "row_attention_type", "icl_attention_type"):
        if kwargs.get(key) != "standard":
            raise ValueError(
                f"{key}={kwargs.get(key)!r}: only 'standard' is on the exported path "
                "(the released Orion-BiX-v1.1 checkpoint uses standard attention "
                "throughout). Extend orion_bix_patches before exporting this config."
            )


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
    kwargs = dict(_FIXTURE_KWARGS)
    assert_shipped_path(kwargs)
    return ExportConfig(
        name="fixture", model_kwargs=kwargs,
        dim_rows=DIM_ROWS_FIXTURE, dim_train=DIM_TRAIN_FIXTURE,
        dim_features=DIM_FEATURES_FIXTURE,
        example=(20, 5, 12), parity_shapes=((40, 7, 30),),
    )


def real() -> ExportConfig:
    kwargs = dict(_REAL_KWARGS)
    assert_shipped_path(kwargs)
    return ExportConfig(
        name="real", model_kwargs=kwargs,
        dim_rows=DIM_ROWS, dim_train=DIM_TRAIN, dim_features=DIM_FEATURES,
        example=(20, 8, 12), parity_shapes=((16, 4, 10), (60, 20, 40)),
    )


def get(name: str) -> ExportConfig:
    if name == "fixture":
        return fixture()
    if name == "real":
        return real()
    raise ValueError(f"unknown config {name!r} (fixture|real)")
