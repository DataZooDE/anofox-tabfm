"""Export configurations: fixture (tiny) / real (the released v1.1 architecture).

The exported ONNX graph is architecture-only and weight-free, so ``real`` builds
the model with RANDOM weights at the dims the released checkpoint declares — no
Lexsi checkpoint bytes are ever involved.

``_REAL_KWARGS`` is transcribed from the ``config`` dict embedded in
``Lexsi/Orion-MSP::OrionMSP-classifier-v1.5-202603.ckpt`` (361 tensors).

Two of those values are worth flagging, because both are places where the
release is narrower than the paper:

* ``row_scales = (1,)`` — the "multi-scale" row interaction ships with a SINGLE
  scale. The 1/4/16 hierarchy the paper describes is not what the published
  weights use.
* ``perc_num_latents = 0`` — the Perceiver memory is disabled.

``row_num_random = 2`` IS live, which is what forces the frozen-mask patch in
``orion_msp_patches`` (upstream redraws it per call with ``torch.randperm``).

``assert_shipped_path`` enforces the assumptions, so a future checkpoint that
flips one fails loudly rather than silently exporting a different architecture.
"""

from __future__ import annotations

import dataclasses

OPSET = 18

# Verbatim from OrionMSP-classifier-v1.5-202603.ckpt["config"].
_REAL_KWARGS = dict(
    max_classes=10, embed_dim=128,
    features_per_group=2, feature_pos_emb="subspace",
    col_num_blocks=3, col_nhead=4, col_num_inds=128,
    row_num_blocks=9, row_nhead=8, row_num_cls=4, row_num_global=4,
    row_num_random=2, row_window=500, row_scales=(1,),
    row_group_mode="pma", row_rope_base=100000.0,
    scale_combine_method="enhanced_attention",
    icl_num_blocks=12, icl_nhead=4, ff_factor=2,
    num_memory_heads=4, num_thinking_tokens=4,
    perc_layers=2, perc_num_latents=0, use_memory_gating=True,
    dropout=0.0, activation="gelu", norm_first=True,
)

# Tiny unit-test dims for the committed CI fixture. row_num_random is KEPT at 2:
# the frozen-mask patch is the whole reason this exporter differs from
# export_orion_bix, so a fixture with it switched off would not exercise it.
_FIXTURE_KWARGS = dict(
    max_classes=3, embed_dim=16,
    features_per_group=2, feature_pos_emb="subspace",
    col_num_blocks=1, col_nhead=2, col_num_inds=8,
    row_num_blocks=1, row_nhead=2, row_num_cls=2, row_num_global=2,
    row_num_random=2, row_window=500, row_scales=(1,),
    row_group_mode="pma", row_rope_base=100000.0,
    scale_combine_method="enhanced_attention",
    icl_num_blocks=1, icl_nhead=2, ff_factor=2,
    num_memory_heads=2, num_thinking_tokens=2,
    perc_layers=1, perc_num_latents=0, use_memory_gating=True,
    dropout=0.0, activation="gelu", norm_first=True,
)

DIM_ROWS = ("rows", 4, 100_000)
DIM_TRAIN = ("train", 2, 100_000)
DIM_FEATURES = ("features", 2, 512)
DIM_ROWS_FIXTURE = ("rows", 4, 4096)
DIM_TRAIN_FIXTURE = ("train", 2, 4096)
DIM_FEATURES_FIXTURE = ("features", 2, 64)


def assert_shipped_path(kwargs: dict) -> None:
    """Fail loudly if the config leaves the traced (standard-attention) path.

    The patch set in ``orion_msp_patches`` covers exactly the shape the released
    checkpoint instantiates. Each assumption below, if broken, would either fail
    the export or -- much worse -- trace a subtly different architecture that
    still loads the real weights and quietly predicts wrong.
    """
    scales = tuple(kwargs.get("row_scales", (1,)))
    if scales != (1,):
        raise ValueError(
            f"row_scales={scales!r}: the released checkpoint ships a SINGLE scale, "
            "and only that path is traced. A genuine multi-scale config would run "
            "the row stage once per scale and combine them; extend "
            "orion_msp_patches (and re-check the frozen mask, which is built per "
            "sequence length) before exporting it."
        )
    if kwargs.get("perc_num_latents", 0) != 0:
        raise ValueError(
            f"perc_num_latents={kwargs.get('perc_num_latents')!r}: the released "
            "checkpoint disables the Perceiver memory, so PerceiverMemory is not on "
            "the traced path and its branches are unpatched."
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
