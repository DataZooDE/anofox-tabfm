"""Export configurations for TabPFN-3.5.

`REAL35` / `REAL35_FAST` are transcribed from the `config` entry of the
safetensors `__metadata__` block of the released checkpoints
(Prior-Labs/tabpfn_3_5 :: tabpfn-v3.5-20260909.safetensors and
tabpfn-v3.5-fast-20260909.safetensors); the weights are never committed.
tests/test_configs.py checks the transcription against a downloaded checkpoint
when one is present. The two differ ONLY in `nlayers` (24 vs 8).

One checkpoint carries both heads (a 160-class many-class decoder AND a
5000-bucket regression head), so there is one architecture config per variant and
the task picks the head at export time.
"""

from __future__ import annotations

import dataclasses

_REAL35_COMMON = dict(
    cell_ecdf_num_buckets=8192, cell_ecdf_num_frequencies=4, cell_embed_row_chunk_size=2048,
    decoder_head_dim=64, decoder_num_heads=6, decoder_use_softmax_scaling=True,
    dist_embed_num_blocks=3, dist_embed_num_heads=8, dist_embed_num_inducing_points=128,
    embed_dim=128, feat_agg_num_blocks=3, feat_agg_num_cls_tokens=8, feat_agg_num_heads=8,
    feat_agg_rope_base=100000.0, feature_group_size=3, ff_factor=2,
    fourier_encoding_num_frequencies=32, icl_num_heads=16, icl_num_kv_heads_test=1,
    inference_col_chunk_size=4, max_num_classes=160, num_buckets=5000,
    softmax_scaling_mlp_hidden_dim=64,
)
REAL35 = dict(_REAL35_COMMON, nlayers=24)
REAL35_FAST = dict(_REAL35_COMMON, nlayers=8)

# Tiny dims for the committed CI fixture. Chosen to exercise every branch the
# released model takes, at small sizes: test rows on ONE KV head, a feature group
# of 3 (shifts 1, 2, 4 wrap more than once on narrow inputs), more than one block
# in the distribution embedder and the aggregator, more than one CLS token, the
# decoder's softmax scaling, a class width that needs padding to a multiple of the
# decoder head dim -- and an ECDF bucket count (6) BELOW the fixture's train-row
# count, so the graph also runs upstream's interpolating ruler, which the released
# model only reaches past 8192 train rows.
FIXTURE35 = dict(
    embed_dim=16, nlayers=2,
    dist_embed_num_blocks=2, dist_embed_num_heads=2, dist_embed_num_inducing_points=6,
    feature_group_size=3, feat_agg_num_blocks=2, feat_agg_num_heads=2,
    feat_agg_num_cls_tokens=2, feat_agg_rope_base=10000.0,
    icl_num_heads=2, icl_num_kv_heads_test=1,
    decoder_head_dim=3, decoder_num_heads=2, decoder_use_softmax_scaling=True,
    softmax_scaling_mlp_hidden_dim=8, ff_factor=2,
    fourier_encoding_num_frequencies=4, cell_ecdf_num_frequencies=2, cell_ecdf_num_buckets=6,
    max_num_classes=4, num_buckets=16,
)


@dataclasses.dataclass(frozen=True)
class ExportConfig:
    name: str
    model_kwargs: dict
    slug: str                      # resource stem: graph_<slug>_<task>.onnx
    example: tuple = (16, 6, 10)   # (T, H, N) traced at export
    parity_shapes: tuple = ((32, 7, 20),)


CONFIGS = {
    "fixture35": ExportConfig("fixture35", FIXTURE35, "tabpfn35", example=(12, 5, 8),
                              parity_shapes=((40, 7, 30), (16, 9, 6), (20, 2, 13))),
    "real35": ExportConfig("real35", REAL35, "tabpfn35"),
    "real35fast": ExportConfig("real35fast", REAL35_FAST, "tabpfn35fast"),
}


def get(name: str) -> ExportConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise ValueError(f"unknown config {name!r} (choose from {sorted(CONFIGS)})") from None
