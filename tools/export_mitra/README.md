# export_mitra — Mitra (AWS AutoGluon Tab2D) → weight-free ONNX

WS-A exporter for onboarding **Mitra** (`autogluon/mitra-classifier`,
`autogluon/mitra-regressor`, Apache-2.0) into the anofox-tabfm engine. Mirrors
`tools/export_onnx` (the TabFM/S01 pipeline): produces a **weight-free** ONNX
graph + a tensor-name map, keeping NO Google/AWS weight bytes in the repo.

```bash
uv sync
# shipping graphs (real dims) -> resources/
uv run export_mitra --task classification --config real --out ./out
uv run export_mitra --task regression     --config real --out ./out
# random-init CI fixture (+ tiny safetensors + golden slice)
uv run export_mitra --task classification --config fixture --out ./fix \
    --emit-weights ./fix/model_classification.safetensors \
    --emit-golden  ./fix/golden_classification.json
uv run pytest   # fast end-to-end + license-wall guard
```

## Exported contract (engine feeds by NAME)

```
inputs:  x [1,T,H] f32   (preprocessed, feature-padded to H)
         y [1,T]   f32   (labels; test rows hold any value)
         train_size [1] i64   (# context/train rows = prefix of T)
         d [1] i64            (active feature count <= H)
output:  logits [1,T,C] f32   (C = 10 classification, 1 regression)
```

**`cat_mask` is OMITTED** — Mitra has no categorical embedding; categoricals
are ordinal-encoded and treated numerically. `d` builds Mitra's feature-padding
mask; `train_size` splits support/query via masks (no data-dependent slicing).
The engine reads predictions on the test rows (`rows >= train_size`).

## What is patched (`mitra_model_patched.py`)

A byte-name-identical re-implementation of Mitra's **CPU forward path**
(`Tab2D` + embeddings) so the released safetensors load unchanged, with the
non-exportable glue rewritten:

| upstream op | rewrite |
|---|---|
| `torch.utils.checkpoint.checkpoint` | plain layer call (inference) |
| `einx` / `einops.pack` / `einops.unpack` | plain torch reshape/cat/permute |
| `torch.quantile(x, q=999, dim=1)` | sort + gather + lerp (over valid rows) |
| `torch.vmap(torch.bucketize)` | broadcast compare + sum |
| in-place boolean-mask assignment | `torch.where` |

Two semantic generalizations over the upstream CPU branch (which ignores its
padding masks and is only correct when support/query are pre-sliced): the
quantile embedding excludes padded support rows from its statistics, and
attention masks padded support rows (row attention) / padded feature columns
(feature attention) as keys. Both reduce to the upstream result at zero padding
and match the GPU flash-attention branch — this is what lets the graph take a
single `[1,T,H]` table + `train_size`/`d` instead of pre-split tensors.

Verified byte-identical (max abs diff **0.0**) to the unmodified upstream
`Tab2D` CPU forward on real weights, both tasks.
