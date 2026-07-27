# export_limix — LimiX-2M (StableAI) → weight-free ONNX — **INCOMPLETE SPIKE**

> **Status: not shipped.** `limix-2m` is deliberately **not** in the built-in
> registry, not in `cmake/embed_resources.cmake`, and has no fixture or
> sqllogictest. The extension builds and tests green without it. This directory
> is the export spike, checked in so the analysis below is not lost.

WS-A exporter for onboarding **LimiX-2M** (`stable-ai/LimiX-2M`) into the
anofox-tabfm engine. Upstream ships no packaging metadata, so it is pinned as
the `vendor/limix` git submodule (the convention `vendor/tabfm` already uses)
and patched at runtime — no upstream source is copied into this repo.

```bash
git submodule update --init vendor/limix
uv sync
uv run export_limix --task classification --config fixture --out ./out   # traces, then FAILS in ORT
```

## Why it is not shipped

The graph **traces and exports** (43 initializers mapped, weight-free), but the
exported graph is **not shape-correct**: a reshape in the feature-grouping step
binds the sequence length to `T_example + (S − S_example)` instead of the row
count `T`, so ORT fails at any shape other than the export example:

```
T=20,H=6,S=12 (export example)  -> OK
T=20,H=6,S=15                   -> Reshape wants {1,23,3,2} from {1,20,6}   (23 = 20+(15-12))
T=30,H=6,S=12                   -> Reshape wants {1,20,3,2} from {1,30,6}   (20 = T_example)
T=40,H=8,S=30                   -> Reshape wants {1,38,4,2} from {1,40,8}   (38 = 20+(30-12))
```

The row dim `T` and the train dim `S` are being unified into one symbol
somewhere between `forward`'s `batch_size, seq_len, num_feature = x.shape` and
the `x[k].reshape(batch_size, seq_len, ...)` regrouping — most likely via the
`y[k].shape[1] < x["data"].shape[1]` guard and the `T - S` padding width. Fixing
it means either driving the reshape from `x[k].shape[1]` directly rather than
the captured `seq_len`, or splitting the y-padding so the two dims never meet.
That is the one remaining blocker; everything below it is solved.

## Effort assessment

LimiX turned out **heavier than the other onboardings**. TabPFN-2.5, TabICL and
Orion-BiX each needed 2–3 surgical monkeypatches; LimiX needs a full patched
re-implementation of `FeaturesTransformer.forward` plus encoder rewrites — the
Mitra tier — and then the dynamic-shape debugging above. It should be re-scoped
as its own piece of work rather than treated as a drop-in.

## What is already solved (patches 1–4, `limix_patches.py`)

The forward maps cleanly onto the engine's existing `(x, y)`-only contract:
upstream's `forward(x, y, eval_pos, task_type=...)` already takes a row-major
table plus a label prefix and a split point, so **no engine change is needed**.
Both tasks run in PyTorch through `ExportWrapper` (classification → `[1,T,10]`,
regression → `[1,T,1]`).

| # | upstream | rewrite | why |
|---|---|---|---|
| 1 | two `if torch.isnan(...).any(): raise` guards in `forward` | proxy whose `.any()` is statically False, installed only inside `model.transformer` | data-dependent branch; pure input validation, and the config enables the NaN encoders |
| 2 | `mixed_y_embedding` splits labels by `y_type` with boolean-mask indexing | call the single live encoder directly | `forward` builds `y_type` as all-zeros or all-ones from `task_type`, so the split is the identity; keeps upstream's float16 round-trip |
| 3 | `y["data"][:, eval_pos:] = torch.nan` | `torch.where` against a row-index mask | in-place dynamic slice assign bakes `index_put` operand shapes |
| 4 | `NanEncoder.forward`'s four boolean-mask assignments | `torch.where` | same `index_put` shape-baking; elementwise-identical |

## Architecture facts (from `LimiX-2M.ckpt`'s embedded `config`)

2,377,837 params over 137 tensors, 12 layers, `embed_dim` 96, `layer_arch`
`"smf"`. Two flags keep the traced path narrow and are pinned by
`configs.assert_shipped_path`:

- `mask_prediction = False` — the imputation head is off, so `forward` returns
  logits rather than a dict with `feature_pred` and preprocessing statistics.
- `feature_positional_embedding_type = "none"` — avoids the runtime
  `torch.randn` positional draw at `transformer.py:277`, untraceable at a
  symbolic column count.

Checkpoint keys are in the **bare** module namespace (no `_orig_mod.` prefix,
unlike Orion-BiX), so `export.CKPT_KEY_PREFIX` is `""`.

## Licensing (decided, if it ever ships)

Upstream **code** is Apache-2.0 (`vendor/limix/LICENSE.txt`). The **weights**
are not: the model card says "fully open for academic research; commercial use
requires official authorization from StableAI" — note this contradicts the HF
repo's `apache-2.0` card tag. It would therefore ship `commercial: false` +
`gate_setting: "accept_hf_license"`, the `tabfm-v1` treatment, with the conflict
recorded in the manifest attribution.
