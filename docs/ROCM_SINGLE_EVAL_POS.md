# ROCm / MIGraphX and the `single_eval_pos` family

> **Update 2026-09-30.** The whole TabPFN family -- v2, v2.5, v2.6 and v3 -- was
> converted and now runs on ROCm (docs/ROCM_TABPFN_PLAN.md), joining tabdpt
> (docs/ROCM_TABDPT_SPIKE.md). What remains positional and refused on ROCm:
> nothing, since 2026-11-01. `exaone-tabular` and `nori-30m` were added positional too and ship a second, masked
> graph for ROCm (tools/export_exaone, tools/export_nori: `--contract mask`), and so now do `tabicl-v2`,
> `limix-2m`, `orion-bix`, `orion-msp` (tools/export_tabicl, tools/export_limix, tools/export_orion_bix,
> tools/export_orion_msp: `--contract mask`; see "TabICL v2 and LimiX-2M" at the end) and `causilo`
> (tools/export_causilo `--contract mask`, docs/ROCM_CAUSILO_PLAN.md). The two Orion graphs
> needed only key masks -- Orion is TabICL's architecture, so these masks are the TabICL ones too: the inducing
> points and every ICL query attend to the context rows, the labels go to the context rows, padded columns are
> masked out of the row attention; Orion-MSP adds its PMA pooling keys and draws its frozen random links from the
> real columns only (an unrestricted top-k over the padded length hands real queries different links).
> `tabfm_backends()` is the source of truth.


Why nine of the eleven built-in models cannot be served on ROCm today, what it
would actually take, and why the reason recorded in
`ExpectedWeightsHeaderShaFor` is only half of it.

Spike, not a plan. Nothing here is implemented.

## Where we are

Only `tabfm-v1` and `mitra` bundle `graph_migraphx_*` graphs. Every other
built-in — TabPFN v2 / 2.5 / 2.5-real / 2.6 / 3, TabICL v2, Orion-BiX,
Orion-MSP, TabDPT — is CPU- or CUDA-only. CUDA is unaffected and complete: all
eleven bundle `graph_ext_*`.

## The mechanism that makes ROCm work today

MIGraphX compiles for a **fixed** input shape, so the plugin rounds each call up
to a bucket (`tabfm_shape_bucket.hpp`: T in {128, 512, 1024, 2048, 4096, 10000},
H in {16, 64, 128, 256, 512}) and caches one compiled `.mxr` per bucket. Padding
is only sound because the padded region is **semantically inert**, and it is
inert because the graph is told what is real:

```
tabfm-v1 :  x[1,T,H]  y[1,T]  train_size[1]  cat_mask[1,H]  d[1]
mitra    :  x[1,T,H]  y[1,T]  train_size[1]                 d[1]
```

- Padded **rows** sit past `train_size`, so the model treats them as query rows
  and the engine discards their predictions.
- Padded **features** are ignored because `d` says how many columns are real.

Both padding axes have a mask. That is the whole trick.

## Why the rest of the catalog cannot use it

The other nine models are the `single_eval_pos` family, and their contract is
just two tensors:

```
tabpfn-v2-5 :  x[1,T,H]  y[1,S]
tabdpt      :  x[1,T,H]  y[1,S]
orion-msp   :  x[1,T,H]  y[1,S]
```

Context membership is **positional**: the model computes `eval_pos = y.shape[0]`
and treats rows `[0:S)` as context, `[S:T)` as queries. Two consequences, and the
second one is not recorded anywhere:

1. **`S` is a shape dimension, not a value.** A different training-set size is a
   different compiled program. This is the reason the code comment gives.
   Padding `y` is not a way out: appending labels *adds fake context rows*, which
   changes the prediction — the opposite of inert.

2. **`H` cannot be padded either.** These graphs declare no `d` input, so there
   is nothing to tell the model that trailing feature columns are filler. The
   existing H bucket relies on a mask these models do not have.

So the blocker is not "one extra bucket dimension". Two of the three axes lose
their inertness argument. Only T padding survives (rows past `S` are queries
anyway).

## What would actually unblock it

Give the family the same contract the ROCm-servable models already have —
`x, y_full[1,T], train_size[1], d[1]` — and derive the split from the *value*
rather than from `y`'s length. The important part is that the split must become a
**mask**, not a slice:

```
# today (shape-dependent, uncompilable at fixed shape)
k = h[:, :eval_pos]

# needed (static shape, value-dependent contents)
is_context = arange(T) < train_size        # [T] bool, T is bucketed
attn_mask  = is_context[None, :]           # queries may attend to context only
```

A slice by a runtime value reintroduces data-dependent shapes and buys nothing;
a mask keeps every tensor's shape a function of the bucket alone, which is
exactly what MIGraphX needs. `d` masks the feature axis the same way.

This is an **export-time change, per model**, not a plugin-side tweak:

- each model's attention has to be patched to consume the mask instead of
  slicing (`tools/export_*/​*_patches.py`),
- parity has to be re-established against upstream for the masked form,
- the graphs, tensor maps, ext/migraphx variants, header shas and fixtures all
  regenerate.

That is roughly the size of the original onboarding for each model, times nine.

## Recommendation

Do not attempt it catalog-wide. Two narrower options, in order of value:

1. **Do nothing beyond documenting.** ROCm users have `tabfm-v1` and `mitra`,
   and `mitra` is Apache-2.0 and genuinely good. CUDA and MLX already cover the
   whole catalog. Make the limitation explicit and make
   `SET anofox_tabfm_device='rocm'` on an unsupported model say *why* rather than
   fall back silently — the repo's own rule is never to trust a silent fallback.

2. **If one model is worth it, pick `tabdpt`.** It is Apache-2.0 and ungated (so
   ROCm users can actually get the weights without a licence gate), it does both
   tasks, and its wrapper is already ours (`tools/export_tabdpt`) with the
   context split in one place — `eval_pos` is threaded through
   `TransformerEncoderLayer.forward`, which the exporter already patches for the
   attention scale. It is the cheapest one to convert and the most useful to
   ROCm users.

## What was checked

- `src/tabfm_migraphx_plugin.cpp` (bucket padding, `.mxr` cache, the
  "padded query rows carry the -100 label sentinel; padded features are masked by
  `d`" invariant) and `src/include/tabfm_shape_bucket.hpp` (the bucket table).
- The declared ONNX inputs of every bundled graph in `resources/`, which is where
  the missing `d` on the `single_eval_pos` family shows up.
- `ExpectedWeightsHeaderShaFor` in `src/include/tabfm_model_spec.hpp`, whose
  comment states the per-`train_size` half of the problem.

## Superseded: tabdpt has been converted

This document recommends documenting the ROCm limitation and, if any model is
worth converting, picking `tabdpt`. That has now been done, and the analysis
here was incomplete in one important way.

**The result:** tabdpt serves on ROCm — zero query-row disagreements against
the CPU, 103.2 ms → 18.59 ms at 100 rows (5.6x). See
[`ROCM_TABDPT_SPIKE.md`](ROCM_TABDPT_SPIKE.md) for the full account.

**What this document missed:** the positional split is not tabdpt's only
MIGraphX blocker. Its graph also contains 32 `SplitToSequence` and 64
`SequenceAt` ops — from a single `chunk` in the SwiGLU feed-forward — which
MIGraphX's parser does not implement, and which are present in the graph tabdpt
ships **today**, independent of the split. A third blocker then appeared in
MIGraphX's own reduction codegen.

So "cheapest to convert" was reasoned from one obstacle out of three. The
recommendation happened to be right; the estimate behind it was not. Any future
statement about which models are worth converting should be read with that in
mind — the split is what analysis can see, and it was the least of it.

## TabICL v2 and LimiX-2M (2026-11-01)

Both converted and verified on the gfx1201 (RX 9070 XT, MIGraphX 2.15, fp32, the T128 x H16 bucket): served by
`rocm:0`, **0/100** label differences from the CPU for both models, regression max |diff| 4.4e-5 (TabICL) and
3.5e-4 (LimiX, target scale 7.9). Cold compile 2.6-3.8 minutes per task and bucket; warm calls 0.02 s against
0.11-0.14 s on the CPU at 100 rows. `test/sql/tabfm_rocm_real_models.test` (TABFM_REAL_WEIGHTS + TABFM_ROCM)
and `tools/gpu_test/scenarios/tabicl_limix_rocm.sql` are the checks.

As this document predicted, the two share an approach -- the masked contract (`x, y_full, train_size, d,
n_rows`) the TabPFN, EXAONE and Nori conversions use -- but the places the split hides differ:

- **TabICL** (tools/export_tabicl/tabicl_mask.py): the circular feature grouping `(j + 2**i) % H` is a
  function of the WIDTH, so padding H changes every group's partners; it is taken modulo `d`, division-free,
  and gathered. The target-aware column embedding and the ICL label embedding go to the context rows only;
  the column ISAB's inducing points and the ICL transformer attend to the context through a key mask; QASSMax
  reads `log(S)` from the value; the row interactor's own `d` mask removes padded groups.
- **LimiX-2M** (tools/export_limix/limix_mask.py): Nori's recipe (Nori is its fork) plus one place Nori does
  not have: `ValidFeatureEncoder` tests "every row equals the first" over the whole table, so a zero padded
  row would make a constant column valid. That test reads `n_rows`, the first model here that needs it for
  a statistic over the real (not the context) rows.

What MIGraphX could not take, each rewritten to the same function (and now caught before a GPU run by
`tests/migraphx_guards.py`, shared with tools/export_causilo):

| construct | where | MIGraphX failure | rewrite |
|---|---|---|---|
| `Tensor.split` of the packed q/k/v weight (cross-attention) | TabICL ISAB, row interactor | parser: no SplitToSequence | slices |
| RoPE `einsum("..., f -> ... f")` | TabICL row interactor | (avoided; no shipped graph uses Einsum) | broadcast multiply |
| `reshape(())` of log(S) | TabICL QASSMax | parser: "reshape has 0 elements whereas the input has 1"; the plugin then crashed | keep it `[1]` |
| `(x == v).all(...)` skip tests | TabICL SkippableLinear / ISAB | GPU codegen: `reducer ... type 'float' does not provide a subscript operator`, then the `tuning_config` assertion -- the TabDPT reducer bug, in a fused `reduce min` | count by GEMM against a ones vector |
| `Linear(1, n)` | TabICL QASSMax `base_mlp`, regression y-encoders | (guard; failed on Nori and Causilo) | elementwise |
| `torch.cat([t, t])` | LimiX doubled context | (guard; preceded a `fuse_concat` assertion on Causilo) | expand + reshape |

Remaining positional on ROCm: `orion-bix`, `orion-msp`, `causilo` (each being converted on its own branch at
the time of writing).
