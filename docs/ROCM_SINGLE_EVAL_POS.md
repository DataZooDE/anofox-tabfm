# ROCm / MIGraphX and the `single_eval_pos` family

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
