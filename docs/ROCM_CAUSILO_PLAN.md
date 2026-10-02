# Causilo on ROCm: what the masked graph has to reproduce

Status: **inventory + masked forward + parity gate, no GPU run yet.** Nothing here has been compiled
by MIGraphX. This is the Causilo counterpart of `docs/ROCM_TABPFN_PLAN.md`; read that for why the
split must become a value (MIGraphX compiles one program per fixed `(T, H)` bucket, so a split read from
`y`'s length would need one compile per distinct training size).

Every claim below was checked against `causilo` 1.0.3 (the pinned release `94f2bd91`) rather than
inherited from the plan; line numbers are in `causilo/` unless noted.

## The contract

```
x          [1, T, H] f32   engine-preprocessed, real data in [:n_rows, :d], zeros elsewhere
y          [1, T]    f32   training targets in [:train_size], -100 elsewhere
train_size [1] int64       rows [0, train_size) are the context
n_rows     [1] int64       rows [0, n_rows) are real (accepted for plugin parity; see "not needed")
d          [1] int64       columns [0, d) are real
```

Output `[1, T, C]` is the same as the positional graph's: every row is a prediction, rows `< train_size`
are in-context fitted values (the context presented a second time as queries; see `causilo_patches.py`).

## Are padded features inert? Measured: no, once the padding spans a whole group

`FeatureEmbedding` (nn/embeddings.py:24) pads the feature axis to a multiple of 3 with **numeric zero**,
turns each cell into `sin/cos` of `2*pi*freq*value`, and sums the three cells of a group. A bucket wider
than the real feature count therefore adds either (a) zero cells inside the last real group, or (b) whole
extra groups. Measured on the shipped positional wrapper with a responsive random-weight model, `d=5`,
against the unpadded run (max abs difference, `|logit|` up to 1.48):

| padding | classification | regression |
|---|---|---|
| H=6: one zero column inside the partial group | **0.0** | **0.0** |
| H=9: one all-zero group | 1.10 | 0.22 |
| H=16: four all-zero groups | 1.51 | 0.30 |
| T=48: 28 zero rows appended as extra query rows | **0.0** | **0.0** |

(a) is inert **by construction**: upstream's own padding is a numeric zero too, and the in-graph
normaliser maps an all-zero column to exactly zero (`spread -> 1`, bounds `+-4e-6`). (b) is not: an
all-zero group is a real token (`cos(0) = 1` for every frequency, so it is not even a zero vector), and the
row stages read every group. Extra query rows are inert because query rows never influence anything
(they are only ever queries).

## Sites that consume the split or a real count

Numbering follows the data flow of `ModelRunner.predict` (execution/runner.py:86).

| # | where | what it does with the split / count | masked form |
|---|---|---|---|
| 1 | `normalise_features` (ours, `causilo_patches.py`; mirrors data/normalization.py) | fill mean, z-score centre/spread, 4-sigma bounds, all over the **first S rows**; `ddof = (S > 1)`; counts `S` | statistics over `train` mask, counts as values |
| 2 | `inject_targets` (runner.py:33), feature-level | `rows[:, :S] + target_embedding`, `S = targets.shape[1]` | add `where(train, emb, 0)`; non-train targets zeroed first (class id -100) |
| 3 | `ColumnBlock.forward` (nn/column.py:33-36) | `layer.prepare(rows[..., :S, :])`: the gather attention's keys are the first S rows | key mask `train` over all T rows |
| 3b | `QueryScale` in that gather (nn/layers/scaling.py:25-27) | `log(key_count)`, `key_count = S` | `log(clamp(n_train))` as a **value** |
| 4 | `ColumnLayer.broadcast` | all rows query the 128 latents; no scaling | none (rows independent, keys fixed) |
| 5 | `RowBlock` / `RowLayer.update_latents` / `RowPool` (nn/row.py:26-27, 53) | `LatentUpdate` reads `cat(latents, features)`: **keys include every feature group** | group key mask `j*3 < d`; `key_count = L + n_real_groups` |
| 5b | `FeatureUpdate` (nn/row.py:40, 53) | queries = latents + groups, keys = the L latents only | none (keys fixed; padded query tokens are independent garbage nobody reads) |
| 6 | `inject_targets`, row-level | the same prefix add on the pooled `[L*D]` rows | same mask as #2 |
| 7 | `PredictionBlock.forward` (nn/prediction.py:20-26) | `layer.prepare(rows[..., :S, :])` keys = first S rows, in **every** of the 12 layers; the last layer takes `rows[..., S:, :]` | key mask `train` per layer; all rows run, the answer rows are selected after |
| 7b | `QueryScale` in the prediction attention | `key_count = S` | value |
| 8 | regression target scaler | mean / population std over the S targets (`causilo_patches.ExportWrapper`) | masked sums, counts as values |
| 9 | `FeatureEmbedding` grouping (nn/embeddings.py:24) | pad to a multiple of 3, group, sum | unchanged; group count `G = ceil(H/3)` is shape-derived, **real** groups from `d` |

### Not needed

* **`n_rows`.** Nothing in Causilo looks at all rows: statistics use the training rows only and every
  other attention uses a key mask. (TabPFN needed it because constant-feature detection compares every row
  with row 0.) The graph still declares the input so the plugin binds it uniformly, exactly as the TabPFN-3
  masked forward does.
* **Constant-column removal.** Upstream's `FeatureEncoder` drops constant columns, but so does the engine
  before the graph, so the graph never sees them. The in-graph normaliser still handles one (spread -> 1).

### The key counts are values, and one of them is the padded-feature trap

`QueryScale` scales the query by `log(key_count)`. Upstream reads `key_count` off `key.shape[-2]`, which under
a bucket is the **padded** length. Three different counts are involved and they must not be confused:

* gather and prediction attention: keys are the training rows -> `N = train_size`;
* `LatentUpdate` (row stage and pool): keys are `L` latents plus the real groups -> `L + ceil(d/3)`;
* `FeatureUpdate`: keys are the latents only -> `L`, an architectural constant.

The second is the one that depends on `d`: using the padded group count there would be a bug that padding
alone (inert for the partial group) never exposes.

## Design decisions

1. **Nothing upstream is monkeypatched in the masked path.** Every block is re-driven from its own weights,
   like `tabpfn_mask_v3.py`. The module-level patches in `causilo_patches.apply()` (which exist for the
   positional export) are not called, so the masked graph cannot depend on whether the process happened to
   patch the right module; `tests/test_mask_export_isolated.py` proves it by exporting in a fresh process.
2. **No integer division or modulo on runtime data.** Real groups are `arange(G) * 3 < d`; their count is
   a `sum`. (MIGraphX's GPU target miscomputes integer floor-division; see `ROCM_TABPFN_PLAN.md`.)
3. **The duplicate route stays.** The context rows are presented a second time as queries
   (`[x ; x]` with `train = idx < N`), so the answer for row `i` is `out[T + i]`. Statistics are computed once
   on the original `x`, before duplicating, as in the positional wrapper.
4. **Four MIGraphX GPU-only failures are guarded structurally from the start** (none is visible to ORT or to
   MIGraphX's reference target): integer division on runtime data, `Reshape` to `[]`, a `Concat` whose
   inputs are all `Transpose`s, and a matmul with contraction dimension 1. Causilo has the last one in
   **two places per attention layer**: `QueryScale.length` and `QueryScale.correction` both start with
   `Linear(1, n)` on a `[1, 1]` input, plus the regression `TargetEmbedding` (`Linear(1, width)`).
   Each is written elementwise (`causilo_mask_forward._linear_k1`).

## The gate

`export_causilo.mask_parity_full`: the reference is the **unpadded** `ExportWrapper` (itself validated against
upstream's `ModelRunner` and `Normalizer` in `tests/test_export.py`); the candidate is the masked wrapper on the
same data padded to a bucket. They must agree on every real row, context and query reported separately.

A gate that cannot fail is worse than none, and this codebase has been blinded by it repeatedly, so:

* the model is `fixture.sensitive()`: a default-initialised Causilo ignores its input, so every comparison
  against it passes vacuously (measured earlier: logits vary by 0.008 across rows);
* a "real-like" fixture runs every loop the released model runs (depths 3/3/3/3, 4 row latents, more than one
  head) at small width, so no branch is unexercised;
* each mechanism has a negative control that must be **caught** (see the table the gate prints).

## What is not done

No GPU compile, no `make_migraphx_graph`, no bundling, no `tabfm_backends()` change, no timing. Whether
MIGraphX accepts the graph is unknown; the TabPFN conversion found four such failures only on the GPU.
