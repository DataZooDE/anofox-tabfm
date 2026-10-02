# Causilo on ROCm: what the masked graph has to reproduce

Status: **inventory, masked forward, parity gate, masked export and structural guards done; no GPU compile yet.**
MIGraphX has only PARSED and reference-compiled the graph (CPU), which cannot see the failures that matter most. This is the Causilo counterpart of `docs/ROCM_TABPFN_PLAN.md`; read that for why the
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
4. **Five MIGraphX failures are guarded structurally from the start.** Four are GPU-pipeline-only and invisible
   to ORT and to MIGraphX's reference target: integer division on runtime data, `Reshape` to `[]`, a `Concat`
   whose inputs are all `Transpose`s, and a matmul with contraction dimension 1. The fifth is new, found on
   Causilo, and fails at PARSE time: ONNX Sequence operators (see "What parsing found" below). Causilo has the last one in
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

## Results (CPU only)

Gate (`mask_parity_full`, 11 cases x {fixture, fixture_r} x {classification, regression}, reference = the unpadded
positional wrapper): worst **7.0e-6** classification, **1.7e-5** regression, against a 1e-4 budget. All 28 negative
controls caught (7 mechanisms x 2 configs x 2 tasks), relative errors 0.54 to 2.2 against a 1e-3 threshold.

Exported masked ONNX under ORT at padded shapes, every row, against the unpadded reference: fixture classification
1.1e-5, regression 9.5e-7; **real dimensions** classification **4.5e-6**, regression **9.5e-7**. Every case differs
from the trace in train size, feature count and bucket; includes d = 2, one query row and one training row.

`pytest`: 110 passed, 6 skipped (the real-weights tests, gated on `TABFM_REAL_WEIGHTS`; run separately below).

**Real weights, padded, against upstream's own estimators** (`tests/test_real_weights.py`, gated on
`TABFM_REAL_WEIGHTS`; the weights are exported into a pytest tmp dir and never into the repo). The masked graph
with the actual Nums AI checkpoints, run in ORT with the table padded in BOTH axes, the bucket adding whole padded
feature groups (asserted), against `CausiloClassifier` / `CausiloRegressor` with `n_estimators=1`: iris, wine and a
5-class table agree to within 1e-4 in probability with identical labels; diabetes regression within 1e-4 of the
prediction scale. Random weights cannot tell a scrambled weight mapping from a right one; this can.

### A parity budget that was not loosened, and why it nearly was

The first real-dimension export FAILED parity at 1.42e-4 against 1e-4. Separating the sources at real dims showed
all three comparisons (masked-PyTorch vs unpadded, ORT vs masked-PyTorch, ORT vs unpadded) at 1e-5 to 1.7e-4, i.e.
**ORT and PyTorch disagreed with each other by 1.1e-4 on the same masked model**: float32 noise amplified through
a deep, sharply-attending network, not a mask error. The fix was to the TEST MODEL, not the budget:
`fixture.sensitive()` scales projections by 2, which at the released depth (12 prediction layers) amplifies 1e-6
relative input noise to 1.2e-4; a gain of 1.5 is still responsive (classification row spread 0.245, label
sensitivity 0.20) and amplifies it to 8.8e-6. `ExportConfig.weight_gain` carries the measured value (1.5 for
`real`). Regression at real dims is much less responsive in absolute terms (row spread 0.0105, outputs ~0.05); a
masking bug would still move it by orders of magnitude more than 1e-4.

### What parsing found: ONNX Sequence operators

Parsing the real masked graph with `migraphx-driver` failed in 3 seconds: **`Unknown operator: SplitToSequence`**.
Upstream's `Attention.prepare_context` does `packed.chunk(2, dim=-1)`, which exports to `SplitToSequence` +
`SequenceAt` (verified on a tiny module, static and dynamic shapes), and MIGraphX's ONNX parser has neither: the
masked graph could not even be LOADED on ROCm. ORT, the PyTorch gate and the MLX op-table check all accept it.
`_prepare_context` now takes the K and V halves as two slices (`Slice` is supported); the gate numbers are identical
to the digit. After the fix MIGraphX parses and reference-compiles both real graphs (rc 0, 38-43 s, bucket
T=128 H=16, CPU). The positional graphs that already ship on CPU/CUDA still contain `SplitToSequence`; that is
fine for ORT and for MLX (its interpreter implements both ops) and is not touched.

### What the MIGraphX checks CANNOT see (measured, not assumed)

* The reference target does **not** reproduce the GPU-pipeline failures: the known-bad TabPFN-3 K=1 repro compiles
  on `--ref`.
* There is no GPU-free target to substitute: `--cpu` fails here with `libmigraphx_cpu.so: cannot open shared
  object file`.
* So parse + reference compile is **necessary, not sufficient**. The structural guards (1-5) are the only
  pre-GPU evidence for the four pipeline failures, and each was shown to flag a deliberately bad graph. Whether the
  real graph has a sixth, unknown failure is exactly what the GPU run is for.

### Size, as a compile-time proxy

Masked real graphs: 9,398 (classification) / 9,436 (regression) nodes, against 8,573 / 8,585 positional and 10,480
for TabPFN-3, which took about 16 min per (T, H) bucket (Causilo's compile time is NOT measured). The duplicate
route processes `2T` rows, so run time (not node count) roughly doubles against the positional graph.

### The masked graph reuses the shipped tensor map

`tensor_map_causilo_<task>.json` from the masked export is byte-identical to the one already bundled, so the
weights-header hash (`ExpectedWeightsHeaderShaFor("causilo", ...)`) and the tensor map need no change.

## What is not done

No GPU compile, no bundling (nothing under `resources/`, no `cmake/embed_resources.cmake` entry), no
`tabfm_backends()` or test-list change, no timing on any device, no plugin involvement. `make_migraphx_graph.py` was
run only into a scratch directory to feed the parser. Whether the GPU pipeline accepts the graph is unknown; the
TabPFN conversion found four such failures only on the GPU.
