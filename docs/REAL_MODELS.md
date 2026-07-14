# Real tabular foundation models in `anofox_tabfm`

Status of onboarding real (non-fixture) tabular foundation models behind the
extension's fixed ONNX engine contract. The registry proves FR-5.1 / M4 ("a
second model is a manifest, not new C++") — but "not new C++" only holds when a
model's forward maps onto the engine's existing input-feeding + preprocessing.
That is exactly what separates Mitra (drop-in) from TabPFN v2 / TabICL (need
engine work).

The engine feeds a fixed signature and preprocessing:
`x[1,T,H] f32, y[1,T] f32, train_size[1] i64, cat_mask[1,H] bool, d[1] i64 →
logits[1,T,C]`, with `tabfm_v1_minimal` preprocessing (z-score + first-appearance
ordinal). It feeds inputs by name and only feeds names the graph declares.

## Summary

| model | license | status | runs today | notes |
|---|---|---|---|---|
| **Google TabFM** (`tabfm-v1`) | non-commercial (gated) | ✅ shipped | yes | the original; 1.6 B params / 6.56 GB |
| **Mitra** (`mitra`) | Apache-2.0 (commercial) | ✅ shipped | **yes — zero C++ changes** | 72 M / ~303 MB; iris 0.962 in ~2.4 s |
| **TabPFN v2** (`tabpfn-v2`) | Prior Labs (Apache-2.0 + attribution) | ✅ shipped | **yes — classify + regress** | ~29 MB; iris **0.962**, wine MSE **0.482**; one-time ckpt→safetensors convert |
| **TabICL v2** (`tabicl-v2`) | BSD-3-Clause (commercial) | ✅ shipped | **yes — classify + regress** | ~110 MB; iris **0.962**, wine MSE **0.586**; one-time ckpt→safetensors convert |

## Mitra — done

Real Tab2D transformer, exported weight-free with exact parity (graph + injected
real weights vs upstream: 4.6e-6 classification, 1.5e-5 regression, 100% argmax).
It maps onto the engine with **no C++ change** because:
- Its exported contract is `x, y, train_size, d → logits` — a subset of what the
  engine already feeds (it simply doesn't declare `cat_mask`).
- It **rank/quantile-normalizes inside the graph**, so the engine's z-score
  preprocessing is harmless (a rank transform is invariant to monotonic external
  transforms).

Registered via `examples/mitra.json`; offline fixture `test/sql/tabfm_mitra.test`;
head-to-head in `examples/compare_models.sql`. Downloadable from HF
(`autogluon/mitra-{classifier,regressor}`, per-file URLs in the manifest),
Apache-2.0, ungated.

## TabPFN v2 — shipped (classification)

TabPFN v2's ONNX contract differs from TabFM's: its graph takes only `(x, y)`
and derives the train/test split from `len(y)` (`single_eval_pos`), so `y` is the
training-label prefix. Two small, generic engine features made it run:

1. **`y`-as-train-prefix feeding** (`Run`, `tabfm_ort_engine.cpp`). A graph that
   declares *no* `train_size` input must derive the split from `len(y)`, so the
   engine feeds `y` as `[1, train_size]` instead of the full `[1, T]`. This is
   *inferred* from the graph's inputs — no manifest flag, no schema change — and
   is inert for TabFM/Mitra (which do declare `train_size`).
2. **`*_raw` preprocessing profile** (`PreprocessBatch`). A model whose
   `preprocessing_profile` ends in `_raw` skips the z-score/outlier stages
   (features are ordinal-encoded + NULL-imputed but passed through). TabPFN
   declares `tabpfn_v2_raw`. (Measured: z-score is actually harmless to TabPFN
   too, but respecting the declared profile is the correct behavior.)

Real-weight run: `tools/export_tabpfn/convert_weights.py` downloads the HF `.ckpt`
(pickle) and writes a safetensors keyed by the committed tensor map into the
extension cache (a one-time, dev-side step — the extension stays pure C++/ORT).
Then `model := 'tabpfn-v2'` scores **iris at 0.962** (matching the Python
reference), ~29 MB weights.

**Regression** works too: the bar-distribution mean (`softmax(bucket_logits) ·
bucket_centers`, with TabPFN's half-normal tail correction and target
de-standardization) is **baked into the graph** at export, so it outputs a plain
`[1,T,1]` point estimate — no model-specific C++ decoder. The graph is
**self-contained (raw-in / raw-out)**: it consumes raw training targets and emits
raw predictions, which the engine's `*_raw` profile honors (feed raw target, skip
the inverse-transform). Real weights: `california_housing` R² 0.827 (agent
parity), `mstz/wine` MSE **0.482** vs 0.860 baseline. The bucket borders live in
the checkpoint (`regression_borders`), so the classification and regression graphs
have different initializers — the manifest uses a **per-task** `graph.tensor_map`.

## TabICL v2 — shipped (classification)

TabICL's graph is *also* `(x, y)`-only, so the same y-prefix inference drives it
with no model-specific code. `tools/export_tabicl/convert_weights.py` downloads
the HF `.ckpt` and writes a safetensors keyed by the committed tensor map (all
391 keys matched the checkpoint's `state_dict` directly). `model := 'tabicl-v2'`
scores **iris at 0.962** (~110 MB, BSD-3, ungated) under the `tabicl_v2_raw`
profile — raw features (its internal normalization prefers them, and they edge
out z-score here).

**Regression** works via the same in-graph reduction: `TabICLRegressor`'s point
estimate is the **mean over the 999 quantiles** (sort-invariant, so no in-graph
sort needed), with the target StandardScaler + inverse baked in → a
self-contained `[1,T,1]` raw-in/raw-out graph. Real weights: `california_housing`
R² 0.821 tracking the sklearn regressor at corr 0.9999; `mstz/wine` MSE **0.586**
vs 0.860 baseline. Note the checkpoints differ (`bias_free_ln`: classifier 391
tensors, regressor 347), so the exporter is task-aware; the regressor keys are a
subset of the classifier's, so a shared tensor_map still drives both graphs.

## License wall (all models)

Every shipped graph is weight-free (all checkpoint initializers externalized,
`.onnx.data` deleted); every fixture is seeded random-init. No real weight bytes
from any model are committed. Real weights are the user's own HF download behind
the manifest's license gate.
