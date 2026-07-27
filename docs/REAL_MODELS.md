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
| **Orion-BiX** (`orion-bix`) | MIT (commercial) | ✅ shipped | **yes — classify only** | 27 M / ~315 MB .ckpt read natively; upstream ships no regressor |
| **TabPFN-2.5** (`tabpfn-v2-5`) | tabpfn-2.5-license-v1.1 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 10.7 M clf / 10.2 M reg; 24- and 18-layer heads; ckpt→safetensors convert required |

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

## Orion-BiX — shipped (classification only)

A TabICL descendant from Lexsi Labs, and the **only MIT-licensed** model in the
catalog: commercially usable with no license gate at all. Its graph is `(x, y)`-only
too, so the same y-prefix inference drives it with **no model-specific C++**.

It is the first built-in with a **single capability** — upstream ships
`sklearn/classifier.py` and no regression head — so `capabilities: ["classify"]`
and `tabfm_regress(..., model := 'orion-bix')` raises the actionable
unsupported-task error (asserted in `test/sql/tabfm_orion_bix.test`).

Three things the export had to establish, all recorded in
`tools/export_orion_bix/README.md`:

1. **The name is misleading.** `Orion-BiX-v1.1.ckpt`'s embedded `config` sets
   `col/row/icl_attention_type = "standard"`, so `BiAxialAttention` and
   `LinearAttentionBlock` are never instantiated by the released weights.
   `configs.assert_shipped_path` fails loudly if a future checkpoint changes that.
2. **Keys carry an `_orig_mod.` prefix** (saved from a `torch.compile`-wrapped
   module). Since `src/tabfm_ckpt.cpp` unwraps `state_dict` but does not rewrite
   parameter names, the tensor map records the prefixed keys —
   `convert_weights.py` confirms **277/277** map keys are present in the real
   checkpoint.
3. **RoPE caching had to be disabled** for export: the freq cache becomes a
   FakeTensor under tracing, and would have baked the export example's row count
   into the rotary table.

Unlike TabPFN/TabICL it does **not** normalize internally (its sklearn wrapper
uses `CustomStandardScaler` / `RTDLQuantileTransformer` externally), so it
declares `orion_bix_v1_minimal` — a normal engine-standardizes profile, not
`_raw`. Export parity **2.24e-07**. The ~315 MB `.ckpt` is read natively by the
extension; no conversion step is needed.

## TabPFN-2.5 — shipped (classify + regress)

The 2.5 line keeps v2's `(x, y)`-only contract, so onboarding it needed **no
engine change at all** — it is purely an exporter change. `tools/export_tabpfn`
is now parameterized by architecture (`tabpfn_patched.ARCHES`, `--config
real25`), since `tabpfn` ≥ 8.1 ships `tabpfn_v2_5.py` as a standalone module
with the same symbol names the v2 patches already target. The v2 path is
unchanged: rebuilding `test/fixtures/tabpfn/` reproduces all 10 files
byte-for-byte.

**License, not accuracy, is the reason it is a separate entry.** v2 ships under
the Prior Labs License (Apache-2.0 + attribution, `commercial: true`); 2.5 ships
under `tabpfn-2.5-license-v1.1`, which forbids commercial *and production* use of
the weights and their outputs. So `tabpfn-v2-5` is `commercial: false` +
`gate_setting: accept_hf_license`. `test/sql/tabpfn25.test` asserts the two rows
differ on exactly that.

Three findings from the export:

1. **The heads are different architectures.** Classifier = 24 layers with a
   linear encoder (10,718,218 params); regressor = 18 layers with an MLP encoder
   (10,186,760). `configs.real25(task)` is task-aware for that reason, and the
   manifest carries per-task graphs *and* per-task tensor maps.
2. **A new data-dependent guard.** `TabPFNV2p5.forward` validates
   `(y > n_out - 1).any()` — untraceable, and it aborts the export with
   `GuardOnDataDependentSymNode`. `self.training` appears exactly once in the
   whole module (that guard) and the released configs set `dropout = 0.0`, so
   pinning the model in train mode is provably inert and skips it. Like the
   existing `_do_encoder_nan_check` patch, it is a pure input-validation branch.
3. **Column embeddings are package data, not weights.** 2.5 replaces v2's seeded
   `randn` with `pre_generated_column_embeddings`, a fixed (2000, 48) table
   shipped inside the `tabpfn` wheel so values match across platforms. It is not
   in the checkpoint at all, so — exactly like v2's `_pos_base` — it stays inline
   in the weight-free graph and never enters the tensor map. Our 512-feature cap
   means the column count can never reach 2000, so slicing it is exact.

Export parity **1.45e-07** (classification). `convert_weights.py --arch=v2.5`
confirms **250/250** and **192/192** tensor-map keys resolve against the real
checkpoints. As for v2 the conversion step is *required*, not optional: the
released `.ckpt` stores fused QKV projections, so only 2 of 250 map keys appear
in it directly — the engine consumes the converted safetensors named by the
tensor map's `safetensors` field.

Note the `Prior-Labs/tabpfn_2_5` repo carries `extra_gated_fields`, but its
`resolve` endpoint currently serves anonymously; if that changes, the download
fails with the actionable 401 described below.

## Gated HuggingFace repositories

Some model repositories are **access-gated**: you must accept the license on the
model page while signed in, and the download must then carry your HF token.
Downloads flow through DuckDB's `httpfs`, so the token is supplied with a
standard DuckDB secret — the extension needs no setting of its own:

```sql
INSTALL httpfs; LOAD httpfs;
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_xxx', SCOPE 'https://huggingface.co');
CALL tabfm_download('classification', model := 'tabpfn-v2-5');
```

Without it the fetch fails with an actionable error naming both steps
(`HttpAuthRemediation`, `src/tabfm_weights.cpp`): HTTP **401** means no token,
HTTP **403** means the token is fine but the license has not been accepted. The
mapping is unit-tested offline in `test/cpp/test_tabfm_weights.cpp` — reproducing
a real 401 would require network, so there is deliberately no sqllogictest for it.

Note this is orthogonal to `anofox_tabfm_accept_hf_license`, which is *our*
gate recording that you accepted a non-commercial license; the secret is
*HuggingFace's* gate on serving you the bytes. Gated models need both.

## License wall (all models)

Every shipped graph is weight-free (all checkpoint initializers externalized,
`.onnx.data` deleted); every fixture is seeded random-init. No real weight bytes
from any model are committed. Real weights are the user's own HF download behind
the manifest's license gate.
