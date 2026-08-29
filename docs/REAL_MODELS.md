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
| **Orion-BiX** (`orion-bix`) | MIT (commercial) | ✅ shipped | **yes — classify only** | 27 M / ~315 MB; needs the ckpt→safetensors convert (native reader rejects opcode 0x65); upstream ships no regressor |
| **TabPFN-2.5** (`tabpfn-v2-5`) | tabpfn-2.5-license-v1.1 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 10.7 M clf / 10.2 M reg; 24- and 18-layer heads; ckpt→safetensors convert required |
| **TabPFN-3** (`tabpfn-v3`) | tabpfn-3-license-v1.0 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 53.2 M; export parity 2.4e-07 clf / 2.4e-06 reg; ckpt→safetensors convert required |
| **RealTabPFN-2.5** (`tabpfn-v2-5-real`) | tabpfn-2.5-license-v1.1 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | same architecture as 2.5, real-data continued pre-training; **reuses 2.5's graph, map and ext graph — zero new embedded bytes** |
| **Orion-MSP** (`orion-msp`) | MIT (commercial) | ✅ shipped | **yes — classify only** | 27 M; frozen seeded attention mask (upstream redraws it per call); ckpt→safetensors convert required |
| **TabDPT** (`tabdpt`) | Apache-2.0 (commercial) | ✅ shipped | **yes — classify + regress** | 63.5 M; **no ckpt conversion — Layer 6 ship safetensors**; both tasks share one file; parity 3.7e-07 clf / 2.9e-06 reg |
| **TabPFN-2.6** (`tabpfn-v2-6`) | tabpfn-2.6-license-v1.0 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 10.7 M clf / 12.9 M reg; rmsnorm; export parity 8.9e-08 clf / 1.4e-06 reg; ckpt→safetensors convert required |

### TabPFN-3 — done

Released 2026-05-12 and **not** 2.5 scaled up: a distribution-embedding stack
with inducing points feeds a feature-aggregation stack, and RoPE replaces the
pre-generated column-embedding table. Released dims (`embed_dim=128`,
`nlayers=24`, 27 config fields) are transcribed into `configs.real3()` from the
checkpoint's own `config` block.

It needed **no engine change and no new export patches**. The only thing
blocking `torch.export` was the multiclass target-range guard — byte-for-byte
the same branch 2.5 has — which `_freeze_in_train_mode` already neutralizes
(`dropout = 0.0` in the released config is what makes pinning train mode
numerically inert). v3 actually needs *fewer* patches than 2.5: with RoPE there
is no runtime `randn` column-embedding table to freeze.

```bash
cd tools/export_tabpfn
uv run python convert_weights.py classification   # writes model.safetensors
uv run python make_fixture.py --arch=v3           # regenerate the CI fixture
```

Offline fixture: `test/sql/tabfm_tabpfn3.test`.

### RealTabPFN-2.5 — done

`Prior-Labs/tabpfn_2_5` ships more than the `_default` checkpoint the
`tabpfn-v2-5` entry uses. `tabpfn-v2.5-{classifier,regressor}-v2.5_real.ckpt` is
the same architecture continued-pre-trained on real tabular data, and TabArena
v0.1.4 ranks it (1600 Elo) above TabICLv2. It ships as its own catalog entry
because the two checkpoints score differently and users pick between them by
name.

It is the cheapest onboarding in the catalog so far, and deliberately so:

- **Everything graph-shaped is reused.** Both checkpoints carry an identical
  `config` block and an identical state_dict signature (154 clf / 121 reg
  tensors — same names, shapes, dtypes; only the values differ). A safetensors
  JSON header is a function of names/shapes/dtypes, so converting either
  checkpoint yields a **byte-identical header** — verified against the real
  weights. The entry therefore reuses `graph_tabpfn25_*`,
  `tensor_map_tabpfn25_*` and `graph_ext_tabpfn25_*` verbatim and embeds **no
  new bytes**. `ExpectedWeightsHeaderShaFor` returns the same two shas for both
  ids, and `test_tabfm_model_spec.cpp` pins that so a divergent future
  checkpoint fails loudly instead of mis-indexing the ext graph's baked offsets.
- **The cache path is NOT shared.** `WeightsManifest::CacheSlug` keys on the HF
  *repo*, and both checkpoints live in `Prior-Labs/tabpfn_2_5`. Identical
  `files[].path` values would make the two entries download over each other and
  silently serve the wrong weights, so the real entry uses
  `classification-real/` and `regression-real/`.

```bash
cd tools/export_tabpfn
uv run python convert_weights.py classification --arch=v2.5 --variant=real
uv run python convert_weights.py regression     --arch=v2.5 --variant=real
```

Offline fixture: `test/sql/tabfm_tabpfn25_real.test` (shares the tabpfn25
fixture — sharing is the point). Real-weight behaviour:
`test/sql/tabfm_real_models.test`.

### TabPFN-2.6 — done

Released alongside the 2.5 line and **#2 single model on TabArena v0.1.4**
(1624 Elo, behind only TabPFN-3). A 2.5-*line* architecture: same
emsize/nhead/features-per-group/thinking-row layout, and — the part that decides
the export — it still carries `pre_generated_column_embeddings`
(2000 × emsize//4), so `prepare_model_for_export` takes the same non-v2 branch
as 2.5.

It needed **no new export patches at all**; registering `"v2.6"` in
`tabpfn_patched.ARCHES` was the whole change. What actually differs is
`layernorm_type="rmsnorm"` and a deeper per-layer parameterisation — 322 clf /
324 reg mapped initializers against 2.5's 250 / 192 — which is why it gets its
own graphs rather than sharing 2.5's the way `tabpfn-v2-5-real` does.

2.6 also widens the documented regime to ≤50 000 samples / ≤2000 features
(2.5: 10 000 / 500), reflected in its `size_regime`.

```bash
cd tools/export_tabpfn
uv run export_tabpfn --task classification --config real26 --out ../../resources
uv run python make_fixture.py --arch=v2.6          # regenerate the CI fixture
uv run python convert_weights.py classification --arch=v2.6
```

Offline fixture: `test/sql/tabfm_tabpfn26.test`.

### TabDPT — done (and the deferral was wrong)

`docs/MULTI_MODEL_PLAN.md` §3 deferred TabDPT behind a hypothetical
`RetrievalOnnxBackend`. That premise does not survive contact with the code:
**retrieval is not part of the model.** `TabDPTEstimator` defaults to
`context_reduction="subsample"` and only reaches for FAISS when the caller asks;
either way the reduction lives in the sklearn wrapper and merely chooses *which
context rows to hand over*. The model itself takes the whole context and derives
the split from the label length:

```
TabDPTModel.forward(x_src[B, T, F], y_src[B, n_ctx], num_features)
eval_pos = y_src.shape[0]
```

That is exactly the engine's `single_eval_pos` family. TabDPT therefore needed
**no engine change** — only an exporter (`tools/export_tabdpt`).

It is worth having beyond the rank (1459 Elo): with Mitra it is one of only two
built-ins that are both permissively licensed **and** capable of both tasks.

Two firsts for the catalog:

- **No `convert_weights.py`.** Layer 6 publish the checkpoint as safetensors
  (`Layer6/TabDPT :: tabdpt1_2.safetensors`, Apache-2.0, ungated) whose keys are
  already the model's state_dict namespace, so the downloaded file is injected
  as-is against the committed tensor map. All 647 initializers map exactly.
- **Both tasks share one weights file.** TabDPT has a single head whose output is
  class logits followed by regression bins, so the two graphs map the same
  tensors and both tasks declare the same `files[].path` — one 254 MB download
  serves both, and `ExpectedWeightsHeaderShaFor` returns the same sha for both
  because it is literally the same file. (Contrast `tabpfn-v2-5` vs `-real`,
  where a shared path would have been a *bug*: same repo, different checkpoints.)

The exporter absorbs the estimator's share of the work so the graph is
self-contained: feature padding to the model's fixed `num_features`, the
train-prefix target standardisation and its inverse, and the bar-distribution
point estimate. Hence `tabdpt_v1_raw` (raw-in / raw-out), like TabPFN's.

Two upstream constructs needed patching, both found by the export failing:

1. **The attention scale.** Upstream passes a context-length-dependent
   temperature to SDPA as `scale=`, which must be a concrete Python float;
   `eval_pos` is symbolic for us, so `torch.export` tried to guard on an unbacked
   symbol. Since SDPA computes `softmax(scale · qkᵀ)v`, folding beta into `q` and
   leaving the default scale is the identical function and fully traceable.
2. **`torch.as_tensor(eval_pos)`** in `get_scale_param` silently *baked the
   context length into a constant*. The export still succeeded — and produced a
   graph whose `y` input was pinned to the tracing example's length, which ORT
   rejects on the first real call with a different context size.
   `torch.ones(eval_pos).sum()` builds the same value without specializing.

`max_features` is a hard 128: the wrapper pads `x` up to the model's fixed width,
so a wider table would make that pad negative.

```bash
cd tools/export_tabdpt
uv run export_tabdpt --task classification --config real \
    --weights ~/.cache/anofox-tabfm/Layer6__TabDPT@main/model.safetensors \
    --out ../../resources
uv run make_tabdpt_fixture ../../test/fixtures/tabdpt
```

Offline fixture: `test/sql/tabfm_tabdpt.test`.

### Orion-MSP — done, with one deliberate deviation

The sibling of `orion-bix` (same vendor, same MIT licence, same
classification-only shape — upstream ships `sklearn/classifier.py` and no
regressor). Not on the TabArena board, so unlike the other three additions its
accuracy claim is vendor-reported; it earns its place by being commercially
clean rather than by rank.

Two things about the released `OrionMSP-classifier-v1.5` checkpoint are narrower
than the paper, and `configs.assert_shipped_path` fails loudly on either
changing:

- `row_scales = (1,)` — the "multi-scale" row interaction ships with a **single**
  scale; the 1/4/16 hierarchy is not what the published weights use.
- `perc_num_latents = 0` — the Perceiver memory is disabled.

**The deviation.** `row_num_random = 2` IS live, and upstream builds those
BigBird links with a Python loop over a tensor plus a `torch.randperm` on *every
forward*:

```python
for i in rng_idx:
    choices = torch.randperm(L - num_special)[:num_random] + num_special
    mask[i, choices] = 0.0
```

That is untraceable once `L` is symbolic, and it means **upstream inference is
non-deterministic** — the same table scored twice gets two different masks. The
exporter freezes one seeded draw into a `[MAX_L, MAX_L]` score table and takes
the top-`num_random` per row, slicing by the runtime `L` — the same treatment
`tools/export_tabpfn` gives TabPFN's runtime `randn` column-embedding table.

The result is deterministic (better than upstream) and structurally faithful:
each non-special query keeps exactly `num_random` extra links. It costs ~1 MB of
inline constant in each graph (`MAX_L = 512`, f32), which is why Orion-MSP's
graphs are larger than its parameter count suggests; upstream leaves a
table-free alternative in a comment (`(i + 1 + arange(num_random)) % ...`) that
would trade that megabyte for a strided rather than random link pattern. But it reproduces
*one particular* upstream draw rather than any specific one, so export parity is
measured against upstream **with the same patch installed** — comparing against
unpatched upstream would be comparing two different random masks and would mean
nothing. `test/sql/tabfm_orion_msp.test` asserts the resulting stability by
scoring the same table twice and requiring identical labels.

**An upstream bug found on the way.** `_build_block_sparse_mask` also computes a
sliding window that never takes effect:

```python
local = (dist <= window).to(mask.dtype) * 0.0          # all zeros
mask  = torch.where(mask.isfinite(), mask, local + float("-inf"))   # no-op
```

`x * 0.0` is zero regardless of `dist`, so `local + -inf` is `-inf` everywhere
and the `where` changes nothing. The effective upstream mask is specials +
random links + diagonal. The frozen mask **reproduces that**, deliberately:
honouring the window would "fix" the architecture out from under weights that
were trained without it (measured at L=24 / num_special=8 / window=3 it opens 14
keys per query where upstream opens 11).

```bash
cd tools/export_orion_msp
uv run export_orion_msp --config real --out ../../resources
uv run python convert_weights.py          # verifies the map covers the ckpt 1:1
uv run make_orion_msp_fixture ../../test/fixtures/orion_msp
```

Offline fixture: `test/sql/tabfm_orion_msp.test`.

### Testing against real weights

Every `test/sql/tabfm_<model>.test` runs the committed random-init fixture:
that proves the graph loads and the engine drives it, but a random-init model
predicts noise whether or not the tensor mapping is correct, so it cannot tell a
right wiring from a scrambled one.

`test/sql/tabfm_real_models.test` is the complement — it runs the **actual
published checkpoints** and asserts they are good at a learnable task
(accuracy ≥ 0.90, R² ≥ 0.80). It is gated behind `require-env
TABFM_REAL_WEIGHTS` because the weights are gated, non-commercial and ~40–50 MB
each, so CI must not fetch them and the licence wall forbids committing them.

```bash
TABFM_REAL_WEIGHTS=1 ./build/debug/test/unittest test/sql/tabfm_real_models.test
```

Measured on that split (accuracy): `tabpfn-v2-5-real` 0.984, `tabpfn-v2-6`
0.984, `tabpfn-v2-5` 0.969, `tabdpt` 0.953, `mitra` 0.938, `tabicl-v2` 0.922 —
the newly onboarded models lead, matching their TabArena order. Regression R² is
0.9999 for all three new entries, which is the assertion that actually exercises
each one's point-estimate decode (bar-distribution mean plus the target
de-standardisation each graph bakes in).

One trap worth recording, because it cost a debugging round: the context table
and the test table must expose the **same feature columns**. Building the
context with `SELECT *` gives it a column the query table lacks (`tgt`), and the
prediction then degrades to roughly chance *without raising* — every model in
the catalog scored 0.27–0.58 until the tables were matched, at which point they
all jumped to 0.92–0.98.

### Capabilities per model

`tabfm_generate` and `tabfm_impute` (see `docs/GENERATE.md`) are built on the
predict engine, so their availability follows directly from the classify /
regress capabilities above — there is no separate per-model work.

| model | classify | regress | `tabfm_generate` | `tabfm_impute` |
|---|---|---|---|---|
| `tabfm-v1` | ✅ | ✅ | ✅ | ✅ all columns |
| `mitra` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2-5` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2-5-real` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2-6` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabdpt` | ✅ | ✅ | ✅ | ✅ all columns |
| `orion-msp` | ✅ | ✗ | ✅ | categorical columns only |
| `tabpfn-v3` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabicl-v2` | ✅ | ✅ | ✅ | ✅ all columns |
| `orion-bix` | ✅ | ✗ | ✅ | categorical columns only |

`tabfm_generate` needs **only** classification, because every chain-rule step is
a classification problem — continuous columns go through quantile bins. That is
why Orion-BiX, which ships no regressor, can still generate numeric columns.
`tabfm_impute` deliberately does *not* bin (it wants full precision), so filling
a numeric column requires the model's regression weights.

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
`_raw`. Export parity **2.24e-07**.

**Conversion IS required** (verified 2026-08-11 against the current upstream
file). This section previously claimed the ~315 MB `.ckpt` is read natively and
needs no conversion step; that is no longer true for `Orion-BiX-v1.1.ckpt`, which
the native reader rejects with `unsupported pickle opcode 0x65` (`APPENDS`).
Until `src/tabfm_ckpt.cpp` learns that opcode, run the one-time conversion — the
engine then prefers the `model.safetensors` it writes next to the checkpoint:

```bash
cd tools/export_orion_bix && uv run python convert_weights.py   # 277/277 tensors
```

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
