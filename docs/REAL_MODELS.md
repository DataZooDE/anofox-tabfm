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
| **Causilo** (`causilo`) | Causilo License v1.0 (**non-commercial**; hosted/API/SaaS needs a separate license) | ✅ shipped | **yes — classify + regress** | 36.1 M clf / 37.1 M reg; **no conversion — Nums AI ship safetensors**, ungated; single-estimator parity with upstream 1e-5; CPU ~0.15 s at 150 rows, ~2.2 s at 1000x50 |
| **LimiX-2M** (`limix-2m`) | Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 + attribution; **commercial use permitted**) | ✅ shipped | **yes — classify + regress** | 2.4 M; **no conversion — the engine reads the `.ckpt` directly**; ONE 9.5 MB file serves both tasks; **single forward only**; built with StableAI LimiX; ROCm refused by name |
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

This is also what surfaced the feature-column contract. The context relation and
the test relation must expose the **same feature columns**: building the context
with `SELECT *` gives it a column the query table lacks (`tgt`), and because the
train/test macro unions the two with `UNION ALL BY NAME`, the missing column is
filled with NULL rather than rejected. Every model in the catalog scored
0.27–0.58 until the tables were matched, at which point they all jumped to
0.92–0.98 — and **nothing raised**.

That is now an error rather than a quiet wrong answer:

```
anofox_tabfm: the context relation and the `test` relation must expose the same
feature columns, but only in the context relation: tgt. A column on one side
only is filled with NULL on the other, which silently degrades the prediction
instead of failing. Drop the column, add it to the other relation, or name the
shared columns explicitly with features := [...].
```

The check runs *before* the union, because the union is what destroys the
evidence; it ignores the target (which by construction exists only on the
context side) and, when `features := [...]` is given, compares only the columns
named there — pinning the shared columns is the documented way to proceed when
the relations genuinely differ. See `test/sql/tabfm_feature_mismatch.test`.

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

## Causilo (Nums AI) — shipped (classify + regress)

A TabICL-style column-then-row model (36.1 M classifier / 37.1 M regressor parameters, native
10-class head) with an extra row-refinement stage. Code is Apache-2.0; the **weights are under
the Causilo License v1.0**: non-commercial research and evaluation only, and *commercial or
production use, and hosted/API/SaaS services whether paid or free, require a separate license*
(contact@nums.world). It is registered like the other non-commercial entries: `commercial: false`,
gated by `SET anofox_tabfm_accept_hf_license = true`, with that clause in the attribution. The
Hugging Face repo itself is ungated.

Like TabICL its graph is `(x, y)`-only, so the engine needs no model-specific code
(`tools/export_causilo`). Nums AI publish safetensors whose keys are already the model's own
namespace, so there is **no converter**: `CALL tabfm_download('classification', model := 'causilo')`.
The weights are pinned to the commit the graph was exported from (`94f2bd91…`); Hugging Face `main`
has moved past it, and a moving ref would pair a newer checkpoint with this graph.

The graph does Causilo's own preprocessing itself (train-prefix z-score, two-stage 4-sigma tail
bounds, arcsinh tail compression, and for regression the target StandardScaler and its inverse), so
the engine's `causilo_v1_raw` profile only encodes and imputes.

**What is verified, on the real weights.** Against upstream's `CausiloClassifier` /
`CausiloRegressor` with `n_estimators=1`: iris 45/45 and wine 54/54 query labels identical,
probabilities within 1e-5 at temperature 1; diabetes regression within 4e-4 on a scale of 157
(query R² 0.359, the same as upstream). That comparison ran through the built-in path end to end
(the engine downloading the weights itself), not only through the exporter.

**Fitted values.** The `is_training` rows are in-context values, taken by presenting the context
rows a second time as queries. The cheap alternative, running the last prediction layer over all
rows, is measurably wrong on the real model: fitted R² 0.50 on a clean linear problem where the
query R² is 0.999, and fitted accuracy 0.457 on a 5-class problem where query accuracy is 0.717.
The second pass gives 1.000 on both and leaves query rows unchanged. It costs `S` extra rows per call.

**What this does not claim.**
- The engine runs **one estimator**. The TabArena figure (Elo 1785) is an 8-member ensemble that
  cycles four normalisations and permutes features; `n_estimators=1` is upstream's own single-member
  path, and that is what is reproduced.
- The engine mean-imputes missing cells before the graph, so Causilo's missing-value embedding is
  unused. Tables with NULLs are answered, but not with upstream's missing-value signal.
- The engine's default `softmax_temperature` is 0.9 for every model; Causilo's own is 1, so default
  probabilities are slightly sharper than upstream's (about 2e-2 on top-class probability). Labels
  are unaffected; pass `opts := MAP {'softmax_temperature': '1.0'}` to match upstream.
- More than 10 classes (upstream handles them with error-correcting output codes) is not supported;
  the engine's class ceiling is the head width.
- ROCm: refused by name like `tabicl-v2` (positional split; `docs/ROCM_SINGLE_EVAL_POS.md`).

**Ext graphs (CUDA, and the CPU default).** `graph_ext_causilo_*` reference the cached safetensors
by offset, so ORT reads weights off disk instead of copying them in; the engine uses them on CUDA
and as the default CPU path whenever the weights' header hash matches the pinned revision. On the
real checkpoints the two paths give **bit-identical** answers (0.0 difference over 770 rows,
classification and regression), checked by forcing the injection path with
`TABFM_DISABLE_EXTERNAL_DATA=1`. **CUDA is prepared but has not been run on a CUDA device**: the
graph and hash are bundled and `tabfm_backends()` should list it, but `SERVED_BY=cuda:0` and
agreement with the CPU still need a GPU box (`tools/gpu_test`).

**MLX.** The interpreter has no per-model code; all 50 distinct ops in Causilo's four graphs are in
its 68-entry op table, so it should not be refused at session creation. That predicts servable, not
correct: numerical agreement needs a Mac (`tools/gpu_test/scenarios/mlx_all_models.sql` now has
Causilo blocks that print the serving device beside the agreement).

CPU speed (ORT, intra-op threads): about 0.15 s at 150 rows x 4 features (4 threads), about 2.2 s at
1000 x 50, against 0.9 s for upstream's PyTorch single pass on the same machine. The ONNX graph is
slower than eager PyTorch here (Transpose is 36% of the time); set `anofox_tabfm_threads` to 4-8.

## LimiX-2M (StableAI) — shipped (classify + regress)

> **Built with StableAI LimiX.** (The attribution Section 10 of the weights licence asks for; see
> "Licence" below.)

LimiX-2M is the small variant of StableAI's LimiX family: 2.38 M parameters, 12 layers, embedding 96,
axis-wise attention over samples and features, and a numeric tokenizer that encodes each value as a
decimal sign, exponent and mantissa. It is **not** LimiX-2, the 400 M-parameter model that tops the
TabArena board: this extension does not ship that one. An export spike showed LimiX-2 exports and matches
upstream, but at about 3 GB of weights and about 12 GB at 1000 rows x 50 features it is a different
proposition, and the spike's recommendation was not to ship it (that is a recommendation, not a measurement
of the model's quality). We have not measured LimiX-2M on TabArena, so no leaderboard figure is claimed for
it.

### Licence

The sources disagree, and the registry follows the newest one.

| source | what it says about the weights |
|---|---|
| `vendor/limix` README (the pinned upstream code repo) | "fully available for academic research and may be used commercially upon obtaining proper authorization" |
| the Hugging Face model card (`stable-ai/LimiX-2M`) | internally inconsistent: front matter `license: other` (`limix-2m-license-v1.0`), body "all model resources are open-sourced under the Apache 2.0 License" |
| `LICENSE.txt` in the weights repo, added 2026-09-15 | the **Stable AI Technology Co., Ltd. License v1.0**: Apache-2.0 plus one added Section 10; **commercial use is permitted** |

The model is therefore registered `commercial: true`, **gated by the licence acknowledgement**
(`SET anofox_tabfm_accept_hf_license = true`), not by a commercial restriction. Section 10 treats the
weights as part of the "Work" and says:

- if you distribute or make available the Work, a Derivative Work, or any product or service that
  incorporates it, you must (a) provide a copy of the License or a reasonably accessible link to it, and
  (b) prominently display **"Built with StableAI LimiX"** where appropriate to the medium (website, UI,
  documentation);
- if you use the weights to create, train, fine-tune, distil or otherwise improve an AI model that is then
  made available to a third party, **"LimiX" must be at the beginning of that model's name**;
- activities **solely for internal research, evaluation, benchmarking or testing**, without making the Work,
  a derivative or a resulting model available to a third party, do not trigger either requirement;
- you may not use the name or the phrase to imply endorsement, sponsorship or affiliation.

License text: https://huggingface.co/stable-ai/LimiX-2M/blob/main/LICENSE.txt.

**Open item, not settled by us:** whether shipping weight-free ONNX graphs that load the user's own
downloaded weights at run time counts as "distributing the Work" under Section 10 has **not been confirmed
with StableAI or counsel**. The attribution is shown here, in the README and in the registry's `attribution`
field in the meantime. The stance on commercial use above is our reading of `LICENSE.txt`; it is not legal
advice.

### How it runs

Like TabICL and Causilo its graph is `(x, y)`-only (the split is the length of `y`), so the engine needs no
model-specific code (`tools/export_limix`). **One** checkpoint, `LimiX-2M.ckpt` (9,558,253 bytes), serves
both tasks and is downloaded once: `CALL tabfm_download('classification', model := 'limix-2m')`. It is pinned
to the commit the graphs were exported from (`641d8b81…`); the file's LFS sha256 (`16f385d5…`) is identical in
every commit since its November 2025 release. **No converter is needed**: the engine's native checkpoint
reader takes the pickle (a `{config, state_dict}` dict) directly, unlike Orion. A one-time
`convert_limix_weights` (to `model.safetensors` beside it) additionally enables the external-data graphs
(CUDA and the CPU low-memory path); the engine's answers are byte-identical either way.

The model standardises its own features, and the export wrapper standardises the regression target in-graph
and inverts it (the model itself does not: an un-normalised target scored query R² -4.2 on diabetes against
+0.36 once standardised), so the engine's `limix_v1_raw` profile only encodes and imputes. The engine
mean-imputes missing cells **before** the graph, so LimiX's own missing-value encoding is unused.

### What is verified, on the real weights

Built-in path, CPU, against upstream's own eager single forward on the same split:

- iris **45/45** and wine **54/54** query labels identical to upstream; diabetes query R² **0.3638**
  (upstream 0.36380679), engine vs eager worst absolute difference 3.8e-2, mean 8.6e-3, on a target spread of
  about 220 (float32 reduction order).
- Three weight paths agree **byte for byte** over 832 rows (the three datasets plus 300 synthetic
  classification and 300 synthetic regression rows): the external-data graph, forced injection
  (`TABFM_DISABLE_EXTERNAL_DATA=1`), and injection straight from the checkpoint. Which path ran is
  observable: the engine stages `graph_ext_limix2m_*.onnx` beside the weights when it uses them.
- The `is_training` rows are real in-context values, taken by presenting the context rows a second time as
  queries. Decoding every row from one pass is measurably wrong on the real model: context accuracy 0.111 on a
  5-class problem against 1.000 for the second pass, and linear-regression context R² 0.50 against 0.993.
  Query rows are identical either way. It costs `S` extra rows per call.

### What it does not claim, and two behaviours to know

- **One estimator.** The engine runs a single forward. Upstream's own packaged no-retrieval default
  (`cls_default_noretrieval.json`, `reg_default_noretrieval.json`) runs **4 pipelines for classification and 8
  for regression** (quantile or power transforms, feature shuffles) and combines them, and its `*_2M_retrieval`
  variants add retrieval on top; this is none of those, and the engine rejects `n_estimators > 1`. LimiX-2M is
  also the small model, so do not expect the leaderboard figures quoted for the family.
- **It saturates when extrapolating.** On a table whose query rows lie beyond the training range (query
  f1 80..99 after training on 0..79) its predictions span 59.2..64.7 where the truth reaches 72.3
  (correlation 0.83). Upstream's own forward gives the identical numbers, so this is the model. Inside the
  training range it is accurate (correlation 0.9998, worst error 0.99 on a target spanning 3..72).
- **A feature value exactly at the training mean is handled badly.** The tokenizer encodes an *exact*
  standardised zero as its own token, and a standardised value near zero is hypersensitive in proportion to
  1/|z|. On a symmetric table the row sitting exactly on the mean was predicted 25.1 against a truth of 37.3
  while its neighbours at z = ±0.17 were off by 0.3–0.4. Discrete and symmetric columns (a binary column with
  mean 0.5, a column of equally spaced integers) can trigger it. It is upstream's behaviour, reproduced by the
  engine, and pinned by `tools/export_limix/tests/test_real_weights.py`.
- **The same sensitivity explains the one unexplained number from the spike.** At (T=33, H=9, S=21) the
  exported graph in ORT differed from PyTorch by 1.2e-3 where every other shape agrees to ~1e-5. Over 200 random
  inputs the median ORT-vs-torch difference is 3.3e-5 and exactly one input exceeds 5e-4: the only one whose
  smallest standardised value is below 1e-5. That input is chaotic in eager PyTorch alone (1-ulp noise moves
  its logits 8e-3, against 3e-5 for a normal input), the first layer where it moves is the tokenizer, and
  nudging one raw element by 0.37 takes the gap from 5.4e-3 to 1.4e-5. So it is the model's input encoding,
  not an export defect. (An earlier guess, the float16 round-trip in the target embedding, was refuted: turning
  every fp16 cast into fp32 left the sensitivity unchanged.)
- More than 10 classes is not supported (the head width is the engine's class ceiling).

### Size and speed (CPU, default threads, 200 query rows)

| rows x features | wall | peak RSS |
|---|---|---|
| 500 x 20 | 7 s | |
| 1000 x 20 | 16 s | 1.3 GB |
| 1000 x 100 | 39 s | 4.6 GB |
| 2000 x 20 | 39 s | 3.7 GB |
| 3000 x 20 | 52 s | 6.2 GB |
| 3000 x 50 | 142 s | 14.3 GB |
| 5000 x 20 | 132 s | 15.3 GB |

Memory grows roughly with rows^1.7 and linearly with features. The engine caps the model at **5,000 rows and
100 features**, which are independent limits: their corner (5000 x 100) would need on the order of 70 GB, so
check the table above against your machine. (The first cap, 10,000 x 500, was a guess and was wrong.)

### Platforms

- **CPU:** verified on Linux x86-64 (this section's numbers). macOS arm64 and Windows x64 build in CI like every
  model but nobody ran LimiX there; the fixture SQL test is `notwindows`, as for the other ORT-forward tests.
- **ROCm:** refused by name, like `tabicl-v2` and `causilo` (positional split;
  `docs/ROCM_SINGLE_EVAL_POS.md`). Verified on a real gfx1201: `SET anofox_tabfm_device = 'rocm'` fails with
  the reason and `SET anofox_tabfm_device = 'cpu'` as the fix; `auto` runs on the CPU and reports it.
- **CUDA:** the external-data graphs and their header hash are bundled, but **no CUDA device was available,
  so none of this was run on CUDA.**
- **MLX:** a static check finds all 54 distinct ops in the four shipped graphs in the interpreter's table (the
  export removes `Unique` and `Relu`, which it lacked). That predicts servable, not correct. **Not run on a
  Mac**; `tools/gpu_test/scenarios/mlx_all_models.sql` has LimiX blocks that print the serving device beside the
  agreement.

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
