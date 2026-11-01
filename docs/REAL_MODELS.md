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
| **TabICL v2** (`tabicl-v2`) | BSD-3-Clause (commercial) | ✅ shipped | **yes — classify + regress** | ~110 MB; iris **0.962**, wine MSE **0.586**; one-time ckpt→safetensors convert; **ROCm served** (masked graph; 0/100 label diffs vs CPU) |
| **Orion-BiX** (`orion-bix`) | MIT (commercial) | ✅ shipped | **yes — classify only** | 27 M / ~315 MB; the `.ckpt` is read natively (no conversion step); upstream ships no regressor; **ROCm served** (masked graph; 0/100 label diffs vs CPU) |
| **TabPFN-2.5** (`tabpfn-v2-5`) | tabpfn-2.5-license-v1.1 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 10.7 M clf / 10.2 M reg; 24- and 18-layer heads; ckpt→safetensors convert required |
| **TabPFN-3** (`tabpfn-v3`) | tabpfn-3-license-v1.0 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 53.2 M; export parity 2.4e-07 clf / 2.4e-06 reg; ckpt→safetensors convert required |
| **RealTabPFN-2.5** (`tabpfn-v2-5-real`) | tabpfn-2.5-license-v1.1 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | same architecture as 2.5, real-data continued pre-training; **reuses 2.5's graph, map and ext graph — zero new embedded bytes** |
| **Orion-MSP** (`orion-msp`) | MIT (commercial) | ✅ shipped | **yes — classify only** | 27 M; frozen seeded attention mask (upstream redraws it per call); the `.ckpt` is read natively; **ROCm served** (masked graph; 0/100 label diffs vs CPU) |
| **TabDPT** (`tabdpt`) | Apache-2.0 (commercial) | ✅ shipped | **yes — classify + regress** | 63.5 M; **no ckpt conversion — Layer 6 ship safetensors**; both tasks share one file; parity 3.7e-07 clf / 2.9e-06 reg |
| **Causilo** (`causilo`) | Causilo License v1.0 (**non-commercial**; hosted/API/SaaS needs a separate license) | ✅ shipped | **yes — classify + regress** | 36.1 M clf / 37.1 M reg; **no conversion — Nums AI ship safetensors**, ungated; single-estimator parity with upstream 1e-5; CPU ~0.15 s at 150 rows, ~2.2 s at 1000x50; **ROCm served** (masked graph; 0/100 label diffs vs CPU, regression 3.6e-5) |
| **LimiX-2M** (`limix-2m`) | Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 + attribution; **commercial use permitted**) | ✅ shipped | **yes — classify + regress** | 2.4 M; **no conversion — the engine reads the `.ckpt` directly**; ONE 9.5 MB file serves both tasks; **single forward only**; built with StableAI LimiX; **ROCm served** (masked graph, reads the `model.safetensors` `tabfm_download` writes; 0/100 label diffs vs CPU) |
| **TabPFN-2.6** (`tabpfn-v2-6`) | tabpfn-2.6-license-v1.0 (**non-commercial**) | ✅ shipped | **yes — classify + regress** | 10.7 M clf / 12.9 M reg; rmsnorm; export parity 8.9e-08 clf / 1.4e-06 reg; reads the downloaded .ckpt directly (no convert step) |
| **Google TabFM 1.1** (`tabfm-v1-1`) | TabFM Non-Commercial License v1.0 (**non-commercial**, same text as v1) | ✅ shipped | **yes — classify + regress, CPU + ROCm** | 410 M / 0.82 GB per task (BF16), 4x fewer params than v1; no conversion; end-to-end parity with upstream 9.4e-06 clf / 4.9e-06 reg; ROCm agrees with CPU 64/64 |
| **Mitra-v2** (`mitra-v2`) | Apache-2.0 (commercial, ungated) | ✅ shipped | **yes — classify + regress** | 75.7 M clf / 76.7 M reg; **classifier reuses mitra's graphs (byte-identical header) — zero new clf bytes**; new graph for the 1,000-bin regressor; 32,768 rows / 256 features; CPU + ROCm |
| **TabDPT v1.3** (`tabdpt-v1-3`) | Apache-2.0 (commercial, ungated) | ✅ shipped | **yes — classify + regress** | 63.0 M; re-export for the per-task y-encoders; one 252 MB file, both tasks, under `tabdpt1_3/`; parity vs upstream 1.3.1 2.3e-05 clf / 2.3e-06 reg; CPU + ROCm |
| **TabPFN-3.5** (`tabpfn-v3-5`) | tabpfn-3-5-license-v1.0 (**non-commercial**, no hosted service) | ✅ shipped | **yes — classify + regress** | 219.0 M / 876 MB; ONE safetensors serves both tasks, **no conversion**; parity vs upstream 1.7e-05 clf / 3.8e-06 reg; ROCm served (0 label diffs; cold compile ~1 h per task) |
| **TabPFN-3.5-Fast** (`tabpfn-v3-5-fast`) | tabpfn-3-5-license-v1.0 (**non-commercial**, no hosted service) | ✅ shipped | **yes — classify + regress** | 83.5 M / 334 MB; 3.5 at 8 ICL layers instead of 24; parity 1.5e-04 clf (logit floor) / 6.2e-06 reg; ROCm served (0 label diffs; cold compile ~17 min per task) |
| **Nori-30M** (`nori-30m`) | Apache-2.0, code and weights (commercial, ungated) | ✅ shipped | **yes — regress only** | 29.2 M; a LimiX fork; **no conversion — the engine reads `nori.pt` natively**; 999-quantile head reduced in-graph to its mean; parity with upstream's single forward 1.0e-4 (california); **ROCm served** (masked graph, after a one-time convert; 3.8e-6 vs CPU) |
| **EXAONE Tabular** (`exaone-tabular`) | EXAONE AI Model License 1.2 - NC (**non-commercial**, gated) | ✅ shipped | **yes — classify + regress** | 20.8 M clf / 21.1 M reg; **no conversion — LG ship safetensors**; iris 45/45 and wine 54/54 labels identical to upstream's single forward; plain attention instead of upstream's pinned FlashAttention; **ROCm served** (masked graph; 0/100 label diffs vs CPU) |
| **Kumo-Tabular** (`kumo-tabular-small` / `-medium` / `-large`) | OpenMDW-1.1 (**commercial**, ungated; keep the agreement + NVIDIA notice when distributing) | ✅ shipped | **yes — classify + regress, CPU and ROCm** | 27.5 M / 61.5 M / 213.7 M (classifier); no conversion on the CPU (the `.pt` is read natively); upstream parity ≤ 1.5e-5 (probabilities); first model onboarded on the masked contract from day one |

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

### TabPFN-3.5 and TabPFN-3.5-Fast — done

Released 2026-09-09 (HF `Prior-Labs/tabpfn_3_5`; tech report arXiv 2609.17895,
which gives TabArena Elo 1866 for 3.5 and 1780 for Fast on the full 51-dataset
board — figures of 1898 / 1805 circulating elsewhere do not appear in it, and were
not verified). Prior Labs' default ensembles are 8 estimators (3.5) and 4 (Fast);
tabfm runs its own ensemble over a single forward. Two built-ins:

| id | checkpoint | params | file | ICL layers |
|---|---|---|---|---|
| `tabpfn-v3-5` | `tabpfn-v3.5-20260909.safetensors` | 219.0 M | 876,027,932 B | 24 |
| `tabpfn-v3-5-fast` | `tabpfn-v3.5-fast-20260909.safetensors` | 83.5 M | 334,181,708 B | 8 |

The ids follow `tabpfn-v2-5` / `tabpfn-v2-5-real`: the version with its dot as a
dash, and the variant as a suffix. Every existing TabPFN id is unchanged.

**Licence.** `tabpfn-3-5-license-v1.0` is `tabpfn-3-license-v1.0` word for word;
only the model name and the revision date differ (diffed). It allows testing,
evaluation and research ("Non-Commercial Purpose"), and forbids commercial or
production use of the weights, derivatives and outputs, and offering the model as
a hosted, managed, API or SaaS service (§3d). So both entries are
`commercial: false` + `gate_setting: accept_hf_license`, exactly like `tabpfn-v3`.
Nothing tabfm does goes past what it already does for v3: it ships no weight bytes
(the user downloads them from Prior Labs' repo), and the graphs are derived from
the Apache-2.0 inference code in the `tabpfn` 9.2.0 wheel. The licence preamble
leaves elements published under a separate licence unrestricted. The repo carries
`extra_gated_fields` while the API reports `gated: false`; `resolve` serves
anonymously today, and if that changes the download fails with the actionable 401
described under "Gated HuggingFace repositories".

**Not shipped:** the experimental `tabpfn-v3.5-20260909_multiclass.safetensors`
(Prior Labs: "slightly different architecture", experimental). It would be a third
graph set for a checkpoint its authors do not recommend by default. "TabPFN-3.5
Thinking" is API-only.

**What changed against v3** (from the checkpoint's own `config` block and
`tabpfn/architectures/tabpfn_v3_5.py`):

- every cell carries a third channel next to its scaled value and NaN indicator:
  its midrank ECDF against the train rows of its column, lifted to sin/cos
  features. Above 8192 train rows (with more than 8192 distinct values) upstream
  ranks against 8192 bucket edges and interpolates;
- a Fourier cell embedding (learned frequencies) replaces v3's linear embedder;
- q/k RMSNorm in every attention (after RoPE in the column aggregator);
- 8 CLS tokens (v3: 4), so the ICL width is 1024 (v3: 512), with 16 heads;
- ONE checkpoint per variant holds both heads (a 160-class many-class decoder and
  a 5000-bucket regression head, each behind its own pre-head MLP), and the task
  is a forward argument.

**Why a new exporter.** `TabPFNV3p5.forward` cannot be traced the way v3's is.
The ECDF uses `torch.searchsorted`, which has no ONNX operator. The decoder's
one-hot width comes from `y.max().item()`. Stage 0-2 chunking and the
attention-backend dispatch are Python loops keyed on runtime shapes. So
`tools/export_tabpfn35` re-drives the blocks from their own weights, the way the
v3 masked graph does. The ECDF becomes a binary search of 24 fixed steps (Gather +
compare, no integer division, nothing data-dependent). It reproduces upstream's
algorithm with both rulers, so the graph agrees with upstream past 8192 train
rows too, not only below. The CI fixture uses 6 ECDF buckets, so the
interpolating ruler runs in the offline tests. One forward serves both contracts:
train rows are masked rather than sliced, so the same code is the positional
CPU/CUDA graph and the masked MIGraphX graph.

It is a separate uv project (pinned `tabpfn==9.2.0`), not a new arch in
`tools/export_tabpfn`. Moving that project off its locked 8.1.0 would change the
v3 config surface (9.x drops `inference_row_chunk_size`) and the byte-reproducible
v2..v3 exports with it. The shared plumbing is reused through a path dependency.

**Engine change (one).** One weights file now holds more tensors than one task's
graph declares. The engine injected every file tensor, unmapped ones under the
`m.` fallback name, and ORT rejected the other head's tensors ("Failed to find
existing initializer"). TabDPT never hit this because both its graphs use the
same tensor set. A tensor map may now declare `"complete": true`. Then the engine
injects exactly the tensors the map names (`TensorMapDeclaresComplete`,
`src/tabfm_engine.cpp`). Maps without the flag keep the fallback: TabICL's
fixture map omits 23 initializers and relies on it. The 3.5 maps carry the flag;
the exporter refuses to set it unless every initializer is mapped. Covered by
`test/sql/tabfm_tabpfn35.test`, whose fixture serves both tasks from one file
(it failed with exactly that ORT error before the change).

**Cache layout.** Both variants live in one HF repo, and `CacheSlug` keys on the
repo, so each gets its own directory (`v3.5/`, `v3.5-fast/`). Within a variant
both tasks read the same file, like TabDPT. There is no conversion step: the keys
are already the model's `state_dict` names.

**Fitted values.** Re-measured on the released weights rather than assumed from
v3. Classification decodes the train rows' own embeddings (fitted accuracy 1.00 on
separable data, same as the duplicate route). Regression presents the rows again
as queries: the embedding route correlates only 0.65 (Fast) / 0.86 (3.5) with the
target, the duplicate route 0.99997 / 0.99998.

**Text and dates.** Upstream 3.5 ships with `TRANSFORM_TEXT` / `TRANSFORM_DATES`
on: it embeds string columns and expands dates inside its sklearn front end,
before the network. That is preprocessing, not part of the graph, and tabfm does
not reproduce it. tabfm does what it does for every model: VARCHAR / ENUM /
BOOLEAN columns are ordinal-encoded categoricals, DATE / TIMESTAMP columns become
the engine's numeric datetime features (`tabfm_preprocess.cpp`), and numbers pass
through raw (the graph scales them). Free-text columns therefore get no semantic
embedding, and results on text-heavy tables can be below Prior Labs' own numbers.

**Verified.**

- Re-driven forward vs upstream `TabPFNV3p5.forward`, every row, released
  weights: 3.5 ≤ 1.7e-05 classification logits / ≤ 3.8e-06 regression; Fast
  ≤ 1.5e-04 / ≤ 6.2e-06. Fast's largest delta is a logit at upstream's
  `log(clamp(p, 1e-5) + 3e-5)` floor; all argmaxes agree. Random-init fixture dims,
  both contracts (masked padded to a larger bucket in rows AND columns): ≤ 1.1e-05,
  with negative controls (wrong `d`, one train row too many) caught.
- Exported ONNX under ORT with the released weights vs PyTorch: 3.5
  3.2e-05 / 2.0e-05, Fast 5.1e-05 / 1.9e-05. At real dims with random weights
  (export gate): ≤ 3.6e-07 classification, ≤ 1.1e-05 regression.
- ECDF vs upstream: 73 cases exact to 1e-6 (both rulers, ties, constant columns,
  padded rows).
- Engine, CPU, the standard split of `tabfm_real_models.test`: 3.5 accuracy
  **0.969**, R² **0.99997**; Fast **0.953**, **0.99993** (v3 on the same split:
  0.969 / 0.99999). Fitted and query accuracy on `sep_t` 1.0, fitted regression
  correlation 0.999998.
- MIGraphX structural guards on the masked graphs (no runtime integer `Div`, no
  `Reshape` to `[]`, no K=1 `MatMul`), exported in a fresh process.

**ROCm (gfx1201, released weights, `tools/gpu_test/scenarios/tabpfn35_rocm.sql`).**
Served by `rocm:0` (read from `tabfm_models().device`, not inferred from agreement)
through the bundled masked graphs and the freshly built plugin. 100 rows × 8
features (70 context, 30 query; the 128×16 bucket), compared with the CPU on every
row:

| model | task | agreement with CPU | cold compile (one bucket) |
|---|---|---|---|
| 3.5-Fast | cls | 0/30 query, 0/70 context labels; max prob diff 2.4e-06 | 1005 s |
| 3.5-Fast | reg | max abs 1.6e-04 (targets span ~±8), corr 1.000000, 100 distinct | 1081 s |
| 3.5 | cls | 0/30 query, 0/70 context labels; max prob diff 1.1e-06 | 3724 s |
| 3.5 | reg | max abs 1.3e-04, corr 1.000000, 100 distinct | ~3060 s (inferred from the .mxr timestamp: the timed run was interrupted) |

The compiles are the longest in the catalog (v3: ~16 min). 3.5's ICL stack is
1024 wide and 24 deep, and MIGraphX compiles per (rows, features) bucket.
`tabfm_gpu_precompile` moves that off the query path. Nothing has been tried yet
to reduce it. Warm, the same call takes 0.10 s (classify) / 0.11 s (regress) on the GPU
against 0.29 s / 0.49 s on the CPU for 3.5 (one shape only, not extrapolated).
Plugin and graphs ship together: an older plugin refuses these
graphs over `n_rows`, like every masked TabPFN graph.


```bash
cd tools/export_tabpfn35
uv run pytest                                   # ECDF, forward parity, guards, configs
uv run python make_fixture.py                   # test/fixtures/tabpfn35
for c in real35 real35fast; do for t in classification regression; do
  uv run export_tabpfn35 --task $t --config $c --out ../../resources
  uv run export_tabpfn35 --task $t --config $c --contract mask --out /tmp/mask35
done; done
uv run python parity_real.py ~/.cache/anofox-tabfm/Prior-Labs__tabpfn_3_5@main/v3.5/model.safetensors --onnx ../../resources
# then tools/make_external_graph.py (ext) and tools/make_migraphx_graph.py --dynamic (masked)
# against the downloaded safetensors; header shas 8a237b22... (3.5), 733da85d... (Fast)
```

Offline fixture: `test/sql/tabfm_tabpfn35.test`. ROCm scenario:
`tools/gpu_test/scenarios/tabpfn35_rocm.sql`.

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
uv run python convert_weights.py classification --arch=v2.6   # optional since v2026.11.01
```

Both tasks now run from the downloaded `.ckpt` with no conversion: the registry pins Hugging Face revision
`24148873` (the regressor's byte count was first recorded as 51955328, which no revision ever served, so
every fresh download was refused), and the regression tensor map declares `"ckpt_aliases":
{"regression_borders": "criterion.borders"}`, the one name the checkpoint stores differently.
`tabfm_download` writes the GPU `model.safetensors` too, byte-identical to `convert_weights.py`'s.
`test/sql/tabfm_fresh_download.test` covers both tasks from an empty cache.

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

### TabDPT v1.3 — done (`tabdpt-v1-3`, beside `tabdpt`)

Layer 6 updated `Layer6/TabDPT` on 2026-09-08 with `tabdpt1_3.safetensors` (252 MB, Apache-2.0,
ungated; PyPI `tabdpt` 1.3.x). TabArena: **1545** Elo for the v1.3 default against **1457** for
v1.2. It ships as a **new entry**; `tabdpt` keeps v1.2 unchanged.

**Not a drop-in for the v1.2 graphs.** The checkpoint's `cfg` carries v1.2's exact dims
(emsize 512, 32 layers, 16 classes, 2048 bins, ...), so a dims check passes — but the tensor
signature does not: 647 → 679 tensors. v1.2's per-layer target MLP `y_encoders.{l}`
(Linear(1,256) → GELU → Linear(256,128)) is replaced by a per-**task** pair:

| | v1.2 | v1.3 |
|---|---|---|
| classification | `y_encoders.{l}` MLP over the float label | `cls_y_encoders.{l}` = `Embedding(16, 128)` over `y.long().clamp(0, 15)` |
| regression | the same MLP | `reg_y_encoders.{l}` = Linear(1,128) → GELU → Linear(128,128) → LayerNorm |

`forward(x, y, num_features)` became `forward(x, y, is_cls)`. Everything else — encoder, the 32
layers, the attention-temperature buffers, the head (16 logits + 2048 regression bins) — is
unchanged. The "full probabilistic regression outputs" of the v1.3 README are an estimator API
(`predict(output_type="full")` returns the same logits plus bin borders); the network's regression
head is the one v1.2 had, and the engine returns its point estimate (the bar-distribution mean),
as for v1.2.

**How it was exported.** `tools/export_tabdpt` keeps its `tabdpt==1.2.0` pin (the pin is what
re-exports the shipped v1.2 graphs, and 1.3 cannot load `tabdpt1_2.safetensors`).
`tabdpt_patches.upgrade_to_v13` performs the y-encoder swap on the pinned model; every existing
patch (traceable attention scale, head over every data row) and the masked ROCm conversion apply
unchanged. `configs.assert_state_dict_matches_checkpoint` compares names and shapes against the
file's header — the check a dims comparison cannot make. The v1.2 artifacts regenerate
byte-identically with the changed exporter.

**Both tasks, one file, two subsets.** Each task's graph reads only its own encoder (551
classification / 647 regression of the 679 tensors). The engine injects the whole file, and ORT
refuses an initializer its graph does not declare, so the tensor maps list the other task's
tensors under `unused_by_graph` and the engine skips exactly those (opt-in; a map without the list
behaves as before; every other catalog map was checked to cover its checkpoint fully). This is
the one engine change, covered by `test/sql/tabfm_tabdpt13.test`.

**Cache path.** Same repo ⇒ same cache slug (`Layer6__TabDPT@main`) as v1.2, so the file lives
at `tabdpt1_3/model.safetensors` — the `tabpfn-v2-5-real` rule. The bundled GPU graphs are
staged beside it (stem `tabdpt13`), so v1.2's staged graphs are untouched too.

**Verified on the real weights.**

| check | result |
|---|---|
| strict `load_state_dict` of `tabdpt1_3.safetensors` into the patched model | 679/679 |
| exported ONNX (ORT) vs upstream `tabdpt` 1.3.1 `TabDPTModel`, same weights and inputs, T=96/300 | classification logits **2.3e-05** max abs (argmax 100 %), regression point **2.3e-06** |
| export parity (graph vs PyTorch wrapper, random weights) | 3.7e-07 clf / 2.9e-06 reg |
| masked (ROCm) vs positional, incl. the three negative controls | rel 8.9e-07 clf / 1.5e-07 reg |
| `tabfm_real_models.test` split, CPU | accuracy **0.969** (v1.2: 0.938), R² **0.99990** (v1.2: 0.99995) |
| CPU latency, that split (168 context / 72 query rows) | ~0.9 s per task (v1.2: ~1.0 s) |

ROCm: served on `rocm:0`; numbers in the ROCm table of the Mitra-v2 section.

```bash
cd tools/export_tabdpt
W=~/.cache/anofox-tabfm/Layer6__TabDPT@main/tabdpt1_3/model.safetensors
for t in classification regression; do
  uv run export_tabdpt --task $t --config real13 --weights $W --out ../../resources
  uv run export_tabdpt --task $t --config real13 --contract mask --weights $W --out /tmp/mask13
done
uv run make_tabdpt_fixture --config fixture13 ../../test/fixtures/tabdpt13
# then tools/make_external_graph.py and tools/make_migraphx_graph.py --dynamic exactly as for v1.2
# (docs/ROCM_TABDPT_SPIKE.md), with the tabdpt13 names and --inline-ok bin_centres for regression
```

Offline fixture: `test/sql/tabfm_tabdpt13.test` (`test/fixtures/tabdpt13`; its seeding keeps the
attention-temperature buffers `n0`/`max_len_f`/`kappa` at their constructed values — random ones
can make `log(min(n, max_len_f) / n0)` NaN, which the v1.2 fixture escapes only by luck).

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

**ROCm: served on `rocm:0`, verified on a real gfx1201 with the real weights.** The CPU graph's split is
positional, so a second, MASKED graph is bundled (`graph_migraphx_orion_msp_classification.onnx`,
`tools/export_orion_msp --contract mask`, `orion_mask.py`): `x, y_full, train_size, d`. On top of Orion-BiX's
masks (below) the row stage needs two of its own: the PMA pooling's keys exclude padded columns (upstream's own
`valid_counts` mask), and the frozen random links are the top-2 of the score table over the REAL columns only --
the CPU graph takes the top-2 of `table[:L, :L]` with `L` the real length, and an unrestricted top-2 over the
padded length would hand a real query a different link whenever a padded column scores higher. The gate compares
padded + masked with the positional forward on the unpadded table: worst 7.2e-4 on logits up to 11 (fp32
reassociation; in float64 the two agree to 1.4e-12), and each of six negative controls (one mask broken each,
random links included) is caught by 0.26 or more. Through the engine, 100 rows x 8 features (the 128 x 16 bucket),
fp32, against the CPU: **0 / 100 label disagreements** (context and query). First call on a bucket compiles for
about **4.3 minutes** (259 s); warm calls take **0.023 s on ROCm against 0.31 s on the CPU**
(`tools/gpu_test/scenarios/orion_rocm.sql`).

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
0.984, `tabpfn-v2-5` 0.969, `tabdpt-v1-3` 0.969, `mitra-v2` 0.953, `tabdpt` 0.953 (0.938 on the
2026-10-31 build), `mitra` 0.938, `tabicl-v2` 0.922 —
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
| `tabfm-v1-1` | ✅ | ✅ | ✅ | ✅ all columns |
| `mitra` | ✅ | ✅ | ✅ | ✅ all columns |
| `mitra-v2` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2-5` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2-5-real` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v2-6` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabdpt` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabdpt-v1-3` | ✅ | ✅ | ✅ | ✅ all columns |
| `orion-msp` | ✅ | ✗ | ✅ | categorical columns only |
| `tabpfn-v3` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v3-5` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabpfn-v3-5-fast` | ✅ | ✅ | ✅ | ✅ all columns |
| `tabicl-v2` | ✅ | ✅ | ✅ | ✅ all columns |
| `orion-bix` | ✅ | ✗ | ✅ | categorical columns only |
| `nori-30m` | ✗ | ✅ | ✗ (needs classification) | numeric columns only |
| `exaone-tabular` | ✅ | ✅ | ✅ | ✅ all columns |
| `kumo-tabular-*` | ✅ | ✅ | ✅ | ✅ all columns |

`tabfm_generate` needs **only** classification, because every chain-rule step is
a classification problem — continuous columns go through quantile bins. That is
why Orion-BiX, which ships no regressor, can still generate numeric columns.
`tabfm_impute` deliberately does *not* bin (it wants full precision), so filling
a numeric column requires the model's regression weights.

`nori-30m` is the mirror image of Orion: regression only. `tabfm_generate` refuses it (every chain-rule
step is a classification), and `tabfm_impute` fills numeric columns but refuses a categorical one, both
naming the missing task (`test/sql/tabfm_nori.test`). A refused `tabfm_classify` names the models that do
classify ("Models that support it: ...").

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

## Mitra-v2 — done (`mitra-v2`, beside `mitra`)

AWS AutoGluon's second-generation Mitra (arXiv 2609.04540; `autogluon/mitra-classifier-2`
2026-09-07, `autogluon/mitra-regressor-2` 2026-09-08; Apache-2.0, ungated). A **new entry**;
`mitra` keeps v1 unchanged.

**The classifier is a drop-in; the regressor is not.** `config.json` of classifier-2 is
byte-identical to classifier-1's, and so is the safetensors JSON header (sha `cb1a261b…`, 392
tensors — same names, shapes, dtypes and offsets; only the values differ). The entry therefore
reuses mitra's classification graph, tensor map, ext graph and MIGraphX graph verbatim and embeds
no new classification bytes; `ExpectedWeightsHeaderShaFor("mitra-v2", "classification")` is
mitra's sha, pinned in `test_tabfm_model_spec.cpp`. The regressor-2 file is 306,776,952 bytes
(v1: 302,683,140), `dim_output` 1 → **1000**: regression is cast as classification over 1,000
target bins, `final_layer` is `[1000, 512]` and every support target is embedded by its **bin**
(`y_embedding` becomes an `Embedding [1000, 512]`, no bias; 393 → 392 tensors). That needs its own
graph, `graph_mitra2_regression` (`tools/export_mitra --config real2`), which does upstream's
target pipeline in-graph:

1. min-max scale over the context rows (AutoGluon `Preprocessor.normalize_y`);
2. `bucketize(y_norm, linspace(-0.5, 1.5, 1001)) - 1`, clamped to [0, 999]
   (`trainer_finetune`; written as a compare-and-count so it exports and MIGraphX parses it);
3. the network predicts a distribution over the bins;
4. the point prediction is its **mean** over the bin centres (the card; installed upstream by
   `mitra_finetune.patches._mean_decode_source` — stock AutoGluon's argmax decode is for v1's
   head), then the min-max scaling is undone.

The graph is raw-in / raw-out, so the entry's profile is `mitra_v2_raw`. Features are passed raw
too, which is what upstream feeds: Mitra's quantile embedding is rank-based, so the engine's
monotone feature scaling is irrelevant either way. A consequence worth knowing: because the target
scaling lives in the graph, v2 regression does **not** depend on the engine's target scaling —
the v1 regressor's shrinkage on a z-scored target (R² 0.75, slope 1.78 on a noise-free linear
target, measured here; being fixed separately for `mitra`) does not occur (R² 0.9998, slope 0.998).
Predictions always lie in `[y_min - range/2, y_max + range/2]` and are affine-equivariant in the
target, for any weights; `test/sql/tabfm_mitra_v2.test` asserts both on the random-init fixture.

**Size regime.** v2 was pre-trained on 1–50 features and 160–5,120 support rows ("10x longer
context, 3x features" than v1's 1–16), and its documented operating range is up to **32,768**
support rows at prediction time and a **256**-feature budget (Technical Report §3.1, §3.5;
`mitra-finetune` README). The entry declares 32,768 / 256 (v1: 10,000 / 100). The v1 graph the
classifier reuses was traced with H ≤ 100, but its ONNX axis is symbolic: a 160-feature table runs
(real-weights test; accuracy 0.938, v1 0.922 on the same table, ~60 s CPU at 168 context rows).
tabfm has no feature reduction, so wider than 256 is beyond the documented range (the global
`anofox_tabfm_max_features` still applies).

**In-context only — read the card's numbers with that in mind.** The card's benchmark results
(TabArena Elo 1774.6) use the `mitra-finetune` recipe: 50-step fine-tuning with 8-fold bagging,
class-balanced support sampling and "heldout in support". tabfm runs one in-context forward pass,
so those numbers do not transfer. Measured in-context here instead.

**Verified on the real weights.**

| check | result |
|---|---|
| classifier-2 header == classifier-1 header | byte-identical (sha `cb1a261b…`) |
| shipped mitra classification graph + v2 weights (ORT) vs AutoGluon's own `Tab2D` (CPU branch), S=60/250 | logits **1.6e-04** max abs at max \|logit\| ≈ 25 (argmax 100 %; the same harness gives 1.6e-05 at \|logit\| ≈ 7 for v1) |
| `graph_mitra2_regression` + v2 weights vs `Tab2D` + upstream min-max/bucketize/mean decode | **1.6e-03** max abs on targets spanning ~100–130 (≈1e-05 of the range) |
| export parity (graph vs PyTorch wrapper, random weights) | 1.5e-05 (raw target units) |
| `tabfm_real_models.test` split, CPU | accuracy **0.953** (v1: 0.938); R² **0.99983** (v1: 0.693 with the current z-scored target) |
| noise-free linear target, original scale and ×1000 + 7000 | R² > 0.999, slope 0.998 |
| CPU latency, that split | ~2.3 s per task, as v1 |

```bash
cd tools/export_mitra
W=~/.cache/anofox-tabfm/autogluon__mitra-regressor-2@main/model.safetensors
uv run export_mitra --task regression --config real2 --weights $W --out ../../resources
uv run export_mitra --task regression --config fixture2 --out ../../test/fixtures/mitra2 \
    --emit-weights ../../test/fixtures/mitra2/model_regression.safetensors \
    --emit-golden ../../test/fixtures/mitra2/golden_regression.json
cd ../.. && python tools/make_external_graph.py regression --weights $W \
    --graph resources/graph_mitra2_regression.onnx --tensor-map resources/tensor_map_mitra2_regression.json \
    --out resources/graph_ext_mitra2_regression.onnx
python tools/make_migraphx_graph.py regression --weights $W --dynamic --inline-ok bin_edges,bin_centres \
    --graph resources/graph_mitra2_regression.onnx --tensor-map resources/tensor_map_mitra2_regression.json \
    --out resources/graph_migraphx_mitra2_regression.onnx
```

Offline fixture: `test/sql/tabfm_mitra_v2.test` (classification shares `test/fixtures/mitra`;
regression uses `test/fixtures/mitra2`). MLX: not run (Linux box). The MLX plugin's hand-ported
fast path keys on the id `mitra`, so `mitra-v2` goes through the general graph interpreter there.

### ROCm (Radeon RX 9070 XT, gfx1201) — mitra-v2 and tabdpt-v1-3

Both serve on `rocm:0` with the real weights (`SELECT device FROM tabfm_models()` reads `rocm:0`
for all four model/task pairs, `SET anofox_tabfm_device = 'rocm'`, release MIGraphX plugin), on
the `tabfm_real_models.test` split (168 context / 72 query rows → bucket T512 × H16):

| model / task | GPU vs CPU | GPU quality | cold MIGraphX compile |
|---|---|---|---|
| `mitra-v2` classification | 64/64 labels agree | accuracy 0.953 (= CPU) | 580 s |
| `mitra-v2` regression | max abs 3.8e-05, corr 1 − 6e-12 | R² 0.99983 (= CPU) | 619 s |
| `tabdpt-v1-3` classification | 64/64 labels agree | accuracy 0.969 (= CPU) | 551 s |
| `tabdpt-v1-3` regression | max abs 8.8e-06, corr 1 − 2e-13 | R² 0.99990 (= CPU) | 819 s |

Compile times are cold, per shape bucket, on a shared machine; the `.mxr` is cached afterwards.
`mitra-v2` classification compiles mitra's graph against v2's weights: the program cache keys on
a weights fingerprint as well as the graph (`tabfm_mxr_cache_key.hpp`), so it gets its own
`…_w52cd48ae_…` entry and never reuses v1's program.
`tools/gpu_test/scenarios/model_upgrades_rocm.sql` reproduces this table and checks that
mitra and mitra-v2 answer differently on the GPU.

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
- ROCm: served on `rocm:0` through a second, MASKED graph per task (`tools/export_causilo --contract mask`;
  `docs/ROCM_CAUSILO_PLAN.md`), not the positional CPU graph.

**ROCm (Radeon RX 9070 XT, gfx1201, real weights).** `graph_migraphx_causilo_*` take `x, y` (full length,
-100 off the context), `train_size`, `n_rows`, `d`: context statistics and every attention over rows are masked,
whole padded feature groups are masked out of the row stages, and the `QueryScale` key counts come from the values.
They reuse the shipped tensor map byte for byte, so the existing header sha applies. On 100 rows x 8 features
(the 128 x 16 bucket, 70 context rows; `test/sql/tabfm_rocm_causilo.test`,
`tools/gpu_test/scenarios/causilo_rocm.sql`), with `tabfm_models()` reading `rocm:0`:

| | classification | regression |
|---|---|---|
| agreement with the CPU, 100 rows | 0 label differences | max abs diff 3.6e-5, corr 0.9999999999998 |
| first compile per bucket (`migraphx-driver compile --gpu`) | ~13 min (791 s) | ~16 min (937 s) |
| warm call, ROCm vs CPU | 0.049 s vs 0.22 s | 0.047 s vs 0.18 s |

The CPU warm call uses ~3.3 s of CPU time over its threads; the ROCm one ~0.05 s.

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

Memory grows roughly with rows^1.7 and linearly with features. The registry declares **5,000 rows and
100 features** for the model as measured, recommended limits; they are not enforced (only the global
`anofox_tabfm_max_rows` / `anofox_tabfm_max_features` are), so set `SET anofox_tabfm_max_rows = 5000` to hold
to the row limit. They are independent limits: their corner (5000 x 100) would need on the order of 70 GB, so
check the table above against your machine. (The first declared limit, 10,000 x 500, was a guess and was wrong.)

### Platforms

- **CPU:** verified on Linux x86-64 (this section's numbers). macOS arm64 and Windows x64 build in CI like every
  model but nobody ran LimiX there; the fixture SQL test is `notwindows`, as for the other ORT-forward tests.
- **ROCm:** served through a second, MASKED graph per task (`tools/export_limix --contract mask`;
  `docs/ROCM_SINGLE_EVAL_POS.md`), which reads the `model.safetensors` `tabfm_download` writes beside
  `model.ckpt` (header sha `0c6a8684...`, the same file the ext graphs use). Verified on a gfx1201 (RX 9070 XT,
  fp32, 100 x 8, 70 context rows; `test/sql/tabfm_rocm_real_models.test`): served by `rocm:0`, **0/100** label
  differences from the CPU, regression max |diff| 3.5e-4 on a target scale of 7.9 (corr 0.9999999996); cold
  compile 162 s (classification) / 155 s (regression) at T128 H16; warm 0.017 s against 0.11-0.12 s on the CPU.
- **CUDA:** the external-data graphs and their header hash are bundled, but **no CUDA device was available,
  so none of this was run on CUDA.**
- **MLX:** a static check finds all 54 distinct ops in the four shipped graphs in the interpreter's table (the
  export removes `Unique` and `Relu`, which it lacked). That predicts servable, not correct. **Not run on a
  Mac**; `tools/gpu_test/scenarios/mlx_all_models.sql` has LimiX blocks that print the serving device beside the
  agreement.

## Nori-30M (Synthefy) — shipped (regression only)

Nori-30M is the 29.2 M-parameter Nori from Synthefy: an in-context regression model trained only on
synthetic data. Upstream's NOTICE says the code "originated as a fork of LimiX", and it shows: the same
`smf` layer (sample attention, MLP, feature attention), the same decimal (sign, exponent, mantissa) RBF
tokenizer, features in pairs. Nori grows it to 28 pre-norm layers with RMSNorm, SwiGLU, DeepNorm residual
scaling (0.189), a query-adaptive softmax temperature in the sample attention (QASSMax), a target-aware
embedding of the context labels, and a 999-quantile pinball regression head.

**Licence.** Apache-2.0 for code and weights: the model card front matter and body agree, and the upstream
repo carries LICENSE + NOTICE. The Hugging Face repo is ungated. It is therefore registered like `tabdpt`:
`commercial: true`, no licence gate. Verified on Hugging Face, `Synthefy/Nori-30M` @ `63c9f7fa`.

**Regression only.** The checkpoint config still carries a 10-class head, but upstream's `build_model`
refuses every `task_type` except `"reg"`, and the model card calls it a regression model. `capabilities` is
`["regress"]`: `tabfm_classify` and `tabfm_download('classification', ...)` are refused before any download,
and the error names the models that do classify.

**How it runs.** The graph is `(x, y)`-only, so the engine needs no model-specific code
(`tools/export_nori`, which pins `synthefy-nori==0.22.0` from PyPI and patches it at run time). There is no
converter: `nori.pt` is a `torch.save` archive `{model_config, ema_state_dict}`, and the engine's native
checkpoint reader now prefers `ema_state_dict` the way upstream's `load_model` does (the one engine change,
covered by `test/cpp/test_tabfm_ckpt.cpp` with a 2.5 KB random checkpoint in that layout). The committed
fixture is a checkpoint in the same layout, so the offline test exercises the native reader on it. Four
upstream constructs needed patching, each proved identical to upstream by `tools/export_nori/tests`:

| upstream | rewrite | why |
|---|---|---|
| the feature-pair pad `if (-H) % 2 > 0: cat zeros` | always pad, then slice | a Python branch on the symbolic width freezes at the trace's parity |
| the label pad `nan * zeros(B, T - S, 1)` | the tail of a `T`-long NaN tensor | torch.export concretises the pad length |
| QASSMax's `math.log(float(key_len))` | `log(ones(S).sum())` | a Python float of the symbolic context length specialises the graph |
| `int(eval_pos)` and module-attribute writes in two encoders | removed | same specialisation; the arithmetic is unchanged |

One more was found by the export rather than by reading: the label embedding's `.squeeze(2)` is a torch
no-op on the 3-D regression encoder output, but exports as an unconditional `Squeeze` that ORT rejects.

**The point estimate is the mean.** The head predicts 999 quantiles on the grid `tau_k = (k + 1) / 1000`.
Upstream's default (`NoriRegressor(quantile_collapse="mean")`) sorts them and integrates the
piecewise-linear quantile function; on that uniform grid the integral is `(sum(q) + min(q)/2 + max(q)/2) /
1000`, so the graph needs no sort (`nori_patches.quantile_mean`, equal to upstream's to 1e-12 including on
crossed quantiles). The mean rather than the median because it is upstream's default and what its benchmark
R² comes from, and because the engine scores regression by squared error, for which the mean is optimal.
The wrapper z-scores the raw target with the population std (as `NoriRegressor` does) and inverts it.

**What is verified, on the real weights.** Built-in path, CPU, against upstream's own eager forward on the
same split (`model.forward` + `quantile_dist_mean_simple`):

- california_housing (1000-row sample, 700 / 300): worst absolute difference **1.0e-4**, query R² 0.79172
  (engine) vs 0.79172 (upstream).
- diabetes (309 / 133): worst difference 9.4e-2 on a target std of 71, query R² 0.3804 vs 0.3804. The graph
  run standalone in ORT gives the identical 9.4e-2, so it is not the engine; and 1e-7 relative noise on the
  inputs moves upstream's own eager output by 7.7e-2 there. It is float32 rounding amplified by the decimal
  tokenizer, as for LimiX-2M (`tools/export_nori/tests/test_real_weights.py` pins both numbers).
- `test/sql/tabfm_real_models.test`: shared split query R² **0.99982**; reg_t context corr 0.99997 and query
  corr 0.9955 with predictions reaching 74.2 against a truth of 72.3 (it extrapolates, unlike LimiX-2M);
  reg_in query corr 0.99996, worst error 0.44.

**What it does not claim.**
- **One forward.** TabArena (leaderboard space @ `68988c6e`, 2026-10-31: regression Elo **1791**; the EXAONE
  model card quotes 1753 from an earlier snapshot) evaluates upstream's default `NoriRegressor`: an
  8-member ensemble of preprocessing pipelines (quantile and adaptive feature transforms, polynomial
  interaction features, fingerprint features, feature shuffles, SVD for wide tables). The engine runs one
  forward on the raw table. On diabetes the single forward scored R² 0.348 against the ensemble's 0.328 on
  one split; that is one noisy dataset, not a claim that the ensemble is worse.
- The engine mean-imputes missing cells before the graph, so Nori's missing-value embedding is unused.
- No classification, `tabfm_generate`, or categorical `tabfm_impute`.

**Size and speed** (CPU, default threads, 200 query rows; the engine caps the model at 5,000 rows and 100
features, independent limits):

| rows x features | wall | peak RSS |
|---|---|---|
| 500 x 20 | 5.6 s | 0.85 GB |
| 1000 x 20 | 10.8 s | 1.5 GB |
| 2000 x 20 | 28 s | 3.3 GB |
| 3000 x 20 | 53 s | 6.1 GB |
| 5000 x 20 | 127 s | 15.4 GB |
| 1000 x 100 | 40 s | 5.0 GB |

**Platforms.**
- **CPU:** verified on Linux x86-64 (the numbers above).
- **ROCm: served on `rocm:0`, verified on a real gfx1201 with the real weights.** MIGraphX reads the weights
  itself, as external data in a `model.safetensors` beside `nori.pt`. `tabfm_download` writes that file from the
  checkpoint (status `converted`), byte for byte what `tools/export_nori`'s `convert_nori_weights` writes, and
  only when its header matches the one the bundled graph pins; without it `tabfm_backends()` reports the bundled
  ROCm graph as not matching the weights and the CPU serves. The CPU graph's split is positional, so a second, MASKED graph is bundled
  (`graph_migraphx_nori30m_regression.onnx`, `tools/export_nori --contract mask`, `nori_mask.py`): `x, y_full,
  train_size, d, n_rows`, with every statistic over the context, padded feature groups masked out of the feature
  attention (`2 * g < d`, division-free), non-context keys masked in the sample attention and QASSMax reading
  `log(S)` from the value. The gate compares it, padded to a larger bucket, with the positional forward (worst
  6.9e-6 on the real weights) and only passes when each of four negative controls is caught. Getting MIGraphX
  to compile it took four rewrites, each bisected on the GPU with small extracted graphs and each the same
  function (`tools/export_nori/README.md`): the tokenizer without integer floor-division/modulo and `split`,
  GEMMs instead of einsums, LayerNorm written out, the target statistics masked before transposing, and the
  QASSMax base MLP written out (its `Linear` on a `[1, 1]` input made `simplify_reshapes` throw). Through the
  engine, 100 rows x 8 features (the 128 x 16 bucket), fp32, against the CPU: **max |diff| 3.8e-6** on a
  scale of 8, corr 1.0000000, in-context corr 0.99998. First call on a bucket compiles for about **37
  minutes** (2229 s); warm calls take **0.18 s on ROCm against 2.6 s on the CPU** (about 15x). The CPU path
  gives byte-identical answers from `nori.pt` and from the converted file.
- **CUDA / MLX:** no external-data graph is bundled and nothing was run on either.

```bash
cd tools/export_nori && uv sync
uv run export_nori --config real --out ./out        # resources/graph_nori30m_regression.onnx + map
uv run make_nori_fixture ../../test/fixtures/nori   # the committed CI fixture
TABFM_REAL_WEIGHTS=1 uv run pytest                  # real-weight parity
```

## EXAONE Tabular (LG AI Research) — shipped (classify + regress)

EXAONE Tabular v1 is LG AI Research's Cross-Axis Summary Transformer (CAST; arXiv 2608.25774): width 192, 6
heads, 12 blocks, one column per feature token, 3 item-summary tokens per row and 32 feature-summary tokens
per column, and an SSMax attention temperature (keys scaled by the log of the context length). The
classifier (20.8 M parameters, a 10-class head) and the regressor (21.1 M, a 999-quantile head) are separate
checkpoints in one repo, `LG-AI-Research/EXAONE-Tabular` @ `093ad1c2` (ungated).

### Licence

Code: BSD-3-Clause-LG AI Research License (the BSD 3-clause text plus a reservation of patents and
trademarks). Weights: the **EXAONE AI Model License Agreement 1.2 - NC**, read in full. The clauses that
decide what tabfm may do:

- 2.1(a): a licence to "access, download, install, and use the Model solely for research and educational
  purposes. This includes evaluation, testing, academic research, experimentation, learning, teaching ...".
- 3.1: "The Licensee is expressly prohibited from using the Model, Derivatives, or Output for any commercial
  purposes, including but not limited to, developing or deploying products, services, or applications that
  generate revenue, whether directly or indirectly."
- 2.1(d) and 2.2: distribution of the Model or Derivatives only "with a copy of this Agreement", and "any
  commercial application or external distribution, is strictly prohibited unless explicitly agreed upon in
  writing".
- 2.1(c): a modified Model "must include "EXAONE" at the beginning of its name".
- 3.2: no decompiling, reverse engineering, or attempting "to derive the source code, underlying ideas,
  algorithms, or structure of the Model".
- 4.3: publications must attribute the Model's name and version.

Our reading, which is not legal advice: what tabfm does is what the licence permits for research and
evaluation, so it may offer EXAONE as a **non-commercial built-in gated by `accept_hf_license`**, exactly
like TabPFN-2.5 and Causilo:

- tabfm never distributes the weights. The user downloads them from LG's own repo, after `SET
  anofox_tabfm_accept_hf_license = true`, and the gate message calls them non-commercial. The registry's
  `attribution` carries the restriction and the licence link (`commercial: false`, `redistributable:
  false`).
- The architecture in the shipped graphs comes from LG's published BSD-licensed code, not from the weights,
  so 3.2 is not engaged; the graphs are weight-free.
- The id starts with "EXAONE" (`exaone-tabular`), which satisfies 2.1(c) even if a single-forward export is
  read as a modification.
- What the user then does with the Output is governed by 3.1: commercial use needs a separate agreement
  with LG.

TabArena itself lists the entry as `Commercial: False`, `EXAONE AI Model License 1.2 - NC`.

### How it runs

The graph is `(x, y)`-only like TabICL's, so the engine needs no model-specific code. LG publish safetensors
keyed by the module namespace, so there is **no converter**: `CALL tabfm_download('classification', model :=
'exaone-tabular')` and the same for `'regression'`; the two checkpoints are cached under `classification/`
and `regression/` because they share a repo.

Upstream's forward is written for a CUDA runtime and could not be traced as it stands: every module
validates `isinstance(support_size, int)`, which a symbolic size is not; the attention packs its weights
into buffers keyed by `data_ptr()` and caches the support keys/values in module state between the two halves
of a layer; part of it is pinned to FlashAttention (fp16/bf16 only, hence "float32 is a CPU-only path"); the
SSMax scale is `math.log` of a symbolic length; and the classification label encoder uses `torch.unique`.
`tools/export_exaone` (which pins `exaonetabular` to commit `d149d0f0`) therefore **re-expresses the
inference forward** over upstream's own parameters (`exaone_patches.py`): the plain `softmax(q k^T / sqrt(d))
v` attention, the queries' multi-query cross-attention onto the first context head, a first-occurrence rank
table instead of `Unique`, and branchless label imputation. It is proved against upstream's forward:
2.9e-6 on random inputs with the real weights, and `tests/test_export.py` checks it at odd shapes, one
feature, S = 2 and a constant column. Upstream's final layer skips the support rows' update (nothing reads
it); the export computes it, which changes nothing a query sees.

The wrapper standardises the regression target as `EXAONETabularRegressor.fit` does (float32, population
variance, a variance at rounding level counts as constant) and inverts it. The **point estimate is the mean
of the 999 quantiles**: that is the released regressor's own readout (`RegressionConfig.point_estimate =
"mean"`), which LG describe as targeting "the conditional mean, which is what RMSE scores and what the
median misses on skewed targets"; for one member upstream's across-member trimming is a no-op.

### What is verified, on the real weights

Built-in path, CPU, against upstream's own eager single forward (`model.forward`, float32 CPU):

- iris **45/45** and wine **54/54** query labels identical to upstream (accuracy 0.978 and 1.000 both).
- diabetes (309 / 133): worst difference **1.7e-4** on a target std of 71, query R² 0.39104 both;
  california_housing (1000-row sample): worst difference **2.6e-5**, R² 0.80731 both.
- `test/sql/tabfm_real_models.test`: shared split accuracy **0.953** and query R² **0.99998**; sep_t context
  and query 1.000; reg_t context corr 0.999997, query corr 0.9993 (predictions 59.1..67.6 against a truth
  reaching 72.3); reg_in query corr 0.999997, worst error 0.09.

### What it does not claim

- **One forward.** The TabArena figures (leaderboard space @ `68988c6e`, 2026-10-31: overall Elo **1774**,
  classification 1760, regression 1913; the model card quotes 1755 / 1759 / 1883) come from upstream's
  8-member ensemble: four numeric transforms (identity, power, robust, rank), random sign flips and feature
  permutations, member trimming, and error-correcting output codes for more than 10 classes. The engine runs
  one forward on the raw table.
- More than 10 classes is not supported (upstream uses ECOC); the head width is the engine's class ceiling.
- The engine mean-imputes missing cells before the graph, so the model's non-finite code channel never sees
  a NaN.
- The engine's default `softmax_temperature` is 0.9; upstream's is 1, so default probabilities are slightly
  sharper. Labels are unaffected.
- `fitted_t` in the real-weights test is not asserted for this model: its label (`i % 3`) is not learnable
  from its features by EXAONE, and upstream's own eager forward also answers one class for every row
  (probabilities 0.34 / 0.33 / 0.33).

### Size and speed (CPU, default threads, 200 query rows)

| rows x features | wall | peak RSS |
|---|---|---|
| 500 x 20 | 4.3 s | 1.2 GB |
| 1000 x 20 | 9.7 s | 2.5 GB |
| 2000 x 20 | 26 s | 6.7 GB |
| 3000 x 20 | 49 s | 13.1 GB |
| 1000 x 100 | 37 s | 10.5 GB |

Memory grows roughly with rows^1.7 and linearly in features, about twice Nori's at the same shape (the 32
feature-summary tokens join every column's sample attention). The engine caps it at **3,000 rows and 100
features**, independent limits measured here, not upstream's 100,000-row support limit, which its cached,
chunked CUDA runtime serves.

### Platforms

- **CPU:** verified on Linux x86-64.
- **ROCm: served on `rocm:0`, verified on a real gfx1201 with the real weights.** The CPU graph's split is
  positional, so a second, MASKED graph per task is bundled (`graph_migraphx_exaone_tabular_<task>.onnx`,
  `tools/export_exaone --contract mask`, `exaone_mask.py`): `x, y_full, train_size, d, n_rows`, with the
  feature-encoder statistics and the label encoder over the context, padded columns masked out of the feature
  attention, the support self-attention and the queries' cross-attention masking non-context keys, and both
  SSMax lengths (`3 + d + 1`, `32 + S`) taken from the values. The gate (`mask_parity.gate`) compares it,
  padded to a larger bucket, with the positional forward on the unpadded table (worst 3.8e-6 classification,
  1.6e-5 regression on the real weights) and only passes when each of four negative controls (one mask
  broken) is caught. Through the engine, 100 rows x 8 features (the 128 x 16 bucket), fp32, against the CPU:
  **0 / 100 label disagreements** (context and query), regression **max |diff| 1.4e-5**, corr 1.0000000; the
  GPU's in-context accuracy 1.000. First call on a bucket compiles for about **19 minutes per task** (1155 s
  classification, 1132 s regression); warm calls take **0.10 s on ROCm against 3.6 s on the CPU** (about
  35x). `tabfm_gpu_precompile` moves the compile off the query path.
- **CUDA / MLX:** no external-data graph is bundled and nothing was run on either. Upstream's own CUDA path
  runs in fp16; this export is float32.

```bash
cd tools/export_exaone && uv sync
uv run export_exaone --task classification --config real --out ./out
uv run export_exaone --task regression     --config real --out ./out
uv run make_exaone_fixture ../../test/fixtures/exaone
TABFM_REAL_WEIGHTS=1 uv run pytest
```

## Kumo-Tabular (NVIDIA) — shipped (classify + regress, three sizes)

NVIDIA's tabular foundation model, released on Hugging Face as `nvidia/Kumo-Tabular` with code in
`NVIDIA/structured-data-models` (`sdm/models/kumo/tabular`). It descends from TabFM and TabICL: Fourier
cell embeddings with separate numerical / categorical frequencies and a missingness projection, induced
column attention over feature groups of three, RoPE row attention that reads the row out through four
learned tokens, then an in-context-learning transformer with test-time GQA. Three sizes ship as three
catalog entries:

| id | params (classifier / regressor) | download (clf + reg) | TabArena Elo (NVIDIA, 8 estimators) |
|---|---|---|---|
| `kumo-tabular-small` | 27.5 M / 28.5 M | 110 + 114 MB | 1820 |
| `kumo-tabular-medium` | 61.5 M / 62.5 M | 246 + 250 MB | 1936 |
| `kumo-tabular-large` | 213.7 M / 215.7 M | 855 + 863 MB | 2017 |

The ids spell out upstream's own size names (`KumoTabular(size="small")`, the `small/` directory on HF)
rather than abbreviating them. The Elo figures are NVIDIA's and use `num_estimators=8`; the engine runs
one estimator, so they are not a claim about these entries.

### Licence

Code Apache-2.0. Weights **OpenMDW-1.1** (`LICENSE` in the HF repo, `Copyright (c) 2026 NVIDIA
CORPORATION & AFFILIATES`): permission "to deal in the Model Materials without restriction", commercial
use included, with no field-of-use clause and no claim on outputs. The repo is not gated. So all three
entries are `commercial: true`, `redistributable: true` and have **no licence gate**
(`tabfm_download` works without `anofox_tabfm_accept_hf_license`). The agreement's one duty applies to
distribution: whoever distributes any portion of the Model Materials must retain a copy of the agreement
and NVIDIA's copyright and origin notices. "Model Materials" is defined to include the architecture, so the weight-free graphs
bundled into the binary may count as "a portion". We take that reading, because complying costs nothing and
makes the question moot: the binary carries the agreement verbatim (`resources/license_openmdw_1_1_nvidia.txt`,
the HF `LICENSE` at the pinned revision, NVIDIA's copyright line included) and `tabfm_list_models()` shows it,
after the registry's notice, in the `attribution` column of each Kumo row. `THIRD_PARTY_NOTICES.md` says the
same for the source tree. There is also a patent-retaliation clause (rights end for anyone who sues claiming the
materials infringe). This is our reading, not legal advice.

### How it runs

**The graph contract is tabfm-v1's five inputs**, `x[1,T,H]`, `y[1,T]`, `train_size[1]`, `cat_mask[1,H]`,
`d[1]`, not the positional `(x, y[1,S])` the TabICL family uses. Upstream already has the masked form:
`_KumoTabular.forward_train(..., num_columns=, train_size=)` is its own padded-context / padded-column
path (structured-data-models #1204, #1297), where every split is a mask. The exporter
(`tools/export_kumo`) calls that path, so the split and the real width are values, padding is inert, and
**one graph family serves the CPU, CUDA (ext graph) and bucket-padded ROCm (migraphx graph)**. There
was no positional version to convert later. Gated in `tools/export_kumo/tests/test_export.py`: padded in
both axes equals unpadded (< 1e-5), the masked path equals upstream's own positional inference forward
(< 1e-5), and the negative controls (padding declared real, split off by one) are caught.

**Preprocessing is the engine's.** Kumo's single-estimator numeric recipe is, step for step, the engine's
standardising profile: `DropConstantColumns` = the unique-value filter, `Standardize(eps=1e-6)` +
`Clip(-100, 100)` = the z-score with population std + 1e-6 and the clip, `ClipSigma(4, "log")` = the
two-stage 4-sigma bounds with the log1p soft clip, and the regression target's `Standardize()` = the
engine's target z-score and its inverse. Both sides compute it in float64. So the profile is
`kumo_tabular_v1_minimal` (engine standardises), not `*_raw`. The regression graph emits the mean of the
999 quantiles, which is upstream's own point estimate (`benchmark/tabular/model.py` takes
`out.numerical.mean(-1)` for TabArena). The mean commutes with the affine target inverse.

**Weights.** The published `<size>/{classifier,regressor}.pt` is a plain float32 state_dict, declared as
`<size>/<task>/model.ckpt`. On the CPU the engine's native reader injects it directly, with no conversion.
The GPU graphs (ext for CUDA and the CPU low-memory path, migraphx for ROCm) reference a safetensors file by
offset. `tabfm_download` writes it beside each `.pt` (a second row, status `converted`): every key in sorted
order as contiguous float32 with no metadata, so the header is deterministic and `ExpectedWeightsHeaderShaFor`
pins it per size and task. The bytes are identical to what `tools/export_kumo`'s `convert_kumo_weights` writes,
which stays as the reference.

**The classifier checkpoints store two tensors transposed.** `row_embedding.y_emb.weight` and
`icl_block.y_emb.weight` are saved as views with stride `(1, n)`. The native reader used to hand out raw
storage, so on a cache holding only the `.pt` (every fresh download, before this release wrote the
safetensors) the CPU fed those embeddings in the wrong order: kumo-tabular-small classified a 3-class probe at
28-34% instead of 94%. The parity numbers above were measured with the converted file present, which the
engine prefers, so they did not see it. The reader now gathers every strided view in index order;
`test/sql/tabfm_kumo_tabular.test` pins it on a fixture checkpoint saved with transposed views. Pinned to HF tag `v1.0.1` (`b2f5a9d6…`), the
revision upstream's loader asks for; the LFS hashes are identical on `main`.

Four things the export had to change, each found by a failure: integer `index % num_columns` in the group
wrap became conditional subtraction (MIGraphX's GPU target miscomputes data-dependent integer division,
`docs/ROCM_TABPFN_PLAN.md`); upstream hands ONE `RotaryEmbedding` to every row block, so 8 checkpoint keys
were one ONNX initializer and the engine refused the graph, and unsharing the leaf module gives a 1:1 map;
the qkv / RoPE / readout splits lowered to `SplitToSequence` + `SequenceAt` (166 in the small graph), which
MIGraphX and the MLX interpreter lack, and a graph rewrite turns them into `Slice`; and query rows carry the
engine's `-100` label sentinel, which `Embedding` cannot index (upstream zeroes those rows' label
embedding anyway).

### What is verified, on the real weights

**Parity with upstream** (`tools/export_kumo/tests/test_real_weights.py`, CPU). The reference is upstream's
public `KumoTabular(...)(x_context, y_context, x_query)` with `num_estimators=1` and the default recipe made
deterministic: the random `FlipSign`, `ShuffleColumns` and `ShuffleCategories` are removed and the
round-robin `Choice` is pinned to its first member, `Identity`. The candidate is the shipped ext graph under
ONNX Runtime on the float64 replica of the engine's preprocessing, which `test/sql/tabfm_kumo_tabular.test`
in turn checks against the engine on every fixture row.

| size | iris / wine / breast cancer | diabetes (spread 321) | synthetic regression (spread 24) |
|---|---|---|---|
| small | labels 100 %, probabilities ≤ 6.0e-6 | ≤ 2.4e-4, query R² 0.560 | ≤ 7.4e-6, R² 0.9994 |
| medium | labels 100 %, ≤ 6.7e-6 | ≤ 3.8e-4, R² 0.561 | ≤ 1.1e-5 |
| large | labels 100 %, ≤ 1.5e-5 | ≤ 4.1e-4, R² 0.559 | ≤ 9.5e-6 |

For scale: upstream's own *default* recipe with one estimator is not this function. On breast cancer its
random sign flips and shuffles move probabilities by up to 0.34 against the deterministic member (labels
agree 98.8 %); with 8 estimators, 0.20. The engine reproduces the deterministic member.

**Through the engine** (built-in path, `test/sql/tabfm_real_models.test` split: 168 context rows, 64 query
rows, 4 features), CPU:

| size | accuracy | R² | warm call (240 rows) | peak RSS |
|---|---|---|---|---|
| small | 0.969 | 0.999998 | 0.22 s | 0.6 GB |
| medium | 0.938 | 0.999995 | 0.7 s | 1.3 GB |
| large | 0.969 | 0.999998 | 1.0 s | 3.0 GB |

The real-weights test also asserts in-context fitted values (`is_training` rows) and query rows separately,
on all three sizes.

**ROCm** (gfx1201, RX 9070 XT, fp32, built-in path with the converted weights;
`tools/gpu_test/scenarios/kumo_rocm.sql`, 240 rows x 4 features padded to the 512 x 16 bucket, every
row compared with the CPU in the same session):

| size | served by | label disagreements (query / context) | regression max abs diff | warm call ROCm / CPU | cold compile per bucket (clf / reg) |
|---|---|---|---|---|---|
| small | `rocm:0` | 0 / 0 of 240 | 5.9e-6 | 0.04 s / 0.22 s | ~5 min / ~5 min |
| medium | `rocm:0` | 0 / 0 of 240 | 5.5e-5 | 0.18 s / 0.7 s | 39 min / 39 min |
| large | `rocm:0` | 0 / 0 of 240 | 6.9e-5 | 0.19 s / 1.0 s | 41 min / 33 min |

The first compile of each (size, task, bucket) is long; `tabfm_gpu_precompile` moves it off the query path.
Only the 512 x 16 bucket was measured; larger buckets were not compiled.

**A MIGraphX miscompile, found and worked around.** The first medium run on ROCm was wrong: 6 of 64 query
labels differed from the CPU and regression was off by 0.67 on average (R² 0.96 against 0.99999), while
small was exact. Small is the one size without test-time GQA (medium and large attend query rows with 2 of
their 8 / 16 K/V heads). Bisected on random-init models at medium dims with 2 layers, CPU vs ROCm through
the engine: with GQA the GPU is off by about half the output spread; the same model without GQA agrees to
2e-7; the fixture (1 K/V head) agrees too. `migraphx-driver verify -b` (reference target vs GPU) puts the
first divergence in the first ICL layer's query path, at a concat of head slices of a transposed K/V
tensor. Rewriting the head expansion as slices + Concat did not help; a one-hot [Hq, Hkv] GEMM over the
reshaped K/V did (`verify` passes, engine difference 2.3e-7). It is the identical function: the upstream
parity numbers above did not move and the fixture goldens are unchanged. The ORT paths (CPU, CUDA) were
never affected; this only shows up on the GPU, so a CPU-only parity gate cannot see it
(`tools/export_kumo/tests/test_export.py` checks the rewrite against torch's own `enable_gqa`).

### Size and speed (CPU, 8 threads, 80 % context)

| model | rows x features | wall | peak RSS |
|---|---|---|---|
| small | 2000 x 20 | 5.4 s | 1.8 GB |
| small | 5000 x 20 | 19 s | 5.0 GB |
| small | 10000 x 50 | 73 s | 30 GB |
| large | 2000 x 20 | 26 s | 4.5 GB |
| large | 5000 x 20 | 123 s | 14.7 GB |

The masked graph attends over all T rows (context rows attend to the context as queries do), so memory grows
faster than linearly in rows. The engine caps all three at **5,000 rows**, 500 features (upstream's own
`SelectColumns(500)`) and 10 classes.

### What it does not claim

- **One estimator.** NVIDIA's leaderboard figures are 8-member ensembles (sign flips, column shuffles, class
  shifts, four numeric transforms in rotation).
- **More than 10 classes** is refused by the head width (`supports at most 10 classes`). Upstream reaches
  more with error-correcting output codes (random codebooks over several forward passes), which the engine
  does not reproduce.
- **Categoricals differ in two details.** The engine codes a category by first appearance in the context
  (and drops categories seen once); upstream sorts by value. The category-count columns upstream adds for
  columns with 50 or more categories are not added. The categorical embedding itself is used: `cat_mask`
  reaches the graph (asserted in the fixture test). Upstream's `infer_stypes` also treats a two-valued
  numeric column as categorical; the engine treats any numeric column as numeric.
- **NULLs are mean-imputed by the engine** before the graph, so Kumo's missing-value projection is unused.
- **Context K/V caching** (upstream's prepare/query split) is not exported; every call re-encodes the
  context.
- **CUDA was not run**; no CUDA device was available. The ext graphs and header hashes are bundled.
- **MLX serves all three sizes**, both tasks. On an Apple M3 with the real weights they agree with the CPU
  (0/100 label differences, regression within 5e-5); `tabfm_mlx_real_models.test` pins that.
- **The GPU paths need the converted file**, which `tabfm_download` now writes. Without it the weights are
  the raw `.pt`, whose header no GPU graph is pinned to, so ROCm and CUDA report the weights as not matching
  and `auto` runs on the CPU.

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

**ROCm** serves both tasks through a second, MASKED graph per task (`tools/export_tabicl --contract mask`,
`docs/ROCM_SINGLE_EVAL_POS.md`): the same initializers and safetensors, the split as a value. On the real
checkpoints the masked forward matches the positional one to 2.6e-5 (five negative controls caught). On a
gfx1201 (RX 9070 XT, fp32, 100 x 8, 70 context rows; `test/sql/tabfm_rocm_real_models.test`): served by
`rocm:0`, **0/100** label differences from the CPU, regression max |diff| 4.4e-5 on a scale of 7.3; cold
compile 228 s (classification) / 203 s (regression) at T128 H16; warm 0.020 s against 0.134 s on the CPU.

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

**No conversion step.** `Orion-BiX-v1.1.ckpt` carries its `config` and optimizer state as lists beside
the state_dict, which a protocol-2 pickle builds with `APPEND` / `APPENDS`. The native reader used to reject
opcode 0x65 (`APPENDS`), so this section required `tools/export_orion_bix/convert_weights.py`; it now reads
past lists, and the CPU serves the `.ckpt` alone with predictions identical to the converted file.
`tabfm_download` also writes the `model.safetensors` the CUDA ext graph reads, byte for byte what
`convert_weights.py` writes (277/277 tensors), which stays as the dev-side check that the tensor map covers
the checkpoint.

**ROCm: served on `rocm:0`, verified on a real gfx1201 with the real weights.** The CPU graph's split is
positional (`len(y)`), so a second, MASKED graph is bundled (`graph_migraphx_orion_bix_classification.onnx`,
`tools/export_orion_bix --contract mask`, `orion_mask.py`): `x, y_full, train_size, d`, reading the same
`model.safetensors` as the CUDA ext graph (so the pinned header sha is unchanged). Orion-BiX touches the split in
two places and the padded columns in one, so three key masks suffice: the column embedder's inducing points attend
to the context rows only, every ICL query attends to the context rows only (upstream's split attention is exactly
that) with the label embedding on the context rows only, and padded columns are masked out of the row attention.
There is no `n_rows`: a padded row is a query everywhere, never a key. The gate compares padded + masked with the
positional forward on the unpadded table: worst 9.3e-5 on logits up to 12 (fp32 reassociation; in float64 they
agree to 1.5e-13), and each of four negative controls is caught by 2.8 or more. No MIGraphX rewrite was needed
beyond the conversions' usual habits (attention written out with a -1e4 key bias, RoPE without einsum). Through
the engine, 100 rows x 8 features (the 128 x 16 bucket), fp32, against the CPU: **0 / 100 label
disagreements** (context and query). First call on a bucket compiles for about **2.2 minutes** (134 s); warm calls
take **0.017 s on ROCm against 0.12 s on the CPU** (`tools/gpu_test/scenarios/orion_rocm.sql`).

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

## Google TabFM 1.1 (`tabfm-v1-1`) — shipped (classify + regress)

Google's second TabFM release (HF `google/tabfm-1.1.0-pytorch`, published 2026-08-01, pinned to commit
`d7cfd3b8…`). It is a **separate catalog entry**; `tabfm-v1` is unchanged and still serves 1.0.0.

### What changed from v1, and what did not

The architecture code is the same: `vendor/tabfm`'s `model.py` at the pinned submodule loads the 1.1
checkpoint with `strict=True` (913 classification / 918 regression tensors, every key the same as v1's).
Upstream's public code has **no 1.1-specific module** (the model card's `tabfm_v1_1_0_pytorch` import
exists neither on GitHub `main` nor in the PyPI releases 1.0.0 / 1.0.1), so the submodule was **not
bumped**: newer upstream commits change estimator code (context caching, hold-out sets) and none touches
the forward. What changed is the checkpoint:

| | v1 (1.0.0) | 1.1 (1.1.0) |
|---|---|---|
| `embed_dim` (icl_dim = 8 x) | 256 (2048) | **128 (1024)** |
| `decoder_hidden` in config.json | `null` → icl_dim x 2 = 4096 | **2048** (also icl_dim x 2) |
| parameters | 1.64 B | **410 M** clf / 412 M reg |
| tensor dtype on disk | F32 | **BF16** |
| file per task | 6.56 GB | **0.82 GB** |

`tools/export_onnx` now takes `--hf-repo/--hf-revision` and passes an explicit `decoder_hidden` through
(it dropped the field only when null, so the value happened to be right; the 1.1 test pins it). The
exported **tensor maps are byte-identical to v1's** and are reused; the graphs are new because the dims are
baked into Reshape constants. BF16 needs nothing new on the CPU injection path (the safetensors reader
upcasts exactly), but the external-data and MIGraphX graphs point ONNX initializers straight at the file,
so `tools/make_migraphx_graph.py` (`retype_external`, also used by `make_external_graph.py`) now retypes a
BF16-stored initializer to BF16 and adds a `Cast(to=FLOAT)`; it refuses any other dtype mismatch and any
byte length that does not equal numel x itemsize. The v1 ext and MIGraphX graphs regenerate byte-identical
with the changed tools.

### Licence

`LICENSE` in the 1.1 repo is byte-identical to v1's (sha256 `26d8c474…`): the TabFM Non-Commercial License
v1.0 — testing, evaluation and research only, no commercial or production use, no redistribution. So the
entry is `commercial: false`, gated by `anofox_tabfm_accept_hf_license`. The Hugging Face repo itself is
**not access-gated** (API `gated: false`, like v1's): no token is needed.

### What is verified, on the real weights

- **Graph vs upstream PyTorch** (shipped weight-free graph + injected 1.1 weights in ORT, against
  `vendor/tabfm`'s `TabFM` with the same weights): worst test-row difference 7.6e-05 on classification
  logits and 2.5e-06 on regression, argmax identical, at (T, H) = (16, 4), (240, 4), (300, 25). The
  external-data and MIGraphX graph variants run in ORT CPU agree to 3.1e-05 / 9.8e-06.
- **SQL end to end vs upstream's estimator** (the standard split of `tabfm_real_models.test`, engine vs
  `TabFMClassifier` / `TabFMRegressor` with `n_estimators=1`, norm none, no shuffle, no class shift, no AMP —
  the single-member configuration the engine implements): probabilities within **9.4e-06**, labels 64/64,
  regression within **4.9e-06** on a target spread of 13.8.
- **Quality on that split (CPU):** accuracy **0.969**, R² **0.99996**. `tabfm-v1` on the same split: 0.969 and
  0.99991. The context-row (`is_training`) oracles pass as for every other model.
- The two CPU weight paths (external-data graph, the default; forced injection,
  `TABFM_DISABLE_EXTERNAL_DATA=1`) give byte-identical answers.
- Offline: `test/sql/tabfm_v1_1.test` (catalog entry, licence gate before any I/O, and a random-init BF16
  fixture with a non-default `decoder_hidden` whose 20 rows per task are pinned against the upstream
  estimator).

### Size and speed

CPU, default threads, the 240-row split (168 context + 64 query rows, 5 features), cold: **~6 s** per task
including the model load, against ~39-54 s for `tabfm-v1`.

### Platforms

- **CPU:** Linux x86-64, as above.
- **ROCm (gfx1201, RX 9070 XT, fp32 default):** served on **`rocm:0`** (`tools/gpu_test/scenarios/tabfm_v1_1_rocm.sql`
  prints `V11_GPU_SERVED_BY=rocm:0`). Against the CPU on the standard split: **0/64 label disagreements**, top
  probability within 1.1e-05, GPU accuracy 0.969; regression max abs diff 1.1e-05, correlation 1.000000, GPU
  R² 0.99996. Warm predict 0.27 s. The cold MIGraphX compile of the T512/H16 bucket took **36 min
  (classification) and 24 min (regression)** — measured while other builds shared the CPU, so not faster than
  v1's ~25 min despite the 4x smaller model; the `.mxr` is 1.66 GB per bucket. Precompile
  (`tabfm_gpu_precompile`) or stage `.mxr` files (`anofox_tabfm_mxr_source`) as for v1.
- **CUDA:** the external-data graphs and header hashes are bundled, but **no CUDA device was available, so
  nothing was run on CUDA**.
- **MLX:** not run.

### TabArena

TabArena lists **"TabFM (default)"** (1815 Elo on the current all-datasets board, no imputation; 1823 in the
open-entrants view) and **"TabFM+"**. It does not say which checkpoint it measured. TabFM was added on
2026-07-08 (v0.1.5.2), **before 1.1.0 existed**, and the 2026-09-17 re-run notes only a timing change with
Elo "unchanged within their confidence intervals"; the entry links the public GitHub code, which loads 1.0.0.
So the figure is most likely **1.0.0, i.e. `tabfm-v1`**, and no leaderboard figure is claimed for 1.1.

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
