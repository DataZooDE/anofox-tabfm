# anofox-tabfm

**Zero-shot machine learning for tabular data, inside DuckDB.** This extension
runs **tabular foundation models** — TabPFN-style in-context learners — so
classification and regression become a single SQL statement. No Python, no
training loop, no MLOps: the model reads your labelled rows as context and
predicts the rest.

**Seven models are built in and selectable by name** — `mitra` (AWS AutoGluon,
Apache-2.0), `tabpfn-v2`, `tabpfn-v2-5` and `tabpfn-v3` (Prior Labs),
`tabicl-v2` (Inria),
`orion-bix` (Lexsi Labs, MIT), and `tabfm-v1` (Google TabFM) — and you can
register your own entirely in SQL. Everything is
operated in SQL: no manifest file, no config.

---

## Quickstart

### 1. Install & load

```sql
INSTALL httpfs;          -- weights are fetched over HTTPS
LOAD httpfs;
LOAD anofox_tabfm;
```

### 2. Pick a model and download its weights (once)

`SELECT * FROM tabfm_list_models();` shows the seven built-in models. `mitra` is a
good default — Apache-2.0, ~303 MB. It also has a permissive license. The extension ships only
**weight-free** graphs; the weights have to be downloaded from Hugging Face before using the model:

```sql
CALL tabfm_download('classification', model := 'mitra');    -- ~303 MB, cached in ~/.cache/anofox-tabfm
```

Models whose license needs an explicit nod (`tabfm-v1`, `tabpfn-v2`, `tabpfn-v2-5`,
`tabpfn-v3` — "(gated)" in the table below) additionally need
`SET anofox_tabfm_accept_hf_license = true;` before `tabfm_download`.

Some repositories are additionally **gated by Hugging Face** — accept the license on
the model page while signed in, then pass your token via a standard DuckDB secret:

```sql
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_xxx', SCOPE 'https://huggingface.co');
```

A download that needs this fails with an error naming both steps. See
[`docs/REAL_MODELS.md`](docs/REAL_MODELS.md#gated-huggingface-repositories).

### 3. Predict

A concrete example — the classic **iris** dataset (classifying four flower measurements into 
species), read straight from Hugging Face, with 20% of the species hidden as
`NULL` so we can predict them:

```sql
CREATE TABLE iris AS
SELECT SepalLengthCm AS sepal_len, SepalWidthCm AS sepal_wid,
       PetalLengthCm AS petal_len, PetalWidthCm AS petal_wid,
       CASE WHEN row_number() OVER () % 5 = 0 THEN NULL ELSE Species END AS species
FROM 'hf://datasets/scikit-learn/iris/**/*.csv';
```

Rows with a known `species` are the **context**; the `NULL` rows are the ones to
score:

| sepal_len | sepal_wid | petal_len | petal_wid | species |
|--:|--:|--:|--:|:--|
| 4.9 | 3.0 | 1.4 | 0.2 | Iris-setosa |
| 5.0 | 3.6 | 1.4 | 0.2 | **NULL** |
| 5.2 | 2.7 | 3.9 | 1.4 | **NULL** |
| 6.5 | 3.0 | 5.8 | 2.2 | **NULL** |

Predict the hidden species — zero-shot, no training. Every function takes
`model :=` (or `SET anofox_tabfm_default_model = 'mitra'` once for the session):

```sql
SELECT sepal_len, sepal_wid, petal_len, petal_wid, yhat AS predicted_species, yhat_score
FROM tabfm_classify('iris', 'species', model := 'mitra')
WHERE species IS NULL;
```

| sepal_len | sepal_wid | petal_len | petal_wid | predicted_species | yhat_score |
|--:|--:|--:|--:|:--|--:|
| 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa | 1.00 |
| 5.2 | 2.7 | 3.9 | 1.4 | Iris-versicolor | 1.00 |
| 6.5 | 3.0 | 5.8 | 2.2 | Iris-virginica | 1.00 |

That's it — a foundation model scoring your data, from SQL. (On a held-out split
`mitra` gets iris **96%** right; see [`examples/`](examples/README.md).)

---

## We support classification and regression

Both functions have the same signature; the task (classification vs
regression) is fixed by which one you call.

```sql
tabfm_classify(data, target [, test] [, features] [, opts])
tabfm_regress (data, target [, test] [, features] [, opts])
```

**Explicit train / test** — like `clf.fit(X_train, y_train).predict(X_test)`.
Only the scored rows are returned.

```sql
SELECT * FROM tabfm_classify('history', 'churned', test := 'prospects');
SELECT * FROM tabfm_regress('sold_homes', 'price', test := 'listings');
```

**Single relation** — rows whose target `IS NULL` are the ones to score; every
row comes back with an `is_training` flag (context rows get in-context fitted
values, handy for a sanity check).

```sql
SELECT * FROM tabfm_classify('customers', 'churned');
```

**A subquery** works anywhere where a table name does:

```sql
SELECT * FROM tabfm_classify(
    '(SELECT * FROM history WHERE signup_year = 2025)', 'churned',
    test := 'prospects');
```

**Feature selection and options** (named parameters lead to the best readability):

```sql
SELECT * FROM tabfm_classify(
    'history', 'churned',
    test     := 'prospects',
    features := ['age', 'plan', 'usage_gb'],   -- default: all other columns
    opts     := MAP{'seed': '42', 'output_mode': 'detail'});
```

### Output columns

Every column of the scored rows, plus:

| column | meaning |
|---|---|
| `yhat` | predicted label (classification) or value (regression) |
| `yhat_score` | top-class probability; `NULL` for regression |
| `proba` | `MAP(label → probability)` — classification, `output_mode = 'detail'` (the default) |
| `is_training` | present in single-relation mode: was this a context row? |

---

## Generate synthetic data and fill gaps

The same in-context engine also works *backwards*: instead of predicting one
column, it can model the whole table as a joint distribution and sample from it.

```sql
tabfm_generate(data, n [, features] [, opts] [, model])           -- sample new rows
tabfm_impute  (data [, columns] [, features] [, opts] [, model])  -- fill NULLs
```

```sql
-- 500 synthetic customers that look like the real ones
SELECT * FROM tabfm_generate('customers', 500);

-- ...and straight back into the table, since the shape matches
INSERT INTO customers SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('customers', 500);

-- fill every NULL cell, conditioned on the rest of each row
CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw');
SELECT * FROM tabfm_impute('raw', columns := ['income', 'plan']);
```

Generation works column by column: each column is sampled conditioned on the
ones already generated (the chain rule), so the **relationships between columns
survive** — not just each column's marginal. It needs only classification
weights, and works with every model in the registry, including classify-only
ones.

Imputation is the deterministic sibling: it takes the conditional best estimate
rather than sampling, so continuous fills keep full precision. Non-NULL cells
are never modified.

### Output columns

`tabfm_generate` returns every column of `data`, plus:

| column | meaning |
|---|---|
| `synthetic_id` | `1..n` in generation order — joinable and reproducible |

`tabfm_impute` returns **exactly** the columns of `data`, so it round-trips.

### Options

| key | default | applies to | meaning |
|---|---|---|---|
| `seed` | `42` | both | RNG seed; same seed → same rows within a build |
| `temperature` | `1.0` | generate | diversity; higher explores more, lower hugs the data |
| `bins` | `10` | generate | quantile buckets per continuous column, 2–10 (clamped to the model's class-head width) |
| `column_order` | `random` | both | `random` \| `natural` \| `missingness` |
| `rounds` | `1` | impute | MICE-style refinement sweeps, 1–16 |

Cost is one model call per column, run sequentially — it scales with the number
of **columns**, not with `n`. Continuous columns are sampled through quantile
bins, so generated values never leave the observed range and resolution is
capped at `bins` levels; high-cardinality and temporal columns cannot be
generation targets and must be excluded with `features :=`.

**[`docs/GENERATE.md`](docs/GENERATE.md)** explains the method, what the binning
does and does not preserve, and why this is *not* a privacy mechanism.
Runnable samples: [`examples/generate_synthetic.sql`](examples/generate_synthetic.sql),
[`examples/impute_missing.sql`](examples/impute_missing.sql),
[`examples/generate_fidelity.sql`](examples/generate_fidelity.sql),
[`examples/generate_conditional.sql`](examples/generate_conditional.sql).

---

## Multiple models (the registry)

`anofox_tabfm` is one entrypoint for many **tabular foundation models** — "TabFM"
is the *category*, not a single model. Seven models are **built in** and usable by
name with no config, no manifest file:

```sql
SELECT * FROM tabfm_list_models();          -- the registry: every known model
```

| `model` | family | license | `commercial` |
|---|---|---|---|
| `tabfm-v1` | Google TabFM | non-commercial (gated) | `false` |
| `mitra` | AWS AutoGluon | Apache-2.0 | `true` |
| `tabpfn-v2` | Prior Labs | Apache-2.0 + attribution (gated) | `true` |
| `tabicl-v2` | Inria | BSD-3-Clause | `true` |
| `orion-bix` | Lexsi Labs | MIT | `true` (classify only) |
| `tabpfn-v2-5` | Prior Labs | TabPFN-2.5 (non-commercial, gated) | `false` |
| `tabpfn-v3` | Prior Labs | TabPFN-3 (non-commercial, gated) | `false` |

Pick a model per call (a first-class argument, promoted out of `opts`), or set a
session default; precedence is **per-call → `anofox_tabfm_default_model` → a
single SQL-registered model**:

```sql
SELECT * FROM tabfm_classify('customers', 'churned', model := 'mitra');
SET anofox_tabfm_default_model = 'mitra';            -- session-wide
```

Register your **own** model entirely in SQL — no JSON file — with
`CALL tabfm_register_model(id := 'my', classification_graph := '<path|url>', …)`
(and `tabfm_unregister_model('my')`). The only external thing is the weight-free
ONNX graph blob; every other field is a named argument. An unknown `model :=`, or
a model that lacks the requested task, is a clean error naming the alternatives.
`tabfm_download` / `tabfm_load` / `tabfm_unload` / `tabfm_remove` /
`tabfm_gpu_precompile` each take the same `model :=` argument (or fall back to the
default) so the whole lifecycle is per-model.

---

## A full worked example

Zero-shot churn prediction on a public dataset, split into train/test, scored,
and evaluated — **entirely in SQL**:

```sql
INSTALL httpfs; LOAD httpfs; LOAD anofox_tabfm;

-- 1. load + a deterministic 70/30 split
CREATE TABLE churn AS
SELECT *, hash(customerID) % 100 AS bucket
FROM 'hf://datasets/scikit-learn/churn-prediction/**/*.csv';

CREATE TABLE train AS SELECT * EXCLUDE (bucket) FROM (FROM churn WHERE bucket < 70) USING SAMPLE 500 ROWS (reservoir, 42);
CREATE TABLE test  AS SELECT * EXCLUDE (bucket) FROM (FROM churn WHERE bucket >= 70) USING SAMPLE 150 ROWS (reservoir, 42);
CREATE TABLE test_features AS SELECT * EXCLUDE (Churn) FROM test;   -- no target for the test rows

-- 2. predict the test rows using the train rows as context
--    (assumes `model := 'tabfm-v1'`'s weights are downloaded — see Quickstart)
CREATE TABLE preds AS
SELECT customerID, yhat AS pred
FROM tabfm_classify('train', 'Churn', test := 'test_features', model := 'tabfm-v1');

-- 3. F1 of the positive class, in SQL
WITH cm AS (
  SELECT count(*) FILTER (WHERE p.pred AND t.Churn)         AS tp,
         count(*) FILTER (WHERE p.pred AND NOT t.Churn)     AS fp,
         count(*) FILTER (WHERE NOT p.pred AND t.Churn)     AS fn
  FROM preds p JOIN test t USING (customerID))
SELECT 2.0*tp / nullif(2.0*tp + fp + fn, 0) AS f1 FROM cm;
```

On this dataset `tabfm-v1` reaches **F1 0.667 / accuracy 0.827** zero-shot. The
same recipe generalizes across models and tasks — swap `model :=` to compare: on
iris, `mitra` / `tabpfn-v2` / `tabicl-v2` all reach **0.962** vs `tabfm-v1`'s
0.943, at a fraction of the size and ~40–60× the speed. The runnable scripts
(including `compare_models.sql`, a real head-to-head) and full numbers are in
[`examples/`](examples/README.md).

---

## Managing weights and devices

```sql
SELECT * FROM tabfm_models();     -- what's cached / loaded
```
```
┌───────────┬────────────────┬──────────┬──────────────┬────────────┬─────────┬───────────────────────────┬─────────┐
│   model   │      task      │ revision │     path     │   bytes    │ loaded  │          license          │ device  │
├───────────┼────────────────┼──────────┼──────────────┼────────────┼─────────┼───────────────────────────┼─────────┤
│ tabfm-v1  │ classification │ main     │ …/model.saf… │ 6557888408 │ true    │ tabfm-non-commercial-v1.0 │ rocm:0  │
└───────────┴────────────────┴──────────┴──────────────┴────────────┴─────────┴───────────────────────────┴─────────┘
```

`device` is which backend is serving the model right now — `cpu`, `cuda:0`,
`rocm:0` — and is empty until a predict warms a session. It is the way to check
that a GPU is actually being used: a device that silently fell back to the CPU
produces the same answers, only slower, so agreement alone will not tell you.

```sql
CALL tabfm_load('classification');    -- warm the model (else lazy on first predict)
CALL tabfm_unload();                  -- free the weights, memory returns to baseline
CALL tabfm_remove('regression');      -- delete from the cache
SELECT * FROM tabfm_devices();         -- discovered execution devices (CPU / CUDA / ROCm / CoreML)
```

For a gated or private HF repo, add a bearer token — no custom credential store:

```sql
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_…', SCOPE 'https://huggingface.co');
```

---

## Settings

| Setting | Default | Purpose |
|---|---|---|
| `anofox_tabfm_accept_hf_license` | `false` | license gate — downloads fail without it |
| `anofox_tabfm_cache_dir` | `~/.cache/anofox-tabfm` | weight cache root (pre-seed for air-gapped use) |
| `anofox_tabfm_threads` | cores / 2 | ONNX Runtime intra-op threads |
| `anofox_tabfm_cpu_prepack` | `true` | prepack weights for faster CPU matmuls (~+16% RSS) |
| `anofox_tabfm_context_cache` | `false` | encode the labelled context once and reuse it, for models that ship a prepare/query graph pair — see [docs/KV_CACHE_DESIGN.md](docs/KV_CACHE_DESIGN.md) |
| `anofox_tabfm_max_rows` | `10000` | guardrail per predict / group |
| `anofox_tabfm_max_features` | `500` | guardrail |
| `anofox_tabfm_device` | `auto` | `auto` / `cpu` / `cuda` / `rocm` / `coreml` (`migraphx` alias for `rocm`) |
| `anofox_tabfm_gpu_precision` | `fp32` | GPU numeric mode: `fp32` (strict — same answers as CPU, measured exact on both GPUs) / `tf32` (CUDA tensor-core rounding) / `bf16` / `fp16` (ROCm quantize; label flips vs fp32 are rare but measured to include high-confidence rows — validate on your data) |
| `anofox_tabfm_max_sessions` | `4` | loaded (device, precision) sessions kept per model; oldest evicted beyond this |
| `anofox_tabfm_max_memory` | `''` | refuse predicts once resident memory is at/above this (e.g. `'16GB'`); `''` disables |
| `anofox_tabfm_default_model` | `''` | session-wide model when `model :=` is not given |
| `anofox_tabfm_mxr_source` | `''` | directory of precompiled ROCm `.mxr` programs to stage from |
| `anofox_tabfm_ep_path` | `''` | directory holding the GPU backend plugin (and the runtime libraries beside it) |
| `anofox_tabfm_trace_level` | `warn` | log verbosity: `error` / `warn` / `info` / `debug` / `trace` |

Options (the trailing `opts` MAP, all values VARCHAR): `task`, `n_estimators`
(v1: `1`), `seed`, `output_mode` (`compact` / `detail`), `context_rows`,
`softmax_temperature`, `model`. Unknown keys error.

Every failure is a DuckDB exception that names the fix, e.g.:

```
Invalid Input Error: tabfm: model 'classification' is not downloaded.
Run: CALL tabfm_download('classification');
```

---

## Status & scope

The full SQL surface runs the **real TabFM v1 model** end to end (preprocess →
ONNX Runtime forward → decode); per-row outputs match the PyTorch reference to
~1e-5. The weight-free graphs are compiled into the extension, so after
`tabfm_download` the model works with no companion files. A **model registry**
(`model :=`, `tabfm_list_models()`, `tabfm_register_model`) makes a second model a
data change, not new C++ (FR-5.1). v1 runs a single estimator; it ships CPU
(`cpu`) plus accelerated flavors (`cuda`, `rocm` via a direct MIGraphX backend,
and `coreml` for Apple Silicon — see below). Not yet wired up: the
`n_estimators > 1` ensemble, and the grouped / composable-aggregate / windowed
surfaces (`tabfm_predict_by` / `_agg` / `_win`) — planned on the same engine.

## Flavors (CPU / GPU)

One codebase, four builds (`TABFM_FLAVOR`): `cpu` (default, community-extension
eligible), `cuda` (NVIDIA), `rocm` (AMD via a **direct MIGraphX backend** —
ONNX Runtime's MIGraphX EP can't load the >2 GB model, so ROCm bypasses it and
drives libMIGraphX directly, with a compiled-program `.mxr` cache), and `coreml`
(Apple Silicon via ONNX Runtime's CoreML EP — same macOS archive as `cpu`, GPU/ANE
where the graph is supported, CPU fallback otherwise). GPU builds
link no vendor runtime — CUDA/cuDNN or ROCm resolve from your system, and
`tabfm_devices()` reports what was found. GPU dtype is set by
`anofox_tabfm_gpu_precision` (default `fp32` — strict CPU parity; faster modes are opt-in); `CALL tabfm_gpu_precompile(task)`
warms a shape bucket ahead of the first predict (builds/caches the ROCm `.mxr`).
Released cpu builds are served from the anofox extension repository
(`SET custom_extension_repository = 'https://get.anofox.com'`) as well as from
the DuckDB community repository.

**Using a GPU.** A GPU backend is a plugin the extension `dlopen`s at runtime,
not a separate build of the extension: point `anofox_tabfm_ep_path` at the
directory holding it and select the device.

```sql
CALL tabfm_download_runtime('cuda');    -- fetch the ORT GPU runtime into that directory
SET anofox_tabfm_ep_path = '/path/to/plugin/dir';
SET anofox_tabfm_device  = 'cuda';      -- or 'rocm'
-- confirm it is really being used, rather than trusting the answers:
SELECT model, device FROM tabfm_models() WHERE loaded;
```

The plugins themselves are **not published yet**, so today you build the one
you need from source (`src/tabfm_cuda_plugin.cpp`,
`src/tabfm_migraphx_plugin.cpp`; see [`docs/rocm-build.md`](docs/rocm-build.md)
for the ROCm toolchain and [`docs/DYNAMIC_BACKENDS.md`](docs/DYNAMIC_BACKENDS.md)
for how the two fit together). `tabfm_download_runtime('cuda')` fetches the
ONNX Runtime GPU libraries the CUDA plugin needs, not the plugin itself.

**Any build can host the CUDA plugin now.** Earlier, a host that loaded a
*shared* `libonnxruntime.so` (the local `make debug` build) shadowed the
plugin's own ONNX Runtime — same SONAME, first one loaded wins — so CUDA
required the statically-linked release build. Fixed at both ends
(`docs/GPU_HARDENING_PLAN.md` S2): the plugin's core ships under a private
SONAME (`libanofoxort_gpu.so`, applied by `tabfm_download_runtime` and by the
plugin build itself), and the plugin loader binds with `RTLD_DEEPBIND`.
Verified on hardware including the hostile case — a foreign upstream-SONAME
ORT preloaded into the global scope — which the plugin no longer notices.

`anofox_tabfm_gpu_precision` now means something on both GPUs, with fp32 the
default everywhere: strict fp32 (on CUDA that disables the TF32 tensor-core
rounding ORT would otherwise apply to fp32 matmuls) so a device switch does
not change answers. `tf32` re-enables that rounding (CUDA only); `bf16`/`fp16`
quantize the MIGraphX program (ROCm only) for ~2x speed at the cost of a few
near-tie label flips. A mode a backend cannot honour is an error, never a
silent fp32 run.

## Feedback

If `anofox_tabfm` misbehaves — a model that will not load, a prediction that looks
wrong, a device it will not use — please
[open an issue](https://github.com/DataZooDE/anofox-tabfm/issues). Hardware, drivers and
hub availability differ in ways we cannot reproduce here, so a report with your setup is
the fastest path to a fix. Errors from the model functions end with that link.

If it saved you time, a star on the repo helps other people find it.

The first time you load the extension in an interactive terminal each day, a small
banner says the same thing. It never prints when output is piped, in notebooks, or in
CI. Silence it with `SET datazoo_banner = false;` or `DATAZOO_NO_BANNER=1`.

## License

- **This extension's code:** MIT.
- **Model weights:** each model's weights are **your own Hugging Face download**,
  under that model's license — `mitra` Apache-2.0, `tabpfn-v2` Prior Labs License
  (Apache-2.0 + attribution), `tabicl-v2` BSD-3-Clause, `tabfm-v1`
  `tabfm-non-commercial-v1.0` (non-commercial, gated behind
  `anofox_tabfm_accept_hf_license`). The extension ships only **weight-free**
  graphs and never redistributes weights; `tabfm_list_models()` shows each
  model's license and whether it's commercially usable. The repository and test
  fixtures contain **zero model weight bytes** (fixtures are random-init).

## Telemetry

Sends **anonymous** usage telemetry (extension load + per-function call counts —
no data, no queries, no identifiers beyond a hashed machine id) to PostHog EU,
matching the other anofox extensions. Opt out any time:

```bash
export DATAZOO_DISABLE_TELEMETRY=1        # environment
```
```sql
SET anofox_telemetry_enabled = false;      -- SQL
```
CI environments are auto-detected and telemetry is disabled there.

## Building

```bash
# vcpkg is required for EVERY flavor: vcpkg.json depends on openssl unconditionally.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                     # cpu flavor
GEN=ninja TABFM_FLAVOR=cuda make release   # cuda flavor (linux_amd64 / win_amd64)
make test_release                          # sqllogictests + C++ unit tests
```

Without `VCPKG_TOOLCHAIN_PATH` the build stops during configure with a cascade in which only the
first line is the real cause:

```
CMake Error: Could not find toolchain file:
  .../vcpkg_installed//share/vcpkg/scripts/buildsystems/vcpkg.cmake
CMake Error: CMake was unable to find a build program corresponding to "Ninja".
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

CMake aborts before probing for the generator or the compilers, so the last three are reported
missing even when Ninja and the toolchain are installed and on `PATH`.

Requires CMake ≥ 3.10 (≥ 3.19 to also build the bundled C++ unit tests) and a
C++17 toolchain, and vcpkg. ONNX Runtime is fetched as a
prebuilt archive by default; enable the `ort-vcpkg` manifest feature to build it
from source. See [`CLAUDE.md`](CLAUDE.md) for the module map and
[`examples/`](examples/README.md) for end-to-end examples.
