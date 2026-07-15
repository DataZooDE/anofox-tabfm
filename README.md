# anofox-tabfm

**Zero-shot machine learning for tabular data, inside DuckDB.** This extension
runs **tabular foundation models** — TabPFN-style in-context learners — so
classification and regression become a single SQL statement. No Python, no
training loop, no MLOps: the model reads your labelled rows as context and
predicts the rest.

**Four models are built in and selectable by name** — `mitra` (AWS AutoGluon,
Apache-2.0), `tabpfn-v2` (Prior Labs), `tabicl-v2` (Inria), and `tabfm-v1`
(Google TabFM) — and you can register your own entirely in SQL. Everything is
operated in SQL: no manifest file, no config.

```sql
LOAD anofox_tabfm;
SELECT * FROM tabfm_classify('history', 'churned', test := 'prospects', model := 'mitra');
```

---

## Quickstart

### 1. Install & load

```sql
INSTALL httpfs;          -- weights are fetched over HTTPS
LOAD httpfs;
LOAD anofox_tabfm;
```

### 2. Pick a model and download its weights (once)

`SELECT * FROM tabfm_list_models();` shows the four built-in models. `mitra` is a
good default — Apache-2.0, ~303 MB, no license gate. The extension ships only
**weight-free** graphs; the weights are your own Hugging Face download:

```sql
CALL tabfm_download('classification', model := 'mitra');    -- ~303 MB, cached in ~/.cache/anofox-tabfm
```

The gated Google model (`tabfm-v1`) additionally needs
`SET anofox_tabfm_accept_hf_license = true;`.

### 3. Predict

A concrete example — the classic **iris** dataset (four flower measurements →
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

## The two shapes

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

**A subquery** works anywhere a table name does:

```sql
SELECT * FROM tabfm_classify(
    '(SELECT * FROM history WHERE signup_year = 2025)', 'churned',
    test := 'prospects');
```

**Feature selection and options** (named parameters read best):

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

## Multiple models (the registry)

`anofox_tabfm` is one entrypoint for many **tabular foundation models** — "TabFM"
is the *category*, not a single model. Four models are **built in** and usable by
name with no config, no manifest file:

```sql
SELECT * FROM tabfm_list_models();          -- the registry: every known model
```

| `model` | family | license | `commercial` |
|---|---|---|---|
| `tabfm-v1` | Google TabFM | non-commercial (gated) | `false` |
| `mitra` | AWS AutoGluon | Apache-2.0 | `true` |
| `tabpfn-v2` | Prior Labs | Apache-2.0 + attribution | `true` |
| `tabicl-v2` | Inria | BSD-3-Clause | `true` |

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
┌───────────┬────────────────┬──────────┬──────────────┬────────────┬─────────┬──────────────────────────┐
│   model   │      task      │ revision │     path     │   bytes    │ loaded  │         license          │
├───────────┼────────────────┼──────────┼──────────────┼────────────┼─────────┼──────────────────────────┤
│ tabfm-v1  │ classification │ main     │ …/model.saf… │ 6557888408 │ true    │ tabfm-non-commercial-v1.0│
└───────────┴────────────────┴──────────┴──────────────┴────────────┴─────────┴──────────────────────────┘
```

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
| `anofox_tabfm_max_rows` | `10000` | guardrail per predict / group |
| `anofox_tabfm_max_features` | `500` | guardrail |
| `anofox_tabfm_device` | `auto` | `auto` / `cpu` / `cuda` / `rocm` / `coreml` (`migraphx` alias for `rocm`) |
| `anofox_tabfm_gpu_precision` | `bf16` | GPU dtype: `bf16` / `fp16` / `fp32` |
| `anofox_tabfm_default_model` | `''` | session-wide model when `model :=` is not given |
| `anofox_tabfm_mxr_source` | `''` | directory of precompiled ROCm `.mxr` programs to stage from |
| `anofox_tabfm_ep_path` | `''` | extra search path for execution-provider shared libraries |
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
eligible), `cuda` (NVIDIA, bf16), `rocm` (AMD via a **direct MIGraphX backend** —
ONNX Runtime's MIGraphX EP can't load the >2 GB model, so ROCm bypasses it and
drives libMIGraphX directly, with a compiled-program `.mxr` cache), and `coreml`
(Apple Silicon via ONNX Runtime's CoreML EP — same macOS archive as `cpu`, GPU/ANE
where the graph is supported, CPU fallback otherwise). GPU builds
link no vendor runtime — CUDA/cuDNN or ROCm resolve from your system, and
`tabfm_devices()` reports what was found. GPU dtype is set by
`anofox_tabfm_gpu_precision` (default `bf16`); `CALL tabfm_gpu_precompile(task)`
warms a shape bucket ahead of the first predict (builds/caches the ROCm `.mxr`).
GPU flavors ship from the anofox extension repository
(`SET custom_extension_repository = 'https://ext.anofox.com/tabfm/<flavor>'`).

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
GEN=ninja make release        # cpu flavor
make test_release             # sqllogictests + C++ unit tests
```

Requires CMake ≥ 3.10 (≥ 3.19 to also build the bundled C++ unit tests) and a
C++17 toolchain. ONNX Runtime is fetched as a
prebuilt archive by default; enable the `ort-vcpkg` manifest feature to build it
from source. See [`CLAUDE.md`](CLAUDE.md) for the module map and
[`examples/`](examples/README.md) for end-to-end examples.
