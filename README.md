# anofox-tabfm

**Zero-shot machine learning for tabular data, inside DuckDB.** This extension
embeds Google's [TabFM](https://github.com/google-research/tabfm) foundation
model — a TabPFN-style in-context learner — so classification and regression
become a single SQL statement. No Python, no training loop, no MLOps: the model
reads your labelled rows as context and predicts the rest.

```sql
LOAD anofox_tabfm;
SELECT * FROM tabfm_classify('history', 'churned', test := 'prospects');
```

---

## Quickstart

### 1. Install & load

```sql
INSTALL httpfs;          -- weights are fetched over HTTPS
LOAD httpfs;
LOAD anofox_tabfm;
```

### 2. Download the model (once)

The extension ships only a **weight-free** computation graph; you download the
weights yourself from Hugging Face under Google's license.

```sql
SET anofox_tabfm_accept_hf_license = true;   -- non-commercial, no redistribution
CALL tabfm_download('classification');       -- ~6.6 GB, cached in ~/.cache/anofox-tabfm
CALL tabfm_download('regression');           -- optional, for tabfm_regress
```

```
┌──────────────────────────────┬───────────────────────────────────────┬────────────┬────────────┐
│             file             │                  url                  │   bytes    │   status   │
├──────────────────────────────┼───────────────────────────────────────┼────────────┼────────────┤
│ classification/model.safeten…│ huggingface.co/google/tabfm-1.0.0-…   │ 6557888408 │ downloaded │
└──────────────────────────────┴───────────────────────────────────────┴────────────┴────────────┘
```

### 3. Predict

The weight-free graph is bundled in the extension, so nothing else to configure —
just predict:

```sql
-- customers with a known churn label are the context; NULL-label rows are scored
SELECT age, plan, churned, yhat, yhat_score
FROM tabfm_classify('customers', 'churned')
WHERE churned IS NULL;
```

```
┌───────┬───────┬─────────┬──────┬────────────┐
│  age  │ plan  │ churned │ yhat │ yhat_score │
├───────┼───────┼─────────┼──────┼────────────┤
│ 40.0  │ basic │ NULL    │ no   │       0.83 │
│ 60.0  │ pro   │ NULL    │ yes  │       0.71 │
└───────┴───────┴─────────┴──────┴────────────┘
```

That's it — a state-of-the-art tabular model scoring your data, from SQL.

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
CREATE TABLE preds AS
SELECT customerID, yhat AS pred
FROM tabfm_classify('train', 'Churn', test := 'test_features');

-- 3. F1 of the positive class, in SQL
WITH cm AS (
  SELECT count(*) FILTER (WHERE p.pred AND t.Churn)         AS tp,
         count(*) FILTER (WHERE p.pred AND NOT t.Churn)     AS fp,
         count(*) FILTER (WHERE NOT p.pred AND t.Churn)     AS fn
  FROM preds p JOIN test t USING (customerID))
SELECT 2.0*tp / nullif(2.0*tp + fp + fn, 0) AS f1 FROM cm;
```

On this dataset the zero-shot model reaches **F1 0.667 / accuracy 0.827**. The
same three-line recipe generalizes: multiclass `scikit-learn/iris` reaches
**accuracy 0.943**, and the regression counterpart on `scikit-learn/tips` reaches
**MSE 0.971** vs a mean-predictor baseline of 1.68. The runnable scripts
(`classification_churn.sql`, `classification_iris.sql`, `regression_tips.sql`)
and full numbers are in [`examples/`](examples/README.md).

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
SELECT * FROM tabfm_devices();         -- discovered execution devices (CPU / CUDA / ROCm)
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
| `anofox_tabfm_max_rows` | `10000` | guardrail per predict / group |
| `anofox_tabfm_max_features` | `500` | guardrail |
| `anofox_tabfm_device` | `auto` | `cpu` / `cuda` / `rocm` execution device |
| `anofox_tabfm_model_manifest` | `''` | point at a custom model manifest |

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
`tabfm_download` the model works with no companion files. v1 runs a single
estimator on CPU (`cpu` flavor). Not yet wired up: the `n_estimators > 1`
ensemble, and the grouped / composable-aggregate / windowed surfaces
(`tabfm_predict_by` / `_agg` / `_win`) — planned on the same engine.

## Flavors (CPU / GPU)

One codebase, three builds (`TABFM_FLAVOR`): `cpu` (default, community-extension
eligible), `cuda` (NVIDIA, bf16), `rocm` (AMD via ONNX Runtime's MIGraphX EP).
GPU builds link no vendor runtime — CUDA/cuDNN or ROCm resolve from your system,
and `tabfm_devices()` reports what was found. GPU flavors ship from the anofox
extension repository (`SET custom_extension_repository = 'https://ext.anofox.com/tabfm/<flavor>'`).

## License

- **This extension's code:** MIT.
- **Model weights:** you download them from `google/tabfm-1.0.0-pytorch` under
  Google's `tabfm-non-commercial-v1.0` license (non-commercial, no
  redistribution). The extension ships only a weight-free graph and never
  redistributes or writes converted weights to disk; the license gate is
  enforced before any download. The repository and test fixtures contain zero
  Google weight bytes.

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

Requires CMake ≥ 3.19 and a C++17 toolchain. ONNX Runtime is fetched as a
prebuilt archive by default; enable the `ort-vcpkg` manifest feature to build it
from source. See [`CLAUDE.md`](CLAUDE.md) for the module map and
[`examples/`](examples/README.md) for end-to-end examples.
