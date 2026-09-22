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
| `yhat_dist` | `STRUCT(logits DOUBLE[], borders DOUBLE[])` — regression predictive distribution, `output_mode = 'distribution'` (models that emit one, e.g. the TabPFN v2 family) |
| `yhat_quantiles` | `DOUBLE[]` — quantiles of the predictive distribution, `output_mode = 'distribution'` |
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

## Evaluation, cross-validation & scoring

Beyond prediction, the extension ships **model-agnostic** evaluation primitives —
they operate on plain `(actual, predicted[, proba])` columns, so they work on any
model's output, not just tabfm's. Every function has a full `anofox_tabfm_*` name
and a short `tabfm_*` alias.

**Classification metrics** (SQL aggregates over `(actual, predicted[, proba])`):

```sql
SELECT tabfm_accuracy(actual, predicted)                 AS acc,
       tabfm_f1(actual, predicted, avg := 'macro')       AS f1,      -- avg is REQUIRED
       tabfm_roc_auc(actual, proba, avg := 'ovr')        AS auc,     -- rank-sum, tie-safe
       tabfm_log_loss(actual, proba)                     AS logloss, -- ε-clipped
       tabfm_ece(actual, proba)                          AS ece
FROM scored;

-- precision / recall likewise require an explicit avg ('micro' / 'macro' / 'weighted')
-- confusion matrix is a table macro returning tidy (actual, predicted, count):
SELECT * FROM tabfm_confusion_matrix('scored', 'actual', 'predicted');
```

**Regression metrics** (over `(actual, predicted)`):

```sql
SELECT tabfm_rmse(actual, predicted), tabfm_mae(actual, predicted),
       tabfm_r2(actual, predicted),   -- constant-target safe (sklearn convention)
       tabfm_mape(actual, predicted), -- skips zero-actual rows
       tabfm_medae(actual, predicted)
FROM scored;
```

**Leakage-safe k-fold cross-validation** — assigns rows to `k` deterministic,
seedable folds (`hash(row_key, seed) % k`), trains each fold on the others and
predicts its held-out rows via the two-table (leakage-safe) form, and returns
per-fold plus aggregate (mean ± std) metric results:

```sql
-- positional: data, target, row_key; then optional named k / seed / task / metric
SELECT * FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5, seed := 42);
-- tabfm_fold_assign('customers', 5, 'id', 42) exposes the fold ids directly (data, k, row_key, seed)
```

**Predictive distributions & proper scoring rules** — a model that emits a
regression distribution (e.g. the TabPFN v2 family) exposes it via
`output_mode = 'distribution'` as `yhat_dist STRUCT(logits[], borders[])`; the
proper scoring rules score it (bin widths are honoured, so non-uniform bars are
correct). They are bind-gated on a distribution input and error with a named
remedy if handed a point estimate:

```sql
WITH d AS (SELECT actual, yhat_dist
           FROM tabfm_regress('sold', 'price', test := 'listings',
                              opts := MAP{'model':'tabpfn_v2','output_mode':'distribution'}))
SELECT tabfm_crps(actual, yhat_dist)                        AS crps,   -- closed-form CRPS
       tabfm_log_score(actual, yhat_dist)                   AS nll,
       tabfm_interval_score(actual, yhat_dist, 0.9)             AS is90  -- 3rd arg = coverage, default 0.9
FROM d;
```

**Cross-model comparison** — run the metrics and scores across model families on
your own table (no bundled datasets):

```sql
SELECT * FROM tabfm_compare_models('my_table', 'target');
```

See [`examples/`](examples/README.md) for runnable end-to-end scripts.

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
| `anofox_tabfm_model_manifest` | `''` | point at a custom model manifest |
| `anofox_tabfm_mxr_source` | `''` | directory of precompiled ROCm `.mxr` programs to stage from |
| `anofox_tabfm_ep_path` | `''` | extra search path for execution-provider shared libraries |
| `anofox_tabfm_trace_level` | `warn` | log verbosity: `error` / `warn` / `info` / `debug` / `trace` |

Options (the trailing `opts` MAP, all values VARCHAR): `task`, `n_estimators`
(v1: `1`), `seed`, `output_mode` (`compact` / `detail` / `distribution`),
`context_rows`, `softmax_temperature`, `model`. Unknown keys error.

Non-`tabfm-v1` model families each carry their own license gate — the manifest
names its license id and downloads are gated on `SET anofox_tabfm_accept_<id> =
true` (the error names the exact SET), extending the `anofox_tabfm_accept_hf_license`
pattern.

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
estimator; it ships CPU (`cpu`) plus accelerated flavors (`cuda`, `rocm` via a
direct MIGraphX backend, and `coreml` for Apple Silicon — see below).

On top of prediction, the extension now includes a full **evaluation-and-comparison
platform** (see above): model-agnostic classification/regression metrics,
leakage-safe k-fold cross-validation, regression predictive-distribution output,
distribution-gated proper scoring rules (CRPS / log-score / interval score), and
cross-model comparison — plus a generalized model seam (a preprocessing-profile
registry + per-family license gate) with a weight-free **TabPFN v2** fixture
family exercising the distribution path.

Not yet wired up: the `n_estimators > 1` ensemble; the grouped /
composable-aggregate / windowed predict surfaces (`tabfm_predict_by` / `_agg` /
`_win`); and — deferred to a future milestone, both blocked upstream — a **real
TabPFN v2 ONNX inference export** (its `torch.export` trips on data-dependent
preprocessing + chunked attention; the shipped family is fixture-scoped) and
**TabICL** onboarding (ONNX export infeasible on `tabicl 2.2.0`).

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

Requires CMake ≥ 3.10 (≥ 3.19 to also build the bundled C++ unit tests) and a
C++17 toolchain. ONNX Runtime is fetched as a
prebuilt archive by default; enable the `ort-vcpkg` manifest feature to build it
from source. See [`CLAUDE.md`](CLAUDE.md) for the module map and
[`examples/`](examples/README.md) for end-to-end examples.
