# Functions

Every function is available as `anofox_tabfm_<name>` and as the short alias `tabfm_<name>`.
`duckdb_functions()` carries a description and a runnable example for each.

- [Classification and regression](#classification-and-regression)
- [Output columns](#output-columns)
- [Every function](#every-function)

The examples name `model := 'mitra'` and assume its weights are downloaded, for both tasks:

```sql
CALL tabfm_download('classification', model := 'mitra');
CALL tabfm_download('regression', model := 'mitra');
```

## Classification and regression

Both functions have the same signature; the task (classification vs
regression) is fixed by which one you call.

```sql
tabfm_classify(data, target [, test] [, features] [, opts] [, model])
tabfm_regress (data, target [, test] [, features] [, opts] [, model])
```

`data` and `test` are table names or `'(SELECT ...)'` strings; `target` is a column name. `model` names a
model from the registry; without it the call uses `anofox_tabfm_default_model`, and fails naming both
remedies when neither is set ([Choosing a model](../guides/02-choosing-a-model.md#selecting-a-model)).

**Explicit train / test** — like `clf.fit(X_train, y_train).predict(X_test)`.
Only the scored rows are returned.

```sql
SELECT * FROM tabfm_classify('history', 'churned', test := 'prospects', model := 'mitra');
SELECT * FROM tabfm_regress('sold_homes', 'price', test := 'listings', model := 'mitra');
```

**Single relation** — rows whose target is missing are the ones to score; every
row comes back with an `is_training` flag (context rows get in-context fitted
values, handy for a sanity check).

```sql
SELECT * FROM tabfm_classify('customers', 'churned', model := 'mitra');
```

**What counts as a missing target.** `NULL`, and `NaN` in a `FLOAT`/`DOUBLE`
column (pandas exports missing numbers as `NaN`): both mean *predict this
row*. `+Infinity`, `-Infinity` and a `DATE`/`TIMESTAMP` infinity are an **error**
that names the column, so a stray sentinel cannot become a label or skew a mean;
`CASE WHEN isfinite(y) THEN y END` turns them into `NULL` and keeps the `NaN`
rows. The string `'nan'` is an ordinary class label. A value beyond the float32
range (about 3.4e38) is also an error for a regression target. In the two-table
form a row of the context relation with a missing target is neither context nor
output: only the `test` rows come back. The same rule applies to `tabfm_impute`
(a `NaN` cell in a column being filled is filled), the regression metrics and
scoring functions (a `NaN` operand is skipped, like `NULL`; `Infinity` is an
error) and the typed classification metrics. Feature columns are never an error:
`NaN`, `Infinity`, a value beyond float32 and a date infinity are all imputed like
`NULL`. One limit: the metrics that bind labels as `VARCHAR` (log-loss, ROC-AUC,
ECE) cannot tell a numeric `NaN` from the string `'nan'`.

**A subquery** works anywhere where a table name does:

```sql
SELECT * FROM tabfm_classify(
    '(SELECT * FROM history WHERE signup_year = 2025)', 'churned',
    test := 'prospects', model := 'mitra');
```

**One model per segment.** The model learns in context, so each call sees
exactly the labelled rows you hand it. To predict each segment from its own
history, make one call per segment with a subquery and stack the results; the
segment column is excluded so it does not act as a feature:

```sql
SELECT 'north' AS region, * FROM tabfm_classify(
    '(SELECT * EXCLUDE (region) FROM customers WHERE region = ''north'')', 'churned', model := 'mitra')
UNION ALL
SELECT 'south', * FROM tabfm_classify(
    '(SELECT * EXCLUDE (region) FROM customers WHERE region = ''south'')', 'churned', model := 'mitra');
```

This is the supported way to predict per group. Each call is one forward pass over
that segment's context and queries together, which is why a row-at-a-time form such
as a `LATERAL` join would be slower: it would repeat the whole context for every
row. (The grouped and windowed aggregates are internal; `tabfm_classify` and
`tabfm_regress` are the public surface.)

**Feature selection and options** (named parameters lead to the best readability):

```sql
SELECT * FROM tabfm_classify(
    'history', 'churned',
    test     := 'prospects',
    features := ['age', 'plan', 'usage_gb'],   -- default: all other columns
    opts     := MAP{'seed': '42', 'output_mode': 'detail'},
    model    := 'mitra');
```

A name in `features` that matches no column is an error, not a quietly smaller model. In the train / test
form both relations must expose the same feature columns: a column on one side only is an error naming it,
because it would otherwise be filled with `NULL` on the other side and degrade the prediction silently.
The `opts` keys (`seed`, `softmax_temperature`, `output_mode`, `context_rows`, ...) are listed in
[Settings](05-settings.md#options).

## Output columns

Every column of the scored rows, plus:

| column | meaning |
|---|---|
| `yhat` | predicted label (classification) or value (regression) |
| `yhat_score` | top-class probability; `NULL` for regression |
| `proba` | `MAP(label → probability)` — classification, `output_mode = 'detail'` (the default for `tabfm_classify`) |
| `is_training` | present in single-relation mode: was this a context row? |

## Every function

| Function | Kind | Purpose | Reference |
|---|---|---|---|
| `tabfm_classify`, `tabfm_regress` | table macro | predict | this page |
| `tabfm_generate`, `tabfm_impute` | table macro | sample rows, fill NULLs | [Generation](03-generation.md) |
| `tabfm_list_models()` | table | the registry: every model, downloaded or not | [Models and weights](02-models-and-weights.md) |
| `tabfm_models()` | table | cached and loaded weights, and the device serving each | [Models and weights](02-models-and-weights.md#managing-weights) |
| `tabfm_download`, `tabfm_load`, `tabfm_unload`, `tabfm_remove` | table (`CALL`) | the weight lifecycle | [Models and weights](02-models-and-weights.md#managing-weights) |
| `tabfm_register_model`, `tabfm_unregister_model` | table (`CALL`) | your own model, in SQL | [Models and weights](02-models-and-weights.md#registering-your-own-model) |
| `tabfm_devices()`, `tabfm_backends()` | table | devices found; which device can serve which model | [GPU](04-gpu.md) |
| `tabfm_accelerate`, `tabfm_download_runtime`, `tabfm_gpu_precompile` | table (`CALL`) | GPU setup and the ROCm compile | [GPU](04-gpu.md) |
| `tabfm_accuracy`, `tabfm_precision`, `tabfm_recall`, `tabfm_f1`, `tabfm_roc_auc`, `tabfm_log_loss`, `tabfm_ece` | aggregate | classification metrics | [Evaluation](07-evaluation.md) |
| `tabfm_rmse`, `tabfm_mae`, `tabfm_r2`, `tabfm_mape`, `tabfm_medae` | aggregate | regression metrics | [Evaluation](07-evaluation.md) |
| `tabfm_crps`, `tabfm_log_score`, `tabfm_interval_score` | aggregate | scoring rules for a predictive distribution | [Evaluation](07-evaluation.md#scoring-rules-for-a-predictive-distribution) |
| `tabfm_confusion_matrix`, `tabfm_cross_validate`, `tabfm_fold_assign` | table macro | confusion matrix, k-fold CV | [Evaluation](07-evaluation.md) |
| `tabfm_*_by` (six macros) | table macro | long tables for anofox-visualization | [Plotting](08-plotting.md) |
