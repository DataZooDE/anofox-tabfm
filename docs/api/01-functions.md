# Functions

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

**Single relation** — rows whose target is missing are the ones to score; every
row comes back with an `is_training` flag (context rows get in-context fitted
values, handy for a sanity check).

```sql
SELECT * FROM tabfm_classify('customers', 'churned');
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
    test := 'prospects');
```

**One model per segment.** The model learns in context, so each call sees
exactly the labelled rows you hand it. To predict each segment from its own
history, make one call per segment with a subquery and stack the results; the
segment column is excluded so it does not act as a feature:

```sql
SELECT 'north' AS region, * FROM tabfm_classify(
    '(SELECT * EXCLUDE (region) FROM customers WHERE region = ''north'')', 'churned')
UNION ALL
SELECT 'south', * FROM tabfm_classify(
    '(SELECT * EXCLUDE (region) FROM customers WHERE region = ''south'')', 'churned');
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
    opts     := MAP{'seed': '42', 'output_mode': 'detail'});
```

## Output columns

Every column of the scored rows, plus:

| column | meaning |
|---|---|
| `yhat` | predicted label (classification) or value (regression) |
| `yhat_score` | top-class probability; `NULL` for regression |
| `proba` | `MAP(label → probability)` — classification, `output_mode = 'detail'` (the default) |
| `is_training` | present in single-relation mode: was this a context row? |
