# Evaluation, cross-validation and scoring

Model-agnostic evaluation primitives — they operate on plain `(actual,
predicted[, proba])` columns, so they score **any** model's output, not just
this extension's. Each has a full `anofox_tabfm_*` name and a short `tabfm_*`
alias. A worked example on real data: [Evaluate and compare](../guides/03-evaluate-and-compare.md).

## Classification metrics

**Classification metrics** (SQL aggregates):

```sql
SELECT tabfm_accuracy(actual, predicted)           AS acc,
       tabfm_f1(actual, predicted, avg := 'macro') AS f1,       -- avg is REQUIRED
       tabfm_roc_auc(actual, proba, avg := 'ovr')  AS auc,      -- rank-sum, tie-safe
       tabfm_log_loss(actual, proba)               AS logloss,  -- ε-clipped
       tabfm_ece(actual, proba)                    AS ece
FROM scored;
-- tabfm_precision / tabfm_recall take the same required avg ('micro'|'macro'|'weighted'; roc_auc: 'ovr'|'ovo')
-- confusion matrix is a table macro → tidy (actual, predicted, count):
SELECT * FROM tabfm_confusion_matrix('scored', 'actual', 'predicted');
```

`tabfm_accuracy`, `tabfm_precision`, `tabfm_recall` and `tabfm_f1` take labels of any type.
`tabfm_log_loss`, `tabfm_roc_auc` and `tabfm_ece` take the label as `VARCHAR` and the probabilities as
`MAP(VARCHAR, DOUBLE)`, which is the `proba` column of `tabfm_classify`; cast another label type with
`::VARCHAR`. `NULL` rows are skipped.

## Regression metrics

**Regression metrics** (SQL aggregates over `(actual, predicted)`):

```sql
SELECT tabfm_rmse(actual, predicted), tabfm_mae(actual, predicted),
       tabfm_r2(actual, predicted),    -- constant-target safe
       tabfm_mape(actual, predicted),  -- skips zero-actual rows
       tabfm_medae(actual, predicted)
FROM scored;
```

## Cross-validation

**Leakage-safe k-fold cross-validation** — deterministic, seedable folds
(`hash(row_key, seed) % k`); trains each fold on the others and predicts its
held-out rows via the two-table form; returns per-fold plus aggregate
(mean ± std) results. It takes no `model :=`: it uses `anofox_tabfm_default_model`, so set that first.
`metric` is one of `tabfm_accuracy` (the classification default), `tabfm_rmse` (the regression default),
`tabfm_mae` or `tabfm_r2`. The distribution scoring rules (`tabfm_crps`, `tabfm_log_score`,
`tabfm_interval_score`) are not usable in cross-validation yet: the folds produce point predictions.

```sql
SET anofox_tabfm_default_model = 'mitra';
-- positional: data, target, row_key; then optional named k / seed / task / metric
SELECT * FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5, seed := 42);
-- tabfm_fold_assign('customers', 5, 'id', 42) exposes the fold ids directly (data, k, row_key, seed)
```

## Scoring rules for a predictive distribution

**Proper scoring rules** for a regression predictive distribution shaped
`STRUCT(logits DOUBLE[], borders DOUBLE[])` (softmax'd bar distribution; the
model-provided, possibly non-uniform bin widths are honoured). They are
bind-gated on a distribution input and error with a named remedy if handed a
point estimate:

```sql
SELECT tabfm_crps(actual, dist)                AS crps,   -- closed-form CRPS
       tabfm_log_score(actual, dist)           AS nll,
       tabfm_interval_score(actual, dist, 0.9) AS is90    -- 3rd arg = coverage, default 0.9
FROM predictions;   -- `dist` is a STRUCT(logits[], borders[]) column
```

> Note: these score a distribution column you supply — construct the
> `STRUCT(logits[], borders[])` yourself (as the tests do) or from a model that
> emits one. `tabfm_regress` does not produce a distribution column yet: its
> `output_mode` accepts only `compact` and `detail`.
