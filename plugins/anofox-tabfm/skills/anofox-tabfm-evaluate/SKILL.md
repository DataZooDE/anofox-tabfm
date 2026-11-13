---
name: anofox-tabfm-evaluate
description: >
  Evaluating and comparing predictions with the anofox_tabfm DuckDB extension:
  classification metrics (accuracy, precision, recall, F1, ROC-AUC, log loss,
  ECE, confusion matrix), regression metrics (RMSE, MAE, R², MAPE, MedAE),
  proper scoring rules for predictive distributions (CRPS, log score, interval
  score), leakage-safe k-fold cross-validation, and the `_by` macros that feed
  anofox-visualization plots. Use when measuring how good a prediction is,
  comparing models with `tabfm_cross_validate` or `tabfm_fold_assign`, writing
  `tabfm_f1` / `tabfm_roc_auc` / `tabfm_rmse` aggregates, or plotting ROC,
  calibration, confusion-matrix or model-comparison charts with
  `tabfm_classification_metrics_by` and friends.
version: 2026.11.01
user-invocable: false
---

# Anofox TabFM — Evaluation Cheat Sheet

**Extension:** anofox_tabfm v2026.11.01 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `tabfm_*` and `anofox_tabfm_*` (identical)

The metrics are model-agnostic SQL aggregates over plain `(actual, predicted[, proba])` columns: they score
any model's output, not only this extension's.

## Critical gotchas

1. **`avg` is required** for `tabfm_precision`, `tabfm_recall`, `tabfm_f1` (`'micro'`, `'macro'`,
   `'weighted'`) and `tabfm_roc_auc` (`'ovr'`, `'ovo'`). Leaving it out is an error, not a default.
2. **Probability metrics take `VARCHAR` labels** and a `MAP(VARCHAR, DOUBLE)`: `tabfm_log_loss`,
   `tabfm_roc_auc`, `tabfm_ece`. That is exactly `tabfm_classify`'s `proba`; cast the label with `::VARCHAR`
   (a `BOOLEAN` target's keys are `'true'` / `'false'`).
3. **`tabfm_cross_validate` takes no `model :=`.** Set `anofox_tabfm_default_model` first. `task` is
   `'classification'` (default) or `'regression'`; anything else (`'auto'`, a typo) is an error. A numeric
   target that is a quantity needs `task := 'regression'`, or the folds are classified and the call fails
   on the class limit with an error that says so.
4. **Scoring rules need a distribution column** `STRUCT(logits DOUBLE[], borders DOUBLE[])`. `tabfm_regress`
   does not emit one yet, so they score distributions you construct or get from elsewhere, and they are not
   usable in cross-validation.
5. **`NULL` rows are skipped**; `NaN` operands of the regression metrics are skipped too, `Infinity` is an
   error. The `VARCHAR`-label metrics cannot tell a numeric `NaN` from the string `'nan'`.
6. **Never evaluate on the context rows.** Score a held-out split (train / test form) or use
   cross-validation; the single-relation form's fitted values on context rows are only a sanity check.

## Classification metrics (aggregates)

```sql
SELECT tabfm_accuracy(actual, predicted)                   AS acc,
       tabfm_precision(actual, predicted, avg := 'macro')  AS prec,
       tabfm_recall(actual, predicted, avg := 'macro')     AS rec,
       tabfm_f1(actual, predicted, avg := 'macro')         AS f1,
       tabfm_roc_auc(actual, proba, avg := 'ovr')          AS auc,      -- rank-sum, tie-safe
       tabfm_log_loss(actual, proba)                       AS logloss,  -- epsilon-clipped
       tabfm_ece(actual, proba)                            AS ece       -- top-label calibration error
FROM scored;
```

`tabfm_accuracy`, `tabfm_precision`, `tabfm_recall` and `tabfm_f1` take labels of any type (`actual` and
`predicted` the same type). Per model: add `GROUP BY model`.

Confusion matrix, tidy `(actual, predicted, count)`:

```sql
SELECT * FROM tabfm_confusion_matrix('scored', 'actual', 'predicted');
```

## Regression metrics (aggregates over `DOUBLE`)

```sql
SELECT tabfm_rmse(actual, predicted)  AS rmse,
       tabfm_mae(actual, predicted)   AS mae,
       tabfm_r2(actual, predicted)    AS r2,     -- constant-target safe
       tabfm_mape(actual, predicted)  AS mape,   -- skips zero-actual rows
       tabfm_medae(actual, predicted) AS medae
FROM scored_regression;
```

## Scoring rules for a predictive distribution

`dist` is `STRUCT(logits DOUBLE[], borders DOUBLE[])`: a softmax'd bar distribution, `len(borders) =
len(logits) + 1`, non-uniform bin widths honoured. A point estimate instead of a distribution is a bind error
naming the remedy.

```sql
SELECT tabfm_crps(actual, dist)                AS crps,   -- closed form
       tabfm_log_score(actual, dist)           AS nll,
       tabfm_interval_score(actual, dist, 0.9) AS is90    -- 3rd argument: coverage, default 0.9
FROM predictions;
```

## Cross-validation

Deterministic, seedable folds (`hash(row_key, seed) % k`). Each fold is predicted from the others through the
train / test form, so no row ever sees its own label.

```sql
tabfm_cross_validate(data, target, row_key [, k := 5] [, seed := 42] [, task := 'classification'] [, metric])
tabfm_fold_assign(data, k, row_key, seed)
```

| argument | meaning |
|---|---|
| `data` | table name or `'(SELECT ...)'` |
| `target` | column to predict |
| `row_key` | a unique key column; it decides the folds and joins the labels back. Passed through with `keep`, so it is not a feature |
| `k`, `seed` | folds and the hash seed |
| `task` | `'classification'` (default) or `'regression'`; anything else is an error |
| `metric` | `'tabfm_accuracy'` (classification default), `'tabfm_rmse'` (regression default), `'tabfm_mae'`, `'tabfm_r2'` |

Output `fold_id, metric_value, metric_std`: one row per fold (`metric_std` `NULL`) and a summary row with
`fold_id = -1` holding the mean and the standard deviation.

```sql
SET anofox_tabfm_default_model = 'mitra';
SELECT * FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5, seed := 42);
SELECT * FROM tabfm_cross_validate('sold_homes', 'price', 'home_id', k := 3,
                                   task := 'regression', metric := 'tabfm_mae');
```

Comparing models by CV means one run per model, switching the default:

```sql
SET anofox_tabfm_default_model = 'mitra';
CREATE TABLE cv AS
SELECT 'mitra' AS model, metric_value, metric_std
FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5) WHERE fold_id = -1;
SET anofox_tabfm_default_model = 'tabicl-v2';
INSERT INTO cv
SELECT 'tabicl-v2', metric_value, metric_std
FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5) WHERE fold_id = -1;
SELECT * FROM cv ORDER BY metric_value DESC;
```

`tabfm_fold_assign` returns every input column plus `fold_id` (and a helper column `_cv_rk`), for your own
fold loops, e.g. metrics that `tabfm_cross_validate` does not offer. Drop both helper columns and pass the
key through `keep`, or they become features (the test fold's `fold_id` is a value no context row has):

```sql
CREATE TABLE fa AS SELECT * FROM tabfm_fold_assign('customers', 5, 'id', 42);

-- one fold (fold 0); repeat per fold, or generate the UNION ALL
SELECT 0 AS fold, tabfm_f1(c.churned::VARCHAR, p.yhat::VARCHAR, avg := 'macro') AS f1
FROM tabfm_classify('(SELECT * EXCLUDE (fold_id, _cv_rk) FROM fa WHERE fold_id <> 0)', 'churned',
                    test := '(SELECT * EXCLUDE (fold_id, _cv_rk) FROM fa WHERE fold_id = 0)',
                    keep := ['id'], model := 'mitra') p
JOIN customers c USING (id);
```

Cost: CV is k predict calls per model, each over (k-1)/k of the rows; use `k := 3` on large tables. Comparing
more models than `anofox_tabfm_max_sessions` (default 4) reloads sessions between runs.

## Comparing models on a held-out split

Stack each model's predictions with a `model` column; every metric then works `GROUP BY model`.

```sql
-- the held-out labels, wherever they live
CREATE TABLE truth AS SELECT id AS customer_id, churned FROM customers WHERE id > 30;

CREATE TABLE cmp AS
SELECT 'mitra' AS model, t.churned::VARCHAR AS actual, p.yhat::VARCHAR AS yhat, p.proba
FROM tabfm_classify('history', 'churned', test := 'prospects', keep := ['customer_id'], model := 'mitra') p
JOIN truth t USING (customer_id)
UNION ALL
SELECT 'tabicl-v2', t.churned::VARCHAR, p.yhat::VARCHAR, p.proba
FROM tabfm_classify('history', 'churned', test := 'prospects', keep := ['customer_id'], model := 'tabicl-v2') p
JOIN truth t USING (customer_id);

SELECT model, tabfm_accuracy(actual, yhat) AS acc, tabfm_roc_auc(actual, proba, avg := 'ovr') AS auc
FROM cmp GROUP BY model ORDER BY auc DESC;
```

## Plotting with anofox-visualization (the `_by` macros)

Six table macros turn scored rows into the long tables `anofox_plot(t)` from the
[anofox-visualization](https://github.com/DataZooDE/anofox-visualization) extension draws, one group per
model. Arguments after `data` are column **names** as strings.

| macro | gives | draws |
|---|---|---|
| `tabfm_classification_metrics_by(data, group_col, actual_col, predicted_col, proba_col)` | accuracy, ece, f1_macro, log_loss, roc_auc_ovr per group | model comparison |
| `tabfm_regression_metrics_by(data, group_col, actual_col, predicted_col)` | mae, mape, medae, r2, rmse per group | model comparison |
| `tabfm_confusion_matrix_by(data, group_col, actual_col, predicted_col)` | row (actual), col (predicted), value (count), zero cells included | heatmap per model |
| `tabfm_augment_by(data, group_col, actual_col, predicted_col [, row_id_col])` | y, fitted, residual per row | residual diagnostics |
| `tabfm_roc_curve_by(data, group_col, actual_col, proba_col [, pos_label, drop_intermediate])` | ROC curve, one-vs-rest per class | ROC curves with AUC |
| `tabfm_calibration_curve_by(data, group_col, actual_col, proba_col [, pos_label, n_bins])` | reliability curve per class, 95% Wilson band | calibration curves |

Metric tables: `model, model_id, metric, value, conf_low, conf_high, row_order`. Curves: `model, model_id,
series, curve_type, x, y, y_low, y_high, threshold, weight, n_censor, row_order`. Every macro ends with
`row_order`, the 1-based order the rows come out in.

```sql
SELECT * FROM tabfm_classification_metrics_by('scored', 'model', 'actual', 'yhat', 'proba');
SELECT * FROM tabfm_regression_metrics_by('scored_regression', 'model', 'actual', 'predicted');
SELECT * FROM tabfm_roc_curve_by('scored', 'model', 'actual', 'proba', pos_label := 'yes');
```

With the visualization loaded:

```sql
INSTALL anofox_visualization FROM community; LOAD anofox_visualization;
SELECT anofox_plot(t) FROM tabfm_classification_metrics_by('scored', 'model', 'actual', 'yhat', 'proba') t;
SELECT model_id, anofox_plot(t) FROM tabfm_confusion_matrix_by('scored', 'model', 'actual', 'yhat') t
GROUP BY model_id;                                   -- one heatmap per model
SELECT anofox_plot(t) FROM tabfm_roc_curve_by('scored', 'model', 'actual', 'proba') t;
```

Curves and classes: two classes give one curve per model (for the greater label, scikit-learn's
`classes_[1]`), so models compare in one plot. More classes give one curve per class (`series`); plot those
one model at a time (`GROUP BY model_id`). `pos_label := '<class>'` draws one class. The calibration curve
bins each class's probability while `tabfm_ece` bins the top-label confidence, so they do not give the same
number. A metric undefined for a group (ROC-AUC with one class present) is left out.

## Canonical evaluation pipeline

```sql
SET anofox_tabfm_default_model = 'mitra';

-- 1. how good is it, without a held-out split: 5-fold CV
SELECT metric_value AS accuracy, metric_std
FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5) WHERE fold_id = -1;

-- 2. a held-out split for richer metrics
CREATE TABLE ctx  AS SELECT * FROM customers WHERE hash(id) % 10 < 7;
CREATE TABLE hold AS SELECT * FROM customers WHERE hash(id) % 10 >= 7;
CREATE TABLE ev AS
SELECT 'mitra' AS model, h.churned::VARCHAR AS actual, p.yhat::VARCHAR AS yhat, p.proba
FROM tabfm_classify('ctx', 'churned', test := 'hold', keep := ['id']) p
JOIN hold h USING (id);

-- 3. the numbers, and the long table to plot
SELECT tabfm_f1(actual, yhat, avg := 'macro') AS f1, tabfm_log_loss(actual, proba) AS logloss,
       tabfm_ece(actual, proba) AS ece
FROM ev;
SELECT * FROM tabfm_classification_metrics_by('ev', 'model', 'actual', 'yhat', 'proba');
```
