# Evaluate and compare

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

## Cross-validation

**Leakage-safe k-fold cross-validation** — deterministic, seedable folds
(`hash(row_key, seed) % k`); trains each fold on the others and predicts its
held-out rows via the two-table form; returns per-fold plus aggregate
(mean ± std) results:

```sql
-- positional: data, target, row_key; then optional named k / seed / task / metric
SELECT * FROM tabfm_cross_validate('customers', 'churned', 'id', k := 5, seed := 42);
-- tabfm_fold_assign('customers', 5, 'id', 42) exposes the fold ids directly (data, k, row_key, seed)
```

## Plot with anofox-visualization

```sql
INSTALL anofox_visualization FROM community; LOAD anofox_visualization;

CREATE TABLE scored AS
SELECT 'mitra' AS model, t.churned AS actual, p.yhat, p.proba
FROM tabfm_classify('history', 'churned', test := 'holdout', model := 'mitra') p
JOIN holdout_truth t USING (customer_id);

SELECT anofox_plot(t) FROM tabfm_classification_metrics_by('scored', 'model', 'actual', 'yhat', 'proba') t;
-- a confusion matrix is one plot per model
SELECT model_id, anofox_plot(t) FROM tabfm_confusion_matrix_by('scored', 'model', 'actual', 'yhat') t
GROUP BY model_id;
-- two classes: one curve per model, all models in one plot
SELECT anofox_plot(t) FROM tabfm_roc_curve_by('scored', 'model', 'actual', 'proba') t;
```

See [`examples/`](../../examples/README.md) for runnable scripts.
