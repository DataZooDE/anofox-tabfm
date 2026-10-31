# Evaluate and compare

Score a held-out split, measure it in SQL, cross-validate, compare two models and plot the result. Every
step uses `mitra`; download its classification weights once, as in
[Getting Started](01-getting-started.md#2-pick-a-model-and-download-its-weights-once).

## A full worked example

Zero-shot churn prediction on a public dataset, split into train/test, scored,
and evaluated — **entirely in SQL**:

```sql
INSTALL httpfs; LOAD httpfs; LOAD anofox_tabfm;
CALL tabfm_download('classification', model := 'mitra');   -- once; cached afterwards

-- 1. load, clean the types, and a deterministic 70/30 split
CREATE TABLE churn AS
SELECT * REPLACE (TRY_CAST(TotalCharges AS DOUBLE) AS TotalCharges, Churn::BOOLEAN AS Churn),
       hash(customerID) % 100 AS bucket
FROM 'hf://datasets/scikit-learn/churn-prediction/**/*.csv';

CREATE TABLE train AS SELECT * EXCLUDE (bucket) FROM (FROM churn WHERE bucket < 70) USING SAMPLE 500 ROWS (reservoir, 42);
CREATE TABLE test  AS SELECT * EXCLUDE (bucket) FROM (FROM churn WHERE bucket >= 70) USING SAMPLE 150 ROWS (reservoir, 42);
CREATE TABLE test_features AS SELECT * EXCLUDE (Churn) FROM test;   -- no target for the test rows

-- 2. predict the test rows using the train rows as context
CREATE TABLE preds AS
SELECT customerID, yhat AS pred, proba
FROM tabfm_classify('train', 'Churn', test := 'test_features', model := 'mitra');

-- 3. F1 of the positive class, in SQL
WITH cm AS (
  SELECT count(*) FILTER (WHERE p.pred AND t.Churn)         AS tp,
         count(*) FILTER (WHERE p.pred AND NOT t.Churn)     AS fp,
         count(*) FILTER (WHERE NOT p.pred AND t.Churn)     AS fn
  FROM preds p JOIN test t USING (customerID))
SELECT 2.0*tp / nullif(2.0*tp + fp + fn, 0) AS f1 FROM cm;
```

The same numbers come from the metric aggregates, which work on any model's output
([Evaluation](../api/07-evaluation.md)):

```sql
SELECT tabfm_accuracy(t.Churn, p.pred)                 AS accuracy,
       tabfm_f1(t.Churn, p.pred, 'macro')              AS f1_macro,
       tabfm_roc_auc(t.Churn::VARCHAR, p.proba, 'ovr') AS roc_auc   -- probability metrics take VARCHAR labels
FROM preds p JOIN test t USING (customerID);
```

## Cross-validation

**Leakage-safe k-fold cross-validation** — deterministic, seedable folds
(`hash(row_key, seed) % k`); trains each fold on the others and predicts its
held-out rows via the two-table form; returns per-fold plus aggregate
(mean ± std) results. `tabfm_cross_validate` takes no `model :=`, so set the
default first:

```sql
SET anofox_tabfm_default_model = 'mitra';
-- positional: data, target, row_key; then optional named k / seed / task / metric
SELECT * FROM tabfm_cross_validate('(SELECT * EXCLUDE (bucket) FROM churn USING SAMPLE 600 ROWS (reservoir, 42))',
                                   'Churn', 'customerID', k := 5, seed := 42);
-- tabfm_fold_assign('churn', 5, 'customerID', 42) exposes the fold ids directly (data, k, row_key, seed)
```

## Compare models and plot with anofox-visualization

Stack the predictions of several models in one table with a `model` column, then hand it to the `_by`
macros ([Plotting](../api/08-plotting.md)). A second model needs its own download; `tabicl-v2` is ungated:

```sql
CALL tabfm_download('classification', model := 'tabicl-v2');
INSTALL anofox_visualization FROM community; LOAD anofox_visualization;

CREATE TABLE scored AS
SELECT 'mitra' AS model, t.Churn::VARCHAR AS actual, p.yhat::VARCHAR AS yhat, p.proba
FROM tabfm_classify('train', 'Churn', test := 'test_features', model := 'mitra') p
JOIN test t USING (customerID)
UNION ALL
SELECT 'tabicl-v2', t.Churn::VARCHAR, p.yhat::VARCHAR, p.proba
FROM tabfm_classify('train', 'Churn', test := 'test_features', model := 'tabicl-v2') p
JOIN test t USING (customerID);

SELECT anofox_plot(t) FROM tabfm_classification_metrics_by('scored', 'model', 'actual', 'yhat', 'proba') t;
-- a confusion matrix is one plot per model
SELECT model_id, anofox_plot(t) FROM tabfm_confusion_matrix_by('scored', 'model', 'actual', 'yhat') t
GROUP BY model_id;
-- two classes: one curve per model, all models in one plot
SELECT anofox_plot(t) FROM tabfm_roc_curve_by('scored', 'model', 'actual', 'proba') t;
```

See [`examples/`](../../examples/README.md) for runnable scripts, and
[Choosing a model](02-choosing-a-model.md#how-they-compare) for how the models compared on public datasets.
