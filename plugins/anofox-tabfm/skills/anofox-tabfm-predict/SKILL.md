---
name: anofox-tabfm-predict
description: >
  Zero-shot classification and regression in SQL with the anofox_tabfm DuckDB
  extension (tabular foundation models: TabPFN, Mitra, TabICL, TabDPT, LimiX and
  more). Use when predicting a label or a number for table rows from labelled
  example rows without training a model, writing `tabfm_classify` /
  `tabfm_regress` calls (train/test pair or one table with NULL targets),
  choosing features or `opts`, reading `yhat` / `yhat_score` / `proba`, scoring
  per segment, or bounding memory with `query_batch`.
version: 2026.11.01
user-invocable: false
---

# Anofox TabFM — Predict Cheat Sheet

**Extension:** anofox_tabfm v2026.11.01 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `tabfm_*` and `anofox_tabfm_*` (identical)

A tabular foundation model is an in-context learner: it reads your labelled rows as context and predicts
the rest in one forward pass. There is no fit step, no training loop and no saved model: each call sees
exactly the rows you hand it.

## Critical gotchas

1. **Weights first, and name the model.** The extension ships weight-free graphs. Each model and task needs
   `CALL tabfm_download('<task>', model := '<id>')` once (regression is a separate download). Downloading
   does not select a model: pass `model := '<id>'` in every call or `SET anofox_tabfm_default_model`.
   Picking a model and the licence gate: the `anofox-tabfm-models` skill.
2. **Same feature columns on both sides.** In the train / test form a column on one side only is an error
   naming it. Drop it, add it to the other side, or name the shared columns with `features := [...]`.
   Every column except the target is a feature by default, so **exclude leakage columns**
   (`'(SELECT * EXCLUDE (leak) FROM t)'`).
3. **The `test` relation must not contain the target column.** If it does, the call fails with a binder
   error about duplicate names in `UNION (ALL) BY NAME`. Use `'(SELECT * EXCLUDE (target) FROM t)'`.
4. **Pass row keys with `keep := [...]`, not as features.** Without `features` every column, an ID too,
   is a model input; with `features := [...]` only the listed columns come back. `keep := ['id']` returns
   the column unchanged without feeding it to the model, and it may exist in the `test` relation only.
5. **`data` and `test` are strings**: a table name or a `'(SELECT ...)'` subquery, with quotes inside the
   subquery doubled (`''north''`). Not a bare relation.
6. **Missing target = predict this row.** `NULL`, and `NaN` in a `FLOAT`/`DOUBLE` target. `±Infinity` in the
   target is an error naming the column (`CASE WHEN isfinite(y) THEN y END`). Feature cells may be anything:
   `NULL`, `NaN` and infinities are imputed.
7. **Guardrails are enforced globally**: `anofox_tabfm_max_rows` (10000 rows per call or group) and
   `anofox_tabfm_max_features` (500). The per-model `max_rows` / `max_features` in `tabfm_list_models()` are
   recommendations only; `max_classes` (usually 10) is enforced.
8. **One call per segment.** To predict each group from its own history, `UNION ALL` one call per segment.
   Never a `LATERAL` join per row: it repeats the whole context for every row.
9. **`opts` values are strings**: `MAP{'seed': '42'}`, not `MAP{'seed': 42}`. An unknown key is an error
   listing the valid ones.

## `tabfm_classify` / `tabfm_regress`

```sql
tabfm_classify(data, target [, test] [, features] [, opts] [, model] [, keep])
tabfm_regress (data, target [, test] [, features] [, opts] [, model] [, keep])
```

| argument | type | meaning |
|---|---|---|
| `data` | `VARCHAR` | table name or `'(SELECT ...)'`: the labelled context (and, without `test`, the rows to score) |
| `target` | `VARCHAR` | the column to predict |
| `test` | `VARCHAR` | optional table name or subquery: the rows to score (the train / test form) |
| `features` | `VARCHAR[]` | optional; default every column except the target. A name that matches no column is an error (matching is case-insensitive) |
| `opts` | `MAP(VARCHAR, VARCHAR)` | optional, see below |
| `model` | `VARCHAR` | the model id; default `anofox_tabfm_default_model` |
| `keep` | `VARCHAR[]` | optional; columns returned unchanged and never used as features (a row key). Any type; may exist in `test` only |

**Explicit train / test**, like `clf.fit(X_train, y_train).predict(X_test)`: only the test rows come back.

```sql
SELECT customer_id, yhat AS will_churn, yhat_score
FROM tabfm_classify('history', 'churned', test := 'prospects', model := 'mitra');

SELECT sqft, bedrooms, yhat AS predicted_price
FROM tabfm_regress('sold_homes', 'price', test := 'listings', model := 'mitra');
```

**Single relation**: the rows whose target is `NULL` are scored; every row comes back, and the context rows
carry in-context fitted values (in-sample, a sanity check only). For tabfm-v1, tabfm-v1-1, tabicl-v2 and
kumo-tabular-* they cost a second pass over the context rows; `opts := MAP{'context_values': 'false'}` skips it.

```sql
CREATE TABLE customers_masked AS
SELECT * REPLACE (CASE WHEN id % 4 = 0 THEN NULL ELSE churned END AS churned) FROM customers;

SELECT id, churned, yhat, yhat_score, is_training
FROM tabfm_classify('customers_masked', 'churned', model := 'mitra')
WHERE NOT is_training;
```

**Features, options and a subquery**:

```sql
SELECT age, plan, usage_gb, yhat, proba      -- with features, only those columns come back
FROM tabfm_classify(
    '(SELECT * FROM history WHERE signup_year >= 2024)', 'churned',
    test     := 'prospects',
    features := ['age', 'plan', 'usage_gb'],
    opts     := MAP{'seed': '7', 'softmax_temperature': '0.9'},
    model    := 'mitra');
```

## Output columns

The scored relation's feature columns (all of its columns when `features` is omitted, only the listed
ones otherwise), the `keep` columns, the target column (`NULL` on scored rows), plus:

| column | type | meaning |
|---|---|---|
| `yhat` | the target's type | predicted label (classify) or value (regress) |
| `yhat_score` | `DOUBLE` | top-class probability; `NULL` for regression |
| `is_training` | `BOOLEAN` | `true` for a context row (single-relation form); `false` for every scored row |
| `proba` | `MAP(VARCHAR, DOUBLE)` | label → probability; classification with `output_mode = 'detail'`, the default for `tabfm_classify` |

`proba` keys are the labels as `VARCHAR`
(`'true'` / `'false'` for a `BOOLEAN` target), which is what the probability metrics expect
(`anofox-tabfm-evaluate` skill). Read one probability with `proba['yes']`.

## `opts` keys (predict)

| key | values | default | meaning |
|---|---|---|---|
| `seed` | integer | `42` | RNG seed |
| `output_mode` | `compact`, `detail` | `detail` for classify | `compact` drops `proba` |
| `softmax_temperature` | positive number | `0.9` | temperature on the class probabilities |
| `query_batch` | positive integer | all at once | score the test rows N per forward pass: bounds peak memory of a large call; keep N in the hundreds |
| `task` | `auto`, `classification`, `regression` | set by the function | leave it to `tabfm_classify` / `tabfm_regress` |
| `n_estimators` | `1` | `1` | ensembles are not available yet; above 1 is an error |
| `context_rows` | integer | `''` | accepted, no effect today |
| `context_values` | `true`, `false` | `true` | single-relation form: compute the context rows' fitted values (a second pass for the models that need one); query rows are unchanged either way |
| `model` | model id | | same as `model :=` |

There is no `dist` output mode yet: `tabfm_regress` returns point predictions.

## One model per segment

```sql
SELECT 'north' AS region, * FROM tabfm_classify(
    '(SELECT * EXCLUDE (id, region) FROM customers WHERE region = ''north'')', 'churned', model := 'mitra')
UNION ALL
SELECT 'south', * FROM tabfm_classify(
    '(SELECT * EXCLUDE (id, region) FROM customers WHERE region = ''south'')', 'churned', model := 'mitra');
```

Exclude the segment column itself so it does not act as a constant feature.

## Large calls and repeated calls

- **Memory:** `opts := MAP{'query_batch': '256'}` keeps the context and scores the test rows in batches.
  Same answers as one pass, up to float rounding. `SET anofox_tabfm_max_memory = '16GB'` refuses a call
  when the process is already that large, instead of an OOM kill.
- **More rows than the guardrail:** `SET anofox_tabfm_max_rows = 20000;` (and check the model's recommended
  `max_rows`), or sample the context: `'(SELECT * FROM big USING SAMPLE 5000 ROWS (reservoir, 42))'`.
- **Scoring against the same training table again and again** is fastest with `tabicl-v2` on the CPU: its
  encoded context is cached and reused from the third identical call on (details in the models skill).
  Keep the context's row order stable (`ORDER BY` a key) to hit the cache.

```sql
SELECT customer_id, yhat
FROM tabfm_classify('(SELECT * FROM history ORDER BY customer_id)', 'churned',
                    test := 'prospects', model := 'tabicl-v2',
                    opts := MAP{'query_batch': '128'});
```

## Canonical pipeline

```sql
-- 0. once per machine (needs httpfs): CALL tabfm_download('classification', model := 'mitra');
SET anofox_tabfm_default_model = 'mitra';

-- 1. a deterministic split
CREATE TABLE ctx  AS SELECT * FROM customers WHERE hash(id) % 10 < 7;
CREATE TABLE hold AS SELECT * FROM customers WHERE hash(id) % 10 >= 7;

-- 2. predict the held-out rows; the test side drops the target, and id stays in
--    both relations so the result can be joined back
CREATE TABLE pred AS
SELECT h.id, h.churned AS actual, p.yhat, p.yhat_score, p.proba
FROM tabfm_classify('ctx', 'churned', test := '(SELECT * EXCLUDE (churned) FROM hold)') p
JOIN hold h USING (id);

-- 3. measure (anofox-tabfm-evaluate)
SELECT tabfm_accuracy(actual, yhat) AS acc,
       tabfm_roc_auc(actual::VARCHAR, proba, avg := 'ovr') AS auc
FROM pred;
```

## Errors and fixes

| You see | Do this |
|---|---|
| `N models are registered (...) and none is selected` | add `model := '<id>'` or `SET anofox_tabfm_default_model = '<id>';` |
| `has no downloaded weights for task '<task>'` | `CALL tabfm_download('<task>', model := '<id>');` |
| ``the context relation and the `test` relation must expose the same feature columns`` | drop or add the column, or `features := [...]` |
| `features := [...] names '<x>', which matched no column` | fix the name |
| `keep := [...] names '<x>', which matched no column` | fix the name |
| `'<x>' is in both features := [...] and keep := [...]` | name it in only one: `keep` to pass it through, `features` to model it |
| `input exceeds anofox_tabfm_max_rows` | fewer rows, or `SET anofox_tabfm_max_rows = <n>;` |
| `'<model>' supports at most N classes` | fewer classes, or decompose (one-vs-rest) |
| `UNION (ALL) BY NAME ... the name "<target>" occurs multiple times` | the `test` relation still has the target column: `EXCLUDE` it |
| `task 'regression' requires a numeric target` | cast the target, or use `tabfm_classify` |
| `model '<id>' does not support task '<task>'` | e.g. `orion-bix` is classify-only, `nori-30m` regress-only |
| `unknown option '<key>'` | the message lists the valid keys |
| `... is refused rather than risking an OOM kill` | smaller calls, `query_batch`, or `CALL tabfm_unload();` |
