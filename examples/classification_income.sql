-- Real-model verification, pure SQL: zero-shot ADULT-INCOME classification.
-- Dataset: scikit-learn/adult-census-income (HF), binary target `income`
-- ('>50K' vs '<=50K'). Also demonstrates the multi-model surface:
-- tabfm_list_models() (the registry) and an explicit model := 'tabfm-v1'.
--
-- Run:  duckdb :memory: < examples/classification_income.sql
-- Needs: real classification weights (CALL tabfm_download('classification'))
--        (built-in model 'tabfm-v1' — no manifest file needed).

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;
SET anofox_tabfm_max_rows = 20000;

-- What models can I run? (the registry — downloaded or not)
.print '================ registry (tabfm_list_models) ================'
SELECT model, family, capabilities, commercial, downloaded FROM tabfm_list_models() ORDER BY model;

-- 1. Load + a deterministic 70/30 split on a hashed row id. The dotted column
--    names in this dataset must be quoted.
CREATE TABLE income AS
SELECT
    row_number() OVER () AS row_id,
    age::INT AS age, workclass, education, "education.num"::INT AS education_num,
    "marital.status" AS marital_status, occupation, relationship, race, sex,
    "hours.per.week"::INT AS hours_per_week, "native.country" AS native_country,
    income,
    hash(row_number() OVER ()) % 100 AS bucket
FROM 'hf://datasets/scikit-learn/adult-census-income/**/*.csv';

-- 2. ~600 in-context (train) rows, ~200 scored (test) rows.
CREATE TABLE train AS
    SELECT * EXCLUDE (bucket) FROM (FROM income WHERE bucket < 70) USING SAMPLE 600 ROWS (reservoir, 42);
CREATE TABLE test_full AS
    SELECT * EXCLUDE (bucket) FROM (FROM income WHERE bucket >= 70) USING SAMPLE 200 ROWS (reservoir, 42);
CREATE TABLE test_features AS SELECT * EXCLUDE (income) FROM test_full;
CREATE TABLE test_actuals  AS SELECT row_id, income FROM test_full;

-- 3. Zero-shot predict with the model chosen explicitly (model := 'tabfm-v1').
.timer on
CREATE TABLE preds AS
SELECT row_id, yhat AS pred
FROM tabfm_classify('train', 'income', test := 'test_features', model := 'tabfm-v1');

-- 4. F1 of the positive class ('>50K'), plus accuracy, in SQL.
CREATE TABLE cm AS
SELECT
    count(*) FILTER (WHERE p.pred = '>50K' AND a.income = '>50K')  AS tp,
    count(*) FILTER (WHERE p.pred = '>50K' AND a.income <> '>50K') AS fp,
    count(*) FILTER (WHERE p.pred <> '>50K' AND a.income = '>50K') AS fn,
    count(*) FILTER (WHERE p.pred <> '>50K' AND a.income <> '>50K') AS tn
FROM preds p JOIN test_actuals a USING (row_id);

.print '================ ADULT-INCOME classification (TabFM zero-shot) ================'
SELECT
    tp, fp, fn, tn,
    (tp + tn)::DOUBLE / (tp + fp + fn + tn)              AS accuracy,
    tp::DOUBLE / nullif(tp + fp, 0)                      AS precision,
    tp::DOUBLE / nullif(tp + fn, 0)                      AS recall,
    2.0 * tp / nullif(2.0 * tp + fp + fn, 0)             AS f1
FROM cm;
