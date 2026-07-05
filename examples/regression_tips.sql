-- Real-model verification, pure SQL: zero-shot TIP regression.
-- Dataset: scikit-learn/tips (HF), numeric target `tip`.
-- The train split is the in-context set, the test split is scored. We compute
-- the mean squared error (and RMSE / MAE) of the predicted tip entirely in SQL.
--
-- Run:  duckdb :memory: < examples/regression_tips.sql
-- Needs: real regression weights downloaded (CALL tabfm_download('regression'))
--        and the resources/ graph (see examples/tabfm_real_regression.json).

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;
SET anofox_tabfm_model_manifest = 'examples/tabfm_real_regression.json';

-- 1. Load; add a VARCHAR row id (categorical → inert as an unseen value in the
--    test split, so it is safe to carry through for the join-back).
CREATE TABLE tips AS
SELECT
    't' || (row_number() OVER ())::VARCHAR AS row_id,
    total_bill::DOUBLE AS total_bill,
    sex, smoker, day, time, size::INT AS size,
    tip::DOUBLE AS tip,
    hash(row_number() OVER ()) % 100 AS bucket
FROM 'hf://datasets/scikit-learn/tips/**/*.csv';

-- 2. Split ~180 context / ~64 scored.
CREATE TABLE train AS SELECT * EXCLUDE (bucket) FROM tips WHERE bucket < 75;
CREATE TABLE test_full AS SELECT * EXCLUDE (bucket) FROM tips WHERE bucket >= 75;

CREATE TABLE test_features AS SELECT * EXCLUDE (tip) FROM test_full;
CREATE TABLE test_actuals  AS SELECT row_id, tip FROM test_full;

-- 3. Zero-shot predict the tip for the test rows.
CREATE TABLE preds AS
SELECT row_id, yhat AS pred
FROM tabfm_regress('train', 'tip', test := 'test_features');

-- 4. MSE / RMSE / MAE, plus a naive mean-predictor baseline, in SQL.
.print '================ TIPS regression (TabFM zero-shot) ================'
.timer on
WITH scored AS (
    SELECT p.pred, a.tip AS actual
    FROM preds p JOIN test_actuals a USING (row_id)
), baseline AS (
    SELECT avg(tip) AS mean_tip FROM train
)
SELECT
    count(*)                                             AS n_test,
    avg((pred - actual) ^ 2)                             AS mse,
    sqrt(avg((pred - actual) ^ 2))                       AS rmse,
    avg(abs(pred - actual))                              AS mae,
    (SELECT avg((mean_tip - actual) ^ 2) FROM scored, baseline) AS baseline_mse
FROM scored;
