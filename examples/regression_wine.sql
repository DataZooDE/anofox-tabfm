-- Real-model verification, pure SQL: zero-shot WINE-QUALITY regression.
-- Dataset: mstz/wine (HF), numeric target `quality` (physicochemical → score).
-- The train split is the in-context set, the test split is scored; we compute
-- MSE / RMSE / MAE (and a mean-predictor baseline) entirely in SQL.
--
-- Run:  duckdb :memory: < examples/regression_wine.sql
-- Needs: real regression weights (CALL tabfm_download('regression'))
--        (built-in model 'tabfm-v1' — no manifest file needed).

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;

-- 1. Load; deterministic split key.
CREATE TABLE wine AS
SELECT
    row_number() OVER () AS row_id,
    * EXCLUDE (quality),
    quality::DOUBLE AS quality,
    hash(row_number() OVER ()) % 100 AS bucket
FROM 'hf://datasets/mstz/wine/**/*.csv';

-- 2. ~200 in-context / ~64 scored.
CREATE TABLE train AS
    SELECT * EXCLUDE (bucket) FROM (FROM wine WHERE bucket < 75) USING SAMPLE 200 ROWS (reservoir, 42);
CREATE TABLE test_full AS
    SELECT * EXCLUDE (bucket) FROM (FROM wine WHERE bucket >= 75) USING SAMPLE 64 ROWS (reservoir, 42);
CREATE TABLE test_features AS SELECT * EXCLUDE (quality) FROM test_full;
CREATE TABLE test_actuals  AS SELECT row_id, quality FROM test_full;

-- 3. Zero-shot predict the quality score for the test rows.
.timer on
CREATE TABLE preds AS
SELECT row_id, yhat AS pred
FROM tabfm_regress('train', 'quality', test := 'test_features', model := 'tabfm-v1');

-- 4. MSE / RMSE / MAE + a naive mean-predictor baseline, in SQL.
.print '================ WINE-QUALITY regression (TabFM zero-shot) ================'
WITH scored AS (
    SELECT p.pred, a.quality AS actual
    FROM preds p JOIN test_actuals a USING (row_id)
), baseline AS (
    SELECT avg(quality) AS mean_q FROM train
)
SELECT
    count(*)                                              AS n_test,
    avg((pred - actual) ^ 2)                              AS mse,
    sqrt(avg((pred - actual) ^ 2))                        AS rmse,
    avg(abs(pred - actual))                               AS mae,
    (SELECT avg((mean_q - actual) ^ 2) FROM scored, baseline) AS baseline_mse
FROM scored;
