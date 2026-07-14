-- Real-model verification, pure SQL: zero-shot CHURN classification.
-- Dataset: scikit-learn/churn-prediction (HF), binary target `Churn`.
-- TabFM is in-context: the train split is the context, the test split is scored.
-- We compute the F1 score of the positive class (churned) entirely in SQL.
--
-- Run:  duckdb :memory: < examples/classification_churn.sql
-- Needs: real classification weights downloaded (CALL tabfm_download('classification'))
--        (built-in model 'tabfm-v1' — no manifest file needed).

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;
SET anofox_tabfm_max_rows = 20000;

-- 1. Load, clean, and take a manageable, reproducible sample.
--    TotalCharges has blanks for tenure=0; cast safely. Drop the id from the
--    feature set implicitly by NOT selecting it into the feature views; keep it
--    only in the actuals table for the join-back.
CREATE TABLE churn AS
SELECT
    customerID,
    gender, SeniorCitizen::INT AS SeniorCitizen, Partner, Dependents,
    tenure::INT AS tenure, PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges::DOUBLE AS MonthlyCharges,
    TRY_CAST(TotalCharges AS DOUBLE) AS TotalCharges,
    Churn::BOOLEAN AS Churn,
    -- deterministic split key
    hash(customerID) % 100 AS bucket
FROM 'hf://datasets/scikit-learn/churn-prediction/**/*.csv';

-- 2. Split: ~500 context (train) rows, ~150 scored (test) rows, class-balanced
--    enough for a meaningful F1 without a multi-minute forward pass.
CREATE TABLE train AS
    SELECT * EXCLUDE (bucket) FROM (FROM churn WHERE bucket < 70) USING SAMPLE 500 ROWS (reservoir, 42);
CREATE TABLE test_full AS
    SELECT * EXCLUDE (bucket) FROM (FROM churn WHERE bucket >= 70) USING SAMPLE 150 ROWS (reservoir, 42);

-- test features (no target) for prediction, and held-out actuals for scoring
CREATE TABLE test_features AS SELECT * EXCLUDE (Churn) FROM test_full;
CREATE TABLE test_actuals  AS SELECT customerID, Churn FROM test_full;

-- 3. Zero-shot predict the test rows using the train rows as context.
.timer on
CREATE TABLE preds AS
SELECT customerID, yhat AS pred
FROM tabfm_classify('train', 'Churn', test := 'test_features', model := 'tabfm-v1');

-- 4. F1 of the positive class (churned = true), in SQL.
CREATE TABLE cm AS
SELECT
    count(*) FILTER (WHERE p.pred AND a.Churn)          AS tp,
    count(*) FILTER (WHERE p.pred AND NOT a.Churn)      AS fp,
    count(*) FILTER (WHERE NOT p.pred AND a.Churn)      AS fn,
    count(*) FILTER (WHERE NOT p.pred AND NOT a.Churn)  AS tn
FROM preds p JOIN test_actuals a USING (customerID);

.print '================ CHURN classification (TabFM zero-shot) ================'
SELECT
    tp, fp, fn, tn,
    (tp + tn)::DOUBLE / (tp + fp + fn + tn)                          AS accuracy,
    tp::DOUBLE / nullif(tp + fp, 0)                                  AS precision,
    tp::DOUBLE / nullif(tp + fn, 0)                                  AS recall,
    2.0 * tp / nullif(2.0 * tp + fp + fn, 0)                         AS f1
FROM cm;
