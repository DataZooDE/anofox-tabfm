-- Real-model verification, pure SQL: zero-shot IRIS classification (3 classes).
-- Dataset: scikit-learn/iris (HF), target `species` (setosa/versicolor/virginica).
-- TabFM is in-context: the train split is the context, the test split is scored.
-- We compute multiclass accuracy (overall + per-species) entirely in SQL.
--
-- Run:  duckdb :memory: < examples/classification_iris.sql
-- Needs: real classification weights downloaded (CALL tabfm_download('classification'))
--        (built-in model 'tabfm-v1' — no manifest file needed).

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;

-- 1. Load the 150 rows; friendly column names + a deterministic split key.
--    hash(Id) scatters the three (row-ordered) species across both splits so the
--    context is class-balanced. Id is kept only for the join-back (an unseen
--    categorical in the test split → inert, no leakage).
CREATE TABLE iris AS
SELECT
    Id::VARCHAR              AS row_id,
    SepalLengthCm::DOUBLE    AS sepal_length,
    SepalWidthCm::DOUBLE     AS sepal_width,
    PetalLengthCm::DOUBLE    AS petal_length,
    PetalWidthCm::DOUBLE     AS petal_width,
    Species                  AS species,
    hash(Id) % 100           AS bucket
FROM 'hf://datasets/scikit-learn/iris/**/*.csv';

-- 2. Split ~100 context (train) / ~50 scored (test).
CREATE TABLE train      AS SELECT * EXCLUDE (bucket) FROM iris WHERE bucket <  67;
CREATE TABLE test_full  AS SELECT * EXCLUDE (bucket) FROM iris WHERE bucket >= 67;

CREATE TABLE test_features AS SELECT * EXCLUDE (species) FROM test_full;
CREATE TABLE test_actuals  AS SELECT row_id, species FROM test_full;

-- 3. Zero-shot predict the species of the test rows using the train rows as context.
.timer on
CREATE TABLE preds AS
SELECT row_id, yhat AS pred
FROM tabfm_classify('train', 'species', test := 'test_features', model := 'tabfm-v1');

-- 4. Accuracy, in SQL: overall and per true species.
CREATE TABLE scored AS
SELECT a.species AS actual, p.pred, (a.species = p.pred) AS correct
FROM preds p JOIN test_actuals a USING (row_id);

.print '================ IRIS classification (TabFM zero-shot, 3 classes) ================'
SELECT
    count(*)                                  AS n_test,
    count(*) FILTER (WHERE correct)           AS n_correct,
    avg(correct::INT)                         AS accuracy
FROM scored;

.print '---- per-species recall ----'
SELECT
    actual                                    AS species,
    count(*)                                  AS n,
    count(*) FILTER (WHERE correct)           AS n_correct,
    avg(correct::INT)                         AS recall
FROM scored
GROUP BY actual
ORDER BY actual;
