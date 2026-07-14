-- Multi-model comparison, pure SQL: the SAME dataset + SAME split scored by TWO
-- registered models, reporting per-model ACCURACY and wall-clock RUNTIME.
--
-- This is the multi-model harness (FR-5.1 / M4): a model is selected per call
-- with `model :=`, so comparing models is just two queries over one registry.
--
-- Models compared (both discovered via tabfm_list_models()):
--   * tabfm-v1           — the real Google TabFM foundation model (6.56 GB,
--                          zero-shot in-context learner).
--   * fixture-commercial — a committed random-init fixture (v2 schema,
--                          apache-2.0), reusing the tiny test graph. It is NOT a
--                          trained model; it is here so the comparison runs
--                          end-to-end today. Its ~chance accuracy is EXPECTED —
--                          it makes the point that a foundation model buys a lot
--                          of accuracy for its runtime/size. Swap in a second
--                          REAL model (e.g. a future Mitra export) for a
--                          meaningful head-to-head.
--
-- Run:  duckdb :memory: < examples/compare_models.sql   (from the repo root)
-- Needs: CALL tabfm_download('classification');  (real weights, once)
--        and SET anofox_tabfm_accept_hf_license = true.

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;
SET anofox_tabfm_accept_hf_license = true;

-- A DIRECTORY manifest registers fixture-commercial ALONGSIDE the built-in
-- tabfm-v1 (built-ins are always present) → two models in one registry.
SET anofox_tabfm_model_manifest = 'test/fixtures/multi';

.print '================ registered models ================'
SELECT model, capabilities, license, commercial, downloaded
FROM tabfm_list_models() ORDER BY model;

-- IRIS: 3 classes, 4 features. Deterministic split — the train split is TabFM's
-- in-context set, the test split is scored. row_id is carried for the join-back
-- (an unseen categorical in the test split → inert, no leakage).
CREATE TABLE iris AS
SELECT
    Id::VARCHAR           AS row_id,
    SepalLengthCm::DOUBLE AS sepal_length,
    SepalWidthCm::DOUBLE  AS sepal_width,
    PetalLengthCm::DOUBLE AS petal_length,
    PetalWidthCm::DOUBLE  AS petal_width,
    Species               AS species,
    hash(Id) % 100        AS bucket
FROM 'hf://datasets/scikit-learn/iris/**/*.csv';

CREATE TABLE train         AS SELECT * EXCLUDE (bucket)  FROM iris WHERE bucket <  67;
CREATE TABLE test_full     AS SELECT * EXCLUDE (bucket)  FROM iris WHERE bucket >= 67;
CREATE TABLE test_features AS SELECT * EXCLUDE (species) FROM test_full;
CREATE TABLE test_actuals  AS SELECT row_id, species    FROM test_full;

-- Each predict is its own timed statement, so the printed "Run Time" is that
-- model's end-to-end cost (tabfm-v1's first call also pays the 6.56 GB load).
.timer on

.print ''
.print '================ predict: fixture-commercial (tiny, random-init) ================'
CREATE TABLE preds_fixture AS
SELECT row_id, yhat AS pred
FROM tabfm_classify('train', 'species', test := 'test_features', model := 'fixture-commercial');

.print ''
.print '================ predict: tabfm-v1 (real, loads 6.56 GB on first call) ================'
CREATE TABLE preds_tabfm AS
SELECT row_id, yhat AS pred
FROM tabfm_classify('train', 'species', test := 'test_features', model := 'tabfm-v1');

.timer off

-- Side-by-side accuracy. Runtime is the per-predict "Run Time (s)" above.
.print ''
.print '================ results: accuracy by model ================'
WITH scored AS (
    SELECT 'fixture-commercial' AS model, (a.species = p.pred)::INT AS correct
    FROM preds_fixture p JOIN test_actuals a USING (row_id)
    UNION ALL
    SELECT 'tabfm-v1' AS model, (a.species = p.pred)::INT AS correct
    FROM preds_tabfm p JOIN test_actuals a USING (row_id)
)
SELECT
    model,
    count(*)          AS n_test,
    sum(correct)      AS n_correct,
    avg(correct)      AS accuracy
FROM scored
GROUP BY model
ORDER BY accuracy DESC;
