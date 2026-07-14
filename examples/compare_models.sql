-- Multi-model comparison, pure SQL: the SAME dataset + SAME split scored by TWO
-- REAL tabular foundation models, reporting per-model ACCURACY and wall-clock
-- RUNTIME. Selecting a model is just `model :=` over one registry (FR-5.1 / M4).
--
-- Models compared (both real, both discovered via tabfm_list_models()):
--   * mitra     — Mitra (AWS AutoGluon), Apache-2.0, 72 M params / ~303 MB.
--                 Commercial-clean. Rank/quantile-normalizes INSIDE the graph.
--   * tabfm-v1  — Google TabFM, 1.6 B params / 6.56 GB, gated non-commercial.
--
-- Run:  duckdb :memory: < examples/compare_models.sql   (from the repo root)
-- Needs, once:
--   SET anofox_tabfm_model_manifest = 'examples/mitra.json';
--   CALL tabfm_download('classification');                 -- Mitra, ~303 MB (ungated)
--   SET anofox_tabfm_accept_hf_license = true;             -- TabFM is gated
--   -- TabFM's 6.56 GB weights: cached from the other examples, or
--   -- SET anofox_tabfm_model_manifest to the built-in and CALL tabfm_download.

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;
SET anofox_tabfm_accept_hf_license = true;

-- A single-file manifest registers `mitra` AND leaves the built-in `tabfm-v1`
-- in the registry (built-ins are always present) → two real models, one query.
SET anofox_tabfm_model_manifest = 'examples/mitra.json';

.print '================ registered models ================'
SELECT model, capabilities, license, commercial, downloaded
FROM tabfm_list_models() ORDER BY model;

-- IRIS: 3 classes, 4 features. Deterministic split — the train split is the
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
CREATE TABLE test_features AS SELECT * EXCLUDE (bucket, species) FROM iris WHERE bucket >= 67;
CREATE TABLE test_actuals  AS SELECT row_id, species    FROM iris WHERE bucket >= 67;

-- Each predict is its own timed statement, so the printed "Run Time" is that
-- model's end-to-end cost (each model loads its weights on its first call).
.timer on

.print ''
.print '================ predict: mitra (real, ~303 MB) ================'
CREATE TABLE preds_mitra AS
SELECT row_id, yhat AS pred
FROM tabfm_classify('train', 'species', test := 'test_features', model := 'mitra');

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
    SELECT 'mitra' AS model, (a.species = p.pred)::INT AS correct
    FROM preds_mitra p JOIN test_actuals a USING (row_id)
    UNION ALL
    SELECT 'tabfm-v1' AS model, (a.species = p.pred)::INT AS correct
    FROM preds_tabfm p JOIN test_actuals a USING (row_id)
)
SELECT
    model,
    count(*)     AS n_test,
    sum(correct) AS n_correct,
    avg(correct) AS accuracy
FROM scored
GROUP BY model
ORDER BY accuracy DESC;
