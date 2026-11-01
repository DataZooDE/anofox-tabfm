-- The model catalog on CUDA against the CPU, one (model, task) per DuckDB process.
--
-- Driven by tools/gpu_test/cuda_catalog.sh, which runs this file once per database and then
-- cuda_catalog_classification.sql / cuda_catalog_regression.sql once per (model, task), each in a
-- FRESH process: a model that crashes the process then costs its own row, not every later one.
-- The results accumulate in the persistent database's `report` table.
--
-- Same 100-row tables for every model: 4 numeric features, 70 labelled context rows, 30 query rows
-- (row_id % 10 >= 7, interleaved, so context and query cover the same phase of the sin/cos series).
-- Three classes for classification, a smooth target for regression.
CREATE TABLE IF NOT EXISTS base AS
SELECT i AS row_id,
       sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
       CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) >  0.3 THEN 'a'
            WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END AS y,
       sin(i * 0.11) * 5 + cos(i * 0.07) * 2 AS tgt
FROM range(100) t(i);
-- One context table per task: a context column the query lacks is a hard error (feature guard).
CREATE TABLE IF NOT EXISTS ctx_c AS SELECT * EXCLUDE (tgt) FROM base WHERE row_id % 10 < 7;
CREATE TABLE IF NOT EXISTS ctx_r AS SELECT * EXCLUDE (y)   FROM base WHERE row_id % 10 < 7;
CREATE TABLE IF NOT EXISTS qry   AS SELECT * EXCLUDE (y, tgt) FROM base WHERE row_id % 10 >= 7;
CREATE TABLE IF NOT EXISTS truth AS SELECT row_id, y, tgt FROM base WHERE row_id % 10 >= 7;

-- One row per (model, task). served_by is THE column: anything but cuda:N means the "GPU" side ran
-- on the CPU and every number beside it is cpu-vs-cpu. n_gpu = 0 means the CUDA call errored (the
-- driver's log has the message).
CREATE TABLE IF NOT EXISTS report (
    model VARCHAR, task VARCHAR, served_by VARCHAR,
    n_cpu BIGINT, n_gpu BIGINT,
    label_diff BIGINT,          -- classification: rows whose label differs CPU vs CUDA
    score_max_abs_diff DOUBLE,  -- classification: yhat_score; regression: yhat
    reg_max_rel_diff DOUBLE,    -- regression: max |cpu - gpu| / (max - min of the CPU predictions)
    repeat_diff BIGINT,         -- CUDA cold vs CUDA warm: labels (clf) / values differing > 1e-6 (reg)
    cpu_quality DOUBLE,         -- accuracy (clf) / R^2 (reg) on the CPU
    gpu_quality DOUBLE,         -- the same on CUDA
    cpu_ms BIGINT, gpu_cold_ms BIGINT, gpu_warm_ms BIGINT);
