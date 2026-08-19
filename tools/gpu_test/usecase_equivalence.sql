-- Realistic use-case check: the same classification query on CPU and on ROCm,
-- through the SQL surface, against the REAL TabFM weights.
--
-- This is the first thing to exercise the actual user path on a GPU:
-- SET anofox_tabfm_device -> TryMIGraphXBackend dispatch -> ep_path -> the
-- plugin -> the aggregate's chunking. Everything before this drove the plugin
-- from a C++ host with five hand-built rows.
--
-- Sized to stay inside the already-compiled T128/H16 shape bucket (<=128 total
-- rows, <=16 features) so it costs no MIGraphX recompile.
-- Run:  ./build/debug/duckdb -c "SET anofox_tabfm_ep_path='<plugin dir>'" -f tools/gpu_test/usecase_equivalence.sql
--       (or prepend the SET, see tools/gpu_test/README.md)
LOAD anofox_tabfm;

SET anofox_tabfm_gpu_precision = 'bf16';   -- matches the cached .mxr
SET anofox_tabfm_default_model = 'tabfm-v1';  -- the real Google TabFM weights

-- A small but real-shaped problem: 8 numeric features, 3 classes, 70 labelled
-- rows and 30 to predict. Deterministic, so CPU and GPU see identical input.
CREATE TABLE churn AS
SELECT
    i                                              AS id,
    (i * 37 % 100) / 100.0                         AS tenure,
    (i * 53 % 100) / 100.0                         AS monthly_spend,
    (i * 71 % 100) / 100.0                         AS support_tickets,
    (i * 13 % 100) / 100.0                         AS logins,
    (i * 29 % 100) / 100.0                         AS discount,
    (i * 47 % 100) / 100.0                         AS latency,
    (i * 61 % 100) / 100.0                         AS errors,
    (i * 17 % 100) / 100.0                         AS nps,
    CASE WHEN i < 70
         THEN CASE WHEN (i * 37 % 100) < 33 THEN 'churn'
                   WHEN (i * 37 % 100) < 66 THEN 'stay'
                   ELSE 'upgrade' END
         ELSE NULL END                             AS segment
FROM range(100) t(i);

SELECT 'rows' AS k, count(*) AS v FROM churn
UNION ALL SELECT 'to_predict', count(*) FROM churn WHERE segment IS NULL;

-- CPU reference
SET anofox_tabfm_device = 'cpu';
-- (device is asserted after each run below: a cached session used to be reused
-- across a device switch, which made this whole comparison cpu-vs-cpu while
-- reporting a perfect score.)
CREATE TABLE pred_cpu AS
SELECT id, yhat FROM tabfm_classify('churn', 'segment') ORDER BY id;
SELECT 'cpu_done' AS k, count(*)::VARCHAR AS v FROM pred_cpu
UNION ALL SELECT 'cpu_served_by', coalesce(max(device), 'NONE') FROM tabfm_models() WHERE loaded;

-- ROCm through the plugin
SET anofox_tabfm_device = 'rocm';
CREATE TABLE pred_rocm AS
SELECT id, yhat FROM tabfm_classify('churn', 'segment') ORDER BY id;
-- THE check. Anything but rocm:N here means the GPU never ran and every number
-- below is cpu-vs-cpu, however green it looks.
SELECT 'rocm_done' AS k, count(*)::VARCHAR AS v FROM pred_rocm
UNION ALL SELECT 'rocm_served_by', coalesce(max(device), 'NONE') FROM tabfm_models() WHERE loaded;

-- The thing that matters: does switching device change the answer?
SELECT 'total_rows'    AS k, count(*)::VARCHAR AS v FROM pred_cpu
UNION ALL
SELECT 'disagreements', count(*)::VARCHAR
  FROM pred_cpu c JOIN pred_rocm g USING (id)
 WHERE c.yhat IS DISTINCT FROM g.yhat
UNION ALL
SELECT 'agreement_pct',
       round(100.0 * sum(CASE WHEN c.yhat IS NOT DISTINCT FROM g.yhat THEN 1 ELSE 0 END) / count(*), 2)::VARCHAR
  FROM pred_cpu c JOIN pred_rocm g USING (id);

-- Show the predicted-class distribution on both, so a degenerate
-- all-one-class GPU result cannot hide behind a high agreement number.
SELECT 'cpu'  AS backend, yhat, count(*) AS n FROM pred_cpu  WHERE yhat IS NOT NULL GROUP BY ALL
UNION ALL
SELECT 'rocm' AS backend, yhat, count(*) AS n FROM pred_rocm WHERE yhat IS NOT NULL GROUP BY ALL
ORDER BY backend, yhat;
