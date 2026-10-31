-- TabPFN-3.5 / 3.5-Fast on ROCm, real weights: served by rocm:0, and the GPU
-- agrees with the CPU on EVERY row (context and query), both tasks.
--
-- Run with the weights downloaded and the licence accepted, the model id in a
-- variable, and the freshly built MIGraphX plugin (graphs and plugin ship
-- together; an older plugin refuses these graphs over `n_rows`):
--
--   SET anofox_tabfm_ep_path = 'build/release/extension/anofox_tabfm';  -- harness
--   SET VARIABLE m = 'tabpfn-v3-5-fast';                                  -- harness
--
-- The first call per (rows, features) bucket compiles the graph (minutes; see
-- docs/REAL_MODELS.md for the measured times), later calls hit the .mxr cache.
-- Never trust "cpu and gpu agree" (CLAUDE.md): read the SERVED_BY lines.

SET anofox_tabfm_accept_hf_license = true;

CREATE OR REPLACE TABLE g35 AS
SELECT i AS row_id,
       sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
       (i % 5)::DOUBLE f5, sin(i * 0.3) * cos(i * 0.05) f6, (i * 7 % 11)::DOUBLE f7, i / 100.0 f8,
       CASE WHEN i < 70 THEN (CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > 0.3 THEN 'a'
                                   WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END)
       END AS label,
       CASE WHEN i < 70 THEN sin(i * 0.11) * 5 + cos(i * 0.07) * 2 + f7 * 0.1 END AS target
FROM range(100) t(i);

CREATE OR REPLACE TABLE g35_cls AS SELECT * EXCLUDE (target) FROM g35;
CREATE OR REPLACE TABLE g35_reg AS SELECT * EXCLUDE (label) FROM g35;

SET anofox_tabfm_device = 'cpu';
CREATE OR REPLACE TABLE cpu_c AS SELECT row_id, yhat, is_training, proba FROM tabfm_classify('g35_cls', 'label', model := getvariable('m'));
CREATE OR REPLACE TABLE cpu_r AS SELECT row_id, yhat, is_training FROM tabfm_regress('g35_reg', 'target', model := getvariable('m'));
SELECT 'CPU_SERVED_BY=' || string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND model = getvariable('m');

SET anofox_tabfm_device = 'rocm';
CREATE OR REPLACE TABLE gpu_c AS SELECT row_id, yhat, is_training, proba FROM tabfm_classify('g35_cls', 'label', model := getvariable('m'));
SELECT 'CLS_SERVED_BY=' || coalesce(max(device), 'NONE') FROM tabfm_models()
WHERE loaded AND model = getvariable('m') AND task = 'classification';
CREATE OR REPLACE TABLE gpu_r AS SELECT row_id, yhat, is_training FROM tabfm_regress('g35_reg', 'target', model := getvariable('m'));
SELECT 'REG_SERVED_BY=' || coalesce(max(device), 'NONE') FROM tabfm_models()
WHERE loaded AND model = getvariable('m') AND task = 'regression';

-- agreement over every row
SELECT 'CLS_LABEL_DIFF query=' || count(*) FILTER (WHERE NOT a.is_training AND a.yhat <> b.yhat)
       || '/' || count(*) FILTER (WHERE NOT a.is_training)
       || ' context=' || count(*) FILTER (WHERE a.is_training AND a.yhat <> b.yhat)
       || '/' || count(*) FILTER (WHERE a.is_training)
       || ' max_proba_diff=' || max(greatest(abs(a.proba['a'] - b.proba['a']), abs(a.proba['b'] - b.proba['b']),
                                             abs(a.proba['c'] - b.proba['c'])))
FROM cpu_c a JOIN gpu_c b USING (row_id);
SELECT 'REG_DIFF max_abs=' || max(abs(a.yhat - b.yhat)) || ' corr=' || corr(a.yhat, b.yhat)
       || ' distinct_gpu=' || count(DISTINCT round(b.yhat, 6))
FROM cpu_r a JOIN gpu_r b USING (row_id);
-- and the GPU answers are real answers, not a constant fill
SELECT 'GPU_FITTED_ACC=' || (sum(CASE WHEN b.yhat = g.label THEN 1 ELSE 0 END)::DOUBLE / count(*))
FROM gpu_c b JOIN g35 g USING (row_id) WHERE b.is_training;
