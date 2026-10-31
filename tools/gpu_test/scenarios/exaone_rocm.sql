-- EXAONE Tabular (classify + regress) on ROCm, real weights: served by rocm:0, and the GPU agrees with the CPU on every row.
--
-- Run with the weights downloaded, the licence accepted, and the freshly built MIGraphX plugin:
--
--   SET anofox_tabfm_ep_path = 'build/release/extension/anofox_tabfm';  -- harness
--
-- The first call per (rows, features) bucket compiles the graph (see docs/REAL_MODELS.md for the
-- measured times); later calls hit the .mxr cache. Never trust "cpu and gpu agree" (CLAUDE.md):
-- read the SERVED_BY lines.

SET anofox_tabfm_accept_hf_license = true;
.timer on
SELECT backend, model, task, supported, reason FROM tabfm_backends() WHERE model IN ('nori-30m', 'exaone-tabular') AND backend = 'rocm';
-- 100 rows x 8 features -> the 128 x 16 bucket; 70 context rows, 30 queries
CREATE TABLE t AS SELECT i AS rid,
  sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
  sin(i * 0.31) f5, cos(i * 0.17) f6, (i % 7)::DOUBLE f7, (i * 37 % 100) / 100.0 f8,
  CASE WHEN i < 70 THEN CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > 0.3 THEN 'a'
                             WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END END AS label,
  CASE WHEN i < 70 THEN sin(i * 0.11) * 5 + cos(i * 0.07) * 2 + (i % 7) * 0.3 END AS target
FROM range(100) r(i);
CREATE TABLE tc AS SELECT * EXCLUDE (target) FROM t;
CREATE TABLE tr AS SELECT * EXCLUDE (label) FROM t;
SET anofox_tabfm_device = 'cpu';
CREATE TABLE cpu_c AS SELECT rid, yhat, is_training FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular') JOIN tc USING (f1, f2, f3, f4, f5, f6, f7, f8);
CREATE TABLE cpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular') JOIN tr USING (f1, f2, f3, f4, f5, f6, f7, f8);
SET anofox_tabfm_device = 'rocm';
-- first call compiles the bucket (timer)
CREATE TABLE gpu_c AS SELECT rid, yhat, is_training FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular') JOIN tc USING (f1, f2, f3, f4, f5, f6, f7, f8);
SELECT 'SERVED_BY_CLS', model, task, device FROM tabfm_models() WHERE model = 'exaone-tabular';
CREATE TABLE gpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular') JOIN tr USING (f1, f2, f3, f4, f5, f6, f7, f8);
SELECT 'SERVED_BY', model, task, device FROM tabfm_models() WHERE model = 'exaone-tabular';
-- warm calls
SELECT 'WARM_CLS', count(*) FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular');
SELECT 'WARM_REG', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular');
SET anofox_tabfm_device = 'cpu';
SELECT 'WARM_CPU_CLS', count(*) FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular');
SELECT 'WARM_CPU_REG', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'exaone-tabular');
SELECT 'AGREE_CLS', count(*) AS n, count(*) FILTER (WHERE c.yhat <> g.yhat) AS disagree,
       count(*) FILTER (WHERE c.yhat <> g.yhat AND NOT c.is_training) AS disagree_query
FROM cpu_c c JOIN gpu_c g USING (rid);
SELECT 'AGREE_REG', count(*) AS n, max(abs(c.yhat - g.yhat)) AS max_abs, corr(c.yhat, g.yhat) AS corr,
       max(abs(c.yhat - g.yhat)) FILTER (WHERE NOT c.is_training) AS max_abs_query
FROM cpu_r c JOIN gpu_r g USING (rid);
SELECT 'ACC_GPU', avg(CASE WHEN g.yhat = t.label THEN 1.0 ELSE 0 END) FROM gpu_c g JOIN t USING (rid) WHERE g.is_training;
-- Nori is refused by name on ROCm
SET anofox_tabfm_device = 'rocm';
SELECT 'ROCM_NORI', count(*) FROM tabfm_regress('tr', 'target', model := 'nori-30m');
SET anofox_tabfm_device = 'auto';
SELECT 'AUTO_NORI', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-30m');
SELECT 'SERVED_BY_NORI', model, task, device FROM tabfm_models() WHERE model = 'nori-30m';
