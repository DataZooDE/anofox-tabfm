-- Nori-30M (regression only) on ROCm, real weights: served by rocm:0, and the GPU agrees with the CPU on every row.
--
-- Run with the weights downloaded by CALL tabfm_download (which also writes the model.safetensors beside
-- nori.pt that the masked graph reads), and the freshly built MIGraphX plugin:
--
--   SET anofox_tabfm_ep_path = 'build/release/extension/anofox_tabfm';  -- harness
--
-- The first call per (rows, features) bucket compiles the graph (see docs/REAL_MODELS.md for the
-- measured times); later calls hit the .mxr cache. Never trust "cpu and gpu agree" (CLAUDE.md):
-- read the SERVED_BY lines.

.timer on
SELECT backend, model, task, supported, reason FROM tabfm_backends() WHERE model = 'nori-30m' AND backend = 'rocm';
CREATE TABLE t AS SELECT i AS rid, sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
  sin(i * 0.31) f5, cos(i * 0.17) f6, (i % 7)::DOUBLE f7, (i * 37 % 100) / 100.0 f8,
  CASE WHEN i < 70 THEN sin(i * 0.11) * 5 + cos(i * 0.07) * 2 + (i % 7) * 0.3 END AS target FROM range(100) r(i);
SET anofox_tabfm_device = 'cpu';
CREATE TABLE cpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-30m') JOIN t USING (f1,f2,f3,f4,f5,f6,f7,f8);
SET anofox_tabfm_device = 'rocm';
CREATE TABLE gpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-30m') JOIN t USING (f1,f2,f3,f4,f5,f6,f7,f8);
SELECT 'SERVED_BY', model, task, device FROM tabfm_models() WHERE model = 'nori-30m';
SELECT 'WARM_GPU', count(*) FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-30m');
SET anofox_tabfm_device = 'cpu';
SELECT 'WARM_CPU', count(*) FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-30m');
SELECT 'AGREE_REG', count(*) AS n, max(abs(c.yhat - g.yhat)) AS max_abs, corr(c.yhat, g.yhat) AS corr,
       max(abs(c.yhat - g.yhat)) FILTER (WHERE NOT c.is_training) AS max_abs_query, max(abs(c.yhat)) AS scale
FROM cpu_r c JOIN gpu_r g USING (rid);
SELECT 'R2_GPU_CTX', corr(g.yhat, t.target) FROM gpu_r g JOIN t USING (rid) WHERE g.is_training;
