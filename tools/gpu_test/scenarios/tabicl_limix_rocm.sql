-- TabICL v2 and LimiX-2M (classify + regress) on ROCm, real weights: served by rocm:0 through their MASKED
-- graphs, and the GPU agrees with the CPU on every row. test/sql/tabfm_rocm_real_models.test asserts the same
-- (TABFM_REAL_WEIGHTS=1 TABFM_ROCM=1); this file adds the timings.
--
-- Run with the weights downloaded (CALL tabfm_download, which for LimiX also writes the model.safetensors
-- beside model.ckpt that the masked graph reads), and the freshly built MIGraphX plugin:
--
--   SET anofox_tabfm_ep_path = 'build/release/extension/anofox_tabfm';  -- harness
--
-- The first call per (rows, features) bucket compiles the graph (docs/REAL_MODELS.md has the measured
-- times); later calls hit the .mxr cache. Never trust "cpu and gpu agree" (CLAUDE.md): read SERVED_BY.

SET anofox_tabfm_accept_hf_license = true;
.timer on
SELECT backend, model, task, supported, reason FROM tabfm_backends() WHERE model IN ('tabicl-v2', 'limix-2m') AND backend = 'rocm';
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

-- ---------------------------------------------------------------- tabicl-v2
SET anofox_tabfm_device = 'cpu';
CREATE TABLE tabicl_cpu_c AS SELECT rid, yhat, is_training FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2') JOIN tc USING (f1, f2, f3, f4, f5, f6, f7, f8);
CREATE TABLE tabicl_cpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2') JOIN tr USING (f1, f2, f3, f4, f5, f6, f7, f8);
SET anofox_tabfm_device = 'rocm';
CREATE TABLE tabicl_gpu_c AS SELECT rid, yhat, is_training FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2') JOIN tc USING (f1, f2, f3, f4, f5, f6, f7, f8);
CREATE TABLE tabicl_gpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2') JOIN tr USING (f1, f2, f3, f4, f5, f6, f7, f8);
SELECT 'SERVED_BY', model, task, device FROM tabfm_models() WHERE loaded AND model = 'tabicl-v2';
SELECT 'WARM_GPU_CLS', count(*) FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2');
SELECT 'WARM_GPU_REG', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2');
SET anofox_tabfm_device = 'cpu';
SELECT 'WARM_CPU_CLS', count(*) FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2');
SELECT 'WARM_CPU_REG', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'tabicl-v2');
SELECT 'AGREE_CLS', count(*) AS n, count(*) FILTER (WHERE c.yhat <> g.yhat) AS disagree,
       count(*) FILTER (WHERE c.yhat <> g.yhat AND NOT c.is_training) AS disagree_query
FROM tabicl_cpu_c c JOIN tabicl_gpu_c g USING (rid);
SELECT 'AGREE_REG', count(*) AS n, max(abs(c.yhat - g.yhat)) AS max_abs, corr(c.yhat, g.yhat) AS corr,
       max(abs(c.yhat - g.yhat)) FILTER (WHERE NOT c.is_training) AS max_abs_query, max(abs(c.yhat)) AS scale
FROM tabicl_cpu_r c JOIN tabicl_gpu_r g USING (rid);

-- ---------------------------------------------------------------- limix-2m
SET anofox_tabfm_device = 'cpu';
CREATE TABLE limix_cpu_c AS SELECT rid, yhat, is_training FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m') JOIN tc USING (f1, f2, f3, f4, f5, f6, f7, f8);
CREATE TABLE limix_cpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m') JOIN tr USING (f1, f2, f3, f4, f5, f6, f7, f8);
SET anofox_tabfm_device = 'rocm';
CREATE TABLE limix_gpu_c AS SELECT rid, yhat, is_training FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m') JOIN tc USING (f1, f2, f3, f4, f5, f6, f7, f8);
CREATE TABLE limix_gpu_r AS SELECT rid, yhat, is_training FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m') JOIN tr USING (f1, f2, f3, f4, f5, f6, f7, f8);
SELECT 'SERVED_BY', model, task, device FROM tabfm_models() WHERE loaded AND model = 'limix-2m';
SELECT 'WARM_GPU_CLS', count(*) FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m');
SELECT 'WARM_GPU_REG', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m');
SET anofox_tabfm_device = 'cpu';
SELECT 'WARM_CPU_CLS', count(*) FROM tabfm_classify('tc', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m');
SELECT 'WARM_CPU_REG', count(*) FROM tabfm_regress('tr', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'limix-2m');
SELECT 'AGREE_CLS', count(*) AS n, count(*) FILTER (WHERE c.yhat <> g.yhat) AS disagree,
       count(*) FILTER (WHERE c.yhat <> g.yhat AND NOT c.is_training) AS disagree_query
FROM limix_cpu_c c JOIN limix_gpu_c g USING (rid);
SELECT 'AGREE_REG', count(*) AS n, max(abs(c.yhat - g.yhat)) AS max_abs, corr(c.yhat, g.yhat) AS corr,
       max(abs(c.yhat - g.yhat)) FILTER (WHERE NOT c.is_training) AS max_abs_query, max(abs(c.yhat)) AS scale
FROM limix_cpu_r c JOIN limix_gpu_r g USING (rid);
