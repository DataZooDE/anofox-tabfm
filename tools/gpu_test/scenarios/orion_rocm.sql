-- Orion-BiX and Orion-MSP (classification only) on ROCm, real weights: served by rocm:0, and the GPU agrees with the
-- CPU on every row.
--
-- Run with both weights downloaded and the freshly built MIGraphX plugin:
--
--   SET anofox_tabfm_ep_path = 'build/release/extension/anofox_tabfm';  -- harness
--
-- The first call per (rows, features) bucket compiles the masked graph (see docs/REAL_MODELS.md for the measured
-- times); later calls hit the .mxr cache. Never trust "cpu and gpu agree" (CLAUDE.md): read the SERVED_BY lines.

.timer on
SELECT backend, model, task, supported, reason FROM tabfm_backends()
WHERE model IN ('orion-bix', 'orion-msp') AND backend = 'rocm';
-- 100 rows x 8 features -> the 128 x 16 bucket; 70 context rows, 30 queries
CREATE TABLE t AS SELECT i AS rid,
  sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
  sin(i * 0.31) f5, cos(i * 0.17) f6, (i % 7)::DOUBLE f7, (i * 37 % 100) / 100.0 f8,
  CASE WHEN i < 70 THEN CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > 0.3 THEN 'a'
                             WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END END AS label
FROM range(100) r(i);

-- ---- Orion-BiX ----
SET anofox_tabfm_device = 'cpu';
CREATE TABLE bix_cpu AS SELECT rid, yhat, is_training FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-bix') JOIN t USING (f1, f2, f3, f4, f5, f6, f7, f8);
SET anofox_tabfm_device = 'rocm';
-- first call compiles the bucket (timer)
CREATE TABLE bix_gpu AS SELECT rid, yhat, is_training FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-bix') JOIN t USING (f1, f2, f3, f4, f5, f6, f7, f8);
SELECT 'BIX_SERVED_BY', model, task, device FROM tabfm_models() WHERE model = 'orion-bix';
SELECT 'BIX_WARM_GPU', count(*) FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-bix');
SET anofox_tabfm_device = 'cpu';
SELECT 'BIX_WARM_CPU', count(*) FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-bix');
SELECT 'BIX_AGREE', count(*) AS n, count(*) FILTER (WHERE c.yhat <> g.yhat) AS disagree,
       count(*) FILTER (WHERE c.yhat <> g.yhat AND NOT c.is_training) AS disagree_query
FROM bix_cpu c JOIN bix_gpu g USING (rid);

-- ---- Orion-MSP ----
SET anofox_tabfm_device = 'cpu';
CREATE TABLE msp_cpu AS SELECT rid, yhat, is_training FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-msp') JOIN t USING (f1, f2, f3, f4, f5, f6, f7, f8);
SET anofox_tabfm_device = 'rocm';
CREATE TABLE msp_gpu AS SELECT rid, yhat, is_training FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-msp') JOIN t USING (f1, f2, f3, f4, f5, f6, f7, f8);
SELECT 'MSP_SERVED_BY', model, task, device FROM tabfm_models() WHERE model = 'orion-msp';
SELECT 'MSP_WARM_GPU', count(*) FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-msp');
SET anofox_tabfm_device = 'cpu';
SELECT 'MSP_WARM_CPU', count(*) FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-msp');
SELECT 'MSP_AGREE', count(*) AS n, count(*) FILTER (WHERE c.yhat <> g.yhat) AS disagree,
       count(*) FILTER (WHERE c.yhat <> g.yhat AND NOT c.is_training) AS disagree_query
FROM msp_cpu c JOIN msp_gpu g USING (rid);

-- auto now picks the GPU for both (it picked the CPU while the graphs were positional)
SET anofox_tabfm_device = 'auto';
SELECT 'AUTO', count(*) FROM tabfm_classify('t', 'label', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'orion-bix');
SELECT 'AUTO_SERVED_BY', model, task, device FROM tabfm_models() WHERE model = 'orion-bix';
