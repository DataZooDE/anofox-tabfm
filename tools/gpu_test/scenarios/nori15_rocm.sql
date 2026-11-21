-- Nori 1.5 (regression only) on ROCm, real weights: served by rocm:0, and the GPU agrees with the CPU on every row.
--
-- Works for the built-in model or one registered under the id 'nori-1-5' (tools/gpu_test/nori15_rocm.sh registers it
-- against the released extension). The first call per (rows, features) bucket compiles the MASKED graph; later calls
-- hit the .mxr cache. Never trust "cpu and gpu agree" (CLAUDE.md): the verdict at the end reads the serving device.
-- Same 100 x 8 table as nori30m_rocm.sql (the 128 x 16 bucket), so the two models' numbers are comparable.
.timer on
SELECT backend, model, task, supported, reason FROM tabfm_backends() WHERE model = 'nori-1-5' AND backend = 'rocm';
CREATE TABLE t AS SELECT i AS rid, sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
  sin(i * 0.31) f5, cos(i * 0.17) f6, (i % 7)::DOUBLE f7, (i * 37 % 100) / 100.0 f8,
  CASE WHEN i < 70 THEN sin(i * 0.11) * 5 + cos(i * 0.07) * 2 + (i % 7) * 0.3 END AS target FROM range(100) r(i);
CREATE TABLE cpu_r (rid BIGINT, yhat DOUBLE, is_training BOOLEAN);
CREATE TABLE gpu_r (rid BIGINT, yhat DOUBLE, is_training BOOLEAN);
SET anofox_tabfm_device = 'cpu';
INSERT INTO cpu_r SELECT rid, yhat, is_training FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-1-5') JOIN t USING (f1,f2,f3,f4,f5,f6,f7,f8);
SET anofox_tabfm_device = 'rocm';
INSERT INTO gpu_r SELECT rid, yhat, is_training FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-1-5') JOIN t USING (f1,f2,f3,f4,f5,f6,f7,f8);
SELECT 'SERVED_BY', model, task, device FROM tabfm_models() WHERE model = 'nori-1-5';
CREATE TABLE served AS SELECT coalesce(string_agg(DISTINCT device, ','), 'NONE') AS dev FROM tabfm_models() WHERE model = 'nori-1-5' AND loaded AND device <> 'cpu';
CREATE TABLE warm AS SELECT epoch_ms(now()) AS t0;
SELECT 'WARM_GPU', count(*) FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-1-5');
CREATE TABLE warm_gpu_ms AS SELECT epoch_ms(now()) - (SELECT t0 FROM warm) AS ms;
SET anofox_tabfm_device = 'cpu';
CREATE TABLE warm2 AS SELECT epoch_ms(now()) AS t0;
SELECT 'WARM_CPU', count(*) FROM tabfm_regress('t', 'target', features := ['f1','f2','f3','f4','f5','f6','f7','f8'], model := 'nori-1-5');
CREATE TABLE warm_cpu_ms AS SELECT epoch_ms(now()) - (SELECT t0 FROM warm2) AS ms;
SELECT 'AGREE_REG', count(*) AS n, max(abs(c.yhat - g.yhat)) AS max_abs, corr(c.yhat, g.yhat) AS corr,
       max(abs(c.yhat - g.yhat)) FILTER (WHERE NOT c.is_training) AS max_abs_query, max(abs(c.yhat)) AS scale
FROM cpu_r c JOIN gpu_r g USING (rid);
SELECT 'R2_GPU_CTX', corr(g.yhat, t.target) FROM gpu_r g JOIN t USING (rid) WHERE g.is_training;
.timer off
.headers off
.mode list
SELECT 'VERDICT ' || CASE
  WHEN (SELECT dev FROM served) NOT LIKE 'rocm%' THEN 'FAIL: the GPU side was served by ' || (SELECT dev FROM served) || ' (not rocm): every number above is cpu-vs-cpu'
  WHEN (SELECT count(*) FROM gpu_r) <> (SELECT count(*) FROM cpu_r) OR (SELECT count(*) FROM gpu_r) = 0 THEN 'FAIL: row counts differ'
  WHEN (SELECT max(abs(c.yhat - g.yhat)) / nullif(max(abs(c.yhat)), 0) FROM cpu_r c JOIN gpu_r g USING (rid)) > 1e-2 THEN 'FAIL: ROCm disagrees with the CPU (max relative diff ' || (SELECT max(abs(c.yhat - g.yhat)) / nullif(max(abs(c.yhat)), 0) FROM cpu_r c JOIN gpu_r g USING (rid)) || ')'
  ELSE 'PASS: nori-1-5 served by ' || (SELECT dev FROM served) || ', max relative diff vs CPU ' || (SELECT max(abs(c.yhat - g.yhat)) / nullif(max(abs(c.yhat)), 0) FROM cpu_r c JOIN gpu_r g USING (rid)) || ', warm ' || (SELECT ms FROM warm_gpu_ms) || ' ms on rocm vs ' || (SELECT ms FROM warm_cpu_ms) || ' ms on cpu'
END;
