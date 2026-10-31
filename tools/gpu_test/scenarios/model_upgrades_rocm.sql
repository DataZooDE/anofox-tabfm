-- mitra-v2 and tabdpt-v1-3 on a GPU, against the CPU, on the real weights.
--
-- Run on a machine with a real GPU and its plugin installed:
--   SET anofox_tabfm_ep_path='<dir>';  -- prepended by the harness
-- Weights: CALL tabfm_download('classification'|'regression', model := 'mitra-v2' | 'tabdpt-v1-3').
--
-- Why these two need their own check: mitra-v2's CLASSIFIER runs mitra's MIGraphX graph on
-- different weights (byte-identical safetensors header), so the .mxr cache must key on the weights
-- too, not just the graph (tabfm_mxr_cache_key.hpp) -- otherwise ROCm would silently serve v1's
-- compiled program. tabdpt-v1-3 is a masked re-export of a changed architecture. Never trust
-- "cpu and gpu agree" (CLAUDE.md): the *_SERVED_BY lines are the evidence; a silent CPU
-- fallback prints agreement too.

CREATE TABLE base AS SELECT i AS row_id, sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
 CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > 0.3 THEN 'a' WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END AS y,
 sin(i * 0.11) * 5 + cos(i * 0.07) * 2 AS tgt FROM range(240) t(i);
CREATE TABLE ctx_cls AS SELECT row_id, f1, f2, f3, f4, y FROM base WHERE hash(row_id) % 100 < 70;
CREATE TABLE ctx_reg AS SELECT row_id, f1, f2, f3, f4, tgt FROM base WHERE hash(row_id) % 100 < 70;
CREATE TABLE qry AS SELECT row_id, f1, f2, f3, f4 FROM base WHERE hash(row_id) % 100 >= 70;
CREATE TABLE truth AS SELECT row_id, y, tgt FROM base WHERE hash(row_id) % 100 >= 70;

SET anofox_tabfm_device = 'cpu';
CREATE TABLE cpu_c AS SELECT 'mitra-v2' m, row_id, yhat FROM tabfm_classify('ctx_cls','y',test:='qry',model:='mitra-v2')
  UNION ALL SELECT 'tabdpt-v1-3', row_id, yhat FROM tabfm_classify('ctx_cls','y',test:='qry',model:='tabdpt-v1-3');
CREATE TABLE cpu_r AS SELECT 'mitra-v2' m, row_id, yhat FROM tabfm_regress('ctx_reg','tgt',test:='qry',model:='mitra-v2')
  UNION ALL SELECT 'tabdpt-v1-3', row_id, yhat FROM tabfm_regress('ctx_reg','tgt',test:='qry',model:='tabdpt-v1-3');

SET anofox_tabfm_device = 'rocm';
CREATE TABLE gpu_c AS SELECT 'mitra-v2' m, row_id, yhat FROM tabfm_classify('ctx_cls','y',test:='qry',model:='mitra-v2')
  UNION ALL SELECT 'tabdpt-v1-3', row_id, yhat FROM tabfm_classify('ctx_cls','y',test:='qry',model:='tabdpt-v1-3');
CREATE TABLE gpu_r AS SELECT 'mitra-v2' m, row_id, yhat FROM tabfm_regress('ctx_reg','tgt',test:='qry',model:='mitra-v2')
  UNION ALL SELECT 'tabdpt-v1-3', row_id, yhat FROM tabfm_regress('ctx_reg','tgt',test:='qry',model:='tabdpt-v1-3');

SELECT upper(replace(model, '-', '_')) || '_' || upper(task) || '_SERVED_BY=' || device
FROM tabfm_models() WHERE loaded AND model IN ('mitra-v2', 'tabdpt-v1-3') AND device <> 'cpu' ORDER BY ALL;

SELECT 'CLS_AGREE ' || c.m || ' agree=' || round(avg((c.yhat = g.yhat)::INT), 4) || ' gpu_acc=' ||
       round(avg((g.yhat = t.y)::INT), 4) || ' cpu_acc=' || round(avg((c.yhat = t.y)::INT), 4)
FROM cpu_c c JOIN gpu_c g USING (m, row_id) JOIN truth t USING (row_id) GROUP BY c.m ORDER BY c.m;

SELECT 'REG_AGREE ' || c.m || ' maxabs=' || max(abs(c.yhat - g.yhat)) || ' corr=' || corr(c.yhat, g.yhat)
FROM cpu_r c JOIN gpu_r g USING (m, row_id) GROUP BY c.m ORDER BY c.m;

-- mitra-v2 classification on the GPU must NOT be mitra's program: same graph, different weights.
SET anofox_tabfm_device = 'rocm';
SELECT 'MITRA_V1_V2_GPU_DIFFER=' || (max(abs(a.proba['a'] - b.proba['a'])) > 1e-4)::VARCHAR
FROM (SELECT row_id, proba FROM tabfm_classify('ctx_cls','y',test:='qry',model:='mitra')) a
JOIN (SELECT row_id, proba FROM tabfm_classify('ctx_cls','y',test:='qry',model:='mitra-v2')) b USING (row_id);
