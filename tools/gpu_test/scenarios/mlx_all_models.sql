-- Every registered model on MLX, checked against the CPU path (docs/MLX_PLAN.md).
--
-- The MLX backend runs the model's own ONNX graph (src/tabfm_mlx_graph.cpp), so
-- coverage is a property of the interpreter's op table rather than of a
-- per-model list. That is exactly why this file exists: an op-table change can
-- break one model and no other, and nothing else in the suite would notice.
--
-- Requires the converted checkpoints. They are NOT optional and NOT
-- MLX-specific -- the .ckpt releases do not load on any backend, cpu included,
-- until converted:
--
--   CALL tabfm_download('classification', model := 'tabpfn-v2');   -- etc.
--   cd tools/export_tabpfn   && uv run python convert_weights.py classification
--   cd tools/export_tabpfn   && uv run python convert_weights.py classification --arch=v2.5
--   cd tools/export_tabpfn   && uv run python convert_weights.py classification --arch=v3
--   cd tools/export_tabicl   && uv run python convert_weights.py classification
--   cd tools/export_orion_bix && uv run python convert_weights.py
--
-- tabdpt needs only the download -- no conversion step:
--
--   CALL tabfm_download('classification', model := 'tabdpt');
--   CALL tabfm_download('regression',     model := 'tabdpt');
--
-- It is here because MLX consumes the same `ext_graph` the CUDA plugin does
-- (GpuGraphKindFor), and tabdpt's ext graphs were re-exported for the fitted-
-- values fix. CUDA was re-verified on those graphs directly; MLX reads them
-- through its OWN interpreter (src/tabfm_mlx_graph.cpp), so agreement there is
-- a separate claim and has to be measured separately.
--
-- Run with anofox_tabfm_ep_path pointing at the built plugin.

INSTALL httpfs; LOAD httpfs;
LOAD anofox_tabfm;
SET anofox_tabfm_accept_hf_license = true;

CREATE TABLE base AS
SELECT i AS row_id, sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
       CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) >  0.3 THEN 'a'
            WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END AS y,
       sin(i * 0.11) * 5 + cos(i * 0.07) * 2 AS tgt
FROM range(240) t(i);
-- Hash split, not positional: a positional split of a sin/cos series puts
-- context and query in different phase regions, which drives accuracy to chance
-- while leaving cpu/mlx agreement untouched. Agreement is insensitive to
-- whether the task is learnable; anything else here would not be.
-- One context table PER TASK, and this is not cosmetic.
--
-- `qry` drops both label columns, so a single `ctx` carrying both means the
-- classification call sees a `tgt` the query side lacks and the regression
-- call sees a `y` it lacks. Since the feature-column guard landed, that is
-- a hard error -- every call in this file failed at the first block, which is
-- why this scenario had silently stopped running.
--
-- Dropping the other task's label here, rather than naming `features := [...]`
-- at each of the call sites below, keeps the feature set derived from the table
-- instead of restated beside it: adding a feature above cannot then quietly
-- narrow what these scenarios actually exercise.
CREATE TABLE ctx_c AS SELECT * EXCLUDE (tgt) FROM base WHERE hash(row_id) % 100 <  70;
CREATE TABLE ctx_r AS SELECT * EXCLUDE (y)   FROM base WHERE hash(row_id) % 100 <  70;
CREATE TABLE qry AS SELECT * EXCLUDE (y, tgt) FROM base WHERE hash(row_id) % 100 >= 70;

CREATE TABLE report(model VARCHAR, task VARCHAR, mlx_device VARCHAR,
                    agreement DOUBLE, max_abs_diff DOUBLE, n BIGINT);

-- Each model gets its OWN result tables. Reusing one name lets a failed MLX run
-- leave the previous model's rows in place, and the comparison then silently
-- scores the wrong pair -- which is how an earlier version of this file
-- reported 0.6875 for two models that had in fact errored outright.

.print '--- mitra: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_mitra AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'mitra');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_mitra AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'mitra');
INSERT INTO report SELECT 'mitra', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_mitra a JOIN c_mlx_mitra b USING (row_id);

.print '--- tabpfn-v2: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_tabpfnv2 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabpfn-v2');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_tabpfnv2 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabpfn-v2');
INSERT INTO report SELECT 'tabpfn-v2', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_tabpfnv2 a JOIN c_mlx_tabpfnv2 b USING (row_id);

.print '--- tabpfn-v2-5: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_tabpfnv25 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabpfn-v2-5');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_tabpfnv25 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabpfn-v2-5');
INSERT INTO report SELECT 'tabpfn-v2-5', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_tabpfnv25 a JOIN c_mlx_tabpfnv25 b USING (row_id);

.print '--- tabpfn-v3: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_tabpfnv3 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabpfn-v3');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_tabpfnv3 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabpfn-v3');
INSERT INTO report SELECT 'tabpfn-v3', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_tabpfnv3 a JOIN c_mlx_tabpfnv3 b USING (row_id);

.print '--- tabicl-v2: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_tabiclv2 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabicl-v2');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_tabiclv2 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabicl-v2');
INSERT INTO report SELECT 'tabicl-v2', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_tabiclv2 a JOIN c_mlx_tabiclv2 b USING (row_id);

.print '--- orion-bix: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_orionbix AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'orion-bix');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_orionbix AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'orion-bix');
INSERT INTO report SELECT 'orion-bix', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_orionbix a JOIN c_mlx_orionbix b USING (row_id);

.print '--- tabdpt: classification ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu_tabdpt AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabdpt');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE c_mlx_tabdpt AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabdpt');
INSERT INTO report SELECT 'tabdpt', 'classification',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    avg((a.yhat = b.yhat)::INT), NULL, count(*)
FROM c_cpu_tabdpt a JOIN c_mlx_tabdpt b USING (row_id);

.print '--- mitra: regression ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE r_cpu_mitra AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'mitra');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE r_mlx_mitra AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'mitra');
INSERT INTO report SELECT 'mitra', 'regression',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    corr(a.yhat, b.yhat), max(abs(a.yhat - b.yhat)), count(*)
FROM r_cpu_mitra a JOIN r_mlx_mitra b USING (row_id);

.print '--- tabpfn-v2: regression ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE r_cpu_tabpfnv2 AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabpfn-v2');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE r_mlx_tabpfnv2 AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabpfn-v2');
INSERT INTO report SELECT 'tabpfn-v2', 'regression',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    corr(a.yhat, b.yhat), max(abs(a.yhat - b.yhat)), count(*)
FROM r_cpu_tabpfnv2 a JOIN r_mlx_tabpfnv2 b USING (row_id);

.print '--- tabpfn-v2-5: regression ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE r_cpu_tabpfnv25 AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabpfn-v2-5');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE r_mlx_tabpfnv25 AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabpfn-v2-5');
INSERT INTO report SELECT 'tabpfn-v2-5', 'regression',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    corr(a.yhat, b.yhat), max(abs(a.yhat - b.yhat)), count(*)
FROM r_cpu_tabpfnv25 a JOIN r_mlx_tabpfnv25 b USING (row_id);

.print '--- tabicl-v2: regression ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE r_cpu_tabiclv2 AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabicl-v2');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE r_mlx_tabiclv2 AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabicl-v2');
INSERT INTO report SELECT 'tabicl-v2', 'regression',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    corr(a.yhat, b.yhat), max(abs(a.yhat - b.yhat)), count(*)
FROM r_cpu_tabiclv2 a JOIN r_mlx_tabiclv2 b USING (row_id);

-- tabpfn-v3 REGRESSION is absent on purpose: its released checkpoint carries no
-- FullSupportBarDistribution criterion.borders, so convert_weights.py cannot
-- build the point-estimate head. It fails on cpu too -- not an MLX gap.

.print '--- tabdpt: regression ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE r_cpu_tabdpt AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabdpt');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE r_mlx_tabdpt AS SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := 'tabdpt');
INSERT INTO report SELECT 'tabdpt', 'regression',
    (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%'),
    corr(a.yhat, b.yhat), max(abs(a.yhat - b.yhat)), count(*)
FROM r_cpu_tabdpt a JOIN r_mlx_tabdpt b USING (row_id);

-- tabdpt's FITTED rows, which the rest of this file cannot see.
--
-- Every block above compares query rows only. That was sufficient while the
-- exported head ran over query rows alone and the wrapper zero-padded the
-- context rows. It no longer is: the fitted-values fix made the head run over
-- EVERY data row, so the graph's output went from (T-S, B, O) to (T, B, O) and
-- the context rows carry real in-context predictions instead of zeros.
--
-- Those newly-live rows are the entire behavioural change, and a query-row
-- comparison is blind to them by construction -- cpu and mlx would agree
-- perfectly on the query slice while disagreeing on every fitted row. Hence a
-- second comparison over ALL rows, with no test table, where is_training rows
-- are returned.
.print '--- tabdpt: fitted values (all rows, cpu vs mlx) ---'
SET anofox_tabfm_device = 'cpu';
CREATE TABLE f_cpu_tabdpt AS
SELECT row_id, yhat, is_training FROM tabfm_classify('ctx_c', 'y', model := 'tabdpt');
SET anofox_tabfm_device = 'mlx';
CREATE TABLE f_mlx_tabdpt AS
SELECT row_id, yhat, is_training FROM tabfm_classify('ctx_c', 'y', model := 'tabdpt');

SELECT 'TABDPT_FITTED' AS marker,
       (SELECT string_agg(DISTINCT device, ',') FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%') AS mlx_device,
       count(*) AS n_rows,
       count(*) FILTER (WHERE a.is_training) AS n_fitted,
       -- A constant column is what the defect produced; > 1 distinct is the
       -- floor, not the proof. The agreement column is the proof.
       (SELECT count(DISTINCT yhat) FROM f_mlx_tabdpt WHERE is_training) AS mlx_fitted_distinct,
       count(*) FILTER (WHERE a.yhat <> b.yhat) AS ROWS_DISAGREEING
FROM f_cpu_tabdpt a JOIN f_mlx_tabdpt b USING (row_id);

.print ''
.print '=== every model, cpu vs mlx ==='
SELECT * FROM report ORDER BY model, task;

.print ''
.print '-- classification must agree EXACTLY; regression correlates --'
SELECT 'MODELS_DISAGREEING' AS marker,
       count(*) FILTER (WHERE task = 'classification' AND agreement < 1.0) AS classification,
       count(*) FILTER (WHERE task = 'regression' AND max_abs_diff > 1e-3)  AS regression
FROM report;
