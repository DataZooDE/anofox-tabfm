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
-- tabdpt appears only in the refusal check at the end, and needs only the
-- download -- no conversion step:
--
--   CALL tabfm_download('classification', model := 'tabdpt');
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

.print ''
.print '=== every model, cpu vs mlx ==='
SELECT * FROM report ORDER BY model, task;

.print ''
.print '-- classification must agree EXACTLY; regression correlates --'
SELECT 'MODELS_DISAGREEING' AS marker,
       count(*) FILTER (WHERE task = 'classification' AND agreement < 1.0) AS classification,
       count(*) FILTER (WHERE task = 'regression' AND max_abs_diff > 1e-3)  AS regression
FROM report;

-- tabdpt is NOT servable on MLX, and that is the assertion here.
--
-- Worth stating plainly because the obvious reading is wrong. tabdpt looks
-- like it belongs in this file: MLX and CUDA both consume the model's
-- `ext_graph` (GpuGraphKindFor), and tabdpt has one. But the MLX backend runs
-- that graph through its OWN interpreter rather than ONNX Runtime, and the
-- interpreter has no `ConstantOfShape` -- which tabdpt's graph uses 32 times.
--
-- Measured, not assumed, on an M3 (macOS 27.0) against the re-exported graphs:
--
--   Invalid Input Error: anofox_tabfm: the 'mlx' backend could not be
--   initialised: ... this graph needs ONNX ops the mlx backend does not
--   implement (ConstantOfShape). SET anofox_tabfm_device='cpu' ...
--
-- The op count is 32 in the graph on `main` AND 32 after the fitted-values
-- re-export, so this is a pre-existing gap, not something the tabdpt work
-- changed. It is why this file never listed tabdpt: an incapability, not an
-- oversight.
--
-- Kept as an error contract because the refusal is the valuable behaviour. An
-- op the interpreter does not implement must raise and name the op, NOT quietly
-- produce numbers on the CPU -- "cpu and gpu agree" is exactly what a silent
-- fallback prints (CLAUDE.md). Run this last: it is expected to RAISE, and the
-- CLI stops here.
.print '--- tabdpt on mlx must REFUSE, naming the op (expected to raise) ---'
SET anofox_tabfm_device = 'mlx';
SELECT 'TABDPT_MLX_SHOULD_NOT_REACH_HERE' AS marker, count(*) AS n
FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := 'tabdpt');
