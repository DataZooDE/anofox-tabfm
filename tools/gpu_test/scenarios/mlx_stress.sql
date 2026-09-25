-- Apple MLX stress + regression scenario (docs/MLX_PLAN.md M1).
--
-- The small end-to-end check proves routing. This one is here to break things,
-- at sizes a user would actually reach, and it is written to catch the specific
-- failure modes the GPU hardening work paid for once already
-- (docs/GPU_HARDENING_PLAN.md) rather than to produce a green tick:
--
--   * every stage prints *_SERVED_BY. "cpu and mlx agree" is also what a silent
--     CPU fallback prints, so agreement is worthless without the device column.
--   * device and precision are switched BACK AND FORTH mid-session, because the
--     bug (70a6800) was a cached session outliving the setting that built it,
--     and it is invisible unless you alternate.
--   * wide, tall, padded, single-class and constant-feature tables, since the
--     quantile embedding divides by a standard deviation that a constant
--     column drives to zero.
--
-- Run:
--   duckdb -unsigned < tools/gpu_test/scenarios/mlx_stress.sql
-- with anofox_tabfm_ep_path pointing at the built plugin.

LOAD anofox_tabfm;

.print '=== 0. environment ==='
SELECT device_id, name, usable FROM tabfm_devices() ORDER BY device_id;

-- A learnable signal, so the model is deciding something rather than guessing.
-- 4000 rows x 30 features is a realistic analytics table and well past the
-- 2500-row point where S-M4 measured ORT CPU taking three minutes.
CREATE TABLE big AS
SELECT
    i                                              AS row_id,
    sin(i * 0.013)                                 AS f1,
    cos(i * 0.007)                                 AS f2,
    (i % 97) / 97.0                                AS f3,
    ln(1 + (i % 501))                              AS f4,
    ((i * 7919) % 1000) / 1000.0                   AS f5,
    3.14159                                        AS f_const,   -- zero variance
    CASE WHEN i % 11 = 0 THEN NULL ELSE (i % 53) / 53.0 END AS f_sparse,
    CASE
        WHEN sin(i * 0.013) + 0.5 * cos(i * 0.007) >  0.6 THEN 'high'
        WHEN sin(i * 0.013) + 0.5 * cos(i * 0.007) > -0.2 THEN 'mid'
        ELSE 'low'
    END                                            AS label,
    sin(i * 0.013) * 10 + cos(i * 0.007) * 3       AS target
FROM range(4000) t(i);

-- Split by a HASH of row_id, not by position. A positional split on a series
-- built from sin/cos of the row index hands the model a context drawn from one
-- phase region and a query set from another, so accuracy collapses toward
-- chance and the "better than guessing" guard below stops guarding anything.
-- (Measured: 0.422 against a 0.333 baseline on the positional split. The
-- cpu/mlx agreement was still exact -- which is the point: agreement is
-- insensitive to whether the task is learnable, so the accuracy check has to
-- be independently meaningful or it is decoration.)
-- One context table PER TASK, and this is not cosmetic.
--
-- `qry` drops both label columns, so a single `ctx` carrying both means the
-- classification call sees a `target` the query side lacks and the regression
-- call sees a `label` it lacks. Since the feature-column guard landed, that is
-- a hard error -- every call in this file failed at the first block, which is
-- why this scenario had silently stopped running.
--
-- Dropping the other task's label here, rather than naming `features := [...]`
-- at each of the call sites below, keeps the feature set derived from the table
-- instead of restated beside it: adding a feature above cannot then quietly
-- narrow what these scenarios actually exercise.
CREATE TABLE ctx_c AS SELECT * EXCLUDE (target) FROM big WHERE hash(row_id) % 100 <  62;
CREATE TABLE ctx_r AS SELECT * EXCLUDE (label)  FROM big WHERE hash(row_id) % 100 <  62;
CREATE TABLE qry  AS SELECT * EXCLUDE (label, target) FROM big WHERE hash(row_id) % 100 >= 62;
CREATE TABLE act  AS SELECT row_id, label, target FROM big WHERE hash(row_id) % 100 >= 62;

.print ''
.print '=== 1. classification at scale: cpu reference ==='
SET anofox_tabfm_device = 'cpu';
.timer on
CREATE TABLE c_cpu AS
SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
.timer off
SELECT 'CPU_SERVED_BY' AS marker, model, device FROM tabfm_models() WHERE loaded;

.print ''
.print '=== 2. classification at scale: mlx ==='
SET anofox_tabfm_device = 'mlx';
.timer on
CREATE TABLE c_mlx AS
SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
.timer off
SELECT 'MLX_SERVED_BY' AS marker, model, device, bytes FROM tabfm_models() WHERE loaded;

.print ''
.print '-- a device switch must not change the answer (fp32) --'
SELECT 'CLS_AGREEMENT' AS marker,
       count(*)                                        AS n,
       count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) AS disagreements
FROM c_cpu a JOIN c_mlx b USING (row_id);

.print '-- and both must be better than guessing, or agreement proves nothing --'
SELECT 'CLS_ACCURACY' AS marker,
       avg((c.yhat = act.label)::INT)                       AS mlx_accuracy,
       -- the majority-class rate is the bar a useless model clears
       (SELECT max(share) FROM (
            SELECT count(*) * 1.0 / sum(count(*)) OVER () AS share
            FROM act GROUP BY label))                       AS baseline
FROM c_mlx c JOIN act USING (row_id);

.print ''
.print '=== 3. alternate devices mid-session (the 70a6800 bug) ==='
-- Each switch must be honoured on the NEXT call, not on the next session load.
SET anofox_tabfm_device = 'cpu';
CREATE TABLE alt1 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
SELECT 'ALT_CPU_SERVED_BY' AS marker, string_agg(DISTINCT device, ',') AS devices FROM tabfm_models() WHERE loaded;
SET anofox_tabfm_device = 'mlx';
CREATE TABLE alt2 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
SELECT 'ALT_MLX_SERVED_BY' AS marker, string_agg(DISTINCT device, ',') AS devices FROM tabfm_models() WHERE loaded;
SELECT 'ALT_STABLE' AS marker,
       count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) AS drift
FROM alt1 a JOIN alt2 b USING (row_id);

.print ''
.print '=== 4. regression at scale ==='
SET anofox_tabfm_device = 'cpu';
CREATE TABLE r_cpu AS
SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'target', test := 'qry', model := 'mitra');
SELECT 'REG_CPU_SERVED_BY' AS marker, model, device FROM tabfm_models() WHERE loaded AND device = 'cpu';

SET anofox_tabfm_device = 'mlx';
CREATE TABLE r_mlx AS
SELECT row_id, yhat FROM tabfm_regress('ctx_r', 'target', test := 'qry', model := 'mitra');
SELECT 'REG_MLX_SERVED_BY' AS marker, model, device FROM tabfm_models() WHERE loaded AND device LIKE 'mlx%';

SELECT 'REG_AGREEMENT' AS marker,
       count(*)                              AS n,
       max(abs(a.yhat - b.yhat))             AS max_abs_diff,
       corr(a.yhat, b.yhat)                  AS correlation
FROM r_cpu a JOIN r_mlx b USING (row_id);

.print ''
.print '=== 5. reduced precision: class agreement, per the contract ==='
SET anofox_tabfm_device = 'mlx';
SET anofox_tabfm_gpu_precision = 'bf16';
CREATE TABLE c_bf16 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
SET anofox_tabfm_gpu_precision = 'fp16';
CREATE TABLE c_fp16 AS SELECT row_id, yhat FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
SET anofox_tabfm_gpu_precision = 'fp32';

SELECT 'FLIPS_BF16' AS marker, count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) AS flips, count(*) AS n
FROM c_mlx a JOIN c_bf16 b USING (row_id);
SELECT 'FLIPS_FP16' AS marker, count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) AS flips, count(*) AS n
FROM c_mlx a JOIN c_fp16 b USING (row_id);
-- Sessions are cached per (model, device, precision): each mode above is its
-- own entry, so switching back to fp32 must not have evicted the others.
SELECT 'SESSION_COUNT' AS marker, count(*) AS n FROM tabfm_models() WHERE loaded;

.print ''
.print '=== 6. awkward shapes ==='
-- Single class in the context: softmax over one attested label.
CREATE TABLE one_class AS SELECT row_id, f1, f2, f3, 'only' AS label FROM big WHERE row_id < 200;
CREATE TABLE one_q     AS SELECT row_id, f1, f2, f3 FROM big WHERE row_id >= 200 AND row_id < 260;
SELECT 'ONE_CLASS' AS marker, count(DISTINCT yhat) AS distinct_preds, count(*) AS n
FROM tabfm_classify('one_class', 'label', test := 'one_q', model := 'mitra');

-- Constant feature -> zero variance in the quantile embedding (a division the
-- reference implementation guards with a where(var==0, 0, x); if the port lost
-- that guard this returns NaN and the count below drops).
CREATE TABLE const_ctx AS SELECT row_id, f_const, f1, label FROM big WHERE row_id < 300;
CREATE TABLE const_q   AS SELECT row_id, f_const, f1 FROM big WHERE row_id >= 300 AND row_id < 360;
SELECT 'CONST_FEATURE' AS marker, count(*) AS n, count(yhat) AS non_null
FROM tabfm_classify('const_ctx', 'label', test := 'const_q', model := 'mitra');

-- Sparse feature with NULLs.
CREATE TABLE sparse_ctx AS SELECT row_id, f_sparse, f1, f2, label FROM big WHERE row_id < 300;
CREATE TABLE sparse_q   AS SELECT row_id, f_sparse, f1, f2 FROM big WHERE row_id >= 300 AND row_id < 360;
SELECT 'SPARSE_NULLS' AS marker, count(*) AS n, count(yhat) AS non_null
FROM tabfm_classify('sparse_ctx', 'label', test := 'sparse_q', model := 'mitra');

.print ''
.print '=== 7. error contracts ==='
-- This stage USED to assert that an unimplemented model was refused. That
-- contract no longer exists: the backend runs the model's own ONNX graph, so
-- every registered model is served (tools/gpu_test/scenarios/mlx_all_models.sql
-- checks all of them). Left as a marker named for what it now proves, because
-- the old version returned 1484 rows under the name `should_not_reach_here` --
-- a passing check whose label said the opposite.
SET anofox_tabfm_device = 'mlx';
SELECT 'OTHER_MODEL_SERVED_ON_MLX' AS marker, count(*) AS n
FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'tabpfn-v2');
SELECT 'OTHER_MODEL_SERVED_BY' AS marker, string_agg(DISTINCT device, ',') AS devices
FROM tabfm_models() WHERE loaded AND model = 'tabpfn-v2';

-- What IS still a refusal: a device named but no plugin to load. It must name
-- the fix and must never fall back to CPU.
.print '-- ep_path unset must error, not silently serve cpu --'
SET anofox_tabfm_ep_path = '';
SELECT 'NO_PLUGIN_MUST_ERROR' AS marker, count(*) AS should_not_reach_here
FROM tabfm_classify('ctx_c', 'label', test := 'qry', model := 'mitra');
