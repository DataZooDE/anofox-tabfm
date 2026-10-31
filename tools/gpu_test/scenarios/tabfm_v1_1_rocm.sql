-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
--
-- Google TabFM 1.1.0 (`tabfm-v1-1`) on ROCm with its REAL weights, against the CPU on the same split
-- test/sql/tabfm_real_models.test uses. Needs the weights staged first (ungated repo, non-commercial
-- licence, 0.82 GB per task):
--   SET anofox_tabfm_accept_hf_license = true;
--   CALL tabfm_download('classification', model := 'tabfm-v1-1');
--   CALL tabfm_download('regression', model := 'tabfm-v1-1');
--
-- The 1.1 checkpoint is BF16, so its MIGraphX graph retypes every initializer to BF16 and Casts it to
-- FLOAT in-graph (tools/make_migraphx_graph.py retype_external) -- the one thing about this model
-- the v1 ROCm path never exercised. A misread here predicts at chance and still "runs", so the
-- assertions are GPU accuracy and GPU-vs-CPU agreement, each next to *_SERVED_BY (CLAUDE.md: never
-- trust "cpu and gpu agree", a silent CPU fallback prints that too). fp32 is the default precision.
-- .timer on shows the cold MIGraphX compile (once per shape bucket; .mxr-cached afterwards).
LOAD anofox_tabfm;
SET anofox_tabfm_accept_hf_license = true;

CREATE TABLE base AS
SELECT i AS row_id,
       sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
       CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) >  0.3 THEN 'a'
            WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END AS y,
       sin(i * 0.11) * 5 + cos(i * 0.07) * 2 AS tgt
FROM range(240) t(i);
CREATE TABLE ctx_cls AS SELECT row_id, f1, f2, f3, f4, y FROM base WHERE hash(row_id) % 100 < 70;
CREATE TABLE ctx_reg AS SELECT row_id, f1, f2, f3, f4, tgt FROM base WHERE hash(row_id) % 100 < 70;
CREATE TABLE qry AS SELECT row_id, f1, f2, f3, f4 FROM base WHERE hash(row_id) % 100 >= 70;
CREATE TABLE truth AS SELECT row_id, y, tgt FROM base WHERE hash(row_id) % 100 >= 70;

.timer on
SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu AS SELECT row_id, yhat, yhat_score FROM tabfm_classify('ctx_cls', 'y', test := 'qry', model := 'tabfm-v1-1');
CREATE TABLE r_cpu AS SELECT row_id, yhat FROM tabfm_regress('ctx_reg', 'tgt', test := 'qry', model := 'tabfm-v1-1');
.mode list
SELECT 'V11_CPU_SERVED_BY=' || coalesce(string_agg(DISTINCT device, ','), 'NONE') FROM tabfm_models() WHERE loaded AND model = 'tabfm-v1-1';
.mode duckbox

SET anofox_tabfm_device = 'rocm';
CREATE TABLE c_gpu AS SELECT row_id, yhat, yhat_score FROM tabfm_classify('ctx_cls', 'y', test := 'qry', model := 'tabfm-v1-1');
CREATE TABLE r_gpu AS SELECT row_id, yhat FROM tabfm_regress('ctx_reg', 'tgt', test := 'qry', model := 'tabfm-v1-1');
-- warm: the programs are compiled (and .mxr-cached) now
CREATE TABLE c_gpu2 AS SELECT row_id, yhat FROM tabfm_classify('ctx_cls', 'y', test := 'qry', model := 'tabfm-v1-1');
.timer off
.mode list
SELECT 'V11_GPU_SERVED_BY=' || coalesce(string_agg(DISTINCT device, ','), 'NONE')
FROM tabfm_models() WHERE loaded AND model = 'tabfm-v1-1' AND device LIKE 'rocm%';
SELECT 'V11_ROWS=' || count(*) FROM c_gpu;
SELECT 'V11_CPU_ACC=' || round(avg((c.yhat = t.y)::INT), 4) FROM c_cpu c JOIN truth t USING (row_id);
SELECT 'V11_GPU_ACC=' || round(avg((c.yhat = t.y)::INT), 4) FROM c_gpu c JOIN truth t USING (row_id);
SELECT 'V11_LABEL_DISAGREE=' || count(*) FILTER (WHERE c.yhat <> g.yhat) || '/' || count(*)
FROM c_cpu c JOIN c_gpu g USING (row_id);
SELECT 'V11_SCORE_MAX_ABS_DIFF=' || round(max(abs(c.yhat_score - g.yhat_score)), 6) FROM c_cpu c JOIN c_gpu g USING (row_id);
SELECT 'V11_GPU_REPEAT_DISAGREE=' || count(*) FILTER (WHERE a.yhat <> b.yhat) FROM c_gpu a JOIN c_gpu2 b USING (row_id);
SELECT 'V11_REG_CPU_R2=' || round(1 - sum(power(r.yhat - t.tgt, 2)) / sum(power(t.tgt - (SELECT avg(tgt) FROM truth), 2)), 5)
FROM r_cpu r JOIN truth t USING (row_id);
SELECT 'V11_REG_GPU_R2=' || round(1 - sum(power(r.yhat - t.tgt, 2)) / sum(power(t.tgt - (SELECT avg(tgt) FROM truth), 2)), 5)
FROM r_gpu r JOIN truth t USING (row_id);
SELECT 'V11_REG_MAX_ABS_DIFF=' || round(max(abs(c.yhat - g.yhat)), 6) FROM r_cpu c JOIN r_gpu g USING (row_id);
SELECT 'V11_REG_CORR=' || round(corr(c.yhat, g.yhat), 6) FROM r_cpu c JOIN r_gpu g USING (row_id);
.mode duckbox
