-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>'; SET VARIABLE m = 'kumo-tabular-small';
--
-- Kumo-Tabular on ROCm against the CPU, real weights, both tasks, EVERY row (context and query).
-- Needs the weights downloaded AND converted (tools/export_kumo: convert_kumo_weights --size <size>):
-- the bundled migraphx graph references model.safetensors by offset and is only used when its header
-- matches ExpectedWeightsHeaderShaFor. As always, the *_SERVED_BY lines are the assertion: "cpu and gpu
-- agree" is also what a silent CPU fallback prints.
LOAD anofox_tabfm;
CREATE TABLE base AS SELECT i AS row_id, sin(i * 0.11) f1, cos(i * 0.07) f2, (i % 13) / 13.0 f3, ln(1 + (i % 31)) f4,
  CASE WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) >  0.3 THEN 'a'
       WHEN sin(i * 0.11) + 0.5 * cos(i * 0.07) > -0.3 THEN 'b' ELSE 'c' END AS y,
  sin(i * 0.11) * 5 + cos(i * 0.07) * 2 AS tgt FROM range(240) t(i);
CREATE TABLE cls AS SELECT row_id, f1, f2, f3, f4, CASE WHEN hash(row_id) % 100 < 70 THEN y END AS y FROM base;
CREATE TABLE reg AS SELECT row_id, f1, f2, f3, f4, CASE WHEN hash(row_id) % 100 < 70 THEN tgt END AS tgt FROM base;

SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_cpu AS SELECT f1, f2, f3, f4, is_training, yhat FROM tabfm_classify('cls', 'y', features := ['f1','f2','f3','f4'], model := getvariable('m'));
CREATE TABLE r_cpu AS SELECT f1, f2, f3, f4, is_training, yhat FROM tabfm_regress('reg', 'tgt', features := ['f1','f2','f3','f4'], model := getvariable('m'));

SET anofox_tabfm_device = 'rocm';
CREATE TABLE c_gpu AS SELECT f1, f2, f3, f4, is_training, yhat FROM tabfm_classify('cls', 'y', features := ['f1','f2','f3','f4'], model := getvariable('m'));
.mode list
SELECT 'CLS_SERVED_BY=' || coalesce(string_agg(DISTINCT device, ','), 'NONE') FROM tabfm_models() WHERE loaded AND model = getvariable('m') AND device LIKE 'rocm%';
CREATE TABLE r_gpu AS SELECT f1, f2, f3, f4, is_training, yhat FROM tabfm_regress('reg', 'tgt', features := ['f1','f2','f3','f4'], model := getvariable('m'));
SELECT 'REG_SERVED_BY=' || coalesce(string_agg(DISTINCT device, ','), 'NONE') FROM tabfm_models() WHERE loaded AND model = getvariable('m') AND device LIKE 'rocm%';

SELECT 'QUERY_DISAGREE=' || count(*) FILTER (WHERE NOT a.is_training AND a.yhat <> b.yhat)
    || ' CONTEXT_DISAGREE=' || count(*) FILTER (WHERE a.is_training AND a.yhat <> b.yhat) || ' ROWS=' || count(*)
FROM c_cpu a JOIN c_gpu b USING (f1, f2, f3, f4);
SELECT 'REG_MAXDIFF=' || max(abs(a.yhat - b.yhat)) || ' REG_CORR=' || corr(a.yhat, b.yhat)
    || ' FITTED_DISTINCT=' || count(DISTINCT round(b.yhat, 6)) FILTER (WHERE b.is_training)
FROM r_cpu a JOIN r_gpu b USING (f1, f2, f3, f4);
.timer on
SELECT count(*) FROM tabfm_classify('cls', 'y', features := ['f1','f2','f3','f4'], model := getvariable('m'));
SET anofox_tabfm_device = 'cpu';
SELECT count(*) FROM tabfm_classify('cls', 'y', features := ['f1','f2','f3','f4'], model := getvariable('m'));
