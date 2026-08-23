-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
LOAD anofox_tabfm;
SET anofox_tabfm_default_model='tabfm-v1';
SET anofox_tabfm_gpu_precision='bf16';
CREATE TABLE reg AS
SELECT i AS id, (i*37%100)/100.0 AS f1, (i*53%100)/100.0 AS f2, (i*71%100)/100.0 AS f3,
       (i*13%100)/100.0 AS f4, (i*29%100)/100.0 AS f5, (i*47%100)/100.0 AS f6,
       (i*61%100)/100.0 AS f7, (i*17%100)/100.0 AS f8,
       CASE WHEN i<70 THEN ((i*37%100) + (i*53%100)) / 200.0 ELSE NULL END AS target
FROM range(100) t(i);
SET anofox_tabfm_device='cpu';
CREATE TABLE r_cpu AS SELECT id, yhat FROM tabfm_regress('reg','target');
.mode list
SELECT 'REG_CPU_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded AND model='tabfm-v1';
.mode duckbox
SET anofox_tabfm_device='rocm';
CREATE TABLE r_gpu AS SELECT id, yhat FROM tabfm_regress('reg','target');
.mode list
SELECT 'REG_GPU_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded AND model='tabfm-v1';
SELECT 'REG_ROWS=' || count(*) FROM r_gpu;
SELECT 'REG_MAX_ABS_DIFF=' || round(max(abs(c.yhat - g.yhat))::DOUBLE, 6)
  FROM r_cpu c JOIN r_gpu g USING (id);
SELECT 'REG_CORR=' || round(corr(c.yhat, g.yhat)::DOUBLE, 6) FROM r_cpu c JOIN r_gpu g USING (id);
.mode duckbox
