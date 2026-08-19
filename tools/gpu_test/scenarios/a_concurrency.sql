-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
LOAD anofox_tabfm;
SET anofox_tabfm_default_model='tabfm-v1';
SET anofox_tabfm_gpu_precision='bf16';
SET threads=8;
CREATE TABLE base AS
SELECT i AS id, (i*37%100)/100.0 AS f1, (i*53%100)/100.0 AS f2, (i*71%100)/100.0 AS f3,
       (i*13%100)/100.0 AS f4, (i*29%100)/100.0 AS f5, (i*47%100)/100.0 AS f6,
       (i*61%100)/100.0 AS f7, (i*17%100)/100.0 AS f8,
       CASE WHEN i<70 THEN CASE WHEN (i*37%100)<33 THEN 'a' WHEN (i*37%100)<66 THEN 'b' ELSE 'c' END ELSE NULL END AS label
FROM range(100) t(i);
CREATE TABLE s1 AS SELECT * FROM base; CREATE TABLE s2 AS SELECT * FROM base;
CREATE TABLE s3 AS SELECT * FROM base; CREATE TABLE s4 AS SELECT * FROM base;
SET anofox_tabfm_device='rocm';
-- sequential reference
CREATE TABLE seq AS SELECT id, yhat FROM tabfm_classify('s1','label');
-- four predicts in one query: DuckDB runs the UNION ALL branches as parallel
-- pipelines, so their finalizes contend for the per-device lock (#42).
CREATE TABLE par AS
SELECT 1 AS b, id, yhat FROM tabfm_classify('s1','label') UNION ALL
SELECT 2, id, yhat FROM tabfm_classify('s2','label') UNION ALL
SELECT 3, id, yhat FROM tabfm_classify('s3','label') UNION ALL
SELECT 4, id, yhat FROM tabfm_classify('s4','label');
.mode list
SELECT 'CONC_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded;
SELECT 'CONC_ROWS=' || count(*) FROM par;
-- every parallel branch must equal the sequential answer; a locking bug here
-- corrupts results rather than erroring, so this is the assertion that matters
SELECT 'CONC_MISMATCHES=' || count(*) FROM par p JOIN seq s USING (id)
 WHERE p.yhat IS DISTINCT FROM s.yhat;
.mode duckbox
