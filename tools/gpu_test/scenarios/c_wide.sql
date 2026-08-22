-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
LOAD anofox_tabfm;
SET anofox_tabfm_default_model='tabfm-v1';
SET anofox_tabfm_gpu_precision='bf16';
CREATE TABLE wide AS SELECT i AS id,
  (i*3%100)/100.0 c1,(i*5%100)/100.0 c2,(i*7%100)/100.0 c3,(i*11%100)/100.0 c4,
  (i*13%100)/100.0 c5,(i*17%100)/100.0 c6,(i*19%100)/100.0 c7,(i*23%100)/100.0 c8,
  (i*29%100)/100.0 c9,(i*31%100)/100.0 c10,(i*37%100)/100.0 c11,(i*41%100)/100.0 c12,
  (i*43%100)/100.0 c13,(i*47%100)/100.0 c14,(i*53%100)/100.0 c15,(i*59%100)/100.0 c16,
  (i*61%100)/100.0 c17,(i*67%100)/100.0 c18,(i*71%100)/100.0 c19,(i*73%100)/100.0 c20,
  CASE WHEN i<70 THEN CASE WHEN (i*37%100)<33 THEN 'a' WHEN (i*37%100)<66 THEN 'b' ELSE 'c' END ELSE NULL END AS label
FROM range(100) t(i);
SET anofox_tabfm_device='cpu';
CREATE TABLE w_cpu AS SELECT id, yhat FROM tabfm_classify('wide','label');
SET anofox_tabfm_device='rocm';
CREATE TABLE w_gpu AS SELECT id, yhat FROM tabfm_classify('wide','label');
.mode list
SELECT 'WIDE_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded AND model='tabfm-v1';
SELECT 'WIDE_FEATURES=20 -> H64 bucket';
SELECT 'WIDE_DISAGREEMENTS=' || count(*) FROM w_cpu c JOIN w_gpu g USING (id)
 WHERE c.yhat IS DISTINCT FROM g.yhat;
.mode duckbox
