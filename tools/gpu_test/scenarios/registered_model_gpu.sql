-- S4 end-to-end: a REGISTERED model carrying its own migraphx_graph runs on
-- the ROCm plugin through SQL — the dispatch path that did not exist before
-- this change. Weights + graph are the real tabfm ones already in the cache,
-- registered as a user model, so the bundled-graph gate plays no part.
LOAD anofox_tabfm;
SET anofox_tabfm_gpu_precision = 'bf16';

CALL tabfm_register_model(
  id := 'tabfm-as-user-model',
  base_dir := '/home/jr/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main/classification',
  classification_graph := 'graph_ext_classification.onnx',
  classification_migraphx_graph := 'graph_migraphx_classification.onnx',
  classification_weights := 'model.safetensors',
  license := 'tabfm-non-commercial-v1.0', preprocessing_profile := 'tabfm_v1_minimal');

CREATE TABLE churn AS
SELECT i AS id,
       (i * 37 % 100) / 100.0 AS f1, (i * 53 % 100) / 100.0 AS f2,
       (i * 71 % 100) / 100.0 AS f3, (i * 13 % 100) / 100.0 AS f4,
       (i * 29 % 100) / 100.0 AS f5, (i * 47 % 100) / 100.0 AS f6,
       (i * 61 % 100) / 100.0 AS f7, (i * 17 % 100) / 100.0 AS f8,
       CASE WHEN i < 70 THEN CASE WHEN (i * 37 % 100) < 33 THEN 'churn'
                                  WHEN (i * 37 % 100) < 66 THEN 'stay'
                                  ELSE 'upgrade' END
            ELSE NULL END AS segment
FROM range(100) t(i);

SET anofox_tabfm_device = 'rocm';

-- the registered model, through the NEW model-provided dispatch
CREATE TABLE pred_registered AS
SELECT id, yhat FROM tabfm_classify('churn','segment', model := 'tabfm-as-user-model');
.mode list
SELECT 'REGISTERED_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded AND model = 'tabfm-as-user-model';
.mode duckbox

-- the built-in model, through the OLD bundled-graph dispatch (same weights)
CREATE TABLE pred_builtin AS
SELECT id, yhat FROM tabfm_classify('churn','segment', model := 'tabfm-v1');
.mode list
SELECT 'BUILTIN_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded AND model = 'tabfm-v1';
-- same weights, same graph, same device => identical predictions or the new
-- path changed the answer, which is exactly what must never happen
SELECT 'PATHS_DISAGREE=' || count(*) FROM pred_registered r JOIN pred_builtin b USING (id)
 WHERE r.yhat IS DISTINCT FROM b.yhat;
.mode duckbox
