-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
--
-- NaN-marked and NULL-marked missing targets must give the same answer on the GPU as on the CPU. The
-- rule (NULL or NaN = predict this row, Infinity = error) is applied before any device is chosen, so this
-- is expected to hold by construction; the scenario exists because "by construction" is exactly what a
-- silent CPU fallback also looks like. Weight-free: the committed fixture's MIGraphX variant.
-- As always the SERVED_BY lines are the assertion, read them first.
LOAD anofox_tabfm;
CALL tabfm_register_model(id := 'mgx-fixture', base_dir := 'test/fixtures',
  classification_graph := 'graph_fixture.onnx',
  classification_migraphx_graph := 'graph_migraphx_fixture.onnx',
  classification_weights := 'model.safetensors',
  classification_tensor_map := 'tensor_map_fixture.json',
  license := 'fixture-mit', preprocessing_profile := 'tabfm_v1_minimal');

-- 20 rows, 12 labelled (classes 0/1, as DOUBLE so a NaN can mark the rest); features unique per row
CREATE TABLE tn AS SELECT ((i*7) % 11) * 0.25 AS f1, ((i*3) % 7) * 0.5 AS f2,
  CASE WHEN i < 12 THEN (i % 2)::DOUBLE END AS y FROM range(20) r(i);
CREATE TABLE tq AS SELECT f1, f2, CASE WHEN y IS NULL THEN 'nan'::DOUBLE ELSE y END AS y FROM tn;

SET anofox_tabfm_device = 'rocm';
CREATE TABLE g_null AS SELECT f1, f2, yhat, is_training FROM tabfm_classify('tn', 'y', model := 'mgx-fixture');
SELECT 'NULL_SERVED_BY=' || device FROM tabfm_models() WHERE model = 'mgx-fixture' AND loaded;
CREATE TABLE g_nan AS SELECT f1, f2, yhat, is_training FROM tabfm_classify('tq', 'y', model := 'mgx-fixture');
SELECT 'NAN_SERVED_BY=' || device FROM tabfm_models() WHERE model = 'mgx-fixture' AND loaded;

SET anofox_tabfm_device = 'cpu';
CREATE TABLE c_nan AS SELECT f1, f2, yhat, is_training FROM tabfm_classify('tq', 'y', model := 'mgx-fixture');
SELECT 'CPU_SERVED_BY=' || device FROM tabfm_models() WHERE model = 'mgx-fixture' AND loaded;

SELECT 'GPU_ROWS=' || count(*) || ' GPU_TRAINING=' || count(*) FILTER (WHERE is_training) FROM g_nan;
-- the three equalities the rule promises (all must be 0 disagreements, 20 rows joined)
SELECT 'GPU_NAN_VS_GPU_NULL=' || count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat
                                              OR a.is_training IS DISTINCT FROM b.is_training) || '/' || count(*)
FROM g_null a JOIN g_nan b USING (f1, f2);
SELECT 'GPU_NAN_VS_CPU_NAN=' || count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) || '/' || count(*)
FROM g_nan a JOIN c_nan b USING (f1, f2);
