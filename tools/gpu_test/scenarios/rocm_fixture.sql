-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
--
-- The weight-free ROCm smoke test: the committed fixture's external-data graph (graph_ext_fixture.onnx)
-- serves in seconds on the ggml GPU plugin with the random-init fixture weights. No download, no license,
-- no real model: run it before pushing anything that touches the ROCm path.
-- As always: the SERVED_BY line is the assertion, not "it returned rows".
LOAD anofox_tabfm;
CALL tabfm_register_model(id := 'gpu-fixture', base_dir := 'test/fixtures',
  classification_graph := 'graph_fixture.onnx',
  classification_ext_graph := 'graph_ext_fixture.onnx',
  classification_weights := 'model.safetensors',
  classification_tensor_map := 'tensor_map_fixture.json',
  license := 'fixture-mit', preprocessing_profile := 'tabfm_v1_minimal');
SET anofox_tabfm_device = 'rocm';
CREATE TABLE t AS SELECT (hash(i*3)%100)/25.0 AS f1, (hash(i*7)%100)/25.0 AS f2,
  (CASE WHEN i<6 THEN 'c'||(i%2)::VARCHAR END) AS label FROM range(9) r(i);
SELECT count(*) AS rows_predicted FROM tabfm_classify('t','label', model := 'gpu-fixture');
SELECT 'FIXTURE_SERVED_BY=' || device FROM tabfm_models() WHERE model='gpu-fixture' AND loaded;
