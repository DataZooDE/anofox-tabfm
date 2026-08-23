-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
--
-- The weight-free ROCm smoke test (PHASE_COMPLETION_PLAN C3): the committed
-- fixture's MIGraphX variant (graph_migraphx_fixture.onnx, the Shape-rewrite
-- of graph_fixture.onnx — U1's "MIGraphX cannot run the fixture" applied to
-- the PLAIN ext format, not the rewritten one) compiles and serves in
-- seconds with the random-init fixture weights. No download, no license,
-- no real model: run it before pushing anything that touches the ROCm path.
-- As always: the SERVED_BY line is the assertion, not "it returned rows".
LOAD anofox_tabfm;
CALL tabfm_register_model(id := 'mgx-fixture', base_dir := 'test/fixtures',
  classification_graph := 'graph_fixture.onnx',
  classification_migraphx_graph := 'graph_migraphx_fixture.onnx',
  classification_weights := 'model.safetensors',
  classification_tensor_map := 'tensor_map_fixture.json',
  license := 'fixture-mit', preprocessing_profile := 'tabfm_v1_minimal');
SET anofox_tabfm_device = 'rocm';
CREATE TABLE t AS SELECT (hash(i*3)%100)/25.0 AS f1, (hash(i*7)%100)/25.0 AS f2,
  (CASE WHEN i<6 THEN 'c'||(i%2)::VARCHAR END) AS label FROM range(9) r(i);
SELECT count(*) AS rows_predicted FROM tabfm_classify('t','label', model := 'mgx-fixture');
SELECT 'FIXTURE_SERVED_BY=' || device FROM tabfm_models() WHERE model='mgx-fixture' AND loaded;
