-- The ggml GPU plugin through the SQL surface (docs/GGML_PLAN.md "P8 result"): Nori 1.5 regression, real weights, plain external-data graph.
-- Run with the plugin built by tools/ggml_plugin placed alone in EP_DIR (libanofox_tabfm_ggml-rocm_plugin.so or ggml-cuda), and
--   DEVICE=rocm|cuda  NORI_DIR=<dir holding graph_ext_nori15_regression.onnx + model.safetensors>  EP_DIR=<plugin dir>
-- substituted below. Never trust "cpu and gpu agree": the SERVED_BY row must name the GPU.
LOAD 'build/release/extension/anofox_tabfm/anofox_tabfm.duckdb_extension';
CALL tabfm_register_model(id := 'nori-ext', base_dir := '${NORI_DIR}', regression_graph := 'graph_ext_nori15_regression.onnx',
  regression_ext_graph := 'graph_ext_nori15_regression.onnx', regression_weights := 'model.safetensors', license := 'apache-2.0',
  commercial := true, preprocessing_profile := 'nori_v15_raw', max_rows := 4096, max_features := 64);
CREATE TABLE t AS SELECT i AS rid, sin(i*0.11) f1, cos(i*0.07) f2, (i%13)/13.0 f3, ln(1+(i%31)) f4,
  CASE WHEN i<70 THEN sin(i*0.11)*5+cos(i*0.07)*2 END AS target FROM range(100) r(i);
SET anofox_tabfm_device='cpu';
CREATE TABLE cpu_r AS SELECT f1,f2,f3,f4,yhat FROM tabfm_regress('t','target',features:=['f1','f2','f3','f4'],model:='nori-ext');
SET anofox_tabfm_ep_path='${EP_DIR}';
SET anofox_tabfm_device='${DEVICE}';
CREATE TABLE gpu_r AS SELECT f1,f2,f3,f4,yhat FROM tabfm_regress('t','target',features:=['f1','f2','f3','f4'],model:='nori-ext');
SELECT 'SERVED_BY', model, device FROM tabfm_models() WHERE model='nori-ext';
SELECT 'AGREE', count(*) AS n, max(abs(c.yhat-g.yhat)) AS max_abs_diff, max(abs(c.yhat)) AS scale FROM cpu_r c JOIN gpu_r g USING (f1,f2,f3,f4);
