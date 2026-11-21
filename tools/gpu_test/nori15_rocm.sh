#!/usr/bin/env bash
# Nori 1.5 on a ROCm machine (gfx1201): CPU vs ROCm through the SQL surface with the REAL weights, using the masked
# MIGraphX graph. Builds nothing: the released DuckDB CLI + extension, and an EXISTING migraphx plugin
# (EP_DIR; the plugin is model-agnostic, so any current build of libanofox_tabfm_migraphx_plugin.so works).
# The model is registered in SQL (the released extension does not know it yet), with the plain graph for the CPU
# side and the masked graph for ROCm, both over one model.safetensors.
#
#   WORKDIR=~/nori15-rocm EP_DIR=<dir holding libanofox_tabfm_migraphx_plugin.so> nice -n 19 bash nori15_rocm.sh
#
# Needs in WORKDIR: graph_nori_regression.onnx graph_migraphx_nori15_regression.onnx tensor_map_nori_regression.json
# model.safetensors scenarios/nori15_rocm.sql. The first call compiles the masked graph for the 128 x 16 bucket:
# for Nori-30M that took ~37 minutes, so this runs under nohup and the log is the record.
set -uo pipefail
WORKDIR=${WORKDIR:-$HOME/nori15-rocm}
EP_DIR=${EP_DIR:?set EP_DIR to the directory holding libanofox_tabfm_migraphx_plugin.so}
DUCKDB_VERSION=${DUCKDB_VERSION:-v1.5.6}
cd "$WORKDIR" || exit 1
mkdir -p cache ep && cp -f "$EP_DIR"/libanofox_tabfm_migraphx_plugin.so ep/ || exit 1   # a COPY: the source is never touched

if [ ! -x duckdb ]; then
  curl -sSL -o duckdb_cli.zip "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip" \
    && python3 -c "import zipfile; zipfile.ZipFile('duckdb_cli.zip').extractall('.')" && chmod +x duckdb || exit 1
fi
if [ ! -f anofox_tabfm.duckdb_extension ]; then
  curl -sSL -o anofox_tabfm.duckdb_extension.gz "https://get.anofox.com/${DUCKDB_VERSION}/linux_amd64/anofox_tabfm.duckdb_extension.gz" \
    && gunzip -f anofox_tabfm.duckdb_extension.gz || exit 1
fi
echo "extension sha256 $(sha256sum anofox_tabfm.duckdb_extension | cut -d' ' -f1)"
sha256sum model.safetensors graph_migraphx_nori15_regression.onnx | cut -c1-72

{ cat <<SQL
LOAD '$WORKDIR/anofox_tabfm.duckdb_extension';
SET anofox_tabfm_cache_dir = '$WORKDIR/cache'; SET anofox_tabfm_ep_path = '$WORKDIR/ep';
CALL tabfm_register_model(id := 'nori-1-5', base_dir := '$WORKDIR',
  regression_graph := 'graph_nori_regression.onnx', regression_migraphx_graph := 'graph_migraphx_nori15_regression.onnx',
  regression_weights := 'model.safetensors', regression_tensor_map := 'tensor_map_nori_regression.json',
  license := 'apache-2.0', commercial := true, weights_repo := 'local:nori15', weights_revision := 'local',
  preprocessing_profile := 'nori_v15_raw', max_rows := 2000, max_features := 100);
SELECT * FROM tabfm_devices();
SQL
  cat scenarios/nori15_rocm.sql; } | DATAZOO_DISABLE_TELEMETRY=1 ./duckdb -unsigned 2>&1
