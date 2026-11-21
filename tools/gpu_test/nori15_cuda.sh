#!/usr/bin/env bash
# Nori 1.5 on a rented NVIDIA GPU: CPU vs CUDA through the SQL surface, with the REAL weights.
#
# Runs RELEASED artefacts (nothing is built on the pod): the DuckDB CLI, the extension from get.anofox.com and
# the CUDA plugin that CALL tabfm_accelerate() fetches. The model is not in a released extension yet, so it is
# REGISTERED in SQL (tabfm_register_model) with the plain graph for the CPU side and the ext graph for the CUDA
# side, both over the same model.safetensors; that exercises exactly the files the bundled entry will ship.
#
#   tools/gpu_test/runpod_run.py --secure --max-candidates 4 \
#       --upload tools/gpu_test/nori15_cuda.sh tools/gpu_test/scenarios/cuda_catalog.sql \
#                tools/gpu_test/scenarios/cuda_catalog_regression.sql \
#                <dir>/graph_nori_regression.onnx <dir>/graph_ext_nori15_regression.onnx \
#                <dir>/tensor_map_nori_regression.json <dir>/model.safetensors \
#       --command 'bash /workspace/nori15_cuda.sh'
#
# The last lines are the verdict: PASS only if the "GPU" side was really served by cuda:N (a silent CPU fallback
# prints cpu-vs-cpu numbers that agree perfectly, CLAUDE.md rule 2), the answers agree with the CPU's, and a
# second CUDA call repeats the first.
set -uo pipefail
WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR" || exit 1
DUCKDB_VERSION=${DUCKDB_VERSION:-v1.5.6}
CACHE=$WORKDIR/cache
mkdir -p "$CACHE" nori15 logs

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=== host ==="; nproc; free -g | head -2; ldd --version | head -1

echo "=== fetch DuckDB ${DUCKDB_VERSION} CLI + the released extension ==="
curl -sSL -o duckdb_cli.zip "https://github.com/duckdb/duckdb/releases/download/${DUCKDB_VERSION}/duckdb_cli-linux-amd64.zip" \
  && python3 -c "import zipfile; zipfile.ZipFile('duckdb_cli.zip').extractall('.')" && chmod +x duckdb || exit 1
if [ ! -f anofox_tabfm.duckdb_extension ]; then
  curl -sSL -o anofox_tabfm.duckdb_extension.gz "https://get.anofox.com/${DUCKDB_VERSION}/linux_amd64/anofox_tabfm.duckdb_extension.gz" \
    && gunzip -f anofox_tabfm.duckdb_extension.gz || exit 1
fi
echo "extension sha256 $(sha256sum anofox_tabfm.duckdb_extension | cut -d' ' -f1)"

for f in graph_nori_regression.onnx graph_ext_nori15_regression.onnx tensor_map_nori_regression.json model.safetensors; do
  [ -f "$f" ] && mv -f "$f" nori15/
  [ -f "nori15/$f" ] || { echo "MISSING upload: $f"; exit 1; }
done
(cd nori15 && sha256sum model.safetensors graph_ext_nori15_regression.onnx | cut -c1-72)

PRE="INSTALL httpfs; LOAD httpfs; LOAD '$WORKDIR/anofox_tabfm.duckdb_extension';
SET anofox_tabfm_cache_dir = '$CACHE'; SET anofox_tabfm_accept_hf_license = true;"
REGISTER="CALL tabfm_register_model(id := 'nori-1-5', base_dir := '$WORKDIR/nori15',
  regression_graph := 'graph_nori_regression.onnx', regression_ext_graph := 'graph_ext_nori15_regression.onnx',
  regression_weights := 'model.safetensors', regression_tensor_map := 'tensor_map_nori_regression.json',
  license := 'apache-2.0', commercial := true, weights_repo := 'local:nori15', weights_revision := 'local',
  preprocessing_profile := 'nori_v15_raw', max_rows := 5000, max_features := 100);
SET VARIABLE model = 'nori-1-5';"
duck() { ./duckdb -unsigned "$@"; }

echo "=== tabfm_accelerate (fetches the CUDA plugin + ORT GPU runtime) ==="
{ echo "$PRE"; echo "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_tabfm';
CALL tabfm_accelerate(); SELECT * FROM tabfm_devices();"; } | duck || exit 1

echo "=== nori-1-5 regression: CPU vs CUDA ==="
rm -f results.duckdb
{ echo "$PRE"; cat cuda_catalog.sql; echo "$REGISTER"; cat cuda_catalog_regression.sql; } | duck results.duckdb 2>&1 | tee logs/nori15.log

echo "=== verdict ==="
./duckdb -unsigned results.duckdb -noheader -list "
SELECT CASE
  WHEN served_by NOT LIKE 'cuda%'                 THEN 'FAIL: the GPU side was served by ' || served_by || ' (not cuda): every number above is cpu-vs-cpu'
  WHEN n_gpu <> n_cpu OR n_gpu = 0                THEN 'FAIL: CUDA returned ' || n_gpu || ' rows, CPU ' || n_cpu
  WHEN reg_max_rel_diff > 1e-3                    THEN 'FAIL: CUDA disagrees with the CPU: max rel diff ' || reg_max_rel_diff
  WHEN repeat_diff <> 0                           THEN 'FAIL: a second CUDA call differs from the first in ' || repeat_diff || ' rows'
  WHEN gpu_quality < cpu_quality - 0.01           THEN 'FAIL: CUDA R2 ' || gpu_quality || ' vs CPU ' || cpu_quality
  ELSE 'PASS: nori-1-5 served by ' || served_by || ', max rel diff vs CPU ' || reg_max_rel_diff || ', R2 cpu ' || round(cpu_quality, 4) || ' / cuda ' || round(gpu_quality, 4) || ', cuda cold ' || gpu_cold_ms || ' ms / warm ' || gpu_warm_ms || ' ms (cpu ' || cpu_ms || ' ms)'
END FROM report WHERE model = 'nori-1-5' AND task = 'regression';" 2>&1 | tail -3
