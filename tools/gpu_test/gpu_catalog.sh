#!/usr/bin/env bash
# The whole model catalog on a GPU through the ggml GPU plugin, against the CPU, through the SQL surface (the real-weights quality loop of
# docs/GGML_PLAN.md). Uses a BUILT tree (build/release/duckdb + the extension) and a built plugin (anofox-ggml-adapter's, built in
# vendor/anofox-ggml-adapter by its tools/ggml_plugin, or fetched by tabfm_download_runtime), so it runs on any box with the GPU:
#
#   GPU=rocm EP=~/ep tools/gpu_test/gpu_catalog.sh                      # every (model, task) the catalog lists
#   GPU=cuda EP=/work/ep MODELS="mitra:cr tabicl-v2:c" tools/gpu_test/gpu_catalog.sh
#
# Each (model, task) runs in a FRESH DuckDB process (a crash costs its own row) using the scenarios cuda_catalog_{classification,
# regression}.sql, with the device rewritten to $GPU. Weights come from anofox_tabfm_cache_dir (CACHE, default ~/.cache/anofox-tabfm);
# missing ones are fetched with tabfm_download (needs network and the licenses accepted, which ACCEPT_LICENSES=1 does for this run).
# served_by is the column that matters: anything but $GPU:N means the "GPU" side ran on the CPU.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GPU=${GPU:?GPU=cuda|rocm|mlx}
EP=${EP:?EP=<dir with the plugin: libanofox_ggml-cuda_plugin.so, libanofox_ggml-rocm_plugin.so or libanofox_mlx_plugin.dylib>}
CACHE=${CACHE:-$HOME/.cache/anofox-tabfm}
OUT=${OUT:-$PWD/gpu_catalog_$GPU}
PER_MODEL_TIMEOUT=${PER_MODEL_TIMEOUT:-1500}
DUCK="$ROOT/build/release/duckdb"
EXT="$ROOT/build/release/extension/anofox_tabfm/anofox_tabfm.duckdb_extension"
mkdir -p "$OUT/logs"
DB="$OUT/results.duckdb"
LICENSE=""
[ "${ACCEPT_LICENSES:-0}" = 1 ] && LICENSE="SET anofox_tabfm_accept_hf_license = true;"
PRE="LOAD '$EXT'; SET anofox_tabfm_cache_dir = '$CACHE'; SET anofox_tabfm_ep_path = '$EP'; $LICENSE"
duck() { DATAZOO_DISABLE_TELEMETRY=1 "$DUCK" -unsigned "$@"; }
scenario() { sed "s/'cuda'/'$GPU'/g; s/ cuda=/ $GPU=/g" "$ROOT/tools/gpu_test/scenarios/cuda_catalog_$1.sql"; }

{ echo "$PRE"; cat "$ROOT/tools/gpu_test/scenarios/cuda_catalog.sql"; } | duck "$DB" || exit 1
if [ -z "${MODELS:-}" ]; then
  # every catalog model with the tasks it lists
  MODELS=$(echo "$PRE SELECT model || ':' || (CASE WHEN capabilities LIKE '%classify%' THEN 'c' ELSE '' END)
                                  || (CASE WHEN capabilities LIKE '%regress%' THEN 'r' ELSE '' END)
                    FROM tabfm_list_models() ORDER BY model;" | duck -list -noheader | tr '\n' ' ')
fi
echo "MODELS: $MODELS"
for entry in $MODELS; do
  m=${entry%%:*}; tasks=${entry##*:}
  for (( i=0; i<${#tasks}; i++ )); do
    case "${tasks:$i:1}" in c) task=classification;; r) task=regression;; *) continue;; esac
    echo "##### $m / $task"
    { echo "INSTALL httpfs; LOAD httpfs; $PRE"; echo "CALL tabfm_download('$task', model := '$m');"; } | duck > "$OUT/logs/${m}_${task}_download.log" 2>&1
    { echo "$PRE"; echo "SET VARIABLE model = '$m';"; scenario "$task"; } |
      timeout "$PER_MODEL_TIMEOUT" "$DUCK" -unsigned "$DB" > "$OUT/logs/${m}_${task}.log" 2>&1
    echo "exit=$? $(grep -m1 -E 'Error|error' "$OUT/logs/${m}_${task}.log" | cut -c1-160)"
  done
done
duck -box "$DB" -c "SELECT model, task, served_by, n_cpu, n_gpu, label_diff, round(score_max_abs_diff, 6) AS max_abs,
  round(reg_max_rel_diff, 6) AS max_rel, repeat_diff, round(cpu_quality, 4) AS cpu_q, round(gpu_quality, 4) AS gpu_q,
  cpu_ms, gpu_cold_ms, gpu_warm_ms FROM report ORDER BY model, task"
duck "$DB" -c "COPY (SELECT * FROM report ORDER BY model, task) TO '$OUT/report.csv'"
echo "GPU_CATALOG_DONE"
