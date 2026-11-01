#!/usr/bin/env bash
# The model catalog on a rented NVIDIA GPU, CPU vs CUDA through the SQL surface
# (tools/gpu_test/scenarios/cuda_catalog*.sql). Runs the RELEASED artefacts, so nothing is
# built on the pod: the DuckDB CLI from duckdb/duckdb, the extension from get.anofox.com (pinned
# by sha256 -- it is the manylinux build, so the pod's older glibc loads it), and the CUDA plugin
# + ONNX Runtime GPU libraries that CALL tabfm_accelerate() fetches for the release tag the
# extension pins. Weights are downloaded on the pod (tabfm_download) except the ones that need a
# Python conversion step (TabPFN, TabICL, LimiX's optional safetensors), which are uploaded
# pre-converted as weights_*.tar in the cache-dir layout.
#
#   tools/gpu_test/runpod_run.py --secure --gpu "NVIDIA GeForce RTX 4090" \
#       --upload tools/gpu_test/cuda_catalog.sh tools/gpu_test/scenarios/cuda_catalog*.sql \
#                weights_converted.tar \
#       --command 'EXT_SHA256=<sha> MODELS="limix-2m:cr causilo:cr" bash /workspace/cuda_catalog.sh'
#
# MODELS is a space-separated list of <model>:<tasks>, tasks being c (classification) and/or r
# (regression), run in that order; an entry prune:<glob> deletes those cache directories first.
# DEADLINE_S (default 3000) stops starting new models once that
# many seconds have passed, so a slow pod cannot run up the bill. A weights_late.tar that appears
# in /workspace while the run is in progress (uploaded in parallel) is unpacked before the first
# model that needs it; WAIT_LATE_S bounds the wait. (To find the pod's ssh endpoint for that upload, run
# runpod_run.py with `python3 -u`: redirected to a file, its own prints are otherwise block-buffered.)
set -uo pipefail
WORKDIR=${WORKDIR:-/workspace}
cd "$WORKDIR" || exit 1
START=$(date +%s)
DEADLINE_S=${DEADLINE_S:-3000}
WAIT_LATE_S=${WAIT_LATE_S:-600}
DUCKDB_VERSION=${DUCKDB_VERSION:-v1.5.6}
CACHE=$WORKDIR/cache
mkdir -p "$CACHE" logs

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
GOT=$(sha256sum anofox_tabfm.duckdb_extension | cut -d' ' -f1)
echo "extension sha256 $GOT"
if [ -n "${EXT_SHA256:-}" ] && [ "$GOT" != "$EXT_SHA256" ]; then
  echo "EXTENSION SHA MISMATCH: wanted $EXT_SHA256 -- refusing to measure a different build"; exit 1
fi
for t in weights_*.tar; do
  [ -f "$t" ] && [ "$t" != weights_late.tar ] && { echo "unpack $t"; tar xf "$t" -C "$CACHE"; }
done

PRE="INSTALL httpfs; LOAD httpfs; LOAD '$WORKDIR/anofox_tabfm.duckdb_extension';
SET anofox_tabfm_cache_dir = '$CACHE'; SET anofox_tabfm_accept_hf_license = true;"
duck() { ./duckdb -unsigned "$@"; }

echo "=== tabfm_accelerate (fetches the CUDA plugin + ORT GPU runtime) ==="
# SKIP_ACCEL=1 is for a CPU-only dry run of this script off the pod (the CUDA calls then fail, and
# the report must say n_gpu = 0 / served_by NONE for every row).
if [ -z "${SKIP_ACCEL:-}" ]; then
  { echo "$PRE"; echo "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_tabfm';
  CALL tabfm_accelerate(); SELECT * FROM tabfm_devices();"; } | duck || exit 1
fi

{ echo "$PRE"; cat cuda_catalog.sql; } | duck results.duckdb || exit 1

LATE_DONE=0
needs_late() { case "$1" in tabpfn-v2|tabpfn-v2-5|tabpfn-v3|tabicl-v2) return 0;; esac; return 1; }
for entry in ${MODELS}; do
  m=${entry%%:*}; tasks=${entry##*:}
  # prune:<glob> frees container disk before a large download (e.g. prune:nvidia__Kumo* before tabfm-v1)
  if [ "$m" = prune ]; then echo "prune $CACHE/$tasks"; rm -rf "$CACHE"/$tasks; df -h "$CACHE" | tail -1; continue; fi
  el=$(( $(date +%s) - START ))
  if [ "$el" -gt "$DEADLINE_S" ]; then echo "DEADLINE: ${el}s elapsed, skipping $m and the rest"; break; fi
  if needs_late "$m" && [ $LATE_DONE -eq 0 ]; then
    w=0; while [ ! -f weights_late.tar ] && [ $w -lt "$WAIT_LATE_S" ]; do sleep 5; w=$((w+5)); done
    if [ -f weights_late.tar ]; then tar xf weights_late.tar -C "$CACHE" && LATE_DONE=1; echo "unpacked weights_late.tar"; fi
  fi
  for (( i=0; i<${#tasks}; i++ )); do
    case "${tasks:$i:1}" in c) task=classification;; r) task=regression;; *) continue;; esac
    echo; echo "##### $m / $task  (t=$(( $(date +%s) - START ))s)"
    { echo "$PRE"; echo "CALL tabfm_download('$task', model := '$m');"; } | duck 2>&1 | tail -n 3 | cut -c1-200
    { echo "$PRE"; echo "SET VARIABLE model = '$m';"
      echo ".mode list"; echo ".headers off"
      echo "SELECT 'BACKEND cuda ' || coalesce(supported::VARCHAR, 'NULL') || ' ' || coalesce(reason, '')
              FROM tabfm_backends() WHERE model = '$m' AND task = '$task' AND backend = 'cuda';"
      cat "cuda_catalog_${task}.sql"; } | timeout 1500 ./duckdb -unsigned results.duckdb 2>&1 | tee "logs/${m}_${task}.log"
    echo "exit=${PIPESTATUS[1]}"
  done
done

echo; echo "=== REPORT ==="
duck -box results.duckdb -c "SELECT model, task, served_by, n_cpu, n_gpu, label_diff,
  round(score_max_abs_diff, 7) AS max_abs, round(reg_max_rel_diff, 7) AS max_rel, repeat_diff,
  round(cpu_quality, 4) AS cpu_q, round(gpu_quality, 4) AS gpu_q, cpu_ms, gpu_cold_ms, gpu_warm_ms
  FROM report ORDER BY rowid"
duck results.duckdb -c "COPY report TO 'report.csv'"
echo "CUDA_CATALOG_DONE elapsed=$(( $(date +%s) - START ))s"
