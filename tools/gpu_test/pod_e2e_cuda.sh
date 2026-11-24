#!/usr/bin/env bash
# The Phase 3 GPU end-to-end on a rented NVIDIA GPU (runpod_run.py): anofox-ggml-adapter's CUDA plugin, as CI built it, under both
# extensions. Builds tabfm, decide and the adapter's tests from their pushed branches ON the pod (binaries built elsewhere need a newer
# glibc than the pod image), downloads the real weights, and runs:
#   - the adapter's plugin tests against the plugin, then again with the GPU hidden (create must refuse: proof the GPU served);
#   - tabfm's GPU suites (tabfm_gpu_use_cases, tabfm_gpu_registered_model), which assert the serving device;
#   - decide's GPU ticket E2E (decide_local_e2e_gpu.test: every block served by plugin:cuda, 1e-3 of ONNX Runtime, labels exact).
#
#   runpod_run.py --secure --gpu "NVIDIA GeForce RTX 4090" --disk 90 \
#       --upload <libanofox_ggml-cuda_plugin.so> tools/gpu_test/pod_e2e_cuda.sh --command 'bash /workspace/pod_e2e_cuda.sh'
#
# Env: TABFM_BRANCH, DECIDE_BRANCH (default: the Phase 3 branches); SKIP_TABFM=1 runs only the adapter and decide parts.
# ONNX Runtime 1.29.0 is gone from GitHub (decide's vcpkg port and its prebuilt archive both 404) and decide still links it
# until Phase 5: upload onnxruntime-linux-x64-1.29.0.tgz (the PyPI wheel's library + the headers) next to this script and
# decide builds against it. Ubuntu 22.04's GCC 11 cannot build decide: GCC 13 from the toolchain PPA does.
# Prints one RESULT line per suite at the end.
set -uo pipefail
TABFM_BRANCH=${TABFM_BRANCH:-feat/ggml-adapter}
DECIDE_BRANCH=${DECIDE_BRANCH:-feat/ggml-gpu}
W=/workspace
EP=$W/ep
mkdir -p "$EP" && cp "$W/libanofox_ggml-cuda_plugin.so" "$EP/"
declare -A RESULT
run() { # name, command...: runs it, keeps the tail of its output, records PASS/FAIL
	local name=$1
	shift
	echo "=== $name"
	if "$@" > "$W/$name.log" 2>&1; then RESULT[$name]=PASS; else RESULT[$name]="FAIL (exit $?)"; fi
	# the first errors, then the tail (a failed build's cause is far above its last lines)
	grep -nE "error:|Error [0-9]|FAILED:|Killed|internal compiler error" "$W/$name.log" | head -15
	tail -25 "$W/$name.log"
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# a full DuckDB + extension build at one job per core runs a pod out of memory (the compiler is killed): about 3 GB per job
MEM_GB=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
# a container sees the host's /proc/meminfo: its own limit is the cgroup's
CG=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null || echo max)
if [[ $CG =~ ^[0-9]+$ ]] && [ $((CG / 1073741824)) -lt "$MEM_GB" ]; then MEM_GB=$((CG / 1073741824)); fi
JOBS=$((MEM_GB / 3))
[ "$JOBS" -lt 1 ] && JOBS=1
[ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
echo "cores $(nproc), memory ${MEM_GB} GB, build jobs $JOBS"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake ninja-build build-essential python3 zip unzip curl pkg-config tar ccache software-properties-common > /dev/null
add-apt-repository -y ppa:ubuntu-toolchain-r/test > /dev/null 2>&1 && apt-get update -qq && apt-get install -y -qq gcc-13 g++-13 > /dev/null
export CC=gcc-13 CXX=g++-13
export GEN=ninja DATAZOO_DISABLE_TELEMETRY=1 CCACHE_DIR=$W/ccache
# both builds take their dependencies from vcpkg (decide's release build still compiles ONNX Runtime through it, until Phase 5)
git clone -q https://github.com/microsoft/vcpkg.git $W/vcpkg && $W/vcpkg/bootstrap-vcpkg.sh -disableMetrics > /dev/null
export VCPKG_ROOT=$W/vcpkg VCPKG_TOOLCHAIN_PATH=$W/vcpkg/scripts/buildsystems/vcpkg.cmake

cd $W && git clone -q --recursive --depth 1 -b "$TABFM_BRANCH" https://github.com/DataZooDE/anofox-tabfm.git tabfm & # the adapter tests build from its copy
cd $W && git clone -q --recursive --depth 1 -b "$DECIDE_BRANCH" https://github.com/DataZooDE/anofox-decide.git decide &
wait
[ "${SKIP_TABFM:-0}" = 1 ] || run build_tabfm make -C $W/tabfm release
ORT_TGZ=$W/onnxruntime-linux-x64-1.29.0.tgz
if [ -f "$ORT_TGZ" ]; then
	mkdir -p $W/ort && tar -xzf "$ORT_TGZ" -C $W/ort --strip-components=1
	export LD_LIBRARY_PATH=$W/ort/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
	run build_decide make -C $W/decide release DECIDE_ORT_VCPKG=0 EXT_FLAGS="-DDECIDE_ORT_URL=file://$ORT_TGZ"
else
	run build_decide make -C $W/decide release
fi
run build_adapter bash -c "cmake -S $W/tabfm/vendor/anofox-ggml-adapter -B $W/adapter-build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build $W/adapter-build"

# the real weights (the unittest has no httpfs): through each extension's own CLI
[ "${SKIP_TABFM:-0}" = 1 ] || run weights_tabfm bash -c "cd $W/tabfm && ./build/release/duckdb -c \"INSTALL httpfs; LOAD httpfs; CALL tabfm_download('classification', model := 'tabicl-v2'); CALL tabfm_download('regression', model := 'mitra');\""
run weights_decide bash -c "cd $W/decide && ./build/release/duckdb -c \"CALL decide_download('julia-1'); CALL decide_download('laya-multilingual'); CALL decide_download('laya-typed-decisions');\""

cd $W/tabfm/vendor/anofox-ggml-adapter
run adapter_plugin_tests env ANOFOX_GGML_TEST_PLUGIN=$EP/libanofox_ggml-cuda_plugin.so $W/adapter-build/adapter_tests "[plugin]"
# with the GPU hidden the same plugin must refuse at create: the passing run above was served by the GPU
run adapter_gpu_hidden_refuses bash -c "! env CUDA_VISIBLE_DEVICES= ANOFOX_GGML_TEST_PLUGIN=$EP/libanofox_ggml-cuda_plugin.so $W/adapter-build/adapter_tests 'plugin_ggml: the generic plugin runs*'"

cd $W/tabfm
for t in tabfm_gpu_use_cases tabfm_gpu_registered_model; do
	[ "${SKIP_TABFM:-0}" = 1 ] && break
	run "$t" env TABFM_REAL_WEIGHTS=1 TABFM_GPU_DEVICE=cuda TABFM_GPU_EP_PATH=$EP ./build/release/test/unittest "test/sql/$t.test"
done
./build/release/duckdb -c "SET anofox_tabfm_ep_path='$EP'; SELECT * FROM tabfm_devices();" 2>&1 | tail -6

cd $W/decide
run decide_local_e2e_gpu env DECIDE_REAL_WEIGHTS=1 DECIDE_CACHE_DIR=$HOME/.cache/anofox-decide DECIDE_E2E_DEVICE=cuda DECIDE_E2E_EP_PATH=$EP \
	./build/release/test/unittest test/sql/decide_local_e2e_gpu.test
./build/release/duckdb -c "SET anofox_decide_ep_path='$EP'; SET anofox_decide_device='cuda'; SELECT decide_probability('Refund me, please.', 'A refund is requested.', model := 'julia-1') AS p; SELECT model, engine, runs FROM __anofox_decide_local_sessions();" 2>&1 | tail -8

echo "=== SUMMARY"
for k in build_tabfm build_decide build_adapter weights_tabfm weights_decide adapter_plugin_tests adapter_gpu_hidden_refuses tabfm_gpu_use_cases tabfm_gpu_registered_model decide_local_e2e_gpu; do
	echo "RESULT $k: ${RESULT[$k]:-not run}"
done
