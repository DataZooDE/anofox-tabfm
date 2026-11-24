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
# Env: TABFM_BRANCH, DECIDE_BRANCH (default: the Phase 3 branches). Prints one RESULT line per suite at the end.
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
	tail -25 "$W/$name.log"
}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake ninja-build build-essential python3 zip unzip curl pkg-config tar ccache > /dev/null
export GEN=ninja DATAZOO_DISABLE_TELEMETRY=1 CCACHE_DIR=$W/ccache
# both builds take their dependencies from vcpkg (decide's release build still compiles ONNX Runtime through it, until Phase 5)
git clone -q https://github.com/microsoft/vcpkg.git $W/vcpkg && $W/vcpkg/bootstrap-vcpkg.sh -disableMetrics > /dev/null
export VCPKG_ROOT=$W/vcpkg VCPKG_TOOLCHAIN_PATH=$W/vcpkg/scripts/buildsystems/vcpkg.cmake

cd $W && git clone -q --recursive --depth 1 -b "$TABFM_BRANCH" https://github.com/DataZooDE/anofox-tabfm.git tabfm &
cd $W && git clone -q --recursive --depth 1 -b "$DECIDE_BRANCH" https://github.com/DataZooDE/anofox-decide.git decide &
wait
run build_tabfm make -C $W/tabfm release
run build_decide make -C $W/decide release
run build_adapter bash -c "cmake -S $W/tabfm/vendor/anofox-ggml-adapter -B $W/adapter-build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build $W/adapter-build"

# the real weights (the unittest has no httpfs): through each extension's own CLI
run weights_tabfm bash -c "cd $W/tabfm && ./build/release/duckdb -c \"INSTALL httpfs; LOAD httpfs; CALL tabfm_download('classification', model := 'tabicl-v2'); CALL tabfm_download('regression', model := 'mitra');\""
run weights_decide bash -c "cd $W/decide && ./build/release/duckdb -c \"CALL decide_download('julia-1'); CALL decide_download('laya-multilingual'); CALL decide_download('laya-typed-decisions');\""

cd $W/tabfm/vendor/anofox-ggml-adapter
run adapter_plugin_tests env ANOFOX_GGML_TEST_PLUGIN=$EP/libanofox_ggml-cuda_plugin.so $W/adapter-build/adapter_tests "[plugin]"
# with the GPU hidden the same plugin must refuse at create: the passing run above was served by the GPU
run adapter_gpu_hidden_refuses bash -c "! env CUDA_VISIBLE_DEVICES= ANOFOX_GGML_TEST_PLUGIN=$EP/libanofox_ggml-cuda_plugin.so $W/adapter-build/adapter_tests 'plugin_ggml: the generic plugin runs*'"

cd $W/tabfm
for t in tabfm_gpu_use_cases tabfm_gpu_registered_model; do
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
