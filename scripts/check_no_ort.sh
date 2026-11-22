#!/usr/bin/env bash
# The C++ extension, its build and its CI contain no ONNX Runtime: the CPU runs on ggml, CUDA/ROCm on the ggml GPU plugin, Apple GPUs on MLX.
# Python tooling under tools/ keeps onnxruntime as the reference for goldens and parity checks; docs keep their history. Usage: scripts/check_no_ort.sh
set -uo pipefail
cd "$(dirname "$0")/.."
pattern='onnxruntime|Ort::|TABFM_WITH_ORT|TABFM_FLAVOR|TABFM_EP_[A-Z]|libanofoxort_gpu|TABFM_ORT_[A-Z]'
hits=$(grep -rnE "$pattern" src CMakeLists.txt cmake Makefile test/cpp test/sql .github/workflows extension_config.cmake 2>/dev/null \
	| grep -v '^scripts/check_no_ort.sh' )
if [ -n "$hits" ]; then
	echo "FAIL: ONNX Runtime is still referenced ($(echo "$hits" | wc -l) lines):"
	echo "$hits" | cut -c1-160 | head -40
	exit 1
fi
for gone in cmake/ort.cmake src/tabfm_cuda_plugin.cpp src/tabfm_migraphx_plugin.cpp src/include/tabfm_soname_patch.hpp; do
	if [ -e "$gone" ]; then
		echo "FAIL: $gone still exists"
		exit 1
	fi
done
echo "OK: no ONNX Runtime in the extension, its build or its CI"
