#!/usr/bin/env bash
# Fails when a plugin library depends (DT_NEEDED) on a shared library the user would have to install: the ggml GPU plugins must be self-contained
# apart from the GPU driver. Usage: scripts/check_plugin_deps.sh <plugin.so> [forbidden-regex]
#   default regex = the CUDA runtime libraries (libcudart, libcublas, libcublasLt, libnccl)
set -euo pipefail
plugin=${1:?usage: check_plugin_deps.sh <plugin.so> [forbidden-regex]}
forbidden=${2:-'lib(cudart|cublas|cublasLt|nccl)\.so'}
needed=$(readelf -d "$plugin" | awk '/\(NEEDED\)/ {gsub(/[\[\]]/, "", $NF); print $NF}')
echo "NEEDED by $plugin:"; echo "$needed" | sed 's/^/  /'
if echo "$needed" | grep -Eq "$forbidden"; then
	echo "FAIL: $plugin links a shared library matching /$forbidden/ (it must be linked statically):" >&2
	echo "$needed" | grep -E "$forbidden" | sed 's/^/  /' >&2
	exit 1
fi
echo "OK: no library matching /$forbidden/ is needed"
