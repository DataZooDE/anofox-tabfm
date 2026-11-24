#!/usr/bin/env bash
# The extension, its build, its CI and its test tools contain no ONNX Runtime and no trace of the removed multi-backend
# design (MIGraphX, the ORT-CUDA plugin, CoreML, build flavors, shape buckets): the CPU runs on ggml, CUDA/ROCm on the ggml GPU plugin,
# Apple GPUs on MLX (both plugins now live in vendor/anofox-ggml-adapter, on plugin ABI v3; tabfm's own plugin sources, its
# ABI v2 header and the old libanofox_tabfm_* plugin names are gone). Python exporters and tools/ggml_parity keep onnxruntime as the reference for goldens and parity checks; CHANGELOG and
# docs/archive keep the history.
#
# The deprecated SQL surface stays accepted on purpose ('migraphx' as an alias of 'rocm', the no-op anofox_tabfm_mxr_source setting, the
# ignored *_migraphx_graph registration arguments, the ABI's mxr_source/precompile slots until the next ABI version): those use the
# lowercase spellings, which the legacy pattern does not match. A line that must name a removed thing (a deprecation message, a test of
# it) says so with "deprecated", "removed", "no-op", "no effect" or "ignored" and is skipped.
#
# Usage: scripts/check_no_ort.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
report() { # <title> <hits>
	echo "FAIL: $1 ($(echo "$2" | wc -l) lines):"
	echo "$2" | cut -c1-180 | head -60
	fail=1
}
allowed='deprecated|removed|no-op|no effect|ignored'

code_paths=(src CMakeLists.txt cmake Makefile extension_config.cmake test/cpp test/sql test/fixtures .github/workflows tools/gpu_test tools/bench scripts resources)
# -I: skip binary files (the committed graphs). Self-excluded: this script names every pattern. The fixture generators under test/fixtures
# are Python dev tooling that uses onnxruntime as the oracle, like tools/ggml_parity.
code_grep() { grep -rnIE "$1" "${code_paths[@]}" 2>/dev/null | grep -vE '^scripts/check_no_ort\.sh:|^test/fixtures/[^:]*\.py:'; }
# Drop hits whose CONTENT (not path) carries an allowed word.
not_allowed() { awk -v re="$allowed" 'BEGIN { IGNORECASE = 1 } { line = $0; sub(/^[^:]*:[0-9]+:/, "", line); if (tolower(line) !~ re) print }'; }

ort='onnxruntime|Ort::|TABFM_WITH_ORT|TABFM_FLAVOR|TABFM_EP_[A-Z]|libanofoxort_gpu|TABFM_ORT_[A-Z]'
# Exporter provenance (the onnxruntime version the parity oracle ran) is fine.
hits=$(code_grep "$ort" | grep -vE '^(resources|test/fixtures)/[^:]*export_report_[^:]*\.json:|^resources/README\.md:[0-9]+:- \*\*Exporter versions')
[ -n "$hits" ] && report "ONNX Runtime is still referenced" "$hits"

legacy='MIGraphX|libanofox_tabfm_cuda_plugin|libanofox_tabfm_migraphx_plugin|ExecutionProvider|\.mxr\b|PadToShapeBucket|ShapeBucket|TryExternalDataSession|OrtBackend|graph_migraphx_|make_migraphx_graph|TABFM_DISABLE_EXTERNAL_DATA|CoreML|cpu flavor|pin_cuda_shape|pin_dynamic_slice|contract mask|lsan\.supp|libanofox_tabfm_ggml-|libanofox_tabfm_mlx_plugin|tabfm_plugin_abi\.h|TABFM_PLUGIN_ABI_VERSION|TABFM_GGML_GPU'
hits=$(code_grep "$legacy" | not_allowed)
[ -n "$hits" ] && report "the removed multi-backend design is still referenced" "$hits"

doc_paths=(README.md CLAUDE.md docs plugins .claude-plugin)
doc_legacy='libanofox_tabfm_cuda_plugin|libanofox_tabfm_migraphx_plugin|libanofoxort_gpu|ExecutionProvider|TABFM_FLAVOR|TABFM_ORT_|\.mxr\b|onnxruntime-gpu|graph_migraphx_|make_migraphx_graph|TABFM_DISABLE_EXTERNAL_DATA|tabfm_ort_engine'
hits=$(grep -rnIE "$doc_legacy" "${doc_paths[@]}" 2>/dev/null | grep -v '^docs/archive/' | not_allowed)
[ -n "$hits" ] && report "the docs still describe the removed backends" "$hits"
# The GPU plugins moved to vendor/anofox-ggml-adapter (plugin ABI v3, libanofox_ggml-*/libanofox_mlx_plugin). The plan and spike write-ups
# keep the old names as history under a status note.
doc_moved='libanofox_tabfm_ggml-|libanofox_tabfm_mlx_plugin|tabfm_plugin_abi\.h|TABFM_PLUGIN_ABI_VERSION|TABFM_GGML_GPU|gpu_plugins\.yml|check_plugin_release_assets'
hits=$(grep -rnIE "$doc_moved" "${doc_paths[@]}" 2>/dev/null | grep -vE '^docs/(archive/|GGML_PLAN\.md:|MLX_PLAN\.md:|MLX_SPIKE_RESULTS\.md:)' | not_allowed)
[ -n "$hits" ] && report "the docs still name tabfm's own GPU plugins (now the adapter's)" "$hits"

gone=(cmake/ort.cmake src/tabfm_cuda_plugin.cpp src/tabfm_migraphx_plugin.cpp src/tabfm_ort_engine.cpp src/include/tabfm_soname_patch.hpp
	src/include/tabfm_mxr_cache_key.hpp src/include/tabfm_shape_bucket.hpp test/cpp/test_tabfm_cuda_plugin.cpp
	test/cpp/test_tabfm_migraphx_plugin.cpp test/lsan.supp tools/make_migraphx_graph.py tools/check_migraphx_int_div.py
	tools/gpu_test/ort_ep_check.py tools/gpu_test/equivalence.py tools/experimental/mgx_smoke.cpp .github/workflows/graph_invariants.yml
	tools/pin_cuda_shape_operands.py tools/pin_dynamic_slice_bounds.py tools/gpu_test/cuda_catalog.sh tools/gpu_test/nori15_cuda.sh
	test/sql/tabfm_rocm_limix_ceiling.test src/tabfm_ggml_plugin.cpp src/tabfm_mlx_plugin.cpp src/tabfm_mlx_graph.cpp
	src/tabfm_mlx_graph_ops.inc src/include/tabfm_mlx_graph.hpp src/include/tabfm_plugin_abi.h tools/ggml_plugin .github/workflows/gpu_plugins.yml
	scripts/check_plugin_deps.sh scripts/check_plugin_release_assets.py tools/gpu_test/plugin_load_check.c)
for g in "${gone[@]}"; do
	[ -e "$g" ] && { echo "FAIL: $g still exists"; fail=1; }
done
for glob in 'resources/graph_migraphx_*' 'resources/export_report_mask_*' 'test/fixtures/graph_migraphx_*' 'tools/gpu_test/scenarios/*_rocm.sql' 'tools/export_*/src/*/*mask*.py'; do
	# shellcheck disable=SC2086
	if compgen -G "$glob" >/dev/null; then echo "FAIL: $glob still exists"; fail=1; fi
done

[ "$fail" = 0 ] && echo "OK: no ONNX Runtime and no removed backend in the extension, its build, CI, test tools or docs"
exit "$fail"
