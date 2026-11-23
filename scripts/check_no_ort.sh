#!/usr/bin/env bash
# The extension, its build, its CI, its GPU plugin and its test tools contain no ONNX Runtime and no trace of the removed multi-backend
# design (MIGraphX, the ORT-CUDA plugin, CoreML, build flavors, shape buckets): the CPU runs on ggml, CUDA/ROCm on the ggml GPU plugin,
# Apple GPUs on MLX. Python exporters and tools/ggml_parity keep onnxruntime as the reference for goldens and parity checks; CHANGELOG and
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

code_paths=(src CMakeLists.txt cmake Makefile extension_config.cmake test/cpp test/sql .github/workflows tools/ggml_plugin tools/gpu_test scripts resources)
code_grep() { grep -rnE "$1" "${code_paths[@]}" 2>/dev/null | grep -v '^scripts/check_no_ort.sh:'; }

ort='onnxruntime|Ort::|TABFM_WITH_ORT|TABFM_FLAVOR|TABFM_EP_[A-Z]|libanofoxort_gpu|TABFM_ORT_[A-Z]'
# Exporter provenance (the onnxruntime version the parity oracle ran) and the release-asset guard naming what must not ship are fine.
hits=$(code_grep "$ort" | grep -vE '^resources/export_report_[^:]*\.json:|^resources/README\.md:[0-9]+:- \*\*Exporter versions|^scripts/check_plugin_release_assets\.py:[0-9]+:for ort in')
[ -n "$hits" ] && report "ONNX Runtime is still referenced" "$hits"

legacy='MIGraphX|libanofox_tabfm_cuda_plugin|libanofox_tabfm_migraphx_plugin|ExecutionProvider|\.mxr\b|PadToShapeBucket|ShapeBucket|TryExternalDataSession|OrtBackend|graph_migraphx_|make_migraphx_graph|TABFM_DISABLE_EXTERNAL_DATA|CoreML|cpu flavor'
hits=$(code_grep "$legacy" | grep -viE "$allowed")
[ -n "$hits" ] && report "the removed multi-backend design is still referenced" "$hits"

doc_paths=(README.md CLAUDE.md docs plugins .claude-plugin)
doc_legacy='libanofox_tabfm_cuda_plugin|libanofox_tabfm_migraphx_plugin|libanofoxort_gpu|ExecutionProvider|TABFM_FLAVOR|TABFM_ORT_|\.mxr\b|onnxruntime-gpu|graph_migraphx_|make_migraphx_graph|TABFM_DISABLE_EXTERNAL_DATA|tabfm_ort_engine'
hits=$(grep -rnE "$doc_legacy" "${doc_paths[@]}" 2>/dev/null | grep -v '^docs/archive/' | grep -viE "$allowed")
[ -n "$hits" ] && report "the docs still describe the removed backends" "$hits"

gone=(cmake/ort.cmake src/tabfm_cuda_plugin.cpp src/tabfm_migraphx_plugin.cpp src/tabfm_ort_engine.cpp src/include/tabfm_soname_patch.hpp
	src/include/tabfm_mxr_cache_key.hpp src/include/tabfm_shape_bucket.hpp test/cpp/test_tabfm_cuda_plugin.cpp
	test/cpp/test_tabfm_migraphx_plugin.cpp test/lsan.supp tools/make_migraphx_graph.py tools/check_migraphx_int_div.py
	tools/gpu_test/ort_ep_check.py tools/gpu_test/equivalence.py tools/experimental/mgx_smoke.cpp .github/workflows/graph_invariants.yml)
for g in "${gone[@]}"; do
	[ -e "$g" ] && { echo "FAIL: $g still exists"; fail=1; }
done
for glob in 'resources/graph_migraphx_*' 'resources/export_report_mask_*' 'test/fixtures/graph_migraphx_*'; do
	# shellcheck disable=SC2086
	if compgen -G "$glob" >/dev/null; then echo "FAIL: $glob still exists"; fail=1; fi
done

[ "$fail" = 0 ] && echo "OK: no ONNX Runtime and no removed backend in the extension, its build, CI, plugin, test tools or docs"
exit "$fail"
