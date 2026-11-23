PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# VCPKG setup: Set the VCPKG_TOOLCHAIN_PATH variable that the included makefile expects.
ifeq ($(VCPKG_ROOT),)
	export VCPKG_TOOLCHAIN_PATH ?= $(PROJ_DIR)/vcpkg_installed/$(VCPKG_TARGET_TRIPLET)/share/vcpkg/scripts/buildsystems/vcpkg.cmake
else
	export VCPKG_TOOLCHAIN_PATH ?= $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
endif

# Configuration of extension
EXT_NAME=anofox_tabfm
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# The CPU runs on the vendored ggml (vendor/ggml); GPUs are served by plugins built separately (tools/ggml_plugin for CUDA and ROCm, the
# anofox_tabfm_mlx_plugin target on Apple Silicon). Nothing here links ONNX Runtime.

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Force unified "build" directory regardless of generator (ninja or make).
# TABFM_BUILD_ROOT allows parallel agents/worktrees to keep isolated build
# dirs (concurrent ninja runs in one dir corrupt it).
BUILD_ROOT:=$(or $(TABFM_BUILD_ROOT),build)

# Override test targets to disable telemetry during test runs.
# This prevents local tests and CI/CD from polluting PostHog telemetry data.
#
# TWO invocations per test target, because one filter cannot match both: the
# sqllogictests register under their PATH ("test/sql/foo.test"), while the
# Catch2 TUs in TABFM_CPP_TEST_SOURCES register under their tag ([tabfm]).
# "test/*" alone silently skipped every C++ case in CI.
# No LSan suppressions: the one leak that needed them was inside ONNX Runtime, which no build links any more.
TABFM_TEST_ENV = DATAZOO_DISABLE_TELEMETRY=1

test_release_internal:
	$(TABFM_TEST_ENV) ./$(BUILD_ROOT)/release/test/unittest "test/*"
	$(TABFM_TEST_ENV) ./$(BUILD_ROOT)/release/test/unittest "[tabfm]"

test_debug_internal:
	$(TABFM_TEST_ENV) ./$(BUILD_ROOT)/debug/test/unittest "test/*"
	$(TABFM_TEST_ENV) ./$(BUILD_ROOT)/debug/test/unittest "[tabfm]"

test_reldebug_internal:
	$(TABFM_TEST_ENV) ./$(BUILD_ROOT)/reldebug/test/unittest "test/*"
	$(TABFM_TEST_ENV) ./$(BUILD_ROOT)/reldebug/test/unittest "[tabfm]"

# anofox-visualization picks up the contract producers (docs/ANOFOX_CONTRACT_PLAN.md): the
# release artifact next to a released anofox_visualization in the official DuckDB CLI.
test_visualization:
	python3 scripts/visualization_pickup.py $(BUILD_ROOT)/release/extension/anofox_tabfm/anofox_tabfm.duckdb_extension

# Docs and Claude Code skills (plugins/anofox-tabfm): run every ```sql example against the
# release build with the weight-free fixture standing in for each named model, and check
# function, setting and model names, the model tables and the plugin version
# (scripts/check_doc_snippets.py). check-docs-static needs no build.
check-docs:
	python3 scripts/check_doc_snippets.py --duckdb $(BUILD_ROOT)/release/duckdb

check-docs-static:
	python3 scripts/check_doc_snippets.py --static
