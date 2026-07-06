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

# Flavor selection (cpu | cuda | rocm), see cmake/ort.cmake.
# The flavor decides which ONNX Runtime build is linked; sources are identical.
TABFM_FLAVOR ?= cpu
EXT_FLAGS += -DTABFM_FLAVOR=$(TABFM_FLAVOR)

# rocm flavor: install tree of an ONNX Runtime built with --use_migraphx.
TABFM_ORT_ROCM_DIR ?=
ifneq ($(TABFM_ORT_ROCM_DIR),)
EXT_FLAGS += -DTABFM_ORT_ROCM_DIR=$(TABFM_ORT_ROCM_DIR)
endif
# MIGraphX prefix for the direct GPU backend (rocm flavor; defaults to /opt/rocm).
TABFM_MIGRAPHX_DIR ?=
ifneq ($(TABFM_MIGRAPHX_DIR),)
EXT_FLAGS += -DTABFM_MIGRAPHX_DIR=$(TABFM_MIGRAPHX_DIR)
endif
# Optional mirror override for the prebuilt ORT archive (cpu/cuda flavors).
TABFM_ORT_URL ?=
ifneq ($(TABFM_ORT_URL),)
EXT_FLAGS += -DTABFM_ORT_URL=$(TABFM_ORT_URL)
endif

# Release/distribution builds (cpu flavor) link ONNX Runtime statically via the
# vcpkg "ort-vcpkg" manifest feature, so the loadable extension is a single
# self-contained file with no companion libonnxruntime.so — required for the
# DuckDB Community Extensions single-file distribution model. Local debug builds
# keep the fast prebuilt-archive path (dynamic). Override with TABFM_ORT_VCPKG=0.
TABFM_ORT_VCPKG ?= 1
ifeq ($(TABFM_FLAVOR),cpu)
ifeq ($(TABFM_ORT_VCPKG),1)
EXT_RELEASE_FLAGS += -DVCPKG_MANIFEST_FEATURES=ort-vcpkg
endif
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Force unified "build" directory regardless of generator (ninja or make).
# TABFM_BUILD_ROOT allows parallel agents/worktrees to keep isolated build
# dirs (concurrent ninja runs in one dir corrupt it).
BUILD_ROOT:=$(or $(TABFM_BUILD_ROOT),build)

# Override test targets to disable telemetry during test runs.
# This prevents local tests and CI/CD from polluting PostHog telemetry data.
test_release_internal:
	DATAZOO_DISABLE_TELEMETRY=1 ./$(BUILD_ROOT)/release/test/unittest "test/*"

test_debug_internal:
	DATAZOO_DISABLE_TELEMETRY=1 ./$(BUILD_ROOT)/debug/test/unittest "test/*"

test_reldebug_internal:
	DATAZOO_DISABLE_TELEMETRY=1 ./$(BUILD_ROOT)/reldebug/test/unittest "test/*"
