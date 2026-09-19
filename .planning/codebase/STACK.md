# Technology Stack

**Analysis Date:** 2026-09-19

## Languages

**Primary:**
- C++ 17 - Core extension logic, model inference engine, preprocessing, ensemble operations
- SQL - DuckDB function interface for zero-shot tabular classification and regression
- Python 3.11+ - Utility tools for ONNX export, fixture generation, model validation

**Secondary:**
- CMake - Build system configuration and dependency management
- Bash - Makefile targets and build orchestration

## Runtime

**Environment:**
- DuckDB 1.5.4 (pinned submodule) - Host database engine for the extension
- ONNX Runtime 1.23.2 - Inference engine for TabFM model execution
  - Multi-flavor support: CPU (default), CUDA, ROCm (MIGraphX)
  - Prebuilt archives from GitHub Releases (configurable via `TABFM_ORT_URL`)
  - vcpkg port available for build-from-source (`ort-vcpkg` feature)

**Package Manager:**
- CMake 3.10–3.29 - C++ build configuration
- vcpkg - C++ dependency management with overlay ports and triplets
- uv - Python project management for tools/export_onnx and tools/make_fixture

**Lockfile:**
- `vcpkg.json` - Manifest-based vcpkg configuration with OpenSSL dependency
- `pyproject.toml` - uv projects for Python tools (Python 3.11+ requirement)
- Git submodules - DuckDB, extension-ci-tools, posthog-telemetry, vendor/tabfm

## Frameworks

**Core:**
- DuckDB Extension API - SQL function registration, scalar/aggregate function binding
- ONNX Runtime C++ API - Model graph loading, tensor management, inference session execution

**Testing:**
- Catch2 - C++ unit testing (compiled into `build/debug/test/unittest` binary)
- SQLLogicTest - SQL regression testing (`.test` files in `test/sql/`)
- pytest - Python tool testing (fixtures, export validation)

**Build/Dev:**
- extension-ci-tools v1.5-variegata - DuckDB extension scaffold and build helpers
- CMake FetchContent - ONNX Runtime prebuilt archive download
- Hatchling - Python package builder for uv projects

## Key Dependencies

**Critical:**
- OpenSSL 3.x - TLS/SSL for PostHog telemetry HTTP requests; required for OpenSSL SUPPORT in httplib
- ONNX Runtime (v1.23.2) - Foundation model inference; sourced either from vcpkg port or prebuilt archive

**Infrastructure:**
- cpp-httplib - HTTP client with OpenSSL support for telemetry API calls (`duckdb_httplib_openssl::Client`)
- safetensors - Model weight serialization format parser (C++ implementation in `src/tabfm_safetensors.cpp`)
- Postgres query parser - SQL preprocessing validation (embedded from DuckDB codebase)

**Model & Data:**
- TabFM (Google Research) - Upstream Apache-2.0 submodule at `vendor/tabfm/`; weight-free model graphs only
- Test fixture model - Pre-built ONNX graph + safetensors weights committed in `test/fixtures/`

## Configuration

**Environment:**
- `TABFM_FLAVOR` (default: `cpu`) - Selects ONNX Runtime execution provider: `cpu`, `cuda`, `rocm`
- `TABFM_ORT_ROCM_DIR` - Install tree path of ROCm flavor ORT build with MIGraphX
- `TABFM_MIGRAPHX_DIR` (default: `/opt/rocm`) - MIGraphX prefix for direct GPU backend
- `TABFM_ORT_URL` - Optional mirror override for prebuilt ORT archive downloads
- `TABFM_BUILD_ROOT` (default: `build`) - Parallel build directory isolation
- `DATAZOO_DISABLE_TELEMETRY` - Opt-out of PostHog telemetry; recognized values: `1`, `true`, `yes`
- `VCPKG_ROOT` - vcpkg installation root (optional; defaults to `vcpkg_installed` relative path)
- `VCPKG_TARGET_TRIPLET` - vcpkg target triplet for dependency resolution

**Build:**
- `CMakeLists.txt` - Main extension build configuration
- `cmake/ort.cmake` - Flavor-aware ONNX Runtime acquisition (vcpkg port or prebuilt archive)
- `cmake/embed_resources.cmake` - Bundled weight-free ONNX graphs + tensor maps (no companion files required at runtime)
- `extension_config.cmake` - Extension-specific CMake variables
- `Makefile` - High-level build targets; delegates to extension-ci-tools

## Platform Requirements

**Development:**
- C++ compiler (GCC 9+, Clang 10+, MSVC 2019+) with C++17 support
- CMake 3.10 or later
- OpenSSL development headers (system or vcpkg)
- vcpkg for C++ dependencies
- Python 3.11+ for tools (uv managed)
- macOS specific: IOKit + CoreFoundation frameworks for machine ID telemetry
- Windows specific: Windows IP Helper API (iphlpapi.dll, ws2_32.dll) for machine ID

**Production:**
- DuckDB 1.5.4 host instance
- ONNX Runtime library (`libonnxruntime.so`, `onnxruntime.dll`, or `libonnxruntime.dylib`) matching the flavor
- OpenSSL 3.x (for telemetry; optional if telemetry disabled)
- CUDA 11.x or 12.x (CUDA flavor only; runtime requirement, not link-time)
- ROCm + MIGraphX (ROCm flavor only; runtime requirement)

## Build Artifacts

**Output Targets:**
- Static extension: `build/debug/anofox_tabfm_extension.a` or `anofox_tabfm_extension.lib`
- Loadable extension: `build/debug/anofox_tabfm_loadable_extension.so/dll/dylib`
- Unit tests binary: `build/debug/test/unittest`
- Test binaries link telemetry object files + OpenSSL

**Test Modes:**
- `make test_debug` - Run C++ Catch2 tests + SQL logic tests with `DATAZOO_DISABLE_TELEMETRY=1`
- `make test_release` - Optimized test suite (same telemetry disable)
- Individual SQL test: `./build/debug/test/unittest test/sql/settings.test`

---

*Stack analysis: 2026-09-19*
