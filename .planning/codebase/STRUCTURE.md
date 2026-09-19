# Codebase Structure

**Analysis Date:** 2026-09-19

## Directory Layout

```
anofox-tabfm/
├── src/                          # Extension source code (14 .cpp modules + 11 headers)
│   ├── anofox_tabfm_extension.cpp # Entry point, extension loader, telemetry init
│   ├── tabfm_settings.cpp         # Settings registry (anofox_tabfm_*)
│   ├── tabfm_state.cpp            # Per-DB model cache (ObjectCache entry)
│   ├── tabfm_manifest.cpp         # Model manifest parsing (JSON → struct)
│   ├── tabfm_safetensors.cpp      # Weight file reader, memory arena
│   ├── tabfm_weights.cpp          # Download/cache lifecycle (FR-1/2/4)
│   ├── tabfm_devices.cpp          # Device discovery, tabfm_devices() function
│   ├── tabfm_ort_engine.cpp       # ONNX Runtime session management
│   ├── tabfm_migraphx.cpp         # Direct libMIGraphX backend (ROCm bypass)
│   ├── tabfm_preprocess.cpp       # Encoding/filtering/scaling (C++ port of Python)
│   ├── tabfm_ensemble.cpp         # Multi-estimator support (Phase 2)
│   ├── tabfm_predict_agg.cpp      # __anofox_tabfm_predict_agg aggregate
│   ├── tabfm_macros.cpp           # tabfm_classify/tabfm_regress macros
│   ├── tabfm_engine.cpp           # Integration layer (preprocess → ORT → decode)
│   └── include/                   # Public headers for each module
│       ├── anofox_function_alias.hpp    # Function alias macros
│       ├── anofox_tabfm_extension.hpp   # Extension class declaration
│       ├── tabfm_bundled_resources.hpp  # Embedded graphs/tensor maps
│       ├── tabfm_ensemble.hpp           # Ensemble structs/functions
│       ├── tabfm_manifest.hpp           # ModelManifest struct, parser
│       ├── tabfm_migraphx.hpp           # MIGraphX engine declarations
│       ├── tabfm_ort_engine.hpp         # ORT API wrapping, device info
│       ├── tabfm_predict.hpp            # Predict options, PredictEngine seam
│       ├── tabfm_preprocess.hpp         # PreprocessedBatch, encoders
│       ├── tabfm_registration.hpp       # Function registration entry points
│       ├── tabfm_safetensors.hpp        # SafetensorsView, weight access
│       └── tabfm_state.hpp              # TabFMState, LoadedModel
│
├── test/                          # Test suites (SQL + C++)
│   ├── sql/                       # SQLLogicTests (.test files run against DuckDB shell)
│   │   ├── settings.test          # anofox_tabfm_* settings tests
│   │   ├── devices.test           # tabfm_devices() function tests
│   │   ├── weights_lifecycle.test # tabfm_download/models/load/unload/remove tests
│   │   ├── preprocess_*.test      # Preprocessing pipeline tests
│   │   ├── predict_*.test         # Predict aggregate tests
│   │   └── macros_*.test          # Macro expansion tests
│   ├── cpp/                       # Catch2 unit tests
│   │   ├── test_tabfm_bundled_resources.cpp  # Graph/tensor-map loading
│   │   ├── test_tabfm_ensemble.cpp           # Ensemble-specific logic
│   │   ├── test_tabfm_manifest.cpp           # Manifest parsing/validation
│   │   ├── test_tabfm_ort_engine.cpp         # ORT session creation
│   │   ├── test_tabfm_preprocess.cpp         # Preprocessing pipeline, parity vs Python
│   │   ├── test_tabfm_safetensors.cpp        # Weight file parsing
│   │   └── test_tabfm_scaffold.cpp           # Test infrastructure
│   └── fixtures/                  # Golden test data
│       ├── ort_engine/            # ORT session fixtures (graphs, tensors)
│       ├── regression/            # Regression model fixtures
│       ├── weights_test/          # Mock weight files for testing
│       └── golden_preprocess.json # Bit-for-bit parity reference vs Python
│
├── resources/                     # Bundled weight-free resources
│   ├── graph_classification.onnx        # TabFM v1 classification graph (weight-free)
│   ├── graph_regression.onnx            # TabFM v1 regression graph (weight-free)
│   ├── graph_ext_*.onnx                 # Extended graphs (ORT format variations)
│   ├── graph_migraphx_*.onnx            # MIGraphX-compiled graphs
│   ├── tensor_map_*.json                # ONNX initializer → safetensors key mapping
│   └── export_report_*.json             # Export logs from WS-A (export_onnx tool)
│
├── tools/                         # Utility scripts (WS-A Python uv projects)
│   ├── export_onnx/              # Export TabFM to ONNX graphs (Python/uv project)
│   ├── make_fixture/             # Generate random-init test fixtures (Python/uv)
│   └── experimental/             # Experimental/developmental scripts
│
├── vendor/                        # Upstream dependencies (never edit, patch in tools/)
│   └── tabfm/                    # Apache-2.0 TabFM reference (github.com/google-research/tabfm)
│       └── tabfm/src/classifier_and_regressor.py  # Reference for preprocessing logic
│
├── cmake/                         # CMake configuration
│   ├── ort.cmake                  # ONNX Runtime flavor selection & fetching
│   └── embed_resources.cmake      # Bundled graph/tensor-map embedding into binary
│
├── duckdb/                        # DuckDB submodule (v1.5.4)
│   ├── src/                       # DuckDB source (used for extension API)
│   ├── test/                      # DuckDB test suite
│   └── ...
│
├── extension-ci-tools/            # CI infrastructure (duckdb/extension-ci-tools submodule)
│   ├── vcpkg_ports/              # Custom vcpkg ports (ort-vcpkg, etc.)
│   └── ...
│
├── posthog-telemetry/             # Telemetry library (submodule)
│   ├── include/telemetry.hpp      # PostHog telemetry API
│   ├── src/telemetry.cpp          # Implementation
│   └── test/cpp/                  # Telemetry tests
│
├── vcpkg_ports/                   # Local vcpkg port overrides
│   └── ort-vcpkg/                # ONNX Runtime build-from-source manifest (optional)
│
├── examples/                      # Runnable SQL examples
│   ├── classification_churn.sql   # Churn prediction (scikit-learn dataset)
│   ├── classification_iris.sql    # Iris classification
│   ├── regression_tips.sql        # Tip amount regression
│   └── README.md                  # Example guide + results
│
├── docs/                          # Additional documentation
│   ├── WINDOWS_INFERENCE.md       # Windows onnxruntime.dll loading notes
│   └── ...
│
├── .github/                       # CI workflow definitions
│   └── workflows/
│
├── cmake/                         # Build system config
├── CMakeLists.txt                 # Root CMake file
├── extension_config.cmake         # DuckDB extension config
├── Makefile                       # Convenience build targets
├── vcpkg.json                     # vcpkg manifest (dependencies)
├── README.md                      # User-facing quickstart + API reference
├── CHANGELOG.md                   # Version history
├── CLAUDE.md                      # Agent guide (this project's working rules)
├── LICENSE                        # MIT license
│
└── .planning/
    └── codebase/                  # Generated analysis documents (this agent writes here)
        ├── ARCHITECTURE.md        # System design, layers, data flow
        ├── STRUCTURE.md           # Directory layout, file purposes (this file)
        ├── CONVENTIONS.md         # Code style, naming, patterns
        ├── TESTING.md             # Test framework, structure, patterns
        ├── STACK.md               # Languages, frameworks, dependencies
        └── INTEGRATIONS.md        # External services, SDKs, APIs
```

## Directory Purposes

**`src/`:**
- Purpose: Extension implementation (one module per `.cpp` file)
- Contains: C++ source + public headers
- Key files: `anofox_tabfm_extension.cpp` (entry), `tabfm_registration.hpp` (declarations), module `.cpp` files
- Rule: One developer per module; no file sharing between workstreams

**`src/include/`:**
- Purpose: Public module interfaces (seams for testing/integration)
- Contains: Header-only APIs, struct definitions, enum declarations
- Pattern: `#pragma once` guards, no implementation details, pure interface

**`test/sql/`:**
- Purpose: SQLLogicTests (run via DuckDB shell with `./unittest test/sql/settings.test`)
- Contains: `.test` files with SQL statements + expected output + validation rules
- Tool: DuckDB's built-in SQLLogicTest parser (part of extension-ci-tools v1.5-variegata)
- Run via: `make test_debug` or `make test_release`

**`test/cpp/`:**
- Purpose: Catch2 unit tests (C++ fixtures, parity checks, integration tests)
- Contains: `.cpp` files with `#include "catch.hpp"`, compiled into `build/debug/test/unittest` binary
- Declare in: `CMakeLists.txt` via `TABFM_CPP_TEST_SOURCES`
- Run via: `./build/debug/test/unittest [filter]`

**`test/fixtures/`:**
- Purpose: Golden test data (models, weights, preprocessed outputs)
- Contains: Committed fixture data verified against sha256 pinned in CI
- Never regenerated in CI (immutable reference for parity checks)

**`resources/`:**
- Purpose: Bundled weight-free graphs and tensor maps embedded into the binary
- Contains: `.onnx` files (weight-free computation graphs), `.json` tensor maps
- Embedded via: `cmake/embed_resources.cmake` (generates `src/tabfm_bundled_resources.cpp`)
- Never downloaded: Part of the extension binary, no companion files needed after `tabfm_download()`

**`tools/`:**
- Purpose: WS-A build tools (Python uv projects)
- Contains: `export_onnx` (TabFM → ONNX exporter), `make_fixture` (random-init fixture generator), `parity` (Python reference comparison)
- These are NOT part of the C++ extension; they run offline to generate `resources/`

**`vendor/`:**
- Purpose: Upstream reference code (read-only, never edit)
- Contains: `vendor/tabfm` Apache-2.0 TabFM source
- Usage: Reference for preprocessing logic; C++ implementation lives in `src/tabfm_preprocess.cpp`
- Patches: Applied via tool scripts, never directly edited

**`cmake/`:**
- Purpose: Build configuration
- Key files: `ort.cmake` (flavor/ORT setup), `embed_resources.cmake` (graph bundling)

**`duckdb/` and `extension-ci-tools/` (submodules):**
- Purpose: DuckDB core + CI infrastructure
- Do not edit: Only used as build dependencies
- Update via git submodule commands if needed

**`examples/`:**
- Purpose: Runnable SQL examples with expected outputs
- Contains: `.sql` files + README documenting datasets + results
- Run via: `duckdb < examples/classification_churn.sql` (after `LOAD anofox_tabfm`)

## Key File Locations

**Entry Points:**
- `src/anofox_tabfm_extension.cpp`: DuckDB extension loader hook, telemetry init, registration orchestrator
- `src/tabfm_registration.hpp`: Declaration of all `RegisterXxxFunctions()` entry points
- `src/anofox_tabfm_extension.hpp`: Extension class declaration (minimal, forwards to LoadInternal)

**Configuration:**
- `src/tabfm_settings.cpp`: `RegisterTabfmSettings()` registers `anofox_tabfm_*` options
- `src/include/tabfm_state.hpp`: `TabFMState` ObjectCache entry (per-database model cache)

**Core Logic:**
- `src/tabfm_predict_agg.cpp`: `__anofox_tabfm_predict_agg` aggregate (bind/update/combine/finalize triplet)
- `src/tabfm_engine.cpp`: `PredictEngine::Predict()` integration layer (orchestrates preprocessing → ORT → decode)
- `src/tabfm_preprocess.cpp`: `PreprocessBatch()` function (encode/filter/scale/clip)
- `src/tabfm_ort_engine.cpp`: ORT session creation, device discovery, forward pass

**Testing:**
- `test/sql/settings.test`: Settings validation
- `test/sql/predict_*.test`: Predict aggregate behavior
- `test/cpp/test_tabfm_preprocess.cpp`: Parity checks vs Python reference (golden_preprocess.json)
- `test/cpp/test_tabfm_manifest.cpp`: Manifest parsing validation

## Naming Conventions

**Files:**
- `tabfm_*.cpp` / `tabfm_*.hpp`: Core modules (one module per pair)
- `anofox_*.cpp` / `anofox_*.hpp`: Extension-level or shared infrastructure
- `test_tabfm_*.cpp`: Catch2 unit tests
- `*.test`: SQLLogicTest files (run via DuckDB shell)
- Resource files: `graph_*.onnx`, `tensor_map_*.json` (variants like `_classification`, `_ext`, `_migraphx`)

**Functions:**
- `anofox_tabfm_*`: Full SQL function names (registered)
- `tabfm_*`: Alias names via `anofox_function_alias.hpp` macros
- `Register*()`: Module registration functions (declared in `tabfm_registration.hpp`, called from extension loader)
- `Predict::*`: Prefixed with module (e.g., `PreprocessBatch::encode`, `TabFMState::Register`)

**Variables & Classes:**
- `TabFMXxx`: Major struct/class names (e.g., `TabFMState`, `TabFMPredictOptions`)
- `k*`: Constants (e.g., `kTargetPadSentinel = -100.0`)
- `TABFM_*`: Preprocessor defines for flavors/features (e.g., `TABFM_EP_CUDA`, `TABFM_FLAVOR`)

**Settings:**
- `anofox_tabfm_*`: Full setting names (all options start with this prefix)
- Example: `anofox_tabfm_device`, `anofox_tabfm_cache_dir`, `anofox_tabfm_threads`

## Where to Add New Code

**New SQL Function (e.g., `tabfm_foo()`):**
1. Implement in a new file `src/tabfm_foo.cpp` (or reuse an existing module if closely related)
2. Define registration function `void RegisterFooFunctions(ExtensionLoader &loader)` in that file
3. Declare in `src/include/tabfm_registration.hpp`
4. Call from `src/anofox_tabfm_extension.cpp` LoadInternal()
5. Add tests to `test/sql/foo.test` (SQLLogicTest) and optionally `test/cpp/test_tabfm_foo.cpp` (Catch2)

**New Module (e.g., WS-X feature layer):**
1. Create `src/tabfm_ws_x.cpp` + `src/include/tabfm_ws_x.hpp`
2. Public header exports the module's seam (e.g., struct definitions, function signatures)
3. Implement in `.cpp`, include only necessary DuckDB + other module headers
4. Define registration function; declare in `tabfm_registration.hpp`
5. Add C++ tests in `test/cpp/test_tabfm_ws_x.cpp` (fixtures in `test/fixtures/`)

**New Test (SQL):**
- Location: `test/sql/<area>.test`
- Format: SQLLogicTest syntax (statements, results, validation rules)
- Run: `./build/debug/test/unittest test/sql/<area>.test`

**New Test (C++):**
- Location: `test/cpp/test_<module>.cpp`
- Include: `#include "catch.hpp"` + module headers
- Pattern: `TEST_CASE("description") { ... REQUIRE(...); }`
- Compilation: Declare in CMakeLists.txt `TABFM_CPP_TEST_SOURCES`
- Run: `./build/debug/test/unittest [test_<module>]`

**New Setting:**
- Add to `src/tabfm_settings.cpp` RegisterTabfmSettings()
- Use `config.AddExtensionOption("anofox_tabfm_name", "...", type, default, callback)`
- Document in README.md settings table
- Test in `test/sql/settings.test`

**New Bundled Resource (graph, tensor map):**
- Place in `resources/`
- Reference in CMakeLists.txt indirectly via `cmake/embed_resources.cmake`
- The build system automatically generates `src/tabfm_bundled_resources.cpp`
- Access in code via functions from `tabfm_bundled_resources.hpp`

**New Example:**
- SQL file: `examples/<task>_<dataset>.sql`
- Document expected results in `examples/README.md`
- Include: LOAD instructions, dataset source, F1/accuracy/MSE results

## Special Directories

**`.planning/codebase/`:**
- Purpose: Generated analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated by: `/gsd-map-codebase` agent
- Committed: Yes (for reference)
- Edited by: Analysis agent only

**`build/`:**
- Purpose: Build artifacts (generated at compile time)
- Contains: `debug/` and `release/` subdirs with compiled binaries, objects, test executables
- Committed: No (in .gitignore)
- Generated by: CMake / `make debug|release`

**`.git/`:**
- Purpose: Git metadata
- Committed: Yes (repo tracking)

**`duckdb_unittest_tempdir/`:**
- Purpose: Temporary directory for test runtime state
- Created by: DuckDB test harness
- Committed: No

---

*Structure analysis: 2026-09-19*
