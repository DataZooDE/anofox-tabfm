<!-- GSD:project-start source:PROJECT.md -->

## Project

**anofox-tabfm**

A DuckDB C++ extension that embeds tabular foundation models (TabPFN-style
in-context learners) via ONNX Runtime, exposing zero-shot classification and
regression as SQL — no Python, no training loops. This milestone extends it from
a single-model prediction surface into an **evaluation-and-comparison platform**:
composable, model-agnostic metric primitives and cross-validation in SQL, plus
first-class support for multiple foundation-model families so users can measure
and compare model quality on their own tables.

**Core Value:** Users can trust and compare tabular-foundation-model predictions directly in
SQL — computing standard metrics and cross-validation on their own data, across
more than one model family — without leaving DuckDB.

### Constraints

- **Tech stack**: DuckDB v1.5.4 (pinned submodule), ONNX Runtime 1.23.2, C++ extension via extension-ci-tools v1.5-variegata — new code follows existing module conventions (CLAUDE.md)
- **Community-extension eligibility**: cpu flavor must have zero GPU code; metric/eval code must be flavor-independent and license-clean
- **License wall**: no Google/vendor weight bytes anywhere in the repo; fixtures random-init (S06)
- **Model-agnostic metrics**: eval primitives operate on plain `(actual, predicted)` columns so they're useful independent of tabfm output
- **File ownership**: one module = one `src/tabfm_*.cpp` (+ headers); scaffold-owned shared files require coordination (CLAUDE.md rule #2)
- **TDD**: red-green — failing sqllogictest/Catch2 test first, then implement; error-path tests are first-class
- **Telemetry**: every user-facing function calls `CaptureFunctionExecution` once per execution
- **Dependency**: proper scoring rules for regression are blocked until a model emits predictive distributions (TabPFN v2 provides this natively)

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- C++ 17 - Core extension logic, model inference engine, preprocessing, ensemble operations
- SQL - DuckDB function interface for zero-shot tabular classification and regression
- Python 3.11+ - Utility tools for ONNX export, fixture generation, model validation
- CMake - Build system configuration and dependency management
- Bash - Makefile targets and build orchestration

## Runtime

- DuckDB 1.5.4 (pinned submodule) - Host database engine for the extension
- ONNX Runtime 1.23.2 - Inference engine for TabFM model execution
- CMake 3.10–3.29 - C++ build configuration
- vcpkg - C++ dependency management with overlay ports and triplets
- uv - Python project management for tools/export_onnx and tools/make_fixture
- `vcpkg.json` - Manifest-based vcpkg configuration with OpenSSL dependency
- `pyproject.toml` - uv projects for Python tools (Python 3.11+ requirement)
- Git submodules - DuckDB, extension-ci-tools, posthog-telemetry, vendor/tabfm

## Frameworks

- DuckDB Extension API - SQL function registration, scalar/aggregate function binding
- ONNX Runtime C++ API - Model graph loading, tensor management, inference session execution
- Catch2 - C++ unit testing (compiled into `build/debug/test/unittest` binary)
- SQLLogicTest - SQL regression testing (`.test` files in `test/sql/`)
- pytest - Python tool testing (fixtures, export validation)
- extension-ci-tools v1.5-variegata - DuckDB extension scaffold and build helpers
- CMake FetchContent - ONNX Runtime prebuilt archive download
- Hatchling - Python package builder for uv projects

## Key Dependencies

- OpenSSL 3.x - TLS/SSL for PostHog telemetry HTTP requests; required for OpenSSL SUPPORT in httplib
- ONNX Runtime (v1.23.2) - Foundation model inference; sourced either from vcpkg port or prebuilt archive
- cpp-httplib - HTTP client with OpenSSL support for telemetry API calls (`duckdb_httplib_openssl::Client`)
- safetensors - Model weight serialization format parser (C++ implementation in `src/tabfm_safetensors.cpp`)
- Postgres query parser - SQL preprocessing validation (embedded from DuckDB codebase)
- TabFM (Google Research) - Upstream Apache-2.0 submodule at `vendor/tabfm/`; weight-free model graphs only
- Test fixture model - Pre-built ONNX graph + safetensors weights committed in `test/fixtures/`

## Configuration

- `TABFM_FLAVOR` (default: `cpu`) - Selects ONNX Runtime execution provider: `cpu`, `cuda`, `rocm`
- `TABFM_ORT_ROCM_DIR` - Install tree path of ROCm flavor ORT build with MIGraphX
- `TABFM_MIGRAPHX_DIR` (default: `/opt/rocm`) - MIGraphX prefix for direct GPU backend
- `TABFM_ORT_URL` - Optional mirror override for prebuilt ORT archive downloads
- `TABFM_BUILD_ROOT` (default: `build`) - Parallel build directory isolation
- `DATAZOO_DISABLE_TELEMETRY` - Opt-out of PostHog telemetry; recognized values: `1`, `true`, `yes`
- `VCPKG_ROOT` - vcpkg installation root (optional; defaults to `vcpkg_installed` relative path)
- `VCPKG_TARGET_TRIPLET` - vcpkg target triplet for dependency resolution
- `CMakeLists.txt` - Main extension build configuration
- `cmake/ort.cmake` - Flavor-aware ONNX Runtime acquisition (vcpkg port or prebuilt archive)
- `cmake/embed_resources.cmake` - Bundled weight-free ONNX graphs + tensor maps (no companion files required at runtime)
- `extension_config.cmake` - Extension-specific CMake variables
- `Makefile` - High-level build targets; delegates to extension-ci-tools

## Platform Requirements

- C++ compiler (GCC 9+, Clang 10+, MSVC 2019+) with C++17 support
- CMake 3.10 or later
- OpenSSL development headers (system or vcpkg)
- vcpkg for C++ dependencies
- Python 3.11+ for tools (uv managed)
- macOS specific: IOKit + CoreFoundation frameworks for machine ID telemetry
- Windows specific: Windows IP Helper API (iphlpapi.dll, ws2_32.dll) for machine ID
- DuckDB 1.5.4 host instance
- ONNX Runtime library (`libonnxruntime.so`, `onnxruntime.dll`, or `libonnxruntime.dylib`) matching the flavor
- OpenSSL 3.x (for telemetry; optional if telemetry disabled)
- CUDA 11.x or 12.x (CUDA flavor only; runtime requirement, not link-time)
- ROCm + MIGraphX (ROCm flavor only; runtime requirement)

## Build Artifacts

- Static extension: `build/debug/anofox_tabfm_extension.a` or `anofox_tabfm_extension.lib`
- Loadable extension: `build/debug/anofox_tabfm_loadable_extension.so/dll/dylib`
- Unit tests binary: `build/debug/test/unittest`
- Test binaries link telemetry object files + OpenSSL
- `make test_debug` - Run C++ Catch2 tests + SQL logic tests with `DATAZOO_DISABLE_TELEMETRY=1`
- `make test_release` - Optimized test suite (same telemetry disable)
- Individual SQL test: `./build/debug/test/unittest test/sql/settings.test`

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- C++ source/header pairs: `tabfm_*.cpp` + `src/include/tabfm_*.hpp` (one module = one source file)
- Full function names: `anofox_tabfm_*` (e.g., `anofox_tabfm_classify`)
- Settings: `anofox_tabfm_*` (e.g., `anofox_tabfm_device`, `anofox_tabfm_cache_dir`)
- Workstream prefixes in comments: WS-A through WS-F denote functional ownership areas
- Test files: `test/cpp/test_*.cpp` or `test/sql/*.test` (SQL files are DuckDB sqllogictest format)
- PascalCase for public class/struct methods: `CreateSession()`, `Initializers()`, `RunInput()`
- snake_case for free functions: `cast_f32_to_f16()`, `discover_devices()`, `infer_task()`
- Private functions in anonymous namespace: use snake_case, always namespaced
- Getter/setter conventions: `GetOrtEnv()`, `SetEnabled()`, `SetAPIKey()`
- camelCase for local variables and parameters: `isTraining`, `trainSize`, `catMask`
- UPPER_SNAKE_CASE for compile-time constants: `kNsPerDay` (prefixed with `k`)
- Member variables: snake_case with no prefix: `device_id`, `vram_total`, `device_ordinal`
- Loop counters: `i`, `j`, `k` (single letters acceptable in tight loops)
- PascalCase for classes, structs, enums: `TabFMDeviceInfo`, `TabFMTensorDtype`, `PreprocessTask`
- `_t` suffix for typedef'd scalar types inherited from DuckDB: `idx_t`, `int64_t`
- Enum class members: UPPER_SNAKE_CASE: `TabFMTensorDtype::F32`, `TabFMTensorDtype::BF16`

## Code Style

- clang-format (LLVM style, inherited from DuckDB submodule)
- Config: `.clang-format` (inherited from `duckdb/.clang-format`)
- Tab width: 4 spaces, use tabs for indentation
- Line limit: 120 characters
- Pointer alignment: right-aligned (`*` attached to type, not name)
- Implicit: follows LLVM/DuckDB conventions, no explicit linter config
- Banned symbols checked by DuckDB's `banned_symbols_check.py` during build

## Import Organization

- All extension code: `namespace duckdb { namespace anofox { ... } }`
- Anonymous namespaces for private (file-local) helpers and constants
- Using declarations: none in headers; `using namespace duckdb::anofox;` acceptable in .cpp files
- No CMake path aliases; all includes are explicit relative to `src/include/` (added to include path in CMakeLists.txt)

## Error Handling

- Always throw DuckDB exceptions: `InvalidInputException`, `IOException`, `InternalException`
- Exception messages always name the user-facing fix (SQL-API §5 requirement):
- Settings validators throw `InvalidInputException` with the setting name and constraint
- ORT API failures wrapped with actionable text: `"the ONNX Runtime loaded at runtime is version %s, too old for this build"`
- No silent failures or returns of error codes; exceptions propagate to DuckDB exception handling
- `src/tabfm_settings.cpp`: `ValidateDevice()`, `ValidateTraceLevel()` (all throw with the setting name)
- `src/tabfm_devices.cpp:422`: `throw InvalidInputException(...)` for invalid device selections
- `src/include/tabfm_ort_engine.hpp` comments: lifecycle contract (ORT buffer ownership rules)

## Logging

- Telemetry via PostHog (not stderr logs): `PostHogTelemetry::Instance().CaptureFunctionExecution("<short_name>")`
- Telemetry invoked ONCE per user-facing function at bind time or global state init
- Trace levels controlled via `SET anofox_tabfm_trace_level = {error|warn|info|debug|trace}`
- Errors always include the full error message (no silent swallowing)
- Comments in code document INVARIANTS and LIFETIME CONTRACTS (see `tabfm_ort_engine.cpp:1-22`)
- `src/anofox_tabfm_extension.cpp:19-26`: Telemetry opt-out detection (DATAZOO_DISABLE_TELEMETRY env var)
- `src/tabfm_manifest.cpp`: Manifest loading with explicit error messages

## Comments

- **Invariants and contracts:** Always document non-obvious memory ownership, lifetime rules
- **Algorithm references:** Link to upstream Python or academic papers (e.g., `vendor/tabfm/src/classifier_and_regressor.py`)
- **Integration points:** Mark where WS-A (tools) / WS-B (readers) / WS-C (engine) boundaries touch
- **Workarounds:** Explain why a "wrong-looking" pattern is necessary (e.g., ORT API version guard)
- **Tests:** Name the spike or spec section the code implements (e.g., `S02`, `HLD §4.4`, `SQL-API §3`)
- No JSDoc in C++; use `///` for top-level function/struct declarations in headers
- Example: `src/include/tabfm_ort_engine.hpp` file header documents the module's invariants (lines 1-23)
- Use `//===------` dividers to separate logical sections within a file
- File-level comments at the top explain workstream ownership and spec references
- Inline comments explain the "why", not the "what"

## Function Design

- Prefer single-responsibility functions under 50 lines; OK to exceed for complex numeric code
- Example: `F32ToF16Bits()` (35 lines, bitwise conversion) vs. `ValidateDevice()` (15 lines, parameter check)
- Use const references for large objects: `const vector<TabFMTensorRef> &initializers`
- Pass simple types by value: `idx_t count`, `PreprocessTask task`
- Out parameters via references when necessary: `double &epoch_ns_out` (see `DatetimeDayIndex()`)
- Return by value for owned objects: `vector<uint16_t> CastF32ToF16(...)`
- Return by reference only for static/stable data: `Ort::Env &GetOrtEnv()`
- Use Optional-like patterns: `bool FileExists()` or throw on missing file (`REQUIRE(file.good())` in tests)

## Module Design

- Headers in `src/include/` define public module interfaces (included by other modules or tests)
- One public header per module: `#pragma once` guard, full documentation
- Implementation details (.cpp) in `src/`, never reached directly
- No barrel exports; each module's header is self-contained
- The main extension registration (`src/include/anofox_tabfm_extension.hpp`) includes specific modules as needed
- SQL function aliases managed via `anofox_function_alias.hpp` helpers
- Aliases are FULL COPIES with only the name changed (preserves all metadata)
- Example: `anofox_tabfm_classify` (primary) + `tabfm_classify` (alias registered via `RegisterTableFunctionWithAlias`)

## Conventions Specific to This Codebase

- Module comments identify workstream (WS-A through WS-F) for ownership tracking
- Example: `src/tabfm_ort_engine.cpp:1-8` names WS-C as the owner
- Code cites specific sections of the normative spec:
- Example: `src/tabfm_preprocess.cpp:1-17` references upstream Python and the preprocessing chain
- Every user-facing function must call `PostHogTelemetry::Instance().CaptureFunctionExecution(...)` exactly once
- Weight downloads require explicit `SET anofox_tabfm_accept_hf_license = true` (enforced in `src/tabfm_weights.cpp`)
- Committed fixtures in `test/fixtures/` with deterministic sha256 verification in CI
- Golden files (golden.json, expected.json) documenting expected outputs
- Random-init weights to avoid embedding Google-licensed tensors

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Extension Entry** | DuckDB extension loader hook, telemetry init | `src/anofox_tabfm_extension.cpp` |
| **Registration** | Function/macro registration entry points | `src/include/tabfm_registration.hpp` |
| **Settings** | `anofox_tabfm_*` configuration options | `src/tabfm_settings.cpp` |
| **State** | Per-database model cache (ObjectCache) | `src/tabfm_state.cpp`, `src/include/tabfm_state.hpp` |
| **Manifest** | Model manifest parsing (JSON → struct) | `src/tabfm_manifest.cpp`, `src/include/tabfm_manifest.hpp` |
| **Safetensors** | Weight file reader, buffer arena | `src/tabfm_safetensors.cpp`, `src/include/tabfm_safetensors.hpp` |
| **Weights Lifecycle** | Download, cache, models(), load, unload, remove | `src/tabfm_weights.cpp` |
| **Device Discovery** | GPU detection, device selection, tabfm_devices() | `src/tabfm_devices.cpp`, `src/include/tabfm_ort_engine.hpp` |
| **ORT Engine** | ONNX Runtime session management, tensor injection | `src/tabfm_ort_engine.cpp`, `src/include/tabfm_ort_engine.hpp` |
| **MIGraphX** | Direct libMIGraphX backend for ROCm (bypasses ORT EP) | `src/tabfm_migraphx.cpp`, `src/include/tabfm_migraphx.hpp` |
| **Preprocessing** | Encode/filter/scale/clip (C++ port of Python logic) | `src/tabfm_preprocess.cpp`, `src/include/tabfm_preprocess.hpp` |
| **Ensemble** | Multi-estimator support, ensemble tensor prep | `src/tabfm_ensemble.cpp`, `src/include/tabfm_ensemble.hpp` |
| **Predict Core** | `__anofox_tabfm_predict_agg` aggregate function | `src/tabfm_predict_agg.cpp`, `src/include/tabfm_predict.hpp` |
| **Predict Macros** | `tabfm_classify` / `tabfm_regress` table macros | `src/tabfm_macros.cpp` |
| **Predict Engine** | Integration layer: preprocess → ORT → decode | `src/tabfm_engine.cpp` |

## Pattern Overview

- **Module-per-file:** One module = one `src/tabfm_*.cpp` + headers in `src/include/`. Multiple workstreams never touch the same file (CLAUDE.md rule #2).
- **Seam-based testing:** Each module exports a public interface (seams in headers) that downstream modules depend on; tests inject via those seams.
- **Single-threaded aggregate:** DuckDB aggregate functions execute the Update/Combine/Finalize triplet; concurrency is per-device via mutexes during finalize (ORT Run is serialized per device, groups accumulate in parallel).
- **Flavor-aware compilation:** One codebase, four builds (cpu/cuda/rocm/coreml) selected by `TABFM_FLAVOR`. GPU code paths compile out of cpu flavor (community-extension eligible).
- **Opaque state wrapping:** `TabFMState` holds `shared_ptr<void>` sessions (actual `SessionHolder` erased) so state layer never depends on ORT headers.

## Layers

- Purpose: Accept SQL syntax, bind arguments, type inference, macro expansion
- Location: `src/tabfm_macros.cpp` (user-facing), `src/tabfm_predict_agg.cpp` (aggregate bind/update/combine/finalize)
- Contains: Macro body SQL, bind callbacks (argument checking, output schema computation), aggregate function triplet
- Depends on: DuckDB API, telemetry
- Used by: DuckDB query executor
- Purpose: Parse `opts` MAP, task inference from target column type, feature selection
- Location: `src/include/tabfm_predict.hpp` (TabFMPredictOptions), `src/tabfm_predict_agg.cpp` (bind logic)
- Contains: Options struct, validation rules, task type enum
- Depends on: DuckDB types
- Used by: Predict aggregate bind
- Purpose: Settings registry, per-database model cache, device mutex tracking
- Location: `src/tabfm_settings.cpp`, `src/tabfm_state.cpp`
- Contains: `anofox_tabfm_*` options, `TabFMState` object cache entry, loaded model snapshots
- Depends on: DuckDB config/catalog/object cache
- Used by: Predict finalize (model snapshot), devices discovery, weight lifecycle
- Purpose: Parse model metadata (JSON), resolve file URLs, layout cache directories
- Location: `src/tabfm_manifest.cpp`, `src/tabfm_weights.cpp`
- Contains: `ModelManifest` struct (task/repo/revision/files/graph/license), cache layout, download manifest
- Depends on: yyjson (JSON parser), filesystem
- Used by: Weight download, model loading
- Purpose: Download files from HF, verify checksums, manage cache; read safetensors headers and allocate weight buffers
- Location: `src/tabfm_weights.cpp` (download/cache), `src/tabfm_safetensors.cpp` (parse/allocate)
- Contains: `SafetensorsView` (memory arena with tensor accessors), download with resume/restart, checksum validation
- Depends on: DuckDB FileSystem (httpfs integration), filesystem ops
- Used by: Model load (Layer 6)
- Purpose: Assemble manifest + weights + graph + tensor injection; create ORT session
- Location: `src/tabfm_engine.cpp` (SessionHolder, integration), `src/tabfm_ort_engine.cpp` (ORT calls)
- Contains: Manifest file resolution, graph loading (bundled or custom), weight buffer injection by tensor name, device resolution
- Depends on: Manifest (Layer 4), Safetensors (Layer 5), ORT engine (Layer 7)
- Used by: Predict finalize (cached in TabFMState)
- Purpose: ONNX Runtime environment, session management, device probing, dtype casting
- Location: `src/tabfm_ort_engine.cpp` (ORT API wrapping), `src/tabfm_devices.cpp` (discovery), `src/tabfm_migraphx.cpp` (direct libMIGraphX)
- Contains: `Ort::Env` singleton, session creation, execution provider selection, device info structs, GPU dtype conversion
- Depends on: ONNX Runtime library (flavor-specific), filesystem (device file probing)
- Used by: Model loading (Layer 6), predict finalize (forward pass)
- Purpose: C++ port of TabFM's minimal encoding/filtering/scaling logic (bit-for-bit match to Python reference)
- Location: `src/tabfm_preprocess.cpp`
- Contains: `PreprocessedBatch` (x/y/cat_mask matrices, label decoder, target scaling), column encoding (categorical/numeric/datetime), unique-feature filter, outlier removal
- Depends on: DuckDB ColumnDataCollection (input rows), manifest (preprocessing profile id validation)
- Used by: Predict finalize (feature matrix assembly)
- Purpose: Multi-estimator support, ensemble tensor layout
- Location: `src/tabfm_ensemble.cpp`
- Contains: Ensemble-specific preprocessing (currently n_estimators > 1 is rejected)
- Depends on: Preprocessing (Layer 8)
- Used by: Predict finalize (future)
- Purpose: Classification softmax/argmax, regression inverse-transform, scatter predictions back to input row order
- Location: `src/tabfm_engine.cpp` (decode step)
- Contains: Label lookup, softmax temperature application, regression scaling inverse, output struct assembly
- Depends on: Preprocessing (layer decoder info), ORT output tensors
- Used by: Predict finalize (last step before returning results)

## Data Flow

### Primary Request Path (Single-Relation Predict)

### Train/Test Explicit Form

- Macro splits data/test into two subqueries (data rows + test rows with NULL label)
- Finalize returns only test rows (no is_training filter needed; training rows never in the frame)

### State Management

- **Model snapshots:** Lazy-loaded on first predict, cached in `TabFMState` object cache (never evicted by LRU — user-controlled lifecycle via SQL)
- **Weight buffers:** Held in `SafetensorsView` arena, kept alive by `SessionHolder` for the session lifetime (ORT reads lazily during inference)
- **Device mutex:** Per-device serialization during finalize (ORT::Session::Run); groups accumulate concurrently before finalize

## Key Abstractions

- Purpose: Parsed and validated engine options (task, n_estimators, seed, output_mode, context_rows, softmax_temperature, model)
- Examples: `src/include/tabfm_predict.hpp` (struct def), `src/tabfm_predict_agg.cpp` (parsing)
- Pattern: All values arrive as VARCHAR (SQL-API Level 2 spec), validated to enum/int at bind time
- Purpose: Structured metadata for one (model, task) pair
- Examples: `src/include/tabfm_manifest.hpp`, built-in manifests in `src/tabfm_manifest.cpp`
- Pattern: JSON → parsed and strictly validated; unknown fields ignored (forward compat)
- Purpose: Fully preprocessed feature matrix, ready for ORT feeding
- Examples: `src/include/tabfm_preprocess.hpp`, output of `PreprocessBatch()` function
- Pattern: Row-major [T, H] matrices; train rows first, test rows last; encoders kept for introspection
- Purpose: DB-instance-level model cache, opaque session wrapping
- Examples: `src/include/tabfm_state.hpp`, object cache entry
- Pattern: `LoadedModel.session` is `shared_ptr<void>` (actually `SessionHolder` erased) to avoid ORT header dependency in state layer
- Purpose: Mutable memory arena holding all weights, indexable by tensor name
- Examples: `src/include/tabfm_safetensors.hpp`
- Pattern: Allocates one contiguous arena, parses safetensors header to build name → offset map

## Entry Points

- Location: `src/anofox_tabfm_extension.cpp` AnofoxTabfmExtension::Load()
- Triggers: DuckDB loads the extension (dynamic or static)
- Responsibilities: Register telemetry options, settings, function/macro registrations
- `tabfm_classify(data, target [, test] [, features] [, opts])` → Table macro
- `tabfm_regress(data, target [, test] [, features] [, opts])` → Table macro
- `__anofox_tabfm_predict_agg(row STRUCT, target VARCHAR [, opts MAP])` → Aggregate function (internal)
- `tabfm_download(task)` → Procedure
- `tabfm_models()` → Table function
- `tabfm_load(task)` → Procedure
- `tabfm_unload()` → Procedure
- `tabfm_remove(task)` → Procedure
- `tabfm_devices()` → Table function

## Architectural Constraints

- **Threading:** Single-threaded event loop (DuckDB aggregate triplet). ORT Run serialized per device via `DeviceMutex()` during finalize; groups accumulate concurrently before finalize.
- **Global state:** One process-wide `Ort::Env` (ONNX Runtime), one per-database `TabFMState` in ObjectCache. PreCompiled GPU programs cached on disk (`.mxr` for ROCm).
- **Circular imports:** Avoided via seam pattern: each module's public interface lives in a header, implementation in a `.cpp`; includes flow one direction (leaf → root).
- **Model eviction:** Never by LRU; only by explicit SQL `CALL tabfm_unload()`. If already loaded, new load replaces (old one marked evicted, freed when last snapshot releases).
- **Weight buffer lifetime:** Must outlive ORT::Session (ORT reads lazily during Run). Caller keeps `SafetensorsView` arena alive via `SessionHolder`.
- **Flavor isolation:** GPU code paths guarded by `#ifdef TABFM_EP_CUDA / TABFM_EP_MIGRAPHX / TABFM_EP_COREML`. CPU flavor has zero GPU code (community-extension requirement).

## Anti-Patterns

### Circular Manifest Dependency

### Skipping Preprocessing

### Mutating Weight Buffers

## Error Handling

- Model not downloaded: `"tabfm: model 'classification' is not downloaded. Run: CALL tabfm_download('classification');"`
- License not accepted: `"tabfm: non-commercial license must be explicitly accepted. Run: SET anofox_tabfm_accept_hf_license = true;"`
- Device not found: `"tabfm: device 'rocm:1' is not available. Run: SELECT * FROM tabfm_devices();"`
- Max rows exceeded: `"tabfm: max_rows guardrail exceeded (10000 rows seen). Run: SET anofox_tabfm_max_rows = ...;"`

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
