# Coding Conventions

**Analysis Date:** 2026-09-19

## Naming Patterns

**Files:**
- C++ source/header pairs: `tabfm_*.cpp` + `src/include/tabfm_*.hpp` (one module = one source file)
- Full function names: `anofox_tabfm_*` (e.g., `anofox_tabfm_classify`)
- Settings: `anofox_tabfm_*` (e.g., `anofox_tabfm_device`, `anofox_tabfm_cache_dir`)
- Workstream prefixes in comments: WS-A through WS-F denote functional ownership areas
- Test files: `test/cpp/test_*.cpp` or `test/sql/*.test` (SQL files are DuckDB sqllogictest format)

**Functions:**
- PascalCase for public class/struct methods: `CreateSession()`, `Initializers()`, `RunInput()`
- snake_case for free functions: `cast_f32_to_f16()`, `discover_devices()`, `infer_task()`
- Private functions in anonymous namespace: use snake_case, always namespaced
- Getter/setter conventions: `GetOrtEnv()`, `SetEnabled()`, `SetAPIKey()`

**Variables:**
- camelCase for local variables and parameters: `isTraining`, `trainSize`, `catMask`
- UPPER_SNAKE_CASE for compile-time constants: `kNsPerDay` (prefixed with `k`)
- Member variables: snake_case with no prefix: `device_id`, `vram_total`, `device_ordinal`
- Loop counters: `i`, `j`, `k` (single letters acceptable in tight loops)

**Types:**
- PascalCase for classes, structs, enums: `TabFMDeviceInfo`, `TabFMTensorDtype`, `PreprocessTask`
- `_t` suffix for typedef'd scalar types inherited from DuckDB: `idx_t`, `int64_t`
- Enum class members: UPPER_SNAKE_CASE: `TabFMTensorDtype::F32`, `TabFMTensorDtype::BF16`

## Code Style

**Formatting:**
- clang-format (LLVM style, inherited from DuckDB submodule)
- Config: `.clang-format` (inherited from `duckdb/.clang-format`)
- Tab width: 4 spaces, use tabs for indentation
- Line limit: 120 characters
- Pointer alignment: right-aligned (`*` attached to type, not name)

**Linting:**
- Implicit: follows LLVM/DuckDB conventions, no explicit linter config
- Banned symbols checked by DuckDB's `banned_symbols_check.py` during build

## Import Organization

**Order in C++ files:**
1. `#define DUCKDB_EXTENSION_MAIN` (if entry point: `src/anofox_tabfm_extension.cpp`)
2. Project headers: `#include "anofox_*.hpp"` or `#include "tabfm_*.hpp"`
3. DuckDB system headers: `#include "duckdb/..."`
4. External SDK headers: `#include <onnxruntime_cxx_api.h>`, `#include <openssl/...>`
5. Standard library: `#include <algorithm>`, `#include <cmath>`, `#include <map>`
6. Optional OS-specific: `#include <dlfcn.h>` (wrapped in `#if defined()`)

**Namespace:**
- All extension code: `namespace duckdb { namespace anofox { ... } }`
- Anonymous namespaces for private (file-local) helpers and constants
- Using declarations: none in headers; `using namespace duckdb::anofox;` acceptable in .cpp files

**Path aliases:**
- No CMake path aliases; all includes are explicit relative to `src/include/` (added to include path in CMakeLists.txt)

## Error Handling

**Patterns:**
- Always throw DuckDB exceptions: `InvalidInputException`, `IOException`, `InternalException`
- Exception messages always name the user-facing fix (SQL-API §5 requirement):
  ```cpp
  throw InvalidInputException("anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm'..., got '%s'", value);
  ```
- Settings validators throw `InvalidInputException` with the setting name and constraint
- ORT API failures wrapped with actionable text: `"the ONNX Runtime loaded at runtime is version %s, too old for this build"`
- No silent failures or returns of error codes; exceptions propagate to DuckDB exception handling

**Examples:**
- `src/tabfm_settings.cpp`: `ValidateDevice()`, `ValidateTraceLevel()` (all throw with the setting name)
- `src/tabfm_devices.cpp:422`: `throw InvalidInputException(...)` for invalid device selections
- `src/include/tabfm_ort_engine.hpp` comments: lifecycle contract (ORT buffer ownership rules)

## Logging

**Framework:** DuckDB's native logging (no external log library)

**Patterns:**
- Telemetry via PostHog (not stderr logs): `PostHogTelemetry::Instance().CaptureFunctionExecution("<short_name>")`
- Telemetry invoked ONCE per user-facing function at bind time or global state init
- Trace levels controlled via `SET anofox_tabfm_trace_level = {error|warn|info|debug|trace}`
- Errors always include the full error message (no silent swallowing)
- Comments in code document INVARIANTS and LIFETIME CONTRACTS (see `tabfm_ort_engine.cpp:1-22`)

**Example:**
- `src/anofox_tabfm_extension.cpp:19-26`: Telemetry opt-out detection (DATAZOO_DISABLE_TELEMETRY env var)
- `src/tabfm_manifest.cpp`: Manifest loading with explicit error messages

## Comments

**When to Comment:**
- **Invariants and contracts:** Always document non-obvious memory ownership, lifetime rules
- **Algorithm references:** Link to upstream Python or academic papers (e.g., `vendor/tabfm/src/classifier_and_regressor.py`)
- **Integration points:** Mark where WS-A (tools) / WS-B (readers) / WS-C (engine) boundaries touch
- **Workarounds:** Explain why a "wrong-looking" pattern is necessary (e.g., ORT API version guard)
- **Tests:** Name the spike or spec section the code implements (e.g., `S02`, `HLD §4.4`, `SQL-API §3`)

**JSDoc/TSDoc:**
- No JSDoc in C++; use `///` for top-level function/struct declarations in headers
- Example: `src/include/tabfm_ort_engine.hpp` file header documents the module's invariants (lines 1-23)

**Style:**
- Use `//===------` dividers to separate logical sections within a file
- File-level comments at the top explain workstream ownership and spec references
- Inline comments explain the "why", not the "what"

## Function Design

**Size:** 
- Prefer single-responsibility functions under 50 lines; OK to exceed for complex numeric code
- Example: `F32ToF16Bits()` (35 lines, bitwise conversion) vs. `ValidateDevice()` (15 lines, parameter check)

**Parameters:**
- Use const references for large objects: `const vector<TabFMTensorRef> &initializers`
- Pass simple types by value: `idx_t count`, `PreprocessTask task`
- Out parameters via references when necessary: `double &epoch_ns_out` (see `DatetimeDayIndex()`)

**Return Values:**
- Return by value for owned objects: `vector<uint16_t> CastF32ToF16(...)`
- Return by reference only for static/stable data: `Ort::Env &GetOrtEnv()`
- Use Optional-like patterns: `bool FileExists()` or throw on missing file (`REQUIRE(file.good())` in tests)

## Module Design

**Exports:**
- Headers in `src/include/` define public module interfaces (included by other modules or tests)
- One public header per module: `#pragma once` guard, full documentation
- Implementation details (.cpp) in `src/`, never reached directly

**Barrel Files:**
- No barrel exports; each module's header is self-contained
- The main extension registration (`src/include/anofox_tabfm_extension.hpp`) includes specific modules as needed

**Aliases:**
- SQL function aliases managed via `anofox_function_alias.hpp` helpers
- Aliases are FULL COPIES with only the name changed (preserves all metadata)
- Example: `anofox_tabfm_classify` (primary) + `tabfm_classify` (alias registered via `RegisterTableFunctionWithAlias`)

## Conventions Specific to This Codebase

**Workstream Ownership:**
- Module comments identify workstream (WS-A through WS-F) for ownership tracking
- Example: `src/tabfm_ort_engine.cpp:1-8` names WS-C as the owner

**Spec References:**
- Code cites specific sections of the normative spec:
  - SQL-API: SQL function signatures and error messages (SQL-API §3-5)
  - HLD: Architecture, device discovery, preprocessing (HLD D9, §4.4, §9)
  - Spikes: S01-S06 document design decisions; spike RESULTS.md must be read before implementation
- Example: `src/tabfm_preprocess.cpp:1-17` references upstream Python and the preprocessing chain

**Telemetry and License Gates:**
- Every user-facing function must call `PostHogTelemetry::Instance().CaptureFunctionExecution(...)` exactly once
- Weight downloads require explicit `SET anofox_tabfm_accept_hf_license = true` (enforced in `src/tabfm_weights.cpp`)

**Fixture and Test Data:**
- Committed fixtures in `test/fixtures/` with deterministic sha256 verification in CI
- Golden files (golden.json, expected.json) documenting expected outputs
- Random-init weights to avoid embedding Google-licensed tensors

---

*Convention analysis: 2026-09-19*
