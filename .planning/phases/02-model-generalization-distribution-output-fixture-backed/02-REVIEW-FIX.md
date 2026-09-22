---
phase: 02-model-generalization-distribution-output-fixture-backed
fixed_at: 2026-09-22T10:30:00Z
review_path: .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-09-22T10:30:00Z
**Source review:** `.planning/phases/02-model-generalization-distribution-output-fixture-backed/02-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (2 Critical + 3 Warning; 4 Info excluded per fix_scope=critical_warning)
- Fixed: 5
- Skipped: 0

**Verification:** Build and tests ran in the main checkout after fast-forwarding the
`gsd-reviewfix/02-4180253` worktree branch. All 387 assertions passed in 15 test cases
(`make test_debug`, `DATAZOO_DISABLE_TELEMETRY=1`).

---

## Fixed Issues

### CR-01: Built-in manifests use `"tabfm-v1"` but the profile registry only knows `"tabfm_v1_minimal"`

**Files modified:** `src/tabfm_manifest.cpp`, `test/cpp/test_tabfm_profile_registry.cpp`
**Commit:** `70446f1`
**Applied fix:** Changed both built-in manifest JSON literals in `BUILTIN_TABFM_V1_CLASSIFICATION`
and `BUILTIN_TABFM_V1_REGRESSION` from `"preprocessing_profile": "tabfm-v1"` to
`"preprocessing_profile": "tabfm_v1_minimal"`. This aligns the production built-in manifest path
with the registry key (`kPreprocessProfileId = "tabfm_v1_minimal"`).

Added a new Catch2 test case `"built-in manifests use the registered preprocessing profile id (CR-01)"`
in `test/cpp/test_tabfm_profile_registry.cpp`. The test:
1. Parses both built-in manifests via `BuiltinTabFMManifest()` (same path as production)
2. Asserts `preprocessing_profile == kPreprocessProfileId` for both tasks
3. Dispatches through the registry with the built-in profile id (end-to-end regression guard)

Also added `#include "tabfm_manifest.hpp"` to the test file to expose `BuiltinTabFMManifest()`.

---

### CR-02: `ValidateDistributionOutput` named `n_test` but called with `T`; contract inversion

**Files modified:** `src/include/tabfm_ort_engine.hpp`, `src/tabfm_ort_engine.cpp`, `src/tabfm_engine.cpp`
**Commit:** `d14c54b`
**Applied fix:** Adopted the `[T, K]` (all-rows) design — this matches the current fixture
semantics (the decode loop iterates all T rows) and avoids breaking the working distribution tests.

Changes:
- `src/include/tabfm_ort_engine.hpp` line 215 comment: updated `[n_test,K]` to `[T,K]` with a
  NOTE explaining that the real TabPFN v2 wire format emits `[n_test, K]` and what must change
  when a real-export graph is integrated.
- `ValidateDistributionOutput` declaration: renamed parameter `n_test` to `n_all_rows` with an
  extended doc comment that clearly states the current `[T,K]` contract, references the deferred
  real-model `[n_test,K]` contract, and says what to change when integrating a real export.
- `src/tabfm_ort_engine.cpp` implementation: renamed `n_test` to `n_all_rows` in the function
  signature, the condition, and the error message text. Updated the inline comment.
- `src/tabfm_engine.cpp` `DecodeDistribution`: added a block comment documenting the current
  `[T,K]` contract, the deferred real-model contract, and the exact code change needed at
  integration time. Updated the call-site comment from `[n_test, K]` to `[T, K]`.

This also resolves IN-01 (advisory note added) and IN-02 (header doc mismatch fixed) as side effects.

---

### WR-01: `StageBundledGraph` writes ONNX graph bytes using `std::ofstream`, bypassing DuckDB VFS

**Files modified:** `src/tabfm_engine.cpp`
**Commit:** `a22d913`
**Applied fix:** Replaced the `std::ofstream` write path with DuckDB `FileSystem` VFS calls,
consistent with the cache I/O pattern in `tabfm_weights.cpp`. The fix:
- Opens the file via `fs.OpenFile(graph_path, FILE_FLAGS_WRITE | FILE_FLAGS_FILE_CREATE_NEW)`
- Calls `h->Write()` (with a safe const-cast since `Write` does not mutate the buffer)
- Calls `h->Sync()` before returning
- Wraps the write path in `try/catch(...)` returning `false` on failure (matches the fallback
  contract the callers expect)
- Adds a comment explaining why VFS is used (uniformity, future VFS-override safety)

Also removed the now-unused `#include <fstream>` from the translation unit.

Decision note: the graph_path is always inside the local cache directory, so in practice this
lands on `LocalFileSystem`. Using the VFS abstraction is intentional — future VFS overrides
(e.g., an in-memory filesystem in tests) will see the write consistently.

---

### WR-02: `PrepareSessionOptions` holds a function-local `static` `Ort::MemoryInfo` used concurrently

**Files modified:** `src/tabfm_ort_engine.cpp`
**Commit:** `c0f7f88`
**Applied fix:** Removed the `static` qualifier from `memory_info`, making it a stack variable
instantiated on each call. Added a comment explaining: `Ort::MemoryInfo` is a lightweight value
type (wraps a single OrtMemoryInfo* handle), construction cost is negligible, and moving it off
static storage eliminates the theoretical data-race from concurrent `PrepareSessionOptions` calls
for different devices.

---

### WR-03: `DistributionQuantile` does not guard against `q` outside `(0, 1)`

**Files modified:** `src/tabfm_engine.cpp`
**Commit:** `d726c67`
**Applied fix:** Added boundary guards immediately after the empty-input check:
```cpp
if (q <= 0.0) { return borders[0]; }
if (q >= 1.0) { return borders[K]; }
```
Added a comment noting that production callers always pass `kQuantileLevels` (all strictly in
`(0,1)`) and this guard is for out-of-contract callers. The clamp-to-boundary convention is
consistent with the existing fallthrough-to-`borders[K]` at the end of the loop (numerical
rounding).

---

## Skipped Issues

None.

---

_Fixed: 2026-09-22T10:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
