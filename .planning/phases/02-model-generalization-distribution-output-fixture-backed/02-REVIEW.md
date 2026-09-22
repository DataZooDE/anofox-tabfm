---
phase: 02-model-generalization-distribution-output-fixture-backed
reviewed: 2026-09-22T09:00:00Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - src/tabfm_profile_registry.cpp
  - src/include/tabfm_profile_registry.hpp
  - src/tabfm_preprocess_tabpfn_v2.cpp
  - src/tabfm_engine.cpp
  - src/tabfm_ort_engine.cpp
  - src/include/tabfm_ort_engine.hpp
  - src/tabfm_predict_agg.cpp
  - src/include/tabfm_predict.hpp
  - src/tabfm_manifest.cpp
  - src/include/tabfm_manifest.hpp
  - src/tabfm_weights.cpp
  - src/tabfm_settings.cpp
  - src/anofox_tabfm_extension.cpp
  - src/include/tabfm_registration.hpp
  - CMakeLists.txt
  - test/sql/tabfm_distribution.test
  - test/sql/tabfm_license.test
  - test/sql/tabfm_profile_registry.test
  - test/cpp/test_tabfm_distribution_decode.cpp
  - test/cpp/test_tabfm_profile_registry.cpp
  - tools/parity/src/parity/check_tabpfn_v2.py
  - tools/parity/src/parity/build_fixture.py
findings:
  critical: 2
  warning: 3
  info: 4
  total: 9
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-09-22T09:00:00Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

This phase adds: a preprocessing-profile registry (self-registering, function-local-static
singleton), the `tabpfn_v2` fixture-scoped preprocessing profile, distribution-output decode
(FullSupportBarDistribution mean + quantiles over non-uniform borders), a generic
manifest-keyed license gate, and the K=16 weight-free tabpfn_v2 fixture.

Most of the implementation is sound. The registry pattern is correctly structured (no
static-init order fiasco, linker-stripping defended by `ForceProfileInit`, fail-closed unknown
profile). The distribution decode math (softmax, non-uniform CDF interpolation,
`sqrt(pi/2)` half-normal outer-bin correction, z-space→raw-space affine transform) is
correct and well-tested. The license gate is correctly layered, backward-compatible, and
`fixture-mit` is genuinely ungated via `IsGated()`. The fixture is weight-free
(random-init only, no Google bytes).

Two blocking defects are present:

1. A profile-id mismatch means every predict call against the built-in tabfm-v1 weights
   (real-model path) immediately throws an "unknown preprocessing profile" error. Tests pass
   only because all test fixtures override `anofox_tabfm_model_manifest` to a file that uses
   the correct id (`tabfm_v1_minimal`). Production users who download weights and run the
   built-in model get a misleading error.

2. `ValidateDistributionOutput` is documented and named with `n_test` semantics but is called
   with `T` (total rows, train + test). The fixture graph is deliberately built to output
   `[T, K]` for all rows, so this works today. Any future distribution model that follows the
   documented `[n_test, K]` contract (test rows only — the true TabPFN v2 wire format) will be
   rejected by the validator, and the decode loop will scatter predictions into wrong positions.

---

## Critical Issues

### CR-01: Built-in manifests use `"tabfm-v1"` but the profile registry only knows `"tabfm_v1_minimal"`

**File:** `src/tabfm_manifest.cpp:288,307`
**Cross-reference:** `src/include/tabfm_preprocess.hpp:70`, `src/tabfm_profile_registry.cpp:70`

**Issue:** The two built-in manifest JSON literals ship `"preprocessing_profile": "tabfm-v1"`.
The profile registry registers the v1 implementation under the key `kPreprocessProfileId =
"tabfm_v1_minimal"`. These strings do not match. Any call to `tabfm_classify` or
`tabfm_regress` against the production built-in weights (no custom manifest set) reaches
`DispatchPreprocess("tabfm-v1", ...)`, finds nothing in the registry, and throws
`InvalidInputException("tabfm: preprocessing profile 'tabfm-v1' is not registered ...")`.

All existing tests escape this because every test file calls
`SET anofox_tabfm_model_manifest = 'test/fixtures/.../manifest.json'` before predicting, and
every test fixture uses `"preprocessing_profile": "tabfm_v1_minimal"`. The mismatch is
invisible in CI.

**Fix:** Align the profile id in all three places. Either:
- Change `kPreprocessProfileId` in `src/include/tabfm_preprocess.hpp:70` to `"tabfm-v1"` (dash), OR
- Change both built-in manifest literals in `src/tabfm_manifest.cpp:288` and `:307` to
  `"tabfm_v1_minimal"` (underscore, no dash).

The fixture manifests already use `"tabfm_v1_minimal"` so option 2 is internally consistent
and avoids touching any header constant that other code may depend on:

```cpp
// src/tabfm_manifest.cpp line 288 and 307 — change:
"preprocessing_profile": "tabfm-v1",
// to:
"preprocessing_profile": "tabfm_v1_minimal",
```

Also add a static-assert or registry-lookup check in a startup smoke-test to catch this class
of mismatch in future (a test that calls `tabfm_regress` without overriding the manifest
setting would do).

---

### CR-02: `ValidateDistributionOutput` parameter is named `n_test` but called with `T` (train + test); contract inversion breaks future real-model graphs

**File:** `src/tabfm_ort_engine.cpp:597-626`, `src/tabfm_engine.cpp:868`
**Cross-reference:** `src/include/tabfm_ort_engine.hpp:251-254`, `src/include/tabfm_ort_engine.hpp:215`

**Issue:** `ValidateDistributionOutput(const TabFMRunOutput &out, idx_t n_test)` checks
`out.shape[0] == n_test`. The header documents the distribution model output shape as
`"[n_test,K] for distribution models (tabpfn_v2)"` — test rows only. The call site is:

```cpp
// tabfm_engine.cpp:868
ValidateDistributionOutput(out, T);   // T = batch.T = train_size + n_test
```

The fixture graph is intentionally built to emit `[T, K]` logits for ALL rows (train + test),
which means today the call passes. But:

1. The TabPFN v2 real tensor contract (per the SPIKE and the Python reference) outputs logits
   only for test rows (`[n_test, K]`), not all rows. A graph that follows the documented
   contract will be rejected.
2. Even with the fixture, if a user's `train_size` differs from what the graph sees, `T`
   passed to `ValidateDistributionOutput` may not equal `out.shape[0]`, producing a confusing
   mismatch error.
3. The decode loop at `tabfm_engine.cpp:901` iterates `for (idx_t t = 0; t < T; t++)` and
   reads `out.logits[t * K + k]`. If a real-model graph outputs `[n_test, K]` and this call
   path is reached, the loop will read out of bounds for the `n_test < T` rows.

**Fix:** Decide and document the canonical shape contract, then make the code match it
consistently. The simplest fix that matches the documented header contract (test-rows-only
output):

```cpp
// tabfm_engine.cpp — pass n_test, not T
const idx_t n_test = T - batch.train_size;
ValidateDistributionOutput(out, n_test);
// ... change the loop:
for (idx_t t = 0; t < n_test; t++) {
    const idx_t src = batch.row_source_index[batch.train_size + t]; // test rows only
```

If the design intent is to keep `[T, K]` output (all rows including train):
- Rename the parameter from `n_test` to `n_all_rows` in `ValidateDistributionOutput`
- Update `tabfm_ort_engine.hpp` line 215 and line 251 comments to say `[T, K]` not `[n_test, K]`
- Update `build_fixture.py` and `check_tabpfn_v2.py` comments to say the fixture emits `[T, K]`
- Add a test that a graph emitting only `[n_test, K]` is explicitly rejected with a clear error

Whichever choice is made, the function signature name, the header comment, the call site, and
the fixture must all agree.

---

## Warnings

### WR-01: `StageBundledGraph` writes ONNX graph bytes using `std::ofstream`, bypassing DuckDB VFS

**File:** `src/tabfm_engine.cpp:433`

**Issue:** The bundled external-data graph is staged to disk using `std::ofstream`:

```cpp
std::ofstream out(graph_path, std::ios::binary | std::ios::trunc);
out.write(graph.data, NumericCast<std::streamsize>(graph.size));
out.close();
return static_cast<bool>(out);
```

This bypasses the DuckDB `FileSystem` abstraction used everywhere else in the codebase.
On systems where the cache directory is on a virtual filesystem (e.g., remote or in-memory
filesystems configured via DuckDB's VFS), the write would silently write to the host
filesystem instead. The read path (`WeightsHeaderMatches`, `TryMapFile`) correctly uses either
`FileSystem` or direct POSIX calls but only for local files, so a mixed behavior emerges.

**Fix:** Replace with DuckDB `FileSystem` write:

```cpp
bool StageBundledGraph(FileSystem &fs, const BundledResource &graph, const string &graph_path) {
    try {
        if (fs.FileExists(graph_path)) {
            auto h = fs.OpenFile(graph_path, FileFlags::FILE_FLAGS_READ);
            if (NumericCast<idx_t>(fs.GetFileSize(*h)) == graph.size) {
                return true;
            }
        }
    } catch (...) {}
    try {
        auto h = fs.OpenFile(graph_path,
            FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
        h->Write(const_cast<void*>(static_cast<const void*>(graph.data)),
                 NumericCast<int64_t>(graph.size));
        h->Sync();
        return true;
    } catch (...) {
        return false;
    }
}
```

The caller `TryExternalDataSession` and `TryMIGraphXBackend` must also thread the `FileSystem`
reference through (currently they use `fs` from the outer scope — it can be passed directly).

---

### WR-02: `PrepareSessionOptions` holds a function-local `static` `Ort::MemoryInfo` used concurrently

**File:** `src/tabfm_ort_engine.cpp:358`

**Issue:**

```cpp
static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
```

This is initialized once (thread-safe per C++11 §6.7), but `PrepareSessionOptions` can be
called concurrently from multiple DuckDB worker threads when different groups are loading
sessions simultaneously (the per-device lock is held only around `LoadOrGetSession`, not
`PrepareSessionOptions` for separate devices). If `Ort::MemoryInfo` is not a const/immutable
value after construction — its internal ORT C API pointer could be read concurrently while
`Ort::Value::CreateTensor` uses it — this is a data race under the C++ memory model even if
ORT's own threading is correct.

**Fix:** Move the `memory_info` instantiation to a stack variable inside the loop, or document
that ORT guarantees `MemoryInfo` is immutable after construction (cite the ORT docs/source).
The object is small so a per-call stack instantiation has negligible cost:

```cpp
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
for (auto &tensor : initializers) {
    ...
    values.push_back(Ort::Value::CreateTensor(memory_info, ...));
}
```

---

### WR-03: `DistributionQuantile` does not guard against `q` outside `(0, 1)`

**File:** `src/tabfm_engine.cpp:689-713`

**Issue:** The function is called only with `kQuantileLevels` values (all in (0,1)) from
production code, but the public declaration in `tabfm_predict.hpp:206` advertises `q in (0, 1)`
without enforcing it. A caller passing `q = 0.0` causes `q <= curr_cum` to be true on the
first iteration when `probs[0] > 0`, returning `borders[0] + 0 = borders[0]` — numerically
harmless but semantically the 0th-percentile is the leftmost border by convention. A caller
passing `q < 0` produces a negative interpolation offset, returning a value left of
`borders[0]` (logically wrong output with no error). A caller passing `q > 1` falls through
to the final `return borders[K]` (the rightmost border).

The Catch2 test and the SQL test do not exercise `q <= 0` or `q >= 1`.

**Fix:** Add a guard at the top of `DistributionQuantile`:

```cpp
double DistributionQuantile(const vector<double> &probs, const vector<double> &borders, double q) {
    const size_t K = probs.size();
    if (K == 0 || borders.size() != K + 1) {
        return 0.0;
    }
    if (q <= 0.0) {
        return borders[0];
    }
    if (q >= 1.0) {
        return borders[K];
    }
    // ... existing body ...
}
```

---

## Info

### IN-01: `ModelManifest::distribution_output` is parsed and stored but never consulted in the engine dispatch

**File:** `src/tabfm_engine.cpp:807`, `src/include/tabfm_manifest.hpp:85-88`

**Issue:** The `distribution_output` boolean in `ModelManifest` is documented as controlling
`ValidateDistributionOutput` vs. `ValidateTabFMOutput` dispatch (header comment). In practice,
the engine ignores it and dispatches solely on `!out.borders.empty()` (line 807). A manifest
author who sets `"distribution_output": false` while connecting a graph that emits borders will
still enter the distribution decode path. Conversely, the field is listed in `manifest.hpp` as
a dispatch control, misleading future contributors.

**Fix (editorial):** Either (a) enforce the field in the dispatch:

```cpp
const bool model_emits_dist = resolved.manifest.distribution_output
    && !out.borders.empty() && task == TabFMTask::REGRESSION;
```

or (b) remove the field from the manifest schema and update all comments to say dispatch is
purely based on `!out.borders.empty()`. Option (a) is safer: it closes a path where a
regression model that accidentally emits a borders output (graph bug) silently enters
distribution decode.

---

### IN-02: `ValidateDistributionOutput` parameter name `n_test` and header comment say test-rows-only, but both the header doc (`tabfm_ort_engine.hpp:215`) and the function doc (`tabfm_ort_engine.hpp:251`) disagree with each other

**File:** `src/include/tabfm_ort_engine.hpp:215,251-254`

**Issue:** Line 215 says `"[n_test,K] for distribution models"`. Lines 251-254 say
`"shape must be rank-2 [n_test, K]"`. Neither comment mentions that the current fixture emits
`[T, K]` for all rows and that `ValidateDistributionOutput` is called with `T`. Future
developers reading the header will implement a `[n_test, K]` graph and be surprised.

**Fix:** Update the header comments to precisely state which shape the current code validates
against (`[T, K]` for all rows in the context). This is a documentation-only change that
prevents future graph authors from building conformant-but-rejected models.

---

### IN-03: `tabfm_weights.cpp` `ParseManifestJson` does not use RAII for the `yyjson_doc`

**File:** `src/tabfm_weights.cpp:168-212`

**Issue:** `tabfm_manifest.cpp` uses a `YyjsonDoc` RAII guard for all JSON parsing. The
private `ParseManifestJson` in `tabfm_weights.cpp` uses a raw `yyjson_doc *doc` with a manual
`yyjson_doc_free(doc)` before the final validation throws. The two `throw` statements at lines
205 and 209 come after the `free` at line 204, so today there is no leak. But the pattern is
fragile — adding a new field extraction between lines 179 and 204 without guarding could
introduce a leak.

**Fix:** Replace the manual free with the same `YyjsonDoc` RAII wrapper used in
`tabfm_manifest.cpp`:

```cpp
WeightsManifest ParseManifestJson(const string &json, const string &manifest_path) {
    using namespace duckdb_yyjson;
    yyjson_read_err err {};
    YyjsonDoc guard(yyjson_read_opts(const_cast<char*>(json.c_str()), json.size(),
                                     YYJSON_READ_NOFLAG, nullptr, &err));
    if (!guard.doc) { throw ...; }
    ...
    // no manual yyjson_doc_free needed
}
```

Note: `YyjsonDoc` is defined in `tabfm_manifest.cpp`'s anonymous namespace so it is not
visible from `tabfm_weights.cpp`. Either move it to a shared internal header, or duplicate
the trivial struct.

---

### IN-04: `tabfm_distribution.test` asserts training rows get distribution columns, but this is undocumented behavior

**File:** `test/sql/tabfm_distribution.test:77-83`

**Issue:** The test at lines 79-83 asserts that ALL rows (including `is_training = true`)
have non-NULL `logits`, `borders`, and `yhat_quantiles` when `output_mode = 'distribution'`.
This is consistent with the current implementation (the fixture graph outputs `[T, K]` for all
rows and the decode loop iterates over all T rows). However, the SQL-API comment in
`tabfm_predict.hpp` and the distribution feature documentation do not mention that training rows
receive decoded distribution columns. Users may be surprised that fitted values for training
rows include a full distributional predictive — especially since `is_training = true` rows
normally represent the context, not novel inputs.

This is not a code defect but a documentation gap. The test passing is evidence the behavior is
intentional; it should be explicitly noted in the SQL-API docs and the `tabfm_predict.hpp`
`TabFMPredictResult` struct comment alongside `yhat_dist_logits`.

---

_Reviewed: 2026-09-22T09:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
