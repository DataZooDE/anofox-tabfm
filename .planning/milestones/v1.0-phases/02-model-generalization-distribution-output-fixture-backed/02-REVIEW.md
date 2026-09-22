---
phase: 02-model-generalization-distribution-output-fixture-backed
reviewed: 2026-09-22T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - src/tabfm_manifest.cpp
  - src/tabfm_ort_engine.cpp
  - src/include/tabfm_ort_engine.hpp
  - src/tabfm_engine.cpp
  - src/tabfm_profile_registry.cpp
  - src/tabfm_preprocess_tabpfn_v2.cpp
  - src/tabfm_predict_agg.cpp
  - src/tabfm_weights.cpp
  - src/tabfm_settings.cpp
  - test/cpp/test_tabfm_profile_registry.cpp
  - test/cpp/test_tabfm_distribution_decode.cpp
  - test/sql/tabfm_distribution.test
  - test/sql/tabfm_license.test
findings:
  critical: 1
  warning: 0
  info: 1
  total: 2
status: issues_found
---

# Phase 02: Code Review Report (Re-review)

**Reviewed:** 2026-09-22
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Re-review after targeted fixes for CR-01, CR-02, WR-01, WR-02, WR-03.

**All five prior fixes are confirmed correct and complete:**

- **CR-01** — Both built-in manifest JSON literals now carry `"preprocessing_profile": "tabfm_v1_minimal"`, which is exactly `kPreprocessProfileId`. The regression test (Test 3 in `test_tabfm_profile_registry.cpp`) calls `BuiltinTabFMManifest(TabFMTask::CLASSIFICATION/REGRESSION)` directly (not a fixture path), verifies the string equality, and then exercises the full registry dispatch path with `REQUIRE_NOTHROW`. The production path from built-in manifest to registry dispatch is end-to-end covered.

- **CR-02** — `ValidateDistributionOutput` parameter is now `n_all_rows`; header comment (`tabfm_ort_engine.hpp:264-267`) and implementation comments correctly document the fixture `[T,K]` contract and the deferred real-model `[n_test,K]` contract. The call site (`tabfm_engine.cpp:900`) passes `T` (all rows) as documented.

- **WR-01** — `StageBundledGraph` uses `FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW`, which maps to `O_CREAT | O_TRUNC` on POSIX and the Windows equivalent. `LocalFileSystem::Write` retries partial writes in a loop and throws `IOException` on error; the outer `catch(...)` block correctly falls back to the injection path on any failure. `h->Sync()` is called before returning `true`. The directory is always valid because `StageBundledGraph` is only reached after `ResolveWeightsPath` succeeds (so `DirName(weights_path)` exists).

- **WR-02** — `Ort::MemoryInfo` is now stack-local in both `PrepareSessionOptions` and `Run`. `Ort::Value::CreateTensor` takes `const OrtMemoryInfo*` via `Base<T>::operator contained_type*()` implicit conversion; the `MemoryInfo` object outlives all `CreateTensor` calls and the subsequent `session.Run()` within each function. The fix eliminates the theoretical data-race on the former static instance without introducing a lifetime hazard.

- **WR-03** — `DistributionQuantile` guards `q <= 0.0 → borders[0]` and `q >= 1.0 → borders[K]` before any loop. Production callers only pass values from `kQuantileLevels` (all strictly in `(0,1)`) so this path exists only for out-of-contract callers.

**One new critical finding** was uncovered during this re-review: `ValidateDistributionOutput` still does not validate `out.logits.size() == n_all_rows * K`, while `ValidateTabFMOutput` performs the equivalent check (lines 592-599). The `DecodeDistribution` loop at `tabfm_engine.cpp:939` indexes `out.logits[t * K + k]` without any bounds guard, making it reachable as an out-of-bounds access if the ORT output tensor element count does not match the declared shape.

## Critical Issues

### CR-01: `ValidateDistributionOutput` does not check `logits.size() == T * K`

**File:** `src/tabfm_ort_engine.cpp:602-635`
**Issue:** `ValidateDistributionOutput` verifies that `out.shape` is rank-2 `[n_all_rows, K]` and that `out.borders.size() == K + 1`, but it never asserts that `out.logits.size() == n_all_rows * K`. The immediately following `DecodeDistribution` in `tabfm_engine.cpp:939` indexes `out.logits[t * K + k]` for `t` in `[0, T)` and `k` in `[0, K)` with no additional guard. An ORT model that declares shape `[T, K]` but returns fewer elements — possible with a corrupt or non-conformant graph — would result in a read beyond the vector bounds rather than a clean exception.

Compare `ValidateTabFMOutput` (lines 592-599) which guards the exact same invariant for the standard path:
```cpp
const idx_t expected = expected_t * C;
if (out.logits.size() != expected) {
    throw InvalidInputException(
        "anofox_tabfm: model returned %llu logits but the [1, %llu, %llu] output requires %llu ...",
        ...);
}
```

**Fix:** Add the size check inside `ValidateDistributionOutput` after the existing borders check:

```cpp
const size_t expected_logits =
    static_cast<size_t>(n_all_rows) * static_cast<size_t>(K);
if (out.logits.size() != expected_logits) {
    throw InvalidInputException(
        "anofox_tabfm: distribution model returned %llu logits but shape [%llu, %lld] requires %llu — "
        "truncated or malformed model output. Check the manifest and SET anofox_tabfm_model_manifest.",
        static_cast<unsigned long long>(out.logits.size()),
        static_cast<unsigned long long>(n_all_rows),
        static_cast<long long>(K),
        static_cast<unsigned long long>(expected_logits));
}
```

Extend Test 4 in `test/cpp/test_tabfm_distribution_decode.cpp` with a case where shape is correct but `logits` is one element short:

```cpp
// Bad: shape claims [N, 8] but logits vector is truncated
{
    TabFMRunOutput bad;
    bad.shape = {static_cast<int64_t>(N), 8};
    bad.logits.assign(N * 8 - 1, 0.0f); // one element short
    bad.borders.assign(9, 0.0f);
    REQUIRE_THROWS_AS(ValidateDistributionOutput(bad, N), InvalidInputException);
}
```

## Info

### IN-01: Test 3 mixes classification manifest with `PreprocessTask::REGRESSION`

**File:** `test/cpp/test_tabfm_profile_registry.cpp:138`
**Issue:** The end-to-end dispatch check in Test 3 uses `cls_manifest.preprocessing_profile` (from the classification built-in manifest) but passes `PreprocessTask::REGRESSION` to `DispatchPreprocess`. There is no runtime bug — both built-in manifests share the same profile id (`tabfm_v1_minimal`) and `PreprocessBatch` accepts either task — but the combination looks unintentional and creates a small readability gap.

**Fix:** Either use `PreprocessTask::CLASSIFICATION` when dispatching via `cls_manifest.preprocessing_profile`, or add a second `REQUIRE_NOTHROW` call using `reg_manifest.preprocessing_profile` with `PreprocessTask::REGRESSION` to cover both.

---

_Reviewed: 2026-09-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
