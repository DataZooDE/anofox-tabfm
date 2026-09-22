---
phase: 02-model-generalization-distribution-output-fixture-backed
fixed_at: 2026-09-22T19:10:00Z
review_path: .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-REVIEW.md
iteration: 2
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report (Iteration 2)

**Fixed at:** 2026-09-22T19:10:00Z
**Source review:** `.planning/phases/02-model-generalization-distribution-output-fixture-backed/02-REVIEW.md`
**Iteration:** 2

**Summary:**
- Findings in scope (critical + warning): 1
- Fixed: 1
- Skipped: 0

Note: IN-01 (Test 3 task-enum mismatch) was also addressed in this pass as it was trivial (one-line change) and co-located with the test file touched for CR-01's test addition.

---

## Fixed Issues

### CR-01: `ValidateDistributionOutput` does not check `logits.size() == T * K`

**Files modified:** `src/tabfm_ort_engine.cpp`, `test/cpp/test_tabfm_distribution_decode.cpp`, `test/cpp/test_tabfm_profile_registry.cpp`
**Commit:** `effadf1`
**Applied fix:**

1. **`src/tabfm_ort_engine.cpp`** — Added `logits.size() == n_all_rows * K` check inside `ValidateDistributionOutput` immediately after the existing borders-size check (line 635 area). The guard uses `size_t` multiplication to avoid integer overflow and throws `InvalidInputException` naming the actual element count, declared shape, and required count, with the same error-message style as the surrounding checks (names the fixing SET). This mirrors the equivalent check in `ValidateTabFMOutput` (lines 592-599) and closes the OOB-read hole in `DecodeDistribution`'s `out.logits[t * K + k]` indexing (MGEN-03).

2. **`test/cpp/test_tabfm_distribution_decode.cpp`** — Extended Test 4 (`ValidateDistributionOutput rejects bad contracts`) with a new sub-case: shape `[N, 8]` is correct and borders length is correct (`K+1=9`), but `logits` is assigned `N * 8 - 1` elements (one short). Asserts `REQUIRE_THROWS_AS(..., InvalidInputException)`. Comment documents that this catches corrupt/non-conformant graphs that declare the right shape but return fewer elements.

3. **`test/cpp/test_tabfm_profile_registry.cpp`** (IN-01) — Fixed the unintentional task-enum mismatch on line 138: changed `PreprocessTask::REGRESSION` to `PreprocessTask::CLASSIFICATION` for the dispatch via `cls_manifest.preprocessing_profile`, and added a second `REQUIRE_NOTHROW` using `reg_manifest.preprocessing_profile` with `PreprocessTask::REGRESSION` to cover both tasks explicitly. No runtime behavior change (both built-in manifests share the same profile id), but removes the readability gap flagged by IN-01.

**Verification:** Verification ran in the **main checkout** (incremental `make debug` from the main repo; worktree held modified files via git worktree mechanism). `make test_debug` passed all 387 assertions in 15 test cases (`DATAZOO_DISABLE_TELEMETRY=1`).

Targeted Catch2 runs (main checkout, same binary):
- `[tabfm][distribution_decode]`: 15 assertions in 4 test cases — all passed (including new truncated-logits sub-case)
- `[tabfm][profile_registry]`: 10 assertions in 3 test cases — all passed

Results are reproducible from the main checkout at commit `effadf1`.

---

## Skipped Issues

None — the single in-scope critical finding was fixed.

---

_Fixed: 2026-09-22T19:10:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
