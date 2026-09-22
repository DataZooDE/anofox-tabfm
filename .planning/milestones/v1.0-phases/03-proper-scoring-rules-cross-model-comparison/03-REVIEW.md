---
phase: 03-proper-scoring-rules-cross-model-comparison
reviewed: 2026-09-22T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/tabfm_scoring.cpp
  - src/tabfm_crossval.cpp
  - test/cpp/test_tabfm_scoring.cpp
  - test/sql/tabfm_scoring.test
  - test/sql/tabfm_crossval.test
findings:
  critical: 0
  warning: 1
  info: 0
  total: 1
status: issues_found
---

# Phase 03: Code Review Report (Re-Review after Fixes)

**Reviewed:** 2026-09-22
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Re-review of 6 fix commits (e593d3a..7151857) targeting 2 Critical + 3 Warning + 2 Info findings
from the prior review. All 6 fixes are confirmed correct and complete for their stated goals. No
new Critical issues were introduced. One pre-existing Warning surfaces as a result of the CR-02
fix: `tabfm_cross_validate` injects the `data` parameter without single-quote escaping in three
places, creating a visible inconsistency with the now-fixed `tabfm_compare_models`.

### Fix confirmation

**CR-01 — `std::isnan(coverage)` guard (`tabfm_scoring.cpp:703`).**
Correct. `std::isnan` is prepended to the disjunction before the range comparisons. `<cmath>` was
already included. The SQL regression test at `tabfm_scoring.test:292` uses `0.0/0.0` to produce
NaN: DuckDB's IEEE float mode returns `nan` for `0.0 / 0.0::DOUBLE` (verified in
`duckdb/test/sql/types/float/ieee_floating_points.test:22`), so `ExpressionExecutor::EvaluateScalar`
delivers a genuine NaN `double` to `IScoreBind`, and `std::isnan` intercepts it. Fix is complete
and the test is exercising the real path.

**CR-02 — `data` single-quote-escaped in `tabfm_compare_models` (`tabfm_crossval.cpp:322, 331`).**
Both occurrences are fixed. The escape pattern `replace(CAST(data AS VARCHAR), '''', '''''')` is
identical to the `target` escape in the same macro and in `tabfm_macros.cpp`. The description
string at line 341 is updated to document the fix. Correct and complete for `tabfm_compare_models`.

**WR-01 — metric whitelist in `tabfm_cross_validate` (`tabfm_crossval.cpp:181-191`).**
The `CASE` whitelist covers all seven expected metric names: `tabfm_rmse`, `tabfm_accuracy`,
`tabfm_crps`, `tabfm_log_score`, `tabfm_interval_score`, `tabfm_mae`, `tabfm_r2`. The `ELSE`
branch calls `error()` with the named remedy. The injection-attempt test in
`tabfm_crossval.test:323-327` confirms the error fires. Note that three whitelisted names
(`tabfm_crps`, `tabfm_log_score`, `tabfm_interval_score`) accept `(actual, yhat_dist STRUCT(...))`
as their second argument and will produce a PSR-04 bind-gate error when used with
`tabfm_cross_validate`'s `p.yhat` (a plain DOUBLE). This is a pre-existing design limitation, not
a regression introduced by WR-01 — the whitelist correctly prevents injection while the
distribution-metric incompatibility existed before.

**WR-02 — per-element `IsNull()` checks in all three Update loops.**
All three Update functions (CRPSUpdate, LogScoreUpdate, IScoreUpdate) have identical null-checking
structure: a loop over indices `[0, K-1]` checking both `logit_vals[k].IsNull()` and
`border_vals[k].IsNull()`, followed by a separate post-loop check for `border_vals[K].IsNull()`.
This covers the full K+1 border array. The fix is symmetric across all three aggregates. The SQL
tests in `tabfm_scoring.test:325-351` verify NULL-in-logits and NULL-in-borders for `tabfm_crps`
and `tabfm_log_score`; `tabfm_interval_score` is tested for NULL-in-logits only (missing
NULL-in-borders test for interval score, but the code path is identical to CRPS/log_score and
the gap is a test coverage omission, not a code defect).

**WR-03 — telemetry-at-load deviation documented.**
`RegisterCVMacroWithAlias` at `tabfm_crossval.cpp:403-411` now contains a detailed comment
explaining the known limitation and its root cause. The same pattern exists in
`tabfm_macros.cpp`; the comment brings parity. No code change was required.

**IN-01 — unreachable post-loop fallback marked.**
`tabfm_scoring.cpp:217` now has `D_ASSERT(false && "ComputeLogScore: unreachable post-loop fallback")`
followed by `return kMaxPenalty`. The comment explains the invariant. Correct.

**IN-02 — 3-arg `(DOUBLE, DOUBLE, DOUBLE)` bind-gate added.**
Overload 4 of `anofox_tabfm_interval_score` is registered at `tabfm_scoring.cpp:880-894`.
It throws `InvalidInputException` with the PSR-04 remedy text. The SQL test at
`tabfm_scoring.test:306-310` confirms the error fires and the message matches the expected prefix.
Correct and complete.

---

## Warnings

### WR-01: `tabfm_cross_validate` does not single-quote-escape `data` in generated SQL

**File:** `src/tabfm_crossval.cpp:210, 225, 237`
**Issue:** The `data` parameter is interpolated into three inner SQL string literals that become
arguments to `tabfm_classify`/`tabfm_regress` inside the `query()` call, without applying
`replace(..., '''', '''''')`. The three affected sites are:

```cpp
|| ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'   // line 210
|| ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'   // line 225
|| ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'   // line 237
```

If `data` resolves to a string containing a single quote (for example, a user passes
`data := 'my''table'` — an already-quoted table reference), the generated SQL string literal
becomes malformed, producing a parser error rather than a controlled DuckDB exception.
This is inconsistent with `tabfm_compare_models` (fixed by CR-02), which now correctly escapes
`data` at both its interpolation sites (lines 322 and 331).

The same gap does not affect `target` in `tabfm_cross_validate` — `target` is single-quote-escaped
at line 215 and double-quote-escaped at lines 221, 233. Only `data` is unescaped.

**Fix:**

```cpp
// Replace all three occurrences of:
|| ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'

// With:
|| ')::INTEGER AS _cv_fold_id FROM (FROM ' || replace(CAST(data AS VARCHAR), '''', '''''') || '))'
```

Apply to lines 210, 225, and 237. This brings `tabfm_cross_validate` to parity with the
`tabfm_compare_models` fix from CR-02.

---

_Reviewed: 2026-09-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
