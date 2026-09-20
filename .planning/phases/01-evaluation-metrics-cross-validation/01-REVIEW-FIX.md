---
phase: 01-evaluation-metrics-cross-validation
fixed_at: 2026-09-21T00:30:00Z
review_path: .planning/phases/01-evaluation-metrics-cross-validation/01-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 5
skipped: 1
status: partial
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-09-21
**Source review:** `.planning/phases/01-evaluation-metrics-cross-validation/01-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 6 (2 critical, 4 warnings; info items excluded per fix_scope=critical_warning)
- Fixed: 5 (CR-01, CR-02, WR-02, WR-03, WR-04)
- Skipped: 1 (WR-01 — accepted as intentional deviation)

**Verification ran in:** isolated git worktree, built against the main checkout (cmake build system
resolves sources from the main repo path). All three target SQL test files passed:
`tabfm_metrics_classification.test` (54 assertions), `tabfm_metrics_regression.test` (28 assertions),
`tabfm_crossval.test` (33 assertions). The pre-existing heap-use-after-free in `test_tabfm_crossval.cpp:251`
(IN-03 related static DuckDB shared instance) was confirmed pre-existing and is NOT caused by these fixes.

## Fixed Issues

### CR-01: F1/Precision/Recall/ROC-AUC silently return NULL for non-constant `avg` expressions

**Files modified:** `src/tabfm_metrics_classification.cpp`
**Commit:** 96ae060
**Applied fix:** Replaced `FlatVector::SetNull` + `continue` in both `F1Finalize` and `AUCFinalize`
when `avg_mode.empty()` with `throw InvalidInputException(...)` carrying an actionable message naming
the fixing parameter and valid values. The `catch (...)` in `BindF1Metric` / `AUCBind` is retained
(non-constant expressions cannot be evaluated at bind time). Tests that expect the exact error
substring (`tabfm_f1: 'avg' is required`, `tabfm_precision: 'avg' is required`,
`tabfm_recall: 'avg' is required`, `tabfm_roc_auc: 'avg' is required`) now pass.

Note: WR-03 changes were included in this commit (same file).

### CR-02: R² catastrophic cancellation for large-magnitude targets

**Files modified:** `src/tabfm_metrics_regression.cpp`
**Commit:** 2cf53c9
**Applied fix:** Replaced the naive `sum_y2 - n*ybar^2` SS_tot formula with Welford's online
variance algorithm (Knuth Vol. 2 §4.2.2). `R2State` struct changed from `{sum_y, sum_y2, sum_res, n}`
to `{mean, M2, sum_res, n}`. `R2Update` uses the two-delta Welford update (`delta = a - mean_before`,
`mean += delta/n`, `M2 += delta * delta_after`). `R2Combine` uses the Chan et al. parallel Welford
formula. `R2Finalize` reads `ss_tot = state.M2` (exactly 0.0 for constant targets regardless of
magnitude). This resolves both CR-02 (catastrophic cancellation) and WR-04 (threshold asymmetry)
together.

### WR-02: Hash seed type inconsistency in generated `tabfm_cross_validate` SQL

**Files modified:** `src/tabfm_crossval.cpp`
**Commit:** 9627231
**Applied fix:** All three occurrences of the hash expression in the generated SQL now emit
`CAST(seed AS BIGINT)` and `CAST(k AS UBIGINT)` explicitly (e.g. `hash(id, CAST(42 AS BIGINT)) %
CAST(5 AS UBIGINT)`), matching the type casts in `tabfm_fold_assign`'s macro body. Previously the
bare integer literal (e.g. `42`) was an `INTEGER` in the generated SQL while `fold_assign` used
`BIGINT`, creating a silent fold-assignment mismatch when users compared the two outputs.

### WR-03: `GetValue(i)` per-row heap allocation in hot comparison loops

**Files modified:** `src/tabfm_metrics_classification.cpp`
**Commit:** 96ae060 (same commit as CR-01)
**Applied fix:** Replaced `inputs[x].GetValue(i).ToString()` in `AccuracyUpdate` and `F1Update`
with direct `UnifiedVectorFormat::GetData<string_t>(actual_data)[aidx]` raw pointer access.
`AccuracyUpdate` now uses `string_t::operator==` directly (no heap alloc, zero copy).
`F1Update` uses `.GetString()` on the raw `string_t` to produce `std::string` for map lookups
(one allocation per row instead of two). `AUCUpdate` and `ECEUpdate` iterate over MAP values
using `MapValue::GetChildren` which requires a `Value` object; those were left unchanged as the
API does not offer a zero-allocation MAP iterator.

### WR-04: `ss_res < 1e-12` threshold misclassifies near-perfect predictions as perfect

**Files modified:** `src/tabfm_metrics_regression.cpp`
**Commit:** 2cf53c9 (same commit as CR-02)
**Applied fix:** Replaced `(ss_res < 1e-12) ? 1.0 : 0.0` with `(ss_res == 0.0) ? 1.0 : 0.0`.
`ss_res` is a sum of squared residuals; it is exactly 0.0 at the floating-point level only when
all residuals are identically zero, making the exact equality check safe and correct. The Welford
fix for CR-02 and this fix were applied in the same refactor.

## Skipped Issues

### WR-01: Telemetry fires at extension-load time for CV macros, not at per-use time

**File:** `src/tabfm_crossval.cpp:302-313`
**Reason:** Accepted as intentional documented deviation. The code comment at line 304 already
acknowledges this limitation: "For macros, telemetry fires at load time via registration rather than
per-bind (macros don't have a separate bind callback; see tabfm_macros.cpp pattern)." The
`tabfm_macros.cpp` file uses the identical pattern for `tabfm_classify`/`tabfm_regress` macros.
Changing this would require either (a) injecting a sentinel scalar function into the macro body
(fragile and poorly tested), or (b) waiting for DuckDB to expose a macro bind callback. Neither
approach is warranted for a warning-level finding.

---

_Fixed: 2026-09-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
