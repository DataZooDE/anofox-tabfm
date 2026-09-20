---
phase: 01-evaluation-metrics-cross-validation
verified: 2026-09-21T01:20:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-01-PLAN.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-01-SUMMARY.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-02-PLAN.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-02-SUMMARY.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-03-PLAN.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-03-SUMMARY.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-04-PLAN.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-04-SUMMARY.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-REVIEW-FIX.md
  - .planning/phases/01-evaluation-metrics-cross-validation/01-REVIEW.md
  - src/include/tabfm_crossval.hpp
  - src/include/tabfm_metrics_classification.hpp
  - src/include/tabfm_metrics_regression.hpp
  - src/tabfm_crossval.cpp
  - src/tabfm_metrics_classification.cpp
  - src/tabfm_metrics_regression.cpp
  - test/cpp/test_tabfm_crossval.cpp
  - test/cpp/test_tabfm_metrics.cpp
  - test/cpp/test_tabfm_metrics_regression.cpp
  - test/sql/tabfm_crossval.test
  - test/sql/tabfm_metrics_classification.test
  - test/sql/tabfm_metrics_regression.test
  - tools/golden/generate_metric_fixtures.py
covered_digest: "v1:sha256:bd89c959fd3f5abf604a5ef53a46988aa0c19c91468fc9f48ab0cddb3d2fbc19"
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "tabfm_accuracy (and tabfm_f1/precision/recall) now bind and compute correctly on INTEGER (non-VARCHAR) columns — registration restored to {ANY, ANY} and label comparison replaced with Value-based path (commit fda45f8)"
  gaps_remaining: []
  regressions: []
advisory:
  - finding: "IN-02 stale comment: R2 registration comment at src/tabfm_metrics_regression.cpp:630 reads '|SS_tot| < 1e-12 → 1.0' but the post-review code uses exact 'ss_tot == 0.0'. A future reader may be confused about whether a tolerance is intended."
    category: other
    reason: "Info severity per 01-REVIEW.md; code is correct; only the comment is stale. Not in scope of fda45f8 gap-closure commit."
    evidence_status: "comment at line 630 confirmed stale — code at line 336 uses exact equality; pre-dates this gap-closure round unflagged"
  - finding: "Post-shutdown heap-use-after-free (ASan) in DuckDB's own BlockAllocator thread-local cleanup (duckdb/src/storage/block_allocator.cpp:156, duckdb/src/main/database.cpp:73). All 1520 Catch2 assertions pass before the abort."
    category: other
    reason: "Present on commit 650b3a7 (prior to fda45f8) — confirmed pre-existing by re-running '[tabfm]' suite on that commit. Stack trace contains zero phase-modified files. Likely a DuckDB v1.5.4 + GCC 16 shared_ptr destruction ordering issue. Not introduced by this phase."
    evidence_status: "identical ASan trace reproduced on prior commit 650b3a7; new-scope, no deterministic evidence attributing it to phase changes"
---

# Phase 1: Evaluation Metrics + Cross-Validation Verification Report

**Phase Goal:** Users can measure classification and regression prediction quality — and run leakage-safe k-fold cross-validation — directly in SQL over plain `(actual, predicted[, proba])` columns, independent of any model.
**Verified:** 2026-09-21T01:20:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (commit fda45f8)

---

## Re-verification Summary

The single gap from the initial verification (CMET-01 blocked by `{VARCHAR, VARCHAR}` registration breaking INTEGER inputs under DuckDB v1.5.4) is **closed**. Commit `fda45f8` restored `{ANY, ANY}` registration and replaced raw `GetData<string_t>()` with type-safe `Value::GetValue(i)` comparison. All four test suites now pass.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `tabfm_accuracy(actual, predicted)` returns correct/total as DOUBLE matching sklearn golden (CMET-01) | VERIFIED | `tabfm_metrics_classification.test` — **56 assertions pass** (was: FAIL at line 74). `{ANY, ANY}` registration at line 1050; `inputs[0].GetValue(i) == inputs[1].GetValue(i)` at line 114; `preds_empty` fixtures use `NULL::VARCHAR`; `preds_int` INTEGER test binds and returns golden 0.666667. |
| 2 | `tabfm_f1`/`tabfm_roc_auc` with required avg match sklearn, no silent default (CMET-02, CMET-04) | VERIFIED | SQL test passes; Catch2 `[tabfm][metrics]` 53 assertions pass; `{ANY, ANY}` registration confirmed at lines 1078, 1108, 1138; `Value::ToString()` used for F1 map keys at lines 294-295; golden values confirmed. |
| 3 | `tabfm_log_loss` clips probabilities (no infinities), `tabfm_ece` computes ECE from MAP, `tabfm_confusion_matrix` returns table-valued result (CMET-03, CMET-05, CMET-06) | VERIFIED | Catch2 `[tabfm][metrics]` passes; MAP access, clipping, and macro quoting present in source; log_loss golden 0.424322, ECE golden 0.333333. |
| 4 | Regression edge cases: `tabfm_r2` on constant target returns 1.0/0.0 (never NaN/Inf), `tabfm_mape`/`tabfm_medae` on zero actuals return documented results (RMET-01..04) | VERIFIED | `tabfm_metrics_regression.test` passes (28 assertions), Catch2 `[tabfm][metrics_regression]` passes (81 assertions); Welford R² with exact `ss_tot == 0.0` guard at lines 283–346. |
| 5 | `tabfm_cross_validate` uses leakage-safe two-table form, returns k per-fold + aggregate mean±std rows, safely quotes target identifier (CV-01..04) | VERIFIED | `tabfm_crossval.test` passes (33 assertions), Catch2 `[tabfm][crossval]` passes (42 assertions); NON-NEGOTIABLE leakage test passes (aggregate accuracy < 0.8); `replace(target,'"','""')` at all interpolation sites confirmed. |

**Score:** 5/5 truths verified

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Classification SQL test | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest test/sql/tabfm_metrics_classification.test` | All tests passed (56 assertions in 1 test case) | PASS |
| Regression SQL test | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest test/sql/tabfm_metrics_regression.test` | All tests passed (28 assertions in 1 test case) | PASS |
| Cross-validation SQL test | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest test/sql/tabfm_crossval.test` | All tests passed (33 assertions in 1 test case) | PASS |
| Catch2 classification metrics | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest "[tabfm][metrics]"` | All tests passed (53 assertions in 11 test cases) | PASS |
| Catch2 regression metrics | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest "[tabfm][metrics_regression]"` | All tests passed (81 assertions in 6 test cases) | PASS |
| Catch2 cross-validation | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest "[tabfm][crossval]"` | All tests passed (42 assertions in 9 test cases) | PASS |
| Leakage detection (NON-NEGOTIABLE) | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest "tabfm_cross_validate: NON-NEGOTIABLE leakage-detecting golden test"` | All tests passed (11 assertions) — aggregate accuracy < 0.8 | PASS |
| Full tabfm suite | `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest "[tabfm]"` | All tests passed (1520 assertions in 77 test cases) | PASS |
| INTEGER column binding (gap gate) | Confirmed via `preds_int` table in classification SQL test (line 101–108) — explicit INTEGER columns, `tabfm_accuracy(actual, predicted)` binds and returns 0.666667 | Pass — line reached and asserted | PASS |

> Note: After test completion, an ASan heap-use-after-free fires in DuckDB's own `BlockAllocator` thread-local shutdown (`duckdb/src/storage/block_allocator.cpp:156`). Confirmed pre-existing on commit 650b3a7 (before fda45f8). Stack trace contains zero phase-modified files. Recorded as advisory.

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/tabfm_metrics_classification.cpp` | Full classification metric aggregates (CMET-01..06) | VERIFIED | 1239 lines, substantive; `{ANY, ANY}` registration at line 1050 (accuracy), 1078 (precision), 1108 (recall), 1138 (f1); `Value::GetValue(i)` comparison at line 114; no `GetData<string_t>()` in label comparison paths |
| `src/tabfm_metrics_regression.cpp` | Full regression metric aggregates (RMET-01..04) | VERIFIED | 705 lines; RMSEState, MAEState, R2State (Welford), MAPEState, MedAEStateSlot all present |
| `src/tabfm_crossval.cpp` | fold_assign + cross_validate macros (CV-01..04) | VERIFIED | 331 lines; both macros with leakage-safe two-table form, safe quoting, correct BIGINT casts |
| `src/include/tabfm_metrics_classification.hpp` | Header declaring RegisterClassificationMetrics | VERIFIED | Present |
| `src/include/tabfm_metrics_regression.hpp` | Header declaring RegisterRegressionMetrics | VERIFIED | Present |
| `src/include/tabfm_crossval.hpp` | Header declaring RegisterCrossValidateMacros | VERIFIED | Present |
| `test/sql/tabfm_metrics_classification.test` | SQL sqllogictest for CMET-01..06 | VERIFIED | 56 assertions all pass; `preds_empty` uses `NULL::VARCHAR`; `preds_int` INTEGER test binds and asserts |
| `test/sql/tabfm_metrics_regression.test` | SQL sqllogictest for RMET-01..04 | VERIFIED | 28 assertions all pass |
| `test/sql/tabfm_crossval.test` | SQL sqllogictest for CV-01..04 with leakage test | VERIFIED | 33 assertions all pass |
| `test/cpp/test_tabfm_metrics.cpp` | Catch2 tests for classification metrics | VERIFIED | 53 assertions pass |
| `test/cpp/test_tabfm_metrics_regression.cpp` | Catch2 tests for regression metrics | VERIFIED | 81 assertions pass |
| `test/cpp/test_tabfm_crossval.cpp` | Catch2 tests for CV with leakage test | VERIFIED | 42 assertions pass |
| `tools/golden/generate_metric_fixtures.py` | sklearn golden value reference | VERIFIED | All CMET/RMET sections present |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `anofox_tabfm_extension.cpp` Load() | `RegisterClassificationMetrics` | Direct call at line 119 | WIRED | Confirmed |
| `anofox_tabfm_extension.cpp` Load() | `RegisterRegressionMetrics` | Direct call at line 120 | WIRED | Confirmed |
| `anofox_tabfm_extension.cpp` Load() | `RegisterCrossValidateMacros` | Direct call at line 121 | WIRED | Confirmed |
| `RegisterAggregateFunctionSetWithAlias` | `anofox_tabfm_* + tabfm_*` aliases | `anofox_function_alias.hpp:115` | WIRED | All metrics register primary + alias |
| `tabfm_accuracy` | ANY registration | `src/tabfm_metrics_classification.cpp:1050` | WIRED | `{LogicalType::ANY, LogicalType::ANY}` — binds INTEGER, VARCHAR, all DuckDB types |
| CV macro body | `replace(target, '"', '""')` quoting | `src/tabfm_crossval.cpp:209,221,228` | WIRED | Three distinct interpolation sites all guarded |
| F1/ROC-AUC 2-arg overload | Named exception on missing avg | bind lambda at registration | WIRED | Both `tabfm_f1` and `tabfm_roc_auc` throw named exception |
| `tabfm_cross_validate` | two-table predict form | `test := '(...WHERE _cv_fold_id = f)'` in macro body | WIRED | Each fold trains on other folds, predicts held-out via two-table form |

---

## Requirements Coverage

| Requirement | Phase | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| CMET-01 | Plan 01-01 | Classification accuracy SQL aggregate | SATISFIED | SQL test passes (56 assertions); `{ANY, ANY}` registration; integer and NULL inputs both handled; golden 0.666667 asserted |
| CMET-02 | Plan 01-02 | Precision/Recall/F1 with required averaging mode | SATISFIED | Golden values match sklearn, required avg enforced, Catch2 passes |
| CMET-03 | Plan 01-02 | Log-loss with probability clipping | SATISFIED | eps clipping confirmed, no infinities, golden 0.424322 |
| CMET-04 | Plan 01-02 | ROC-AUC with rank-sum ties, required avg | SATISFIED | Mann-Whitney U, OvR/OvO, golden 0.958333 |
| CMET-05 | Plan 01-02 | Confusion matrix table-valued result | SATISFIED | TABLE MACRO present, double-quote escaping confirmed |
| CMET-06 | Plan 01-02 | Expected Calibration Error from MAP | SATISFIED | 10-bin ECE, golden 0.333333 |
| RMET-01 | Plan 01-03 | RMSE SQL aggregate | SATISFIED | Golden 0.264575, 28 SQL assertions pass |
| RMET-02 | Plan 01-03 | MAE SQL aggregate | SATISFIED | Golden 0.220000 |
| RMET-03 | Plan 01-03 | R² with constant-target edge case | SATISFIED | Welford implementation, exact 1.0/0.0 guard, never NaN/Inf |
| RMET-04 | Plan 01-03 | MAPE (zero-actual safe) and median absolute error | SATISFIED | MAPE skips actual==0, MedAE even-N mean-of-middles, heap-owning StateDestroy |
| CV-01 | Plan 01-04 | k deterministic seedable folds | SATISFIED | hash(row_key, CAST(seed AS BIGINT)) % k, order-independent |
| CV-02 | Plan 01-04 | Leakage-safe k-fold CV via two-table predict | SATISFIED | Two-table form confirmed in source, leakage test passes (< 0.8) |
| CV-03 | Plan 01-04 | Per-fold + aggregate mean±std output | SATISFIED | k rows + fold_id=-1 aggregate row with stddev_pop |
| CV-04 | Plan 01-04 | Safe target identifier quoting | SATISFIED | `replace(target,'"','""')` at all interpolation sites |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/tabfm_metrics_regression.cpp` | 630 | Stale comment: `\|SS_tot\| < 1e-12` when code uses `ss_tot == 0.0` (IN-02 from 01-REVIEW.md, not fixed) | Advisory | No runtime impact; misleads future readers about whether a tolerance is intended |

No TBD/FIXME/XXX markers found in any phase-modified files. No placeholder or stub implementations found.

---

### Advisory: Pre-Existing Post-Shutdown ASan Error

After all 1520 Catch2 assertions complete, an AddressSanitizer abort fires in DuckDB's own storage layer:

```
heap-use-after-free in duckdb::shared_ptr<std::atomic<bool>, true>::~shared_ptr()
  at duckdb/src/storage/block_allocator.cpp:156 (BlockAllocatorThreadLocalState::Initialize)
  via duckdb/src/main/database.cpp:73 (DBConfig::~DBConfig)
```

This is a thread-local destruction ordering issue between DuckDB's `BlockAllocator` and the atomic flag in its `shared_ptr`, triggered at process exit. The stack trace contains zero phase-modified files. The error was confirmed present on commit `650b3a7` (before the gap closure commit `fda45f8`) and is therefore definitively pre-existing. It does not affect correctness of any test assertions.

---

## Probe Execution

Step 7c: No probe scripts declared in PLAN/SUMMARY files. No `scripts/*/tests/probe-*.sh` found. Skipped.

---

## Human Verification Required

None. All must-haves verified programmatically.

---

_Verified: 2026-09-21T01:20:00Z_
_Verifier: Claude (gsd-verifier)_
