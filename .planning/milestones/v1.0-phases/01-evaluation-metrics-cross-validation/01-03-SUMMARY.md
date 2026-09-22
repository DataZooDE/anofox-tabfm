---
phase: 01-evaluation-metrics-cross-validation
plan: 03
subsystem: metrics-regression
tags: [metrics, regression, rmse, mae, r2, mape, medae, tdd, rmet-01, rmet-02, rmet-03, rmet-04]
status: complete

dependency_graph:
  requires:
    - 01-01 (scaffold, RegisterAggregateFunctionSetWithAlias, empty RegisterRegressionMetrics stub)
  provides:
    - tabfm_rmse / anofox_tabfm_rmse (RMET-01, sklearn-golden)
    - tabfm_mae / anofox_tabfm_mae (RMET-02, sklearn-golden)
    - tabfm_r2 / anofox_tabfm_r2 (RMET-03, constant-target safe, sklearn-golden)
    - tabfm_mape / anofox_tabfm_mape (RMET-04, zero-actual skip, sklearn-golden)
    - tabfm_medae / anofox_tabfm_medae (RMET-04, heap-owning state, sklearn-golden)
  affects:
    - src/tabfm_metrics_regression.cpp (full implementation from stub)
    - test/sql/tabfm_metrics_regression.test (28 assertions, new file)
    - test/cpp/test_tabfm_metrics_regression.cpp (81 assertions, from placeholder)
    - tools/golden/generate_metric_fixtures.py (RMET-01..04 golden value documentation)

tech_stack:
  added:
    - RMSEState {double sum_sq; int64_t n} aggregate (sum_sq += (p-a)^2; finalize sqrt(sum_sq/n))
    - MAEState {double sum_abs; int64_t n} aggregate (finalize sum_abs/n)
    - R2State {sum_y, sum_y2, sum_res, n} online accumulator with SS_tot==0 constant-target guard
    - MAPEState {double sum_ape; int64_t n} with zero-actual skip (a==0 → skip row)
    - MedAEStateSlot (heap-owning vector<double> with StateDestroy for sorted-sample median)
  patterns:
    - UnifiedVectorFormat NULL-skip in all five Update callbacks (01-PATTERNS.md)
    - FlatVector::SetNull on n==0 for empty/all-NULL input in all Finalize callbacks
    - StateDestroy for heap-owning MedAEStateSlot (same pattern as F1StateSlot in 01-02)
    - RegisterAggregateFunctionSetWithAlias for primary+alias in one call (all five metrics)
    - Telemetry-at-bind for all five functions (once per execution, never per row)
    - FunctionDescription on all five registrations (enforced by tabfm_function_docs.test)

key_files:
  created:
    - test/sql/tabfm_metrics_regression.test
  modified:
    - src/tabfm_metrics_regression.cpp (stub → full implementation of all 5 metrics)
    - test/cpp/test_tabfm_metrics_regression.cpp (placeholder → 6 test cases, 81 assertions)
    - tools/golden/generate_metric_fixtures.py (RMET-01..04 golden documentation added)

decisions:
  - id: mape-dimensionless-ratio
    summary: "MAPE returned as dimensionless ratio (not %, not *100) matching sklearn.mean_absolute_percentage_error"
    rationale: "sklearn returns a fraction; the test comment documents this explicitly; users multiply by 100 for a percentage display"
  - id: r2-constant-target-tolerance
    summary: "R² constant-target guard uses |SS_tot| < 1e-12 as 'effectively zero' threshold"
    rationale: "Tight tolerance prevents false-positive triggering on genuine near-zero variance; same numeric scale as float64 rounding in SS_tot computation"
  - id: medae-even-n-mean-of-middles
    summary: "MedAE even-N returns (residuals[n/2-1] + residuals[n/2]) / 2.0, matching sklearn median_absolute_error"
    rationale: "sklearn uses numpy.median which returns the mean of two middle elements for even arrays; match ensures golden parity"
  - id: all-metrics-in-one-file
    summary: "All five regression metrics implemented in src/tabfm_metrics_regression.cpp in one TDD sweep (RED all tests, GREEN full implementation)"
    rationale: "Plan had tasks split RMSE/MAE, R², MAPE/MedAE but they share the same file; writing all tests upfront as a single RED pass avoids repeated build cycles while preserving TDD discipline"

metrics:
  duration_minutes: 9
  completed_date: "2026-09-20"
  tasks_completed: 3
  tasks_total: 3

actuals:
  tokens: 11481     # chars/4 over the 45924-char diff
  tasks: 3
  commits: 2        # MEASURED: git rev-list --count 40bda7bb44e3816dd40f71a7da69007ed0e66ebe..HEAD
  plan_head_before: 40bda7bb44e3816dd40f71a7da69007ed0e66ebe
---

# Phase 01 Plan 03: Regression Metrics RMET-01..04 Summary

**One-liner:** Five regression metrics — RMSE, MAE, R² (constant-target safe), MAPE (zero-actual skip), and MedAE (heap-owning sorted-sample) — shipped sklearn-golden with full NULL-safety and documented edge-case behavior.

## What Was Built

### Task 1: RMSE and MAE Aggregates (RMET-01, RMET-02)

- **RMSEState `{double sum_sq; int64_t n}`**: Update accumulates `(predicted-actual)^2`; Finalize returns `sqrt(sum_sq/n)`. SetNull when `n==0`.
- **MAEState `{double sum_abs; int64_t n}`**: Update accumulates `|predicted-actual|`; Finalize returns `sum_abs/n`. SetNull when `n==0`.
- Both use the UnifiedVectorFormat NULL-skip pattern — rows where actual OR predicted is NULL are excluded from both numerator and denominator.
- `anofox_tabfm_rmse` (primary) + `tabfm_rmse` (alias) and `anofox_tabfm_mae` + `tabfm_mae` via `RegisterAggregateFunctionSetWithAlias`.
- Golden values match sklearn: RMSE=0.264575, MAE=0.22 on the 5-row fixture.

### Task 2: R² with Constant-Target Edge Case (RMET-03)

- **R2State `{sum_y, sum_y2, sum_res, n}`**: Online accumulator avoids a separate pass for mean computation. Finalize computes `SS_tot = sum_y2 - n*y_bar^2`.
- **Constant-target guard (`|SS_tot| < 1e-12`)**: returns 1.0 if `SS_res < 1e-12` (perfect prediction), 0.0 otherwise — never NaN or Inf. Documents the sklearn convention in the FunctionDescription.
- Golden value: R²=0.965 on the 5-row fixture. Exact 1.0/0.0 on constant-target fixtures.

### Task 3: MAPE (Zero-Actual Safe) and Median Absolute Error (RMET-04)

- **MAPEState `{double sum_ape; int64_t n}`**: Skips rows where `actual == 0` (documented, avoids division by zero). Returns a dimensionless ratio matching sklearn. All-zero-actual input → NULL.
- **MedAEStateSlot (heap-owning `vector<double>*`)**: StateInit sets pointer to nullptr; StateDestroy frees the heap-allocated vector. Update appends `|actual-predicted|`; Combine merges vectors; Finalize sorts and returns the median (odd N: middle element; even N: mean of two middle elements).
- MAPE golden: 0.076667 on the base fixture; 0.375 on the zero-skip fixture. MedAE golden: 0.2 (N=5), 0.15 (N=4, even).

## Verification Evidence

- `make debug`: succeeded, 4 files changed (0 errors)
- `./build/debug/test/unittest test/sql/tabfm_metrics_regression.test`: **All tests passed (28 assertions in 1 test case)**
- `./build/debug/test/unittest "[tabfm][metrics_regression]"`: **All tests passed (81 assertions in 6 test cases)**
- Full suite: **All tests passed (2101 assertions in 92 test cases)**

### Golden Value Verification

| Metric | Golden | Source |
|--------|--------|--------|
| tabfm_rmse | 0.264575 | sklearn root_mean_squared_error |
| tabfm_mae | 0.220000 | sklearn mean_absolute_error |
| tabfm_r2 | 0.965000 | sklearn r2_score |
| tabfm_r2 (constant perfect) | 1.0 | sklearn r2_score convention |
| tabfm_r2 (constant imperfect) | 0.0 | sklearn r2_score convention |
| tabfm_mape | 0.076667 | sklearn mean_absolute_percentage_error (ratio) |
| tabfm_mape (zero-skip) | 0.375 | non-zero rows only |
| tabfm_mape (all-zero) | NULL | no contributing rows |
| tabfm_medae (N=5) | 0.200000 | sklearn median_absolute_error |
| tabfm_medae (N=4) | 0.150000 | sklearn median_absolute_error (even-N mean) |

### Edge Cases Verified

| Function | Input | Result |
|----------|-------|--------|
| tabfm_rmse | empty | NULL |
| tabfm_rmse | all-NULL | NULL |
| tabfm_mae | empty | NULL |
| tabfm_r2 | constant target, perfect | 1.0 (finite) |
| tabfm_r2 | constant target, imperfect | 0.0 (finite) |
| tabfm_r2 | empty | NULL |
| tabfm_mape | actual==0 rows | skipped |
| tabfm_mape | all-zero actuals | NULL |
| tabfm_mape | empty | NULL |
| tabfm_medae | empty | NULL |

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Informational Notes

**1. [Informational - TDD ordering] All 5 metrics written as one TDD sweep**
- The three plan tasks share a single source file and a single test file. Writing the failing tests in one RED pass (all 28 assertions for all 5 metrics) before implementing any of them is the natural TDD interpretation for a single-file module.
- RED was observed: the first build attempt against the stub produced "Catalog Error: Scalar Function with name tabfm_rmse does not exist!" confirming the failing-test gate.
- No behavior change from plan intent — all metrics are correctly decomposed into the three tasks' done criteria.

## Known Stubs

None — all RMET-01..04 metrics are fully implemented with golden values.

## Threat Surface Scan

All mitigations from the plan's threat register are implemented:

| Threat ID | Status | Evidence |
|-----------|--------|----------|
| T-01-03-01 | Mitigated | R² guards `\|SS_tot\|<1e-12`; MAPE skips `actual==0`; all Finalizers SetNull when `n==0` |
| T-01-03-02 | Accepted | 64-bit IEEE 754 sums; documented in plan |
| T-01-03-03 | Accepted | MedAE O(N) heap documented in FunctionDescription; StateDestroy frees vector |
| T-01-03-04 | N/A | No package installs |

No new threat surface beyond the plan's threat model (no new endpoints, no auth paths, no schema changes).

## Self-Check: PASSED

- `test/sql/tabfm_metrics_regression.test`: FOUND
- `src/tabfm_metrics_regression.cpp`: FOUND (full implementation)
- `test/cpp/test_tabfm_metrics_regression.cpp`: FOUND (6 test cases)
- `tools/golden/generate_metric_fixtures.py`: FOUND (RMET docs added)
- Commit 5d34185 (RED) and b7b68e3 (GREEN) both in git log
- SQL test: 28 assertions passed
- Catch2 [tabfm][metrics_regression]: 81 assertions in 6 cases passed
- Full suite: 2101 assertions, 92 test cases, 0 failed
