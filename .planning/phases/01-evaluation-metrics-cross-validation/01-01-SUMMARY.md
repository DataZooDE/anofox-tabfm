---
phase: 01-evaluation-metrics-cross-validation
plan: 01
subsystem: metrics-scaffold
tags: [metrics, classification, accuracy, scaffold, tdd, tracer]
status: complete

dependency_graph:
  requires: []
  provides:
    - RegisterClassificationMetrics (entry point for plan 01-02)
    - RegisterRegressionMetrics (entry point for plan 01-03)
    - RegisterCrossValidateMacros (entry point for plan 01-04)
    - tabfm_accuracy / anofox_tabfm_accuracy (CMET-01, fully wired)
    - RegisterAggregateFunctionSetWithAlias helper
    - Three test TU OBJECT library sources in CMakeLists
  affects:
    - CMakeLists.txt (EXTENSION_SOURCES + TABFM_CPP_TEST_SOURCES)
    - src/include/tabfm_registration.hpp
    - src/anofox_tabfm_extension.cpp (LoadInternal)
    - src/include/anofox_function_alias.hpp (new aggregate helper)

tech_stack:
  added:
    - RegisterAggregateFunctionSetWithAlias (anofox_function_alias.hpp inline helper)
    - AccuracyState aggregate triplet (DuckDB AggregateFunction API)
    - UnifiedVectorFormat NULL-skip pattern for metric aggregates
  patterns:
    - DuckDB aggregate triplet: StateSize/StateInit/Update/Combine/Finalize
    - UnifiedVectorFormat for NULL-safe per-row access in Update
    - FlatVector::SetNull for empty/all-NULL aggregate result
    - Telemetry-at-bind (once per execution, never per row)
    - RegisterAggregateFunctionSetWithAlias for primary+alias in one call

key_files:
  created:
    - src/include/tabfm_metrics_classification.hpp
    - src/include/tabfm_metrics_regression.hpp
    - src/include/tabfm_crossval.hpp
    - src/tabfm_metrics_classification.cpp
    - src/tabfm_metrics_regression.cpp
    - src/tabfm_crossval.cpp
    - test/cpp/test_tabfm_metrics.cpp
    - test/cpp/test_tabfm_metrics_regression.cpp
    - test/cpp/test_tabfm_crossval.cpp
    - test/sql/tabfm_metrics_classification.test
    - tools/golden/generate_metric_fixtures.py
  modified:
    - CMakeLists.txt (EXTENSION_SOURCES + TABFM_CPP_TEST_SOURCES)
    - src/include/tabfm_registration.hpp (three new declarations)
    - src/anofox_tabfm_extension.cpp (three new Register* calls in LoadInternal)
    - src/include/anofox_function_alias.hpp (new aggregate helper + includes)

decisions:
  - id: golden-hardcoded
    summary: "Golden values hard-coded in .test files; generate_metric_fixtures.py traces their sklearn origin but is not a build artifact"
    rationale: "CI reproducibility without sklearn runtime dependency; values are stable math, not ML predictions"
  - id: accuracy-implementation-in-task1
    summary: "Full accuracy implementation included in scaffold commit (Task 1) rather than stub, then refined in Task 2"
    rationale: "Plan instructions said 'create tabfm_metrics_classification.cpp with the real accuracy implementation' as part of Task 1; stub would compile but leave an unregistered aggregate"
  - id: function-description-required
    summary: "FunctionDescription added to accuracy registration (deviation Rule 2) after tabfm_function_docs.test caught the gap"
    rationale: "Every tabfm_*/anofox_tabfm_* function must carry a non-empty description per existing test contract"

metrics:
  duration_minutes: 10
  completed_date: "2026-09-20"
  tasks_completed: 2
  tasks_total: 2

actuals:
  tokens: 6767      # chars/4 over the 27066-char diff
  tasks: 2
  commits: 3        # MEASURED: git rev-list --count bacefd034567212a2717f2febe0b2869cb20f6b5..HEAD
  plan_head_before: bacefd034567212a2717f2febe0b2869cb20f6b5
---

# Phase 01 Plan 01: Scaffold Wiring + tabfm_accuracy Tracer Summary

**One-liner:** Metric/CV scaffold wired once for all three parallel plans; tabfm_accuracy aggregate (CMET-01) ships fully with UnifiedVectorFormat NULL-skip, telemetry-at-bind, and sklearn-golden sqllogictest.

## What Was Built

### Task 1: Coordinated Scaffold Wiring

All scaffold-owned shared-file edits for the entire phase done in one coordinated batch, so plans 01-02/03/04 never touch these files:

- `RegisterAggregateFunctionSetWithAlias` added to `anofox_function_alias.hpp` — mirrors `RegisterScalarFunctionSetWithAlias` exactly; copy-then-rename preserves all aggregate function metadata.
- Three module public headers created (`tabfm_metrics_classification.hpp`, `tabfm_metrics_regression.hpp`, `tabfm_crossval.hpp`), each declaring one `Register*` function.
- Three declarations added to `tabfm_registration.hpp`.
- Three `Register*` calls added to `LoadInternal()` in `anofox_tabfm_extension.cpp`.
- Three new `.cpp` added to `EXTENSION_SOURCES`; three test TUs added to `TABFM_CPP_TEST_SOURCES` in `CMakeLists.txt` (all Wave-2 plans pre-registered so they never touch CMakeLists again).
- Stub bodies for `RegisterRegressionMetrics` and `RegisterCrossValidateMacros` compile cleanly; `RegisterClassificationMetrics` carries the full accuracy implementation.

### Task 2: tabfm_accuracy End-to-End Tracer (CMET-01)

- `AccuracyState {int64_t correct; int64_t total}` aggregate triplet with correct DuckDB aggregate Update signature (5th parameter is row count, 3rd is input_count).
- UnifiedVectorFormat NULL-skip: rows where `actual` OR `predicted` is NULL excluded from both numerator and denominator.
- Divide-by-zero guard (T-01-01): `total == 0` → `FlatVector::SetNull` (returns NULL, never divides).
- Telemetry called once at bind via `AccuracyBind` callback (CLAUDE.md rule #3).
- `RegisterAggregateFunctionSetWithAlias` registers `anofox_tabfm_accuracy` (primary) and `tabfm_accuracy` (alias) with full metadata and `FunctionDescription`.
- Golden sqllogictest: 3-class fixture achieves `0.666667` (2/3, matching `sklearn.metrics.accuracy_score`).
- `tools/golden/generate_metric_fixtures.py`: traceable reference for committed golden values; not a build artifact.
- 4 Catch2 test cases added to `test_tabfm_metrics.cpp`: golden value, NULL-skip, empty-input NULL, alias parity.

**Tracer gate result:** Both `tabfm_accuracy` and `anofox_tabfm_accuracy` resolve and agree; all three scaffold Register* functions wired in `Load()` and the build links cleanly; aggregate-alias helper is in use; full test suite passes (240 assertions in 11 test cases).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed aggregate Update signature (input_count vs row_count)**
- **Found during:** Task 2 RED phase — test returned `0.5` instead of `0.666667`
- **Issue:** `AccuracyUpdate` had `(inputs[], data, idx_t count, Vector &states, idx_t)` — the 3rd param is `input_count` (=2 for 2 inputs), not row count. With `count=2`, only 2 rows of 3 were processed.
- **Fix:** Corrected signature to `(inputs[], data, idx_t, Vector &states, idx_t count)` per 01-PATTERNS.md
- **Files modified:** `src/tabfm_metrics_classification.cpp`
- **Commit:** `db352e1`

**2. [Rule 2 - Missing Critical Functionality] Added FunctionDescription to accuracy registration**
- **Found during:** Full test suite run — `tabfm_function_docs.test` returned count=2 (expected 0) for functions without descriptions
- **Issue:** `RegisterAggregateFunctionSetWithAlias` called without `FunctionDescription`; existing contract requires all registered functions to carry a non-empty description
- **Fix:** Added `FunctionDescription` with description text and two examples
- **Files modified:** `src/tabfm_metrics_classification.cpp`
- **Commit:** `1cf2fb6`

**3. [Informational - TDD ordering] Accuracy implementation in Task 1, not Task 2**
- **Situation:** Plan Task 1 instructed "create tabfm_metrics_classification.cpp with the real accuracy implementation"; full implementation was included in the scaffold commit rather than as a stub that Task 2 would fill.
- **Impact:** The RED phase was observed via the failing Update signature test (the test returned wrong values, not "function not found"), satisfying TDD intent — the test revealed a correctness bug before it was fixed.
- No code change — informational only.

## Verification Evidence

- `make debug`: succeeds, all three modules linked (3 commits verified above)
- `./build/debug/test/unittest test/sql/tabfm_metrics_classification.test`: All tests passed (11 assertions in 1 test case)
- `tabfm_accuracy` and `anofox_tabfm_accuracy`: return identical `0.666667` on golden fixture
- NULL rows skipped: valid (non-NULL) pairs only counted
- Empty / all-NULL input returns NULL (divide-by-zero guarded)
- Full suite: All tests passed (240 assertions in 11 test cases)

## Known Stubs

- `src/tabfm_metrics_regression.cpp`: `RegisterRegressionMetrics` body is a stub — registers nothing. Filled by plan 01-03.
- `src/tabfm_crossval.cpp`: `RegisterCrossValidateMacros` body is a stub — registers nothing. Filled by plan 01-04.
- `test/cpp/test_tabfm_metrics_regression.cpp`: placeholder Catch2 case only. Filled by plan 01-03.
- `test/cpp/test_tabfm_crossval.cpp`: placeholder Catch2 case only. Filled by plan 01-04.

## Threat Surface Scan

No new threat surface beyond what the plan's threat model covers. T-01-01 (divide-by-zero in Finalize) mitigated. T-01-02 (int64 overflow) accepted. T-01-03 (package installs) N/A — no installs.

## Self-Check: PASSED

- All 15 artifact files verified present on disk
- All 3 commits (3f5f93f, db352e1, 1cf2fb6) verified in git log
- Full test suite: 240 assertions passed, 0 failed
