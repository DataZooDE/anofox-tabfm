---
phase: 03-proper-scoring-rules-cross-model-comparison
verified: 2026-09-22T00:00:00Z
status: passed
score: 12/12 must-haves verified
covered_files:
  - .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-01-PLAN.md
  - .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-01-SUMMARY.md
  - .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-02-PLAN.md
  - .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-02-SUMMARY.md
  - .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-REVIEW-FIX.md
  - .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-REVIEW.md
  - CMakeLists.txt
  - src/anofox_tabfm_extension.cpp
  - src/include/tabfm_registration.hpp
  - src/include/tabfm_scoring.hpp
  - src/tabfm_crossval.cpp
  - src/tabfm_scoring.cpp
  - test/cpp/test_tabfm_scoring.cpp
  - test/sql/tabfm_scoring.test
  - tools/parity/src/parity/crps_reference.py
covered_digest: "v1:sha256:b2c94bfdc4cd9559610047832e147303dad408d9f24b21c80260437f0430d4bd"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 03: Proper Scoring Rules + Cross-Model Comparison Verification Report

**Phase Goal:** Users can score probabilistic regression predictions with proper scoring rules (CRPS, log-score, interval score) over the Phase 2 predictive-distribution output, and compare model families head-to-head on their own tables using the Phase 1 evaluation primitives.
**Verified:** 2026-09-22T00:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | tabfm_crps(actual, yhat_dist) over a K=16 non-uniform distribution matches Python reference (0.530862) within 1e-6 (PSR-01) | VERIFIED | `./build/debug/test/unittest test/sql/tabfm_scoring.test` — 45/45 assertions passed; test block asserts `round(tabfm_crps(y_val, {...}), 6) == 0.530862`; Python reference `uv run python -m parity.crps_reference` prints identical value |
| 2  | tabfm_crps(actual, plain_double) fails at bind with error containing "output_mode" (PSR-04) | VERIFIED | SQL test `statement error` block at `tabfm_scoring.test:70-74` asserts error text `tabfm_crps: second argument must be a distribution`; C++ fn_point lambda throws `InvalidInputException` with "Use output_mode := 'distribution' in tabfm_regress" |
| 3  | tabfm_crps and anofox_tabfm_crps are both registered and callable (alias parity) | VERIFIED | `RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_crps", ...)` at `tabfm_scoring.cpp:773`; SQL test at line 59-63 asserts alias returns same golden value |
| 4  | CRPS uses model-provided non-uniform bin widths (wi = borders[i+1]-borders[i]), never a uniform 1/K assumption | VERIFIED | `tabfm_scoring.cpp:122` `const double wi = b_hi - b_lo;` — width computed from borders; no 1/K constant anywhere in file; docstring at line 768 explicitly states "Non-uniform bin widths from borders[i+1]-borders[i]" |
| 5  | NULL actual or NULL yhat_dist rows are skipped; all-empty group returns NULL | VERIFIED | SQL test blocks at lines 79-85 (empty group), 330-351 (NULL logit/border elements via WR-02 fix); Catch2 case "tabfm_crps NULL rows are skipped" in 21 test cases all pass |
| 6  | tabfm_log_score(actual, yhat_dist) returns mean NLL matching golden reference 1.581928 within 1e-6; out-of-support y returns -log(1e-10) with no log(0) (PSR-02) | VERIFIED | SQL test at `tabfm_scoring.test:105-109` asserts `round(tabfm_log_score(...), 6) == 1.581928`; Catch2 cases `ComputeLogScore out-of-support y -> max penalty` and `ComputeLogScore zero-width bin -> max penalty` pass; Python reference prints identical value |
| 7  | tabfm_log_score and tabfm_interval_score are bind-gated on distribution STRUCT; plain DOUBLE second argument throws with "output_mode := 'distribution'" remedy (PSR-04 extended) | VERIFIED | `tabfm_scoring.cpp:800-808` (log_score fn_point), `tabfm_scoring.cpp:860-875` (interval_score fn_point), `tabfm_scoring.cpp:877-894` (IN-02: 3-arg gate); SQL tests at lines 119-124, 197-201, 306-310 all confirmed passing |
| 8  | tabfm_interval_score(actual, yhat_dist, coverage := 0.9) computes IS formula matching golden reference 5.411898; defaults coverage=0.9, rejects coverage<=0 or >=1 at bind; NaN coverage also rejected (PSR-03, CR-01) | VERIFIED | SQL tests at lines 154-158 (default 0.9), 162-166 (explicit 0.9), 168-173 (coverage 0.5 = 2.34192), 182-193 (coverage=0.0 and =1.0 rejected), 291-295 (NaN via 0.0/0.0 rejected); `IScoreBind` at `tabfm_scoring.cpp:703` has `std::isnan(coverage)` guard |
| 9  | tabfm_compare_models runs Phase-1 metrics + Phase-3 scores across tabpfn_v2 (distribution) vs tabfm-v1 (point) on user's own table (CMP-01) | VERIFIED | SQL test at `tabfm_scoring.test:231-237` asserts tabpfn_v2 row returns `rmse=0.756226, crps=0.530862, log_score=1.581928, interval_score=5.411898`; tabfm-v1 row emits NULL for PSR scores (documented: no distribution output); UNION ALL at `tabfm_crossval.cpp:317-338` |
| 10 | tabfm_compare_models safely quotes target and data identifiers preventing SQL injection (T-03-06, CMP-01, WR-01) | VERIFIED | `replace(CAST(target AS VARCHAR), '"', '""')` at all 5 target interpolation sites (lines 318, 319, 321, 323, 331); `replace(CAST(data AS VARCHAR), '''', '''''')` at both data sites (lines 326, 335); SQL identifier-safety test at lines 267-279 passes (double-quote in column name) |
| 11 | tabfm_cross_validate single-quote-escapes data parameter at all three inner SQL generation sites (WR-01) | VERIFIED | `tabfm_crossval.cpp:212, 228, 241` all use `replace(CAST(data AS VARCHAR), '''', '''''')` — confirmed by code review fix commit dd0c7ba; `tabfm_crossval.test` injection-attempt test confirmed passing (35/35 assertions) |
| 12 | Full existing test suite (all Phases 1-3) passes without regressions (make test_debug) | VERIFIED | `make test_debug` output: "All tests passed (434 assertions in 16 test cases)" |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/tabfm_scoring.cpp` | CRPS + log-score + interval-score aggregate functions | VERIFIED | 913 lines; three aggregates with Init/Update/Combine/Finalize, PSR-04 bind-gates, IScoreBindData, full registration |
| `src/include/tabfm_scoring.hpp` | RegisterScoringFunctions + ComputeCRPS/LogScore/IntervalScore declarations | VERIFIED | 95 lines; all three Compute functions declared with external linkage; RegisterScoringFunctions declared |
| `test/cpp/test_tabfm_scoring.cpp` | Catch2 unit tests for ComputeCRPS/LogScore/IntervalScore (synthetic K=4) | VERIFIED | 21 test cases, 92 assertions, all passing |
| `test/sql/tabfm_scoring.test` | End-to-end SQL golden tests for PSR-01/02/03/04, CMP-01, CR-01, IN-02, WR-02 | VERIFIED | 45 assertions in 1 test case, all passing |
| `tools/parity/src/parity/crps_reference.py` | Python reference functions crps_bar + log_score_bar + interval_score_bar | VERIFIED | Documented functions; `uv run python -m parity.crps_reference` runs and prints matching golden values |
| `src/tabfm_crossval.cpp` (compare_models) | tabfm_compare_models macro added to RegisterCrossValidateMacros | VERIFIED | COMPARE_MODELS_MACRO at lines 292-348; registered at line 432 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/anofox_tabfm_extension.cpp` | `RegisterScoringFunctions` | `anofox::RegisterScoringFunctions(loader)` at line 126 | WIRED | Confirmed present |
| `src/include/tabfm_registration.hpp` | `RegisterScoringFunctions` | Declaration at line 19 | WIRED | Confirmed present |
| `CMakeLists.txt` | `src/tabfm_scoring.cpp` | EXTENSION_SOURCES at line 60 | WIRED | Confirmed present |
| `CMakeLists.txt` | `test/cpp/test_tabfm_scoring.cpp` | TABFM_CPP_TEST_SOURCES at line 136 | WIRED | Confirmed present |
| `tabfm_scoring.cpp:244-248` | `DistributionQuantile` | `#include "tabfm_predict.hpp"` + duckdb::vector conversion | WIRED | Interval score uses `DistributionQuantile(dv_probs, dv_borders, q)` |
| `tabfm_crossval.cpp:432` | `COMPARE_MODELS_MACRO` | `RegisterCVMacroWithAlias(loader, "anofox_tabfm_compare_models", "tabfm_compare_models", ...)` | WIRED | Confirmed present |
| CRPS aggregate STRUCT field order | Phase-2 `tabfm_predict_agg.cpp:100-103` | `dist_children[0]=logits, dist_children[1]=borders` | WIRED | Matches Phase-2 STRUCT layout; SQL test proves round-trip |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CRPS golden over K=16 non-uniform fixture | `./build/debug/test/unittest test/sql/tabfm_scoring.test` | 45/45 assertions | PASS |
| PSR-04 bind-gate for all three functions | same test file | statement error blocks pass | PASS |
| NaN coverage rejection (CR-01) | same test file | line 292-295 | PASS |
| 3-arg bind-gate for interval_score (IN-02) | same test file | line 306-310 | PASS |
| NULL list-element skip (WR-02) | same test file | lines 325-351 | PASS |
| Catch2 unit tests (ComputeCRPS/LogScore/IntervalScore) | `./build/debug/test/unittest "[tabfm_scoring]"` | 92/92 assertions | PASS |
| Python reference matches C++ golden values | `cd tools/parity && uv run python -m parity.crps_reference` | CRPS=0.530862, log=1.581928, IS=5.411898 | PASS |
| Identifier safety (SQL injection) | `tabfm_scoring.test:267-279` | 0 rows, no parser error | PASS |
| Full suite regression | `make test_debug` | 434/434 assertions, 16 test cases | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PSR-01 | 03-01 | CRPS over regression predictive distribution via SQL aggregate | SATISFIED | `tabfm_crps` registered; golden value 0.530862 verified in SQL test; 21 Catch2 + 45 SQL assertions pass |
| PSR-02 | 03-02 | Log-score (NLL) over regression predictive distribution | SATISFIED | `tabfm_log_score` registered; golden value 1.581928 verified; out-of-support max-penalty confirmed |
| PSR-03 | 03-02 | Interval score at configurable coverage level | SATISFIED | `tabfm_interval_score` registered; default 0.9 and explicit coverage verified; coverage 0.5 variant tested |
| PSR-04 | 03-01, 03-02 | Proper-scoring-rule functions bind-gated on distribution input | SATISFIED | All three functions have DOUBLE bind-gate overload; error message names "output_mode := 'distribution'"; 3-arg gate (IN-02) added for interval_score |
| CMP-01 | 03-02 | Cross-model comparison across multiple model families | SATISFIED | `tabfm_compare_models` macro in tabfm_crossval.cpp; tabpfn_v2 vs tabfm-v1 UNION ALL; golden verified in SQL test |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | No TBD/FIXME/XXX markers, no stubs, no empty implementations |

### Code Review Fix Verification

Both rounds of code review fixes confirmed applied and holding:

**Round 1 (6 findings, all fixed):**
- CR-01: `std::isnan(coverage)` guard — present at `tabfm_scoring.cpp:703`; NaN SQL test passes
- CR-02: `replace(data, '''', '''''')` in `tabfm_compare_models` — present at lines 326, 335; SQL test passes
- WR-01: Metric whitelist in `tabfm_cross_validate` — `CASE` whitelist at `tabfm_crossval.cpp:181-191`; injection test passes
- WR-02: Per-element `IsNull()` in all three Update loops — 25 null-check occurrences confirmed; NULL-element SQL tests pass
- WR-03: Telemetry-at-load limitation documented — comment at `tabfm_crossval.cpp:406-414`
- IN-01: `D_ASSERT(false)` unreachable marker — present at `tabfm_scoring.cpp:217`
- IN-02: 4th overload (DOUBLE, DOUBLE, DOUBLE) bind-gate for interval_score — present at lines 877-894; SQL test at line 306 passes

**Round 2 (1 finding, fixed):**
- WR-01 (round 2): Single-quote-escape of `data` in `tabfm_cross_validate` at 3 sites — lines 212, 228, 241 all use `replace(CAST(data AS VARCHAR), '''', '''''')` pattern

---

_Verified: 2026-09-22T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
