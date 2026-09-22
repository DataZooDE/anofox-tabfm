---
phase: 03-proper-scoring-rules-cross-model-comparison
plan: "02"
subsystem: scoring-aggregates + crossval-macros
status: complete
tags: [psr, log-score, interval-score, crps, aggregate, tdd, cmp-01, macro]
completed: "2026-09-22"
duration_min: 26

dependency_graph:
  requires:
    - 03-01  # RegisterScoringFunctions entry point + tabfm_crps tracer
    - 02-04  # tabfm_regress with output_mode='distribution' producing yhat_dist STRUCT
  provides:
    - tabfm_log_score aggregate (PSR-02)
    - tabfm_interval_score aggregate with coverage bind-data (PSR-03)
    - tabfm_compare_models cross-model comparison macro (CMP-01)
  affects:
    - src/tabfm_scoring.cpp (extended with log_score + interval_score)
    - src/tabfm_crossval.cpp (extended with compare_models macro)
    - test/cpp/test_tabfm_scoring.cpp (added 13 new test cases)
    - test/sql/tabfm_scoring.test (added PSR-02/03/CMP-01 blocks)
    - tools/parity/src/parity/crps_reference.py (added log_score_bar + interval_score_bar)

tech_stack:
  added:
    - ComputeLogScore: NLL = log(w_i) - log(max(1e-10, p_i)); clip_eps=1e-10 (A2)
    - ComputeIntervalScore: Gneiting & Raftery IS with DistributionQuantile for l/u
    - IScoreBindData (FunctionData subclass): stores coverage at bind time (Pitfall 5)
    - IScoreBind: validates 0 < coverage < 1 at bind via ExpressionExecutor::EvaluateScalar
    - COMPARE_MODELS_MACRO: tabpfn_v2 (distribution) vs tabfm-v1 (point) UNION ALL
  patterns:
    - Deviation fix: expression_executor include path is duckdb/execution/ not duckdb/planner/
    - ComputeIntervalScore converts std::vector → duckdb::vector when calling DistributionQuantile
    - duckdb/execution/expression_executor.hpp (correct include path for EvaluateScalar)
    - IScoreBindData::Equals + Copy pattern from F1BindData (tabfm_metrics_classification.cpp:205)
    - Three-overload AggregateFunctionSet: fn_dist2 (2-arg) + fn_dist3 (3-arg) + fn_point (gate)
    - COMPARE_MODELS_MACRO uses replace(target,'"','""') at every identifier interpolation (T-03-06)

key_files:
  created: []
  modified:
    - src/include/tabfm_scoring.hpp (added ComputeLogScore + ComputeIntervalScore declarations)
    - src/tabfm_scoring.cpp (tabfm_log_score + tabfm_interval_score aggregates)
    - src/tabfm_crossval.cpp (COMPARE_MODELS_MACRO + registration)
    - test/cpp/test_tabfm_scoring.cpp (13 new Catch2 cases for PSR-02/03)
    - test/sql/tabfm_scoring.test (PSR-02/03/CMP-01 blocks)
    - tools/parity/src/parity/crps_reference.py (log_score_bar + interval_score_bar)

decisions:
  - id: PSR-02-external-linkage
    summary: "ComputeLogScore given external linkage (defined outside anonymous namespace) so Catch2 can call directly — same pattern as ComputeCRPS (PSR-01-external-linkage)"
  - id: PSR-02-clip-eps-1e-10
    summary: "Epsilon = 1e-10 for log-score clip (Assumption A2); larger than classification log-loss 1e-15 because bar distributions can have small mass in wide outer bins"
  - id: PSR-03-duckdb-vector-conversion
    summary: "ComputeIntervalScore takes std::vector<double> for API consistency; converts to duckdb::vector<double> internally before calling DistributionQuantile (which uses DuckDB's vector typedef)"
  - id: PSR-03-three-overload-set
    summary: "tabfm_interval_score uses three-overload AggregateFunctionSet: 2-arg STRUCT (default coverage), 3-arg STRUCT+DOUBLE (explicit coverage), 2-arg DOUBLE (PSR-04 gate) — STRUCT overloads registered first per Pitfall 6"
  - id: CMP-01-tabfm-v1-psr-null
    summary: "tabfm-v1 row in compare_models hard-emits NULL for crps/log_score/interval_score because tabfm-v1 has no predictive distribution output; documented in macro description"
  - id: CMP-01-test-only-tabpfn-v2
    summary: "CMP-01 SQL test only verifies the tabpfn_v2 row (WHERE model='tabpfn_v2'); tabfm-v1 inference requires downloading a 6.5GB model which is unavailable in CI"

requirements: [PSR-02, PSR-03, CMP-01]

actuals:
  tokens: 16476
  tasks: 3
  commits: 2
  plan_head_before: c753e303de8f2b3b37bb08a19dc96d05725f24d4
---

# Phase 03 Plan 02: tabfm_log_score + tabfm_interval_score + tabfm_compare_models Summary

**tabfm_log_score (PSR-02) and tabfm_interval_score (PSR-03) added as aggregate extensions of the Plan A CRPS spine; tabfm_compare_models (CMP-01) added as a SQL macro to tabfm_crossval.cpp; all golden-correct, bind-gated, and covered by 37 SQL + 92 Catch2 assertions; full 2486-assertion suite passes.**

## What Was Built

### Task 1+2: tabfm_log_score (PSR-02) and tabfm_interval_score (PSR-03)

1. **`ComputeLogScore(y, probs, borders)`** — external linkage:
   - NLL = `log(w_i) - log(max(1e-10, p_i))` for y in bin i
   - clip_eps = 1e-10 (Assumption A2); out-of-support y or w_i < 1e-300 → max penalty `-log(1e-10)` = 23.025...
   - Guard sequence: check y outside [b_0, b_K] first (Pitfall 7), then scan bins for containing bin, then check w_i (Pitfall 4)

2. **`ComputeIntervalScore(y, probs, borders, coverage)`** — external linkage:
   - IS = (u - l) + (2/alpha) × [(l-y)·1{y<l} + (y-u)·1{y>u}]
   - alpha = 1 - coverage; l/u = DistributionQuantile at alpha/2 and 1-alpha/2
   - Converts std::vector → duckdb::vector to call DistributionQuantile (external linkage in tabfm_predict.hpp)

3. **`IScoreBindData`** — FunctionData subclass storing coverage:
   - F1BindData pattern (tabfm_metrics_classification.cpp:205)
   - Read from arg[2] via `ExpressionExecutor::EvaluateScalar` at bind time
   - Default 0.9 when no 3rd argument; validates 0 < coverage < 1 (T-03-05)

4. **Three-overload `AggregateFunctionSet`** for interval_score:
   - fn_dist2: `(DOUBLE, yhat_dist STRUCT)` → default coverage=0.9
   - fn_dist3: `(DOUBLE, yhat_dist STRUCT, DOUBLE)` → explicit coverage (users write `coverage := 0.9`)
   - fn_point: `(DOUBLE, DOUBLE)` → PSR-04 bind-gate always throws

5. **tools/parity/crps_reference.py** extended with:
   - `log_score_bar()` — documented NLL reference
   - `interval_score_bar()` — Gneiting & Raftery IS reference
   - `_distribution_quantile()` — Python equivalent of C++ DistributionQuantile

### Task 3: tabfm_compare_models (CMP-01)

6. **`COMPARE_MODELS_MACRO`** in `src/tabfm_crossval.cpp`:
   - Params: `data`, `target` (required); `coverage` (optional, default 0.9)
   - UNION-ALLs two rows via `query()`:
     - tabpfn_v2: `tabfm_regress(..., output_mode='distribution')` → rmse + crps + log_score + interval_score
     - tabfm-v1: `tabfm_regress(...)` (point estimate) → rmse + NULL + NULL + NULL
   - Safe quoting at every identifier interpolation: `replace(target,'"','""')` for metric identifiers (T-03-06); `replace(target,"'","''")` for SQL string literals
   - Registered via `RegisterCVMacroWithAlias` as `anofox_tabfm_compare_models` + `tabfm_compare_models`

## Golden Values

| Metric | K=4 Synthetic (logits=[0,0,0,0], borders=[0,1,2,3,4], y=1.5) | K=16 Fixture Mean (4 training rows) |
|--------|---------------------------------------------------------------|--------------------------------------|
| log_score | ln(4) = 1.386294... | round(mean, 6) = **1.581928** |
| interval_score (0.9) | 3.6 | round(mean, 6) = **5.411898** |
| interval_score (0.5) | 2.0 | round(mean, 6) = **2.341920** |

## Test Results

- `cd tools/parity && uv run python -m parity.crps_reference`: Prints golden log_score + interval_score references
- `./build/debug/test/unittest "[tabfm_scoring]"`: **92 assertions in 21 test cases — ALL PASSED**
  - 8 CRPS cases (Plan A), 4 log-score unit, 3 interval-score unit, 6 SQL aggregate cases
- `./build/debug/test/unittest test/sql/tabfm_scoring.test`: **37 assertions in 1 test case — ALL PASSED**
  - PSR-01 (2), PSR-02 (4), PSR-03 (6 including coverage variants), CMP-01 (5 including identifier-safety)
- `make test_debug` (full suite): **2486 assertions in 136 test cases — ALL PASSED**

## Success Criteria Verification

- [x] `tabfm_log_score` / `anofox_tabfm_log_score` registered, callable, golden-correct (PSR-02)
- [x] `tabfm_interval_score` / `anofox_tabfm_interval_score` registered with default and explicit coverage (PSR-03)
- [x] Out-of-support y and zero-width bins return max penalty (`-log(1e-10)`) — no NaN/Inf (T-03-04)
- [x] `coverage := 0.0` and `coverage := 1.0` throw `InvalidInputException` naming the valid range (T-03-05)
- [x] Point-estimate (DOUBLE) second argument rejected at bind for all PSR functions (PSR-04 extended)
- [x] `tabfm_compare_models` compares tabpfn_v2 fixture distribution scores vs tabfm-v1 (CMP-01)
- [x] Identifier-safety: double-quote in target name does not cause SQL injection (T-03-06)
- [x] Full existing test suite passes (`make test_debug`)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking Issue] Wrong include path for ExpressionExecutor**
- **Found during:** Task 1+2 (first build attempt)
- **Issue:** The plan and PATTERNS.md cited `duckdb/planner/expression_executor.hpp` but the actual header in the DuckDB codebase is at `duckdb/execution/expression_executor.hpp`
- **Fix:** Changed include from `duckdb/planner/expression_executor.hpp` to `duckdb/execution/expression_executor.hpp` (verified by checking tabfm_metrics_classification.cpp)
- **Files modified:** src/tabfm_scoring.cpp
- **Commit:** 01d1faf

**2. [Rule 1 - Bug] ComputeLogScore zero-width bin test had wrong input**
- **Found during:** Task 1 (Catch2 run)
- **Issue:** Test used borders=[0,1,1] K=2 with y=1.0 — y falls in bin 0 [0,1] (non-degenerate), not the zero-width bin 1 [1,1]. The test expected max_penalty but the function correctly returned log(2).
- **Fix:** Changed test to use a single zero-width bin: borders=[1.0, 1.0], probs=[1.0], y=1.0 — genuinely degenerate case
- **Files modified:** test/cpp/test_tabfm_scoring.cpp
- **Commit:** 01d1faf

**3. [Rule 3 - Blocking Issue] std::vector vs duckdb::vector type mismatch**
- **Found during:** Task 1+2 (second build attempt after expression_executor fix)
- **Issue:** `ComputeIntervalScore` used `std::vector<double>` (required by the declared API for Catch2 testability) but `DistributionQuantile` (tabfm_predict.hpp:206) takes `duckdb::vector<double>` — an incompatible type
- **Fix:** Added explicit conversion in ComputeIntervalScore body: `duckdb::vector<double> dv_probs(probs.begin(), probs.end())` before the DistributionQuantile calls. Decision PSR-03-duckdb-vector-conversion documented.
- **Files modified:** src/tabfm_scoring.cpp
- **Commit:** 01d1faf

## Threat Mitigations Applied

| Threat | T-ID | Mitigation |
|--------|------|------------|
| log(0) / divide-by-zero in log-score | T-03-04 | `max(1e-10, p_i)` clip; `w_i < 1e-300` → max penalty; y outside [b_0,b_K] → max penalty |
| Invalid coverage parameter | T-03-05 | `IScoreBind` validates 0 < coverage < 1 via `ExpressionExecutor::EvaluateScalar`; throws named error |
| SQL injection via target in compare_models | T-03-06 | `replace(target,'"','""')` at every identifier interpolation; identifier-safety test block |
| OOB index on malformed distribution | T-03-07 | Same bounds-check as CRPSUpdate (K==0, size mismatch → skip row) |

## Known Stubs

None. All functions fully implemented and golden-tested.

## Self-Check: PASSED
- `/home/simonm/projects/duckdb/anofox-tabfm/src/tabfm_scoring.cpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/src/include/tabfm_scoring.hpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/src/tabfm_crossval.cpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/test/cpp/test_tabfm_scoring.cpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/test/sql/tabfm_scoring.test` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/tools/parity/src/parity/crps_reference.py` — FOUND
- Commit 01d1faf (tabfm_log_score + tabfm_interval_score) — FOUND
- Commit ee6ef9d (tabfm_compare_models CMP-01) — FOUND
