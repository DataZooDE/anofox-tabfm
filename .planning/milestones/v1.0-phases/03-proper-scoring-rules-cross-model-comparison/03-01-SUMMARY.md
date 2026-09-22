---
phase: 03-proper-scoring-rules-cross-model-comparison
plan: "01"
subsystem: scoring-aggregates
status: complete
tags: [psr, crps, aggregate, tdd, tracer]
completed: "2026-09-22"
duration_min: 14

dependency_graph:
  requires:
    - 02-04  # tabfm_regress with output_mode='distribution' producing yhat_dist STRUCT
  provides:
    - tabfm_crps aggregate (PSR-01)
    - RegisterScoringFunctions entry point
    - Scaffold batch: CMakeLists.txt + registration.hpp + extension.cpp updated for scoring module
  affects:
    - src/anofox_tabfm_extension.cpp (added RegisterScoringFunctions call)
    - src/include/tabfm_registration.hpp (added declaration)
    - CMakeLists.txt (added scoring.cpp + test source)

tech_stack:
  added:
    - tabfm_scoring.cpp: CRPS aggregate (CRPSState, SoftmaxInPlace copy, ComputeCRPS with external linkage)
    - tabfm_scoring.hpp: RegisterScoringFunctions + ComputeCRPS declarations
    - tools/parity/src/parity/crps_reference.py: Python crps_bar() golden reference
  patterns:
    - AggregateFunctionSet with two overloads (STRUCT normal + DOUBLE PSR-04 bind-gate)
    - inputs[1].GetValue(i) + StructValue::GetChildren + ListValue::GetChildren for STRUCT read
    - ComputeCRPS external linkage for Catch2 direct calls (same as DistributionMean/DistributionQuantile)
    - RegisterAggregateFunctionSetWithAlias for full + short alias atomically

key_files:
  created:
    - src/tabfm_scoring.cpp
    - src/include/tabfm_scoring.hpp
    - test/cpp/test_tabfm_scoring.cpp
    - test/sql/tabfm_scoring.test
    - tools/parity/src/parity/crps_reference.py
  modified:
    - CMakeLists.txt (EXTENSION_SOURCES + TABFM_CPP_TEST_SOURCES)
    - src/include/tabfm_registration.hpp
    - src/anofox_tabfm_extension.cpp
    - tools/parity/pyproject.toml (crps_reference entry point)

decisions:
  - id: PSR-01-external-linkage
    summary: "ComputeCRPS given external linkage (declared in header, defined outside anonymous namespace) so Catch2 TU can call it directly — same pattern as DistributionMean/DistributionQuantile (Phase 2 decision 02-02)"
  - id: PSR-01-tail-a1
    summary: "CRPS integrates only over [b_0, b_K] (no half-normal tail extension) per assumption A1 in 03-RESEARCH.md; documented in code comment naming the research section"
  - id: PSR-04-overload-order
    summary: "STRUCT overload (fn_dist) registered BEFORE DOUBLE overload (fn_point) so DuckDB overload resolution prefers exact STRUCT match and falls through to bind-gate only for plain DOUBLE second args"

requirements: [PSR-01, PSR-04]

actuals:
  tokens: 10000
  tasks: 3
  commits: 3
  plan_head_before: bb23dc82e2adaf3e3db5751ea36abbd7417088e6
---

# Phase 03 Plan 01: tabfm_crps Tracer + Scoring Module Scaffold Summary

JWT auth with refresh rotation — wait, wrong domain. The one-liner:

**tabfm_crps registered as a proper-scoring-rule aggregate over the Phase-2 bar-distribution STRUCT, golden-tested at 0.530862 mean CRPS with non-uniform bins, PSR-04 bind-gate rejecting point estimates, aliased as both tabfm_crps and anofox_tabfm_crps.**

## What Was Built

Delivered `tabfm_crps` (PSR-01) fully end-to-end as the phase tracer:

1. **`src/tabfm_scoring.cpp`** — New WS-scoring module:
   - `SoftmaxInPlace` (copied verbatim from tabfm_engine.cpp anonymous namespace, temperature=1.0)
   - `ComputeCRPS(y, probs, borders)` — analytical closed-form CRPS with per-bin Case A/B/C integrals over non-uniform bins (w_i = borders[i+1]-borders[i]), external linkage for testability
   - `CRPSState {double sum_crps; int64_t n}` aggregate triplet (Init/Update/Combine/Finalize)
   - `CRPSUpdate` reads `inputs[1]` via `GetValue(i)` → `StructValue::GetChildren` → `ListValue::GetChildren` (T-03-01 bounds-checked: K==0, size mismatch, NULL children — all skip)
   - `CRPSBind` emits telemetry once per bind
   - PSR-04 bind-gate: second `{DOUBLE, DOUBLE}` overload that always throws `InvalidInputException` naming `output_mode := 'distribution'`
   - `RegisterScoringFunctions` building `AggregateFunctionSet` with both overloads + `FunctionDescription`, registered via `RegisterAggregateFunctionSetWithAlias`

2. **`src/include/tabfm_scoring.hpp`** — Header declaring `RegisterScoringFunctions` and `ComputeCRPS` (external linkage)

3. **Scaffold batch** (all scaffold-owned files edited together per CLAUDE.md rule #2):
   - `CMakeLists.txt`: `src/tabfm_scoring.cpp` → EXTENSION_SOURCES; `test/cpp/test_tabfm_scoring.cpp` → TABFM_CPP_TEST_SOURCES
   - `src/include/tabfm_registration.hpp`: `RegisterScoringFunctions` declaration added
   - `src/anofox_tabfm_extension.cpp`: `anofox::RegisterScoringFunctions(loader)` added after `RegisterCrossValidateMacros`

4. **`tools/parity/src/parity/crps_reference.py`** — Documented Python `crps_bar()` reference with `main()` printing synthetic K=4 anchor + fixture K=16 mean CRPS (0.530862); registered as `crps_reference` console entry point in `pyproject.toml`

5. **`test/cpp/test_tabfm_scoring.cpp`** — 8 Catch2 test cases:
   - `ComputeCRPS` direct unit tests (K=4 y=1.5 inside, y=-1.0 outside/left tail, empty probs guard, size mismatch guard)
   - SQL aggregate tests (empty group → NULL, PSR-04 bind-gate, K=4 synthetic via SQL, NULL-skip)

6. **`test/sql/tabfm_scoring.test`** — 7 sqllogictest assertions:
   - PSR-01: `round(tabfm_crps(y_val, {'logits':logits,'borders':borders}), 6) == 0.530862` against fixture
   - PSR-01 alias: `anofox_tabfm_crps(...)` returns same value
   - PSR-04: `statement error` with `tabfm_crps: second argument must be a distribution`
   - NULL/empty: `WHERE 1=0` returns NULL

## CRPS Math Implementation

The CRPS closed-form over the bar distribution's piecewise-linear CDF:
- **Case A (y < b_i)**: `w_i * ((C_i-1)^2 + (C_i-1)*p_i + p_i^2/3)`
- **Case B (y > b_{i+1})**: `w_i * (C_i^2 + C_i*p_i + p_i^2/3)`
- **Case C (y inside)**: `L + R` where L is the left-of-y integral (indicator=0) and R is the right-of-y integral (indicator=1)

w_i = borders[i+1] - borders[i] (non-uniform, never 1/K). Zero-width bins skipped.

## Test Results

- `make debug`: PASS (build/debug/test/unittest produced)
- `./build/debug/test/unittest "[tabfm_scoring]"`: 31 assertions in 8 test cases — ALL PASSED
- `./build/debug/test/unittest test/sql/tabfm_scoring.test`: 7 assertions in 1 test case — ALL PASSED
- `cd tools/parity && uv run python -m parity.crps_reference`: prints synthetic K=4 = 0.3958... and mean fixture CRPS = 0.530862

## Success Criteria Verification

- [x] `tabfm_crps` / `anofox_tabfm_crps` registered, callable, golden-correct within 1e-6 (PSR-01)
- [x] Point-estimate second argument rejected at bind with named remedy (PSR-04)
- [x] CRPS uses non-uniform bin widths (w_i = borders[i+1]-borders[i])
- [x] Malformed distributions and NULLs are skipped safely (T-03-01, T-03-02)
- [x] Scaffold batch complete: module in CMakeLists, registration declared and called
- [x] `make debug` builds; existing suite regression deferred to Plan B capstone

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] NULL struct construction in Catch2 test**
- **Found during:** Task 2 (Catch2 test execution)
- **Issue:** `NULL::{logits DOUBLE[], borders DOUBLE[]}` is invalid DuckDB SQL cast syntax for anonymous struct types; DuckDB requires `NULL::STRUCT(logits DOUBLE[], borders DOUBLE[])` with the STRUCT keyword
- **Fix:** Rewrote the NULL-row test case to use UNION ALL with explicit `NULL::STRUCT(logits DOUBLE[], borders DOUBLE[])` cast
- **Files modified:** test/cpp/test_tabfm_scoring.cpp
- **Commit:** 418ac9c

**2. [Rule 1 - Bug] statement error sqllogictest syntax**
- **Found during:** Task 3 (SQL test execution)
- **Issue:** `statement error` blocks require the expected error text after a `----` separator; a bare `statement error` line without `----` is not valid sqllogictest syntax
- **Fix:** Added `----` + expected error prefix after each `statement error` block
- **Files modified:** test/sql/tabfm_scoring.test
- **Commit:** 26b67b9

## Threat Mitigations Applied

| Threat | T-ID | Mitigation |
|--------|------|------------|
| OOB index on malformed distribution | T-03-01 | K==0 || border_vals.size()!=K+1 → skip row |
| Zero-width bin divide-by-zero | T-03-02 | wi <= 0.0 → skip bin (cum accumulates pi) |
| Point-estimate passed as distribution | T-03-03 | PSR-04 bind-gate; DOUBLE overload always throws |

## Known Stubs

None. All functions fully implemented and golden-tested.

## Self-Check: PASSED

- `/home/simonm/projects/duckdb/anofox-tabfm/src/tabfm_scoring.cpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/src/include/tabfm_scoring.hpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/test/cpp/test_tabfm_scoring.cpp` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/test/sql/tabfm_scoring.test` — FOUND
- `/home/simonm/projects/duckdb/anofox-tabfm/tools/parity/src/parity/crps_reference.py` — FOUND
- Commit 93f7337 — FOUND (feat(03-01): tabfm_crps scoring module + scaffold wiring)
- Commit 418ac9c — FOUND (feat(03-01): Python CRPS reference + Catch2 unit tests)
- Commit 26b67b9 — FOUND (feat(03-01): tabfm_scoring.test — end-to-end CRPS golden)
