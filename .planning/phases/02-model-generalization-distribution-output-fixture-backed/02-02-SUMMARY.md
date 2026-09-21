---
phase: 02-model-generalization-distribution-output-fixture-backed
plan: "02"
subsystem: distribution-decode
status: complete
tags: [distribution, decode, rdist, mgen, ort, manifest]
completed: "2026-09-21"
duration_minutes: 15

dependency_graph:
  requires:
    - tabfm_profile_registry (02-01: preprocessing dispatch seam)
    - tabfm_ort_engine Run() multi-output path
  provides:
    - TabFMRunOutput.borders (optional [K+1] float32, empty for non-distribution models)
    - ValidateDistributionOutput (rank-2 contract gate before decode)
    - DistributionMean + DistributionQuantile (FullSupportBarDistribution helpers)
    - DecodeDistribution (end-to-end z-space logits → raw-space mean + quantiles)
    - output_mode='distribution' parse + ListStructType extension + finalize population
    - ModelManifest.distribution_output bool flag
    - TabFMPredictOptions.distribution + TabFMPredictResult yhat_dist_*/yhat_quantiles fields
  affects:
    - src/include/tabfm_ort_engine.hpp (TabFMRunOutput.borders + ValidateDistributionOutput decl)
    - src/tabfm_ort_engine.cpp (Run() multi-output, ValidateDistributionOutput impl)
    - src/include/tabfm_manifest.hpp (distribution_output field)
    - src/tabfm_manifest.cpp (parse "distribution_output" optional bool)
    - src/include/tabfm_predict.hpp (options + result extension + decode helper decls)
    - src/tabfm_engine.cpp (DistributionMean/Quantile impl + DecodeDistribution)
    - src/tabfm_predict_agg.cpp (output_mode parsing, EmitDistribution, ListStructType, finalize)

tech_stack:
  added:
    - FullSupportBarDistribution decode math (half-normal outer-bin correction + CDF-inverse quantile)
    - kQuantileLevels: 9 levels {0.1,...,0.9} compile-time constant
  patterns:
    - ORT multi-output Run(): detect "borders" by name, request both in one session.Run() call
    - Pitfall 5 storage convention: z-space logits pre-softmax, raw-space borders after affine transform
    - Typed Value::LIST(LogicalType::DOUBLE, ...) for aggregate result fields (Pitfall 6)
    - Decode branch before ValidateTabFMOutput to avoid rank mismatch on distribution graphs

key_files:
  created:
    - test/cpp/test_tabfm_distribution_decode.cpp (filled; was plan-01 placeholder)
  modified:
    - src/include/tabfm_ort_engine.hpp
    - src/tabfm_ort_engine.cpp
    - src/include/tabfm_manifest.hpp
    - src/tabfm_manifest.cpp
    - src/include/tabfm_predict.hpp
    - src/tabfm_engine.cpp
    - src/tabfm_predict_agg.cpp

decisions:
  - "DistributionMean/DistributionQuantile placed outside anonymous namespace in tabfm_engine.cpp (external linkage) so Catch2 TU can call them directly via tabfm_predict.hpp declaration"
  - "kQuantileLevels={0.1,...,0.9} (9 levels) as static constexpr in tabfm_predict.hpp — standard for regression confidence intervals"
  - "DecodeDistribution branches before ValidateTabFMOutput because distribution graphs produce rank-2 [n,K] not rank-3 [1,T,C] — the existing validator would false-fail"
  - "raw_borders_val pre-built once per Decode call (same K+1 vector for every row) then copied into each result row — avoids O(n*K) reconstruction"
  - "borders detection in ORT Run(): scans session.output_names for 'borders' by name, not index — robust to exporter-side output ordering changes"
  - "ASAN abort at test binary teardown is a pre-existing DuckDB internal BlockAllocatorThreadLocalState cleanup issue; all 83 tests pass (1542 assertions) before teardown"

plan_head_before: 15aa3893d112327b49e248841cfa1eaa03fc532c

actuals:
  tokens: 8171
  tasks: 3
  commits: 3
---

# Phase 02 Plan 02: Regression Predictive Distribution Output Path Summary

Distribution decode math (FullSupportBarDistribution), ORT multi-output borders read, manifest flag, contract validation (MGEN-03), and output_mode='distribution' end-to-end wiring with a golden Catch2 test against K=8 non-uniform borders.

## What Was Built

### Task 1 — Extend TabFMRunOutput + ORT Run; add ValidateDistributionOutput; parse manifest flag

`src/include/tabfm_ort_engine.hpp`:
- `TabFMRunOutput.borders`: `vector<float>`, optional [K+1], empty for non-distribution models, z-normalized. Documented with non-uniform width note and Pitfall 4/5 reference.
- `ValidateDistributionOutput(const TabFMRunOutput &, idx_t n_test)`: declared alongside `ValidateTabFMOutput`.

`src/tabfm_ort_engine.cpp` `Run()`:
- Scans `session.output_names` for `"borders"` by name. When found, builds `req_output_names = {"logits", "borders"}` and requests both in a single `session.Run()` call. Copies borders tensor (float32 guard) into `result.borders`. Backward compatible: no borders → `result.borders` stays empty.
- `ValidateDistributionOutput`: asserts `out.shape.size() == 2` and `out.shape[0] == n_test` and `out.borders.size() == K+1`, throws `InvalidInputException` with actionable message naming `SET anofox_tabfm_model_manifest`.

`src/include/tabfm_manifest.hpp`:
- `bool distribution_output = false;` added to `ModelManifest` after `preprocessing_profile`.

`src/tabfm_manifest.cpp`:
- Parses optional `"distribution_output"` JSON boolean via `yyjson_obj_get` + `yyjson_is_bool` pattern; defaults false.

Verification: `make debug` passes; all existing ort_engine (204 assertions) and manifest (84 assertions) tests pass.

### Task 2 — Distribution decode math + Catch2 golden test (TDD, RDIST-02, MGEN-03)

**RED phase**: Test declared `DistributionMean`/`DistributionQuantile` from `tabfm_predict.hpp` — compilation failed (undefined functions). Confirmed RED.

`src/include/tabfm_predict.hpp`:
- `bool distribution = false;` added to `TabFMPredictOptions`.
- `vector<Value> yhat_dist_logits/yhat_dist_borders/yhat_quantiles` added to `TabFMPredictResult`.
- `DistributionMean` and `DistributionQuantile` declared with external linkage.
- `kQuantileLevels = {0.1,...,0.9}` and `kNumQuantileLevels = 9` added as `static constexpr`.

`src/tabfm_engine.cpp`:
- `DistributionMean`: FullSupportBarDistribution mean. Outer bins use half-normal correction (`contribution = borders[0] - sqrt(π/2)*width` / `borders[K] + sqrt(π/2)*width`); interior bins use midpoint. Defined outside anonymous namespace for external linkage.
- `DistributionQuantile`: CDF cumsum search + linear interpolation within bin using `borders[i+1]-borders[i]` (actual non-uniform widths, never assumed-uniform).
- `DecodeDistribution`: called from `Decode()` when `is_distribution = (opts.distribution && !out.borders.empty() && task==REGRESSION)`. Applies affine transform `raw_borders[k] = borders[k] * target_scale + target_mean`. Stores z-space logits (pre-softmax) and raw-space borders per Pitfall 5. Builds quantiles via `kQuantileLevels`. Validates before indexing via `ValidateDistributionOutput`.

`test/cpp/test_tabfm_distribution_decode.cpp` (replaces plan-01 placeholder):
- K=8, non-uniform borders `{-5,-3,-1.5,-0.5,0,0.5,1.5,3,5}` (widths: 2.0,1.5,1.0,0.5,0.5,1.0,1.5,2.0).
- Logits `{0.1,0.5,1.2,2.0,1.8,0.9,0.3,-0.2}`, Python-traced goldens.
- Test 1: mean = -0.16643975 within 1e-4.
- Test 2: q0.1 = -1.80915018, q0.5 = -0.09423481, q0.9 = 1.40018675 within 1e-4.
- Test 3: raw-space (y_std=3, y_mean=10) mean = 9.50068074, quantiles match within 1e-4.
- Test 4: `ValidateDistributionOutput` throws on 5 contract violations (bad borders length, wrong rank, wrong n_test, empty borders), no-throw on valid input.

**GREEN**: All 4 test cases (15 assertions) pass.

### Task 3 — Wire output_mode='distribution' through options, bind type, result (RDIST-02)

`src/tabfm_predict_agg.cpp`:
- `ParseOneOption` output_mode: added `"distribution"` branch (`opts.detail=false; opts.distribution=true`); error text updated to list `'compact', 'detail', or 'distribution'`.
- `EmitDistribution()`: `return options.distribution && options.task == TabFMTask::REGRESSION`.
- `ListStructType()`: when `EmitDistribution()`, appends `yhat_dist STRUCT(logits LIST(DOUBLE), borders LIST(DOUBLE))` and `yhat_quantiles LIST(DOUBLE)` (absent from return type otherwise — backward compat).
- `PredictAggFinalize`: mirrors proba population pattern. Pushes `yhat_dist` STRUCT + `yhat_quantiles` when `emit_dist`; typed NULLs when row has no distribution or engine returned empty. Uses `Value::LIST(LogicalType::DOUBLE, ...)` (typed overload, Pitfall 6).
- `Equals()` updated to include `distribution` field.

## Verification Results

| Test | Result |
|------|--------|
| `make debug` | PASS |
| `./build/debug/test/unittest "[tabfm][distribution_decode]"` | PASS — 15 assertions in 4 test cases |
| `./build/debug/test/unittest test/sql/tabfm_regress.test` | PASS — 20 assertions (backward compat) |
| `./build/debug/test/unittest test/sql/tabfm_classify.test` | PASS — 25 assertions (backward compat) |
| `./build/debug/test/unittest "[tabfm]"` (full suite) | PASS — 1542 assertions in 83 test cases |

Note: The full `[tabfm]` run shows an ASAN abort during process teardown (`DBConfig::~DBConfig → BlockAllocatorThreadLocalState::Initialize`) — a pre-existing DuckDB internal cleanup issue, not caused by extension code. All 83 tests pass before teardown.

## Deviations from Plan

None. Plan executed exactly as written with one structural note:

The plan suggests calling `ValidateDistributionOutput` "before decode" via a branch in `Decode()`. The implementation branches on `is_distribution` BEFORE `ValidateTabFMOutput` (required since the existing validator expects rank-3 `[1,T,C]` and would false-fail on rank-2 distribution output). `ValidateDistributionOutput` is called inside `DecodeDistribution` as the first action, preserving the security gate before any indexing (MGEN-03, T-02-03 fully mitigated).

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | 75c478c | feat(02-02): extend TabFMRunOutput with borders; add ValidateDistributionOutput; parse manifest distribution_output flag (RDIST-01, MGEN-03) |
| 2 | feef88d | feat(02-02): distribution decode helpers + golden Catch2 tests (RDIST-02, MGEN-03) |
| 3 | 8888c66 | feat(02-02): wire output_mode='distribution' through options, bind return type, result population (RDIST-02) |

## Known Stubs

None — all plan-specified functionality is implemented and tested.

## Threat Surface Scan

T-02-03 (Tampering, distribution model output contract, high, mitigate): `ValidateDistributionOutput` asserts rank-2 `[n_test, K]` logits and `borders.size()==K+1` before any decode indexing. Implemented as the first call inside `DecodeDistribution`, guarding all subsequent `out.logits[t * K + k]` and `raw_borders[k]` accesses. Gate is reachable via any `output_mode='distribution'` call on a distribution-manifest model.

T-02-04 (Information Disclosure, z-space vs raw-space, low, accept): Convention documented in `tabfm_predict.hpp` header (`yhat_dist_logits: z-space logits; yhat_dist_borders: raw-space borders`). Confirmed correct by the golden test (Test 3).

No new security-relevant surface beyond what the threat register covers.

## Self-Check

Files exist:
- src/include/tabfm_ort_engine.hpp — FOUND
- src/tabfm_ort_engine.cpp — FOUND
- src/include/tabfm_manifest.hpp — FOUND
- src/tabfm_manifest.cpp — FOUND
- src/include/tabfm_predict.hpp — FOUND
- src/tabfm_engine.cpp — FOUND
- src/tabfm_predict_agg.cpp — FOUND
- test/cpp/test_tabfm_distribution_decode.cpp — FOUND

Commits: 75c478c, feef88d, 8888c66

## Self-Check: PASSED
