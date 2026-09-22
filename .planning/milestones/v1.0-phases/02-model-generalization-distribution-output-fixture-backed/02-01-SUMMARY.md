---
phase: 02-model-generalization-distribution-output-fixture-backed
plan: "01"
subsystem: profile-registry
status: complete
tags: [registry, dispatch, mgen, modl, scaffold]
completed: "2026-09-21"
duration_minutes: 90

dependency_graph:
  requires: []
  provides:
    - tabfm_profile_registry (DispatchPreprocess seam for all Phase-2 plans)
    - ForceProfileInit (load-time force-link of registry TUs)
    - tabfm_preprocess_tabpfn_v2 (compiling stub, plan 03 fills body)
    - per-license settings (anofox_tabfm_accept_tabfm_non_commercial_v1_0, anofox_tabfm_accept_tabpfn_v2_cc_by_nc_4_0)
    - CMakeLists.txt registry+tabpfn_v2 sources wired (whole Phase-2 batch)
  affects:
    - src/tabfm_engine.cpp (dispatch seam, ResolveModel moved before preprocessing)
    - src/anofox_tabfm_extension.cpp (ForceProfileInit called at Load)
    - src/include/tabfm_registration.hpp (ForceProfileInit declaration added)
    - src/tabfm_settings.cpp (per-license options added)

tech_stack:
  added:
    - preprocessing-profile registry (unordered_map function-local static, C++17)
    - ProfileRegistration self-registration RAII pattern (mirrors GetPredictEngine)
  patterns:
    - Factory-registration via static initializers (ForceProfileInit prevents stripping)
    - Exact-match dispatch with fail-closed InvalidInputException (T-02-01)
    - DuckDB Load()-time option pre-registration loop (kKnownLicenses table)

key_files:
  created:
    - src/include/tabfm_profile_registry.hpp
    - src/tabfm_profile_registry.cpp
    - src/tabfm_preprocess_tabpfn_v2.cpp
    - test/cpp/test_tabfm_profile_registry.cpp
    - test/cpp/test_tabfm_distribution_decode.cpp
    - test/sql/tabfm_profile_registry.test
    - test/fixtures/regression/manifest_bad_profile.json
  modified:
    - src/tabfm_engine.cpp (ResolveModel moved before preprocess; DispatchPreprocess replaces PreprocessBatch)
    - src/anofox_tabfm_extension.cpp (ForceProfileInit call added)
    - src/include/tabfm_registration.hpp (ForceProfileInit declaration)
    - src/tabfm_settings.cpp (per-license options)
    - CMakeLists.txt (all Phase-2 sources + test sources)

decisions:
  - "ResolveModel moved before preprocessing (not after) so manifest.preprocessing_profile is available for DispatchPreprocess; ordering change is safe (no dependency on batch data)"
  - "kKnownLicenses uses pre-sanitized ids in the struct to avoid runtime SanitizeLicenseId call at Load(); matches SanitizeLicenseId output exactly"
  - "tabpfn_v2 stub registers immediately so registry is consistent across all Phase-2 parallel plans; body throws NotImplementedException until plan 03 fills it"
  - "test_tabfm_distribution_decode.cpp created as hidden placeholder ([.]) so CMakeLists.txt can list it now; plan 02 replaces body"

plan_head_before: f19775fa915927b68b047217e6a163ae6456650a

actuals:
  tokens: 7140
  tasks: 3
  commits: 3
---

# Phase 02 Plan 01: Preprocessing-Profile Registry + Scaffold Batch Summary

Registry dispatch seam wiring tabfm_v1_minimal through DispatchPreprocess with backward compat, unknown-profile fail-closed error, tabpfn_v2 compiling stub, per-license options pre-registered, and all Phase-2 CMakeLists.txt edits done in one coordinated batch.

## What Was Built

### Task 1 — End-to-end registry dispatch (tracer)

New module `src/tabfm_profile_registry.cpp` + header implements the preprocessing-profile
registry as a function-local-static `unordered_map<string, PreprocessFn>` (mirrors
`GetPredictEngine()` at `tabfm_engine.cpp:781`). `ProfileRegistration` is a self-registration
RAII struct; `DispatchPreprocess` performs an exact-match lookup and throws
`InvalidInputException` naming the fix for unknown profiles (T-02-01).

The tabfm_v1_minimal profile self-registers via `static const ProfileRegistration kTabFMV1Reg`
(references `kPreprocessProfileId` constant, not a literal). `ForceProfileInit()` chains
`ForceTabPFNV2ProfileInit()` to prevent linker stripping of both TUs (T-02-02, RESEARCH Pitfall 2).

`tabfm_preprocess_tabpfn_v2.cpp` is a compiling stub: the "tabpfn_v2" profile self-registers
but throws `NotImplementedException` until plan 03 fills the body.

`tabfm_engine.cpp`: `ResolveModel()` moved before preprocessing so `resolved.manifest.
preprocessing_profile` is available; `PreprocessBatch` call replaced with `DispatchPreprocess`.

Scaffold batch (CLAUDE.md rule 2, coordinated for whole Phase 2):
- `CMakeLists.txt`: registry + tabpfn_v2 stub added to `EXTENSION_SOURCES`; both new Catch2
  TUs added to `TABFM_CPP_TEST_SOURCES`; placeholder `test_tabfm_distribution_decode.cpp`
  (hidden `[.]` test) created so the build links — plan 02 replaces body.
- `tabfm_registration.hpp`: `ForceProfileInit()` declaration added.
- `anofox_tabfm_extension.cpp`: `ForceProfileInit()` called before predict registration.

### Task 2 — Registry unit + SQL error-path tests

`test/cpp/test_tabfm_profile_registry.cpp`: two Catch2 cases — unknown profile throws
`InvalidInputException` containing "no_such_profile" and "tabfm_models()" (substring checks);
tabfm_v1_minimal dispatches correctly (T=3, train_size=2, H>=1).

`test/sql/tabfm_profile_registry.test`: bad-profile manifest triggers named error substrings
"no_such_profile" and "tabfm_models()"; positive case with regression fixture returns 6 rows
all finite (MGEN-01 backward compat confirmed via the registry path).

`test/fixtures/regression/manifest_bad_profile.json`: copy of regression manifest with
`preprocessing_profile: "no_such_profile"` for the SQL error-path test.

### Task 3 — Per-license acceptance options at Load() (MODL-03 scaffold)

`src/tabfm_settings.cpp`: `kKnownLicenses` compile-time table with pre-sanitized ids
(`tabfm_non_commercial_v1_0`, `tabpfn_v2_cc_by_nc_4_0`) loops `config.AddExtensionOption`
for each. Existing `anofox_tabfm_accept_hf_license` unchanged (backward compat). Plan 04
reads these via `TryGetCurrentSetting(opt_name)` in the generic license gate.

## Verification Results

| Test | Result |
|------|--------|
| `make debug` | PASS |
| `./build/debug/test/unittest test/sql/tabfm_regress.test` | PASS — 20 assertions (tabfm-v1 backward compat) |
| `./build/debug/test/unittest "[tabfm][profile_registry]"` | PASS — 7 assertions in 2 test cases |
| `./build/debug/test/unittest test/sql/tabfm_profile_registry.test` | PASS — 8 assertions |
| `./build/debug/test/unittest test/sql/settings.test` | PASS — 32 assertions |

## Deviations from Plan

None. Plan executed exactly as written with one minor structural note:

The plan noted that `resolved` (from `ResolveModel`) was created AFTER line 674
(`PreprocessBatch`), creating an ordering conflict for `DispatchPreprocess`. Resolution:
`ResolveModel` was moved to run BEFORE preprocessing (steps renumbered 1→2). This is
architecturally correct — `ResolveModel` does file I/O only and has no dependency on the
preprocessed batch. The tensor-materialization order (steps 3→5) is preserved intact.

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | 95f7bb6 | feat(02-01): preprocessing-profile registry + engine dispatch (MGEN-01/02) |
| 2 | 25e03b9 | test(02-01): registry unit + SQL error-path tests (MGEN-01/02) |
| 3 | 4176e0e | feat(02-01): pre-register per-license acceptance options at Load() (MODL-03) |

## Known Stubs

| File | Description | Resolved by |
|------|-------------|-------------|
| `src/tabfm_preprocess_tabpfn_v2.cpp` | `TabPFNV2PreprocessBatch` throws `NotImplementedException` — fixture-scoped implementation deferred | Plan 02-03 |
| `test/cpp/test_tabfm_distribution_decode.cpp` | Hidden placeholder `[.]` — no real assertions | Plan 02-02 |

## Threat Surface Scan

No new network endpoints, auth paths, or file access patterns introduced beyond plan scope.
The registry dispatch seam is the only new trust boundary: untrusted `preprocessing_profile`
string → exact-match lookup → fail-closed `InvalidInputException` (T-02-01 mitigated).
Static-init stripping prevented by `ForceProfileInit()` chain (T-02-02 mitigated).

## Self-Check

Files exist:
- src/include/tabfm_profile_registry.hpp — FOUND
- src/tabfm_profile_registry.cpp — FOUND
- src/tabfm_preprocess_tabpfn_v2.cpp — FOUND
- test/cpp/test_tabfm_profile_registry.cpp — FOUND
- test/sql/tabfm_profile_registry.test — FOUND
- test/fixtures/regression/manifest_bad_profile.json — FOUND

Commits: 95f7bb6, 25e03b9, 4176e0e — all present in git log.

## Self-Check: PASSED
