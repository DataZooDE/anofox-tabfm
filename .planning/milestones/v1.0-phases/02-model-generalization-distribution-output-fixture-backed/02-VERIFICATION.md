---
phase: 02-model-generalization-distribution-output-fixture-backed
verified: 2026-09-22T21:30:00Z
status: passed
score: 14/14 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-01-PLAN.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-01-SUMMARY.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-02-PLAN.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-02-SUMMARY.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-03-PLAN.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-03-SUMMARY.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-04-PLAN.md
  - .planning/phases/02-model-generalization-distribution-output-fixture-backed/02-04-SUMMARY.md
  - src/include/tabfm_manifest.hpp
  - src/include/tabfm_ort_engine.hpp
  - src/include/tabfm_predict.hpp
  - src/include/tabfm_profile_registry.hpp
  - src/tabfm_engine.cpp
  - src/tabfm_manifest.cpp
  - src/tabfm_ort_engine.cpp
  - src/tabfm_predict_agg.cpp
  - src/tabfm_preprocess_tabpfn_v2.cpp
  - src/tabfm_profile_registry.cpp
  - src/tabfm_settings.cpp
  - src/tabfm_weights.cpp
  - test/cpp/test_tabfm_distribution_decode.cpp
  - test/cpp/test_tabfm_profile_registry.cpp
  - test/fixtures/tabpfn_v2/FIXTURE_SHA256
  - test/fixtures/tabpfn_v2/golden.json
  - test/fixtures/tabpfn_v2/manifest.json
  - test/sql/tabfm_distribution.test
  - test/sql/tabfm_license.test
  - test/sql/tabfm_profile_registry.test
  - tools/parity/src/parity/check_tabpfn_v2.py
  - tools/parity/tests/test_check_tabpfn_v2.py
covered_digest: "v1:sha256:230afaaa28dddda9b7c3999220bb1b5f2b8b67a241aba6a1081822833ec540be"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 02: Model Generalization + Distribution Output (fixture-backed) Verification Report

**Phase Goal:** Generalize the model seam so families beyond tabfm-v1 are first-class, and carry a regression predictive distribution end-to-end against the confirmed TabPFN v2 tensor contract (logits [n,K] + non-uniform borders [K+1]), proven with a committed weight-free random-init fixture family. Real TabPFN v2 export + TabICL are DEFERRED (upstream-blocked).
**Verified:** 2026-09-22T21:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | A model whose manifest names `preprocessing_profile: 'tabfm_v1_minimal'` loads and predicts exactly as before (backward-compat regression guard) (MGEN-01) | ✓ VERIFIED | `tabfm_profile_registry.test` passes 8 assertions including positive case (6 rows, all finite yhats). `BuiltinTabFMManifest` JSON literals at `tabfm_manifest.cpp:288,307` carry `"tabfm_v1_minimal"` and Catch2 Test 3 verifies this explicitly (11 assertions pass). `make test_debug` 387/387. |
| 2  | Loading a model whose manifest names an unregistered `preprocessing_profile` fails with a named, actionable error before any ORT run (MGEN-02) | ✓ VERIFIED | `DispatchPreprocess` exact-match lookup at `tabfm_profile_registry.cpp:55-63` throws `InvalidInputException` naming the profile id and `tabfm_models()` hint. SQL test verifies both substrings. Catch2 Test 1 `REQUIRE_THROWS_WITH(Catch::Contains("no_such_profile") && Catch::Contains("tabfm_models()"))` passes. |
| 3  | The tabfm-v1 profile self-registers via a static initializer forced live at Load() (MGEN-01) | ✓ VERIFIED | `static const ProfileRegistration kTabFMV1Reg(kPreprocessProfileId, PreprocessBatch)` at `tabfm_profile_registry.cpp:70`. `ForceProfileInit()` called at `anofox_tabfm_extension.cpp:118`. `ForceTabPFNV2ProfileInit()` chained from `ForceProfileInit()`. |
| 4  | Model-output validation rejects distribution graph contract violations (wrong rank, wrong borders length, truncated logits) with named errors BEFORE decode (MGEN-03) | ✓ VERIFIED | `ValidateDistributionOutput` at `tabfm_ort_engine.cpp:602-651` checks rank-2 shape, borders.size()==K+1, and logits.size()==T*K before any indexing. Catch2 Test 4 verifies 5 contract violations all throw `InvalidInputException` (16 assertions, all pass). The CR-01 truncated-logits case is explicitly covered. |
| 5  | ORT run output carries optional regression predictive distribution (per-bin logits [n,K] + bin borders [K+1]), empty when a model does not emit one (RDIST-01) | ✓ VERIFIED | `TabFMRunOutput.borders` field at `tabfm_ort_engine.hpp:225`. `Run()` at `tabfm_ort_engine.cpp:510-558` detects "borders" output by name, requests both outputs, copies borders tensor. Empty for sessions without a "borders" output node. |
| 6  | `output_mode='distribution'` emits `yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])` and `yhat_quantiles DOUBLE[]` computed over the model's NON-UNIFORM borders (RDIST-02) | ✓ VERIFIED | `EmitDistribution()` at `tabfm_predict_agg.cpp:69-71`; `ListStructType` appends both fields at lines 99-103. `DecodeDistribution` at `tabfm_engine.cpp:896-976` uses `DistributionMean`+`DistributionQuantile` with actual bucket widths. `tabfm_distribution.test` asserts column presence, list lengths (K=16, K+1=17, 9 quantiles), and golden values within 1e-3 — all 21 assertions pass. |
| 7  | Point-estimate output remains the backward-compatible default; distribution fields are absent unless `output_mode='distribution'` AND task is regression (RDIST-02) | ✓ VERIFIED | `DESCRIBE SELECT * FROM tabfm_regress(...)` compact mode returns 0 rows matching 'logits','borders','yhat_quantiles'. `tabfm_distribution.test` line 49-53 asserts this. `tabfm_regress.test` passes 20 assertions unchanged. |
| 8  | Distribution mean and quantiles match a golden fixture within 1e-4 (decode math: non-uniform borders, FullSupportBarDistribution, affine transform) | ✓ VERIFIED | Catch2 Tests 1-3 in `test_tabfm_distribution_decode.cpp` verify mean(-0.16644), q0.1(-1.80915), q0.5(-0.09423), q0.9(1.40019), raw_mean(9.50068), raw_q0.5(9.71730) all within 1e-4. `tabfm_distribution.test` cross-checks SQL decode vs `golden.json` within 1e-3 for both test rows. All 16 Catch2 assertions and 21 SQL assertions pass. |
| 9  | A committed weight-free random-init tabpfn_v2 fixture family exists (K=16, manifest+ONNX+safetensors+golden+sha256), with no Google/vendor weight bytes in the repo (MODL-01, license wall) | ✓ VERIFIED | All 5 files exist in `test/fixtures/tabpfn_v2/`. `sha256sum -c FIXTURE_SHA256` passes (5/5 OK). Safetensors metadata explicitly states "random init...not Google's / TabPFN's". ONNX graph has random-init W weights (seed=42, not Google weights). `manifest.json` carries `distribution_output:true`, `preprocessing_profile:'tabpfn_v2'`, `license:'fixture-mit'`. |
| 10 | `tools/parity` validates the fixture's TWO distribution tensors (logits [T,K] + borders [K+1], strictly increasing, non-uniform) against the confirmed contract (MODL-04) | ✓ VERIFIED | `uv run pytest -q` in `tools/parity/`: 7 passed. `uv run check_tabpfn_v2` exits 0 with "PASS: logits (6, 16) float32, borders (17,) float32, strictly increasing (min_diff=0.3115, max_diff=0.9317, std=0.1919)". The parity tool explicitly checks non-uniformity (std of diffs > 1e-4). |
| 11 | The tabpfn_v2 preprocessing profile produces a fixture-scoped PreprocessedBatch (correct T/H/train_size/x/y/target_mean/target_scale) that the engine can feed to the fixture graph (MODL-01) | ✓ VERIFIED | `TabPFNV2PreprocessBatch` in `tabfm_preprocess_tabpfn_v2.cpp` is fully implemented (not the stub). Populates all required fields including z-scored targets, target_mean/target_scale, row_source_index. Self-registers via `kTabPFNV2Reg`. End-to-end tabfm_distribution.test passes with no ORT errors. |
| 12 | Each non-tabfm-v1 family enforces its own license-acceptance gate before download, keyed on manifest license id, with an error naming the exact SET to run (MODL-03) | ✓ VERIFIED | `RequireLicenseAccepted` at `tabfm_weights.cpp:323-341` calls `GenericLicenseAccepted` keyed on manifest.license. `SanitizeLicenseId` at line 107 lowercases and replaces non-alphanumeric. `tabfm_license.test` verifies: per-family gate fires naming `SET anofox_tabfm_accept_tabpfn_v2_cc_by_nc_4_0 = true`; gate fires before I/O; accepting option clears gate. 33 assertions pass. |
| 13 | The existing `anofox_tabfm_accept_hf_license` gate still works for tabfm-non-commercial-v1.0 (backward compat) | ✓ VERIFIED | `GenericLicenseAccepted` at `tabfm_weights.cpp:128-131` checks legacy `LicenseAccepted()` for `kBuiltinLicense`. `RequireLicenseAccepted` at line 330-334 preserves the exact original error text containing `anofox_tabfm_accept_hf_license`. License test assertions 1-15 covering this path all pass. |
| 14 | The ungated fixture license (`fixture-mit`) fires no gate; the tabpfn_v2 fixture predicts without acceptance (MODL-03) | ✓ VERIFIED | `IsGated()` at `tabfm_weights.cpp:68-70` explicitly treats `"fixture-mit"` as ungated. License test section verifies tabfm_download with fixture-mit fails at I/O step ("air-gapped manifest"), not at the license gate. Distribution test runs predict on the fixture without any license SET. |

**Score:** 14/14 truths verified (0 present, behavior-unverified)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Real TabPFN v2 ONNX inference export (MODL-01-EXPORT) | Phase v2 | REQUIREMENTS.md v2 section: "Real TabPFN v2 ONNX inference export — blocked on data-dependent preprocessing + chunked attention failing torch.export." |
| 2 | TabICL as a first-class classification family (MODL-02) | Phase v2 | REQUIREMENTS.md v2 section: "MODL-02 (deferred from v1, 2026-09-21): TabICL... needs 2-3 upstream PRs to soda-inria/tabicl." |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/include/tabfm_profile_registry.hpp` | Registry header with PreprocessFn, ProfileRegistration, DispatchPreprocess, ForceProfileInit | ✓ VERIFIED | 63 lines, full implementation |
| `src/tabfm_profile_registry.cpp` | Registry with function-local-static map, kTabFMV1Reg self-registration | ✓ VERIFIED | 89 lines, all symbols present |
| `src/tabfm_preprocess_tabpfn_v2.cpp` | tabpfn_v2 fixture-scoped preprocessing profile | ✓ VERIFIED | 204 lines, fully implemented (not stub) |
| `src/tabfm_ort_engine.cpp` | ValidateDistributionOutput with logits.size check | ✓ VERIFIED | CR-01 fix at lines 640-650 present |
| `src/tabfm_engine.cpp` | DistributionMean, DistributionQuantile, DecodeDistribution | ✓ VERIFIED | All three present with external linkage |
| `src/include/tabfm_predict.hpp` | distribution field in options, yhat_dist_* in result, kQuantileLevels | ✓ VERIFIED | All fields present |
| `src/tabfm_predict_agg.cpp` | EmitDistribution(), ListStructType distribution fields, finalize population | ✓ VERIFIED | All wired correctly |
| `src/tabfm_weights.cpp` | SanitizeLicenseId, GenericLicenseAccepted, RequireLicenseAccepted generic | ✓ VERIFIED | All present |
| `test/cpp/test_tabfm_profile_registry.cpp` | 3 Catch2 test cases including CR-01 regression guard | ✓ VERIFIED | 11 assertions, all pass |
| `test/cpp/test_tabfm_distribution_decode.cpp` | 4 Catch2 test cases with golden values, MGEN-03 contract violations | ✓ VERIFIED | 16 assertions, all pass |
| `test/sql/tabfm_profile_registry.test` | Unknown profile error + tabfm-v1 positive dispatch | ✓ VERIFIED | 8 assertions pass |
| `test/sql/tabfm_license.test` | Per-family gate + legacy compat + ungated fixture | ✓ VERIFIED | 33 assertions pass |
| `test/sql/tabfm_distribution.test` | End-to-end distribution capstone vs golden.json | ✓ VERIFIED | 21 assertions pass |
| `test/fixtures/tabpfn_v2/manifest.json` | distribution_output:true, preprocessing_profile:'tabpfn_v2' | ✓ VERIFIED | All fields correct |
| `test/fixtures/tabpfn_v2/graph_tabpfn_v2.onnx` | K=16 ONNX graph with named 'logits' and 'borders' outputs | ✓ VERIFIED | Both outputs confirmed by onnx inspection |
| `test/fixtures/tabpfn_v2/golden.json` | logits, borders, decoded means/quantiles for C++ parity | ✓ VERIFIED | Present with n_test=2, 9 quantile levels |
| `test/fixtures/tabpfn_v2/FIXTURE_SHA256` | sha256 for all 5 fixture artifacts | ✓ VERIFIED | All 5 checksums pass |
| `tools/parity/src/parity/check_tabpfn_v2.py` | Contract validator: shapes, strictly increasing, non-uniform | ✓ VERIFIED | Full implementation including non-uniformity check |
| `tools/parity/tests/test_check_tabpfn_v2.py` | pytest: positive + negative border + missing output tests | ✓ VERIFIED | 7 tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tabfm_engine.cpp:770` | `tabfm_profile_registry.cpp:DispatchPreprocess` | `DispatchPreprocess(resolved.manifest.preprocessing_profile, ...)` | ✓ WIRED | Verified at line 770; ResolveModel moved before preprocessing |
| `anofox_tabfm_extension.cpp:118` | `tabfm_profile_registry.cpp:ForceProfileInit` | Direct call in `LoadInternal` | ✓ WIRED | Verified at `anofox_tabfm_extension.cpp:118` |
| `TabFMRunOutput.borders` | `tabfm_ort_engine.cpp Run()` | ORT reads "borders" named output when session declares it | ✓ WIRED | Verified at `tabfm_ort_engine.cpp:513-558` |
| `manifest.distribution_output` | `tabfm_engine.cpp Decode()` | `(!out.borders.empty() && task==REGRESSION)` at line 832 | ✓ WIRED | Distribution branch engages on borders presence |
| `ParseOneOption output_mode='distribution'` | `TabFMPredictOptions.distribution` | `tabfm_predict_agg.cpp:161-163` | ✓ WIRED | Sets `opts.distribution = true` |
| `tabfm_predict_agg.cpp ListStructType` | `yhat_dist + yhat_quantiles` | `EmitDistribution()` guard at lines 99-103 | ✓ WIRED | Fields only appended when distribution+regression |
| `tabfm_weights.cpp RequireLicenseAccepted` | `GenericLicenseAccepted(manifest.license)` | Replaces old hf-only check at line 324 | ✓ WIRED | Both legacy and generic paths covered |
| `tabfm_distribution.test SET model_manifest` | tabpfn_v2 fixture → distribution decode | Full chain exercised by 21 SQL assertions | ✓ WIRED | Runs through registry → preprocess → ORT → decode |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `tabfm_distribution.test` yhat_dist | `yhat_dist_logits/borders` in `TabFMPredictResult` | ORT Run → `DecodeDistribution` → `tabfm_predict_agg.cpp finalize` | Yes: borders from ORT tensor, logits computed via softmax | ✓ FLOWING |
| `tabfm_distribution.test` yhat_quantiles | `yhat_quantiles` in result | `DistributionQuantile(probs, raw_borders, kQuantileLevels[qi])` | Yes: real computation from ORT output tensors | ✓ FLOWING |
| `tabfm_distribution.test` yhat (mean) | `result.yhat[src]` | `DistributionMean(probs, raw_borders)` after affine transform | Yes: derived from ORT logits+borders with z→raw transform | ✓ FLOWING |
| `tabfm_profile_registry.test` dispatch | `DispatchPreprocess` return | `PreprocessBatch` called via function pointer from registry | Yes: real preprocessing executed | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite | `DATAZOO_DISABLE_TELEMETRY=1 make test_debug` | 387 assertions in 15 test cases — all passed | ✓ PASS |
| Distribution end-to-end | `./build/debug/test/unittest test/sql/tabfm_distribution.test` | 21 assertions in 1 test case — all passed | ✓ PASS |
| License gate | `./build/debug/test/unittest test/sql/tabfm_license.test` | 33 assertions in 1 test case — all passed | ✓ PASS |
| Profile registry | `./build/debug/test/unittest test/sql/tabfm_profile_registry.test` | 8 assertions in 1 test case — all passed | ✓ PASS |
| Distribution decode Catch2 | `./build/debug/test/unittest "[tabfm][distribution_decode]"` | 16 assertions in 4 test cases — all passed | ✓ PASS |
| Profile registry Catch2 | `./build/debug/test/unittest "[tabfm][profile_registry]"` | 11 assertions in 3 test cases — all passed | ✓ PASS |
| Parity tool pytest | `cd tools/parity && uv run pytest -q` | 7 passed in 0.21s | ✓ PASS |
| Parity contract check | `cd tools/parity && uv run check_tabpfn_v2` | PASS: logits (6, 16) float32, borders (17,) float32, strictly increasing | ✓ PASS |
| Fixture sha256 | `cd test/fixtures/tabpfn_v2 && sha256sum -c FIXTURE_SHA256` | 5/5 OK | ✓ PASS |
| ONNX output names | `onnx.load(graph)` → check outputs | `{'logits', 'borders'}` — both present | ✓ PASS |
| License wall | safetensors metadata | "random init...not Google's / TabPFN's" explicitly confirmed | ✓ PASS |
| Build | `make debug` | 61/61 targets, no errors | ✓ PASS |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared. The parity tool (`uv run check_tabpfn_v2`) serves as the equivalent contract probe and was run directly above.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MGEN-01 | 02-01 | Preprocessing-profile registry maps manifest `preprocessing_profile` to C++ function; tabfm-v1 self-registers | ✓ SATISFIED | Registry implemented; kTabFMV1Reg self-registers; backward-compat test passes |
| MGEN-02 | 02-01 | Unknown profiles fail with named, actionable error before ORT run | ✓ SATISFIED | DispatchPreprocess throws InvalidInputException; tested in SQL + Catch2 |
| MGEN-03 | 02-02 | Model-output validation (shape/rank/class-count) runs before decode for every family | ✓ SATISFIED | ValidateDistributionOutput checks rank-2 + borders.size + logits.size; CR-01 fix verified; Catch2 Test 4 covers all violation cases |
| RDIST-01 | 02-02 | ORT run output can carry regression predictive distribution (logits + borders), empty when model does not emit one | ✓ SATISFIED | TabFMRunOutput.borders optional field; Run() reads "borders" output when present |
| RDIST-02 | 02-02 | output_mode='distribution' emits yhat_dist + yhat_quantiles for regression, backward-compatible default | ✓ SATISFIED | EmitDistribution() gate; SQL test verifies column presence, list lengths, golden values |
| MODL-01 | 02-03 | tabpfn_v2 fixture family: manifest + profile + weight-free K=16 ONNX | ✓ SATISFIED (fixture-scoped) | Fixture committed, sha256 verified, profile implemented, end-to-end test passes |
| MODL-02 | — | TabICL family | CORRECTLY EXCLUDED — deferred to v2 (upstream-blocked) | REQUIREMENTS.md v2 section; not a gap |
| MODL-03 | 02-04 | Per-family license-acceptance gate keyed on manifest license id | ✓ SATISFIED | GenericLicenseAccepted + SanitizeLicenseId; license test covers per-family gate + legacy compat + ungated fixture |
| MODL-04 | 02-03 | tools/parity validates tabpfn_v2 fixture's ONNX distribution contract | ✓ SATISFIED (fixture-scoped) | pytest 7/7 pass; check_tabpfn_v2 passes with non-uniformity verification |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/tabfm_preprocess_tabpfn_v2.cpp` | 1-20 | Comment marks profile as "fixture-scoped; full-fidelity ... deferred" | ℹ️ Info | Intentional and documented; this IS the MODL-01 fixture-scoped deliverable, not a stub |
| `src/tabfm_engine.cpp` | 890-895 | Comment "CURRENT CONTRACT: ...real TabPFN v2 wire format emits [n_test, K] only" | ℹ️ Info | Documents known future work; no code change needed until real export lands |

No TBD/FIXME/XXX debt markers found in any phase-modified file. No HACK/PLACEHOLDER markers. No empty return stubs. No hardcoded empty data in production paths.

### Human Verification Required

None. All phase claims were verified empirically via build + test runs. The distribution decode math was verified against Python-derived golden values in both Catch2 (unit) and SQL (integration) tests.

### Code Review Fix Verification

**CR-01** (`ValidateDistributionOutput` missing `logits.size() == T * K` check) was confirmed fixed:
- `tabfm_ort_engine.cpp:640-650` contains the expected logits count check with `InvalidInputException` naming the manifest SET.
- `test_tabfm_distribution_decode.cpp` Test 4 includes the truncated-logits sub-case (`N*8-1` elements), which passes (throws `InvalidInputException`).

**IN-01** (Test 3 task-enum mismatch) was confirmed fixed:
- `test_tabfm_profile_registry.cpp:139` uses `PreprocessTask::CLASSIFICATION` for `cls_manifest.preprocessing_profile` and adds a second `REQUIRE_NOTHROW` for `reg_manifest.preprocessing_profile` with `PreprocessTask::REGRESSION`.

---

_Verified: 2026-09-22T21:30:00Z_
_Verifier: Claude (gsd-verifier)_
