---
phase: 02-model-generalization-distribution-output-fixture-backed
plan: "02-03"
subsystem: tabpfn-v2-fixture
status: complete
tags: [fixture, tabpfn_v2, parity, modl, weight-free, onnx]
completed: "2026-09-22"
duration_minutes: 240

dependency_graph:
  requires:
    - tabfm_profile_registry (02-01: ProfileRegistration + ForceProfileInit chaining)
    - distribution decode path (02-02: TabFMRunOutput.borders, DecodeDistribution)
  provides:
    - tools/parity (uv project — K=16 fixture builder + distribution contract validator, MODL-04)
    - test/fixtures/tabpfn_v2/ (weight-free random-init ONNX graph + manifest.json + golden.json + FIXTURE_SHA256, MODL-01)
    - tabpfn_v2 fixture-scoped preprocessing profile (self-registers "tabpfn_v2", MODL-01)
  affects:
    - tools/parity/* (new uv project)
    - test/fixtures/tabpfn_v2/* (committed weight-free fixture family)
    - src/tabfm_preprocess_tabpfn_v2.cpp (fixture-scoped profile body)

key_files:
  created:
    - tools/parity/pyproject.toml
    - tools/parity/src/parity/check_tabpfn_v2.py
    - test/fixtures/tabpfn_v2/graph_tabpfn_v2.onnx
    - test/fixtures/tabpfn_v2/model.safetensors
    - test/fixtures/tabpfn_v2/manifest.json
    - test/fixtures/tabpfn_v2/golden.json
    - test/fixtures/tabpfn_v2/tensor_map_tabpfn_v2.json
    - test/fixtures/tabpfn_v2/FIXTURE_SHA256
  modified:
    - src/tabfm_preprocess_tabpfn_v2.cpp

requirements: [MODL-01, MODL-04]
---

# Plan 02-03 Summary — TabPFN v2 fixture family + tools/parity

**Status:** complete (3/3 tasks). MODL-01 (fixture-scoped) and MODL-04 (fixture parity) delivered.

## What was built

- **Task 1 — tools/parity (MODL-04, commit `73bba53`):** a new `uv` project that
  builds the K=16 weight-free fixture graph and validates the distribution output
  **contract** — two named outputs, `logits [n,K]` float32 and `borders [K+1]`
  float32 with strictly-increasing (non-uniform) borders — before the C++ decoder
  is trusted. `uv run pytest -q` → 7 passed.
- **Task 2 — committed fixture family (MODL-01, commit `891aa9e`):**
  `test/fixtures/tabpfn_v2/` holds a **weight-free random-init** ONNX graph
  (1.2 KB), a 864-byte random-init safetensors, `manifest.json`
  (`preprocessing_profile: tabpfn_v2`, `distribution_output: true`, ungated
  fixture license so tests need no license SET), `golden.json`, and
  `FIXTURE_SHA256`. All artifacts sha256-verified stable; **no weight bytes**
  (license wall upheld).
- **Task 3 — fixture-scoped tabpfn_v2 preprocessing profile (MODL-01, commit
  `64d26bc`):** `TabPFNV2PreprocessBatch` self-registers `"tabpfn_v2"` and returns
  a minimal, correct-shape `PreprocessedBatch` (features passed through; train rows
  first / test rows last; `target_mean`/`target_scale` from training targets for
  02-02's affine transform to raw space; regression, no label decoder). Documented
  as fixture-scoped — full-fidelity tabpfn_v2 preprocessing is deferred with the
  real-export work.

## Verification

- `make debug` → exit 0 (profile compiles + links; self-registration intact).
- `cd tools/parity && uv run pytest -q` → 7 passed (contract validator).
- `sha256sum -c test/fixtures/tabpfn_v2/FIXTURE_SHA256` → all OK (byte-stable).
- Existing suite unaffected (registry + Phase 1 tests unchanged).

## Notes / deviations

- **Watchdog stall during execution:** the executor was killed by the stream
  watchdog (600s no-output) during the ~1.9 GB relink — a false-positive stall,
  not broken work. Tasks 1–2 were already committed; the orchestrator completed
  the build (which passed), committed the finished Task-3 profile, verified parity
  + sha, and wrote this SUMMARY.
- The end-to-end distribution **predict** on this fixture (load via
  `SET anofox_tabfm_model_manifest` → assert decode vs `golden.json`) is exercised
  in plan **02-04** (the capstone), per the plan split.
- Broken-windows ledger entry #3 (tabpfn_v2 profile stub) is now filled.
