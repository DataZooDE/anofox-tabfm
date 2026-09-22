---
milestone: v1
milestone_name: Evaluation Framework + Multi-Model Support
audited: 2026-09-22
status: passed
scores:
  requirements: 27/27 active (2 formally deferred to v2)
  phases: 3/3
  integration: 6/6 seams
  flows: 5/5
gaps: {}
tech_debt:
  - phase: 02-model-generalization-distribution-output-fixture-backed
    items:
      - "Deferred to v2 (upstream-blocked, user-approved rescope): real TabPFN v2 ONNX inference export — torch.export fails on data-dependent preprocessing + chunked attention. Fixture-scoped MODL-01 shipped instead."
      - "Deferred to v2 (upstream-blocked): MODL-02 TabICL first-class family — ONNX export infeasible on tabicl 2.2.0 (data-dependent Stage-1 branches); needs upstream PRs."
  - phase: cross-cutting
    items:
      - "Accepted deviation: CV/compare table macros fire telemetry at extension-load, not per-invocation — DuckDB table macros have no per-invocation bind callback (matches existing classify/regress macro pattern)."
deferred_to_v2:
  - "MODL-02 (TabICL) — .planning/spikes/SPIKE-tabicl-onnx-export.md"
  - "MODL-01-EXPORT (real TabPFN v2 inference export) — .planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md"
---

# Milestone v1 Audit — Evaluation Framework + Multi-Model Support

**Status: PASSED** (against the user-approved, spike-informed rescoped definition of done)

anofox-tabfm was extended from a single-model prediction surface into an
evaluation-and-comparison platform. All three phases verified `passed`
individually and the cross-phase integration is confirmed clean.

## Requirements coverage (3-source cross-referenced)

All active v1 requirements are **satisfied** (VERIFICATION.md passed + SUMMARY
frontmatter + REQUIREMENTS.md traceability agree):

| Group | Requirements | Phase | Status |
|-------|--------------|-------|--------|
| Classification metrics | CMET-01..06 | 1 | ✓ satisfied |
| Regression metrics | RMET-01..04 | 1 | ✓ satisfied |
| Cross-validation | CV-01..04 | 1 | ✓ satisfied |
| Model generalization | MGEN-01/02/03 | 2 | ✓ satisfied |
| Distribution output | RDIST-01/02 | 2 | ✓ satisfied |
| Model onboarding (fixture) | MODL-01 (fixture), MODL-03, MODL-04 (fixture) | 2 | ✓ satisfied |
| Proper scoring rules | PSR-01/02/03/04 | 3 | ✓ satisfied |
| Cross-model comparison | CMP-01 | 3 | ✓ satisfied |

**Formally deferred to v2 (not milestone gaps — user-approved rescope after spikes):**
- **MODL-02** (TabICL first-class family) — ONNX export infeasible on `tabicl 2.2.0`.
- **Real TabPFN v2 ONNX inference export** — blocked upstream; Phase 2 ships the
  fixture-scoped family + confirmed tensor contract; real inference needs a
  cleaned export path.

Both are recorded in REQUIREMENTS.md §v2 and STATE.md Deferred Items with spike
evidence under `.planning/spikes/`.

## Phase verifications

| Phase | Score | Status |
|-------|-------|--------|
| 1 — Evaluation Metrics + Cross-Validation | 5/5 must-haves | passed |
| 2 — Model Generalization + Distribution Output (fixture-backed) | 14/14 must-haves | passed |
| 3 — Proper Scoring Rules + Cross-Model Comparison | 12/12 must-haves | passed |

## Cross-phase integration (integration checker: PASS)

6/6 seams verified, 5/5 end-to-end flows wired, 1547 test assertions pass
(434 SQL + 1113 Catch2), no orphaned exports:
1. Predict → metrics (model-agnostic, `{ANY,…}`)
2. CV → predict → metrics (leakage-safe two-table + JOIN-back)
3. Registry → preprocess → predict (profile dispatch; unknown → named error)
4. **Distribution output STRUCT contract (Phase 2 → Phase 3)** — `yhat_dist
   STRUCT(logits[], borders[])` field-order contract matches end-to-end (the
   critical seam)
5. PSR → distribution (bind-gated; bounds-checked)
6. CMP-01 capstone — `tabfm_compare_models` composes Phase-1 metrics + Phase-3
   scores across `tabpfn_v2` (fixture) vs `tabfm-v1` on a user table

## Quality signals

- **Broken-windows ledger:** 0 open (all 4 stubs filled).
- **Code review:** every phase reviewed + re-reviewed; all Critical/Warning
  findings fixed and re-verified (incl. a production profile-id regression and
  several input-safety/bounds/injection issues caught before merge).
- **License wall:** upheld — all fixtures weight-free random-init; no vendor
  weight bytes; cpu-flavor-clean.
- **Known pre-existing (not milestone-introduced):** an ASan
  heap-use-after-free in DuckDB's own `BlockAllocator` shutdown path (submodule
  code; present before this milestone; all assertions pass before teardown).

## Verdict

The rescoped milestone is **complete, verified, and integration-clean**. Proceed
to complete-milestone. The two deferred model-onboarding items are tracked as v2
work with concrete spike evidence and unblocking steps.
