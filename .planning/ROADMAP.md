# Roadmap: anofox-tabfm

## Milestones

- ✅ **v1.0 — Evaluation Framework + Multi-Model Support** — Phases 1-3 (shipped 2026-09-22)

## Phases

<details>
<summary>✅ v1.0 — Evaluation Framework + Multi-Model Support (Phases 1-3) — SHIPPED 2026-09-22</summary>

Full detail archived at `.planning/milestones/v1.0-ROADMAP.md`; requirements at
`.planning/milestones/v1.0-REQUIREMENTS.md`; audit at
`.planning/milestones/v1.0-MILESTONE-AUDIT.md`; phase artifacts under
`.planning/milestones/v1.0-phases/`.

- [x] Phase 1: Evaluation Metrics + Cross-Validation (4 plans) — completed 2026-09-21
      — model-agnostic classification/regression metric aggregates + leakage-safe k-fold CV
- [x] Phase 2: Model Generalization + Distribution Output (fixture-backed) (4 plans) — completed 2026-09-22
      — preprocessing-profile registry, regression distribution output vs the confirmed TabPFN v2
      contract, generic license gate, weight-free TabPFN v2 fixture family
- [x] Phase 3: Proper Scoring Rules + Cross-Model Comparison (2 plans) — completed 2026-09-22
      — CRPS / log-score / interval-score (distribution-gated) + `tabfm_compare_models`

**Deferred to v2 (upstream-blocked; spike evidence under `.planning/milestones/v1.0-phases/`):**
- Real TabPFN v2 ONNX *inference* export (torch.export blocked on data-dependent preprocessing + chunked attention)
- TabICL first-class family (ONNX export infeasible on tabicl 2.2.0)

</details>
