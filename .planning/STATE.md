---
gsd_state_version: "1.0"
current_phase: 2
current_phase_name: Model Generalization + Multi-Family Onboarding
status: planning
stopped_at: Phase 01 complete, ready to plan Phase 2
last_updated: "2026-09-20T23:10:30.539Z"
last_activity: 2026-09-21
last_activity_desc: Phase 01 complete, transitioned to Phase 2
state_head: fda45f8c40ca03a6acf7d428f565c7131808ff73
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
  percent: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-19)

**Core value:** Users can trust and compare tabular-foundation-model predictions directly in SQL — computing standard metrics and cross-validation on their own data, across more than one model family — without leaving DuckDB.
**Current focus:** Phase 01 — Evaluation Metrics + Cross-Validation

## Current Position

Phase: 2 — Model Generalization + Multi-Family Onboarding
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-21 — Phase 01 complete, transitioned to Phase 2

Progress: [███░░░░░░░] 33%

## Performance Metrics

**Velocity:**

- Total plans completed: 4
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 10 | 2 tasks | 15 files |
| Phase 01 P02 | 18 | 3 tasks | 4 files |
| Phase 01 P03 | 11 | 3 tasks | 4 files |
| Phase 01-evaluation-metrics-cross-validation P04 | multi-session (~600 minutes) | 4 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Evaluation framework ships before adding models (measure quality before comparing).
- Roadmap: Coarse granularity — research's 5 suggested phases merged into 3 (registry + distribution decode + onboarding folded into one model-seam phase).
- Roadmap: Hard dependency gate — PSR-* (Phase 3) blocked on RDIST-* + MODL-01 + MGEN-* (Phase 2); point metrics + CV (Phase 1) unblocked.
- [Phase 01]: Golden values hard-coded in .test files; generate_metric_fixtures.py traces sklearn origin but is not a build artifact — CI reproducibility without sklearn runtime dependency
- [Phase 01]: FunctionDescription required for all tabfm_*/anofox_tabfm_* registered functions per tabfm_function_docs.test contract
- [Phase 01]: Required avg enforced via 2-arg overload that always throws at bind — no silent default for F1/precision/recall/ROC-AUC
- [Phase 01]: tabfm_confusion_matrix implemented as TABLE MACRO (not C++ table function) wrapping GROUP BY with replace() identifier quoting
- [Phase 01]: MAPE returned as dimensionless ratio matching sklearn.mean_absolute_percentage_error
- [Phase 01]: R² constant-target guard uses |SS_tot|<1e-12 per sklearn convention; returns 1.0/0.0 never NaN/Inf
- [Phase 01]: MedAE even-N returns mean of two middle residuals matching sklearn median_absolute_error
- [Phase 01]: CV macro uses JOIN-back on row_key to recover actual labels after tabfm_classify forces label=NULL in two-table form
- [Phase 01]: tabfm_cross_validate uses list_transform(range(k), f -> sql_frag) + array_to_string for k-fold UNION ALL without C++ loop
- [Phase 01]: SELECT * EXCLUDE target used in test subquery to prevent UNION ALL BY NAME duplicate column error in tabfm_classify body

### Pending Todos

None yet.

### Blockers/Concerns

- ✓ RESOLVED 2026-09-21: Both Phase 2 spikes executed (`.planning/spikes/`). TabPFN v2 tensor contract CONFIRMED (K=5000, logits `[n,K]`, borders `[K+1]` non-uniform). TabICL ONNX export INFEASIBLE on tabicl 2.2.0 → deferred. Real TabPFN v2 inference export blocked → deferred. Phase 2 rescoped to fixture-backed slice (user-approved).
- ✓ RESOLVED (Phase 1): CV leakage-detecting golden test shipped in 01-04 (`test/sql/tabfm_crossval.test`).

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Model family | MODL-02 TabICL first-class family — ONNX export infeasible (tabicl 2.2.0 data-dependent Stage-1 branches); needs upstream PRs | Deferred → v2 | 2026-09-21 (Phase 2 rescope) | current |
| Model export | Real TabPFN v2 ONNX inference export — blocked on data-dependent preprocessing + chunked attention; Phase 2 ships fixture-scoped MODL-01 instead | Deferred → v2 | 2026-09-21 (Phase 2 rescope) | current |

## Session Continuity

Last session: 2026-09-20T22:06:50.924Z
Stopped at: Phase 01 complete, ready to plan Phase 2
Resume file: None
