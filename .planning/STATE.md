---
gsd_state_version: "1.0"
current_phase: 01
current_phase_name: Evaluation Metrics + Cross-Validation
status: verifying
stopped_at: Completed 01-04-PLAN.md
last_updated: "2026-09-20T22:06:50.942Z"
last_activity: 2026-09-20
last_activity_desc: Phase 01 execution started
state_head: c34192ebe1ebbd114eb9ad4921fc8f4ad236c86c
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 4
  completed_plans: 4
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-19)

**Core value:** Users can trust and compare tabular-foundation-model predictions directly in SQL — computing standard metrics and cross-validation on their own data, across more than one model family — without leaving DuckDB.
**Current focus:** Phase 01 — Evaluation Metrics + Cross-Validation

## Current Position

Phase: 01 (Evaluation Metrics + Cross-Validation) — EXECUTING
Plan: 4 of 4
Status: Phase complete — ready for verification
Last activity: 2026-09-20 — Phase 01 execution started

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

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

- Two spikes warranted before Phase 2 planning: TabPFN v2 ONNX tensor contract (logits + bin borders + K) and TabICL ONNX export feasibility (dynamo traceability unconfirmed). See ROADMAP.md §Recommended Spikes.
- CV leakage-detecting golden test must be written before Phase 1 CV ships (non-negotiable, research §Gaps).
- REQUIREMENTS.md header says "26 total" but 28 IDs are enumerated (CMET/RMET/CV/MGEN/RDIST/MODL/PSR/CMP). Traceability covers all 28; header count is stale.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| *(none)* | | | | |

## Session Continuity

Last session: 2026-09-20T22:06:50.924Z
Stopped at: Completed 01-04-PLAN.md
Resume file: None
