---
gsd_state_version: "1.0"
status: Awaiting next milestone
stopped_at: Phase 03 complete — all phases complete
last_updated: "2026-09-22T21:15:42.545Z"
last_activity: 2026-09-22
last_activity_desc: Milestone v1.0 completed and archived
state_head: 1a84844fc0fdf6ae9632211f4e8002ac27ff8811
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 10
  completed_plans: 10
  percent: 33
current_phase: 03
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-19)

**Core value:** Users can trust and compare tabular-foundation-model predictions directly in SQL — computing standard metrics and cross-validation on their own data, across more than one model family — without leaving DuckDB.
**Current focus:** Phase 03 — Proper Scoring Rules + Cross-Model Comparison

## Current Position

Phase: Milestone v1.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-09-22 — Milestone v1.0 completed and archived

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 4 | - | - |
| 02 | 4 | - | - |
| 03 | 2 | - | - |

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
| Phase 02 P01 | 525689 | 3 tasks | 12 files |
| Phase 02-model-generalization-distribution-output-fixture-backed P02 | 15 | 3 tasks | 8 files |
| Phase 02-model-generalization-distribution-output-fixture-backed P04 | 90 | 2 tasks | 9 files |
| Phase 03 P01 | 14 | 3 tasks | 9 files |
| Phase 03-proper-scoring-rules-cross-model-comparison P02 | 26 | 3 tasks | 6 files |

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
- [Phase 02]: ResolveModel moved before PreprocessBatch so manifest.preprocessing_profile is available for DispatchPreprocess dispatch (02-01)
- [Phase 02]: kKnownLicenses compile-time table with pre-sanitized option names; fixture-mit license excluded (ungated, no gate fires) (02-01)
- [Phase 02]: DistributionMean/DistributionQuantile placed outside anonymous namespace for external linkage so Catch2 TU can call them directly (02-02)
- [Phase 02]: kQuantileLevels={0.1,...,0.9} 9 levels in tabfm_predict.hpp; standard for regression confidence intervals (02-02)
- [Phase 02]: DecodeDistribution branches before ValidateTabFMOutput — distribution graphs produce rank-2 [n,K] not rank-3 [1,T,C] (02-02)
- [Phase 02]: GenericLicenseAccepted: backward-compat kBuiltinLicense path keeps hf_license SET working for tabfm-v1; non-builtin licenses use anofox_tabfm_accept_<sanitized>
- [Phase 02]: Distribution model backward-compat: model_emits_dist = !out.borders.empty() activates DecodeDistribution regardless of opts.distribution; emit_dist_cols flag suppresses columns when !opts.distribution
- [Phase 03]: PSR-01-external-linkage: ComputeCRPS given external linkage (declared in header) so Catch2 TU can call directly — same pattern as DistributionMean/DistributionQuantile (Phase 2 decision 02-02)
- [Phase 03]: PSR-04-overload-order: STRUCT overload registered BEFORE DOUBLE overload so DuckDB overload resolution prefers exact STRUCT match; bind-gate fires only for plain DOUBLE second arg
- [Phase 03]: PSR-02-external-linkage: ComputeLogScore given external linkage (defined outside anonymous namespace) so Catch2 can call directly — same pattern as ComputeCRPS (PSR-01-external-linkage)
- [Phase 03]: PSR-02-clip-eps-1e-10: Epsilon = 1e-10 for log-score clip (Assumption A2); larger than classification log-loss 1e-15 because bar distributions can have small mass in wide outer bins
- [Phase 03]: PSR-03-duckdb-vector-conversion: ComputeIntervalScore takes std::vector<double> for API consistency; converts to duckdb::vector<double> internally before calling DistributionQuantile
- [Phase 03]: CMP-01-tabfm-v1-psr-null: tabfm-v1 row in compare_models hard-emits NULL for crps/log_score/interval_score because tabfm-v1 has no predictive distribution output

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

Last session: 2026-09-22T20:34:27.138Z
Stopped at: Phase 03 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
