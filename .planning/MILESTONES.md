# Milestones

## v1.0 Evaluation Framework + Multi-Model Support (Shipped: 2026-09-22)

**Phases completed:** 3 phases, 10 plans, 18 tasks

**Key accomplishments:**

- Metric/CV scaffold wired once for all three parallel plans; tabfm_accuracy aggregate (CMET-01) ships fully with UnifiedVectorFormat NULL-skip, telemetry-at-bind, and sklearn-golden sqllogictest.
- Six classification metrics (F1/precision/recall with required avg, log-loss with eps clipping, ROC-AUC with rank-sum tie handling, ECE, and a safely-quoted confusion matrix TABLE MACRO) shipped sklearn-golden and error-path tested.
- Five regression metrics — RMSE, MAE, R² (constant-target safe), MAPE (zero-actual skip), and MedAE (heap-owning sorted-sample) — shipped sklearn-golden with full NULL-safety and documented edge-case behavior.
- complete (3/3 tasks). MODL-01 (fixture-scoped) and MODL-04 (fixture parity) delivered.
- tabfm_log_score (PSR-02) and tabfm_interval_score (PSR-03) added as aggregate extensions of the Plan A CRPS spine; tabfm_compare_models (CMP-01) added as a SQL macro to tabfm_crossval.cpp; all golden-correct, bind-gated, and covered by 37 SQL + 92 Catch2 assertions; full 2486-assertion suite passes.

---
