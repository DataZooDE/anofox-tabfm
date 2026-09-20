# Roadmap: anofox-tabfm — Evaluation Framework + Multi-Model Support

## Overview

This milestone turns a single-model prediction surface into an evaluation-and-comparison
platform. The journey follows the one hard dependency gate in the research: composable,
model-agnostic **point metrics and k-fold cross-validation** ship first (they work against
today's outputs with zero core-predict changes), then the **model seam is generalized and
new families onboarded** (profile registry + regression distribution output + TabPFN v2 /
TabICL), which finally **unlocks proper scoring rules and cross-model comparison** over the
user's own tables. All metric code stays model-agnostic, cpu-flavor-clean, and weight-free
(random-init fixtures per the license wall). Granularity is coarse: research's five suggested
phases are merged into three broad delivery boundaries.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Evaluation Metrics + Cross-Validation** - Model-agnostic classification/regression metric aggregates and leakage-safe k-fold CV over `(actual, predicted[, proba])` (completed 2026-09-21)
- [ ] **Phase 2: Model Generalization + Multi-Family Onboarding** - Preprocessing-profile registry, regression distribution output, TabPFN v2 + TabICL as first-class families
- [ ] **Phase 3: Proper Scoring Rules + Cross-Model Comparison** - Distribution-gated CRPS / log-score / interval score and evaluation across model families on user tables

## Phase Details

### Phase 1: Evaluation Metrics + Cross-Validation

**Goal**: Users can measure classification and regression prediction quality — and run leakage-safe k-fold cross-validation — directly in SQL over plain `(actual, predicted[, proba])` columns, independent of any model.
**Mode:** mvp
**Depends on**: Nothing (achievable with today's model outputs)
**Requirements**: CMET-01, CMET-02, CMET-03, CMET-04, CMET-05, CMET-06, RMET-01, RMET-02, RMET-03, RMET-04, CV-01, CV-02, CV-03, CV-04
**Success Criteria** (what must be TRUE):

  1. `SELECT tabfm_accuracy(actual, predicted) FROM t` and `tabfm_rmse(actual, predicted)` return values matching a golden fixture (verified against scikit-learn)
  2. `tabfm_f1(actual, predicted, avg := 'macro')` and `tabfm_roc_auc(actual, proba, avg := 'ovr')` require an explicit averaging mode (no silent default) and match sklearn on a 3-class fixture, ties handled via rank-sum
  3. `tabfm_log_loss(actual, proba)` clips probabilities (no infinities) and `tabfm_ece(actual, proba)` computes Expected Calibration Error from the per-class MAP; `tabfm_confusion_matrix(actual, predicted)` returns a table-valued result
  4. Regression aggregates handle edge cases: `tabfm_r2` on a constant target and `tabfm_mape`/`tabfm_medae` on zero actuals return documented, non-crashing results
  5. `tabfm_cross_validate(...)` assigns rows to k deterministic seedable folds, trains each fold's context and predicts its held-out rows via the two-table (leakage-safe) predict form, returns per-fold and aggregate (mean ± std) results, and safely quotes the target identifier (fixes the P1 bug at `tabfm_macros.cpp:91`)

**Plans**: 4/4 plans executed

- [x] 01-01-PLAN.md — Scaffold wiring for all three modules + end-to-end accuracy tracer (CMET-01)
- [x] 01-02-PLAN.md — Classification metrics: precision/recall/F1, log-loss, ECE, ROC-AUC, confusion matrix (CMET-02..06)
- [x] 01-03-PLAN.md — Regression metrics: RMSE, MAE, R², MAPE, median absolute error (RMET-01..04)
- [x] 01-04-PLAN.md — Cross-validation: seedable folds, leakage-safe k-fold, per-fold+aggregate output, safe quoting, leakage test (CV-01..04)

### Phase 2: Model Generalization + Multi-Family Onboarding

**Goal**: Users can load and predict with more than one foundation-model family — TabPFN v2 (with a native regression predictive distribution) and TabICL — each with its own preprocessing profile, weight-free fixture, and license gate.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: MGEN-01, MGEN-02, MGEN-03, RDIST-01, RDIST-02, MODL-01, MODL-02, MODL-03, MODL-04
**Success Criteria** (what must be TRUE):

  1. A model whose manifest names `preprocessing_profile: 'tabpfn_v2'` (or `tabicl_v2`) loads and predicts through the registry-dispatched preprocessing; an unknown profile fails with a named, actionable error, and the existing `tabfm-v1` profile still works via self-registration
  2. Model-output validation (shape/rank/class-count) runs before decode for every family and rejects contract violations with a named error (extends the existing P0 gap)
  3. `tabfm_regress(..., opts := {'output_mode': 'distribution'})` against TabPFN v2 emits `yhat_dist` (per-bin logits) and `yhat_quantiles`, while point-estimate output remains the backward-compatible default and is empty when a model does not emit a distribution
  4. `tabpfn_v2` and `tabicl_v2` are available as first-class families (manifest + profile + committed weight-free random-init ONNX fixture), and predicting with each returns results verified against a golden fixture
  5. Each non-`tabfm-v1` family enforces its own license-acceptance gate before download, and `tools/parity` validates each family's ONNX output contract (including both TabPFN v2 distribution tensors) before its C++ decoder is trusted

**Plans**: TBD

### Phase 3: Proper Scoring Rules + Cross-Model Comparison

**Goal**: Users can score probabilistic regression predictions with proper scoring rules and compare model families head-to-head on their own tables using the Phase 1 evaluation primitives.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: PSR-01, PSR-02, PSR-03, PSR-04, CMP-01
**Success Criteria** (what must be TRUE):

  1. `tabfm_crps(actual, yhat_dist)` computes CRPS over a bar distribution (energy-score identity) and matches a golden fixture; `tabfm_log_score(actual, yhat_dist)` computes NLL over the predictive distribution
  2. `tabfm_interval_score(actual, yhat_dist, coverage := 0.9)` computes the interval score at a configurable coverage level
  3. Proper-scoring-rule functions are bind-gated on distribution input and fail with a named remedy (pointing at `output_mode := 'distribution'`) when given point estimates
  4. A user can run the Phase 1 metrics and Phase 3 scoring rules across `tabpfn_v2`, `tabicl_v2`, and `tabfm-v1` on their own table to compare model quality, with no bundled datasets involved

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Evaluation Metrics + Cross-Validation | 4/4 | Complete    | 2026-09-21 |
| 2. Model Generalization + Multi-Family Onboarding | 0/TBD | Not started | - |
| 3. Proper Scoring Rules + Cross-Model Comparison | 0/TBD | Not started | - |

## Recommended Spikes

Two spikes are warranted before their downstream Phase 2 work is planned (research flags,
`SUMMARY.md` §Research Flags):

- **TabPFN v2 tensor contract** (before Phase 2 distribution decode): confirm exact ONNX
  output names/shapes for per-bin logits **and** bin borders, plus K (bin count), via a
  checkpoint inspection + `tools/export_onnx` run. Assuming uniform bins yields
  plausible-but-wrong quantiles/CRPS.
- **TabICL ONNX export feasibility** (immediately, gates Phase 2 scope): the three-stage
  Transformer's `torch.dynamo` traceability with dynamic shapes is unconfirmed, and no
  published ONNX export path exists. Spike before committing the TabICL onboarding scope.
