# Requirements: anofox-tabfm — Evaluation Framework + Multi-Model Support

**Defined:** 2026-09-19
**Core Value:** Users can trust and compare tabular-foundation-model predictions directly in SQL — computing standard metrics and cross-validation on their own data, across more than one model family — without leaving DuckDB.

## v1 Requirements

Requirements for this milestone. Each maps to a roadmap phase. All metric
functions are model-agnostic (operate on plain `(actual, predicted[, proba])`
columns) and must stay cpu-flavor-clean and weight-free.

### Classification Metrics

- [x] **CMET-01**: User can compute classification accuracy over `(actual, predicted)` columns via a SQL aggregate
- [x] **CMET-02**: User can compute precision, recall, and F1 with an explicit averaging mode (micro / macro / weighted) — no silent default
- [x] **CMET-03**: User can compute log-loss (cross-entropy) from a per-class probability MAP, with probability clipping to avoid infinities
- [x] **CMET-04**: User can compute ROC-AUC with an explicit multiclass averaging mode and correct tie handling (rank-sum form)
- [x] **CMET-05**: User can compute a confusion matrix (as a table-valued result) over `(actual, predicted)`
- [x] **CMET-06**: User can compute Expected Calibration Error (ECE) from the per-class probability MAP

### Regression Metrics

- [x] **RMET-01**: User can compute RMSE over `(actual, predicted)` via a SQL aggregate
- [x] **RMET-02**: User can compute MAE over `(actual, predicted)`
- [x] **RMET-03**: User can compute R² over `(actual, predicted)`, handling the constant-target edge case
- [x] **RMET-04**: User can compute MAPE and median absolute error, handling zero actuals

### Cross-Validation

- [x] **CV-01**: User can assign rows to k deterministic folds via a documented, seedable rule
- [x] **CV-02**: User can run k-fold cross-validation via a SQL macro that trains on each fold's context and predicts its held-out rows using the leakage-safe two-table predict form
- [x] **CV-03**: CV returns per-fold and aggregate (mean ± std) metric results
- [x] **CV-04**: The macro quotes the target identifier safely (fixes the P1 interpolation bug at `tabfm_macros.cpp:91`)

### Model Generalization

- [x] **MGEN-01**: A preprocessing-profile registry maps a manifest `preprocessing_profile` string to a C++ preprocessing function; the existing `tabfm-v1` profile self-registers
- [x] **MGEN-02**: Model loading dispatches preprocessing through the registry (rejecting unknown profiles with a named, actionable error)
- [x] **MGEN-03**: Model-output validation (shape/rank/class-count) runs before decode for every family, extending the existing P0 gap

### Regression Distribution Output

- [x] **RDIST-01**: The ORT run output can carry a regression predictive distribution (per-bin logits + bin borders), empty when a model does not emit one
- [x] **RDIST-02**: `output_mode='distribution'` emits a predictive distribution and quantiles for regression, backward-compatible with point-estimate output

### Model Onboarding

- [ ] **MODL-01** *(fixture-scoped — Phase 2)*: TabPFN v2 is available as a first-class model family (manifest + `tabpfn_v2` preprocessing profile + committed weight-free random-init ONNX fixture whose outputs match the confirmed `[n,K]` logits + `[K+1]` non-uniform borders contract), including its bar-distribution regression output. *Real TabPFN v2 ONNX inference export is deferred (upstream-blocked: data-dependent preprocessing + chunked attention — see `.planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md`).*
- [ ] ~~**MODL-02**: TabICL as a first-class classification family~~ **→ DEFERRED to v2** (ONNX export infeasible on `tabicl 2.2.0`: data-dependent Stage-1 branches; needs upstream PRs — see `.planning/spikes/SPIKE-tabicl-onnx-export.md`)
- [ ] **MODL-03**: Each non-tabfm-v1 family has its own license-acceptance gate (separate from the existing HF-license flag), enforced before download
- [ ] **MODL-04** *(fixture-scoped — Phase 2)*: `tools/parity` validates the TabPFN v2 fixture family's ONNX output contract (both distribution tensors) before its C++ decoder is trusted. *TabICL parity deferred with MODL-02.*

### Proper Scoring Rules

- [ ] **PSR-01**: User can compute CRPS over a regression predictive distribution via a SQL aggregate (energy-score identity for bar distributions)
- [ ] **PSR-02**: User can compute log-score (NLL) over a regression predictive distribution
- [ ] **PSR-03**: User can compute interval score at a configurable coverage level
- [ ] **PSR-04**: Proper-scoring-rule functions are bind-gated on distribution input and fail with a named remedy when given point estimates

### Model Comparison

- [ ] **CMP-01**: User can run the evaluation metrics across multiple model families on their own tables to compare model quality (no bundled datasets)

## v2 Requirements

Deferred to a future milestone. Tracked but not in this roadmap.

### Advanced Scoring

- **ASCR-01**: CRLS (log-score with singularity handling) and multivariate scores (energy/variogram)
- **ASCR-02**: Calibration curves / reliability diagrams as data output

### Model Families

- **MFAM-01**: TabPFN v3 (post-cutoff; tensor contract unknown)
- **MFAM-02**: Regression predictive distributions for TabICL (if/when upstream supports it)
- **MODL-02** *(deferred from v1, 2026-09-21)*: TabICL as a first-class classification family — ONNX export infeasible on `tabicl 2.2.0` (data-dependent Stage-1 branches: `SkippableLinear` if-branch + `num_classes = y_train.max()`). Needs 2–3 upstream PRs to `soda-inria/tabicl`; re-spike after. (`.planning/spikes/SPIKE-tabicl-onnx-export.md`)
- **MODL-01-EXPORT** *(deferred from v1, 2026-09-21)*: Real TabPFN v2 ONNX *inference* export — blocked on data-dependent preprocessing (`_remove_constant_features`, `_impute_nan_and_inf_with_mean`) + chunked attention failing `torch.export`. Phase 2 delivers the fixture-scoped MODL-01 (weight-free graph matching the confirmed contract); real-model inference needs a cleaned export path. (`.planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md`)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Bundled / auto-downloaded benchmark dataset suite | Comparison runs on the user's own tables; avoids dataset-licensing burden and keeps the extension lean |
| `autorank` / critical-difference diagrams in-engine | Statistical ranking is a downstream/notebook concern, not SQL |
| Post-hoc calibration fitting (Platt/isotonic) | Measuring calibration (ECE) is in scope; correcting it is not |
| Training / fine-tuning models | These are in-context (zero-shot) models; training is not the product |
| Google/vendor weight bytes in the repo | License wall (S06); fixtures stay random-init, graphs weight-free |
| GPU-only code paths in the cpu flavor | Community-extension eligibility (PR #2181) |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CMET-01 | Phase 1 | Complete |
| CMET-02 | Phase 1 | Complete |
| CMET-03 | Phase 1 | Complete |
| CMET-04 | Phase 1 | Complete |
| CMET-05 | Phase 1 | Complete |
| CMET-06 | Phase 1 | Complete |
| RMET-01 | Phase 1 | Complete |
| RMET-02 | Phase 1 | Complete |
| RMET-03 | Phase 1 | Complete |
| RMET-04 | Phase 1 | Complete |
| CV-01 | Phase 1 | Complete |
| CV-02 | Phase 1 | Complete |
| CV-03 | Phase 1 | Complete |
| CV-04 | Phase 1 | Complete |
| MGEN-01 | Phase 2 | Complete |
| MGEN-02 | Phase 2 | Complete |
| MGEN-03 | Phase 2 | Complete |
| RDIST-01 | Phase 2 | Complete |
| RDIST-02 | Phase 2 | Complete |
| MODL-01 | Phase 2 | Pending (fixture-scoped; real export → v2) |
| MODL-02 | v2 | Deferred (TabICL export infeasible) |
| MODL-03 | Phase 2 | Pending |
| MODL-04 | Phase 2 | Pending (fixture parity; TabICL parity → v2) |
| PSR-01 | Phase 3 | Pending |
| PSR-02 | Phase 3 | Pending |
| PSR-03 | Phase 3 | Pending |
| PSR-04 | Phase 3 | Pending |
| CMP-01 | Phase 3 | Pending |

**Coverage:**

- v1 requirements: 28 enumerated; 1 deferred to v2 post-spike (MODL-02, TabICL export infeasible), 1 export sub-requirement split off (MODL-01-EXPORT → v2)
- Active v1 mapped to phases: 27 (Phase 1: 14 ✓ complete, Phase 2: 8 [MGEN-01/02/03, RDIST-01/02, MODL-01 fixture, MODL-03, MODL-04 fixture], Phase 3: 5)
- Deferred (upstream-blocked, 2026-09-21): MODL-02 (TabICL), real TabPFN v2 inference export
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-19*
*Last updated: 2026-09-19 after initial definition*
