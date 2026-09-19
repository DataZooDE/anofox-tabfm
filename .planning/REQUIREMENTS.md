# Requirements: anofox-tabfm — Evaluation Framework + Multi-Model Support

**Defined:** 2026-09-19
**Core Value:** Users can trust and compare tabular-foundation-model predictions directly in SQL — computing standard metrics and cross-validation on their own data, across more than one model family — without leaving DuckDB.

## v1 Requirements

Requirements for this milestone. Each maps to a roadmap phase. All metric
functions are model-agnostic (operate on plain `(actual, predicted[, proba])`
columns) and must stay cpu-flavor-clean and weight-free.

### Classification Metrics

- [ ] **CMET-01**: User can compute classification accuracy over `(actual, predicted)` columns via a SQL aggregate
- [ ] **CMET-02**: User can compute precision, recall, and F1 with an explicit averaging mode (micro / macro / weighted) — no silent default
- [ ] **CMET-03**: User can compute log-loss (cross-entropy) from a per-class probability MAP, with probability clipping to avoid infinities
- [ ] **CMET-04**: User can compute ROC-AUC with an explicit multiclass averaging mode and correct tie handling (rank-sum form)
- [ ] **CMET-05**: User can compute a confusion matrix (as a table-valued result) over `(actual, predicted)`
- [ ] **CMET-06**: User can compute Expected Calibration Error (ECE) from the per-class probability MAP

### Regression Metrics

- [ ] **RMET-01**: User can compute RMSE over `(actual, predicted)` via a SQL aggregate
- [ ] **RMET-02**: User can compute MAE over `(actual, predicted)`
- [ ] **RMET-03**: User can compute R² over `(actual, predicted)`, handling the constant-target edge case
- [ ] **RMET-04**: User can compute MAPE and median absolute error, handling zero actuals

### Cross-Validation

- [ ] **CV-01**: User can assign rows to k deterministic folds via a documented, seedable rule
- [ ] **CV-02**: User can run k-fold cross-validation via a SQL macro that trains on each fold's context and predicts its held-out rows using the leakage-safe two-table predict form
- [ ] **CV-03**: CV returns per-fold and aggregate (mean ± std) metric results
- [ ] **CV-04**: The macro quotes the target identifier safely (fixes the P1 interpolation bug at `tabfm_macros.cpp:91`)

### Model Generalization

- [ ] **MGEN-01**: A preprocessing-profile registry maps a manifest `preprocessing_profile` string to a C++ preprocessing function; the existing `tabfm-v1` profile self-registers
- [ ] **MGEN-02**: Model loading dispatches preprocessing through the registry (rejecting unknown profiles with a named, actionable error)
- [ ] **MGEN-03**: Model-output validation (shape/rank/class-count) runs before decode for every family, extending the existing P0 gap

### Regression Distribution Output

- [ ] **RDIST-01**: The ORT run output can carry a regression predictive distribution (per-bin logits + bin borders), empty when a model does not emit one
- [ ] **RDIST-02**: `output_mode='distribution'` emits a predictive distribution and quantiles for regression, backward-compatible with point-estimate output

### Model Onboarding

- [ ] **MODL-01**: TabPFN v2 is available as a first-class model family (manifest + `tabpfn_v2` preprocessing profile + weight-free ONNX fixture), including its bar-distribution regression output
- [ ] **MODL-02**: TabICL is available as a first-class classification model family (manifest + `tabicl_v2` preprocessing profile + weight-free ONNX fixture)
- [ ] **MODL-03**: Each non-tabfm-v1 family has its own license-acceptance gate (separate from the existing HF-license flag), enforced before download
- [ ] **MODL-04**: `tools/parity` validates each new family's ONNX output contract (including both distribution tensors for TabPFN v2) before its C++ decoder is trusted

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
| (populated by roadmapper) | — | Pending |

**Coverage:**
- v1 requirements: 26 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 26 ⚠️

---
*Requirements defined: 2026-09-19*
*Last updated: 2026-09-19 after initial definition*
