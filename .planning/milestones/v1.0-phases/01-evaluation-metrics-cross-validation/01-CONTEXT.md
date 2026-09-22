# Phase 1: Evaluation Metrics + Cross-Validation - Context

**Gathered:** 2026-09-20
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous)

<domain>
## Phase Boundary

Users can measure classification and regression prediction quality — and run
leakage-safe k-fold cross-validation — directly in SQL over plain
`(actual, predicted[, proba])` columns, independent of any model. All metric
code is model-agnostic, cpu-flavor-clean, and weight-free. Delivers CMET-01..06,
RMET-01..04, CV-01..04.

Out of this phase: proper scoring rules (Phase 3, gated on distributions),
multi-model support (Phase 2), post-hoc calibration fitting.

</domain>

<decisions>
## Implementation Decisions

### Module & Function Surface
- **File split:** three modules — `src/tabfm_metrics_classification.cpp`
  (accuracy, precision/recall/F1, log-loss, ROC-AUC, ECE, confusion matrix),
  `src/tabfm_metrics_regression.cpp` (RMSE, MAE, R², MAPE, median abs error),
  and `src/tabfm_crossval.cpp` (k-fold CV macro + fold-assignment helper).
  Honors the one-module-per-file rule and enables parallel plan execution.
- **Naming:** every metric gets a full `anofox_tabfm_*` name plus a short
  `tabfm_*` alias via `anofox_function_alias.hpp` helpers (CLAUDE.md rule #4).
- **Averaging mode:** exposed as a named parameter `avg := 'macro'` and
  **required** (no silent default) for the multiclass metrics
  precision/recall/F1 (`tabfm_f1`) and ROC-AUC (`tabfm_roc_auc`); omitting it
  throws a named `InvalidInputException` telling the user to pass
  `avg := 'micro'|'macro'|'weighted'` (AUC: `'ovr'|'ovo'`).
- **Confusion matrix:** table function returning tidy long form
  `(actual, predicted, count)` — DuckDB-idiomatic; users pivot downstream.

### Numerical Semantics & Edge Cases
- **NULL handling:** metric aggregates skip rows where any required input
  (actual, predicted, or proba) is NULL, matching standard SQL aggregate
  semantics. Documented in each function's description.
- **log-loss clipping:** clip probabilities to `[ε, 1−ε]` with `ε = 1e-15`
  (scikit-learn default) to avoid infinities.
- **R² constant target:** match scikit-learn — when the target variance is 0,
  return 1.0 for a perfect prediction and 0.0 otherwise; document the rule.
- **MAPE / median absolute error zero actuals:** MAPE skips rows where
  `actual = 0` (documented, avoids division by zero); median absolute error is
  unaffected. Behavior must be non-crashing and documented per success
  criterion 4.

### Cross-Validation Design
- **Fold assignment (CV-01):** deterministic `hash(row_key, seed) % k` over a
  user-supplied row-key expression — seedable, order-independent, leakage-safe,
  and documented. Not `row_number()`-based (order-dependent).
- **Stratification:** none in v1 — uniform hash-based folds; stratified folds
  deferred to a future milestone.
- **Metric selection (CV-02/03):** the `tabfm_cross_validate` macro accepts a
  metric name (or list) argument and dispatches to the Phase-1 metric
  aggregates; a task-inferred default (accuracy for classification, RMSE for
  regression) applies when none is given.
- **Output shape (CV-03):** one result — per-fold rows plus aggregate
  (mean ± std) row(s).
- **Leakage-safe predict (CV-02):** CV trains each fold's context and predicts
  its held-out rows via the existing two-table (leakage-safe) predict form.
- **Identifier quoting (CV-04):** the macro safely quotes the target identifier,
  fixing the P1 interpolation bug at `tabfm_macros.cpp:91` (double-quote,
  escape embedded quotes).

### Claude's Discretion
- Aggregate state layout, intermediate accumulator types, and exact sklearn
  parity math (rank-sum AUC, micro/macro/weighted formulas) are at Claude's
  discretion, grounded in scikit-learn reference behavior.
- Golden fixtures verified against scikit-learn outputs; test structure and
  tolerance thresholds at Claude's discretion (TDD red-green mandated).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Aggregate registration pattern:** `src/tabfm_predict_agg.cpp` shows the
  DuckDB `AggregateFunction` triplet (bind/update/combine/finalize),
  `CreateAggregateFunctionInfo`, and telemetry-at-bind convention.
- **Alias helpers:** `src/include/anofox_function_alias.hpp` — register full
  `anofox_tabfm_*` + short `tabfm_*` names as full metadata copies.
- **Table macros:** `src/tabfm_macros.cpp` — pattern for the CV macro (SQL body,
  `query()` composition, the two-table leakage-safe predict form). Note the
  identifier-quoting bug at line 91 that CV-04 must fix.
- **Registration entry:** `src/include/tabfm_registration.hpp` and
  `src/anofox_tabfm_extension.cpp` are scaffold-owned — coordinate any edits to
  the source list / registration hooks (CLAUDE.md rule #2).

### Established Patterns
- Namespaces: `namespace duckdb { namespace anofox { ... } }`; file-local
  helpers in anonymous namespaces.
- Errors: DuckDB exceptions whose message names the fixing SET/CALL/param
  (SQL-API §5).
- Telemetry: `PostHogTelemetry::Instance().CaptureFunctionExecution("<short>")`
  once per execution at bind/global-init.
- Tests: sqllogictest `.test` files in `test/sql/`; Catch2 TUs listed in
  `TABFM_CPP_TEST_SOURCES` in CMakeLists.txt.

### Integration Points
- New aggregates/macros register through the scaffold-owned registration path;
  add source files to the CMakeLists source list (coordinate per rule #2).
- CV macro composes the existing `tabfm_classify`/`tabfm_regress` predict
  surface and the new metric aggregates.

</code_context>

<specifics>
## Specific Ideas

- Function names are pinned by the ROADMAP success criteria: `tabfm_accuracy`,
  `tabfm_rmse`, `tabfm_f1(..., avg := 'macro')`,
  `tabfm_roc_auc(..., avg := 'ovr')`, `tabfm_log_loss`, `tabfm_ece`,
  `tabfm_confusion_matrix`, `tabfm_r2`, `tabfm_mape`, `tabfm_medae`,
  `tabfm_cross_validate(...)`.
- Golden fixtures must match scikit-learn (accuracy, RMSE, F1 macro on a 3-class
  fixture, ROC-AUC ovr with rank-sum tie handling).
- A CV leakage-detecting golden test is non-negotiable and must be written
  before CV ships (STATE.md Blockers, research §Gaps).

</specifics>

<deferred>
## Deferred Ideas

- Stratified CV folds — v2.
- Post-hoc calibration fitting (Platt/isotonic) — out of scope (measuring ECE is
  in scope; correcting calibration is not).
- Proper scoring rules (CRPS/log-score/interval) — Phase 3, gated on regression
  predictive distributions.

</deferred>
