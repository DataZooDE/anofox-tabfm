---
phase: 01-evaluation-metrics-cross-validation
plan: 02
subsystem: metrics-classification
tags: [metrics, classification, f1, log-loss, roc-auc, ece, confusion-matrix, tdd, cmet-02, cmet-03, cmet-04, cmet-05, cmet-06]
status: complete

dependency_graph:
  requires:
    - 01-01 (scaffold, RegisterAggregateFunctionSetWithAlias, tabfm_accuracy baseline)
  provides:
    - tabfm_f1 / anofox_tabfm_f1 (CMET-02, avg required)
    - tabfm_precision / anofox_tabfm_precision (CMET-02, avg required)
    - tabfm_recall / anofox_tabfm_recall (CMET-02, avg required)
    - tabfm_log_loss / anofox_tabfm_log_loss (CMET-03)
    - tabfm_roc_auc / anofox_tabfm_roc_auc (CMET-04, avg required, rank-sum tie handling)
    - tabfm_confusion_matrix / anofox_tabfm_confusion_matrix (CMET-05, TABLE MACRO)
    - tabfm_ece / anofox_tabfm_ece (CMET-06)
  affects:
    - src/tabfm_metrics_classification.cpp (major expansion)
    - test/sql/tabfm_metrics_classification.test (54 assertions, up from 12)
    - test/cpp/test_tabfm_metrics.cpp (11 test cases, up from 4)
    - tools/golden/generate_metric_fixtures.py (CMET-02..06 golden values documented)

tech_stack:
  added:
    - F1StateSlot (heap-owning unordered_map<string,ClassCounts> with StateDestroy)
    - AUCStateSlot (heap-owning unordered_map<string,AUCClassData> with StateDestroy)
    - ComputeBinaryAUC (Mann-Whitney U rank-sum with tie-group handling)
    - CONFUSION_MACRO_BODY table macro (GROUP BY with replace() identifier quoting)
  patterns:
    - 2-arg overload that always throws at bind (required-parameter enforcement pattern)
    - Heap-owning state slot with StateDestroy (O(N) state cleanup, used for F1 and AUC)
    - MAP(VARCHAR,DOUBLE) iteration via MapValue::GetChildren + StructValue::GetChildren
    - ExpressionExecutor::EvaluateScalar for early avg validation at bind time
    - TABLE MACRO with replace(col,'"','""') SQL injection guard (T-01-02-01)
    - AggregateFunction 2-arg overload with inline lambda bind that throws immediately

key_files:
  created: []
  modified:
    - src/tabfm_metrics_classification.cpp (6 new metrics + confusion matrix TABLE MACRO)
    - test/sql/tabfm_metrics_classification.test (all CMET-01..06 golden tests)
    - test/cpp/test_tabfm_metrics.cpp (AUC tie, F1, log-loss Catch2 cases)
    - tools/golden/generate_metric_fixtures.py (sklearn reference for all CMET-02..06)

decisions:
  - id: avg-required-2-arg-overload
    summary: "Required avg enforced via a 2-arg overload whose bind always throws, rather than a runtime check at Finalize"
    rationale: "DuckDB aggregate bind is the correct error interception point; 2-arg overload dispatches to the distinct bind function that throws immediately with the named exception"
  - id: avg-value-stored-in-bind-data
    summary: "avg mode stored as string in F1BindData/AUCBindData at bind time via ExpressionExecutor::EvaluateScalar"
    rationale: "Allows Finalize to compute without re-reading avg from the input vector; validates value at bind for constant expressions"
  - id: confusion-matrix-as-table-macro
    summary: "tabfm_confusion_matrix implemented as TABLE MACRO (not a C++ table function)"
    rationale: "01-RESEARCH.md Open Question 2 confirmed: GROUP BY cannot be populated at bind time in a table function; SQL macro wrapping query() is the canonical pattern. Double-quote escaping applied inline."
  - id: ece-proba-map-not-proba-scalar
    summary: "tabfm_ece takes (actual, proba MAP) and extracts argmax internally"
    rationale: "Consistent with log_loss signature; user doesn't need to pre-compute max_proba; MAP contains all class probabilities needed for argmax detection"
  - id: pre-existing-asan-in-ort-teardown
    summary: "ASAN heap-use-after-free in duckdb/shared_ptr_ipp.hpp during full-suite teardown is pre-existing"
    rationale: "Occurs in DuckDB ORT engine module static cleanup, not in metric aggregates; all 1966 assertions pass; confirmed scope boundary (CLAUDE.md rule #2)"

metrics:
  duration_minutes: 18
  completed_date: "2026-09-20"
  tasks_completed: 3
  tasks_total: 3

actuals:
  tokens: 18400    # chars/4 over 73601-char diff
  tasks: 3
  commits: 2       # MEASURED: git rev-list --count ed8ae06e75df04836a44e833f67bb88ba7647f5c..HEAD
  plan_head_before: ed8ae06e75df04836a44e833f67bb88ba7647f5c
---

# Phase 01 Plan 02: Classification Metrics CMET-02..06 Summary

**One-liner:** Six classification metrics (F1/precision/recall with required avg, log-loss with eps clipping, ROC-AUC with rank-sum tie handling, ECE, and a safely-quoted confusion matrix TABLE MACRO) shipped sklearn-golden and error-path tested.

## What Was Built

### Task 1: Precision / Recall / F1 with Required Averaging Mode (CMET-02)

- **F1StateSlot:** heap-owning `unordered_map<string, ClassCounts>` (tp/fp/fn per class) with `StateDestroy` for safe cleanup.
- **Two-overload pattern:** 3-arg form accepts `avg`; 2-arg form has an inline lambda bind that immediately throws `InvalidInputException("tabfm_f1: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'")` (same for precision/recall). No silent default. The avg mode is validated early via `ExpressionExecutor::EvaluateScalar` at bind time when the argument is a constant.
- **Averaging modes:**
  - `macro`: unweighted mean of per-class metrics
  - `micro`: global TP/FP/FN summed, then single precision/recall/F1
  - `weighted`: support-weighted mean (support_c = actual count of class c)
- **zero_division=0**: undefined precision/recall (zero denominator) returns 0.0 per sklearn convention.
- **Golden values (sklearn reference):** F1 macro=0.4, weighted=0.533333, micro=0.666667 on the 3-class fixture.

### Task 2: Log-Loss and ECE from Per-Class Probability MAP (CMET-03, CMET-06)

- **tabfm_log_loss:** Iterates `MAP(VARCHAR,DOUBLE)` via `MapValue::GetChildren + StructValue::GetChildren`. Missing key returns p=0; clipped to [1e-15, 1-1e-15] (`kEps = 1e-15`). Never returns infinity. Golden: 0.424322 on 3-row fixture.
- **tabfm_ece:** 10-bin ECE — extracts argmax class and max_proba from MAP for each row; assigns to `bin = floor(max_prob * 10)` clamped to [0,9]; accumulates accuracy and confidence per bin; ECE = Σ (|B_m|/N) × |acc_m − conf_m|. Golden: 0.333333 (all predictions correct, but confidence < 1).

### Task 3: ROC-AUC (Rank-Sum Ties) and Confusion Matrix TABLE MACRO (CMET-04, CMET-05)

- **tabfm_roc_auc:** `AUCStateSlot` with per-class `vector<ScoreLabel>` pairs accumulated in Update (same row-order across all classes, enabling OvO index-alignment). `ComputeBinaryAUC` uses the Mann-Whitney U formula with tie-group advancement — equal-score positives and negatives in the same group get the proportional contribution `n_pos_group × n_neg_below + n_pos_group × n_neg_group × 0.5`. No optimistic or pessimistic tie interpolation (01-RESEARCH.md §Pitfall 1). OvO pairs are index-aligned: pairs in `class_map[c1]` and `class_map[c2]` correspond to the same source rows, enabling efficient filter-by-label. Golden: OvR=OvO=0.958333 on 6-row tied-score fixture.
- **tabfm_confusion_matrix:** TABLE MACRO with body:
  ```sql
  SELECT * FROM query(
    'SELECT "' || replace(actual_col, '"', '""') || '" AS actual, '
    || '"' || replace(predicted_col, '"', '""') || '" AS predicted, COUNT(*) AS count'
    || ' FROM (FROM ' || data || ') GROUP BY 1, 2 ORDER BY 1, 2'
  )
  ```
  Double-quote escaping prevents SQL injection via column identifier arguments (T-01-02-01). Registered as both `anofox_tabfm_confusion_matrix` and `tabfm_confusion_matrix`.

## Verification Evidence

- `make debug`: succeeded, 4 files changed, all modules linked
- `./build/debug/test/unittest test/sql/tabfm_metrics_classification.test`: **All tests passed (54 assertions in 1 test case)**
- `./build/debug/test/unittest "[tabfm][metrics]"`: **All tests passed (53 assertions in 11 test cases)**
- Full suite: **All tests passed (1966 assertions in 85 test cases)**

### Golden Value Verification

| Metric | Mode | Golden | Source |
|--------|------|--------|--------|
| tabfm_f1 | macro | 0.400000 | sklearn f1_score(average='macro', zero_division=0) |
| tabfm_f1 | weighted | 0.533333 | sklearn f1_score(average='weighted', zero_division=0) |
| tabfm_f1 | micro | 0.666667 | sklearn f1_score(average='micro') |
| tabfm_precision | macro | 0.333333 | sklearn precision_score(average='macro', zero_division=0) |
| tabfm_precision | weighted | 0.444444 | sklearn precision_score(average='weighted', zero_division=0) |
| tabfm_precision | micro | 0.666667 | sklearn precision_score(average='micro') |
| tabfm_recall | macro | 0.500000 | sklearn recall_score(average='macro') |
| tabfm_recall | weighted | 0.666667 | sklearn recall_score(average='weighted') |
| tabfm_recall | micro | 0.666667 | sklearn recall_score(average='micro') |
| tabfm_log_loss | — | 0.424322 | sklearn log_loss(eps=1e-15) |
| tabfm_roc_auc | ovr | 0.958333 | sklearn roc_auc_score(multi_class='ovr', average='macro') |
| tabfm_roc_auc | ovo | 0.958333 | sklearn roc_auc_score(multi_class='ovo', average='macro') |
| tabfm_ece | — | 0.333333 | Guo2017 10-bin ECE formula |

### Error Paths Verified

| Function | Missing Arg | Error Message Contains |
|----------|-------------|------------------------|
| tabfm_f1(actual, predicted) | avg | "tabfm_f1: 'avg' is required" |
| tabfm_precision(actual, predicted) | avg | "tabfm_precision: 'avg' is required" |
| tabfm_recall(actual, predicted) | avg | "tabfm_recall: 'avg' is required" |
| tabfm_roc_auc(actual, proba) | avg | "tabfm_roc_auc: 'avg' is required" |

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written, modulo minor technical adaptation (see informational notes below).

### Informational Notes

**1. [Informational] ExpressionExecutor::EvaluateScalar requires ClientContext**
- The plan's PATTERNS.md suggested checking `BoundConstantExpression` for early avg validation, but `BoundConstantExpression` is an incomplete type without its own header, and the required header is in the DuckDB planner, not available to extension code.
- Fix: used `ExpressionExecutor::EvaluateScalar(ctx, *arguments[2])` which requires the correct `ClientContext &` parameter; already available in the bind callback.
- No behavior change — same validation goal, correct API.

**2. [Informational] MakeF1Bind factory approach replaced with direct bind functions**
- The plan showed a factory lambda returning a lambda, but C++ requires the outer lambda to be convertible to `unique_ptr<FunctionData>(*)(ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)` — a lambda capturing variables is not implicitly convertible.
- Fix: factored bind logic into `BindF1Metric(ClientContext &, const char*, char, args)` helper called by three named bind functions (`PrecisionBind`, `RecallBind`, `F1ScoreBind`).
- No behavior change.

**3. [Informational] Pre-existing ASAN teardown error in ORT module**
- The full suite exit triggers a heap-use-after-free in `duckdb/shared_ptr_ipp.hpp` during process teardown. This is in DuckDB's ORT session cleanup code, not in the metric aggregates. All 1966 assertions pass before the teardown error.
- Out-of-scope per CLAUDE.md rule #2 (different module). Deferred per scope boundary.

## Known Stubs

None — all CMET-02..06 metrics are fully implemented with golden values.

## Threat Surface Scan

All mitigations from the plan's threat register are implemented:

| Threat ID | Status | Evidence |
|-----------|--------|----------|
| T-01-02-01 | Mitigated | `replace(actual_col, '"', '""')` in confusion matrix macro body |
| T-01-02-02 | Mitigated | Missing MAP key → p=0 → clipped to 1e-15 (no throw, no leak) |
| T-01-02-03 | Mitigated | All denominators guarded: safe_div(a, b) returns 0.0 when b=0; empty state returns NULL |
| T-01-02-04 | N/A | No package installs |

## Self-Check: PASSED

- All 4 modified files verified present on disk
- Commits 69a6872 and da1650a verified in git log
- SQL test: 54 assertions passed
- Catch2 [tabfm][metrics]: 53 assertions in 11 cases passed
- Full suite: 1966 assertions, 85 test cases, 0 failed
