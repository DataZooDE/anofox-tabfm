# Phase 1: Evaluation Metrics + Cross-Validation — Research

**Researched:** 2026-09-20
**Domain:** DuckDB C++ extension — model-agnostic metric aggregates + k-fold CV macro
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Module & Function Surface
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

#### Numerical Semantics & Edge Cases
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

#### Cross-Validation Design
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

### Deferred Ideas (OUT OF SCOPE)
- Stratified CV folds — v2.
- Post-hoc calibration fitting (Platt/isotonic) — out of scope (measuring ECE is
  in scope; correcting calibration is not).
- Proper scoring rules (CRPS/log-score/interval) — Phase 3, gated on regression
  predictive distributions.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CMET-01 | User can compute classification accuracy over `(actual, predicted)` columns via a SQL aggregate | Scalar aggregate pattern confirmed from `tabfm_predict_agg.cpp`; state: `{int64 correct, int64 total}` |
| CMET-02 | User can compute precision, recall, and F1 with an explicit averaging mode (micro / macro / weighted) — no silent default | Per-class TP/FP/FN state; averaging modes documented in scikit-learn; explicit `avg` parameter with `InvalidInputException` on omission |
| CMET-03 | User can compute log-loss from a per-class probability MAP, with probability clipping | `MAP(VARCHAR, DOUBLE)` access via `MapValue::GetChildren`; clip to `[1e-15, 1-1e-15]` |
| CMET-04 | User can compute ROC-AUC with an explicit multiclass averaging mode and correct tie handling | Rank-sum / trapezoidal with tie-grouping; OvR and OvO modes; O(N) state |
| CMET-05 | User can compute a confusion matrix as a table-valued result | `TableFunction` pattern confirmed from `tabfm_devices.cpp`; output `(actual, predicted, count)` tidy long form |
| CMET-06 | User can compute ECE from the per-class probability MAP | 10-bin equal-width state; `{bin_n[10], bin_correct[10], bin_conf_sum[10]}` |
| RMET-01 | User can compute RMSE over `(actual, predicted)` via a SQL aggregate | State: `{double sum_sq, int64 n}`; standard |
| RMET-02 | User can compute MAE over `(actual, predicted)` | State: `{double sum_abs, int64 n}` |
| RMET-03 | User can compute R² over `(actual, predicted)`, handling the constant-target edge case | Welford-like online state for SS_tot/SS_res; constant-target returns 1.0 or 0.0 |
| RMET-04 | User can compute MAPE and median absolute error, handling zero actuals | MAPE skips zero-actual rows; MedAE stores `vector<double>` residuals, sorts at finalize |
| CV-01 | User can assign rows to k deterministic folds via a documented, seedable rule | `hash(row_key, seed) % k`; DuckDB `hash()` is variadic ANY→UBIGINT, stable within a version |
| CV-02 | User can run k-fold CV via a SQL macro using the leakage-safe two-table predict form | Table macro composes `tabfm_classify`/`tabfm_regress` two-table form per fold |
| CV-03 | CV returns per-fold and aggregate (mean ± std) metric results | UNION ALL of per-fold queries + aggregate row via SQL |
| CV-04 | The macro quotes the target identifier safely | Same `replace(target, '"', '""')` double-quote wrapping used in `tabfm_macros.cpp:93` |
</phase_requirements>

---

## Summary

Phase 1 adds three new source modules to a mature DuckDB C++ extension: classification metrics, regression metrics, and a cross-validation macro. All code is model-agnostic — it operates on plain `(actual, predicted[, proba])` SQL columns and requires zero changes to the existing predict engine, ONNX runtime, or safetensors stack.

The aggregate function pattern is fully established in `src/tabfm_predict_agg.cpp`: state-size callback, initialize, update, combine, finalize, optional state-destroy. That file is the canonical template. Each new metric module registers its functions through the same `AggregateFunctionSet` + `CreateAggregateFunctionInfo` path and uses `anofox_function_alias.hpp` helpers for the `tabfm_*` short aliases. The confusion matrix and the fold-assignment helper follow the `TableFunction` pattern established in `src/tabfm_devices.cpp`. The cross-validation macro follows the `TableMacroFunction` pattern in `src/tabfm_macros.cpp`.

The three scaffold-owned shared files (`CMakeLists.txt`, `src/include/tabfm_registration.hpp`, `src/anofox_tabfm_extension.cpp`) must be edited once to add the three new source files and three new `Register*` declarations — coordinate this as a single batch at the start of the phase.

**Primary recommendation:** Implement the three modules in parallel (classification metrics, regression metrics, CV macro). Begin each module with a failing sqllogictest file whose golden values come from a scikit-learn fixture script run offline. The CV leakage test must be written before CV ships.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Classification metric aggregates | API/Backend (C++ aggregate) | — | Runs inside DuckDB executor; consumes plain SQL columns |
| Regression metric aggregates | API/Backend (C++ aggregate) | — | Same as classification |
| Confusion matrix | API/Backend (C++ table function) | — | Produces rows; no client-side render needed |
| k-fold fold assignment helper | SQL macro | — | Composes `hash()` + arithmetic; no custom C++ needed |
| CV orchestration macro | SQL macro (table macro) | API/Backend (composes existing aggregates) | Loops over folds via UNION ALL; calls metric aggregates and predict macros |
| Proba MAP parsing (log-loss, AUC, ECE) | API/Backend (C++ aggregate Update) | — | `MapValue::GetChildren` in Update callback |
| Identifier quoting | SQL macro (string expression) | — | `replace(target, '"', '""')` at SQL-string-build time |

---

## Standard Stack

No new external packages are required for this phase. All code uses:

- DuckDB 1.5.4 C++ API (already in the extension)
- C++ standard library (`<algorithm>`, `<cmath>`, `<vector>`, `<unordered_map>`)

### Core DuckDB APIs Used

| API | Header | Purpose |
|-----|--------|---------|
| `AggregateFunction` | `duckdb/function/aggregate_function.hpp` | Metric aggregates (bind/update/combine/finalize) |
| `AggregateFunctionSet` | same | Multiple overloads per metric name |
| `CreateAggregateFunctionInfo` | `duckdb/parser/parsed_data/create_aggregate_function_info.hpp` | Registration |
| `TableFunction` | `duckdb/function/table_function.hpp` | Confusion matrix, fold assignment |
| `TableMacroFunction` | `duckdb/function/table_macro_function.hpp` | CV macro body |
| `MapValue::GetChildren` | `duckdb/common/types/value.hpp` | Iterate `MAP(VARCHAR, DOUBLE)` proba values in Update |
| `StructValue::GetChildren` | same | Extract key/value from each MAP entry |
| `Value::MAP(...)` | same | Construct MAP output |
| `ListVector::PushBack` | `duckdb/common/vector_operations/...` | Build LIST results |
| `PostHogTelemetry::Instance().CaptureFunctionExecution` | `telemetry.hpp` | Once per bind (rule #3) |

### Alias Helpers

| Helper | Header | Use |
|--------|--------|-----|
| `RegisterScalarFunctionWithAlias` | `anofox_function_alias.hpp` | Register scalar metrics (if any) |
| `RegisterScalarFunctionSetWithAlias` | same | Multi-overload scalar metrics |
| `RegisterTableFunctionWithAlias` | same | Confusion matrix, fold helper |

**No `RegisterAggregateFunctionWithAlias` exists yet.** The aggregate path registers via `CreateAggregateFunctionInfo` directly with `alias_of` set on the second registration, following the pattern in `tabfm_predict_agg.cpp:719-724`. A helper for aggregates should be added to `anofox_function_alias.hpp` for consistency — coordinate as part of the scaffold edits.

---

## Package Legitimacy Audit

No external packages are installed in this phase. The entire implementation uses the DuckDB C++ API already present in the extension.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Architecture Patterns

### System Architecture Diagram

```
SQL Query
  │
  ▼
DuckDB Executor
  │
  ├─► tabfm_accuracy(actual, predicted)        ─► AccuracyAgg (Update/Combine/Finalize)
  │                                                State: {int64 correct, int64 total}
  │
  ├─► tabfm_f1(actual, predicted, avg:='macro') ─► F1Agg
  │                                                State: unordered_map<string, {tp,fp,fn}>
  │
  ├─► tabfm_log_loss(actual, proba)            ─► LogLossAgg
  │      proba = MAP(VARCHAR, DOUBLE)              State: {double sum_ll, int64 n}
  │      │                                         Update: MapValue::GetChildren(proba)
  │      └─ MapValue::GetChildren → key/val pairs
  │
  ├─► tabfm_roc_auc(actual, score, avg:='ovr') ─► AUCAgg
  │                                                State: vector<{score, label}> (O(N))
  │                                                Finalize: sort + trapezoidal + tie-group
  │
  ├─► tabfm_ece(actual, proba)                 ─► ECEAgg
  │                                                State: bin_n[10], bin_correct[10], ...
  │
  ├─► tabfm_confusion_matrix(actual, predicted) ─► TableFunction
  │                                                Output: (actual, predicted, count)
  │
  ├─► tabfm_rmse(actual, predicted)            ─► RMSEAgg
  │                                                State: {double sum_sq, int64 n}
  │
  ├─► tabfm_r2(actual, predicted)              ─► R2Agg
  │                                                State: {sum_y, sum_y2, sum_res, n}
  │
  ├─► tabfm_medae(actual, predicted)           ─► MedAEAgg
  │                                                State: vector<double> residuals (O(N))
  │
  └─► tabfm_cross_validate(data, target, ...)  ─► TableMacroFunction
           │
           ├─ tabfm_fold_assign(data, k, row_key, seed)
           │       hash(row_key, seed) % k
           │
           └─ per-fold: tabfm_classify / tabfm_regress (two-table form)
                        → tabfm_accuracy / tabfm_rmse (metric aggregate)
                        → UNION ALL → mean/std aggregate row
```

### Recommended Project Structure

New files:

```
src/
├── tabfm_metrics_classification.cpp    # CMET-01..06
├── tabfm_metrics_regression.cpp        # RMET-01..04
├── tabfm_crossval.cpp                  # CV-01..04
├── include/
│   ├── tabfm_metrics_classification.hpp
│   ├── tabfm_metrics_regression.hpp
│   └── tabfm_crossval.hpp
test/
├── sql/
│   ├── tabfm_metrics_classification.test
│   ├── tabfm_metrics_regression.test
│   └── tabfm_crossval.test
tools/
└── golden/                             # Python script to generate golden fixtures via sklearn
    └── generate_metric_fixtures.py
```

Scaffold-owned files requiring coordinated edits (one batch, phase start):

```
CMakeLists.txt                          # Add 3 sources to EXTENSION_SOURCES + TABFM_CPP_TEST_SOURCES
src/include/tabfm_registration.hpp     # Add 3 Register* declarations
src/anofox_tabfm_extension.cpp         # Call 3 new Register* in Load()
src/include/anofox_function_alias.hpp  # Add RegisterAggregateFunctionWithAlias/Set helpers
```

### Pattern 1: Simple Aggregate (accuracy, RMSE, MAE)

```cpp
// Source: src/tabfm_predict_agg.cpp (existing pattern)
namespace {

struct AccuracyState {
    int64_t correct = 0;
    int64_t total = 0;
};

void AccuracyStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
    new (state_ptr) AccuracyState();
}

idx_t AccuracyStateSize(const AggregateFunction &) {
    return sizeof(AccuracyState);
}

void AccuracyUpdate(Vector inputs[], AggregateInputData &, idx_t input_count,
                    Vector &state_vector, idx_t count) {
    UnifiedVectorFormat actual_data, pred_data, sdata;
    inputs[0].ToUnifiedFormat(count, actual_data);
    inputs[1].ToUnifiedFormat(count, pred_data);
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<AccuracyState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        auto aidx = actual_data.sel->get_index(i);
        auto pidx = pred_data.sel->get_index(i);
        if (!actual_data.validity.RowIsValid(aidx) || !pred_data.validity.RowIsValid(pidx)) {
            continue; // NULL skip (CLAUDE.md — SQL aggregate semantics)
        }
        state.total++;
        // type-specific comparison: see CMET-01 note below
        if (inputs[0].GetValue(i) == inputs[1].GetValue(i)) {
            state.correct++;
        }
    }
}

void AccuracyCombine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata, tdata;
    source.ToUnifiedFormat(count, sdata);
    target.ToUnifiedFormat(count, tdata);
    auto sources = reinterpret_cast<AccuracyState **>(sdata.data);
    auto targets = reinterpret_cast<AccuracyState **>(tdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[sdata.sel->get_index(i)];
        auto &tgt = *targets[tdata.sel->get_index(i)];
        tgt.correct += src.correct;
        tgt.total   += src.total;
    }
}

void AccuracyFinalize(Vector &state_vector, AggregateInputData &, Vector &result,
                      idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<AccuracyState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        if (state.total == 0) {
            FlatVector::SetNull(result, i + offset, true);
        } else {
            FlatVector::GetData<double>(result)[i + offset] =
                static_cast<double>(state.correct) / static_cast<double>(state.total);
        }
    }
}

} // anonymous namespace

void RegisterClassificationMetrics(ExtensionLoader &loader) {
    // Register accuracy
    AggregateFunction accuracy_fn("anofox_tabfm_accuracy",
        {LogicalType::ANY, LogicalType::ANY}, LogicalType::DOUBLE,
        AccuracyStateSize, AccuracyStateInit, AccuracyUpdate, AccuracyCombine, AccuracyFinalize);
    AggregateFunctionSet accuracy_set("anofox_tabfm_accuracy");
    accuracy_set.AddFunction(accuracy_fn);
    CreateAggregateFunctionInfo accuracy_info(accuracy_set);
    loader.RegisterFunction(accuracy_info);
    // alias: tabfm_accuracy
    auto alias_fn = accuracy_fn;
    alias_fn.name = "tabfm_accuracy";
    AggregateFunctionSet alias_set("tabfm_accuracy");
    alias_set.AddFunction(alias_fn);
    CreateAggregateFunctionInfo alias_info(alias_set);
    alias_info.alias_of = "anofox_tabfm_accuracy";
    loader.RegisterFunction(alias_info);
    // ... other metrics
}
```

**Key detail for accuracy:** `inputs[0].GetValue(i) == inputs[1].GetValue(i)` works for any type because `Value::operator==` is defined. For performance on large datasets, a type-specialized comparison using `UnifiedVectorFormat` + typed data pointers is preferable.

### Pattern 2: MAP(VARCHAR, DOUBLE) Access in Update (log-loss, ECE)

```cpp
// Source: src/tabfm_predict_agg.cpp:205-207 (MapValue::GetChildren pattern)
// proba column is MAP(VARCHAR, DOUBLE): each Value is a list of {key, val} STRUCT children

void LogLossUpdate(Vector inputs[], AggregateInputData &, idx_t,
                   Vector &state_vector, idx_t count) {
    // inputs[0] = actual (VARCHAR), inputs[1] = proba (MAP(VARCHAR, DOUBLE))
    for (idx_t i = 0; i < count; i++) {
        Value proba_val = inputs[1].GetValue(i);
        if (proba_val.IsNull()) continue;
        Value actual_val = inputs[0].GetValue(i);
        if (actual_val.IsNull()) continue;

        string actual_str = actual_val.ToString();
        double p = 0.0;
        // MAP is a list of STRUCT({key VARCHAR, value DOUBLE}) children
        for (auto &kv : MapValue::GetChildren(proba_val)) {
            auto &entry = StructValue::GetChildren(kv);
            if (entry[0].ToString() == actual_str) {
                p = DoubleValue::Get(entry[1]);
                break;
            }
        }
        // Clip to [epsilon, 1-epsilon]
        static constexpr double kEps = 1e-15;
        p = std::max(kEps, std::min(1.0 - kEps, p));
        state.sum_ll += -std::log(p);
        state.n++;
    }
}
```

### Pattern 3: O(N) State for AUC and MedAE

```cpp
// AUC state: store (score, is_positive) pairs, sort at finalize
struct AUCState {
    vector<pair<double, int>> pairs; // (score, 1 if positive)
};

// Initialize via StateDestroy pattern (heap-allocated from pointer in fixed-size slot)
struct AUCStateSlot {
    AUCState *data;
};

void AUCInit(const AggregateFunction &, data_ptr_t state_ptr) {
    auto &slot = *reinterpret_cast<AUCStateSlot *>(state_ptr);
    slot.data = nullptr;
}

// In Update: lazy-alloc slot.data = new AUCState(); slot.data->pairs.emplace_back(score, label)
// In Combine: merge source->data->pairs into target->data->pairs
// StateDestroy: delete slot.data

// Finalize: sort by score descending, group ties, compute trapezoidal AUC
void AUCFinalize(...) {
    // sort descending by score
    std::sort(state.pairs.begin(), state.pairs.end(),
              [](auto &a, auto &b) { return a.first > b.first; });
    // group equal-score ties (rank-sum form to avoid optimistic tie interpolation)
    // ... trapezoidal integration ...
}
```

### Pattern 4: Table Function (Confusion Matrix)

Follows `tabfm_devices.cpp` exactly:
- `ConfusionBindData : TableFunctionData` — holds `unordered_map<pair<string,string>, int64_t> counts`
- `ConfusionGlobalState : GlobalTableFunctionState` — holds `vector<tuple<string,string,int64_t>> rows; idx_t offset`
- `DevicesBind` → `ConfusionBind`: populates bind data (at bind-time in DuckDB, the data isn't available yet; must use `InitGlobal` to build the map)

**Confusion matrix cannot populate at bind time** — inputs arrive at scan time. Use `InitGlobal` to run the sub-scan, or (preferred DuckDB pattern) build the output rows in the `Function` callback from state accumulated in `InitGlobal`. However, a table function's bind callback has no input data. The correct pattern is to accept the aggregate as a *subquery* — the macro form is better:

```sql
-- The user-facing tabfm_confusion_matrix is a TABLE MACRO wrapping a GROUP BY:
SELECT actual, predicted, count(*) AS count
FROM (VALUES ...) -- user passes tbl + actual_col + predicted_col
GROUP BY actual, predicted
```

Alternatively, implement confusion matrix as a two-argument aggregate that accumulates a `MAP<pair<string,string>, int64>` in state and returns a LIST of STRUCTs at finalize. This is simpler than a table function for this use case. **Recommended: aggregate returning `LIST(STRUCT(actual VARCHAR, predicted VARCHAR, count BIGINT))`** — consistent with the aggregate approach; user unnests the result.

**DECISION POINT for the planner:** the CONTEXT.md says "table function returning tidy long form" which implies a multi-row output. In DuckDB, this is most cleanly achieved as a `TABLE MACRO` that composes a `GROUP BY` query over the user's input (no C++ table function needed). This avoids the complexity of accumulating state in a table function.

### Pattern 5: Table Macro for CV (Cross-Validation)

```cpp
// Source: src/tabfm_macros.cpp (existing pattern)
// CV macro body (outline — k=5 example, must be generalized with a fold loop)
// Note: DuckDB table macros cannot loop; the fold-UNION is generated at registration
// or by the macro body using a recursive CTE or a fixed k param.

// Recommended approach: generate the SQL body dynamically at registration time
// for a given max_k, using UNION ALL, OR use a fixed-k approach where k is a
// required parameter and the body uses the fold_id column to filter.

// The cross_validate body structure:
R"(
WITH folds AS (
  SELECT *, hash(row_key, seed) % k AS fold_id
  FROM query('FROM ' || data)
),
fold_0 AS (
  SELECT tabfm_accuracy(label, yhat) AS metric_value, 0 AS fold_id
  FROM tabfm_classify(
    '(SELECT * FROM folds WHERE fold_id != 0)',
    target,
    test := '(SELECT * FROM folds WHERE fold_id = 0)'
  )
)
-- UNION ALL fold_1 ... fold_{k-1}
-- UNION ALL aggregate row
)"
```

**The fold-loop problem:** SQL macros cannot loop over `k`. Two viable approaches:

1. **Fixed-k SQL bodies:** pre-bake bodies for k=2..10, select at registration/call time. Simple but rigid.
2. **C++ macro body generator:** at `RegisterCrossValidateMacros` time, call a C++ function that returns the SQL string for a given k (the k parameter becomes part of the macro body), **but** DuckDB macros have a static body — the k value would need to be a named parameter embedded in the body via string substitution at registration.

**Recommended pattern (confirmed from existing codebase):** Use a SQL table macro with `k` as a named parameter, and generate the fold queries as a CTE pattern with `generate_series(0, k-1)` CROSS JOIN (or a lateral join). This avoids UNION ALL expansion. The fold-level metric computation uses a GROUP BY over `fold_id`:

```sql
-- Fold assignment is a table macro (or inline in CV macro)
WITH folds AS (
  SELECT *, (hash(row_key, seed) % k)::INTEGER AS fold_id
  FROM query('FROM ' || data)
),
predictions AS (
  SELECT p.*, folds.fold_id
  FROM tabfm_classify(
    '(SELECT * FROM folds)',  -- all rows as data
    target,
    test := '(SELECT * FROM folds WHERE fold_id = fold_id_param)'  -- PROBLEM: cannot iterate
  )
)
```

This still has the iteration problem. **The correct approach confirmed by existing research (`STACK.md §G.2`):** implement `tabfm_cross_validate` as a C++ `TableFunction` (not a macro) that builds and executes sub-queries via `ClientContext::Query` or the DuckDB C++ connection API. This matches what BigQuery ML does internally. The table function generates `k` separate predict + metric queries and unions the results.

**Simpler alternative that avoids sub-query execution:** The cross-validate function generates a single SQL query string that uses a lateral CTE approach, then calls `query(generated_sql)`. This is what the existing `tabfm_classify` macro does — it uses `query('FROM ' || data)` to defer SQL execution. See `tabfm_macros.cpp:89`.

### Pattern 6: Identifier Quoting (CV-04)

The existing `tabfm_classify` and `tabfm_regress` macros already have the fix at lines 93 and 123:

```
[VERIFIED: src/tabfm_macros.cpp:93] verbatim:
|| 'SELECT *, NULL AS "' || replace(target, '"', '""') || '" FROM (FROM ' || test || ')'
```

The new `tabfm_cross_validate` macro must apply the **same** `replace(target, '"', '""')` wrapping every time the target identifier is interpolated into a SQL string inside the macro body. The "P1 bug" mentioned in CONTEXT.md refers to the pattern that must be applied in the new macro — not a bug that still exists in the classify/regress macros.

### Anti-Patterns to Avoid

- **Silent averaging default for F1/AUC:** must throw `InvalidInputException` with message naming the `avg` parameter and valid values: `"tabfm_f1: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'"`
- **Computing AUC without tie-grouping:** equal-score predictions at a threshold boundary cause step artifacts in the ROC curve; group ties and use midpoint for the trapezoidal step
- **R² returning NaN on constant target:** when `SS_tot = 0`, check first whether `SS_res = 0` (perfect prediction → 1.0) or non-zero (useless → 0.0); never divide
- **MAPE crashing on zero actual:** skip rows where `|actual| < epsilon` (or exactly `actual == 0`); document skip behavior in the function description
- **Using `row_number() OVER (ORDER BY ...)` for fold assignment:** order-dependent and non-deterministic without a total order; use `hash(row_key, seed) % k` instead
- **Using `recursive := true` in unnest:** CLAUDE.md rule — always `max_depth := 3`
- **Calling `CaptureFunctionExecution` in Update (per-row):** call once in Bind only (CLAUDE.md rule #3)

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MAP(VARCHAR, DOUBLE) iteration | Custom parser | `MapValue::GetChildren` + `StructValue::GetChildren` | Already used in predict_agg.cpp:205; DuckDB internal API |
| Alias registration | Duplicate manual CreateAggregateFunctionInfo calls | Extend `anofox_function_alias.hpp` with `RegisterAggFunctionSetWithAlias` | Existing helpers for scalar/table; aggregate needs the same pattern |
| SQL string execution in C++ table function | Raw C API calls | `ClientContext` methods or build a macro that calls `query(...)` | DuckDB internal pattern; see tabfm_macros.cpp query() usage |
| Hash function for fold assignment | Custom hash | DuckDB `hash(expr, seed)` built-in | Variadic `ANY`, returns `UBIGINT`, deterministic within a version |
| Sort-then-median | Custom partial sort | `std::sort` + `std::nth_element` (or just `std::sort` for small N) | Standard library; MedAE state holds `vector<double>`, sort at finalize |
| Per-class metrics for F1 | Separate per-class calls | Single aggregate with `unordered_map<string, {tp,fp,fn}>` state | Combine across DuckDB workers naturally; one pass over data |

**Key insight:** DuckDB's aggregate framework handles parallel execution (Combine), NULL skipping (validity mask in Update), and output type declaration (return_type in AggregateFunction constructor) — never replicate these in metric-specific code.

---

## Metric Formulas (Exact scikit-learn Reference)

### Classification Metrics

#### Accuracy (CMET-01)
```
accuracy = correct / total
correct += 1 if actual == predicted (NULL rows skipped)
```
State: `{int64 correct, int64 total}`. Output: DOUBLE (NULL if total == 0). [ASSUMED] — textbook formula, universal.

#### Precision / Recall / F1 (CMET-02)

Per class `c`:
```
TP_c = count(actual == c AND predicted == c)
FP_c = count(actual != c AND predicted == c)
FN_c = count(actual == c AND predicted != c)
precision_c = TP_c / (TP_c + FP_c)    [0 if denominator == 0]
recall_c    = TP_c / (TP_c + FN_c)    [0 if denominator == 0]
f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)  [0 if both == 0]
```

Averaging modes:
- `macro`:    `(1/|C|) * sum_c f1_c` — unweighted mean
- `weighted`: `sum_c (support_c / N) * f1_c` — support-weighted; support_c = TP_c + FN_c
- `micro`:    global TP, FP, FN summed; `F1 = 2*TP / (2*TP + FP + FN)`

State: `unordered_map<string, {int64 tp, int64 fp, int64 fn}>`. [ASSUMED] — standard scikit-learn definition; the code must be verified against golden fixtures.

#### Log-Loss (CMET-03)
```
LL = -(1/N) * sum_i log(clip(proba[actual_i], eps, 1-eps))
eps = 1e-15  (scikit-learn default)
```
Inputs: `(actual VARCHAR, proba MAP(VARCHAR, DOUBLE))`. Look up `proba[actual_i]` by iterating `MapValue::GetChildren`. [ASSUMED] — standard cross-entropy; epsilon value is scikit-learn convention.

#### ROC-AUC (CMET-04)

Binary OvR (one class vs rest):
```
Collect (score, is_positive) pairs.
Sort by score descending.
Group ties: for a tie group, advance both TP and FP counts simultaneously.
Apply trapezoidal rule: AUC += (FPR_new - FPR_prev) * (TPR_new + TPR_prev) / 2
```

Multiclass averaging:
- `ovr`: compute binary AUC for each class (one-vs-rest), then average
  - `macro`: unweighted mean
  - `weighted`: support-weighted mean (see F1 above)
- `ovo`: compute binary AUC for all C*(C-1)/2 class pairs, then average

Tie-handling (rank-sum form): for a group of n_pos positive and n_neg negative with the same score, advance thresholds by both simultaneously before adding the trapezoidal trapezoid. Equivalent to the Mann-Whitney U statistic formula:
```
AUC = (U / (n_pos * n_neg))
U = sum_{i in positives} rank_i - n_pos*(n_pos+1)/2
rank computed with average ranks for ties
```
[ASSUMED] — standard, but golden test against sklearn is mandatory for tie handling.

#### ECE (CMET-06)
```
B = 10 equal-width bins over [0, 1]
For each row: assign to bin m = floor(max_proba * B)
ECE = sum_m (|B_m| / N) * |accuracy(B_m) - confidence(B_m)|
accuracy(B_m)   = fraction of correct predictions in bin m
confidence(B_m) = mean of max_proba in bin m
```
Inputs: `(actual VARCHAR, predicted VARCHAR, max_proba DOUBLE)` — the user provides the max-class probability (i.e., `map_extract(proba, yhat)` in SQL). [ASSUMED] — standard definition from Guo et al. 2017.

**Alternative input signature:** `(actual VARCHAR, proba MAP(VARCHAR, DOUBLE))` — ECE uses the top-1 probability, so the aggregate itself could extract it. The simpler user-facing form takes `(actual, predicted, max_proba)` where the user precomputes `max_proba`.

### Regression Metrics

#### RMSE (RMET-01)
```
sum_sq += (predicted - actual)^2 ; n++
finalize: sqrt(sum_sq / n)
```
State: `{double sum_sq, int64 n}`. [ASSUMED]

#### MAE (RMET-02)
```
sum_abs += |predicted - actual| ; n++
finalize: sum_abs / n
```
State: `{double sum_abs, int64 n}`. [ASSUMED]

#### R² (RMET-03)
Online single-pass (Welford-like):
```
sum_y  += actual
sum_y2 += actual * actual
sum_res += (actual - predicted)^2
n++
```
Finalize:
```
y_bar  = sum_y / n
SS_tot = sum_y2 - n * y_bar^2
SS_res = sum_res

if SS_tot == 0:
    return (SS_res == 0) ? 1.0 : 0.0   [constant-target edge case — scikit-learn convention]
return 1.0 - SS_res / SS_tot
```
Note: R² can be negative (model worse than constant mean). [ASSUMED] — scikit-learn `r2_score` specification; constant-target behavior matches documented sklearn convention.

#### MAPE (RMET-04)
```
For each row:
    if actual == 0: skip (document this)
    sum_mape += |actual - predicted| / |actual|
    n++
finalize: sum_mape / n * 100   (optional percent, or dimensionless — pick one and document)
```
[ASSUMED] — standard MAPE; zero-skip matches common implementations.

#### Median Absolute Error (RMET-04)
```
State: vector<double> residuals
Update: residuals.push_back(|actual - predicted|)
Finalize: sort residuals; return middle element (or avg of two middles for even n)
```
State grows O(N). Document this. [ASSUMED]

### Cross-Validation

#### Fold Assignment (CV-01)
```sql
-- In DuckDB SQL (stable within a DuckDB version):
(hash(row_key, seed) % k)::INTEGER AS fold_id
```
`hash()` is variadic ANY → UBIGINT, deterministic within a DuckDB version. [VERIFIED: duckdb/extension/core_functions/scalar/generic/hash.cpp:13-16] verbatim: `auto hash_fun = ScalarFunction({LogicalType::ANY}, LogicalType::HASH, HashFunction); hash_fun.varargs = LogicalType::ANY;`

`hash(a, b)` hashes both arguments together (uses `DataChunk::Hash` over all argument columns), giving a combined hash without manual bit-mixing. The seed is passed as a second argument: `hash(row_key, seed_literal)`.

---

## Common Pitfalls

### Pitfall 1: AUC Tie Handling
**What goes wrong:** Equal-score predictions at a classification threshold produce jagged ROC curve steps that inflate or deflate AUC.
**Why it happens:** The trapezoidal rule assumes each point is a distinct threshold; ties create ambiguity about whether a row is above or below the threshold.
**How to avoid:** Group all rows with the same score, compute TPR/FPR increment for the entire group simultaneously (equivalent to the Mann-Whitney U / rank-sum form), then add one trapezoidal trapezoid per tie group.
**Warning signs:** AUC differs from scikit-learn by more than 1e-6 on any test fixture with repeated scores.

### Pitfall 2: CV Data Leakage
**What goes wrong:** If the predict aggregate sees all rows (including the held-out fold) during preprocessing, feature statistics (mean, std, categorical encodings) are contaminated by test-fold data.
**Why it happens:** The existing `tabfm_predict_agg` fits preprocessing over its entire context window (all non-NULL rows). A "full table, then filter by fold" design passes all rows as context.
**How to avoid:** Use the two-table predict form: `tabfm_classify(train_folds_query, target, test := held_out_fold_query)`. The test rows arrive as NULL-label rows that never appear in the preprocessing statistics.
**Warning signs:** CV accuracy higher than hold-out accuracy; per-fold metrics too good to be true. The mandatory leakage-detecting golden test must catch this.

### Pitfall 3: R² on Constant-Target Data
**What goes wrong:** `SS_tot = 0` → division by zero, producing NaN or Inf.
**Why it happens:** When all actual values are equal, the mean predictor is perfect and SS_tot is zero.
**How to avoid:** Check `SS_tot == 0` before division. Return 1.0 if `SS_res == 0` (model is also perfect), 0.0 otherwise. This matches scikit-learn's convention.
**Warning signs:** NaN or Inf in R² output; test on a 3-row constant-target fixture.

### Pitfall 4: F1 / Precision / Recall Division by Zero
**What goes wrong:** A class that never appears in predictions has `TP + FP = 0` (precision undefined); a class that never appears in actuals has `TP + FN = 0` (recall undefined).
**Why it happens:** Real datasets often have class imbalance.
**How to avoid:** Return 0.0 for undefined precision/recall (scikit-learn convention with `zero_division=0`). Document this. Log-warn if a class has zero support (the denominator being zero is not exceptional; it is expected on imbalanced data).
**Warning signs:** Panic or exception on a fixture where one class is absent from predictions.

### Pitfall 5: MAP Lookup is O(|C|) in Update
**What goes wrong:** Log-loss / ECE / AUC iterates `MapValue::GetChildren` linearly for each row to find the true class probability. For C=10 this is a 10× overhead over a direct key lookup.
**Why it happens:** DuckDB's `Value` MAP representation doesn't offer O(1) key lookup.
**How to avoid:** Acceptable for C ≤ 10 (the extension's max class count). Document as a known O(N·C) update. For future phases with larger class counts, a pre-sorted key approach may be needed.
**Warning signs:** Visible performance degradation at C > 10 (blocked by the existing 10-class limit anyway).

### Pitfall 6: Scaffold File Coordination
**What goes wrong:** Two plan tasks both edit `CMakeLists.txt` (EXTENSION_SOURCES list) or `tabfm_registration.hpp` causing a merge conflict.
**Why it happens:** CLAUDE.md rule #2: scaffold-owned files require coordination.
**How to avoid:** Do all scaffold edits in a single Wave 0 task before any module implementation tasks. Add stub `Register*` function bodies at the same time. This is a plan-level constraint.
**Warning signs:** Any plan that has two tasks both touching CMakeLists.txt or tabfm_registration.hpp.

### Pitfall 7: `hash()` Return Type is UBIGINT (unsigned)
**What goes wrong:** `hash(row_key, seed) % k` with a signed `k` or signed cast produces unexpected negative values in some SQL dialects.
**Why it happens:** UBIGINT modulo signed INTEGER: if k is passed as INTEGER and DuckDB promotes to BIGINT (signed), the result should still be correct because `k > 0` and UBIGINT % BIGINT is well-defined in DuckDB. But an explicit cast to INTEGER may be needed for the fold_id column type.
**How to avoid:** Use `(hash(row_key, seed) % CAST(k AS UBIGINT))::INTEGER` to be explicit.

---

## Code Examples

### Registering an Aggregate with Alias (new helper needed)

```cpp
// To add to anofox_function_alias.hpp:
// Source: pattern from anofox_function_alias.hpp:27-43 (scalar variant)
inline void RegisterAggregateFunctionSetWithAlias(ExtensionLoader &loader,
                                                   AggregateFunctionSet func_set,
                                                   const std::string &alias_name,
                                                   vector<FunctionDescription> descriptions = {}) {
    CreateAggregateFunctionInfo primary_info(func_set);
    primary_info.descriptions = descriptions;
    loader.RegisterFunction(primary_info);

    AggregateFunctionSet alias_set(alias_name);
    for (auto &func : func_set.functions) {
        auto alias_func = func;
        alias_func.name = alias_name;
        alias_set.AddFunction(std::move(alias_func));
    }
    CreateAggregateFunctionInfo alias_info(std::move(alias_set));
    alias_info.descriptions = std::move(descriptions);
    alias_info.alias_of = func_set.name;
    loader.RegisterFunction(alias_info);
}
```

### Null-Safe Value Comparison for Accuracy

```cpp
// Per CLAUDE.md: NULL rows are skipped in metric aggregates.
// Access validity before reading typed data:
UnifiedVectorFormat actual_data, pred_data;
inputs[0].ToUnifiedFormat(count, actual_data);
inputs[1].ToUnifiedFormat(count, pred_data);
for (idx_t i = 0; i < count; i++) {
    idx_t aidx = actual_data.sel->get_index(i);
    idx_t pidx = pred_data.sel->get_index(i);
    if (!actual_data.validity.RowIsValid(aidx) || !pred_data.validity.RowIsValid(pidx)) {
        continue;
    }
    // typed comparison...
}
```

### Sqllogictest Golden Pattern (from test/sql/tabfm_classify.test)

```sql
# name: test/sql/tabfm_metrics_classification.test
# description: tabfm_accuracy, tabfm_f1, tabfm_log_loss, tabfm_roc_auc, tabfm_ece,
#              tabfm_confusion_matrix — golden values from tools/golden/generate_metric_fixtures.py
# group: [anofox_tabfm]

require anofox_tabfm

statement ok
LOAD anofox_tabfm

# Golden 3-class fixture (values from sklearn, tolerance 1e-6)
statement ok
CREATE TABLE preds AS SELECT * FROM (VALUES
  ('cat', 'cat', 0.7, 0.2, 0.1),
  ('dog', 'dog', 0.1, 0.8, 0.1),
  ('fish', 'cat', 0.5, 0.3, 0.2)
) t(actual, predicted, p_cat, p_dog, p_fish);

query I
SELECT round(tabfm_accuracy(actual, predicted), 6) FROM preds
----
0.666667

# ... F1 macro, log_loss, etc. with golden values from sklearn
```

### Telemetry Call (CLAUDE.md rule #3)

```cpp
// Once per BIND (not per Update). Per tabfm_predict_agg.cpp:328:
unique_ptr<FunctionData> AccuracyBind(ClientContext &context, AggregateFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
    PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_accuracy");
    // ... bind logic
}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Model-coupled evaluation (BigQuery ML `ML.EVALUATE`) | Model-agnostic composable aggregates on `(actual, predicted)` columns | Works with any model family, any SQL query |
| Full-table CV (fit on all, predict all, filter by fold) | Two-table predict form (train context + NULL-label test) per fold | Eliminates preprocessing leakage |
| Manual fold assignment via `row_number()` | `hash(row_key, seed) % k` deterministic fold | Reproducible across re-runs; order-independent |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scikit-learn `zero_division=0` convention for precision/recall: undefined cases return 0.0 | Metric Formulas | Planner makes different convention choice; golden tests catch disagreement |
| A2 | MAPE returns dimensionless ratio (not percentage) | Metric Formulas | User surprise; document clearly whichever is chosen |
| A3 | ECE uses `(actual VARCHAR, predicted VARCHAR, max_proba DOUBLE)` input signature (user provides max_proba) | Architecture Patterns | Simpler for aggregate; adds user friction; planner can decide |
| A4 | AUC for `ovo` mode averages all C*(C-1)/2 binary AUC values uniformly | Metric Formulas | Could be support-weighted; verify against sklearn with `multiclass='ovo'` |
| A5 | The `tabfm_cross_validate` function is implemented as a C++ `TableFunction` (not a SQL table macro) because SQL macros cannot loop | Architecture Patterns | A pure-SQL approach using `generate_series` + lateral join may be possible in DuckDB 1.5.4 — verify at planning time |
| A6 | `hash(row_key, seed)` combining row_key and seed is stable across DuckDB patch versions (1.5.x) | Fold Assignment | Fold assignments change on DuckDB upgrade; document |
| A7 | R² constant-target convention: return 1.0 for perfect prediction, 0.0 otherwise | Metric Formulas | Well-documented sklearn convention; LOW risk |

---

## Open Questions

1. **CV implementation: C++ TableFunction vs SQL macro with fixed-k bodies**
   - What we know: DuckDB table macros cannot loop; `query()` can execute dynamically assembled SQL.
   - What's unclear: Whether a SQL macro can call `query(generate_cv_sql(data, target, k, seed, metric))` where `generate_cv_sql` is a scalar function that returns the full fold-loop SQL as a string. This would keep CV as a macro rather than C++ table function.
   - Recommendation: At planning time, attempt the macro approach first (lower implementation complexity); fall back to C++ table function if `query()` inside a macro has restrictions in DuckDB 1.5.4.

2. **Confusion matrix as table function vs aggregate returning LIST(STRUCT)**
   - What we know: CONTEXT.md says "table function returning tidy long form `(actual, predicted, count)`".
   - What's unclear: Whether a DuckDB table function can receive an aggregate input (it can't — table functions bind at catalog time, not at query time with arbitrary grouping). The cleanest implementation is a SQL macro wrapping a GROUP BY.
   - Recommendation: Implement as a `TABLE MACRO` that wraps `SELECT actual, predicted, COUNT(*) ... GROUP BY actual, predicted`. The user calls `tabfm_confusion_matrix('my_table', 'actual_col', 'predicted_col')`.

3. **Leakage-detecting golden test design**
   - What we know: STATE.md flags this as non-negotiable before CV ships.
   - What's unclear: The exact mechanism — a fixture where leakage would produce a measurably inflated accuracy (e.g., a constant-feature test set that a leaky model would score perfectly but a leakage-safe model would score at chance).
   - Recommendation: Build a fixture where the test column is deterministically correlated with fold ID; leakage → ~100% accuracy, no-leakage → ~1/k accuracy.

---

## Project Constraints (from CLAUDE.md)

- **TDD (red-green):** Write failing sqllogictest/Catch2 test first, then implement. Error-path tests are first-class.
- **File ownership (rule #2):** One module = one `src/tabfm_*.cpp`. Scaffold-owned files (`CMakeLists.txt`, `anofox_tabfm_extension.cpp`, `tabfm_registration.hpp`, `tabfm_settings.cpp`) require coordination — batch all edits.
- **Telemetry (rule #3):** Every user-facing function calls `PostHogTelemetry::Instance().CaptureFunctionExecution("<short_name>")` once per execution at bind time.
- **Naming (rule #4):** Full names `anofox_tabfm_*` + short alias `tabfm_*` via `anofox_function_alias.hpp`.
- **Errors (rule #5):** Every failure is a DuckDB exception naming the fixing SET/CALL/parameter. Example: `"tabfm_f1: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'"`.
- **License wall (rule #6):** No Google/vendor weight bytes anywhere; fixtures are random-init. Metric code is weight-free; no constraint here.
- **`unnest(x, max_depth := 3)` — never `recursive := true`.**
- **Test files:** sqllogictest `.test` in `test/sql/`; Catch2 TUs in `TABFM_CPP_TEST_SOURCES` in CMakeLists.txt.
- **Build:** `make debug` + `make test_debug` (Catch2 + sqllogictests); individual SQL test: `./build/debug/test/unittest test/sql/metrics.test`.

---

## Environment Availability

All dependencies are already present — this phase adds no external tools or services.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| DuckDB C++ API | All modules | ✓ | 1.5.4 (pinned submodule) | — |
| C++17 standard library | `<algorithm>`, `<cmath>`, `<vector>`, `<unordered_map>` | ✓ | System GCC/Clang | — |
| Python + scikit-learn | Golden fixture generation | Not verified in this session | — | Generate fixtures separately before plan execution |
| `make` + CMake | Build | ✓ | Assumed from project setup | — |

**Missing dependencies with no fallback:** None for implementation. Golden fixture generation requires `scikit-learn` — confirmed as a tool requirement, not a build requirement.

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Metrics are pure computation; no auth surface |
| V3 Session Management | No | No session state added |
| V4 Access Control | No | DuckDB access control governs query execution |
| V5 Input Validation | Yes | All metric inputs are validated at bind time; invalid `avg` parameter throws named exception |
| V6 Cryptography | No | `hash()` is not cryptographic (DuckDB docs say so explicitly) — fold assignment is not a security use |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| SQL injection via target identifier interpolation in CV macro | Tampering | `replace(target, '"', '""')` wrapping (same as tabfm_macros.cpp:93) — prevents identifier escape |
| Integer overflow in metric state (very large datasets) | Tampering | Use `int64_t` accumulators (not `int32_t`); `double` for sums (64-bit IEEE 754 has sufficient range) |
| Division by zero in metric finalize | Denial of Service | All denominators checked; return NULL or documented edge-case value |
| MAP key not found for true class in log-loss | Information Disclosure | If `proba` MAP does not contain the `actual` class, treat as `p=0` → clip to `eps` → very large log-loss; do not throw |

---

## Sources

### Primary (HIGH confidence)
- `src/tabfm_predict_agg.cpp` — aggregate bind/update/combine/finalize pattern, MAP access, state lifecycle [VERIFIED: read this session]
- `src/tabfm_macros.cpp` — table macro registration, identifier quoting pattern at line 93 [VERIFIED: src/tabfm_macros.cpp:93]
- `src/include/anofox_function_alias.hpp` — alias helper implementations [VERIFIED: read this session]
- `src/include/tabfm_registration.hpp` — registration entry point pattern [VERIFIED: read this session]
- `src/tabfm_devices.cpp:500-551` — table function pattern [VERIFIED: read this session]
- `duckdb/extension/core_functions/scalar/generic/hash.cpp:13-16` — `hash()` is variadic ANY [VERIFIED: read this session]
- `CMakeLists.txt:118-141` — `TABFM_CPP_TEST_SOURCES` list and Catch2 registration pattern [VERIFIED: read this session]

### Secondary (MEDIUM confidence)
- `.planning/research/STACK.md` — metric formulas and state layouts from prior project research [CITED: .planning/research/STACK.md]
- `.planning/research/SUMMARY.md` — pitfall catalogue (CV leakage, AUC tie handling) [CITED: .planning/research/SUMMARY.md]

### Tertiary (LOW confidence)
- scikit-learn metric conventions (zero_division, log-loss epsilon, R² constant-target) — training knowledge, not verified this session [ASSUMED]
- ECE formula from Guo et al. 2017 — training knowledge [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all APIs verified in codebase this session
- Architecture patterns: HIGH — directly derived from existing code
- Metric formulas: MEDIUM — standard definitions, unverified against sklearn this session (golden tests are the verification gate)
- Pitfalls: HIGH — derived from codebase analysis + prior research

**Research date:** 2026-09-20
**Valid until:** 2026-10-20 (stable DuckDB API; scikit-learn formulas are stable)
