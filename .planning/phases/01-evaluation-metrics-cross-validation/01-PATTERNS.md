# Phase 1: Evaluation Metrics + Cross-Validation — Pattern Map

**Mapped:** 2026-09-20
**Files analyzed:** 13 (3 new .cpp, 3 new .hpp, 3 new .test, 1 new Catch2 TU, 3 scaffold-owned edits)
**Analogs found:** 13 / 13

---

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `src/tabfm_metrics_classification.cpp` | aggregate-module | request-response (batch) | `src/tabfm_predict_agg.cpp` | role-match (same aggregate triplet pattern) |
| `src/tabfm_metrics_regression.cpp` | aggregate-module | request-response (batch) | `src/tabfm_predict_agg.cpp` | role-match |
| `src/tabfm_crossval.cpp` | macro-module | request-response | `src/tabfm_macros.cpp` | exact (table macro + SQL body builder) |
| `src/include/tabfm_metrics_classification.hpp` | header | — | `src/include/tabfm_registration.hpp` | role-match (module public interface header) |
| `src/include/tabfm_metrics_regression.hpp` | header | — | `src/include/tabfm_registration.hpp` | role-match |
| `src/include/tabfm_crossval.hpp` | header | — | `src/include/tabfm_registration.hpp` | role-match |
| `test/sql/tabfm_metrics_classification.test` | sqllogictest | — | `test/sql/tabfm_classify.test` | exact |
| `test/sql/tabfm_metrics_regression.test` | sqllogictest | — | `test/sql/tabfm_classify.test` | exact |
| `test/sql/tabfm_crossval.test` | sqllogictest | — | `test/sql/tabfm_classify.test` | exact |
| `test/cpp/test_tabfm_metrics.cpp` | Catch2 TU | — | `test/cpp/test_tabfm_scaffold.cpp` | exact |
| `CMakeLists.txt` (edit) | config | — | `CMakeLists.txt` lines 42–58, 118–126 | exact |
| `src/include/tabfm_registration.hpp` (edit) | config-header | — | `src/include/tabfm_registration.hpp` lines 10–14 | exact |
| `src/anofox_tabfm_extension.cpp` (edit) | entry-point | — | `src/anofox_tabfm_extension.cpp` lines 114–118 | exact |
| `src/include/anofox_function_alias.hpp` (edit) | utility-header | — | `src/include/anofox_function_alias.hpp` lines 46–65 | exact (extend with aggregate variant) |

---

## Pattern Assignments

### `src/tabfm_metrics_classification.cpp` + `src/tabfm_metrics_regression.cpp`
**Role:** aggregate-module | **Data flow:** batch request-response
**Analog:** `src/tabfm_predict_agg.cpp`

#### Imports pattern (lines 1–14)
```cpp
#include "tabfm_metrics_classification.hpp"   // or tabfm_metrics_regression.hpp
#include "tabfm_registration.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/common/types/value.hpp"           // MapValue, StructValue, Value
#include "telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace duckdb {
namespace anofox {
```

#### Namespace layout (lines 16–17, 31, 727)
```cpp
namespace duckdb {
namespace anofox {

namespace {  // anonymous — all state structs and callbacks are file-local

// ... state structs, callbacks ...

} // anonymous namespace

void RegisterClassificationMetrics(ExtensionLoader &loader) { ... }

} // namespace anofox
} // namespace duckdb
```

#### Aggregate state + triplet pattern (lines 393–440, 445–506, 520–560, 701–724)

The full triplet signature set — copy exactly:

```cpp
// State size callback
idx_t MetricStateSize(const AggregateFunction &) {
    return sizeof(MyState);
}

// Initialize (placement-new or zeroing)
void MetricStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
    new (state_ptr) MyState();
}

// Update — UnifiedVectorFormat pattern for NULL-safe per-row access
void MetricUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t,
                  Vector &state_vector, idx_t count) {
    UnifiedVectorFormat sdata, input0_data, input1_data;
    state_vector.ToUnifiedFormat(count, sdata);
    inputs[0].ToUnifiedFormat(count, input0_data);
    inputs[1].ToUnifiedFormat(count, input1_data);
    auto states = reinterpret_cast<MyState **>(sdata.data);

    for (idx_t i = 0; i < count; i++) {
        idx_t sidx  = sdata.sel->get_index(i);
        idx_t aidx  = input0_data.sel->get_index(i);
        idx_t pidx  = input1_data.sel->get_index(i);
        // NULL skip: matches standard SQL aggregate semantics (CONTEXT.md)
        if (!input0_data.validity.RowIsValid(aidx) ||
            !input1_data.validity.RowIsValid(pidx)) {
            continue;
        }
        auto &state = *states[sidx];
        // ... accumulate ...
    }
}

// Combine (for parallel execution)
void MetricCombine(Vector &source_vector, Vector &target_vector,
                   AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);
    auto sources = reinterpret_cast<MyState **>(source_data.data);
    auto targets = reinterpret_cast<MyState **>(target_data.data);
    for (idx_t i = 0; i < count; i++) {
        auto &src = *sources[source_data.sel->get_index(i)];
        auto &tgt = *targets[target_data.sel->get_index(i)];
        // ... merge src into tgt ...
    }
}

// Finalize — write into FlatVector with NULL on empty state
void MetricFinalize(Vector &state_vector, AggregateInputData &, Vector &result,
                    idx_t count, idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<MyState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        if (state.n == 0) {
            FlatVector::SetNull(result, i + offset, true);
        } else {
            FlatVector::GetData<double>(result)[i + offset] = /* compute */;
        }
    }
}
```

#### Telemetry-at-bind pattern (line 328)
```cpp
// Bind callback — telemetry called ONCE here, never in Update
unique_ptr<FunctionData> MetricBind(ClientContext &context, AggregateFunction &function,
                                    vector<unique_ptr<Expression>> &arguments) {
    PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_accuracy");
    // ... validate arguments, set function.return_type if needed ...
    return nullptr; // or a bind-data struct if state needs it
}
```

#### MAP(VARCHAR, DOUBLE) access in Update — for log-loss, ECE, ROC-AUC (line 205)
```cpp
// proba input is MAP(VARCHAR, DOUBLE): each entry is a STRUCT({key VARCHAR, value DOUBLE})
Value proba_val = inputs[1].GetValue(i);
if (proba_val.IsNull()) continue;
string actual_str = inputs[0].GetValue(i).ToString();

double p = 0.0;
for (auto &kv : MapValue::GetChildren(proba_val)) {
    auto &entry = StructValue::GetChildren(kv);
    if (entry[0].ToString() == actual_str) {
        p = DoubleValue::Get(entry[1]);
        break;
    }
}
```

#### StateDestroy callback — required for heap-owning states (AUC vector, MedAE vector) (lines 406–413)
```cpp
void MetricStateDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<HeapState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &s = *states[sdata.sel->get_index(i)];
        delete s.data;
        s.data = nullptr;
    }
}
```

#### Registration pattern (lines 718–724)
```cpp
void RegisterClassificationMetrics(ExtensionLoader &loader) {
    // One block per metric function:
    {
        AggregateFunctionSet set("anofox_tabfm_accuracy");
        AggregateFunction fn("anofox_tabfm_accuracy",
            {LogicalType::ANY, LogicalType::ANY}, LogicalType::DOUBLE,
            AccuracyStateSize, AccuracyStateInit, AccuracyUpdate,
            AccuracyCombine, AccuracyFinalize,
            /*simple_update=*/nullptr, AccuracyBind,
            /*state_destroy=*/nullptr);
        set.AddFunction(fn);
        CreateAggregateFunctionInfo info(set);
        loader.RegisterFunction(info);
        // alias — use new RegisterAggregateFunctionSetWithAlias helper
        // (see anofox_function_alias.hpp section below)
    }
    // ... repeat for tabfm_f1, tabfm_log_loss, tabfm_roc_auc, tabfm_ece
    // ... tabfm_rmse, tabfm_mae, tabfm_r2, tabfm_mape, tabfm_medae
}
```

#### Error messages — InvalidInputException with parameter name (lines 461–463, 558–559)
```cpp
// CONTEXT.md / CLAUDE.md rule #5: every exception names the fixing parameter
throw InvalidInputException(
    "tabfm_f1: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'");

throw InvalidInputException(
    "tabfm_roc_auc: 'avg' is required — pass avg := 'ovr' or 'ovo'");
```

---

### `src/tabfm_crossval.cpp`
**Role:** macro-module | **Data flow:** request-response (SQL body generation)
**Analog:** `src/tabfm_macros.cpp`

#### Imports pattern (lines 1–9)
```cpp
#include "tabfm_crossval.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
```

#### PredictMacroDef struct pattern (lines 63–69) — replicate for CV macro definition
```cpp
struct CVMacroDef {
    const char *parameters[8];      // positional param names, nullptr-terminated
    const char *default_params[8];  // "name=sql_default" for trailing optionals
    const char *body;
    const char *description;
    const char *example;
};
```

#### SQL body with identifier quoting (lines 93, 123) — CRITICAL: always use this form
```cpp
// Whenever the `target` identifier is interpolated into a SQL string inside
// the macro body, quote with double-quotes and escape embedded quotes:
|| 'SELECT *, NULL AS "' || replace(target, '"', '""') || '" FROM (FROM ' || test || ')'
// Apply the same replace(target, '"', '""') in every tabfm_cross_validate
// SQL string that references the target column by name.
```

#### BuildTableMacroFunction + BuildMacroInfo pattern (lines 136–186)
```cpp
// Parse body SQL → extract SelectNode → wrap in TableMacroFunction
unique_ptr<MacroFunction> BuildTableMacroFunction(const CVMacroDef &def) {
    Parser parser;
    parser.ParseQuery(def.body);
    // ... same validation as tabfm_macros.cpp:139-143 ...
    auto node = std::move(parser.statements[0]->Cast<SelectStatement>().node);
    auto function = make_uniq<TableMacroFunction>(std::move(node));
    for (idx_t i = 0; def.parameters[i] != nullptr; i++) {
        function->parameters.push_back(make_uniq<ColumnRefExpression>(def.parameters[i]));
        function->types.push_back(LogicalType::UNKNOWN);
    }
    for (idx_t i = 0; def.default_params[i] != nullptr; i++) {
        // ... same default-param parsing as tabfm_macros.cpp:152-162 ...
    }
    return std::move(function);
}

// Build CreateMacroInfo with alias_of + FunctionDescription
unique_ptr<CreateMacroInfo> BuildMacroInfo(const string &name, const CVMacroDef &def,
                                           const string &alias_of) {
    auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
    info->schema = DEFAULT_SCHEMA;
    info->name = name;
    info->temporary = true;
    info->internal = true;
    info->alias_of = alias_of;
    info->macros.push_back(BuildTableMacroFunction(def));
    // ... FunctionDescription as in tabfm_macros.cpp:178-185 ...
    return info;
}

// Register primary + alias pair
void RegisterMacroWithAlias(ExtensionLoader &loader, const string &full_name,
                            const string &alias_name, const CVMacroDef &def) {
    auto primary = BuildMacroInfo(full_name, def, string());
    loader.RegisterFunction(*primary);
    auto alias = BuildMacroInfo(alias_name, def, full_name);
    loader.RegisterFunction(*alias);
}
```

#### query() + FROM string pattern (lines 89–96) — CV macro SQL body must use this
```cpp
// DuckDB table macros cannot take relation arguments directly.
// Use query('FROM ' || data) to defer SQL execution — same as classify/regress.
// The cross_validate body follows this pattern per fold:
R"(
    WITH folds AS (
      SELECT *, (hash(row_key, seed) % CAST(k AS UBIGINT))::INTEGER AS fold_id
      FROM query('FROM ' || data)
    )
    ...
)"
// Identifier quoting for target in every fold subquery:
// 'SELECT *, NULL AS "' || replace(target, '"', '""') || '" FROM ...'
```

---

### `src/include/tabfm_metrics_classification.hpp`, `tabfm_metrics_regression.hpp`, `tabfm_crossval.hpp`
**Role:** module public interface headers
**Analog:** `src/include/tabfm_registration.hpp`

#### Header pattern (tabfm_registration.hpp lines 1–17)
```cpp
#pragma once

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

/// Register all classification metric aggregates and table functions.
/// Called from LoadInternal() in anofox_tabfm_extension.cpp.
void RegisterClassificationMetrics(ExtensionLoader &loader);

} // namespace anofox
} // namespace duckdb
```

---

### `src/include/anofox_function_alias.hpp` (edit — add aggregate helper)
**Analog:** existing `RegisterScalarFunctionSetWithAlias` pattern (lines 46–65)

#### New helper to add (after line 106, before closing namespaces)
```cpp
// Helper to register an aggregate function set with an alias.
// Pattern mirrors RegisterScalarFunctionSetWithAlias exactly.
// Include: duckdb/parser/parsed_data/create_aggregate_function_info.hpp
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
Also add `#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"` to the header's include block (after line 7).

---

### `src/include/tabfm_registration.hpp` (edit)
**Analog:** existing lines 10–14

#### Lines to add (after line 14, before closing brace)
```cpp
void RegisterClassificationMetrics(ExtensionLoader &loader); // tabfm_metrics_classification.cpp
void RegisterRegressionMetrics(ExtensionLoader &loader);     // tabfm_metrics_regression.cpp
void RegisterCrossValidateMacros(ExtensionLoader &loader);   // tabfm_crossval.cpp
```

---

### `src/anofox_tabfm_extension.cpp` (edit)
**Analog:** lines 114–118

#### Lines to add (after line 118 `RegisterPredictMacros`)
```cpp
anofox::RegisterClassificationMetrics(loader);
anofox::RegisterRegressionMetrics(loader);
anofox::RegisterCrossValidateMacros(loader);
```

---

### `CMakeLists.txt` (edit)
**Analog:** lines 42–58 (EXTENSION_SOURCES) and lines 118–126 (TABFM_CPP_TEST_SOURCES)

#### EXTENSION_SOURCES additions (after line 54 `src/tabfm_macros.cpp`)
```cmake
        src/tabfm_metrics_classification.cpp
        src/tabfm_metrics_regression.cpp
        src/tabfm_crossval.cpp
```

#### TABFM_CPP_TEST_SOURCES additions (after line 125 `test_tabfm_bundled_resources.cpp`)
```cmake
    test/cpp/test_tabfm_metrics.cpp
```

---

### `test/sql/tabfm_metrics_classification.test`, `tabfm_metrics_regression.test`, `tabfm_crossval.test`
**Analog:** `test/sql/tabfm_classify.test`

#### File header pattern (lines 1–15)
```sql
# name: test/sql/tabfm_metrics_classification.test
# description: tabfm_accuracy, tabfm_f1, tabfm_log_loss, tabfm_roc_auc, tabfm_ece,
#              tabfm_confusion_matrix golden values (tools/golden/generate_metric_fixtures.py)
# group: [anofox_tabfm]

require anofox_tabfm

statement ok
LOAD anofox_tabfm
```

#### Golden value query pattern (lines 31–38 of tabfm_classify.test)
```sql
# Values from sklearn; tolerance applied via round():
query I
SELECT round(tabfm_accuracy(actual, predicted), 6) FROM preds
----
0.666667

# Error paths are first-class (CLAUDE.md rule #1):
statement error
SELECT tabfm_f1(actual, predicted) FROM preds
----
tabfm_f1: 'avg' is required
```

---

### `test/cpp/test_tabfm_metrics.cpp`
**Analog:** `test/cpp/test_tabfm_scaffold.cpp`

#### Catch2 TU pattern (lines 1–16 of test_tabfm_scaffold.cpp)
```cpp
// test/cpp/test_tabfm_metrics.cpp — Catch2 unit tests for metric aggregate
// state logic (AccuracyState, F1State, RMSEState, etc.) independent of DuckDB
// query execution.

#include "catch.hpp"
#include "duckdb.hpp"

TEST_CASE("tabfm_accuracy: basic correct/total", "[tabfm][metrics]") {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    // Load extension, run aggregate SQL, check results
    // ...
}
```

---

## Shared Patterns

### Namespace wrapping
**Source:** `src/tabfm_predict_agg.cpp` lines 16–17 and `src/tabfm_macros.cpp` lines 11–12
**Apply to:** All new `.cpp` files
```cpp
namespace duckdb {
namespace anofox {
// All public functions here
} // namespace anofox
} // namespace duckdb
```
File-local helpers always in `namespace { ... }` immediately inside `namespace anofox`.

### Telemetry — once per bind
**Source:** `src/tabfm_predict_agg.cpp` line 328
**Apply to:** Every Bind callback in `tabfm_metrics_classification.cpp`, `tabfm_metrics_regression.cpp`, `tabfm_crossval.cpp`
```cpp
PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_accuracy");
// short name = the tabfm_* alias, not the anofox_tabfm_* full name
```

### NULL skip in Update via UnifiedVectorFormat validity
**Source:** `src/tabfm_predict_agg.cpp` lines 449–466
**Apply to:** Every Update callback (all metric aggregates)
```cpp
UnifiedVectorFormat sdata, d0, d1;
state_vector.ToUnifiedFormat(count, sdata);
inputs[0].ToUnifiedFormat(count, d0);
inputs[1].ToUnifiedFormat(count, d1);
auto states = reinterpret_cast<State **>(sdata.data);
for (idx_t i = 0; i < count; i++) {
    if (!d0.validity.RowIsValid(d0.sel->get_index(i)) ||
        !d1.validity.RowIsValid(d1.sel->get_index(i))) {
        continue;  // SQL aggregate NULL semantics
    }
    // ...
}
```

### FlatVector NULL on empty state in Finalize
**Source:** `src/tabfm_predict_agg.cpp` lines 534–537
**Apply to:** Every Finalize callback
```cpp
if (state.n == 0) {
    FlatVector::SetNull(result, i + offset, true);
} else {
    FlatVector::GetData<double>(result)[i + offset] = computed_value;
}
```

### DuckDB exception with fixing instruction
**Source:** `src/tabfm_predict_agg.cpp` lines 461–463
**Apply to:** All error paths in all new modules
```cpp
throw InvalidInputException(
    "tabfm_<fn>: <problem>. <Fixing action>: <exact SQL>");
```

### Full-name + alias registration for aggregates
**Source:** `src/tabfm_predict_agg.cpp` lines 718–724 + `src/include/anofox_function_alias.hpp`
**Apply to:** Every aggregate in `tabfm_metrics_classification.cpp` and `tabfm_metrics_regression.cpp`

Primary name always `anofox_tabfm_*`; alias always `tabfm_*`.
Use the new `RegisterAggregateFunctionSetWithAlias` helper once it is added to `anofox_function_alias.hpp`.

### Macro alias registration
**Source:** `src/tabfm_macros.cpp` lines 188–194, 198–201
**Apply to:** `tabfm_crossval.cpp`
```cpp
void RegisterCrossValidateMacros(ExtensionLoader &loader) {
    RegisterMacroWithAlias(loader, "anofox_tabfm_cross_validate", "tabfm_cross_validate", CV_MACRO);
    RegisterMacroWithAlias(loader, "anofox_tabfm_fold_assign", "tabfm_fold_assign", FOLD_ASSIGN_MACRO);
}
```

---

## No Analog Found

No new files in this phase lack a close codebase analog. Every pattern has a verified match.

| File | Nearest Analog | Note |
|------|---------------|-------|
| `tools/golden/generate_metric_fixtures.py` | None in codebase | Python script for sklearn golden values; no C++ analog. Follow scikit-learn API directly. Not a build artifact; not listed in CMakeLists. |

---

## Metadata

**Analog search scope:** `src/`, `src/include/`, `test/sql/`, `test/cpp/`, `CMakeLists.txt`
**Files scanned:** 9 tracked source files read in full
**All analog paths verified:** `git ls-files` confirms all paths are tracked (no mirror paths)
**Pattern extraction date:** 2026-09-20
