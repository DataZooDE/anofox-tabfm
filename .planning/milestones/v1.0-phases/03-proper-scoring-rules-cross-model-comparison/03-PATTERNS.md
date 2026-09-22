# Phase 3: Proper Scoring Rules + Cross-Model Comparison — Pattern Map

**Mapped:** 2026-09-22
**Files analyzed:** 9 (4 new source, 2 new test, 3 scaffold-owned modifications)
**Analogs found:** 9 / 9

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/tabfm_scoring.cpp` | aggregate | request-response (streaming update/combine/finalize) | `src/tabfm_metrics_regression.cpp` | exact (same aggregate triplet, DOUBLE state, no StateDestroy) |
| `src/include/tabfm_scoring.hpp` | header | — | `src/include/tabfm_metrics_regression.hpp` | exact (single declaration, #pragma once) |
| `src/tabfm_crossval.cpp` (extend) OR `src/tabfm_macros.cpp` (extend) | macro | request-response (SQL string building + query() executor) | `src/tabfm_crossval.cpp` CVMacroDef pattern | exact |
| `test/sql/tabfm_scoring.test` | test | — | `test/sql/tabfm_distribution.test` | role-match |
| `test/cpp/test_tabfm_scoring.cpp` | test | — | `test/cpp/test_tabfm_metrics_regression.cpp` | exact |
| `CMakeLists.txt` (scaffold) | config | — | lines 57-59 (`EXTENSION_SOURCES`), lines 131-132 (`TABFM_CPP_TEST_SOURCES`) | exact |
| `src/anofox_tabfm_extension.cpp` (scaffold) | config | — | lines 123-125 (`LoadInternal` registration calls) | exact |
| `src/include/tabfm_registration.hpp` (scaffold) | header | — | lines 16-18 (existing `Register*` declarations) | exact |
| `tools/parity/src/parity/crps_reference.py` | utility | transform | existing `tools/parity/` Python scripts | role-match |

---

## Pattern Assignments

### `src/tabfm_scoring.cpp` (aggregate, request-response)

**Primary analog:** `src/tabfm_metrics_regression.cpp`
**Secondary analog:** `src/tabfm_metrics_classification.cpp` (bind-gate + MAP/STRUCT reading)

---

#### Imports pattern (`src/tabfm_metrics_regression.cpp` lines 25-35)

```cpp
#include "tabfm_metrics_regression.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox {
namespace {
```

For `tabfm_scoring.cpp`, also add:
```cpp
#include "tabfm_predict.hpp"   // for DistributionQuantile (external linkage, tabfm_predict.hpp:206)
```

---

#### State pattern — simple DOUBLE accumulator (`src/tabfm_metrics_regression.cpp` lines 53-130)

Use this shape for `tabfm_crps` and `tabfm_log_score` states (no heap allocation, no StateDestroy needed):

```cpp
struct RMSEState {
    double  sum_sq; // sum of (predicted - actual)^2
    int64_t n;      // valid (non-NULL) pairs seen
};

idx_t RMSEStateSize(const AggregateFunction &) { return sizeof(RMSEState); }

void RMSEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
    new (state_ptr) RMSEState{0.0, 0};
}
```

For `tabfm_interval_score` add a `IScoreBindData : FunctionData` (see bind-data pattern below).

---

#### UnifiedVectorFormat NULL-skip in Update — DOUBLE inputs (`src/tabfm_metrics_regression.cpp` lines 68-92)

```cpp
void RMSEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
    UnifiedVectorFormat sdata, actual_data, predicted_data;
    state_vector.ToUnifiedFormat(count, sdata);
    inputs[0].ToUnifiedFormat(count, actual_data);
    inputs[1].ToUnifiedFormat(count, predicted_data);
    auto states = reinterpret_cast<RMSEState **>(sdata.data);

    for (idx_t i = 0; i < count; i++) {
        idx_t sidx = sdata.sel->get_index(i);
        idx_t aidx = actual_data.sel->get_index(i);
        idx_t pidx = predicted_data.sel->get_index(i);

        if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
            continue;
        }

        auto &state     = *states[sidx];
        double a        = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];
        double p        = UnifiedVectorFormat::GetData<double>(predicted_data)[pidx];
        // ... compute metric ...
        state.n++;
    }
}
```

**CRITICAL DEVIATION for PSR aggregates:** `inputs[1]` is `STRUCT(logits DOUBLE[], borders DOUBLE[])`, NOT a plain DOUBLE. Do NOT use `UnifiedVectorFormat` for the STRUCT argument. Instead use `inputs[1].GetValue(i)` with `StructValue::GetChildren` + `ListValue::GetChildren` — see STRUCT reading pattern below. Still use `UnifiedVectorFormat` for `inputs[0]` (the DOUBLE `actual` argument) and for state/validity tracking.

---

#### STRUCT reading in Update — `GetValue(i)` idiom (`src/tabfm_metrics_classification.cpp` lines 460-490)

The MAP reading pattern is the direct analog for STRUCT(LIST, LIST) reading. The idiom is identical: `GetValue(i)` handles all vector representations (flat, constant, dictionary) uniformly.

```cpp
// MAP reading from tabfm_metrics_classification.cpp:466-482 — adapt for STRUCT:
Value proba_val = inputs[1].GetValue(i);
if (proba_val.IsNull()) { continue; }

for (auto &kv : MapValue::GetChildren(proba_val)) {
    auto &entry = StructValue::GetChildren(kv);
    // entry[0] = key, entry[1] = value
}
```

Adapted for `yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])`:

```cpp
Value dist_val = inputs[1].GetValue(i);
if (dist_val.IsNull()) { continue; }

auto &dist_children = StructValue::GetChildren(dist_val);
// dist_children[0] = logits LIST(DOUBLE)  — field order from tabfm_predict_agg.cpp:594-596
// dist_children[1] = borders LIST(DOUBLE)

if (dist_children[0].IsNull() || dist_children[1].IsNull()) { continue; }

auto &logit_vals  = ListValue::GetChildren(dist_children[0]);
auto &border_vals = ListValue::GetChildren(dist_children[1]);

const size_t K = logit_vals.size();
if (K == 0 || border_vals.size() != K + 1) { continue; }

std::vector<double> logits(K), borders(K + 1);
for (size_t k = 0; k < K; k++) {
    logits[k]  = DoubleValue::Get(logit_vals[k]);
    borders[k] = DoubleValue::Get(border_vals[k]);
}
borders[K] = DoubleValue::Get(border_vals[K]);
```

---

#### Combine pattern (`src/tabfm_metrics_regression.cpp` lines 94-106)

```cpp
void RMSECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
    UnifiedVectorFormat source_data, target_data;
    source_vector.ToUnifiedFormat(count, source_data);
    target_vector.ToUnifiedFormat(count, target_data);
    auto sources = reinterpret_cast<RMSEState **>(source_data.data);
    auto targets = reinterpret_cast<RMSEState **>(target_data.data);
    for (idx_t i = 0; i < count; i++) {
        auto &src  = *sources[source_data.sel->get_index(i)];
        auto &tgt  = *targets[target_data.sel->get_index(i)];
        tgt.sum_sq += src.sum_sq;
        tgt.n      += src.n;
    }
}
```

---

#### Finalize pattern — NULL-on-empty (`src/tabfm_metrics_regression.cpp` lines 108-123)

```cpp
void RMSEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                  idx_t offset) {
    UnifiedVectorFormat sdata;
    state_vector.ToUnifiedFormat(count, sdata);
    auto states = reinterpret_cast<RMSEState **>(sdata.data);
    for (idx_t i = 0; i < count; i++) {
        auto &state = *states[sdata.sel->get_index(i)];
        if (state.n == 0) {
            FlatVector::SetNull(result, i + offset, true);
        } else {
            FlatVector::GetData<double>(result)[i + offset] =
                std::sqrt(state.sum_sq / static_cast<double>(state.n));
        }
    }
}
```

For PSR aggregates replace `std::sqrt(...)` with `sum_score / static_cast<double>(n)`.

---

#### Bind — telemetry once per bind (`src/tabfm_metrics_regression.cpp` line 125-128)

```cpp
unique_ptr<FunctionData> RMSEBind(ClientContext &, AggregateFunction &,
                                   vector<unique_ptr<Expression>> &) {
    PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_rmse");
    return nullptr;
}
```

For `tabfm_interval_score`, bind also reads the `coverage` constant and stores it in `IScoreBindData`.

---

#### Bind-data pattern — reading constant at bind time (`src/tabfm_metrics_classification.cpp` lines 205-248)

```cpp
struct F1BindData : FunctionData {
    std::string avg;
    char        metric;

    F1BindData(std::string avg_mode, char m) : avg(std::move(avg_mode)), metric(m) {}

    unique_ptr<FunctionData> Copy() const override { return make_uniq<F1BindData>(avg, metric); }
    bool Equals(const FunctionData &other) const override {
        auto &o = other.Cast<F1BindData>();
        return avg == o.avg && metric == o.metric;
    }
};

// Reading a constant expression at bind time:
Value v = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
if (!v.IsNull()) { avg_mode = v.ToString(); }
```

For `IScoreBindData`, store `double coverage` extracted from `arguments[2]` (the `coverage` parameter). Validate `coverage > 0.0 && coverage < 1.0` at bind time.

Reading in Update:
```cpp
// src/tabfm_metrics_classification.cpp lines 335-364 (F1Finalize analog):
auto &bd = aggr_input.bind_data->Cast<IScoreBindData>();
double coverage = bd.coverage;
```

---

#### PSR-04 Bind-gate — 2-arg overload that always throws (`src/tabfm_metrics_classification.cpp` lines 1077-1088)

```cpp
AggregateFunction fn2(
    "anofox_tabfm_precision", {LogicalType::ANY, LogicalType::ANY}, LogicalType::DOUBLE,
    F1StateSize, F1StateInit, F1Update, F1Combine, F1Finalize,
    /*simple_update=*/nullptr,
    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
        -> unique_ptr<FunctionData> {
        throw InvalidInputException(
            "tabfm_precision: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'");
        return nullptr;
    },
    F1StateDestroy);
set.AddFunction(fn2);
```

**PSR-04 adaptation:** Register the STRUCT overload (`fn_dist`) FIRST, then the `DOUBLE` overload that throws. Error message must name the remedy: `"use output_mode := 'distribution' in tabfm_regress"`. Use `{LogicalType::DOUBLE, LogicalType::DOUBLE}` for the bind-gate overload.

---

#### AggregateFunctionSet registration (`src/tabfm_metrics_classification.cpp` lines 1159-1177)

```cpp
// tabfm_log_loss — one-overload aggregate with complex second argument:
AggregateFunctionSet set("anofox_tabfm_log_loss");
AggregateFunction fn("anofox_tabfm_log_loss",
                     {LogicalType::VARCHAR,
                      LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)},
                     LogicalType::DOUBLE, LogLossStateSize, LogLossStateInit, LogLossUpdate,
                     LogLossCombine, LogLossFinalize, /*simple_update=*/nullptr, LogLossBind,
                     /*state_destroy=*/nullptr);
set.AddFunction(fn);
FunctionDescription fd;
fd.description = "...";
fd.examples = {"SELECT tabfm_log_loss(actual, proba) FROM predictions;"};
RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_log_loss", {std::move(fd)});
```

**STRUCT type for PSR registration:**
```cpp
LogicalType yhat_dist_type = LogicalType::STRUCT({
    {"logits",  LogicalType::LIST(LogicalType::DOUBLE)},
    {"borders", LogicalType::LIST(LogicalType::DOUBLE)}
});
// input types: {LogicalType::DOUBLE, yhat_dist_type}
// Matches tabfm_predict_agg.cpp:594-596 field declaration order exactly.
```

---

#### SoftmaxInPlace — copy from `src/tabfm_engine.cpp` lines 649-665

```cpp
// NOT in external scope — anonymous namespace in tabfm_engine.cpp.
// Copy verbatim into anonymous namespace of tabfm_scoring.cpp:
void SoftmaxInPlace(std::vector<double> &v) {
    if (v.empty()) return;
    double m = v[0];
    for (auto x : v) { m = std::max(m, x); }
    double sum = 0.0;
    for (auto &x : v) { x = std::exp(x - m); sum += x; }
    if (sum > 0.0) { for (auto &x : v) { x /= sum; } }
}
```

`DistributionQuantile` (tabfm_predict.hpp line 206) IS externally linked — include `"tabfm_predict.hpp"` and call directly.

---

### `src/include/tabfm_scoring.hpp` (header)

**Analog:** any existing module header, e.g., `src/include/tabfm_metrics_regression.hpp`

Pattern: `#pragma once`, `namespace duckdb { namespace anofox {`, one declaration:
```cpp
void RegisterScoringFunctions(ExtensionLoader &loader);
```

---

### CMP-01 macro (new macro in `src/tabfm_crossval.cpp` or `src/tabfm_macros.cpp`)

**Analog:** `src/tabfm_crossval.cpp` lines 128-312 (CVMacroDef struct + BuildTableMacroFunction + RegisterCVMacroWithAlias)
**Secondary analog:** `src/tabfm_macros.cpp` lines 63-104 (PredictMacroDef struct, simpler pattern)

Decision for planner: because CMP-01 composes existing predict + metric + scoring aggregates via SQL (no new C++ compute logic), place the macro definition in `src/tabfm_crossval.cpp` (extend `RegisterCrossValidateMacros`) to avoid touching a second scaffold-owned file. If the planner prefers a clean module boundary, add a new `src/tabfm_compare.cpp` (requires one additional `CMakeLists.txt` and `tabfm_registration.hpp` entry).

**Safe identifier quoting (`src/tabfm_crossval.cpp` line 209):**
```cpp
replace(CAST(target AS VARCHAR), '"', '""')
```
Carry this forward at every target column interpolation in the CMP-01 macro body.

**CVMacroDef struct (`src/tabfm_crossval.cpp` lines 128-130):**
```cpp
static const CVMacroDef CROSS_VALIDATE_MACRO = {
    {"data", "target", "row_key", nullptr},          // positional params
    {"k=5", "seed=42", "task='classification'", "metric=NULL", nullptr},  // defaults
    R"( SELECT * FROM query( ... ) )",               // body
    "description...",
    "example..."
};
```

**Macro body using `query()` + `list_transform` + `array_to_string` (`src/tabfm_crossval.cpp` lines 169-237):** The CMP-01 macro body is simpler (no fold loop) but uses the same `query()` executor pattern.

**RegisterCVMacroWithAlias (`src/tabfm_crossval.cpp` lines 306-312):** handles full + short alias atomically.

---

### `test/cpp/test_tabfm_scoring.cpp` (Catch2 unit test)

**Analog:** `test/cpp/test_tabfm_metrics_regression.cpp`

Pattern: `#include "catch.hpp"`, `TEST_CASE("tabfm_scoring: CRPS synthetic K=4", "[tabfm_scoring]")`, hard-coded inputs and expected values with `REQUIRE(std::abs(result - expected) < 1e-9)`.

Use a synthetic K=4 uniform-width case for CRPS where the per-bin integral can be hand-verified. Use the K=16 golden.json fixture for end-to-end SQL test (not in C++ unit test).

---

### `test/sql/tabfm_scoring.test` (sqllogictest)

**Analog:** `test/sql/tabfm_distribution.test`

Pattern: `load tabfm`, `statement ok` / `query R` blocks, one block per PSR-01..04 requirement, one block for PSR-04 bind-gate (expect error with named remedy), one block for CMP-01 macro.

---

### Scaffold-owned modifications (coordinate — CLAUDE.md rule #2)

All three must be edited in a single coordinated batch. They are listed here for reference — no implementation details change compared to the Phase 1 pattern.

#### `CMakeLists.txt` — EXTENSION_SOURCES (lines 57-59)

```cmake
set(EXTENSION_SOURCES
    ...
    src/tabfm_metrics_classification.cpp
    src/tabfm_metrics_regression.cpp
    src/tabfm_crossval.cpp
    src/tabfm_scoring.cpp          # ADD
    ...
)
```

#### `CMakeLists.txt` — TABFM_CPP_TEST_SOURCES (lines 131-132)

```cmake
set(TABFM_CPP_TEST_SOURCES
    ...
    test/cpp/test_tabfm_metrics.cpp
    test/cpp/test_tabfm_metrics_regression.cpp
    test/cpp/test_tabfm_crossval.cpp
    test/cpp/test_tabfm_scoring.cpp   # ADD
    ...
)
```

#### `src/anofox_tabfm_extension.cpp` — LoadInternal (lines 123-125)

```cpp
anofox::RegisterClassificationMetrics(loader);
anofox::RegisterRegressionMetrics(loader);
anofox::RegisterCrossValidateMacros(loader);
anofox::RegisterScoringFunctions(loader);   // ADD
```

#### `src/include/tabfm_registration.hpp` — declarations (lines 16-18)

```cpp
void RegisterClassificationMetrics(ExtensionLoader &loader);
void RegisterRegressionMetrics(ExtensionLoader &loader);
void RegisterCrossValidateMacros(ExtensionLoader &loader);
void RegisterScoringFunctions(ExtensionLoader &loader);  // ADD
```

---

## Shared Patterns

### Telemetry — once per bind
**Source:** `src/tabfm_metrics_regression.cpp` line 127
**Apply to:** All three PSR aggregate Bind callbacks in `tabfm_scoring.cpp`
```cpp
PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_crps");
```

### AggregateFunctionSet + alias registration
**Source:** `src/tabfm_metrics_classification.cpp` lines 1159-1177
**Apply to:** All three PSR aggregates
```cpp
RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_crps", {std::move(fd)});
```

### Error messages name the remedy (SQL-API §5)
**Source:** `src/tabfm_metrics_classification.cpp` lines 1083-1084
**Apply to:** PSR-04 bind-gate overload for all three PSR aggregates
```cpp
throw InvalidInputException(
    "tabfm_crps: second argument must be a distribution "
    "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
    "Use output_mode := 'distribution' in tabfm_regress.");
```

### NULL-on-empty Finalize
**Source:** `src/tabfm_metrics_regression.cpp` lines 116-118
**Apply to:** All three PSR aggregate Finalize callbacks
```cpp
if (state.n == 0) { FlatVector::SetNull(result, i + offset, true); }
```

### namespace double-wrapping
**Source:** `src/tabfm_metrics_regression.cpp` lines 37-40
**Apply to:** `tabfm_scoring.cpp`
```cpp
namespace duckdb {
namespace anofox {
namespace {
// all implementation helpers
} // namespace (anonymous)
```

---

## No Analog Found

All files have adequate analogs. No file requires fallback to RESEARCH.md patterns from scratch.

---

## Metadata

**Analog search scope:** `src/`, `src/include/`, `test/cpp/`, `test/sql/`, `tools/parity/`
**Files scanned:** 9 source files read directly
**Pattern extraction date:** 2026-09-22
**Tracked-source gate:** All analog paths verified as git-tracked source files (not gitignored mirrors).
