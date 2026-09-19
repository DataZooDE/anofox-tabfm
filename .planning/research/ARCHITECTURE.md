# Architecture Patterns

**Project:** anofox-tabfm — Evaluation Framework + Multi-Model Milestone
**Researched:** 2026-09-19
**Source:** Direct codebase analysis (.planning/codebase/ documents + src/ headers)

---

## Executive Statement

The existing codebase is a tight 10-layer stack with strict module-per-file ownership and a clean virtual-dispatch engine seam (`PredictEngine` in `tabfm_predict.hpp`). The new milestone adds four capability clusters — metric aggregates, CV macro, preprocessing-profile dispatch, and regression distributions — each of which can be landed as its own self-contained module without touching the inner layers. The only scaffold-owned files that need a coordinated one-time edit are `tabfm_registration.hpp` (declare new `Register*` entry points) and `anofox_tabfm_extension.cpp` (call them). Every other file is greenfield.

---

## Recommended Architecture

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │              DuckDB SQL Frontend (unchanged macros)                     │
 │  tabfm_classify / tabfm_regress  ─────────────────────► (no change)   │
 │                                                                         │
 │  NEW (model-agnostic)                                                   │
 │  tabfm_accuracy / tabfm_f1 / tabfm_auc / tabfm_logloss …   ◄── L-AGG  │
 │  tabfm_rmse / tabfm_mae / tabfm_r2 …                        ◄── L-AGG  │
 │  tabfm_cv(data, target, k [, model] [, opts])               ◄── L-CV   │
 │  tabfm_crps / tabfm_logscore / tabfm_interval_score …       ◄── L-PSR  │
 └─────────────────────────┬───────────────────────────────────────────────┘
                           │
 ┌─────────────────────────▼───────────────────────────────────────────────┐
 │   NEW Layer 0b — Profile Registry  (tabfm_profile_registry.cpp)        │
 │   Manifest preprocessing_profile → C++ PreprocessProfile interface     │
 │   Registered at extension load; tabfm_v1_minimal is the built-in       │
 └──────┬──────────────────────────────────────────────────────────────────┘
        │ look up by manifest.preprocessing_profile
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │   Existing Layer 8 — Preprocessing Pipeline (tabfm_preprocess.cpp)     │
 │   PreprocessBatch() — no structural change; new profiles call their     │
 │   own Preprocess* functions registered against their profile id         │
 └──────┬───────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │   Existing Layer 7/10 — ORT Engine + Decode (tabfm_ort_engine.cpp,     │
 │   tabfm_engine.cpp)                                                     │
 │                                                                         │
 │   NEW: TabFMRunOutput gains optional quantile_logits[] / bar_dist[]    │
 │   fields (zero-size = not emitted by this model). Decode step reads     │
 │   them when present and adds yhat_q10/q50/q90 + dist fields to output. │
 └──────────────────────────────────────────────────────────────────────────┘
```

### Component Map: New vs Existing

| Layer | File | Status | Touches |
|-------|------|--------|---------|
| SQL metric aggregates | `src/tabfm_metrics.cpp` + `src/include/tabfm_metrics.hpp` | **New** | Own module |
| SQL CV macro | `src/tabfm_cv.cpp` | **New** | Own module |
| Proper scoring rules | `src/tabfm_scoring.cpp` + `src/include/tabfm_scoring.hpp` | **New** | Own module |
| Profile registry | `src/tabfm_profile_registry.cpp` + `src/include/tabfm_profile_registry.hpp` | **New** | Own module |
| TabPFN v2 preprocess | `src/tabfm_preprocess_tabpfnv2.cpp` + `.hpp` | **New** | Registers into profile registry |
| TabICL preprocess | `src/tabfm_preprocess_tabicl.cpp` + `.hpp` | **New** | Registers into profile registry |
| Regression distributions | Extend `TabFMRunOutput` in `tabfm_ort_engine.hpp`; decode in `tabfm_engine.cpp` | **Extend** | Coordinated change (engine seam) |
| Registration declaration | `src/include/tabfm_registration.hpp` | **Extend** | Scaffold-owned — one coordinated edit |
| Extension loader | `src/anofox_tabfm_extension.cpp` | **Extend** | Scaffold-owned — one coordinated edit |

---

## Component Boundaries

### L-AGG — `tabfm_metrics.cpp` / `tabfm_metrics.hpp`

**Responsibility:** Model-agnostic scalar and aggregate SQL functions for classification and regression metrics. Operates exclusively on plain `(actual, predicted[, proba])` columns — no dependency on TabFM internals, ORT, or the predict surface.

**Public interface:**

```cpp
// tabfm_metrics.hpp
namespace duckdb { namespace anofox {

// Aggregate state accumulating pairs of (actual, predicted) or (actual, proba_map)
struct MetricAccumState { ... };

// Called by each metric aggregate's Finalize step:
double ComputeAccuracy(const MetricAccumState &);
double ComputeF1(const MetricAccumState &, const string &average);   // "macro"/"weighted"/"binary"
double ComputeAUC(const MetricAccumState &);                          // one-vs-rest, proba required
double ComputeLogLoss(const MetricAccumState &);                      // proba required
double ComputeRMSE(const MetricAccumState &);
double ComputeMAE(const MetricAccumState &);
double ComputeR2(const MetricAccumState &);

// Registration entry point (declared in tabfm_registration.hpp):
void RegisterMetricsFunctions(ExtensionLoader &loader);

}} // namespace
```

**SQL surface:**

```sql
-- Classification (single-group aggregates, model-agnostic):
SELECT tabfm_accuracy(actual, predicted)           FROM results;
SELECT tabfm_f1(actual, predicted, 'macro')        FROM results;
SELECT tabfm_auc(actual, proba)                    FROM results;  -- proba is MAP(VARCHAR,DOUBLE)
SELECT tabfm_logloss(actual, proba)                FROM results;

-- Regression:
SELECT tabfm_rmse(actual, predicted)               FROM results;
SELECT tabfm_mae(actual, predicted)                FROM results;
SELECT tabfm_r2(actual, predicted)                 FROM results;
```

**Communicates with:** DuckDB aggregate API only. No dependency on any tabfm_*.hpp engine modules.

**Depends on:** DuckDB types (LogicalType, Value), DuckDB aggregate registration.

**Used by:** User queries directly; CV macro (`tabfm_cv`) invokes them inside its fold SQL.

---

### L-CV — `tabfm_cv.cpp`

**Responsibility:** k-fold cross-validation orchestration macro. Assigns fold indices to rows, then for each fold constructs a train partition (all other folds) and a test partition (the held-out fold), calls the predict aggregate internally, and calls the metric aggregates. Returns per-fold + aggregate result rows. Implemented as a SQL table macro (same pattern as `tabfm_classify`/`tabfm_regress`).

**Design constraint:** DuckDB macros expand to SQL, so the CV macro emits a SQL query that uses `tabfm_classify` or `tabfm_regress` (depending on task) per fold. The fold-assignment column is computed with a deterministic hash (`hash(rowid) % k` or a seeded LCG) as a computed column injected into the relation. The result is unnested and aggregated over folds.

**Macro signature:**

```sql
tabfm_cv(data, target [, k=5] [, model='tabfm-v1'] [, features=NULL] [, opts=MAP{}])
-- Returns: fold INTEGER, metric_name VARCHAR, metric_value DOUBLE (per-fold rows)
--          + a 'mean' and 'std' summary row per metric_name (aggregate=true flag)
```

**Fold assignment approach:** Inject `__fold := abs(hash(row_number() OVER ())) % k` as a derived column into the data relation. Train rows for fold f are `WHERE __fold != f`; test rows are `WHERE __fold = f` (with target forced NULL). The macro body constructs this via `query()` string assembly — same mechanism as `tabfm_classify`/`tabfm_regress`.

**Per-fold logic (SQL body sketch):**

```sql
-- For each fold f in 0..k-1 (generated via generate_series):
SELECT f, tabfm_accuracy(actual, predicted) AS accuracy, ...
FROM tabfm_classify(
  '(SELECT *, abs(hash(row_number() OVER ())) % ' || k || ' AS __fold FROM (' || data || ')) WHERE __fold != ' || f,
  target,
  test := '(... WHERE __fold = ' || f || ' ...)',
  opts := opts
)
```

**Communicates with:** `tabfm_classify` / `tabfm_regress` macros (calls them), metric aggregate functions (wraps them in the fold aggregation SQL).

**Depends on:** DuckDB macro registration, query string construction.

**Scaffold-owned interaction:** Adds `void RegisterCVMacros(ExtensionLoader &loader)` — declare in `tabfm_registration.hpp`, call from `anofox_tabfm_extension.cpp`.

---

### L-PSR — `tabfm_scoring.cpp` / `tabfm_scoring.hpp`

**Responsibility:** Proper scoring rules for probabilistic regression predictions. Operates on predictive distribution columns emitted by models that support them (TabPFN v2). Functions are model-agnostic — they consume distribution column types regardless of origin.

**Gating:** These functions compile and register unconditionally but throw `InvalidInputException` at bind time if fed non-distribution inputs (i.e., when called on output from a model that does not emit quantiles/bar-distribution). The error message names the remedy: `"tabfm_crps requires a model that emits predictive distributions. Use tabfm_regress with opts := MAP{'model':'tabpfn-v2'}."`

**SQL surface:**

```sql
SELECT tabfm_crps(actual, yhat_dist)          FROM results;  -- yhat_dist: DOUBLE[]
SELECT tabfm_logscore(actual, yhat_dist)       FROM results;
SELECT tabfm_interval_score(actual, yhat_q10, yhat_q90, alpha := 0.1) FROM results;
```

**Communicates with:** Metric aggregate layer (sibling). No engine dependency.

---

### L-PROF — `tabfm_profile_registry.cpp` / `tabfm_profile_registry.hpp`

**Responsibility:** Registry that maps a `preprocessing_profile` id (string from the model manifest) to a `PreprocessProfile` C++ interface. Decouples manifest parsing from preprocessing implementation, so new model families can register their own profiles without touching existing code.

**Interface:**

```cpp
// tabfm_profile_registry.hpp
namespace duckdb { namespace anofox {

struct PreprocessProfile {
    string id;
    // Pointer to the preprocessing function for this profile.
    // Same signature as PreprocessBatch but with a profile-specific impl.
    // Returns PreprocessedBatch (shared type — profiles must emit the same
    // contract so the ORT/decode layer is unchanged).
    PreprocessedBatch (*preprocess_fn)(const ColumnDataCollection &,
                                      const vector<PreprocessColumnSpec> &,
                                      PreprocessTask);
};

class PreprocessProfileRegistry {
public:
    static PreprocessProfileRegistry &Instance();
    void Register(PreprocessProfile profile);
    // Throws InvalidInputException if id not registered:
    const PreprocessProfile &Get(const string &id) const;
};

}} // namespace
```

**Built-in registration:** At extension load, `tabfm_profile_registry.cpp` calls `Instance().Register({kPreprocessProfileId, PreprocessBatch})` to wire the existing `tabfm_v1_minimal` profile.

**Existing code change:** `tabfm_engine.cpp` currently calls `PreprocessBatch()` directly. After this component lands, it calls `PreprocessProfileRegistry::Instance().Get(manifest.preprocessing_profile).preprocess_fn(...)` instead. This is a single-line change in `tabfm_engine.cpp` — but it is an integration file that touches the engine seam, requiring coordination with the WS-C/WS-F owners.

**New profile modules:**

- `src/tabfm_preprocess_tabpfnv2.cpp` — implements `PreprocessBatchTabPFNv2()` and registers `"tabpfn_v2_minimal"` at init.
- `src/tabfm_preprocess_tabicl.cpp` — implements `PreprocessBatchTabICL()` and registers `"tabicl_minimal"` at init.

Each new preprocess module declares a `void RegisterTabPFNv2Profile()` / `void RegisterTabICLProfile()` function; these are called from `anofox_tabfm_extension.cpp` (scaffold-owned; coordinate).

---

### Regression Distribution Path (cross-cutting extension)

**Problem:** `TabFMRunOutput` today carries only `logits` (a flat float vector) and `shape [1,T,C]`. For classification C = n_classes; for regression C = 1. TabPFN v2 and similar models emit either a bar-distribution (histogram over a discretized real axis) or explicit quantile values — both need to flow through the existing preprocess→ORT→decode path.

**Solution — minimal extension to `TabFMRunOutput`:**

```cpp
// In tabfm_ort_engine.hpp (existing file — coordinate with WS-C):
struct TabFMRunOutput {
    vector<float> logits;
    vector<int64_t> shape;   // [1, T, C]

    // NEW: populated only when the model's graph emits them.
    // bar_logits: [1, T, B] unnormalized bar-distribution (B histogram bins).
    // When empty, scoring-rule functions are blocked at bind time.
    vector<float> bar_logits;
    vector<int64_t> bar_shape;  // [1, T, B] or empty
    // quantile_levels: length Q — the probability levels (0.1, 0.5, 0.9 ...)
    vector<float> quantile_levels;
    // quantile_values: [T, Q] — the corresponding quantile estimates.
    vector<float> quantile_values;
};
```

**Decode step extension (tabfm_engine.cpp — existing file — coordinate with WS-C):**

When `bar_logits` is non-empty the decode step:
1. Applies softmax over the B bins → probability mass vector per row.
2. Computes the expected value (mean) as the scalar `yhat` (existing regression column — backward compatible).
3. Also emits `yhat_dist DOUBLE[]` (the softmax distribution over bins) and `yhat_q10/q50/q90 DOUBLE` (percentile estimates from the CDF).

**Output schema extension (tabfm_predict.hpp):**

```cpp
struct TabFMPredictResult {
    vector<Value> yhat;        // existing
    vector<Value> yhat_score;  // existing
    vector<Value> proba;       // existing

    // NEW: populated only when model emits predictive distributions.
    // yhat_dist: DOUBLE[] — bar-distribution probability mass per bin.
    // yhat_quantiles: MAP(DOUBLE,DOUBLE) — level → quantile value.
    vector<Value> yhat_dist;
    vector<Value> yhat_quantiles;
};
```

**SQL output extension (tabfm_macros.cpp — coordinate):** The `tabfm_regress` macro's body currently unnests exactly the fields produced by the aggregate. When `yhat_dist` and `yhat_quantiles` are present in the result struct (as NULLs for models that don't emit them), DuckDB's `unnest(max_depth:=3)` surfaces them. The macro body does not need to change if the aggregate's output schema is conditioned on `output_mode='distribution'` opt (opt parsed at bind time in `tabfm_predict_agg.cpp`).

---

## Data Flows

### Cross-Validation (CV) Data Flow

```
tabfm_cv('sales', 'revenue', k := 5)
  │
  ▼ (macro expansion)
  FOR fold IN 0..4:
    train_rel = "(SELECT *, abs(hash(row_number() OVER ())) % 5 AS __fold FROM (sales)) WHERE __fold != fold"
    test_rel  = "(... WHERE __fold = fold AND revenue IS NULL override)"
    │
    ▼
    tabfm_regress(train_rel, 'revenue', test := test_rel, opts)
    │  (existing predict path — no change)
    ▼
    yhat per test row
    │
    ▼
    tabfm_rmse(revenue_actual, yhat) over fold rows
    [tabfm_crps(revenue_actual, yhat_dist) over fold rows — if distribution model]
  │
  ▼ (UNION ALL across folds)
  (fold INTEGER, metric VARCHAR, value DOUBLE) rows
    + summary rows: (fold='mean', metric VARCHAR, value DOUBLE)
```

**Key constraint:** The fold loop is unrolled at SQL generation time (the macro emits k subqueries UNION ALL'd), not a SQL LOOP. DuckDB macros do not have iteration; the body must be a single SELECT. This means k is bounded in practice (the macro raises at bind for k > 20 to prevent query bloat).

### Regression Distribution Data Flow

```
tabfm_regress('data', 'y', opts := MAP{'model':'tabpfn-v2','output_mode':'distribution'})
  │
  ▼
__anofox_tabfm_predict_agg (bind: output_mode='distribution' parsed, output schema extended)
  │
  ▼
PredictEngine::Predict → tabfm_engine.cpp
  │
  ├─ PreprocessProfileRegistry.Get('tabpfn_v2_minimal').preprocess_fn(...)
  │    → PreprocessedBatch (same struct as today)
  │
  ├─ Run (ORT session for tabpfn-v2 graph)
  │    → TabFMRunOutput { logits (point est), bar_logits [T,B] }
  │
  └─ Decode:
       yhat         = mean over bar_distribution bins  (backward compat)
       yhat_dist    = softmax(bar_logits) → DOUBLE[]   (NEW)
       yhat_quantiles = CDF inversion at [0.1,0.5,0.9] (NEW)
  │
  ▼
TabFMPredictResult { yhat, yhat_score=NULL, yhat_dist, yhat_quantiles }
  │
  ▼
tabfm_crps(y_actual, yhat_dist)  ← proper scoring rule, operates on plain columns
```

---

## Suggested Build Order

The dependency graph drives the order. Each step is a shippable unit; later steps gate on earlier.

**Step 1 — Metric Aggregates (`tabfm_metrics.cpp`)**
- No dependency on any other new component.
- Deliverable: `tabfm_accuracy`, `tabfm_f1`, `tabfm_auc`, `tabfm_logloss`, `tabfm_rmse`, `tabfm_mae`, `tabfm_r2` as DuckDB aggregate functions.
- Test: `test/sql/metrics.test` (feed synthetic actual/predicted columns, assert values).
- Scaffold edit: add `RegisterMetricsFunctions` to `tabfm_registration.hpp` + call in extension loader.

**Step 2 — CV Macro (`tabfm_cv.cpp`)**
- Depends on: Step 1 (metric functions must exist for the macro body to parse).
- Depends on: Existing predict macros (no change needed there).
- Deliverable: `tabfm_cv(data, target, k)` table macro.
- Test: `test/sql/cv.test` (synthetic table with known labels, k=2, assert per-fold metric rows).
- Scaffold edit: add `RegisterCVMacros` to `tabfm_registration.hpp` + call in extension loader.

**Step 3 — Profile Registry (`tabfm_profile_registry.cpp`)**
- Depends on: Nothing new; only refactors the call site in `tabfm_engine.cpp`.
- Deliverable: `PreprocessProfileRegistry` singleton with `tabfm_v1_minimal` pre-registered. `tabfm_engine.cpp` rerouted through registry.
- Test: `test/cpp/test_tabfm_profile_registry.cpp` — confirm `tabfm_v1_minimal` round-trips through registry and produces same output as direct `PreprocessBatch()` call.
- Scaffold edit: `tabfm_engine.cpp` call site change (WS-C coordination); `tabfm_registration.hpp` + extension loader for registry init.

**Step 4 — TabPFN v2 Model Support (`tabfm_preprocess_tabpfnv2.cpp` + manifest + graph)**
- Depends on: Step 3 (profile registry must exist to register into).
- Depends on: WS-A tools must export the TabPFN v2 ONNX graph + tensor map.
- Deliverable: `tabpfn-v2` manifest registered, preprocessing profile registered, prediction works.
- Test: `test/sql/tabpfnv2_predict.test` (fixture model, synthetic data, assert output shape).

**Step 5 — Regression Distributions (`TabFMRunOutput` extension + decode)**
- Depends on: Step 4 (TabPFN v2 graph must emit bar_logits for this to be testable end-to-end).
- Delivers: `yhat_dist` and `yhat_quantiles` columns in `tabfm_regress` output (behind `output_mode='distribution'`).
- Scaffold edit: extend `TabFMRunOutput` in `tabfm_ort_engine.hpp` (WS-C coordination); extend decode in `tabfm_engine.cpp`; extend output schema in `tabfm_predict_agg.cpp` bind.

**Step 6 — Proper Scoring Rules (`tabfm_scoring.cpp`)**
- Depends on: Step 5 (distribution columns must be present for CRPS/log-score to be testable).
- Depends on: Step 1 (follows same aggregate registration pattern as metrics).
- Deliverable: `tabfm_crps`, `tabfm_logscore`, `tabfm_interval_score` aggregate functions.
- Test: `test/sql/scoring.test` (synthetic bar-distributions with known CRPS values).

**Step 7 — TabICL Model Support (`tabfm_preprocess_tabicl.cpp`)**
- Depends on: Step 3 (profile registry).
- Parallel with Step 4 (no mutual dependency; both register into the registry independently).

---

## Scaffold-Owned Files Requiring Coordination

Two files are scaffold-owned (CLAUDE.md rule #2) and must be edited in a single coordinated batch, not per-workstream:

| File | What changes | When |
|------|-------------|------|
| `src/include/tabfm_registration.hpp` | Add `RegisterMetricsFunctions`, `RegisterCVMacros`, `RegisterScoringFunctions`, `RegisterPreprocessProfileRegistry`, `RegisterTabPFNv2Profile`, `RegisterTabICLProfile` | Before any new module ships (Step 1) |
| `src/anofox_tabfm_extension.cpp` | Call each new `Register*` function in `LoadInternal()` | Matches above |

Two non-scaffold files require coordinated point edits (not owned exclusively by the new workstream):

| File | What changes | When |
|------|-------------|------|
| `src/tabfm_engine.cpp` | Reroute `PreprocessBatch()` call through `PreprocessProfileRegistry::Get()`; add bar_logits decode | Steps 3 and 5 |
| `src/include/tabfm_ort_engine.hpp` | Extend `TabFMRunOutput` with `bar_logits`, `bar_shape`, `quantile_*` | Step 5 |
| `src/tabfm_predict_agg.cpp` | Extend output schema computation at bind for `output_mode='distribution'` | Step 5 |

Recommendation: batch all `tabfm_registration.hpp` + `anofox_tabfm_extension.cpp` declarations in a single PR at Step 1, with stub implementations for Steps 4–7 (declaring but not yet calling the not-yet-written functions is fine; they are stubs until the module lands).

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Metric Functions Depending on TabFM Types

**What goes wrong:** `tabfm_accuracy` imports `tabfm_predict.hpp` or `tabfm_manifest.hpp` to introspect the model.

**Why bad:** Couples model-agnostic metrics to the model stack. Users can no longer run metrics on non-tabfm predictions.

**Instead:** Metrics operate purely on DuckDB Value types (`actual VARCHAR/INTEGER`, `predicted VARCHAR/INTEGER`, `proba MAP(VARCHAR,DOUBLE)`). No tabfm header included.

---

### Anti-Pattern 2: CV Macro Embedding Fold Logic in C++

**What goes wrong:** The CV macro is implemented as a C++ table function that drives a loop internally, rather than as a SQL macro that expands to a UNION ALL.

**Why bad:** A C++ loop that invokes the predict aggregate recursively forces an inner DuckDB execution context that is not how extension aggregates are designed to run. It also bypasses the executor's parallelism.

**Instead:** The CV macro body is SQL (same as `tabfm_classify`/`tabfm_regress`). It unrolls k subqueries at macro expansion time.

---

### Anti-Pattern 3: Bar-Distribution Fields Always Allocated

**What goes wrong:** `TabFMRunOutput.bar_logits` is always allocated (even as empty), causing every model's decode step to branch on a new field.

**Why bad:** Silent allocation overhead; confusion about whether a model "supports distributions."

**Instead:** `bar_logits.empty()` is the canonical "no distribution" signal. The proper scoring rule functions check at bind time whether the input column is non-NULL/non-empty, and throw with a clear message if not. The decode step checks `bar_logits.empty()` once and skips distribution decode for point-estimate models.

---

### Anti-Pattern 4: New Preprocessing Profiles Hardcoded in Manifest Parsing

**What goes wrong:** `tabfm_manifest.cpp` gains an `if (preprocessing_profile == "tabpfn-v2") { ... }` branch.

**Why bad:** Violates module ownership (manifest layer should not know about model-specific preprocessing); creates an unbounded if-chain as new models are added.

**Instead:** Manifest parsing validates that `preprocessing_profile` is a non-empty string (already the case). The engine looks up the profile in the registry at predict time. New model support adds a new registration call — zero changes to manifest parsing.

---

## Scalability Considerations

| Concern | Current (tabfm-v1 only) | After milestone |
|---------|------------------------|-----------------|
| Preprocessing dispatch | Direct call — zero overhead | Registry lookup (string map, one per predict) — negligible |
| CV fold count | n/a | k is bounded at bind (k > 20 → error); avoids k × model-load overhead (sessions are cached) |
| Distribution decode | n/a | bar_logits softmax is O(T × B), B ≈ 1000 bins for TabPFN v2 — small vs ORT forward pass |
| Metric computation | n/a | All metrics are single-pass aggregates; memory O(distinct_classes) for AUC/logloss |

---

## Sources

- `.planning/codebase/ARCHITECTURE.md` — layer map, data flow, seam contracts
- `.planning/codebase/STRUCTURE.md` — file ownership, scaffold files, where to add code
- `.planning/codebase/CONVENTIONS.md` — naming, error handling, registration pattern
- `src/include/tabfm_preprocess.hpp` — PreprocessedBatch, kPreprocessProfileId
- `src/include/tabfm_manifest.hpp` — ModelManifest.preprocessing_profile field
- `src/include/tabfm_predict.hpp` — PredictEngine seam, TabFMPredictResult, PredictContext
- `src/include/tabfm_ort_engine.hpp` — TabFMRunOutput, TabFMRunInput, TabFMSession
- `src/include/tabfm_state.hpp` — LoadedModel, TabFMModelCacheKey
- `src/include/tabfm_registration.hpp` — scaffold-owned registration declarations
- `src/tabfm_macros.cpp` — SQL macro body pattern (query() + unnest(max_depth:=3))
- `src/tabfm_engine.cpp` — integration layer, PreprocessBatch call site
- `.planning/PROJECT.md` — ScoringBench inspiration, constraints, out-of-scope decisions
