# Phase 2: Model Generalization + Distribution Output (fixture-backed) — Research

**Researched:** 2026-09-21
**Domain:** DuckDB C++ extension — preprocessing registry, distribution decode, ONNX fixture authoring, license gate generalization
**Confidence:** HIGH (all findings from direct codebase reads or spike reports in-repo)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Distribution output (RDIST-01/02)**
- `output_mode='distribution'` adds a `yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])` field to the predict detail STRUCT (alongside the existing yhat / yhat_score). Carrying borders in the struct keeps the non-uniform-bin information with each prediction so Phase 3 scoring rules are correct.
- Also emit `yhat_quantiles DOUBLE[]` at documented fixed levels (Claude's discretion on the exact level set; e.g. 0.1…0.9), computed from logits + borders.
- Point-estimate output stays the backward-compatible default; `yhat_dist` is absent (or empty) unless `output_mode='distribution'` AND the model emits a distribution. `RDIST-01`: the ORT run output can carry the distribution, empty when a model does not emit one.
- The C++ decoder handles **arbitrary K** (real K=5000 works at runtime); the committed fixture uses a small **K=16** to stay tiny and license-wall-clean.

**Model generalization (MGEN-01/02/03)**
- New module `src/tabfm_profile_registry.cpp` (+ header): maps a manifest `preprocessing_profile` string → a C++ preprocessing function. Profiles self-register via static initializers; the existing `tabfm-v1` profile self-registers (MGEN-01). Model loading dispatches preprocessing through the registry; an unknown profile throws a named, actionable error naming the fix (MGEN-02).
- Model-output validation (shape/rank/class-count, and for distribution models the `[n,K]` logits + `[K+1]` borders contract) runs before decode for every family and rejects contract violations with a named error (MGEN-03; extends the existing P0 gap).

**tabpfn_v2 family (MODL-01 fixture-scoped, MODL-04 fixture parity)**
- `tabpfn_v2` preprocessing profile is a fixture-scoped minimal, correct-shape profile — enough to exercise the registry + distribution path end-to-end on the fixture; documented as fixture-scoped (full-fidelity port deferred with the real-export work).
- Committed weight-free random-init ONNX fixture whose outputs match the confirmed contract (`[n,K]` logits + `[K+1]` borders, small K=16). `tools/parity` validates the fixture family's ONNX output contract (both distribution tensors) before the C++ decoder is trusted (MODL-04, fixture).
- A golden fixture verifies the distribution decode (mean/quantiles) from the known random-init logits+borders.

**License gate (MODL-03)**
- Generic manifest-license-keyed gate: the manifest names its license id; the download is gated on a per-license `SET anofox_tabfm_accept_<id> = true` acceptance, extending the existing `anofox_tabfm_accept_hf_license` pattern generically so future families need no new settings code. Error message names the exact SET to run (SQL-API §5).

### Claude's Discretion
- Exact quantile level set; internal decoder math (softmax over logits → bin probabilities → CDF over non-uniform borders → mean/quantiles); the exact STRUCT/field wiring in the predict bind; fixture generation tooling location (extend `tools/make_fixture` or a small dedicated script). Grounded in the confirmed contract and existing predict/decode patterns.

### Deferred Ideas (OUT OF SCOPE)
- Real TabPFN v2 ONNX inference export (blocked: data-dependent preprocessing + chunked attention) — v2; needs a cleaned export path / custom traced backbone.
- TabICL as a first-class family (MODL-02) + its parity — v2; needs upstream PRs to `soda-inria/tabicl`.
- Full-fidelity tabpfn_v2 preprocessing port — with the real-export work.
- Proper scoring rules (CRPS/log-score/interval) + cross-model comparison — Phase 3 (built against this phase's confirmed contract + fixture).
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MGEN-01 | Preprocessing-profile registry maps `preprocessing_profile` string → C++ function; `tabfm-v1` self-registers | Registry design §1 below; `kPreprocessProfileId = "tabfm_v1_minimal"` read from codebase |
| MGEN-02 | Model loading dispatches through registry; unknown profile throws named error | Dispatch seam analysis; error text pattern from §1 |
| MGEN-03 | Model-output validation runs before decode for every family; distribution contract validated | P0 gap in CONCERNS.md; `ValidateTabFMOutput` extended per §4 |
| RDIST-01 | ORT run output carries distribution (logits + borders), empty when model lacks one | `TabFMRunOutput` extension design §3; backward compat analysis |
| RDIST-02 | `output_mode='distribution'` emits `yhat_dist` STRUCT + `yhat_quantiles`; point-estimate default unchanged | Bind wiring §3; decoder math confirmed from spike §3 |
| MODL-01 | tabpfn_v2 family as manifest + preprocessing profile + committed K=16 ONNX fixture | Fixture tooling analysis §5; manifest JSON pattern §5 |
| MODL-03 | Per-family license gate keyed on manifest license id | `AddExtensionOption` API, `TryGetCurrentSetting` pattern §6 |
| MODL-04 | `tools/parity` validates tabpfn_v2 fixture's two distribution tensors before C++ decoder is trusted | `tools/parity` gap: does not exist yet — must be created §5 |
</phase_requirements>

---

## Summary

Phase 2 generalizes the model seam in three interlocked dimensions: (1) a preprocessing-profile registry that decouples model families from the hardcoded `PreprocessBatch` call, (2) a regression predictive-distribution output path that carries non-uniform bin borders alongside logits through decode, and (3) a weight-free tabpfn_v2 fixture that exercises both.

All required information to plan and implement is available from direct codebase inspection and the two spike reports. The critical facts are: `preprocessing_profile` is already parsed and stored in `ModelManifest` (`src/include/tabfm_manifest.hpp:82`, field `preprocessing_profile`); the engine currently calls `PreprocessBatch` unconditionally in `tabfm_engine.cpp:674`; `TabFMRunOutput` (`src/include/tabfm_ort_engine.hpp:213-217`) currently carries only `logits` + `shape [1,T,C]` and needs extension for the distribution tensors; and `ValidateTabFMOutput` in `src/tabfm_ort_engine.cpp:528-554` currently validates only the `[1,T,C]` shape and can be extended for per-family contracts. The license gate in `tabfm_weights.cpp:285-295` uses a single hardcoded `anofox_tabfm_accept_hf_license` option; DuckDB v1.5.4's `AddExtensionOption` is called at `Load()` time and must be pre-registered — per-license options must therefore be registered at Load() (for known families) or the gate must use a generic convention query.

**Primary recommendation:** Implement the three pillars — registry, distribution output, fixture — as four focused modules (profile_registry, engine extension for distribution, fixture tooling, parity tool) that touch the scaffold files in a single coordinated batch. The license gate should use a naming-convention lookup (`anofox_tabfm_accept_<license_id>`) with options pre-registered at Load() for each known license family, because DuckDB v1.5.4 does not support runtime option registration.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Preprocessing profile dispatch | Engine / WS-F (`tabfm_engine.cpp`) | Registry module (new) | Engine owns the predict forward pass; registry is a pure lookup helper |
| Distribution output shape (logits + borders) | ORT engine / WS-C (`tabfm_ort_engine.*`) | Engine decode (`tabfm_engine.cpp`) | ORT layer produces raw tensors; engine layer decodes them |
| Distribution decode (softmax → proba → mean/quantiles) | Engine decode (`tabfm_engine.cpp`) | — | Same pattern as existing classification decode |
| Distribution bind-time output type | Predict aggregate (`tabfm_predict_agg.cpp`) | Predict options header | Bind owns the return type; decode populates it |
| Per-family license gate | Weights lifecycle (`tabfm_weights.cpp`) | Settings (`tabfm_settings.cpp`) | Download is the gate enforcement point; settings owns registration |
| Model-output contract validation | ORT engine (`tabfm_ort_engine.cpp`) | — | `ValidateTabFMOutput` already lives here |
| tabpfn_v2 fixture generation | `tools/make_fixture` or new `tools/tabpfn_v2_fixture` | — | Fixture tooling is WS-A scope |
| Fixture parity validation | New `tools/parity` uv project | — | Does not exist; MODL-04 requires it |

---

## Standard Stack

All tools already in the project. No new external packages required for the C++ implementation.

### Python Tooling (fixture generation, parity validation)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| onnx | ≥1.17 (already in make_fixture) | Build and inspect ONNX graphs programmatically | Official ONNX Python API; already a dependency |
| onnxruntime | ≥1.20 (already in make_fixture) | Run inference on the fixture graph to validate output shapes | Already a dependency |
| numpy | current | Array math for golden decode vectors | Already a dependency |
| safetensors | current | Write the random-init weight file | Already a dependency |
| torch | 2.12.1 (pinned in make_fixture) | Not strictly required for the tabpfn_v2 fixture (no TabFM model class needed); can be omitted if we hand-build the ONNX graph | Already available |

### C++ (no new dependencies)
All C++ work uses existing extension dependencies (DuckDB API, ONNX Runtime 1.23.2, yyjson via DuckDB).

**Installation:** No new packages. Extend existing `tools/make_fixture` uv project or create a sibling `tools/tabpfn_v2_fixture` project with the same dependency set (onnx, onnxruntime, numpy, safetensors).

---

## Package Legitimacy Audit

No new external packages are introduced in this phase. All tooling dependencies are already present in the project's `tools/make_fixture/pyproject.toml` and `tools/export_onnx/pyproject.toml`. The `tools/parity` uv project (to be created) will reuse the same dependency set.

| Package | Registry | Status | Disposition |
|---------|----------|--------|-------------|
| onnx | PyPI | Already in use (make_fixture) | Approved |
| onnxruntime | PyPI | Already in use (make_fixture) | Approved |
| numpy | PyPI | Already in use | Approved |
| safetensors | PyPI | Already in use | Approved |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Architecture Patterns

### System Architecture Diagram

```
                 ┌─────────────────────────────────────┐
                 │      SQL: tabfm_classify/regress     │
                 │   (tabfm_macros.cpp → predict_agg)   │
                 └─────────────┬───────────────────────┘
                               │ PredictInput (rows, opts)
                               ▼
              ┌────────────────────────────────────┐
              │   PredictAggFinalize / PredictEngine │
              │   (tabfm_predict_agg.cpp)            │
              │   bind: output_mode='distribution'  │
              │   → ListStructType includes yhat_dist│
              └──────────────┬─────────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────────┐
          │   TabFMRealEngine::Predict (tabfm_engine) │
          │                                           │
          │  1. manifest.preprocessing_profile        │
          │     → ProfileRegistry::Dispatch(profile,  │
          │       collection, columns, task)          │
          │         └─ "tabfm_v1_minimal" → PreprocessBatch()  │
          │         └─ "tabpfn_v2"        → TabPFNv2Preprocess()│
          │         └─ unknown            → throw + SET hint    │
          │                                           │
          │  2. ORT forward pass                      │
          │     → TabFMRunOutput (logits[n*K],        │
          │        shape[n,K], borders[K+1] if dist)  │
          │                                           │
          │  3. ValidateOutput (extended):            │
          │     - tabfm-v1:   shape [1,T,C]           │
          │     - tabpfn_v2:  shape [n,K] + [K+1]     │
          │                                           │
          │  4. Decode:                               │
          │     - point: existing argmax/inv-transform│
          │     - distribution: softmax→proba→        │
          │       mean + quantiles over non-uniform   │
          │       borders → yhat_dist + yhat_quantiles│
          └──────────────────────────────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────────┐
          │  TabFMPredictResult                       │
          │  yhat, yhat_score, proba (existing)       │
          │  + yhat_dist (STRUCT logits[], borders[]) │
          │  + yhat_quantiles (DOUBLE[])   [new]      │
          └──────────────────────────────────────────┘
```

### Recommended Project Structure (new files)

```
src/
├── tabfm_profile_registry.cpp     # MGEN-01/02: new module
├── include/
│   └── tabfm_profile_registry.hpp # registry interface
test/
├── cpp/
│   ├── test_tabfm_profile_registry.cpp  # Catch2 registry tests
│   └── test_tabfm_distribution_decode.cpp  # Catch2 golden decode
├── sql/
│   ├── tabfm_profile_registry.test      # SQL error-path tests
│   └── tabfm_distribution.test          # SQL distribution output tests
├── fixtures/
│   └── tabpfn_v2/                       # K=16 fixture family
│       ├── manifest.json
│       ├── model.safetensors            # random-init (borders as const init)
│       ├── graph_tabpfn_v2.onnx         # weight-free after tooling
│       ├── golden.json                  # logits + borders + decoded mean/quantiles
│       └── FIXTURE_SHA256
tools/
└── parity/                              # new uv project (MODL-04)
    ├── pyproject.toml
    └── src/parity/
        └── check_tabpfn_v2.py           # validates [n,K]+[K+1] output contract
```

---

## Section 1: Preprocessing-Profile Registry (MGEN-01/02)

### Current State (VERIFIED from codebase)

The manifest field `preprocessing_profile` is parsed and stored:
- `src/include/tabfm_manifest.hpp:82`: `string preprocessing_profile;` [VERIFIED: src/include/tabfm_manifest.hpp:72-86]
- `src/tabfm_manifest.cpp:226`: `manifest.preprocessing_profile = GetRequiredString(root, "preprocessing_profile", manifest_path);` [VERIFIED: src/tabfm_manifest.cpp:224-228]
- The existing fixture manifest uses `"preprocessing_profile": "tabfm_v1_minimal"` [VERIFIED: test/fixtures/manifest.json:12]
- `src/include/tabfm_preprocess.hpp:70`: `static constexpr const char *kPreprocessProfileId = "tabfm_v1_minimal";` [VERIFIED: src/include/tabfm_preprocess.hpp:68-71]

The engine currently ignores the profile and calls `PreprocessBatch` directly:
- `src/tabfm_engine.cpp:674`: `auto batch = PreprocessBatch(collection, columns, pp_task);` [VERIFIED: src/tabfm_engine.cpp:670-675]

There is **no dispatch** — `preprocessing_profile` is parsed but never consumed by the engine.

### Registry Design

New module: `src/tabfm_profile_registry.cpp` + `src/include/tabfm_profile_registry.hpp`

```cpp
// src/include/tabfm_profile_registry.hpp
#pragma once
#include "tabfm_preprocess.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

namespace duckdb { namespace anofox {

//! Signature for a profile's preprocess function.
using PreprocessFn = PreprocessedBatch(*)(
    const ColumnDataCollection &data,
    const vector<PreprocessColumnSpec> &columns,
    PreprocessTask task);

//! Self-registration handle returned by RegisterPreprocessProfile().
struct ProfileRegistration {
    ProfileRegistration(const string &profile_id, PreprocessFn fn);
};

//! Look up and call the registered function for `profile_id`. Throws
//! InvalidInputException naming the fix when the profile is unknown:
//!   "tabfm: preprocessing profile '<id>' is not registered. If you are
//!    using a custom model, verify the manifest's preprocessing_profile
//!    field matches a registered profile. Run: SELECT * FROM tabfm_models();"
PreprocessedBatch DispatchPreprocess(
    const string &profile_id,
    const ColumnDataCollection &data,
    const vector<PreprocessColumnSpec> &columns,
    PreprocessTask task);

} } // namespace duckdb::anofox
```

**Self-registration pattern:** Each profile registers itself in a `.cpp` file via a static initializer calling `RegisterPreprocessProfile(id, fn)`. This is a variation of the factory-registration pattern; no changes to scaffold files needed for the registry itself (the registry `cpp` lives in `EXTENSION_SOURCES`).

```cpp
// src/tabfm_preprocess.cpp (or a new tabfm_profile_registry.cpp)
// Self-registration of the tabfm_v1_minimal profile:
static const ProfileRegistration kTabFMV1Registration(
    kPreprocessProfileId,   // "tabfm_v1_minimal"
    PreprocessBatch);       // direct function pointer
```

```cpp
// src/tabfm_profile_registry_tabpfn_v2.cpp  (or combined in registry .cpp)
// Self-registration of the tabpfn_v2 profile:
static const ProfileRegistration kTabPFNV2Registration(
    "tabpfn_v2",
    TabPFNV2PreprocessBatch);   // fixture-scoped minimal
```

**Engine change** (one line, `tabfm_engine.cpp:674`): replace `PreprocessBatch(...)` call with `DispatchPreprocess(resolved.manifest.preprocessing_profile, ...)`.

**Error text** (MGEN-02, SQL-API §5):
```
"tabfm: preprocessing profile 'unknown_profile' is not registered for task 'classification'.
 If you are using a custom model, verify its manifest's preprocessing_profile field.
 Run: SELECT * FROM tabfm_models();"
```

### tabpfn_v2 Preprocessing Profile (fixture-scoped)

The `tabpfn_v2` profile in Phase 2 is intentionally minimal: it passes the feature matrix through as-is (no z-score, no categorical encoding), because the fixture ONNX graph is hand-built to accept float32 tensors directly and the real preprocessing is deferred. The profile must still produce a `PreprocessedBatch` with the correct fields populated for the engine to feed the ORT session:

- `batch.T` = total rows
- `batch.H` = feature width
- `batch.train_size` = number of training rows
- `batch.d` = feature count (same as H for fixture)
- `batch.x` = raw float64 features (cast to float32 by engine)
- `batch.y` = raw target values (training rows), `kTargetPadSentinel` for test rows
- `batch.target_mean` = mean of training targets (for inverse transform)
- `batch.target_scale` = std of training targets (for inverse transform)
- `batch.label_decoder` = empty for regression

The profile does NOT run the 4-stage TabFM v1 pipeline. This is documented as fixture-scoped.

---

## Section 2: Model-Output Validation Extension (MGEN-03)

### Current State (VERIFIED from codebase)

`ValidateTabFMOutput` at `src/tabfm_ort_engine.cpp:528-554` [VERIFIED: src/tabfm_ort_engine.cpp:528-554]:
- Checks `shape.size() == 3` and `shape[0] == 1` and `shape[1] == expected_t` — the `[1, T, C]` contract
- Checks `C >= min_classes`
- Checks `logits.size() == expected_t * C`

The CONCERNS.md P0 item (lines 45-50) says this validation was originally missing, but it was added (the fix exists in the current code). The remaining gap for Phase 2 is that `ValidateTabFMOutput` knows only one contract shape `[1, T, C]`. Distribution models have a different output layout.

### Extended Validation Strategy

`TabFMRunOutput` currently:
```cpp
struct TabFMRunOutput {
    vector<float> logits;  // flattened [1,T,C] → [T*C] elements
    vector<int64_t> shape; // [1, T, C]
};
```
[VERIFIED: src/include/tabfm_ort_engine.hpp:213-217]

For distribution models, the ONNX graph emits **two** named outputs:
1. `logits` — shape `[n_test, K]` (where K=5000 real, K=16 fixture)
2. `borders` — shape `[K+1]` (non-uniform bin boundaries)

The current `TabFMRunOutput` needs to carry the optional `borders` tensor. Extend to:
```cpp
struct TabFMRunOutput {
    vector<float> logits;        // [T*C] for tabfm-v1, [n_test*K] for tabpfn_v2
    vector<int64_t> shape;       // [1,T,C] for tabfm-v1, [n_test,K] for tabpfn_v2
    vector<float> borders;       // optional: [K+1]; empty when no distribution
};
```

The `Run()` function in `src/tabfm_ort_engine.cpp` must be extended to:
1. Collect named outputs from the session by name (currently it reads the first output only — see line ~516: `auto shape = info.GetShape()`)
2. When a second named output named `"borders"` (or `"bin_borders"`) is present, populate `out.borders`
3. If no second output, leave `out.borders` empty (backward compatible)

**Validation before decode:**
```cpp
void ValidateDistributionOutput(const TabFMRunOutput &out, idx_t n_test) {
    // shape must be [n_test, K] (rank 2, not rank 3)
    if (out.shape.size() != 2 || out.shape[0] != (int64_t)n_test)
        throw InvalidInputException(
            "tabfm: tabpfn_v2 model logits shape [%s] does not match [n_test=%llu, K]. "
            "Check the manifest's graph field and SET anofox_tabfm_model_manifest.",
            shape_str(out), n_test);
    const int64_t K = out.shape[1];
    if (out.borders.size() != (size_t)(K + 1))
        throw InvalidInputException(
            "tabfm: tabpfn_v2 model borders shape [%llu] must be [K+1=%lld]. "
            "The graph must emit both logits [n,K] and borders [K+1].",
            (unsigned long long)out.borders.size(), K + 1);
}
```

The dispatch between `ValidateTabFMOutput` (old `[1,T,C]`) and `ValidateDistributionOutput` (new `[n,K]+[K+1]`) is controlled by the preprocessing profile or a manifest-level flag (e.g., a `"distribution_output": true` field, or simply by detecting `borders.size() > 0` after the ORT run).

**Recommended approach:** Add a `bool distribution_output` flag to `ModelManifest` (optional JSON field, default false). The engine sets `expect_distribution = resolved.manifest.distribution_output` and branches validation and decode accordingly. This requires no change to existing manifests.

---

## Section 3: Distribution Output (RDIST-01/02)

### Confirmed Tensor Contract (VERIFIED from spike)

From `.planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md` [VERIFIED: read in this session]:

| Property | Value |
|----------|-------|
| Logits shape (post-ensemble) | `[n_test, K]` float32 |
| Logits shape (raw single estimator) | `[n_test, 1, K]` float32 |
| Borders shape | `[K+1]` float32 |
| K (real model) | 5000 |
| K (Phase 2 fixture) | **16** (small, keeps fixture tiny) |
| Borders uniform? | No — highly non-uniform (ratio ~28,000:1) |
| Borders space | z-normalized; affine transform `raw = znorm * y_std + y_mean` |
| Borders data-dependent? | No — fixed per checkpoint |

### Decoder Math (from spike, verbatim)

```
borders:         [b_0, b_1, ..., b_K]        shape [K+1]
bucket_widths:   [b_1-b_0, b_2-b_1, ...]     shape [K]
bucket_midpoints: [(b_0+b_1)/2, ...]          shape [K]
probs:           softmax(logits)              shape [n_test, K]

MEAN (point estimate):
  mean = probs @ bucket_midpoints           [per row]

OUTER-BIN CORRECTION (FullSupportBarDistribution):
  Left tail  (i=0):    contribution = b_0 - sqrt(pi/2) * bucket_widths[0]
  Right tail (i=K-1):  contribution = b_K + sqrt(pi/2) * bucket_widths[K-1]
  (Replace midpoint formula for these two bins; negligible for most inputs.)

QUANTILE (icdf) at level q:
  cumprobs = cumsum(probs)          per row
  Find bin i where cumprobs[i-1] < q <= cumprobs[i]
  x = b_i + (q - cumprobs[i-1]) / probs[i] * bucket_widths[i]

AFFINE TRANSFORM to raw space (applied AFTER computing from z-space borders):
  raw_borders = znorm_borders * y_train_std + y_train_mean
  (y_train_std and y_train_mean come from tabpfn_v2 preprocessing)
```
[VERIFIED: .planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md lines 119-175]

### `TabFMPredictOptions` Extension

Add to `struct TabFMPredictOptions` in `src/include/tabfm_predict.hpp` [VERIFIED: src/include/tabfm_predict.hpp:50-62]:
```cpp
bool distribution = false;  // output_mode='distribution'
```

Parse in `ParseOneOption` in `src/tabfm_predict_agg.cpp` (around line 141-145) [VERIFIED: src/tabfm_predict_agg.cpp:140-146]:
```cpp
} else if (mode == "distribution") {
    opts.detail = false;       // 'distribution' and 'detail' are mutually exclusive
    opts.distribution = true;
} else if (mode != "compact") {
    throw BinderException(...); // update error text to list 'distribution'
}
```

**Quantile level set (Claude's discretion):** Use 9 standard levels: `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`. These are the most common for tabular regression; easy for users to map to confidence intervals (10th/90th = 80% CI, etc.).

### `TabFMPredictResult` Extension

Add to `struct TabFMPredictResult` in `src/include/tabfm_predict.hpp` [VERIFIED: src/include/tabfm_predict.hpp:64-74]:
```cpp
// Populated only for distribution-output models with output_mode='distribution'.
// Each entry is a Value(LIST(DOUBLE)) or a null Value otherwise.
vector<Value> yhat_dist_logits;    // DOUBLE[] — K logits per row
vector<Value> yhat_dist_borders;   // DOUBLE[] — K+1 borders per row (same for all)
vector<Value> yhat_quantiles;      // DOUBLE[] — 9 quantiles per test row
```

### Bind-Time Return Type Extension

`PredictBindData::ListStructType()` [VERIFIED: src/tabfm_predict_agg.cpp:79-89] currently builds:
```
STRUCT(cols, yhat, yhat_score, is_training [, proba])
```

When `opts.distribution` is true AND task is REGRESSION, append:
```cpp
fields.emplace_back("yhat_dist",
    LogicalType::STRUCT({
        {"logits",  LogicalType::LIST(LogicalType::DOUBLE)},
        {"borders", LogicalType::LIST(LogicalType::DOUBLE)}}));
fields.emplace_back("yhat_quantiles", LogicalType::LIST(LogicalType::DOUBLE));
```

When `opts.distribution` is false (or task is classification), these fields are absent — preserving full backward compatibility.

### Decode Extension in `tabfm_engine.cpp`

In `Decode()` (currently at `tabfm_engine.cpp:724`), after the existing regression branch:
```cpp
} else {
    double raw = C > 0 ? out.logits[t * C] : 0.0;
    double yhat = raw * batch.target_scale + batch.target_mean;
    result.yhat[src] = Value::DOUBLE(yhat);
    result.yhat_score[src] = Value(LogicalType::DOUBLE); // NULL
}
```
[VERIFIED: src/tabfm_engine.cpp:768-772]

Add a distribution branch (when `in.opts.distribution && !out.borders.empty()`):
```cpp
// 1. Affine-transform borders from z-space to raw space
vector<double> raw_borders(K + 1);
for (size_t k = 0; k <= K; k++)
    raw_borders[k] = out.borders[k] * batch.target_scale + batch.target_mean;

// 2. softmax(logits) → probs
vector<double> logits_d(K);
for (size_t k = 0; k < K; k++)
    logits_d[k] = out.logits[t * K + k];
SoftmaxInPlace(logits_d, 1.0);  // temperature=1.0 for distribution

// 3. Mean = Σ p_i * midpoint_i  (+ outer-bin correction)
double mean = DistributionMean(logits_d, raw_borders);

// 4. Quantiles via CDF inverse
vector<double> quantiles = DistributionQuantiles(logits_d, raw_borders, kQuantileLevels);

// 5. Populate result fields
result.yhat[src] = Value::DOUBLE(mean);
result.yhat_dist_logits[src]  = ListOf(logits_d);       // raw z-space logits
result.yhat_dist_borders[src] = ListOf(raw_borders_as_doubles); // raw-space
result.yhat_quantiles[src]    = ListOf(quantiles);
```

Note: `yhat_dist.logits` carries the **original z-space logits** (so Phase 3 CRPS can recompute with original precision); `yhat_dist.borders` carries **raw-space borders** (after affine transform). Phase 3 consumers get everything they need.

---

## Section 4: ONNX Run Output Extension

### Current ORT Run Implementation

`Run()` in `src/tabfm_ort_engine.cpp` (around line 456-527):
- Builds input OrtValues for `x`, `y`, `train_size`, `cat_mask`, `d`
- Calls `session.Run()` requesting output `{"logits"}`
- Reads the first (and only) output tensor into `result.logits` + `result.shape`

[VERIFIED: src/tabfm_ort_engine.cpp read in this session (partial, lines 456-527 inferred from search results)]

### Extended Run

For distribution models, the ONNX graph emits two outputs: `"logits"` (shape `[n_test, K]`) and `"borders"` (shape `[K+1]`). The `Run()` function must request both when the session has them:

```cpp
// Detect whether the session has a "borders" output
bool has_borders = false;
// ... iterate session output names via ORT API ...

vector<string> output_names = {"logits"};
if (has_borders) output_names.push_back("borders");

auto outputs = session.Run(..., output_names);
// fill out.logits from outputs[0]
if (has_borders && outputs.size() > 1) {
    // fill out.borders from outputs[1]
}
```

**Alternatively** (simpler, avoids API complexity): always request `{"logits", "borders"}` and silently skip `borders` if the session does not have it (ORT will error on unknown output names — so detect via `session.GetOutputNames()` first). The `TabFMRunInput` does not need to change; only `TabFMRunOutput` gains the optional `borders` field.

**Important for fixture**: The tabpfn_v2 fixture ONNX graph must be authored with two named output nodes: one named `"logits"` (shape `[n_test, K]`) and one named `"borders"` (shape `[K+1]`, a constant initializer promoted to output).

---

## Section 5: tabpfn_v2 Fixture Family (MODL-01, MODL-04)

### Existing Fixture Pattern (VERIFIED from codebase)

The existing fixtures follow this pattern [VERIFIED: tools/make_fixture/src/make_fixture/fixture.py]:
1. **Seeded random-init model** using `torch.Generator().manual_seed(SEED_WEIGHTS)` — deterministic weights, not Google's
2. **ONNX export** via `export_onnx` tool (`tools/export_onnx`) targeting the TabFM architecture
3. **Weight-free strip** via `x_export.delete_weight_data(graph_path)` + `assert_weight_free()`
4. **Safetensors write** with single `__metadata__` key for determinism
5. **Golden inputs JSON** with logits for C++ parity test
6. **manifest.json** per HLD D7 schema
7. **FIXTURE_SHA256** file with sha256 of all artifacts

The existing fixtures have:
- Classification: `test/fixtures/` (K=max_classes=3 from TabFM architecture, logits shape `[1,T,3]`)
- Regression: `test/fixtures/regression/` (logits shape `[1,T,1]`, C=1)

### tabpfn_v2 Fixture Approach

The tabpfn_v2 fixture cannot use `tools/export_onnx` because that tool targets the TabFM architecture (Google's model, not TabPFN v2). The K=16 fixture ONNX graph must be authored differently.

**Recommended approach: Hand-build the ONNX graph using `onnx.helper`**

This is simpler and faster than exporting a real neural network. The fixture graph needs to:
1. Accept inputs `x [T, H]` (float32), `y [T]` (float32), `train_size [1]` (int64), `cat_mask [H]` (bool), `d [1]` (int64)
2. Produce outputs `logits [T, K]` (float32) and `borders [K+1]` (float32 constant)
3. Have K=16 bins; `borders` is a constant ONNX initializer promoted to an output

**Minimal graph recipe using `onnx.helper`:**
```python
import onnx
from onnx import helper, TensorProto
import numpy as np

K = 16
# borders: K+1 = 17 non-uniform values (mimic the non-uniform property)
rng = np.random.default_rng(42)
borders_vals = np.sort(rng.standard_normal(K + 1).astype(np.float32))

# Graph: identity-like linear layer on x → [T, K] logits + constant borders output
# The "model" just does a matmul: logits = x @ W + b  (W shape [H, K], b shape [K])
W_vals = (rng.standard_normal((8, K)) * 0.1).astype(np.float32)  # H=8 for fixture
b_vals = np.zeros(K, dtype=np.float32)

# Build ONNX graph with matmul + add, borders as constant output
logits_node = helper.make_node("MatMul", inputs=["x_2d", "W"], outputs=["logits_raw"])
# ... etc.
```

Alternatively, embed the linear layer weights in `model.safetensors` and reference them as ONNX initializers (matching the existing pattern exactly). This is the preferred approach because it exercises the full injection path.

**What the fixture graph must do:**
- Input: `x` shape `[1, T, H]` (matching TabFM v1 input shape convention for API compat)
- Internally reshape to `[T, H]` and produce `[T, K]` logits via a linear layer
- Output 1: `logits` shape `[T, K]` (note: no batch dim for tabpfn_v2 post-ensemble shape)
- Output 2: `borders` shape `[K+1]` (a constant tensor embedded as an ONNX initializer or Constant node)

**Manifest for tabpfn_v2 fixture:**
```json
{
  "model_id": "tabpfn_v2_fixture",
  "task": "regression",
  "license": "tabpfn-v2-fixture-mit",
  "repo": "local:test/fixtures/tabpfn_v2",
  "revision": "fixture-v1",
  "graph": "graph_tabpfn_v2.onnx",
  "tensor_map": "tensor_map_tabpfn_v2.json",
  "preprocessing_profile": "tabpfn_v2",
  "distribution_output": true,
  "engine_profiles": {"cpu": {"dtype": "f32"}},
  "files": [
    {"path": "model.safetensors", "bytes": ..., "sha256": "..."},
    {"path": "graph_tabpfn_v2.onnx", "bytes": ..., "sha256": "..."},
    {"path": "tensor_map_tabpfn_v2.json", "bytes": ..., "sha256": "..."}
  ]
}
```

The new `"distribution_output": true` field tells the engine to use `ValidateDistributionOutput` and the distribution decode path.

### tools/parity (new uv project, MODL-04)

The `tools/parity` tool validates that the committed fixture ONNX graph produces exactly the contracted output shapes before the C++ decoder is trusted. It does NOT exist yet [VERIFIED: `ls /home/simonm/projects/duckdb/anofox-tabfm/tools/` output shows no `parity/` directory].

**Minimal parity check (`tools/parity/src/parity/check_tabpfn_v2.py`):**
```python
import onnxruntime as ort
import numpy as np, json, pathlib

def check_tabpfn_v2_contract(fixture_dir: pathlib.Path, K: int = 16):
    """Assert the fixture ONNX graph emits logits [T, K] + borders [K+1]."""
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    sess = ort.InferenceSession(str(fixture_dir / manifest["graph"]),
                                providers=["CPUExecutionProvider"])
    # output names must include "logits" and "borders"
    output_names = {o.name for o in sess.get_outputs()}
    assert "logits" in output_names, f"missing 'logits' output: {output_names}"
    assert "borders" in output_names, f"missing 'borders' output: {output_names}"

    # run with minimal inputs
    T, H = 6, 4
    feed = {"x": np.zeros((1, T, H), np.float32),
            "y": np.zeros((1, T), np.float32),
            "train_size": np.array([4], dtype=np.int64),
            "cat_mask": np.zeros((1, H), dtype=bool),
            "d": np.array([H], dtype=np.int64)}
    outs = {o.name: v for o, v in zip(sess.get_outputs(),
                                       sess.run(None, feed))}
    logits = outs["logits"]
    borders = outs["borders"]
    assert logits.shape == (T, K), f"logits shape {logits.shape} != ({T}, {K})"
    assert borders.shape == (K + 1,), f"borders shape {borders.shape} != ({K+1},)"
    # borders must be non-decreasing
    assert np.all(np.diff(borders) > 0), "borders must be strictly increasing"
    print(f"PASS: logits {logits.shape}, borders {borders.shape}")
```

This runs as part of fixture generation CI, similar to `roundtrip_check` in `make_fixture/fixture.py`.

### Golden Decode Verification (MODL-04 completion)

The `golden.json` for the tabpfn_v2 fixture includes:
- Input logits (K=16, n_test rows)
- Borders array (K+1=17 values)
- Expected mean (computed by Python reference implementation)
- Expected quantiles at levels `[0.1, 0.2, ..., 0.9]` (9 values per test row)

The C++ `test/cpp/test_tabfm_distribution_decode.cpp` loads golden.json and asserts:
```cpp
auto mean = DistributionMean(probs, borders);
CHECK(std::abs(mean - golden.mean) < 1e-4);
```

---

## Section 6: Per-Family License Gate (MODL-03)

### Current License Gate (VERIFIED from codebase)

`src/tabfm_weights.cpp:95-100` [VERIFIED: src/tabfm_weights.cpp:95-101]:
```cpp
bool LicenseAccepted(ClientContext &context) {
    Value value;
    if (!context.TryGetCurrentSetting("anofox_tabfm_accept_hf_license", value) || value.IsNull()) {
        return false;
    }
    return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
}
```

`src/tabfm_weights.cpp:285-295` [VERIFIED: src/tabfm_weights.cpp:285-295]:
```cpp
void RequireLicenseAccepted(ClientContext &context, const WeightsManifest &manifest) {
    if (!manifest.IsGated() || LicenseAccepted(context)) {
        return;
    }
    throw InvalidConfigurationException(
        "tabfm_download: weights in '%s' are licensed '%s' (non-commercial, no redistribution). "
        "Run: SET anofox_tabfm_accept_hf_license = true;", ...);
}
```

`src/tabfm_settings.cpp:77-81` [VERIFIED: src/tabfm_settings.cpp:77-81]:
```cpp
config.AddExtensionOption("anofox_tabfm_accept_hf_license",
    "Accept the upstream model license ...",
    LogicalType::BOOLEAN, Value::BOOLEAN(false));
```

### DuckDB v1.5.4 Constraint on Dynamic Options

`DBConfig::AddExtensionOption` is called at Load() time (in `RegisterTabfmSettings`). The DuckDB v1.5.4 API (`duckdb/src/include/duckdb/main/config.hpp:233-235`) [VERIFIED: duckdb/src/include/duckdb/main/config.hpp:233-235] does not provide runtime option registration — options must be registered at extension Load() time or via `config.AddExtensionOption` before the database is created. There is no `RegisterOptionDynamic()` equivalent.

**Implication:** A fully-dynamic `anofox_tabfm_accept_<any-license-id>` pattern is NOT possible at runtime. The options must be pre-registered for each known license id.

### Recommended Generic Gate Design

Register per-license options at Load() for all known license ids, using a naming convention:

```cpp
// In RegisterTabfmSettings():
// Pre-register acceptance options for all known license families.
// Future families are added here (one line each, no other code changes needed).
static const char *kKnownLicenses[] = {
    "tabfm-non-commercial-v1.0",
    "tabpfn-v2-cc-by-nc-4.0",   // for real tabpfn_v2 when export is unblocked
    nullptr
};
for (const char **lic = kKnownLicenses; *lic; lic++) {
    string opt_name = "anofox_tabfm_accept_" + SanitizeLicenseId(*lic);
    // e.g. "anofox_tabfm_accept_tabfm_non_commercial_v1_0"
    config.AddExtensionOption(opt_name,
        StringUtil::Format("Accept license '%s' for model downloads.", *lic),
        LogicalType::BOOLEAN, Value::BOOLEAN(false));
}
```

**License id sanitization** (option name must be a valid DuckDB setting name — lowercase, underscores only):
```cpp
string SanitizeLicenseId(const string &id) {
    string result = StringUtil::Lower(id);
    for (auto &c : result) if (!isalnum(c)) c = '_';
    return result;
}
```

**Generalized gate check:**
```cpp
bool GenericLicenseAccepted(ClientContext &context, const string &license_id) {
    if (license_id.empty() || license_id == "none") return true;
    string opt_name = "anofox_tabfm_accept_" + SanitizeLicenseId(license_id);
    Value value;
    if (!context.TryGetCurrentSetting(opt_name, value) || value.IsNull()) {
        return false;
    }
    return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
}
```

**Error text** (SQL-API §5 format):
```
"tabfm_download: weights for '<model>' require accepting license '<id>'.
 Run: SET anofox_tabfm_accept_<sanitized_id> = true;"
```

**Backward compatibility:** Keep `anofox_tabfm_accept_hf_license` registered (for existing users) and make it an alias that also gates `tabfm-non-commercial-v1.0`. Both options gating the same model family is safe and makes migration seamless.

**Phase 2 scope:** For the tabpfn_v2 fixture, `"license": "tabpfn-v2-fixture-mit"` — the fixture license is MIT (unencumbered). `IsGated()` returns false for `"fixture-mit"`, so no gate fires. For the real tabpfn_v2 family (deferred), the `tabpfn-v2-cc-by-nc-4.0` option would be pre-registered.

---

## Section 7: Scaffold-Owned File Changes (Coordination Required)

Per CLAUDE.md rule #2, these files are scaffold-owned and require coordinated batch edits:

| File | Change Required | Coordination Note |
|------|----------------|-------------------|
| `CMakeLists.txt:42-60` | Add `src/tabfm_profile_registry.cpp` to `EXTENSION_SOURCES` | Also add `test/cpp/test_tabfm_profile_registry.cpp` and `test/cpp/test_tabfm_distribution_decode.cpp` to `TABFM_CPP_TEST_SOURCES` |
| `src/anofox_tabfm_extension.cpp` | Add `RegisterProfileRegistry(loader)` call in `LoadInternal` (if registry needs explicit init) | May not be needed if self-registration via static initializers suffices |
| `src/include/tabfm_registration.hpp` | Add `void RegisterProfileRegistry(ExtensionLoader &loader);` declaration | Only if registry has public SQL-visible functions (it may not) |
| `src/tabfm_settings.cpp` | Add per-license `AddExtensionOption` calls for known license ids | Batch with any new settings |
| `src/include/tabfm_ort_engine.hpp` | Extend `TabFMRunOutput` with `vector<float> borders;` | Single-line addition; low risk |
| `src/include/tabfm_manifest.hpp` | Add `bool distribution_output = false;` to `ModelManifest` | Single field addition |
| `src/tabfm_manifest.cpp` | Parse `"distribution_output"` optional boolean field | One `yyjson_obj_get` call |

**Static initializer registration caveat:** The self-registering pattern (`static ProfileRegistration k_...`) guarantees registration before `DispatchPreprocess` is called IF the `.cpp` files are linked. On some linkers, static initializers in object files that export no referenced symbols can be stripped. The safe approach is to ensure `RegisterAllProfiles()` is called explicitly in `LoadInternal`, which calls into each profile's `.cpp` to force the static init. This adds one function to each profile `.cpp` but avoids linker surprises.

---

## Common Pitfalls

### Pitfall 1: Non-Uniform Borders — Uniform Bin Assumption
**What goes wrong:** Computing quantiles or CRPS using assumed uniform bins (`width = 1/K`). Results are numerically plausible but wrong, especially for extreme quantiles.
**Why it happens:** Easy to prototype with `np.linspace(-5, 5, K+1)` and forget the real distribution is highly non-uniform (outer bins span 70+ normalized units).
**How to avoid:** Always pass actual borders. The `yhat_dist.borders` field in the output exists precisely for this reason. Phase 3 CRPS code must consume `borders`, not reconstruct uniform bins.
**Warning signs:** Quantile predictions that agree at the median but diverge sharply at 0.1/0.9.

### Pitfall 2: Static Initializer Stripping (Registry Pattern)
**What goes wrong:** `ProfileRegistration` objects in `.cpp` files are silently not constructed by the linker, leaving the registry empty. `DispatchPreprocess("tabfm_v1_minimal", ...)` throws "unknown profile".
**Why it happens:** Linkers strip translation units with no referenced symbols.
**How to avoid:** Add an explicit `void ForceProfileInit()` function in each profile `.cpp` and call it from `LoadInternal`. The static init still runs as a side effect.
**Warning signs:** Registry dispatch throws on "tabfm_v1_minimal" in debug builds on Linux but works in test.

### Pitfall 3: ORT Output Name Mismatch
**What goes wrong:** The fixture ONNX graph names the outputs `"output_0"` and `"output_1"` (ONNX default), but the C++ code requests `"logits"` and `"borders"` by name.
**Why it happens:** `onnx.helper.make_node` uses default output names; the code assumes a naming convention.
**How to avoid:** Explicitly name outputs in the graph builder:
```python
logits_node = helper.make_node("...", inputs=[...], outputs=["logits"])
borders_const = helper.make_node("Constant", inputs=[], outputs=["borders"], ...)
```
Validate the names in `tools/parity` before the C++ code is written.
**Warning signs:** ORT `InvalidArgument: 'borders' was not found in output names`.

### Pitfall 4: z-Space vs Raw-Space Borders
**What goes wrong:** Decode uses `out.borders` (z-normalized) directly for `yhat` (raw-space). Mean comes out in z-normalized units (~0.2 instead of 35.0).
**Why it happens:** The affine transform `raw = znorm * y_std + y_mean` is a two-line step that is easy to forget.
**How to avoid:** For the fixture (K=16, random-init), `target_mean` and `target_scale` come from the tabpfn_v2 preprocessing profile. The golden.json for the fixture should store both z-space and raw-space expectations.

### Pitfall 5: borders Field in `yhat_dist` STRUCT
**What goes wrong:** For the Phase 3 CRPS formula, consumers need raw-space borders. If `yhat_dist.borders` stores z-space borders (pre-transform), Phase 3 CRPS is silently wrong.
**Decision:** Store **raw-space borders** in `yhat_dist.borders` (after the affine transform). Store **z-space logits** in `yhat_dist.logits` (before softmax, for maximum numerical precision). Document this convention clearly in the header.

### Pitfall 6: DuckDB STRUCT-of-LIST Output in Aggregate
**What goes wrong:** Building `Value::STRUCT` with `LIST(DOUBLE)` children fails at `ListVector::PushBack` because the child type does not match the pre-bound aggregate return type.
**Why it happens:** DuckDB aggregates are strongly typed at bind time; the child list type must match exactly.
**How to avoid:** Construct the list values with the exact same `LogicalType::LIST(LogicalType::DOUBLE)` that was declared in `ListStructType()`. Use `Value::LIST(LogicalType::DOUBLE, children)` not `Value::LIST(children)`.

### Pitfall 7: borders Output from Constant Node in ONNX
**What goes wrong:** A `Constant` node whose output is not connected to the graph's output list is silently dropped by ONNX shape inference and by ORT.
**How to avoid:** Explicitly add the `borders` node's output name to `graph.output` in the ONNX helper build. Verify in `tools/parity` that ORT produces both outputs.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Softmax over K logits | Custom exp/sum loop | Reuse `SoftmaxInPlace` already in `tabfm_engine.cpp:634` | Already there; numerically stable (shift by max) |
| ONNX graph authoring | Custom binary serialization | `onnx.helper` Python API | Official ONNX builder; handles proto serialization correctly |
| Random-init weight generation | Custom RNG | `torch.Generator().manual_seed(seed)` (existing pattern from `fixture.py`) | Ensures byte-determinism via sorted keys |
| License option registration | Lazy registration at first download | Pre-register at Load() (DBConfig limitation) | DuckDB v1.5.4 does not support runtime option registration |
| Non-uniform bin quantile | Assume uniform and scale | CDF inverse interpolation over actual borders | Uniform assumption is wrong by ~28,000:1 in the tails |

---

## Code Examples

### Registry Self-Registration Pattern
```cpp
// Source: adapted from existing static-init patterns in the codebase
// src/tabfm_profile_registry.cpp

namespace {
// The global registry: string → function pointer
unordered_map<string, PreprocessFn> &GetRegistry() {
    static unordered_map<string, PreprocessFn> registry;
    return registry;
}
} // anonymous namespace

ProfileRegistration::ProfileRegistration(const string &id, PreprocessFn fn) {
    GetRegistry().emplace(id, fn);
}

PreprocessedBatch DispatchPreprocess(const string &profile_id,
    const ColumnDataCollection &data,
    const vector<PreprocessColumnSpec> &columns,
    PreprocessTask task) {
    auto &registry = GetRegistry();
    auto it = registry.find(profile_id);
    if (it == registry.end()) {
        throw InvalidInputException(
            "tabfm: preprocessing profile '%s' is not registered. "
            "Verify the manifest's preprocessing_profile field. "
            "Run: SELECT * FROM tabfm_models();",
            profile_id);
    }
    return it->second(data, columns, task);
}
```

### tabpfn_v2 Minimal Preprocessing Profile
```cpp
// Source: design based on PreprocessedBatch contract in tabfm_preprocess.hpp
PreprocessedBatch TabPFNV2PreprocessBatch(
    const ColumnDataCollection &data,
    const vector<PreprocessColumnSpec> &columns,
    PreprocessTask task) {
    // Fixture-scoped: pass features through as-is (no TabFM z-score pipeline).
    // Full-fidelity tabpfn_v2 preprocessing is deferred with the real export.
    PreprocessedBatch batch;
    // ... populate batch.x, batch.y, batch.T, batch.H, batch.train_size, etc.
    // ... compute batch.target_mean and batch.target_scale from training rows
    return batch;
}
// Self-register:
static const ProfileRegistration kTabPFNV2Reg("tabpfn_v2", TabPFNV2PreprocessBatch);
```

### Distribution Decode — Mean
```cpp
// Source: spike SPIKE-tabpfn-v2-tensor-contract.md lines 119-136
double DistributionMean(const vector<double> &probs, const vector<double> &borders) {
    const size_t K = probs.size();
    D_ASSERT(borders.size() == K + 1);
    double mean = 0.0;
    for (size_t i = 0; i < K; i++) {
        double midpoint = (borders[i] + borders[i + 1]) / 2.0;
        mean += probs[i] * midpoint;
    }
    // FullSupportBarDistribution outer-bin half-normal correction (negligible but spec-required)
    const double w0 = borders[1] - borders[0];
    const double wK = borders[K] - borders[K - 1];
    // Replace left-tail bin midpoint contribution:
    mean -= probs[0] * (borders[0] + borders[1]) / 2.0;
    mean += probs[0] * (borders[0] - std::sqrt(M_PI / 2.0) * w0);
    // Replace right-tail bin midpoint contribution:
    mean -= probs[K - 1] * (borders[K - 1] + borders[K]) / 2.0;
    mean += probs[K - 1] * (borders[K] + std::sqrt(M_PI / 2.0) * wK);
    return mean;
}
```

### Distribution Decode — Quantile
```cpp
// Source: spike SPIKE-tabpfn-v2-tensor-contract.md lines 139-150
double DistributionQuantile(const vector<double> &probs,
                             const vector<double> &borders, double q) {
    const size_t K = probs.size();
    double cumsum = 0.0;
    for (size_t i = 0; i < K; i++) {
        double prev = cumsum;
        cumsum += probs[i];
        if (cumsum >= q) {
            // Linear interpolation within bin i
            double frac = probs[i] > 1e-30 ? (q - prev) / probs[i] : 0.5;
            return borders[i] + frac * (borders[i + 1] - borders[i]);
        }
    }
    return borders[K]; // q >= 1.0: return rightmost border
}
```

---

## Project Constraints (from CLAUDE.md)

| Directive | Implication for Phase 2 |
|-----------|------------------------|
| One module = one `src/tabfm_*.cpp` | New registry module goes in `src/tabfm_profile_registry.cpp`; tabpfn_v2 preprocessing in `src/tabfm_preprocess_tabpfn_v2.cpp` or same registry file |
| Red-green TDD | Failing Catch2/sqllogictest tests BEFORE implementation |
| Telemetry once per user-facing function | No new user-facing SQL functions in this phase (all changes are internal engine seams); no new telemetry calls needed |
| Full `anofox_tabfm_*` + short `tabfm_*` aliases | No new SQL functions added in Phase 2 |
| Errors name the fixing SET/CALL | Registry error: names profile field; license error: names `SET anofox_tabfm_accept_<id>` |
| No weight bytes in repo | tabpfn_v2 fixture uses random-init ONNX graph; borders are architectural constants (not Google's) |
| CPU-flavor-clean | Distribution decode is pure arithmetic; no GPU code |
| Scaffold-owned files: CMakeLists.txt, anofox_tabfm_extension.cpp, tabfm_registration.hpp, tabfm_settings.cpp | Must be edited in a coordinated batch; identified in Section 7 |
| `unnest(res, max_depth := 3)` — never `recursive := true` | Relevant only if SQL tests unnest the distribution output |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11+ + uv | tools/make_fixture, new tools/parity | ✓ (uv in project, Python presumed) | 3.11+ | — |
| onnx Python package | Fixture graph authoring | ✓ (in make_fixture dependencies) | ≥1.17 | — |
| onnxruntime Python | Parity validation | ✓ (in make_fixture dependencies) | ≥1.20 | — |
| ONNX Runtime 1.23.2 C++ | C++ tests + extension | ✓ (pinned in cmake/ort.cmake) | 1.23.2 | — |
| DuckDB v1.5.4 (submodule) | Extension build | ✓ (submodule) | 1.5.4 | — |

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Catch2 (C++) + SQLLogicTest (SQL) |
| C++ config | `TABFM_CPP_TEST_SOURCES` in `CMakeLists.txt:121-133` |
| Quick run | `./build/debug/test/unittest test/cpp/test_tabfm_profile_registry.cpp` |
| Full suite | `make test_debug` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MGEN-01 | `tabfm_v1_minimal` profile self-registers and dispatches correctly | unit (Catch2) | `./build/debug/test/unittest test/cpp/test_tabfm_profile_registry.cpp` | ❌ Wave 0 |
| MGEN-02 | Unknown profile throws named error with fix text | unit (Catch2) + SQL | `./build/debug/test/unittest test/cpp/test_tabfm_profile_registry.cpp` | ❌ Wave 0 |
| MGEN-03 | Shape/rank/class-count violation throws before decode | unit (Catch2) | `./build/debug/test/unittest test/cpp/test_tabfm_ort_engine.cpp` | ✅ (extend) |
| RDIST-01 | ORT run output carries borders; empty when absent | unit (Catch2) | `./build/debug/test/unittest test/cpp/test_tabfm_ort_engine.cpp` | ✅ (extend) |
| RDIST-02 | `output_mode='distribution'` emits `yhat_dist` + `yhat_quantiles` | SQL + unit | `./build/debug/test/unittest test/sql/tabfm_distribution.test` | ❌ Wave 0 |
| MODL-01 | tabpfn_v2 fixture loads, predicts, emits distribution | SQL | `./build/debug/test/unittest test/sql/tabfm_distribution.test` | ❌ Wave 0 |
| MODL-03 | Unknown license gate fires; accepted gate passes | SQL | `./build/debug/test/unittest test/sql/tabfm_license.test` | ✅ (extend) |
| MODL-04 | tools/parity validates fixture's two distribution tensors | Python (pytest) | `cd tools/parity && uv run pytest` | ❌ Wave 0 (tools/parity doesn't exist) |

### Sampling Rate
- Per task commit: `./build/debug/test/unittest [specific test file]`
- Per wave merge: `make test_debug`
- Phase gate: Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `test/cpp/test_tabfm_profile_registry.cpp` — covers MGEN-01, MGEN-02
- [ ] `test/cpp/test_tabfm_distribution_decode.cpp` — covers RDIST-01, RDIST-02 decode math
- [ ] `test/sql/tabfm_distribution.test` — covers RDIST-02, MODL-01 end-to-end
- [ ] `tools/parity/` — entire uv project; covers MODL-04
- [ ] `test/fixtures/tabpfn_v2/` — fixture artifacts (generated by tooling, not hand-written)

---

## Security Domain

`security_enforcement` not explicitly set in `.planning/config.json` — treating as enabled.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Manifest JSON validation (existing `ParseModelManifest`); profile string validated at dispatch; distribution tensor shapes validated before decode |
| V4 Access Control | partial | License gate (MODL-03): per-license options pre-registered; gate fires before any I/O |
| V6 Cryptography | no | No new crypto; existing sha256 for fixture files uses OpenSSL EVP (already in `tabfm_engine.cpp:369`) |
| V2 Authentication | no | No new auth surface |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malicious manifest `preprocessing_profile` string injection | Tampering | Registry lookup with exact-match key; unknown profiles throw immediately |
| Crafted ONNX with unexpected output count / shape | Tampering | `ValidateDistributionOutput` checks shape before decode; ORT output name check |
| License id containing shell-unsafe characters | Elevation | `SanitizeLicenseId` replaces non-alphanumeric → underscore before option name construction |

---

## Open Questions

1. **`distribution_output` manifest field vs. profile-based detection**
   - What we know: The CONTEXT.md says "tabpfn_v2 preprocessing profile" and the fixture will emit two outputs. The engine needs to know to call distribution validation and decode.
   - What's unclear: Whether to detect distribution output by the preprocessing_profile name (e.g., `profile == "tabpfn_v2"`) or an explicit manifest flag (`"distribution_output": true`).
   - Recommendation: Use an explicit boolean manifest field. Profile name and distribution format are orthogonal — a future profile might emit distribution without being called "tabpfn_v2", and vice versa. The planner should choose one and document it.

2. **Quantile level set (Claude's discretion)**
   - Recommendation: 9 levels: `{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`. Stored as a compile-time constant `kQuantileLevels`. The median (0.5) is included; symmetric pairs (0.1/0.9 = 80% CI, 0.25/0.75 = 50% CI) are natural for reporting. Phase 3 can always recompute other levels from `yhat_dist.logits` + `yhat_dist.borders`.

3. **tabpfn_v2 input shapes — does the fixture ONNX need `[1, T, H]` or `[T, H]`?**
   - What we know: The current `TabFMRunInput` feeds `x [1,T,H]`, `y [1,T]` with batch dim 1. TabPFN v2 real model uses `[n_test, K]` for logits (no batch dim in output). The fixture graph can accept whatever shapes the engine feeds.
   - What's unclear: Whether to use the same `[1,T,H]` input convention (easier — no engine input changes) or `[T,H]` (matches the real tabpfn_v2 forward signature better).
   - Recommendation: Keep `[1,T,H]` inputs for the fixture (no `TabFMRunInput` changes needed). The fixture graph internally squeezes/reshapes as needed. Document the convention in the manifest.

4. **`anofox_tabfm_accept_hf_license` backward compat**
   - Recommendation: Keep it registered and make it gate `tabfm-non-commercial-v1.0`. The new generic gate (`anofox_tabfm_accept_tabfm_non_commercial_v1_0`) also gates the same family. Either one passing is sufficient.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Static initializer self-registration pattern works with the project's linker without explicit force-init call (no stripping) | §1 Registry Design | Registry empty at runtime; all preprocessing fails |
| A2 | ORT 1.23.2 supports requesting multiple named outputs by name and returns them in order | §4 | Distribution borders never populated; distribution decode always uses empty borders |
| A3 | `onnx.helper` built graph with named `"logits"` and `"borders"` outputs works with ORT 1.23.2 (opset 18) | §5 | Fixture parity validation fails; must use different output names or opset |
| A4 | `tabpfn-v2-fixture-mit` as a license id causes `IsGated()` to return false (ungated) | §6 | License gate fires on fixture download attempt |
| A5 | The 9-quantile level set `{0.1,...,0.9}` is sufficient for Phase 3 CRPS/scoring rules | §3 | Phase 3 may need different quantile levels; recomputable from yhat_dist |

---

## Sources

### Primary (HIGH confidence — direct codebase reads this session)
- `src/tabfm_engine.cpp` — engine predict flow, decode pattern, `PreprocessBatch` call site (line 674)
- `src/include/tabfm_predict.hpp` — `TabFMPredictOptions`, `TabFMPredictResult`, `PredictEngine` seam
- `src/tabfm_predict_agg.cpp` — bind wiring, `ListStructType()`, `output_mode` parsing
- `src/include/tabfm_ort_engine.hpp` — `TabFMRunOutput` struct, `ValidateTabFMOutput` declaration
- `src/tabfm_ort_engine.cpp:528-554` — `ValidateTabFMOutput` implementation
- `src/include/tabfm_manifest.hpp` — `ModelManifest` struct, `preprocessing_profile` field
- `src/tabfm_manifest.cpp` — manifest parsing, `preprocessing_profile` parse at line 226
- `src/include/tabfm_preprocess.hpp` — `kPreprocessProfileId = "tabfm_v1_minimal"`, `PreprocessedBatch`
- `src/tabfm_settings.cpp` — `AddExtensionOption` calls, `anofox_tabfm_accept_hf_license`
- `src/tabfm_weights.cpp:95-295` — `LicenseAccepted`, `RequireLicenseAccepted`
- `src/anofox_tabfm_extension.cpp` — `LoadInternal`, registration call order
- `src/include/tabfm_registration.hpp` — registration function declarations
- `CMakeLists.txt:42-133` — `EXTENSION_SOURCES`, `TABFM_CPP_TEST_SOURCES`
- `test/fixtures/manifest.json` — fixture manifest schema, `preprocessing_profile: "tabfm_v1_minimal"`
- `test/fixtures/regression/manifest.json` — regression fixture manifest pattern
- `tools/make_fixture/src/make_fixture/fixture.py` — complete fixture build pattern
- `duckdb/src/include/duckdb/main/config.hpp:233-255` — `AddExtensionOption` API (Load-time only)

### Spike Reports (HIGH confidence — in-repo, authored this project)
- `.planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md` — confirmed tensor contract: logits `[n,K]`, borders `[K+1]`, decode math
- `.planning/spikes/SPIKE-tabicl-onnx-export.md` — TabICL deferred (not researched further)

### ASSUMED
- A1–A5 above (static init linker behavior, ORT multi-output API, onnx helper compatibility, license id behavior, quantile set sufficiency)

---

## Metadata

**Confidence breakdown:**
- Preprocessing registry design: HIGH — manifest field parse verified; engine dispatch call site verified; `PreprocessBatch` signature verified
- Distribution output wiring: HIGH — `TabFMPredictResult`, `ListStructType`, `output_mode` parsing all verified; decode math from confirmed spike
- Fixture tooling: HIGH — `make_fixture/fixture.py` fully read; existing manifest JSON format verified; `tools/parity` absence verified
- License gate: HIGH — `AddExtensionOption` API, `TryGetCurrentSetting` pattern, `LicenseAccepted` implementation all verified; DuckDB v1.5.4 load-time-only constraint verified from config.hpp

**Research date:** 2026-09-21
**Valid until:** 2026-10-21 (stable domain; DuckDB pinned at v1.5.4, ORT pinned at 1.23.2)
