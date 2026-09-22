# Phase 2: Model Generalization + Distribution Output (fixture-backed) - Pattern Map

**Mapped:** 2026-09-21
**Files analyzed:** 17 (new + modified)
**Analogs found:** 15 / 17

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/tabfm_profile_registry.cpp` | service / dispatch | request-response | `src/tabfm_ort_engine.cpp` (output-name dispatch ~line 496-504) | role-match |
| `src/include/tabfm_profile_registry.hpp` | header | request-response | `src/include/tabfm_preprocess.hpp` | role-match |
| `src/tabfm_engine.cpp` (line 674, dispatch + decode) | service | request-response | self (existing decode at lines 630-776) | exact |
| `src/include/tabfm_ort_engine.hpp` (`TabFMRunOutput` extension) | header / struct | request-response | self (lines 213-217) | exact |
| `src/include/tabfm_predict.hpp` (`TabFMPredictResult` + `TabFMPredictOptions` extension) | header / struct | request-response | self (lines 50-74) | exact |
| `src/tabfm_predict_agg.cpp` (`ListStructType` + `output_mode` extension) | controller / aggregate | request-response | self (lines 79-89, 140-145) | exact |
| `src/include/tabfm_manifest.hpp` (`distribution_output` flag) | model / struct | CRUD | self (lines 72-86) | exact |
| `src/tabfm_manifest.cpp` (parse `distribution_output`) | service | CRUD | self (`preprocessing_profile` parse at line 226) | exact |
| `src/tabfm_weights.cpp` (generic license gate) | service | request-response | self (lines 95-101, 285-295) | exact |
| `src/tabfm_settings.cpp` (per-license `AddExtensionOption`) | config | config | self (lines 74-81) | exact |
| `CMakeLists.txt` (new sources + test sources) | config | batch | self (`EXTENSION_SOURCES`, `TABFM_CPP_TEST_SOURCES`) | exact |
| `src/anofox_tabfm_extension.cpp` (registry force-init) | config | request-response | self (`LoadInternal` call order) | exact |
| `src/include/tabfm_registration.hpp` (new declaration) | header | — | self (existing declarations) | exact |
| `test/fixtures/tabpfn_v2/` (K=16 fixture family) | test fixture | file-I/O | `tools/make_fixture/src/make_fixture/fixture.py` | role-match |
| `tools/parity/` (new uv project) | tool / test | file-I/O | `tools/export_onnx/pyproject.toml` + `tools/make_fixture/pyproject.toml` | role-match |
| `test/cpp/test_tabfm_profile_registry.cpp` | test | request-response | `test/cpp/test_tabfm_ort_engine.cpp` | role-match |
| `test/cpp/test_tabfm_distribution_decode.cpp` | test | request-response | `test/cpp/test_tabfm_ort_engine.cpp` | role-match |

---

## Pattern Assignments

### `src/tabfm_profile_registry.cpp` + `src/include/tabfm_profile_registry.hpp` (service, dispatch)

**Primary analog:** `src/tabfm_ort_engine.cpp` (output-name dispatch loop, lines 496-504)
**Secondary analog for header conventions:** `src/include/tabfm_preprocess.hpp` (public module interface)

**Dispatch-by-name pattern** (`src/tabfm_ort_engine.cpp` lines 496-504):
```cpp
idx_t output_index = 0;
for (idx_t i = 0; i < session.output_names.size(); i++) {
    if (session.output_names[i] == "logits") {
        output_index = i;
        break;
    }
}
```
The registry `DispatchPreprocess` follows the same "lookup by string key, throw with actionable text on miss" pattern — but uses a `static unordered_map` inside an anonymous-namespace getter instead of a linear scan, because the registry is mutable at static-init time.

**Error pattern for unknown key** (`src/tabfm_ort_engine.cpp` lines 491-495):
```cpp
if (session.output_names.empty()) {
    throw InvalidInputException(
        "anofox_tabfm: model graph declares no outputs; expected a 'logits' output (HLD §4.4). If you set "
        "anofox_tabfm_model_manifest, point it at a compatible weight-free graph.");
}
```
Copy the `InvalidInputException` + actionable-SET text shape. For the registry the message is:
```
"tabfm: preprocessing profile '%s' is not registered. "
"Verify the manifest's preprocessing_profile field. "
"Run: SELECT * FROM tabfm_models();"
```

**Namespace / header guard pattern** (`src/include/tabfm_preprocess.hpp` lines 1-10, inferred from codebase conventions):
```cpp
#pragma once
#include "duckdb/common/string.hpp"
// ...
namespace duckdb { namespace anofox {
// public declarations
} } // namespace duckdb::anofox
```

**kPreprocessProfileId constant to use for self-registration** (`src/include/tabfm_preprocess.hpp` line 70):
```cpp
static constexpr const char *kPreprocessProfileId = "tabfm_v1_minimal";
```
Reference this constant (not a string literal) when registering the `tabfm_v1_minimal` profile.

**Static-singleton registry pattern** (from `src/tabfm_engine.cpp` lines 781-783):
```cpp
PredictEngine &GetPredictEngine() {
    static TabFMRealEngine engine;
    return engine;
}
```
Use the same "function-local static" pattern for `GetRegistry()` to guarantee construction before first use and avoid the static-init order fiasco:
```cpp
// In anonymous namespace in tabfm_profile_registry.cpp:
unordered_map<string, PreprocessFn> &GetRegistry() {
    static unordered_map<string, PreprocessFn> registry;
    return registry;
}
```

---

### `src/tabfm_engine.cpp` — dispatch change (line 674) + distribution decode (after line 772)

**Analog:** self — existing preprocessing call and decode block.

**Preprocessing dispatch change** (line 674, replace one line):
```cpp
// BEFORE (line 674):
auto batch = PreprocessBatch(collection, columns, pp_task);

// AFTER:
auto batch = DispatchPreprocess(
    resolved.manifest.preprocessing_profile, collection, columns, pp_task);
```

**Existing regression decode pattern to extend** (lines 768-773):
```cpp
} else {
    double raw = C > 0 ? out.logits[t * C] : 0.0;
    double yhat = raw * batch.target_scale + batch.target_mean;
    result.yhat[src] = Value::DOUBLE(yhat);
    result.yhat_score[src] = Value(LogicalType::DOUBLE); // NULL
}
```
The distribution branch inserts BEFORE the closing `}` of the regression else, gated on `in.opts.distribution && !out.borders.empty()`.

**Existing `SoftmaxInPlace` to reuse** (lines 634-650):
```cpp
void SoftmaxInPlace(vector<double> &v, double temperature) {
    double t = temperature > 0 ? temperature : 1.0;
    double m = v.empty() ? 0.0 : v[0];
    for (auto x : v) { m = MaxValue(m, x); }
    double sum = 0;
    for (auto &x : v) { x = std::exp((x - m) / t); sum += x; }
    if (sum > 0) { for (auto &x : v) { x /= sum; } }
}
```
Call `SoftmaxInPlace(logits_d, 1.0)` for the distribution path (temperature=1.0 per confirmed contract).

**Existing proba MAP construction pattern to copy for list values** (lines 759-767):
```cpp
vector<Value> keys, vals;
for (idx_t c = 0; c < n_classes; c++) {
    keys.emplace_back(batch.label_decoder[c].ToString());
    vals.emplace_back(Value::DOUBLE(logits[c]));
}
result.proba[src] =
    Value::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE, std::move(keys), std::move(vals));
```
For distribution list fields, use `Value::LIST(LogicalType::DOUBLE, children)` (not the untyped overload — Pitfall 6 from RESEARCH.md).

---

### `src/include/tabfm_ort_engine.hpp` — `TabFMRunOutput` extension (lines 213-217)

**Analog:** self — current struct definition.

**Current struct** (lines 213-217):
```cpp
struct TabFMRunOutput {
    vector<float> logits;
    //! [1, T, C]
    vector<int64_t> shape;
};
```
Add one field:
```cpp
struct TabFMRunOutput {
    vector<float> logits;
    //! [1,T,C] for tabfm-v1; [n_test,K] for tabpfn_v2
    vector<int64_t> shape;
    //! Optional: [K+1] bin borders for distribution models; empty otherwise.
    //! Non-uniform (outer bins ~67 wide, inner ~0.0024). Carries z-normalized
    //! borders; caller applies affine transform to raw space during decode.
    vector<float> borders;
};
```
Add declaration alongside `ValidateTabFMOutput` (line 244):
```cpp
//! Validates a distribution-output forward pass: shape must be [n_test, K]
//! (rank 2) and borders.size() == K+1. Throws InvalidInputException naming
//! the contract on any mismatch.
void ValidateDistributionOutput(const TabFMRunOutput &out, idx_t n_test);
```

---

### `src/include/tabfm_predict.hpp` — `TabFMPredictOptions` + `TabFMPredictResult` extension

**Analog:** self — current struct definitions (lines 50-74).

**`TabFMPredictOptions` extension** (add after `detail` field, line 57):
```cpp
//! output_mode == 'distribution' → yhat_dist STRUCT + yhat_quantiles
//! Only meaningful when task == REGRESSION and model emits borders.
bool distribution = false;
```

**`TabFMPredictResult` extension** (add after `proba` field, line 73):
```cpp
//! Populated only when opts.distribution && model emits borders.
//! yhat_dist_logits: z-space logits (DOUBLE[], K per test row; pre-softmax,
//!   maximum precision for Phase 3 CRPS recomputation).
//! yhat_dist_borders: raw-space borders (DOUBLE[], K+1; affine-transformed
//!   from z-space; same vector for every row but stored per-row for SQL ergonomics).
//! yhat_quantiles: DOUBLE[], 9 values at levels {0.1,…,0.9} per test row.
vector<Value> yhat_dist_logits;
vector<Value> yhat_dist_borders;
vector<Value> yhat_quantiles;
```

---

### `src/tabfm_predict_agg.cpp` — `ListStructType` + `output_mode` parsing

**Analog:** self — existing `ListStructType()` (lines 79-89) and `ParseOneOption` output_mode block (lines 140-145).

**`ListStructType()` extension pattern** (lines 79-89, copy structure):
```cpp
LogicalType ListStructType() const {
    child_list_t<LogicalType> fields;
    fields.emplace_back("cols", row_type);
    fields.emplace_back("yhat", YhatType());
    fields.emplace_back("yhat_score", LogicalType::DOUBLE);
    fields.emplace_back("is_training", LogicalType::BOOLEAN);
    if (EmitProba()) {
        fields.emplace_back("proba",
            LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE));
    }
    // NEW: distribution fields (regression only, output_mode='distribution')
    if (options.distribution && options.task == TabFMTask::REGRESSION) {
        fields.emplace_back("yhat_dist",
            LogicalType::STRUCT({
                {"logits",  LogicalType::LIST(LogicalType::DOUBLE)},
                {"borders", LogicalType::LIST(LogicalType::DOUBLE)}}));
        fields.emplace_back("yhat_quantiles",
            LogicalType::LIST(LogicalType::DOUBLE));
    }
    return LogicalType::LIST(LogicalType::STRUCT(std::move(fields)));
}
```

**`output_mode` parsing extension** (lines 140-145 — replace the block):
```cpp
} else if (key == "output_mode") {
    auto mode = StringUtil::Lower(val);
    if (mode == "detail") {
        opts.detail = true;
    } else if (mode == "distribution") {
        opts.detail = false;      // mutually exclusive
        opts.distribution = true;
    } else if (mode != "compact") {
        throw BinderException(
            "%s: output_mode must be 'compact', 'detail', or 'distribution', got '%s'",
            fname, val);
    }
```

**`EmitDistribution()` helper** (add alongside `EmitProba()` at line 62):
```cpp
bool EmitDistribution() const {
    return options.distribution && options.task == TabFMTask::REGRESSION;
}
```

**Finalize result-population pattern** — copy the existing proba population block (lines 759-767) for the new distribution fields. Use `Value::LIST(LogicalType::DOUBLE, children)` explicitly (not the untyped overload).

---

### `src/include/tabfm_manifest.hpp` — `distribution_output` flag

**Analog:** self — `ModelManifest` struct (lines 72-86).

**Addition** (after `preprocessing_profile`, line 82):
```cpp
//! True when the model graph emits distribution tensors (logits [n,K] + borders [K+1]).
//! Parsed from optional JSON field "distribution_output" (default: false).
//! Controls ValidateDistributionOutput vs. ValidateTabFMOutput dispatch in engine.
bool distribution_output = false;
```

---

### `src/tabfm_manifest.cpp` — parse `distribution_output`

**Analog:** self — existing `preprocessing_profile` parse (line 226):
```cpp
manifest.preprocessing_profile = GetRequiredString(root, "preprocessing_profile", manifest_path);
```
For the optional boolean field, use the `yyjson_obj_get` + default pattern already present:
```cpp
// In ParseModelManifest(), after preprocessing_profile:
auto dist_val = duckdb_yyjson::yyjson_obj_get(root, "distribution_output");
if (dist_val && duckdb_yyjson::yyjson_is_bool(dist_val)) {
    manifest.distribution_output = duckdb_yyjson::yyjson_get_bool(dist_val);
}
// default false (field is optional; existing manifests need no change)
```

---

### `src/tabfm_weights.cpp` — generic license gate (lines 95-295)

**Analog:** self — existing `LicenseAccepted` (lines 95-101) and `RequireLicenseAccepted` (lines 285-295).

**Current `LicenseAccepted`** (lines 95-101):
```cpp
bool LicenseAccepted(ClientContext &context) {
    Value value;
    if (!context.TryGetCurrentSetting("anofox_tabfm_accept_hf_license", value) || value.IsNull()) {
        return false;
    }
    return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
}
```

**Generalized pattern** — replace with a `SanitizeLicenseId` helper + generic lookup:
```cpp
// Keep SanitizeLicenseId in anonymous namespace (no public header needed):
string SanitizeLicenseId(const string &id) {
    string result = StringUtil::Lower(id);
    for (auto &c : result) { if (!isalnum(c)) c = '_'; }
    return result;
}

bool GenericLicenseAccepted(ClientContext &context, const string &license_id) {
    if (license_id.empty() || license_id == "none") return true;
    // Backward compat: "tabfm-non-commercial-v1.0" also accepts via old option.
    if (license_id == "tabfm-non-commercial-v1.0") {
        if (LicenseAccepted(context)) return true; // legacy option
    }
    string opt_name = "anofox_tabfm_accept_" + SanitizeLicenseId(license_id);
    Value value;
    if (!context.TryGetCurrentSetting(opt_name, value) || value.IsNull()) {
        return false;
    }
    return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
}
```

**`RequireLicenseAccepted` replacement** — copy the existing error shape (lines 285-295):
```cpp
void RequireLicenseAccepted(ClientContext &context, const WeightsManifest &manifest) {
    if (!manifest.IsGated() || GenericLicenseAccepted(context, manifest.license)) {
        return;
    }
    auto what = manifest.repo.empty() ? manifest.model : manifest.repo;
    throw InvalidConfigurationException(
        "tabfm_download: weights for '%s' require accepting license '%s'. "
        "Run: SET anofox_tabfm_accept_%s = true;",
        what, manifest.license,
        SanitizeLicenseId(manifest.license));
}
```

---

### `src/tabfm_settings.cpp` — per-license `AddExtensionOption` registration

**Analog:** self — existing `AddExtensionOption` block (lines 74-81, 77-80):
```cpp
config.AddExtensionOption("anofox_tabfm_accept_hf_license",
    "Accept the upstream model license (tabfm-non-commercial-v1.0: non-commercial use, "
    "no redistribution). Downloads of Google-licensed weights fail without this.",
    LogicalType::BOOLEAN, Value::BOOLEAN(false));
```

**Pattern to copy** — add per-license options after the existing `accept_hf_license` option. Loop over a compile-time table of known license ids (RESEARCH.md §6):
```cpp
// Keep the existing option for backward compat:
config.AddExtensionOption("anofox_tabfm_accept_hf_license", ...);

// Per-license generic options (MODL-03):
static const struct { const char *id; const char *desc; } kKnownLicenses[] = {
    {"tabfm_non_commercial_v1_0",
     "Accept license 'tabfm-non-commercial-v1.0' (non-commercial, no redistribution) for TabFM v1 downloads."},
    {"tabpfn_v2_cc_by_nc_4_0",
     "Accept license 'tabpfn-v2-cc-by-nc-4.0' (CC BY-NC 4.0) for TabPFN v2 downloads."},
    {nullptr, nullptr}
};
for (auto *lic = kKnownLicenses; lic->id; ++lic) {
    config.AddExtensionOption(
        string("anofox_tabfm_accept_") + lic->id,
        lic->desc,
        LogicalType::BOOLEAN, Value::BOOLEAN(false));
}
```
The option names are pre-sanitized (lowercase, underscores) matching `SanitizeLicenseId()` output.

---

### `test/fixtures/tabpfn_v2/` — K=16 random-init ONNX fixture family

**Analog:** `tools/make_fixture/src/make_fixture/fixture.py` (complete pattern).

**Seeded random-init weight pattern** (fixture.py lines 73-82):
```python
def seeded_model(task: str = "classification") -> TabFM:
    model = TabFM(**_cfg(task))
    gen = torch.Generator().manual_seed(SEED_WEIGHTS)
    with torch.no_grad():
        sd = model.state_dict()
        for key in sorted(sd):   # sorted: init-order independent
            p = sd[key]
            p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * WEIGHT_SCALE)
    return model.eval()
```
For the tabpfn_v2 fixture, replace the `TabFM()` instantiation with a hand-built `onnx.helper` graph (no TabFM model class), but keep `numpy.random.default_rng(seed=42)` for the weight matrix and border values, and verify determinism before committing.

**Safetensors single-metadata key pattern** (fixture.py lines 86-92):
```python
save_file(
    model.state_dict(), str(st_path),
    metadata={"origin": "anofox-tabfm CI fixture, random init, Apache-2.0 "
                        "architecture, license fixture-mit (our weights, "
                        "not Google's)"})
```
Copy this single-key metadata convention for the tabpfn_v2 safetensors file (multiple metadata keys produce nondeterministic sha256 via safetensors-rust HashMap ordering).

**Weight-free strip + assert pattern** (fixture.py lines 158-174, after line 155):
```python
# 5. fixture ships weight-free too
x_export.delete_weight_data(graph_path)
x_export.assert_weight_free(graph_path)
```
For the tabpfn_v2 fixture, build the ONNX graph with `onnx.helper` and use a separate `strip_weights(graph_path)` function that zeroes initializer data and verifies all are zero-filled.

**sha256 + FIXTURE_SHA256 file pattern** (fixture.py lines 111-113):
```python
def sha256_file(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
```
Write a `FIXTURE_SHA256` file (one `filename sha256` line per artifact) in the same format as `test/fixtures/FIXTURE_SHA256`.

**manifest.json pattern** (`test/fixtures/manifest.json` — follow that schema exactly, adding the two new fields):
```json
{
  "preprocessing_profile": "tabpfn_v2",
  "distribution_output": true,
  "license": "fixture-mit"
}
```
`"license": "fixture-mit"` → `IsGated()` returns false; no gate fires on fixture use.

---

### `tools/parity/` — new uv project (MODL-04)

**Analog:** `tools/export_onnx/pyproject.toml` (uv project layout) + `tools/make_fixture/pyproject.toml` (test + entry-point conventions).

**`pyproject.toml` skeleton** (copy from `tools/export_onnx/pyproject.toml` lines 1-45):
```toml
[project]
name = "tabfm-parity"
version = "0.1.0"
description = "anofox-tabfm MODL-04: fixture parity validation (distribution tensor contract)"
requires-python = ">=3.11"
dependencies = [
    "onnx>=1.17",
    "onnxruntime>=1.20",
    "numpy",
]
# No torch dependency — parity only needs onnxruntime + numpy

[dependency-groups]
dev = ["pytest>=8"]

[project.scripts]
check_tabpfn_v2 = "parity.check_tabpfn_v2:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/parity"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
No PyTorch index needed (torch is not a dependency of the parity tool).

**Directory layout** (mirror `tools/export_onnx/` and `tools/make_fixture/`):
```
tools/parity/
├── pyproject.toml
├── src/
│   └── parity/
│       ├── __init__.py
│       └── check_tabpfn_v2.py
└── tests/
    └── test_check_tabpfn_v2.py
```

---

### `test/cpp/test_tabfm_profile_registry.cpp` + `test/cpp/test_tabfm_distribution_decode.cpp`

**Analog:** `test/cpp/test_tabfm_ort_engine.cpp` (Catch2 TU in the same directory).

Key patterns from the Catch2 convention (CLAUDE.md):
- `#include "catch.hpp"` (not `catch2/catch.hpp`)
- Listed in `TABFM_CPP_TEST_SOURCES` in `CMakeLists.txt`
- File-level comment names the spec section being implemented (e.g., `// MGEN-01, MGEN-02`)
- `CHECK(std::abs(actual - golden) < 1e-4)` for floating-point comparisons

---

## Shared Patterns

### Error Handling (apply to all modified/new C++ files)
**Source:** `src/tabfm_ort_engine.cpp` lines 491-495, 538-549; `src/tabfm_weights.cpp` lines 290-294
- Always `InvalidInputException` for bad user input (wrong profile, wrong shape, wrong license).
- Always `InvalidConfigurationException` for configuration-level issues (license not accepted).
- Always `InternalException` for engine contract violations (unexpected ORT output type).
- Every error message names the fixing SET or SELECT (SQL-API §5).

### Namespace Wrapping (apply to all new C++ files)
**Source:** `src/tabfm_predict_agg.cpp` lines 16-17, 31
```cpp
namespace duckdb {
namespace anofox {
namespace {
// anonymous namespace for file-local helpers
} // anonymous namespace
// ...
} // namespace anofox
} // namespace duckdb
```

### Settings `AddExtensionOption` (apply to `src/tabfm_settings.cpp`)
**Source:** `src/tabfm_settings.cpp` lines 77-80
```cpp
config.AddExtensionOption(
    "anofox_tabfm_<name>",
    "<description>",
    LogicalType::BOOLEAN, Value::BOOLEAN(false));
```
Options MUST be registered at Load() time; DuckDB v1.5.4 does not support runtime registration.

### `TryGetCurrentSetting` Lookup (apply to `src/tabfm_weights.cpp`)
**Source:** `src/tabfm_weights.cpp` lines 95-101
```cpp
Value value;
if (!context.TryGetCurrentSetting("anofox_tabfm_<option>", value) || value.IsNull()) {
    return false; // or default
}
return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
```

### `Value::LIST` Construction for Aggregate Output (apply to distribution decode in `src/tabfm_engine.cpp`)
**Source:** `src/tabfm_predict_agg.cpp` lines 85-87 (proba MAP construction shows the strongly-typed Value idiom)
```cpp
// WRONG (untyped — Pitfall 6):
//   Value::LIST(children)
// CORRECT (must match bind-time declared type exactly):
Value::LIST(LogicalType::DOUBLE, children)
```

### Python Fixture Determinism (apply to `tools/parity/` and tabpfn_v2 fixture script)
**Source:** `tools/make_fixture/src/make_fixture/fixture.py` lines 86-141
- Single `__metadata__` key in safetensors (nondeterministic sha256 with 2+ keys).
- Sort all dict/state_dict keys before iterating.
- Double-build and assert byte-equality before committing sha256.

---

## Scaffold-Owned Files (Coordinate — CLAUDE.md Rule #2)

These files require a **coordinated batch edit** — do not modify them independently:

| File | Required Change | Risk |
|---|---|---|
| `CMakeLists.txt` | Add `src/tabfm_profile_registry.cpp` to `EXTENSION_SOURCES`; add two new Catch2 TUs to `TABFM_CPP_TEST_SOURCES` | Build breaks if omitted |
| `src/anofox_tabfm_extension.cpp` | Add explicit `ForceProfileInit()` call in `LoadInternal` to prevent static-init stripping (RESEARCH.md Pitfall 2) | Registry empty at runtime on some linkers |
| `src/include/tabfm_registration.hpp` | Add `void ForceProfileInit();` declaration (if registry force-init is routed through registration) | Compile error if declaration missing |
| `src/tabfm_settings.cpp` | Add per-license `AddExtensionOption` calls for known license ids | Gate lookup fails with `TryGetCurrentSetting` returning false for unknown option names |

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `test/sql/tabfm_distribution.test` | test | request-response | SQL logic test format is unique; follow CLAUDE.md TDD rule and existing `.test` files in `test/sql/` |
| `test/sql/tabfm_profile_registry.test` | test | request-response | Same — no exact analog; copy test structure from any existing `test/sql/*.test` file |

---

## Metadata

**Analog search scope:** `src/`, `src/include/`, `tools/`, `test/`
**Files scanned:** 11 analog files read directly
**Key pitfalls from RESEARCH.md to propagate to plans:**
1. Static-init stripping: add explicit `ForceProfileInit()` in `LoadInternal`
2. ORT output name must match exactly: validate with `tools/parity` before writing C++ decoder
3. `Value::LIST(LogicalType::DOUBLE, children)` — typed overload required
4. `AddExtensionOption` must be at Load() time — no runtime registration in DuckDB v1.5.4
5. Borders in `yhat_dist`: store **raw-space** borders (after affine transform); store **z-space** logits (before softmax)
