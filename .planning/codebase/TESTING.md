# Testing Patterns

**Analysis Date:** 2026-09-19

## Test Framework

**Runner:**
- SQL tests: DuckDB sqllogictest format (inherited from DuckDB test infrastructure)
  - Config: `test/sql/*.test` files; no separate config file (part of DuckDB's test runner)
  - Command: `make test_debug` or `make test_release` 
- C++ unit tests: Catch2 framework (compiled into `build/debug/test/unittest` binary)
  - Config: CMakeLists.txt lists source files in `TABFM_CPP_TEST_SOURCES`
  - Include: `#include "catch.hpp"` (DuckDB's vendored Catch2)
  - Command: `./build/debug/test/unittest "test/*"` or single file `./build/debug/test/unittest test/sql/settings.test`

**Assertion Library:**
- Catch2: `REQUIRE()`, `CHECK()`, `SECTION()` for structure
- SQL tests: DuckDB's sqllogictest assertions (`query`, `statement ok/error`)

**Run Commands:**
```bash
make debug                      # Build debug flavor (cpu), run tests with telemetry disabled
make release                    # Build release flavor (cpu), run tests with telemetry disabled
./build/debug/test/unittest "test/sql/settings.test"  # Single SQL test file
./build/debug/test/unittest test/cpp/test_tabfm_ort_engine.cpp  # Single C++ test file (via catch2 pattern)
DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest   # Run all tests, explicit telemetry disable
```

## Test File Organization

**Location:**
- SQL tests: `test/sql/*.test` (co-located with source; mirrors DuckDB structure)
- C++ unit tests: `test/cpp/test_*.cpp` (same directory, same naming convention)
- Test fixtures: `test/fixtures/` (committed, deterministic)
  - Subdirectories: `test/fixtures/ort_engine/` (MLP fixture), `test/fixtures/` (main fixture), `test/fixtures/regression/` (regression fixture)

**Naming:**
- SQL: `test_<component>.test` (e.g., `tabfm_classify.test`, `settings.test`)
- C++: `test_<module>.cpp` (e.g., `test_tabfm_ort_engine.cpp`, `test_tabfm_preprocess.cpp`)
- Fixtures: deterministic, named by model type: `manifest.json`, `graph.onnx`, `model.safetensors`, `golden.json`

**Structure:**
```
test/
├── sql/
│   ├── settings.test              # Settings validation, defaults
│   ├── tabfm_classify.test        # Classification end-to-end
│   ├── tabfm_regress.test         # Regression end-to-end
│   ├── tabfm_devices.test         # Device discovery
│   ├── tabfm_weights.test         # Weight download/lifecycle
│   ├── tabfm_lifecycle.test        # Extension load/unload
│   ├── telemetry.test             # Telemetry opt-out
│   └── ...
├── cpp/
│   ├── test_tabfm_ort_engine.cpp   # Dtype hooks, shape buckets, session creation
│   ├── test_tabfm_preprocess.cpp   # Feature scaling, imputation
│   ├── test_tabfm_ensemble.cpp     # Ensemble tree decoding
│   ├── test_tabfm_safetensors.cpp  # Reader parsing
│   ├── test_tabfm_manifest.cpp     # Manifest loading
│   ├── test_tabfm_scaffold.cpp     # Parity tests (WS-A vs WS-C)
│   └── ...
└── fixtures/
    ├── manifest.json              # Classification fixture model manifest
    ├── graph.onnx                 # Weight-free ONNX graph
    ├── model.safetensors          # Random-init weights
    ├── golden.json                # Expected preprocess + predict outputs
    ├── regression/
    │   └── manifest.json
    └── ort_engine/
        ├── mlp_graph.onnx
        ├── mlp_weights.f32
        └── mlp_expected.json
```

## Test Structure

**Suite Organization (SQL):**
```sql
# name: test/sql/settings.test
# description: anofox_tabfm_* settings — defaults, validation, aliases (SQL-API §4)
# group: [anofox_tabfm]

require anofox_tabfm
statement ok
LOAD anofox_tabfm

# --- Defaults -----------------------------------------------------------------
query I
SELECT current_setting('anofox_tabfm_device')
----
auto
```

**Suite Organization (C++):**
```cpp
#include "catch.hpp"
#include "tabfm_ort_engine.hpp"

namespace {
  string TestCppDir() { ... }  // helpers
  vector<char> ReadFileBytes(const string &path) { ... }
}  // anonymous namespace

TEST_CASE("tabfm_ort_engine: dtype sizes and cast hooks", "[tabfm][ort_engine]") {
  REQUIRE(TabFMDtypeSize(TabFMTensorDtype::F32) == 4);
  
  SECTION("f32 -> bf16 (GPU bf16 profile) and exact upcast back") {
    const float values[] = {0.0f, 1.0f, -1.0f, ...};
    auto bf16 = CastF32ToBF16(values, 6);
    REQUIRE(bf16[1] == 0x3f80);  // 1.0f
  }
}
```

**Patterns:**
- SQL tests use comments to section (e.g., `# --- Defaults ---`) and organize by feature
- Comments name the spec section being tested (e.g., `# SQL-API §4`)
- C++ tests use `SECTION()` for sub-cases within a `TEST_CASE()`
- Helper functions in anonymous namespace (file-scoped, testable imports)

## Mocking

**Framework:** No external mocking library; tests use real fixtures

**Patterns:**
- **Fixtures over mocks:** All tests load committed, deterministic fixtures from `test/fixtures/`
- **Minimal setup:** Tests use real DuckDB connection + real ONNX Runtime (no stubbing)
- **Golden outputs:** Precomputed expected results (golden.json, mlp_expected.json) compared against actual
- **Environment control:** Tests disable telemetry via `DATAZOO_DISABLE_TELEMETRY=1` in Makefile targets

**Example: dtype casting (test_tabfm_ort_engine.cpp:236-273):**
```cpp
TEST_CASE("tabfm_ort_engine: dtype sizes and cast hooks", "[tabfm][ort_engine]") {
  // No mocks — test the actual cast functions against known bit patterns
  SECTION("f32 -> bf16 ... and exact upcast back") {
    const float values[] = {0.0f, 1.0f, -1.0f, 3.140625f, 65504.0f, -0.15625f};
    auto bf16 = CastF32ToBF16(values, 6);
    REQUIRE(bf16[1] == 0x3f80);  // 1.0f (exact bit representation)
  }
}
```

**What to Mock:** Not applicable (no mocking used; see "What NOT to Mock" below)

**What NOT to Mock:**
- ONNX Runtime: use real sessions with fixture graphs (ORT_LOGGING_LEVEL_ERROR suppresses noise)
- DuckDB: use real DB connections + test tables; no connection mocking
- Fixture loading: load real files from `test/fixtures/` (FileExists check + ReadFileBytes)
- Random data generation: use committed random-init weights (no runtime generation to ensure determinism)

## Fixtures and Factories

**Test Data:**
```cpp
// In test_tabfm_ort_engine.cpp
struct MlpFixture {
  vector<char> graph;           // ONNX graph bytes
  vector<char> weights;         // Injection arena
  struct TensorSpec { 
    string name; 
    vector<int64_t> shape; 
    idx_t offset; 
    idx_t nbytes; 
  };
  vector<TensorSpec> tensors;
  // ... input/output arrays, dimensions, metadata
};

MlpFixture LoadMlpFixture() {
  auto dir = OrtEngineFixtureDir();
  fixture.graph = ReadFileBytes(dir + "/mlp_graph.onnx");
  fixture.weights = ReadFileBytes(dir + "/mlp_weights.f32");
  // ... parse mlp_expected.json for tensors, x, y, cat_mask, expected logits
  return fixture;
}
```

**Location:**
- Compiled into binary: `test/fixtures/*.json`, `test/fixtures/*.onnx`, `test/fixtures/*.safetensors`, `test/fixtures/*.f32`
- Generated: `test/fixtures/ort_engine/mlp_*.onnx`, `mlp_expected.json` (produced by `tools/gen_mlp_fixture.py`)
- Never regenerated at test time (CI verifies pinned sha256)

**Determinism:**
- All fixtures are COMMITTED to git with pinned sha256
- Random-init weights (not Google-licensed) regenerated if needed via `tools/` scripts
- Golden outputs (golden.json) precomputed offline

## Coverage

**Requirements:** No explicit coverage target (not enforced in CI)

**View Coverage:** Not configured (code inspection + sqllogictest comprehensiveness is the standard)

## Test Types

**Unit Tests (C++):**
- Scope: Single module in isolation (e.g., dtype casting, shape buckets)
- Approach: Fast, use committed MLP fixture or synthetic data
- Examples:
  - `test_tabfm_ort_engine.cpp`: dtype conversions, ORT API version checking, MIGraphX shape buckets
  - `test_tabfm_preprocess.cpp`: feature type classification, numerical transforms, outlier removal
  - `test_tabfm_safetensors.cpp`: file parsing, tensor extraction
- Location: `test/cpp/test_*.cpp`

**Integration Tests (SQL):**
- Scope: Full SQL function + preprocessing + inference pipeline
- Approach: Use real DuckDB + real fixture model; assert plumbing works
- Examples:
  - `tabfm_classify.test`: single-table mode, two-table mode, error cases (all-NULL target, task mismatch)
  - `tabfm_regress.test`: numerical predictions, two-table form, subselect training relation
  - `settings.test`: default values, validation, aliasing (migraphx -> rocm)
  - `tabfm_weights.test`: download, cache, license gate
- Location: `test/sql/*.test`
- Telemetry: Always run with `DATAZOO_DISABLE_TELEMETRY=1`

**End-to-End Tests (SQL):**
- Scope: Full application: load extension, set manifest, predict on real fixture
- Assertions: Correctness (probabilistic invariants) + determinism (two runs agree)
- Examples:
  - `tabfm_classify.test:29-51`: count predictions, verify class set, check probability distribution sums to 1, confirm determinism
  - `tabfm_regress.test:26-56`: similar for regression (no proba column, predictions are finite)
- Output validation: 
  - Classification: `yhat IN class_set`, `0 < yhat_score <= 1`, `sum(proba values) == 1`
  - Regression: `isfinite(yhat)`, `yhat_score IS NULL`, `proba IS NULL`

## Common Patterns

**Async Testing:** N/A (no async code in extension; C++ is synchronous, SQL is single-threaded per connection)

**Error Testing (SQL):**
```sql
statement error
SELECT * FROM tabfm_regress('houses','price', opts := MAP{'n_estimators':'4'})
----
<REGEX>:.*ensemble|M3|n_estimators.*
```

**Error Testing (C++):**
```cpp
SECTION("f32 -> bf16 (GPU bf16 profile)") {
  const float values[] = {...};
  auto bf16 = CastF32ToBF16(values, 6);
  REQUIRE(bf16[1] == 0x3f80);  // assertion failure -> test fails
}
```

**Fixture Loading Pattern:**
```cpp
MlpFixture LoadMlpFixture() {
  auto dir = OrtEngineFixtureDir();
  auto spec_bytes = ReadFileBytes(dir + "/mlp_expected.json");
  YyDoc doc(spec_bytes);
  auto *root = doc.Root();
  REQUIRE(root);  // file must parse
  // ... extract dims, tensors, inputs, expected logits
  return fixture;
}
```

**Determinism Testing:**
```sql
query I
SELECT count(*) FROM (
  SELECT a.f1, a.yhat AS y1, b.yhat AS y2
  FROM tabfm_classify('t','label') a JOIN tabfm_classify('t','label') b USING (f1,f2)
) WHERE y1 <> y2
----
0
```

**Relative Error Tolerance (numerical):**
```cpp
void RequireLogitsClose(const vector<float> &actual, const vector<float> &expected, 
                         double rtol, double atol) {
  REQUIRE(actual.size() == expected.size());
  double max_rel = 0.0;
  for (size_t i = 0; i < actual.size(); i++) {
    const double abs_diff = std::fabs(double(actual[i]) - double(expected[i]));
    if (abs_diff <= atol) continue;
    const double rel = abs_diff / std::max(std::fabs(double(expected[i])), 1e-12);
    max_rel = std::max(max_rel, rel);
  }
  INFO("max relative delta " << max_rel << " vs rtol " << rtol);
  REQUIRE(max_rel <= rtol);
}
```

## Test Naming and Tagging (Catch2)

**Tag convention:** `[tabfm][module_name]`
- Example: `TEST_CASE("...", "[tabfm][ort_engine]")`
- Allows filtering: `./build/debug/test/unittest "[tabfm]"` runs all anofox-tabfm tests

## Spec Coverage Requirements

All tests must cite the spec section they verify (Red-Green TDD: write test first, name the spec):
- **SQL-API:** Test files comment with the section number (e.g., `# SQL-API §4`)
- **HLD:** File headers document design decision and HLD section (e.g., `# HLD §9, S02`)
- **Spikes:** Reference the spike (S01-S06) that backs the design
- Error path tests are first-class: every failure mode in SQL-API §5 catalog needs a test

---

*Testing analysis: 2026-09-19*
