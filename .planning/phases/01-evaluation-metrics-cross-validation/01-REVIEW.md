---
phase: 01-evaluation-metrics-cross-validation
reviewed: 2026-09-21T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - src/tabfm_metrics_classification.cpp
  - src/tabfm_metrics_regression.cpp
  - src/tabfm_crossval.cpp
  - src/include/tabfm_metrics_classification.hpp
  - src/include/tabfm_metrics_regression.hpp
  - src/include/tabfm_crossval.hpp
  - src/include/anofox_function_alias.hpp
  - src/include/tabfm_registration.hpp
  - src/anofox_tabfm_extension.cpp
  - CMakeLists.txt
  - test/sql/tabfm_metrics_classification.test
  - test/sql/tabfm_metrics_regression.test
  - test/sql/tabfm_crossval.test
  - test/cpp/test_tabfm_metrics.cpp
  - test/cpp/test_tabfm_metrics_regression.cpp
  - test/cpp/test_tabfm_crossval.cpp
  - tools/golden/generate_metric_fixtures.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-09-21
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

This phase implements classification metrics (CMET-01..06), regression metrics (RMET-01..04), and k-fold cross-validation macros (CV-01..04) as DuckDB C++ extension aggregate functions and SQL table macros.

The overall structure is sound: aggregate triplets follow the established pattern from `tabfm_predict_agg.cpp`, NULL-skip via `UnifiedVectorFormat` is consistent, heap-owning states have correct `StateDestroy` callbacks registered, and the formula golden values match sklearn references. The confusion-matrix and cross-validate macros properly apply `replace(col, '"', '""')` quoting to the target/column identifier parameters.

Two blockers require fixes before ship: (1) `tabfm_f1`, `tabfm_precision`, `tabfm_recall`, and `tabfm_roc_auc` silently return NULL instead of computing results when the `avg` parameter is a non-constant SQL expression (column reference or subquery); (2) the R² "constant-target" guard uses a fixed `1e-12` threshold that fails silently — returning a catastrophically wrong value — when the target has large magnitude and the computational `SS_tot = sum_y² − n·ȳ²` formula suffers catastrophic cancellation.

Four warnings and three info items round out the findings.

## Critical Issues

### CR-01: F1/Precision/Recall/ROC-AUC silently return NULL for non-constant `avg` expressions

**File:** `src/tabfm_metrics_classification.cpp:215-233,317-339,566-586,708-728`

**Issue:** `BindF1Metric` (and the analogous `AUCBind`) calls `ExpressionExecutor::EvaluateScalar` inside a bare `catch (...)` block that swallows any exception. When `arguments[2]` is not a constant expression — for example a column reference such as `tabfm_f1(actual, predicted, avg_col)` — `EvaluateScalar` throws, `avg_mode` stays as an empty string, and `F1BindData{avg="", metric='f'}` is stored. At `Finalize` time, the check `if (avg_mode.empty()) { SetNull; continue; }` silently returns NULL for every group, with no error message to the caller.

A user who writes:
```sql
SELECT tabfm_f1(actual, predicted, avg_col) FROM t
```
where `avg_col` is a VARCHAR column containing `'macro'` for all rows gets a NULL result with no explanation. The same issue applies to `tabfm_precision`, `tabfm_recall`, and `tabfm_roc_auc`.

**Fix:** Either (a) remove the empty-string `SetNull` fallback in Finalize and replace it with a thrown `InvalidInputException`, or (b) read `avg` from `inputs[2]` at Update time when `bind_data->avg` is empty, so runtime values are respected:

```cpp
// Option (a): in F1Finalize, replace the silent NULL path:
if (avg_mode.empty()) {
    throw InvalidInputException(
        "tabfm_f1: 'avg' must be a constant expression — "
        "pass avg := 'micro', 'macro', or 'weighted'");
}
```

Option (a) is simpler and more informative. The `catch (...)` swallowing in `BindF1Metric` should stay (non-constant expressions cannot be evaluated at bind time), but the Finalize path must not silently discard valid data.

---

### CR-02: R² computational formula causes catastrophic wrong results for large-magnitude targets

**File:** `src/tabfm_metrics_regression.cpp:306-319`

**Issue:** `R2Finalize` computes `SS_tot` using the single-pass algebraic form:
```cpp
double ss_tot = state.sum_y2 - n * y_bar * y_bar;
```
This form is numerically unstable for large-magnitude constant or near-constant targets. When all actual values are `5e7` (for example), `sum_y2 = 3 * (5e7)² = 7.5e15` and `n * y_bar² = 7.5e15`. Due to IEEE 754 rounding the subtraction produces a small non-zero value — say `1e-3` — rather than `0.0`. The guard `if (std::abs(ss_tot) < 1e-12)` does NOT trigger (because `1e-3 > 1e-12`), so the code falls into the else branch and computes:
```cpp
r2 = 1.0 - ss_res / 1e-3;   // ss_res could be ~1e14 → r2 = -1e17
```
The result is a catastrophically wrong negative value, not `0.0` as the spec requires, with no error or NULL return.

The issue is independent of the 1e-12 threshold: the threshold only protects against the exact-zero case. Cancellation at scale produces a positive but tiny spurious `ss_tot` that bypasses the guard while `ss_res` is large.

**Fix:** Switch to an online mean-shifted accumulation (Welford) or at minimum a two-pass formula that separates `SS_tot` accumulation from raw sums. The simplest correct approach is Welford:

```cpp
// Replace sum_y / sum_y2 with:
struct R2State {
    double  mean;      // running mean of actual (Welford M)
    double  M2;        // running sum of (actual - mean)^2 (Welford S)
    double  sum_res;   // sum of (actual - predicted)^2
    int64_t n;
};

// In R2Update:
double delta  = a - state.mean;
state.mean   += delta / static_cast<double>(state.n + 1);  // increment n AFTER
double delta2 = a - state.mean;
state.M2     += delta * delta2;
state.sum_res += resid * resid;
state.n++;

// In R2Finalize:
double ss_tot = state.M2;  // exact, no cancellation
```

Note: `Combine` still requires care (parallel Welford); an alternative is tracking `sum_y` / `sum_y2` AND adding a scale-factor guard (`ss_tot > std::abs(sum_y2) * 1e-14`), but the Welford path is cleaner.

---

## Warnings

### WR-01: Telemetry fires at extension-load time for CV macros, not at per-use time

**File:** `src/tabfm_crossval.cpp:302-313`

**Issue:** `RegisterCVMacroWithAlias` calls `CaptureFunctionExecution(alias_name.c_str())` at extension registration time (once per `LOAD anofox_tabfm`), not per SQL call to the macro. This fires telemetry even if the user never invokes `tabfm_fold_assign` or `tabfm_cross_validate`. CLAUDE.md rule 3 states: "every user-facing function calls `CaptureFunctionExecution` once per execution (bind or global-state init)". Macros have no bind callback, but firing at load time conflates "extension loaded" with "function used".

The code comment at line 304 acknowledges the limitation but does not justify the false telemetry signal. For consistency with the contract, consider logging only at first actual call (e.g., store a `std::once_flag` guarded by the macro invocation) or documenting this as a known deviation.

**Fix:** Accept as-is if the deviation is intentional, OR add a `TABFM_FIRST_USE_TELEMETRY` pattern with an `std::once_flag` so telemetry fires on the first actual SQL invocation:

```cpp
// In the macro body (as a SQL comment or via a sentinel scalar function injected
// once), or via a static bool per-macro that flips on first Finalize/Update call
// if macros supported bind callbacks.
```

---

### WR-02: Hash seed type inconsistency between `tabfm_fold_assign` and the hash expressions generated by `tabfm_cross_validate`

**File:** `src/tabfm_crossval.cpp:83,192-193`

**Issue:** `tabfm_fold_assign`'s macro body uses `CAST(seed AS BIGINT)` as the second argument to `hash()`:
```sql
hash(_cv_rk, CAST(seed AS BIGINT)) % CAST(k AS UBIGINT)
```
`tabfm_cross_validate`'s generated SQL emits the seed as a bare integer literal (without a `BIGINT` cast):
```sql
CAST(CAST(seed AS BIGINT) AS VARCHAR)  →  '42'
-- resulting generated SQL:
hash(id, 42) % 5   -- 42 is INTEGER, not BIGINT
```
DuckDB's `hash()` function is type-sensitive: `hash(x, 42::INTEGER)` and `hash(x, 42::BIGINT)` may produce different outputs. If they do, a user who pre-computes fold assignments with `tabfm_fold_assign('t', k, 'id', seed)` to inspect the split, then runs `tabfm_cross_validate('t', 'label', 'id', k:=k, seed:=seed)`, will get different fold assignments than expected — silently, with no error.

The `tabfm_crossval.test` test at line 92–99 that verifies `fold_assign`'s formula against `(hash(id,7)%3)::INTEGER` (where `7` is an uncast integer) would catch this if DuckDB hashes the two types differently. If this test has never been executed in CI, the inconsistency could be silent.

**Fix:** Emit `CAST(seed AS BIGINT)` in the generated SQL to match `fold_assign`:
```cpp
// In cross_validate, change:
|| CAST(CAST(seed AS BIGINT) AS VARCHAR)
// to emit:
|| 'CAST(' || CAST(CAST(seed AS BIGINT) AS VARCHAR) || ' AS BIGINT)'
```
Apply the same fix for `k` (use `CAST(k AS UBIGINT)` in the generated modulo expression).

---

### WR-03: `AccuracyUpdate` calls `Vector::GetValue(i)` inside the hot loop — O(1) but allocates a `Value` object per row

**File:** `src/tabfm_metrics_classification.cpp:98-100`

**Issue:** After NULL-checking via `UnifiedVectorFormat` (which is correct and efficient), `AccuracyUpdate` calls `inputs[0].GetValue(i)` and `inputs[1].GetValue(i)` to compare as strings. `GetValue(i)` performs a dictionary-lookup and allocates a `Value` object per call. For ANY/VARCHAR inputs this creates two heap-allocated `Value` objects per row, contrasting with the regression metrics that use `UnifiedVectorFormat::GetData<double>()[idx]` for zero-allocation data access.

For large classification datasets this results in significant unnecessary allocation pressure. The same pattern exists in `F1Update` (lines 276-277), `LogLossUpdate` (line 442), `AUCUpdate` (line 605-606), and `ECEUpdate` (line 855-856).

**Fix:** For string-typed columns, extract the raw `string_t` directly from the `UnifiedVectorFormat` data pointer to avoid `Value` heap allocation:
```cpp
// Replace:
auto actual_val = inputs[0].GetValue(i);
// With:
auto *actual_data_raw = UnifiedVectorFormat::GetData<string_t>(actual_data);
string_t actual_str = actual_data_raw[aidx];
// Compare: actual_str == predicted_str (string_t supports ==)
```

---

### WR-04: `R2Finalize` "constant-target" check asymmetry: uses `ss_res < 1e-12` but `ss_res` accumulates `(a-p)^2` which can only be zero when predictions are perfect — the threshold adds no safety margin

**File:** `src/tabfm_metrics_regression.cpp:312-315`

**Issue:** The constant-target guard is:
```cpp
if (std::abs(ss_tot) < 1e-12) {
    r2 = (ss_res < 1e-12) ? 1.0 : 0.0;
}
```
`ss_res` is a sum of squared residuals, so it is always non-negative. The check `ss_res < 1e-12` works correctly for exactly-zero residuals (e.g., `a=p=5.0` → `ss_res=0`), but the threshold `1e-12` is ad hoc: for large but identical values such as `a=p=1e6`, the sum of squared residuals is exactly `0.0` (IEEE subtraction of equal doubles), so `ss_res < 1e-12` holds correctly. However, for `a=5.0000001, p=5.0`, `ss_res = (5.0000001-5.0)^2 ≈ 1e-14 < 1e-12` — this is an imperfect prediction but the code returns `1.0` (perfect).

This is an off-by-one in the threshold: any residual magnitude less than `~3.16e-7` is classified as perfect. For typical engineering data this is unlikely to matter, but it can produce incorrect results with near-constant-target data.

**Fix:** Use a stricter threshold or compare with zero exactly:
```cpp
r2 = (ss_res == 0.0) ? 1.0 : 0.0;
```
`ss_res` is accumulated as a sum of `double * double` products; it is exactly 0.0 only when all residuals are exactly 0.0 at the floating-point level. Using `== 0.0` is safe here.

---

## Info

### IN-01: Redundant double `CAST` in `tabfm_cross_validate` macro body

**File:** `src/tabfm_crossval.cpp:172`

**Issue:** The `range()` argument is:
```sql
range(CAST(CAST(k AS BIGINT) AS BIGINT), ...)
```
The inner `CAST(k AS BIGINT)` already produces a BIGINT; the outer `CAST(... AS BIGINT)` is a no-op. This adds noise without benefit.

**Fix:** Replace with a single cast:
```sql
range(CAST(k AS BIGINT), ...)
```

---

### IN-02: `generate_metric_fixtures.py` does not emit RMET golden values via actual sklearn calls — regression section is comments-only

**File:** `tools/golden/generate_metric_fixtures.py:63-92`

**Issue:** The docstring header at lines 63-92 lists RMET golden values as free-form comments, but the executable body of the script (lines 93+) only calls sklearn for CMET-01..06. The regression golden values (RMET-01..04) are never verified programmatically by the script. A future sklearn version change affecting `mean_absolute_percentage_error` or `median_absolute_error` would update the comments but not trigger a fixture regeneration check.

**Fix:** Add a `# RMET-01..04` executable section using `sklearn.metrics.root_mean_squared_error`, `mean_absolute_error`, `r2_score`, `mean_absolute_percentage_error`, and `median_absolute_error` on the 5-row fixture to make the golden verification symmetric with the classification section.

---

### IN-03: `make_con()` in `test/cpp/test_tabfm_metrics.cpp` shares a static `DuckDB` instance across all test cases

**File:** `test/cpp/test_tabfm_metrics.cpp:16-21`

**Issue:** The `make_con()` helper declares `db` as `static duckdb::DuckDB`, meaning all `TEST_CASE` functions in that translation unit share a single in-memory database. Each test creates its own `Connection` but they all attach to the same database, so tables created in one test (e.g., `preds`, `preds_null`, `ll_preds`, `auc_preds`) persist across all tests within the process. If Catch2 runs tests in parallel or if table names collide across tests, results may be non-deterministic.

The regression and crossval test files create a fresh `DuckDB db(nullptr)` per `TEST_CASE`, which is the correct pattern.

**Fix:** Remove the `static` qualifier from `db` in `make_con()` (or restructure to create per-test instances):
```cpp
static duckdb::Connection make_con() {
    duckdb::DuckDB db(nullptr);    // <-- remove static
    duckdb::Connection con(db);
    REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());
    return con;
}
```
Note: this causes `db` to be destroyed when `make_con()` returns, which invalidates `con`. The proper fix is to pass `db` by reference or return a `{DuckDB, Connection}` pair, as done in the regression test file.

---

_Reviewed: 2026-09-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
