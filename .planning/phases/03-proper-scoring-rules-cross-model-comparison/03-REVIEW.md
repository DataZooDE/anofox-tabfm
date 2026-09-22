---
phase: 03-proper-scoring-rules-cross-model-comparison
reviewed: 2026-09-22T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/tabfm_scoring.cpp
  - src/include/tabfm_scoring.hpp
  - src/tabfm_crossval.cpp
  - src/anofox_tabfm_extension.cpp
  - src/include/tabfm_registration.hpp
  - CMakeLists.txt
  - test/cpp/test_tabfm_scoring.cpp
  - test/sql/tabfm_scoring.test
  - tools/parity/src/parity/crps_reference.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-09-22
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed the proper scoring rule aggregates (CRPS, log-score, interval-score), the
cross-validation and compare-models macros, and supporting infrastructure. The CRPS
closed-form math is correct and verified against the Python reference; Case A/B/C
formulas match the non-uniform-bin derivation. The log-score NLL (density form) is
correct. Aggregate state design (Combine/Finalize, empty-group NULL, NULL-skip in
Update) is sound. PSR-04 bind-gates are correctly registered and the error messages
name the remedy.

Two blockers were found: a NaN bypass of the coverage validation in `IScoreBind`
that silently produces NaN output, and an unescaped `data` parameter in the
`tabfm_compare_models` macro that breaks generated SQL for table names containing
a single-quote. Three warnings cover a metric-parameter SQL injection surface in
`tabfm_cross_validate`, silent mishandling of NULL elements inside logit/border
lists, and the telemetry-at-load-time pattern for CV macros violating the
per-invocation contract in CLAUDE.md rule 3.

---

## Critical Issues

### CR-01: NaN coverage bypasses IScoreBind validation — silent NaN output

**File:** `src/tabfm_scoring.cpp:655`

**Issue:** The guard `if (coverage <= 0.0 || coverage >= 1.0)` does not catch
`coverage = NaN` because IEEE 754 NaN comparisons always return `false`. A user can
pass `NaN` through an expression such as `tabfm_interval_score(a, d, 0.0/0.0)` or
a column that evaluates to `NaN`. The validation passes silently, `IScoreBindData`
stores `NaN`, and `ComputeIntervalScore` produces `alpha = NaN`, `q_lo = q_hi = NaN`.
`DistributionQuantile` with `q = NaN` falls through its loop (because `NaN <= cumsum`
is always false) and returns `borders[K]`, so `l = u = borders[K]`. The final score
is `(0) + (2/NaN) * ... = NaN`, returned silently with no error or NULL.

The PSR-03 bind validation is the only guard. Downstream `mean(NaN)` propagates NaN
through the aggregate mean, poisoning comparisons silently.

**Fix:**
```cpp
// tabfm_scoring.cpp:655 — replace the guard with:
if (std::isnan(coverage) || coverage <= 0.0 || coverage >= 1.0) {
    throw InvalidInputException(
        "tabfm_interval_score: coverage must be in (0, 1) exclusive — "
        "got %g. Use coverage := 0.9 for 90%% nominal coverage.",
        coverage);
}
```

---

### CR-02: `data` parameter not single-quote-escaped in `tabfm_compare_models`

**File:** `src/tabfm_crossval.cpp:310,319`

**Issue:** In the `COMPARE_MODELS_MACRO` body, the `target` parameter is correctly
single-quote-escaped with `replace(CAST(target AS VARCHAR), '''', '''''')`, but the
`data` parameter is injected bare inside a single-quoted string literal:

```sql
' FROM tabfm_regress(''' || CAST(data AS VARCHAR) || ''','
```

If `data` resolves to a string containing a single quote (e.g., the user passes
`'my''table'`), the generated SQL becomes `tabfm_regress('my'table', ...)` which is a
SQL parse error. The same pattern appears at line 319 (tabfm-v1 branch). The `target`
parameter at lines 311 and 320 has the correct escape applied; `data` does not.

This asymmetry means the security property documented at T-01-04-01 / CMP-01 is
only half-applied. The test suite exercises double-quote injection on `target` but
has no test for single-quote in `data`.

**Fix:**
```sql
-- Replace both occurrences (lines 310 and 319):
-- Before:
' FROM tabfm_regress(''' || CAST(data AS VARCHAR) || ''','
-- After:
' FROM tabfm_regress(''' || replace(CAST(data AS VARCHAR), '''', '''''') || ''','
```

Apply the same fix to the cross-validate macro wherever `data` is embedded inside
a generated SQL string literal (lines 198, 213, 225 each use `FROM (FROM data)` which
is a SQL-expression context and is not affected; lines 310/319 are the only string-literal
occurrences in compare_models).

---

## Warnings

### WR-01: `metric` parameter injected as function identifier without sanitization

**File:** `src/tabfm_crossval.cpp:179-181`

**Issue:** The `metric` parameter in `CROSS_VALIDATE_MACRO` is interpolated directly
as a SQL function name with no quoting or whitelist check:

```sql
coalesce(nullif(CAST(metric AS VARCHAR), ''),
  CASE WHEN ... THEN 'tabfm_rmse' ELSE 'tabfm_accuracy' END) ||
'(orig.__cv_actual, p.yhat) AS m'
```

A caller passing `metric := 'tabfm_rmse); DROP TABLE my_data; SELECT tabfm_rmse'`
would inject arbitrary SQL into the fold query generated by `query()`. While DuckDB
in a local extension context is under the authenticated user's control, this is
still an injection surface for any application that passes user-supplied metric names
to `tabfm_cross_validate`. The spec documents T-01-04-01 SQL-injection mitigation
for `target` but makes no equivalent statement for `metric`.

**Fix:** Whitelist the allowed metric values at macro execution time. The simplest
safe approach is to validate `metric` against a known set:

```sql
-- In the macro body, replace the coalesce(...) expression with:
CASE
  WHEN metric IS NULL OR CAST(metric AS VARCHAR) = ''
    THEN CASE WHEN CAST(task AS VARCHAR) = 'regression'
              THEN 'tabfm_rmse' ELSE 'tabfm_accuracy' END
  WHEN CAST(metric AS VARCHAR) IN (
    'tabfm_rmse', 'tabfm_accuracy', 'tabfm_crps',
    'tabfm_log_score', 'tabfm_interval_score',
    'tabfm_mae', 'tabfm_r2')
    THEN CAST(metric AS VARCHAR)
  ELSE error('tabfm_cross_validate: unknown metric ' || CAST(metric AS VARCHAR))
END
```

---

### WR-02: NULL elements inside logit/border lists are silently coerced to 0.0

**File:** `src/tabfm_scoring.cpp:333-336,449-452,592-595`

**Issue:** `DoubleValue::Get` is implemented as `GetValueUnsafe<double>()` which
calls `D_ASSERT(type == DOUBLE)` then returns `value_.double_` without checking
`is_null`. A NULL `DOUBLE` value in a list has `is_null = true` and `value_.double_
= 0.0` (default-constructed), so `DoubleValue::Get` returns `0.0` silently.

The three Update functions (CRPSUpdate, LogScoreUpdate, IScoreUpdate) check whether
the top-level list children are NULL (line 317, 436, 579) but do not check individual
element nullability before calling `DoubleValue::Get(logit_vals[k])` or
`DoubleValue::Get(border_vals[k])`. A user constructing
`{'logits': [NULL, 0.0, 0.0]::DOUBLE[], 'borders': [0.0, 1.0, 2.0, 3.0]::DOUBLE[]}`
would have the first logit silently treated as 0.0, producing a numerically valid but
semantically wrong CRPS/log-score without any error.

The `tabfm_predict_agg` output path never produces NULL list elements, so this is
not triggered in production. However, manual struct construction by users can trigger
it.

**Fix:** Add a per-element null check in all three Update loops:
```cpp
for (size_t k = 0; k < K; k++) {
    if (logit_vals[k].IsNull() || border_vals[k].IsNull()) {
        goto skip_row;  // or: continue to a label that skips this row
    }
    logits[k]  = DoubleValue::Get(logit_vals[k]);
    borders[k] = DoubleValue::Get(border_vals[k]);
}
if (border_vals[K].IsNull()) { goto skip_row; }
borders[K] = DoubleValue::Get(border_vals[K]);
```

Alternatively, skip the row (same as the malformed-distribution path) rather than
silently proceeding.

---

### WR-03: Telemetry fires once at extension load time for CV macros, not per user invocation

**File:** `src/tabfm_crossval.cpp:393`

**Issue:** `RegisterCVMacroWithAlias` calls
`PostHogTelemetry::Instance().CaptureFunctionExecution(alias_name.c_str())` once
during extension registration (i.e., at `LOAD anofox_tabfm` time), not at each user
invocation of `tabfm_fold_assign`, `tabfm_cross_validate`, or `tabfm_compare_models`.

CLAUDE.md rule 3 states: "every user-facing function calls
`CaptureFunctionExecution("<short_name>")` once per execution (bind or global-state
init, not per chunk)." The aggregate scoring functions (tabfm_crps, etc.) correctly
fire at bind time in their Bind callbacks. SQL macros do not have a bind callback, so
this pattern differs necessarily — but the comment at line 391-392 acknowledges the
gap rather than resolving it.

The practical consequence is that if ten users each call `tabfm_cross_validate` 100
times, the telemetry records only N load events (one per connection that loads the
extension), not 1000 invocation events. Usage tracking for these macros is
systematically under-reported.

**Fix:** The standard workaround for macro bind-time telemetry in DuckDB is to wrap
the macro body in a scalar function call that fires telemetry as a side-effect, or
to use a table function shim that fires telemetry in its bind. If neither is
acceptable, document this as a known limitation in the comment.

---

## Info

### IN-01: Dead code — fallback at `ComputeLogScore` line 211-213 is unreachable

**File:** `src/tabfm_scoring.cpp:211-213`

**Issue:** The comment at line 211 reads "y is exactly on borders[K] and wasn't
caught by the loop (rounding edge case)." However, the loop at lines 197-209 checks
`b_lo <= y && y <= b_hi` for each bin. For the final bin (i = K-1), `b_lo =
borders[K-1]` and `b_hi = borders[K]`. If `y == borders[K]` and the last bin has
positive width, the condition `borders[K-1] <= borders[K] && borders[K] <= borders[K]`
is true and the loop returns from inside. The only scenario where the loop exits
without returning is if every bin has zero width (`wi < 1e-300`); in that case,
`y == borders[K]` implies `borders[K-1] == borders[K]` (all same value), the
`b_lo <= y <= b_hi` is still true for every bin, but the `wi < 1e-300` guard fires
and returns `kMaxPenalty`. So the post-loop fallback is never reached.

**Fix:** Remove or replace with `D_ASSERT(false)` (unreachable marker):
```cpp
// After the for loop:
D_ASSERT(false && "ComputeLogScore: unreachable post-loop fallback");
return kMaxPenalty;
```

---

### IN-02: PSR-04 bind-gate gap for 3-arg `(DOUBLE, DOUBLE, DOUBLE)` call to `tabfm_interval_score`

**File:** `src/tabfm_scoring.cpp:784-826`

**Issue:** Three overloads are registered for `tabfm_interval_score`:
1. `(DOUBLE, STRUCT)` — normal 2-arg
2. `(DOUBLE, STRUCT, DOUBLE)` — 3-arg with explicit coverage
3. `(DOUBLE, DOUBLE)` — PSR-04 bind-gate

If a user calls `tabfm_interval_score(actual, scalar_yhat, 0.9)` where `scalar_yhat`
is a plain `DOUBLE` (3-arg case), DuckDB will try overloads 1 and 2 (arity mismatch
for overload 1; arg2 type mismatch for overload 2), then try overload 3 (arity
mismatch). The result is a generic DuckDB "no function found" error instead of the
named PSR-04 error that names `output_mode := 'distribution'` as the fix.

**Fix:** Add a 4th overload `(DOUBLE, DOUBLE, DOUBLE)` with a bind-gate that throws
the same named error:
```cpp
AggregateFunction fn_point3(
    "anofox_tabfm_interval_score",
    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
    LogicalType::DOUBLE,
    IScoreStateSize, IScoreStateInit, IScoreUpdate, IScoreCombine, IScoreFinalize,
    nullptr,
    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
        -> unique_ptr<FunctionData> {
        throw InvalidInputException(
            "tabfm_interval_score: second argument must be a distribution "
            "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
            "Use output_mode := 'distribution' in tabfm_regress.");
        return nullptr;
    },
    nullptr);
set.AddFunction(fn_point3);
```

---

_Reviewed: 2026-09-22_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
