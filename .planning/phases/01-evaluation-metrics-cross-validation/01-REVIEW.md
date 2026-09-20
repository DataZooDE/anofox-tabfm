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
  critical: 1
  warning: 1
  info: 2
  total: 4
status: issues_found
---

# Phase 01: Code Review Report (Re-review — post-fix)

**Reviewed:** 2026-09-21
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

Re-review of fixes applied for CR-01 (throw named error for non-constant `avg`),
CR-02/WR-04 (Welford online variance for R² + exact zero constant-target check),
WR-02 (matching CAST types in cross_validate SQL), and WR-03 (zero-alloc
`string_t` access in Update hot loops). WR-01 was intentionally accepted as a
documented deviation.

**Confirmed correct:**

- **CR-01 fix** (`F1Finalize`/`AUCFinalize` throw): lines 344-352 and 737-741 now
  throw `InvalidInputException` with actionable messages when `avg_mode` is empty at
  finalize time. The 2-arg overload bind lambda also throws at bind time. Both paths
  are correct.

- **CR-02/WR-04 fix** (Welford R²): The Welford update at lines 283-288 is the
  standard Knuth two-pass form (`delta = a - mean_before`, `mean += delta/(n+1)`,
  `delta2 = a - mean_after`, `M2 += delta * delta2`, then `n++`). The Chan parallel
  combine at lines 309-314 is correct. The `ss_tot == 0.0` exact-equality guard at
  line 336 is appropriate: Welford M2 is exactly `0.0` for constant inputs at any
  magnitude, so the exact check is safe and the `1e-12` threshold from the previous
  implementation is no longer needed.

- **WR-02 fix** (seed CAST type): The cross_validate body now emits
  `CAST(seed AS BIGINT)` and `CAST(k AS UBIGINT)` at every hash/modulo expression,
  matching `tabfm_fold_assign`'s body.

**New finding:** The WR-03 fix introduced a critical type-confusion bug in
`AccuracyUpdate`. The raw `GetData<string_t>()` reinterpretation is only safe when
the backing buffer holds `string_t` data, but `tabfm_accuracy` is registered with
`LogicalType::ANY`, which admits non-VARCHAR inputs. This is described in CR-01 below.

One additional warning and two info items from the previous review remain as
originally classified; no new regressions were found in any other file.

---

## Critical Issues

### CR-01: `AccuracyUpdate` reinterprets `ANY`-typed buffer as `string_t` — crash or silent wrong result on non-VARCHAR inputs

**File:** `src/tabfm_metrics_classification.cpp:100-102`

**Issue:** The WR-03 fix replaced the `GetValue(i).ToString()` comparison in
`AccuracyUpdate` with a direct `UnifiedVectorFormat::GetData<string_t>()` access to
avoid per-row heap allocation. The fix is safe for VARCHAR inputs, but `tabfm_accuracy`
is registered with `{LogicalType::ANY, LogicalType::ANY}` (line 1032), which causes
DuckDB to call `AccuracyUpdate` without inserting an implicit VARCHAR cast for any
column type.

When the user calls `tabfm_accuracy(an_integer_col, another_integer_col)`, the
underlying buffer holds `int32_t` values laid out as flat arrays. `GetData<string_t>()`
reinterprets that memory as an array of `string_t` structs. `string_t` is a 16-byte
union: if the length field (read from the int's bytes) is <= 12 the struct treats it
as inlined chars and compares them; if the length field exceeds 12 it treats the
payload as a `{char *ptr, int32_t size}` — and dereferences the `ptr` field which is
garbage from the integer value. The result is either a wrong equality comparison (silent
data corruption) or a null/wild pointer dereference (crash / undefined behaviour).

The `F1Update` path (lines 280-283) correctly uses `string_t::GetString()` after the
same cast, but is protected because F1/precision/recall inputs are declared `VARCHAR`
(lines 1054, 1082, 1111). `AccuracyUpdate` has no such protection.

**Fix — Option A (recommended):** Change the registered input types from `ANY` to
`VARCHAR`. DuckDB will insert an implicit cast at bind time, so integer, boolean, and
date columns are automatically converted to their string representation before
`AccuracyUpdate` is called. This is semantically correct since accuracy compares
predicted labels as strings regardless of storage type:

```cpp
// In RegisterClassificationMetrics (~line 1032), change:
AggregateFunction fn("anofox_tabfm_accuracy",
                     {LogicalType::ANY, LogicalType::ANY}, LogicalType::DOUBLE, ...);
// to:
AggregateFunction fn("anofox_tabfm_accuracy",
                     {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE, ...);
```

With Option A the raw `GetData<string_t>` access in `AccuracyUpdate` is safe and no
change to the function body is required beyond also hoisting the `GetData` calls outside
the loop (see IN-01).

**Fix — Option B (no registration change):** Restore the safe comparison path inside
the loop and keep `ANY` for broad type support:

```cpp
// Replace lines 100-104:
Value actual_val    = inputs[0].GetValue(i);
Value predicted_val = inputs[1].GetValue(i);
if (actual_val == predicted_val) {
    state.correct++;
}
```

Option B reintroduces the heap allocation that WR-03 was meant to eliminate, so
Option A is preferred.

---

## Warnings

### WR-01: Hash formula verification tests use uncast integer literals — may not match macro formula

**File:** `test/cpp/test_tabfm_crossval.cpp:115`
**File:** `test/sql/tabfm_crossval.test:96`

**Issue:** Both formula-verification tests compare `tabfm_fold_assign` output against
`(hash(id, 42) % 4)::INTEGER` and `(hash(id, 7) % 3)::INTEGER` respectively, where
`42` and `7` are untyped integer literals (DuckDB infers `INTEGER` for unadorned integer
literals in SQL). The `fold_assign` macro body explicitly uses
`hash(_cv_rk, CAST(seed AS BIGINT))`, and the cross_validate body (after the WR-02 fix)
emits `CAST(seed AS BIGINT)` in generated SQL.

The `tabfm_crossval.cpp` comment at lines 192-196 explicitly warns that
`hash(x, 42::INTEGER) != hash(x, 42::BIGINT)` in general because DuckDB's `hash()` is
type-sensitive. If the two produce different values for these seeds, the test would fail
against a correct implementation. If they happen to produce the same value for these
specific seeds, the test passes vacuously and does not actually verify the macro formula.
Either way the tests do not reliably document or gate the contract they claim to verify.

**Fix:** Use an explicit `BIGINT` cast matching the macro:

```sql
-- test/sql/tabfm_crossval.test (line 96):
SELECT count(*) FROM (
  SELECT id, fold_id,
    (hash(id, CAST(7 AS BIGINT)) % 3)::INTEGER AS expected_fold
  FROM tabfm_fold_assign('cv_rows', 3, 'id', 7)
  WHERE fold_id <> expected_fold
)
```

```cpp
// test/cpp/test_tabfm_crossval.cpp (line 115):
auto res = qry(con, R"(
    SELECT count(*) FROM (
      SELECT id, fold_id,
        (hash(id, CAST(42 AS BIGINT)) % 4)::INTEGER AS expected
      FROM tabfm_fold_assign('fa4', 4, 'id', 42)
      WHERE fold_id <> expected
    )
)");
```

---

## Info

### IN-01: `AccuracyUpdate` re-fetches `GetData` pointer on every loop iteration

**File:** `src/tabfm_metrics_classification.cpp:100-101`

**Issue:** `actual_raw` and `predicted_raw` are obtained inside the loop body on every
iteration via `UnifiedVectorFormat::GetData<string_t>(actual_data)`. `GetData` returns
the same pointer on every call (it is a single pointer-cast with no side effects), so
the repeated calls are no-ops, but they obscure the intent and add unnecessary function
call overhead in the hot path.

This is secondary to CR-01: once the input types are restricted to `VARCHAR` (Option A),
hoisting the pointers out of the loop is the natural cleanup.

**Fix:** Hoist the `GetData` calls above the loop:

```cpp
auto *actual_raw    = UnifiedVectorFormat::GetData<string_t>(actual_data);
auto *predicted_raw = UnifiedVectorFormat::GetData<string_t>(predicted_data);
for (idx_t i = 0; i < count; i++) {
    // ...NULL skip...
    if (actual_raw[aidx] == predicted_raw[pidx]) {
        state.correct++;
    }
}
```

---

### IN-02: R² registration comment states `|SS_tot| < 1e-12` but code now uses exact `== 0.0`

**File:** `src/tabfm_metrics_regression.cpp:630`

**Issue:** The comment above the `tabfm_r2` registration block reads:
```
// Constant-target convention (sklearn): |SS_tot| < 1e-12 → 1.0 (perfect) or 0.0 (imperfect).
```
After the CR-02/WR-04 fix the code uses `if (ss_tot == 0.0)` (line 336) — exact
equality, not a tolerance threshold. The `< 1e-12` comment is a leftover from the
previous implementation and is now wrong. Future readers may be confused about whether
a tolerance should be present or whether the exact check is intentional.

**Fix:** Update the registration comment to match the current implementation:

```cpp
// Constant-target convention (sklearn): SS_tot == 0.0 (Welford M2 is exactly 0 for
// constant-magnitude targets at any scale) → 1.0 (perfect prediction) or 0.0
// (imperfect). Never returns NaN or Inf (T-01-03-01).
```

---

_Reviewed: 2026-09-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
