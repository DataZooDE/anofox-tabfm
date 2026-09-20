---
phase: 01-evaluation-metrics-cross-validation
fixed_at: 2026-09-21T01:00:00Z
review_path: .planning/phases/01-evaluation-metrics-cross-validation/01-REVIEW.md
iteration: 2
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report (Iteration 2)

**Fixed at:** 2026-09-21
**Source review:** `.planning/phases/01-evaluation-metrics-cross-validation/01-REVIEW.md`
**Iteration:** 2

**Summary:**
- Findings in scope: 2 (1 Critical, 1 Warning)
- Fixed: 2
- Skipped: 0

**Verification ran in:** isolated worktree `.claude/worktrees/rf-01-3745728-1789943765` on branch
`gsd-reviewfix/01-3745728`. Build and test execution ran against the shared build tree at
`build/debug/`. All affected test suites passed; a pre-existing ASAN heap-use-after-free in DuckDB's
own `BlockAllocator::~BlockAllocator()` teardown is unrelated to any change in this phase and fires
in every run including pre-fix ones.

---

## Fixed Issues

### CR-01: `AccuracyUpdate` reinterprets `ANY`-typed buffer as `string_t`

**Files modified:** `src/tabfm_metrics_classification.cpp`, `test/sql/tabfm_metrics_classification.test`
**Commit:** `7876493`
**Applied fix (Option A from review):**

1. Changed `tabfm_accuracy` registration from `{LogicalType::ANY, LogicalType::ANY}` to
   `{LogicalType::VARCHAR, LogicalType::VARCHAR}` (~line 1045 in `RegisterClassificationMetrics`).
   DuckDB now inserts an implicit cast at bind time for any non-VARCHAR column (INTEGER, BIGINT,
   DATE, etc.), so `AccuracyUpdate`'s `GetData<string_t>()` always sees a real `string_t` buffer
   and never raw integer/float bytes.

2. Hoisted the `GetData<string_t>()` pointer fetches outside the loop body (was being called
   per-iteration as a no-op; also resolves IN-01 opportunistically).

3. Updated the struct-level doc-comment to document the VARCHAR registration invariant and the
   CR-01 safety guarantee.

4. Updated the registration comment to explain why `ANY` was rejected.

5. Added a new test case to `test/sql/tabfm_metrics_classification.test` that passes INTEGER
   columns to `tabfm_accuracy` and asserts the correct value, proving the implicit VARCHAR cast
   is working and that the former string_t reinterpretation bug no longer occurs:
   ```
   -- actual=[1,2,1] INTEGER, predicted=[1,1,1] INTEGER => 2/3 correct => 0.666667
   SELECT round(tabfm_accuracy(actual, predicted), 6) FROM preds_int
   -- expected: 0.666667
   ```

**Verification:**
- Tier 1: re-read modified sections — fix present, surrounding code intact.
- Tier 2: `make debug` succeeded — 0 new compilation errors.
- `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest test/sql/tabfm_metrics_classification.test`
  — **54 assertions passed** (including the new INTEGER test).
- `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest "[tabfm]"`
  — **1520 assertions in 77 test cases passed**.

---

### WR-01: Hash formula verification tests use uncast integer literals

**Files modified:** `test/sql/tabfm_crossval.test`, `test/cpp/test_tabfm_crossval.cpp`
**Commit:** `25ec200`
**Applied fix:**

1. `test/sql/tabfm_crossval.test` (~line 96): changed `(hash(id, 7) % 3)::INTEGER` to
   `(hash(id, CAST(7 AS BIGINT)) % 3)::INTEGER` to match `tabfm_fold_assign`'s macro body
   which uses `CAST(seed AS BIGINT)`.

2. `test/cpp/test_tabfm_crossval.cpp` (~line 115): changed `(hash(id, 42) % 4)::INTEGER`
   to `(hash(id, CAST(42 AS BIGINT)) % 4)::INTEGER` for the same reason.

3. Added explanatory comments to both locations documenting that DuckDB's `hash()` is
   type-sensitive (`hash(x, 42::INTEGER) != hash(x, 42::BIGINT)` in general).

**Verification:**
- Tier 1: re-read modified sections — fixes present, surrounding code intact.
- `DATAZOO_DISABLE_TELEMETRY=1 ./build/debug/test/unittest test/sql/tabfm_crossval.test`
  — **33 assertions passed**.
- Full `[tabfm]` suite: 1520 assertions passed (see CR-01 above).

---

## Info Findings (not in fix scope for this iteration)

### IN-01: `AccuracyUpdate` re-fetches `GetData` pointer on every loop iteration

Resolved opportunistically as part of the CR-01 fix: the `GetData<string_t>()` calls were hoisted
outside the loop in the same commit.

### IN-02: R² registration comment states `|SS_tot| < 1e-12` but code now uses exact `== 0.0`

Not in scope (Info severity). No change applied.

---

_Fixed: 2026-09-21_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
