---
phase: 03-proper-scoring-rules-cross-model-comparison
fixed_at: 2026-09-22T00:00:00Z
review_path: .planning/phases/03-proper-scoring-rules-cross-model-comparison/03-REVIEW.md
iteration: 2
findings_in_scope: 1
fixed: 1
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-22
**Source review:** `.planning/phases/03-proper-scoring-rules-cross-model-comparison/03-REVIEW.md`
**Iteration:** 2

**Summary:**
- Findings in scope: 1
- Fixed: 1
- Skipped: 0

## Fixed Issues

### WR-01: `tabfm_cross_validate` does not single-quote-escape `data` in generated SQL

**Files modified:** `src/tabfm_crossval.cpp`, `test/sql/tabfm_crossval.test`
**Commit:** dd0c7ba
**Applied fix:** Applied `replace(CAST(data AS VARCHAR), '''', '''''')` at all three `data`
interpolation sites inside the `tabfm_cross_validate` macro body:

- Training subquery (`~line 212`): `FROM (FROM ' || replace(CAST(data AS VARCHAR), '''', '''''') || '))`
- Test subquery (`~line 228`): same pattern in the EXCLUDE subquery
- Join-back subquery (`~line 241`): same pattern in the `_cv_folds` label-recovery join

Previously each site used bare `CAST(data AS VARCHAR)`, which would produce malformed
SQL string literals if `data` resolved to a value containing a single quote. The fix
matches identically the escape already applied to `target` in this macro and to `data`
in `tabfm_compare_models` (CR-02, CMP-01, T-03-06). Each changed line is annotated
with a comment citing WR-01 and the CR-02 precedent.

Added a test in `test/sql/tabfm_crossval.test` that passes a `data` argument containing
`' OR 1=1 --`; the query errors with `tabfm_cross_validate` (a meaningful catalog or
inference error), not a SQL syntax error, confirming the escape is in effect.

**Verification:** Tier 1 — file re-read confirmed all three sites fixed and surrounding
code intact. Tier 2 — `make debug` incremental build 64/64 targets clean (pre-existing
SFINAE warnings in DuckDB headers only, no errors in modified files). Full test suite
`make test_debug`: 433 assertions in 16 test cases, all green. Verification ran inside
isolated worktree `.claude/worktrees/rf-03-115177-1790110553`; results are reproducible
from the main checkout after the fast-forward merge.

---

_Fixed: 2026-09-22_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
