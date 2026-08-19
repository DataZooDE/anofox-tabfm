# GPU scenarios — the checks a unit test cannot reach

Each file is SQL you run against a build with a GPU backend plugin available.
Prepend the `SET anofox_tabfm_ep_path=...` line each script names, then:

```bash
{ echo "SET anofox_tabfm_ep_path='$PWD/build/debug/extension/anofox_tabfm';"; \
  cat tools/gpu_test/scenarios/<file>.sql; } | ./build/debug/duckdb
```

They exist because every GPU bug on this branch was found by running the real
SQL surface on real hardware, and none of them was reachable from the C++
suite: those tests build device lists by hand and never touch discovery, the
session cache, or the aggregate.

**Every scenario prints `*_SERVED_BY`. Read it first.** "cpu and gpu agree" is
also exactly what a silent CPU fallback prints — that is not hypothetical, it
is what made this repo's first GPU equivalence result a cpu-vs-cpu comparison
reporting a perfect score.

| scenario | what it is the only cover for | result on gfx1201 (real TabFM v1 weights) |
|---|---|---|
| `a_concurrency.sql` | four predicts in one query as parallel pipelines, finalizing on one device — the path `#42`'s per-device lock guards. A lock bug here corrupts results rather than erroring, so equality with the sequential answer is the assertion, not "it didn't crash". | `rocm:0`, 400 rows, **0 mismatches** |
| `b_regress.sql` | the regression task on a GPU at all. Continuous output, so closeness rather than label agreement. | `rocm:0`, max abs diff `0.004837`, corr `0.999943` |
| `c_wide.sql` | 20 feature columns, so the H64 shape bucket instead of the H16 every other run uses. | `rocm:0`, 7 disagreements / 100 (bf16) |
| `user_workflow.sql` | a workflow shaped like a user's: `read_csv_auto` with inferred types, categorical columns, NULLs in **both** features and label, then joining predictions back and aggregating by a business dimension. | `cpu` then `rocm:0`, **0 disagreements**, 25 rows joined |

Notes worth keeping:

- `customers.csv` has `yes`/`no` in the label column, which `read_csv_auto`
  infers as BOOLEAN — so predictions come back `true`/`false`. Comparing
  against `'yes'` only works because DuckDB casts it. That is the kind of thing
  a synthetic `range()` table never shows you.
- bf16 (the default) flips a small number of near-tie argmaxes; the same
  comparisons at fp32 agree with CPU exactly. Real-shaped data flips fewer than
  synthetic — 0 here versus 7/100 on `c_wide`.
- The first run of any new shape bucket costs a MIGraphX compile (~27 min
  measured for both T4096 and H64). Budget for it, or stay inside a bucket that
  is already cached.
