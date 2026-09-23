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
| `rocm_fixture.sql` | the **weight-free** ROCm smoke test: the committed fixture's MIGraphX variant compiles + serves in seconds, no download/license. Run before pushing ROCm-path changes. | `FIXTURE_SERVED_BY=rocm:0`, 9 rows |
| `registered_model_gpu.sql` | a **registered** model carrying its own `migraphx_graph` served by the ROCm plugin — the model-provided GPU-graph dispatch (GPU_HARDENING_PLAN P3) that bundled-graph runs never touch. Paths inside are machine-local (the weight cache); adjust `base_dir` before running. | `REGISTERED_SERVED_BY=rocm:0`, `PATHS_DISAGREE=0` |
| `user_workflow.sql` | a workflow shaped like a user's: `read_csv_auto` with inferred types, categorical columns, NULLs in **both** features and label, then joining predictions back and aggregating by a business dimension. | `cpu` then `rocm:0`, **0 disagreements**, 25 rows joined |
| `auto_per_model.sql` | per-model `auto` resolving to a *different* device per model in one session — `mitra` and `tabdpt` to the GPU, `tabpfn-v2` to CPU because no MIGraphX graph exists for it, each asserted by `SERVED_BY` plus a `tabfm_backends()` invariant. Re-check the negative model whenever a conversion lands: it silently stops being a negative. | `rocm:0` / `rocm:0` / `cpu`, `AUTO_IS_PER_MODEL=true`, `AUTO_NEVER_UNSUPPORTED=true` |
| `mlx_all_models.sql` | every registered model on MLX against CPU. MLX runs the model's own ONNX graph through its own interpreter, so coverage is a property of the op table, not of a per-model list — an op-table change can break exactly one model and nothing else would notice. | macOS/arm64 only |
| `mlx_stress.sql` | MLX at 4000x30, mid-session device switches, reduced precision, and awkward shapes (single class, zero-variance feature, NULLs). | macOS/arm64 only |

Notes worth keeping:

- **These files are not run by CI, so they rot silently — and they did.** The
  feature-column guard made a context relation carrying a column the `test`
  relation lacks a hard error. Both MLX scenarios built one `ctx` holding both
  label columns and a `qry` holding neither, so after that guard landed every
  call in both files failed at the *first* block. Nothing reported it, because
  nothing ran them. They now build one context table per task. When a guard is
  added to the predict surface, grep this directory before assuming it only
  affects user SQL.
- A scenario that compares only the `test` rows cannot see a change to the
  *fitted* rows. `mlx_all_models.sql` compared query rows exclusively until
  `TABDPT_FITTED` was added — and the fitted-values fix changed nothing else.
  If a change moves output rows the scenario does not select, the scenario
  passes and means nothing.

- `customers.csv` has `yes`/`no` in the label column, which `read_csv_auto`
  infers as BOOLEAN — so predictions come back `true`/`false`. Comparing
  against `'yes'` only works because DuckDB casts it. That is the kind of thing
  a synthetic `range()` table never shows you.
- bf16 flips a small number of near-tie argmaxes; the same comparisons at
  fp32 agree with CPU exactly. fp32 is the default since Track A of
  `docs/GPU_HARDENING_PLAN.md` (these scenarios SET bf16 explicitly, and now
  measure the opt-in rather than the default). Real-shaped data flips fewer
  than synthetic — 0 here versus 7/100 on `c_wide`.
- The first run of any new shape bucket costs a MIGraphX compile (~27 min
  measured for both T4096 and H64). Budget for it, or stay inside a bucket that
  is already cached.
