# GPU scenarios — the checks a unit test cannot reach

Each file is SQL you run against a build with a GPU backend plugin available. Prepend the
`SET anofox_tabfm_ep_path=...` line each script names, then:

```bash
{ echo "SET anofox_tabfm_ep_path='$PWD/vendor/anofox-ggml-adapter/build/ggml-rocm';"; \
  cat tools/gpu_test/scenarios/<file>.sql; } | ./build/release/duckdb
```

They exist because the GPU bugs were found by running the real SQL surface on real hardware, and none of them
was reachable from the C++ suite: those tests build device lists by hand and never touch discovery, the session
cache, or the aggregate.

**Every scenario prints `*_SERVED_BY`. Read it first.** "cpu and gpu agree" is also exactly what a silent CPU
fallback prints; that is what made this repo's first GPU equivalence result a cpu-vs-cpu comparison reporting a
perfect score.

| scenario | what it is the only cover for |
|---|---|
| `cuda_catalog.sql` (+ `_classification`, `_regression`) | the whole catalog on a GPU against the CPU, one (model, task) per fresh process, driven by `tools/gpu_test/gpu_catalog.sh` (which rewrites the device to `cuda`, `rocm` or `mlx`). Every row prints `served_by`; `n_gpu = 0` means the GPU call errored and the log has the message. Results per platform: `docs/GGML_PLAN.md` (P9) and `docs/api/04-gpu.md`. |
| `ggml_plugin_gpu.sql` | the ggml GPU plugin through the SQL surface on the committed fixture. |
| `a_concurrency.sql` | four predicts in one query as parallel pipelines, finalizing on one device (the path the per-device lock guards). A lock bug here corrupts results rather than erroring, so equality with the sequential answer is the assertion. |
| `b_regress.sql` | the regression task on a GPU: continuous output, so closeness rather than label agreement. |
| `c_wide.sql` | 20 feature columns, wider than every other scenario. |
| `rocm_fixture.sql` | the **weight-free** ROCm smoke test: the committed fixture's external-data graph (`graph_ext_fixture.onnx`, registered as `classification_ext_graph`) serves in seconds, no download or license. Run before pushing ROCm-path changes. |
| `user_workflow.sql` | a workflow shaped like a user's: `read_csv_auto` with inferred types, categorical columns, NULLs in **both** features and label, then joining predictions back and aggregating by a business dimension. |
| `nonfinite_targets.sql` | a NaN-marked missing target answers exactly like a NULL-marked one on the GPU, and the GPU agrees with the CPU. Weight-free (the fixture's external-data graph); classification only. |
| `auto_per_model.sql` | per-model `auto` resolving to a *different* device per model in one session: `mitra` to the GPU, a fixture registered without an ext graph to the CPU, each asserted by `SERVED_BY` plus a `tabfm_backends()` invariant. |
| `mlx_all_models.sql` | every registered model on MLX against CPU. MLX runs the model's own ONNX graph through its own interpreter, so coverage is a property of the op table, not of a per-model list: an op-table change can break exactly one model and nothing else would notice. |
| `mlx_stress.sql` | MLX at 4000x30, mid-session device switches, reduced precision (MLX keeps bf16/fp16), and awkward shapes (single class, zero-variance feature, NULLs). macOS/arm64 only. |

Notes worth keeping:

- **These files are not run by CI, so they rot silently, and they did.** The feature-column guard made a context
  relation carrying a column the `test` relation lacks a hard error; both MLX scenarios built one `ctx` holding
  both label columns and a `qry` holding neither, so every call failed at the first block and nothing reported
  it. When a guard is added to the predict surface, grep this directory before assuming it only affects user SQL.
- A scenario that compares only the `test` rows cannot see a change to the *fitted* rows. If a change moves
  output rows the scenario does not select, the scenario passes and means nothing.
- `customers.csv` has `yes`/`no` in the label column, which `read_csv_auto` infers as BOOLEAN, so predictions
  come back `true`/`false`. A synthetic `range()` table never shows you that.
- On CUDA and ROCm only `fp32` (default) and `tf32` exist; `bf16`/`fp16` are refused there (MLX keeps them).
