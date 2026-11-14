# Settings

Set with `SET name = value;`. The texts below are checked against the code; each setting's own description is in
`SELECT name, description FROM duckdb_settings() WHERE name LIKE 'anofox_tabfm%';`. Settings are not
stored: repeat them in each new process.

| Setting | Default | Meaning |
|---|---|---|
| `anofox_tabfm_accept_hf_license` | `false` | Your acknowledgement of a model's licence. Every model marked *gated* in the [Models table](../../README.md#models) fails to download without it ([licence acknowledgement](../guides/01-getting-started.md#licence-acknowledgement-and-gated-repositories)) |
| `anofox_tabfm_default_model` | `''` | Model id used when `model :=` is not given. `''`: a single SQL-registered model if there is one, else every call must name a model |
| `anofox_tabfm_cache_dir` | `~/.cache/anofox-tabfm` | Weight cache root directory (pre-seed it for air-gapped use) |
| `anofox_tabfm_threads` | usable cores / 2 | ONNX Runtime intra-op thread count for CPU inference |
| `anofox_tabfm_cpu_prepack` | `true` | ONNX Runtime weight prepacking on the CPU: faster matmuls at ~+16% resident memory |
| `anofox_tabfm_context_cache` | `true` | Reuse the encoded labelled context across calls, and across the batches of one call (`opts := MAP{'query_batch': 'N'}`), for models that ship an *exact* prepare/query graph pair: today `tabicl-v2`, on the CPU. A cached context is keyed by the exact preprocessed support rows and labels, in order (a hash finds it, a full compare accepts it), so a hit can only answer for the same training data. Every output row, the context rows' fitted values included, equals the combined graph's. A context is kept the second time it is seen (`anofox_tabfm_context_cache_admit`); until it is kept, a call costs about the same as without the cache. `false` is the kill switch: every call runs the combined graph. A session `SET` applies to statements bound after it (on DuckDB 1.x a statement prepared earlier keeps the value it was prepared with; DuckDB 2.0 re-binds it on `EXECUTE`); `SET GLOBAL anofox_tabfm_context_cache = false` is also read when a statement runs, so it stops prepared statements too. Models without an exact pair, and GPU sessions, always run the combined graph. `tabfm_models()` reports what is cached ([Context cache](02-models-and-weights.md#context-cache), [CONTEXT_CACHE_PLAN.md](../CONTEXT_CACHE_PLAN.md)) |
| `anofox_tabfm_context_cache_memory` | `256MB` | Budget for cached contexts, per loaded model. **Instance-wide**: one cache per loaded model serves every connection, so `SET` writes the value for all of them (`SET SESSION` is refused), and it is read when a statement runs. `'0'` keeps nothing. The least recently used context is evicted first; a context larger than the whole budget serves its own call (every `query_batch` batch of it) and is dropped. A `tabicl-v2` context costs about 2 x 12 x S x 512 floats of attention keys and values plus about 0.2 MB per feature column: about 40 MB at 375 support rows and 100 features, 8 MB at 64 rows and 20 features. Cached contexts are resident memory, which `anofox_tabfm_max_memory` counts; with the cache on, a model also keeps a second ORT session (about +300 MB for `tabicl-v2`) |
| `anofox_tabfm_context_cache_admit` | `second` | When the context cache keeps an encoded context. `'second'`: the second time its support rows are seen within the last 4096 contexts, so single-use contexts, such as one per group of a `GROUP BY`, never take memory; the first sighting encodes, answers identically and drops it, and the third call is the first warm one. `'first'`: keep every context at once (the behaviour before this setting): one warm call sooner for a context you know you will reuse, at the cost of filling the budget with contexts used once. **Instance-wide**, like `anofox_tabfm_context_cache_memory` (`SET SESSION` is refused), and read when a statement runs. `tabfm_models().context_first_seen` counts the first sightings |
| `anofox_tabfm_max_rows` | `10000` | Maximum rows per predict call or group. This is the enforced cap; the per-model `max_rows` in `tabfm_list_models()` is a recommendation only (e.g. set `5000` for `limix-2m`) |
| `anofox_tabfm_max_features` | `500` | Maximum feature columns per predict call (enforced; the per-model `max_features` is a recommendation only) |
| `anofox_tabfm_max_sessions` | `4` | Cap on cached model sessions **across all models, devices and precisions**; beyond it the oldest-loaded session is evicted. Sessions are cached per (model, device, precision) so switching devices does not rebuild multi-GB sessions |
| `anofox_tabfm_max_memory` | `''` | Refuse a predict call when this process's resident memory is already at or above this size (e.g. `'16GB'`) before the call starts, so the failure is a DuckDB exception instead of an OOM kill. `''` disables the check. It does not bound how much a single large call can grow memory by itself |
| `anofox_tabfm_device` | `auto` | `auto` / `cpu` / `cuda` / `rocm` / `mlx` / `coreml` (`migraphx` is an alias for `rocm`). `auto` resolves per model: the best device that model can actually be served on (its plugin present and a graph available), else the CPU. Naming a device is a hard request: it errors rather than falling back. `coreml` needs a source build with `TABFM_FLAVOR=coreml` and `auto` does not reach for it ([GPU](04-gpu.md#choosing-the-device)) |
| `anofox_tabfm_gpu_precision` | `fp32` | `fp32` / `tf32` / `bf16` / `fp16`. `fp32` is strict: a device switch does not change answers. `tf32` is CUDA only; `bf16` / `fp16` quantize the ROCm program and are rejected on CUDA. Label flips against fp32 are not confined to near-ties: validate on your own data ([Precision](04-gpu.md#precision)) |
| `anofox_tabfm_ep_path` | `''` | Directory holding the GPU backend plugins and the runtime libraries they load. `''`: the `runtime` subdirectory of `anofox_tabfm_cache_dir`, where `tabfm_download_runtime` and `tabfm_accelerate` put them |
| `anofox_tabfm_mxr_source` | `''` | Directory of precompiled MIGraphX `.mxr` programs. Before compiling a shape bucket, a matching program here is staged into the cache and reused; `''` always compiles on the device. Programs are arch- and ROCm-version-specific ([ROCm compile time](04-gpu.md#rocm-compile-time)) |
| `anofox_tabfm_trace_level` | `warn` | Diagnostic verbosity: `error` / `warn` / `info` / `debug` / `trace` |
| `anofox_telemetry_enabled` | `true` | Anonymous usage telemetry, see [TELEMETRY.md](../../TELEMETRY.md) |
| `datazoo_banner` | `true` | The once-a-day feedback banner in interactive terminals |

## Options

The trailing `opts` MAP of `tabfm_classify` / `tabfm_regress` (all values `VARCHAR`; an unknown key is an
error that lists the valid ones). Generation options are in [Generation](03-generation.md#options).

| key | values | default | meaning |
|---|---|---|---|
| `task` | `auto`, `classification`, `regression` | fixed by the function | which task to run; `tabfm_classify` and `tabfm_regress` set it for you |
| `n_estimators` | `1` | `1` | ensemble size; only `1` today, a value above 1 is an error (not yet available) |
| `seed` | integer | `42` | RNG seed |
| `output_mode` | `compact`, `detail` | `detail` for `tabfm_classify` | `detail` adds the `proba` map |
| `context_rows` | `''` or a non-negative integer | `''` | accepted and validated, but has no effect today: the full context is always used |
| `softmax_temperature` | a positive number | `0.9` | temperature applied to the class probabilities |
| `query_batch` | a positive integer | all at once | score the test rows at most N per forward pass, which bounds the peak memory of a large call. Same answers as one pass, up to float32 rounding (a batch of a different size can take a different kernel: ~1e-9 on the fixture). With the context cache (`tabicl-v2` on the CPU) the context is encoded once for the whole call; for every other model each batch re-runs the context ([Context cache](02-models-and-weights.md#context-cache)) |
| `context_values` | `true`, `false` | `true` | single-relation form: compute the context rows' fitted values. For `tabfm-v1`, `tabfm-v1-1`, `tabicl-v2` and the `kumo-tabular-*` models that costs a second pass over the context rows (their own outputs are not fitted values); `false` skips it and leaves the context rows' `yhat` as the graph's raw output. Query rows are identical either way. The two-table form, window calls, `tabfm_impute` and `tabfm_generate` never pay it |
| `model` | a model id | | the same as `model :=` |

## Environment variables

| Variable | Used for |
|---|---|
| `DATAZOO_DISABLE_TELEMETRY=1` | turns telemetry off, like `SET anofox_telemetry_enabled = false` |
| `DATAZOO_NO_BANNER=1` | silences the feedback banner, like `SET datazoo_banner = false` |
