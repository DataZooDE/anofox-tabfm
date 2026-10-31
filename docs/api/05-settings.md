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
| `anofox_tabfm_context_cache` | `false` | Encode the labelled context once and reuse it across calls, for models that ship a split graph pair (prepare/query). It pays off when the same context is scored more than once (chunked scoring, repeated queries against a fixed training table) and costs extra on a single call. Test-row predictions match the combined graph; the fitted values on *context* rows differ, because the query half no longer sees a context row's own label. Inert for a model that ships no pair |
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
| `model` | a model id | | the same as `model :=` |

## Environment variables

| Variable | Used for |
|---|---|
| `DATAZOO_DISABLE_TELEMETRY=1` | turns telemetry off, like `SET anofox_telemetry_enabled = false` |
| `DATAZOO_NO_BANNER=1` | silences the feedback banner, like `SET datazoo_banner = false` |
