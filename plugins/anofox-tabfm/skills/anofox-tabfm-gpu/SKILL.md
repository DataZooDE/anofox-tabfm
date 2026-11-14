---
name: anofox-tabfm-gpu
description: >
  GPU acceleration and device control for the anofox_tabfm DuckDB extension:
  NVIDIA CUDA, AMD ROCm (MIGraphX) and Apple MLX backend plugins loaded at
  runtime, device discovery, per-model backend support, precision modes, and the
  ROCm first-run compile. Use when making predictions faster on a GPU, running
  `tabfm_accelerate` / `tabfm_download_runtime` / `tabfm_gpu_precompile`, reading
  `tabfm_devices()` / `tabfm_backends()`, setting `anofox_tabfm_device`,
  `anofox_tabfm_gpu_precision`, `anofox_tabfm_ep_path` or
  `anofox_tabfm_mxr_source`, checking which device actually served a query, or
  troubleshooting a GPU that is not used.
version: 2026.11.01
user-invocable: false
---

# Anofox TabFM — GPU Cheat Sheet

**Extension:** anofox_tabfm v2026.11.01 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `tabfm_*` and `anofox_tabfm_*` (identical)

A GPU is not a separate build. Every published build is the CPU build, and each accelerator is a plugin it
loads at runtime: `cuda` (NVIDIA, Linux x64, its own ONNX Runtime GPU libraries), `rocm` (AMD, Linux x64, a
direct MIGraphX backend) and `mlx` (Apple Silicon, macOS arm64). No vendor runtime is linked; CUDA / cuDNN or
ROCm come from the system.

## Critical gotchas

1. **Never trust "CPU and GPU agree".** A silent CPU fallback gives the same answers, only slower. Check
   what served the call: `SELECT model, task, device FROM tabfm_models() WHERE loaded;` (`cpu`, `cuda:0`,
   `rocm:0`, `mlx:0`).
2. **`auto` never promises an accelerator a model cannot use.** `anofox_tabfm_device = 'auto'` (default)
   picks, per model, the best device whose plugin is present and that has a graph for it, else the CPU.
   `tabfm_backends()` gives the choice and the reason. One exception to "GPU is faster": repeated scoring
   against the same context with `tabicl-v2` is fastest on the CPU, whose context cache a GPU session does
   not use; pin `SET anofox_tabfm_device = 'cpu';` there.
3. **Naming a device is a hard request.** `SET anofox_tabfm_device = 'cuda'` (or `'rocm'`, `'mlx'`) errors
   when the model cannot be served there instead of falling back.
4. **ROCm compiles per shape bucket.** The first predict in a (GPU arch, precision, rows x features) bucket
   compiles a MIGraphX program: minutes (about 25 for `tabfm-v1`, 16 for `tabpfn-v3`, 2–9 for most others).
   Later predicts in the bucket are sub-second. Precompile off the query path with `tabfm_gpu_precompile`.
5. **Precision defaults to strict `fp32`** everywhere; `bf16` / `fp16` (ROCm) flip labels on some rows, and
   not only near-ties. Validate on your own data before switching.
6. **Not every model runs on every backend.** CUDA and MLX refuse `nori-30m` and `exaone-tabular` (no GPU
   graph); ROCm serves every built-in model but caps
   `limix-2m`'s bucket size. CoreML is dropped (MLX replaces it).

## Set up

```sql
CALL tabfm_accelerate();   -- find the GPU, fetch its plugin (and runtime), verify it loads
```

Returns `step, status, detail` rows, including what is still ahead (MIGraphX's first compile, `brew install
mlx mlx-c` on macOS). Re-running re-verifies without re-downloading. That is all: with `device = 'auto'` the
next predict uses the GPU for every model it can serve.

By hand:

```sql
CALL tabfm_download_runtime('cuda');   -- 'cuda', 'rocm' or 'mlx': rows file, bytes, status
SET anofox_tabfm_device = 'cuda';
```

Plugins land in the `runtime` subdirectory of `anofox_tabfm_cache_dir`; `SET anofox_tabfm_ep_path = '<dir>'`
only to point somewhere else. A plugin is checked against the extension's plugin ABI on load, so a plugin left
over from an older release is refused (and `auto` uses the CPU); `CALL tabfm_accelerate()` fetches the matching
one.

## Inspect

```sql
SELECT * FROM tabfm_devices();
SELECT model, task, device, backend, supported, reason FROM tabfm_backends() WHERE NOT supported;
```

| function | columns | meaning |
|---|---|---|
| `tabfm_devices()` | `device_id, ep, name, arch, vram_total, vram_free, driver, usable` | always `cpu`, plus every CUDA, ROCm or MLX device found; `usable = false` when present but unsupported |
| `tabfm_backends()` | `model, task, device, backend, supported, reason` | per model, task and device: can it be served, and why not. `supported` is `NULL` while the weights are not downloaded |
| `tabfm_models()` | `..., device, ...` | the device serving each loaded session right now |

Verify a query really ran on the GPU:

```sql
SELECT customer_id, yhat
FROM tabfm_classify('history', 'churned', test := 'prospects', keep := ['customer_id'], model := 'mitra');
SELECT model, task, device FROM tabfm_models() WHERE loaded;
```

## Precompile (ROCm)

```sql
CALL tabfm_gpu_precompile('classification', model := 'mitra');
CALL tabfm_gpu_precompile('classification', 1000, 50, model := 'mitra');   -- task, rows, features
```

Returns `task, rows, features, device, status, model, bucket_rows, bucket_features, seconds, reason`;
`status` is `compiled`, `cached`, `not needed` (CPU, CUDA, MLX) or `refused` (over the model's ROCm ceiling,
`reason` says why). The compiled `.mxr` persists in the weight cache across sessions. A fleet can share
compiles: `SET anofox_tabfm_mxr_source = '<dir>'` stages matching precompiled programs from that directory
(they are GPU-arch and ROCm-version specific). CUDA has no such cost (about 30 s cold start).

`limix-2m` on ROCm: buckets above 4096 rows x 16 features (on a 16 GiB GPU; proportionally less with less
VRAM) are refused before compiling, because MIGraphX would abort the process. Use the CPU for larger tables.

## Precision

| `anofox_tabfm_gpu_precision` | backends | meaning |
|---|---|---|
| `fp32` (default) | all | strict: switching device does not change answers (TF32 rounding off on CUDA) |
| `tf32` | CUDA | TF32 tensor-core matmuls, fp32 storage |
| `bf16`, `fp16` | ROCm | quantized program: about 2x faster on RDNA4, half the VRAM; validate first |

A mode the backend cannot honour is an error, never a silent fp32 run.

## Choosing per query

```sql
SET anofox_tabfm_device = 'cpu';     -- force the CPU, e.g. for a table above a ROCm ceiling
SET anofox_tabfm_device = 'auto';    -- back to per-model routing
```

Sessions are cached per (model, device, precision), up to `anofox_tabfm_max_sessions` (default 4) across all
of them; switching back and forth does not rebuild them. `CALL tabfm_unload();` frees GPU memory.

## Platforms

| backend | platforms | models |
|---|---|---|
| CPU | Linux x64 / arm64, macOS arm64, Windows x64 | all |
| CUDA | Linux x64, CUDA userspace 12.5 or later | all with a GPU graph, i.e. all except `nori-30m`, `exaone-tabular` |
| ROCm | Linux x64; gfx1201 verified, others behind an allowlist | all built-in models (masked graphs for the TabPFN family and others) |
| MLX | macOS arm64 | all with a GPU graph, except `nori-30m`, `exaone-tabular` |

There is no macOS x86_64 build, and Windows is CPU-only. No GPU, or still slow on the CPU: a smaller
context, `anofox_tabfm_threads`, and for repeated scoring the `tabicl-v2` context cache
(`anofox-tabfm-predict`, "Large calls and repeated calls").

## Errors and fixes

| You see | Do this |
|---|---|
| `no usable '<device>' device was discovered` | `SELECT * FROM tabfm_devices();` for the driver state, or `SET anofox_tabfm_device = 'auto';` |
| a predict on a named device fails with the reason the model cannot be served | `SELECT * FROM tabfm_backends() WHERE NOT supported;`, then `auto` or `cpu` |
| `tabfm_accelerate()` reports `verify failed` | install what `detail` names (e.g. MIGraphX), run `CALL tabfm_accelerate();` again; predictions continue on the CPU |
| a GPU precision mode is rejected | `bf16` / `fp16` are ROCm-only, `tf32` CUDA-only |
| the first ROCm predict takes minutes | the bucket compile: `CALL tabfm_gpu_precompile(...)` ahead of time |
| same answers, but slow | it is probably on the CPU: `SELECT model, device FROM tabfm_models() WHERE loaded;` |
