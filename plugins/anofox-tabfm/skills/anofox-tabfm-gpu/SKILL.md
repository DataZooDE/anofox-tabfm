---
name: anofox-tabfm-gpu
description: >
  GPU acceleration and device control for the anofox_tabfm DuckDB extension:
  NVIDIA CUDA and AMD ROCm (both served by the ggml GPU plugin) and Apple MLX
  backend plugins loaded at runtime, device discovery, per-model backend
  support, precision modes and GPU memory. Use when making predictions faster
  on a GPU, running `tabfm_accelerate` / `tabfm_download_runtime`, reading
  `tabfm_devices()` / `tabfm_backends()`,
  setting `anofox_tabfm_device`, `anofox_tabfm_gpu_precision` or
  `anofox_tabfm_ep_path`, checking which device actually served a query, or
  troubleshooting a GPU that is not used.
version: 2026.11.23
user-invocable: false
---

# Anofox TabFM — GPU Cheat Sheet

**Extension:** anofox_tabfm v2026.11.23 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `tabfm_*` and `anofox_tabfm_*` (identical)

A GPU is not a separate build. There is one build per platform; the CPU runs on the vendored ggml engine and
no build contains ONNX Runtime. Each accelerator is a plugin loaded at runtime: `cuda` (NVIDIA, Linux x64) and
`rocm` (AMD, Linux x64) are both the **ggml GPU plugin** (`libanofox_tabfm_ggml-cuda_plugin.so`,
`libanofox_tabfm_ggml-rocm_plugin.so`), the same engine as the CPU on the GPU; `mlx` (Apple Silicon, macOS
arm64) is the MLX plugin. The CUDA plugin links the CUDA runtime statically and needs only the NVIDIA driver;
the ROCm plugin uses the system's ROCm 6.4+ (HIP, hipBLAS, rocBLAS); MLX needs `brew install mlx mlx-c`.

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
   when the model cannot be served there instead of falling back; without the plugin the error names the file
   and `CALL tabfm_download_runtime('cuda')`.
4. **Precision defaults to strict `fp32`** everywhere. `bf16` / `fp16` exist on MLX only; on CUDA and ROCm
   they are refused with an error. `tf32` (CUDA) is faster but rounds each matmul; validate before switching.
5. **A model must fit the card.** Weights are uploaded when the session is created, after a free-memory
   check; a model that does not fit fails at load naming `CALL tabfm_unload()`, `SET anofox_tabfm_max_sessions = 1`
   and `SET anofox_tabfm_device='cpu'`. Sessions are cached per (model, device, precision) and add up.
6. **Which cards are usable.** CUDA needs compute capability 7.5 (Turing) or newer. On ROCm a card (iGPUs
   included) is usable when the published plugin has kernels for its arch: gfx1030/1100/1101/1102/1150/1151/1200/1201,
   gfx90a and gfx942; any other is `usable = false`, and `'auto'` passes over it. Every built-in model ships
   an ext graph, so each runs on CUDA, ROCm and MLX as far as `tabfm_backends()` says.

## Set up

```sql
CALL tabfm_accelerate();   -- find the GPU, fetch its plugin, verify it loads
```

Returns `step, status, detail` rows, including what is still ahead (the ROCm runtime libraries, `brew install
mlx mlx-c` on macOS). Re-running re-verifies without re-downloading. That is all: with `device = 'auto'` the
next predict uses the GPU for every model it can serve.

By hand:

```sql
CALL tabfm_download_runtime('cuda');   -- 'cuda', 'rocm' or 'mlx': rows file, bytes, status
SET anofox_tabfm_device = 'cuda';
```

Plugins land in the `runtime` subdirectory of `anofox_tabfm_cache_dir`, verified against a published sha256;
`SET anofox_tabfm_ep_path = '<dir>'` only to point somewhere else. A plugin is checked against the extension's
plugin ABI on load, so a plugin left over from an older release is refused (and `auto` uses the CPU);
`CALL tabfm_accelerate()` fetches the matching one.

## Inspect

```sql
SELECT * FROM tabfm_devices();
SELECT model, task, device, backend, supported, reason FROM tabfm_backends() WHERE NOT supported;
```

| function | columns | meaning |
|---|---|---|
| `tabfm_devices()` | `device_id, ep, name, arch, vram_total, vram_free, driver, usable` | always `cpu`, plus every CUDA, ROCm or MLX device found; `usable = false` when present but unsupported. `ep` is the serving engine: `ggml` (cpu), `ggml-cuda`, `ggml-rocm`, `mlx` |
| `tabfm_backends()` | `model, task, device, backend, supported, reason` | per model, task and device: can it be served, and why not. `supported` is `NULL` while the weights are not downloaded |
| `tabfm_models()` | `..., device, ...` | the device serving each loaded session right now |

Verify a query really ran on the GPU:

```sql
SELECT customer_id, yhat
FROM tabfm_classify('history', 'churned', test := 'prospects', keep := ['customer_id'], model := 'mitra');
SELECT model, task, device FROM tabfm_models() WHERE loaded;
```

## Warm-up

The first call of a new shape builds its graph and is slower than later ones; to warm a model, run one small
prediction.

## Precision

| `anofox_tabfm_gpu_precision` | backends | meaning |
|---|---|---|
| `fp32` (default) | all | strict: switching device does not change answers beyond float rounding (TF32 off on CUDA, small matmuls padded past ggml's TF32 kernel) |
| `tf32` | CUDA (accepted on ROCm, which has no TF32 switch; refused on MLX) | TF32 tensor-core matmuls, fp32 storage. Process-wide, latched at the first CUDA use |
| `bf16`, `fp16` | MLX | reduced precision; refused on CUDA and ROCm |

A mode the backend cannot honour is an error, never a silent fp32 run.

Deprecated names, accepted so old scripts keep working: `'coreml'` (errors, points to `mlx`), `'migraphx'` (an
alias for `rocm`), `anofox_tabfm_mxr_source` and `anofox_tabfm_cpu_prepack` (no effect), `tabfm_gpu_precompile`
(reports `not needed`); do not use them in new SQL.
