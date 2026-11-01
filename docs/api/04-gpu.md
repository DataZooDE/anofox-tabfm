# GPU acceleration

The steps (set up, check the device, precompile on ROCm) are in the
[GPU acceleration guide](../guides/04-gpu-acceleration.md). This page is the reference.

- [Plugins, not builds](#plugins-not-builds)
- [Functions](#functions)
- [Choosing the device](#choosing-the-device)
- [Backend support matrix](#backend-support-matrix)
- [ROCm compile time](#rocm-compile-time)
- [Precision](#precision)
- [Platform limits](#platform-limits)

## Plugins, not builds

**A GPU is not a build.** The published artifact is the `cpu` flavor on every
platform, and each accelerator is a plugin it loads at runtime: `cuda`
(NVIDIA, via its own ONNX Runtime GPU distribution), `rocm` (AMD via a
**direct MIGraphX backend** — ONNX Runtime's MIGraphX EP can't load the >2 GB
model, so ROCm bypasses it and drives libMIGraphX directly, with a
compiled-program `.mxr` cache), and `mlx` (Apple Silicon, running the shipped
ONNX graph itself). `CALL tabfm_accelerate()` fetches the right one.

No vendor runtime is linked — CUDA/cuDNN or ROCm resolve from your system, and
`tabfm_devices()` reports what was found.

`tabfm_download_runtime` fetches the plugin itself, and for CUDA the ONNX
Runtime GPU libraries it loads alongside it. The plugins are published as
release assets and are verified against the extension's plugin ABI version on
load, so a mismatched pair refuses rather than misbehaves. A plugin also lists the graph inputs it binds,
and the extension refuses a graph that declares one it does not (plugin ABI 2: a plugin left over from
an older release is refused by `device := 'auto'`, which then uses the CPU, and `tabfm_backends()` names
the file and `CALL tabfm_accelerate()` as the fix). They land in the `runtime` subdirectory of
`anofox_tabfm_cache_dir`, unless `anofox_tabfm_ep_path` points elsewhere. Building a plugin from source:
[BUILDING.md](../BUILDING.md#gpu-plugins-from-source).

## Functions

| Call | Returns | Purpose |
|---|---|---|
| `CALL tabfm_accelerate()` | `step, status, detail` | find the accelerator, fetch its plugin (and runtime), verify it loads; re-running re-verifies without re-downloading |
| `CALL tabfm_download_runtime(backend)` | `file, bytes, status` | fetch one plugin: `'cuda'`, `'rocm'` or `'mlx'` |
| `SELECT * FROM tabfm_devices()` | `device_id, ep, name, arch, vram_total, vram_free, driver, usable` | the devices found: always `cpu`, plus any CUDA, ROCm or MLX device, `usable = false` when present but unsupported |
| `SELECT * FROM tabfm_backends()` | `model, task, device, backend, supported, reason` | which device can serve which model, and why not; `supported` is `NULL` while the weights are not downloaded |
| `CALL tabfm_gpu_precompile(task [, rows, features], model := '<id>')` | `task, rows, features, device, status` | compile a shape bucket ahead of the first predict (ROCm); a no-op cost on CPU and CUDA |
| `SELECT model, device FROM tabfm_models()` | | which device serves each loaded model right now |

## Choosing the device

`anofox_tabfm_device` (default `auto`) resolves per model: `auto` picks the best device that model can
actually be served on (its plugin present and a graph available), else the CPU, so it never promises an
accelerator a model cannot use. `SELECT * FROM tabfm_backends()` shows the choice and the reason. Naming a
device explicitly (`cuda`, `rocm`, `mlx`) is a hard request: it errors rather than falling back.

## Backend support matrix

What is verified, not what might work. `SELECT * FROM tabfm_backends()` is the source of truth for what a
backend serves on your machine.

| backend | platforms | models | verified on |
|---|---|---|---|
| CPU | Linux x64/arm64, macOS **arm64**, Windows x64 | every built-in model | CI suites + install-smoke with inference on every platform |
| CUDA (plugin) | Linux x64, CUDA userspace ≥ 12.5 | every built-in model | RTX 4090/3070/A5000/A40: full example suite, catalog parity, 10k-row guardrail max, on the catalog of the time (seven models); `limix-2m` has not been run on CUDA |
| ROCm (plugin) | Linux x64, gfx1201 verified (allowlist gates others) | `tabfm-v1`, `mitra`, `tabdpt` and the TabPFN family (`tabpfn-v2`, `tabpfn-v2-5`, `tabpfn-v2-5-real`, `tabpfn-v2-6`, `tabpfn-v3`), both tasks; also `tabicl-v2` and `limix-2m`, both tasks (masked graphs). Refused: `orion-bix`, `orion-msp`, `causilo` | RX 9070 XT: parity, concurrency, user workflow; tabdpt 5.6x CPU with identical predictions; TabPFN v2/v2.5/v2.6 3.5-7x CPU wall-clock and ~40x less CPU time, every row agreeing with the CPU to float noise; TabPFN-3 ~3-4x CPU wall-clock at small sizes, 0 label differences in 100 rows; `tabicl-v2` and `limix-2m` 0/100 label differences, warm ~6x CPU at 100 rows |
| MLX (plugin) | macOS arm64 (Apple Silicon) | every task of every built-in model that ships an ext graph (40 model×task pairs). Refused: `exaone-tabular`, `nori-30m` (no ext graph) | Apple M3, real weights, 100 rows: all 40 pairs cpu-compared through SQL (`test/sql/tabfm_mlx_real_models.test`): 0/100 label differences for every classifier, regression max \|diff\| ≤ 1.3e-4 (mitra-v2 1.0e-3), corr > 0.9999999; `limix-2m` included since its two interpreter fixes; 4000-row stress, device/precision alternation |
| CoreML | — | — | **dropped** — MLX supersedes it on Apple Silicon ([why](../DYNAMIC_BACKENDS.md#phase-4--coreml--dropped-2026-09-19)) |

The models refused on ROCm read the train/test split from the length of `y`, which MIGraphX's fixed-shape
compilation cannot express ([docs/ROCM_SINGLE_EVAL_POS.md](../ROCM_SINGLE_EVAL_POS.md)); CUDA and the CPU
serve them.

## ROCm compile time

MIGraphX compiles a program per (GPU arch, precision, shape bucket); the first predict in a bucket pays for
it, every later one in that bucket is sub-second. Measured on gfx1201:

| model | first compile per bucket |
|---|---|
| `tabfm-v1` (full size) | about 25 minutes |
| `tabpfn-v3` | about 16 minutes |
| the other ROCm models | 2 to 9 minutes (`orion-bix` 2.2, `orion-msp` 4.3 at 128 x 16) |

`CALL tabfm_gpu_precompile(...)` does it off the query path, and the compiled `.mxr` persists in the weight
cache across sessions. `SET anofox_tabfm_mxr_source = '<dir>'` stages matching precompiled programs from a
shared directory instead of compiling; they are arch- and ROCm-version-specific. CUDA has no analogous cost
(~30 s cold start).

## Precision

`anofox_tabfm_gpu_precision` (default `fp32` everywhere):

| value | backends | meaning |
|---|---|---|
| `fp32` | all | strict: a device switch does not change answers, measured exact on both GPUs. On CUDA it disables the TF32 tensor-core rounding ONNX Runtime would otherwise apply |
| `tf32` | CUDA | re-enables that rounding: fp32 storage, faster matmuls |
| `bf16`, `fp16` | ROCm | quantize the MIGraphX program: ~2x faster on RDNA4, half the VRAM and `.mxr` size |

Quantized label flips against fp32 are model-dependent and **not confined to near-ties**: measured,
`tabfm-v1` bf16 flipped ~2% of rows including high-confidence ones; `mitra` bf16 flipped 30% of query rows
but only at near-ties (margins ≤ 0.023), fp16 10% at margins ≤ 0.002. Validate on your own data. A mode a
backend cannot honour is an error, never a silent fp32 run.

## Platform limits

macOS is **arm64 only**: ONNX Runtime published its last macOS x86_64 /
universal2 archive at v1.23.2, so there is nothing to link an Intel-Mac build
against at the version this project needs. `osx_amd64` is therefore not built.

The CUDA and ROCm plugins are Linux-only; Windows artifacts are CPU-only by
design. On Apple Silicon the MLX plugin is the accelerator — unlike the other
two it runs the model's own ONNX graph directly (no ONNX Runtime), which is why
its model coverage matches CPU's.

Released cpu builds are served from the anofox extension repository
(`SET custom_extension_repository = 'https://get.anofox.com'`, which requires
`duckdb -unsigned`) as well as from the DuckDB community repository (signed, no
flag, but published by a separate submission that can trail a release). See
[Install and load](../guides/01-getting-started.md#1-install-and-load) for both paths and the `httpfs` ordering.
