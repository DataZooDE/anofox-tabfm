# GPU acceleration

## Flavors (CPU / GPU)

**A GPU is not a build.** The published artifact is the `cpu` flavor on every
platform, and each accelerator is a plugin it loads at runtime: `cuda`
(NVIDIA, via its own ONNX Runtime GPU distribution), `rocm` (AMD via a
**direct MIGraphX backend** — ONNX Runtime's MIGraphX EP can't load the >2 GB
model, so ROCm bypasses it and drives libMIGraphX directly, with a
compiled-program `.mxr` cache), and `mlx` (Apple Silicon, running the shipped
ONNX graph itself). `CALL tabfm_accelerate()` fetches the right one.

No vendor runtime is linked — CUDA/cuDNN or ROCm resolve from your system, and
`tabfm_devices()` reports what was found. GPU dtype is set by
`anofox_tabfm_gpu_precision` (default `fp32` — strict CPU parity; faster modes are opt-in).

Released cpu builds are served from the anofox extension repository
(`SET custom_extension_repository = 'https://get.anofox.com'`, which requires
`duckdb -unsigned`) as well as from the DuckDB community repository (signed, no
flag, but published by a separate submission that can trail a release). See
[Install & load](../guides/01-getting-started.md#1-install-and-load) for both paths and the `httpfs` ordering.

## Backend support matrix

**Backend support matrix** (what is verified, not what might work):

| backend | platforms | models | verified on |
|---|---|---|---|
| CPU | Linux x64/arm64, macOS **arm64**, Windows x64 | all 7 built-ins | CI suites + install-smoke with inference on every platform |
| CUDA (plugin) | Linux x64, CUDA userspace ≥ 12.5 | **all 7 built-ins** | RTX 4090/3070/A5000/A40: full example suite, catalog parity, 10k-row guardrail max |
| ROCm (plugin) | Linux x64, gfx1201 verified (allowlist gates others) | tabfm-v1, mitra, **tabdpt**, **tabpfn-v2, tabpfn-v2-5, tabpfn-v2-6, tabpfn-v3** (both tasks) — `SELECT * FROM tabfm_backends()` says which, and why not for the rest | RX 9070 XT: parity, concurrency, user workflow; tabdpt 5.6x CPU with identical predictions; TabPFN v2/v2.5/v2.6 3.5-7x CPU wall-clock and ~40x less CPU time, every row agreeing with the CPU to float noise; TabPFN-3 ~3-4x CPU wall-clock at small sizes, 0 label differences in 100 rows. **First call per (rows, features) bucket compiles for 2.5-9 min (about 16 min for `tabpfn-v3`)** — `CALL tabfm_gpu_precompile(...)` does it off the query path |
| CoreML | — | — | **dropped** — MLX supersedes it on Apple Silicon ([why](../DYNAMIC_BACKENDS.md#phase-4--coreml--dropped-2026-09-19)) |
| MLX (plugin) | macOS arm64 (Apple Silicon) | every model CPU serves (6 verified through SQL; tabfm-v1 via the graph harness) | Apple M3: 10 model×task pairs cpu-compared (0 disagreements), 4000-row stress, device/precision alternation |

## Platform limits

macOS is **arm64 only**: ONNX Runtime published its last macOS x86_64 /
universal2 archive at v1.23.2, so there is nothing to link an Intel-Mac build
against at the version this project needs. `osx_amd64` is therefore not built.

The CUDA and ROCm plugins are Linux-only; Windows artifacts are CPU-only by
design. On Apple Silicon the MLX plugin is the accelerator — unlike the other
two it runs the model's own ONNX graph directly (no ONNX Runtime), which is why
its model coverage matches CPU's. The single_eval_pos models (TabPFN/TabICL/Orion) cannot use ROCm's
bucketed compilation (y's length is the train/test split) — CUDA serves them.

## Plugins

`tabfm_download_runtime` fetches the plugin itself, and for CUDA the ONNX
Runtime GPU libraries it loads alongside it. The plugins are published as
release assets and are verified against the extension's plugin ABI version on
load, so a mismatched pair refuses rather than misbehaves.

## Precision

`anofox_tabfm_gpu_precision` now means something on both GPUs, with fp32 the
default everywhere: strict fp32 (on CUDA that disables the TF32 tensor-core
rounding ORT would otherwise apply to fp32 matmuls) so a device switch does
not change answers. `tf32` re-enables that rounding (CUDA only); `bf16`/`fp16`
quantize the MIGraphX program (ROCm only) for ~2x speed at the cost of a few
near-tie label flips. A mode a backend cannot honour is an error, never a
silent fp32 run.
