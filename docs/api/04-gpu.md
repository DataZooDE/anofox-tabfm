# GPU acceleration

The steps (set up, check the device, place models yourself) are in the
[GPU acceleration guide](../guides/04-gpu-acceleration.md). This page is the reference.

- [Plugins, not builds](#plugins-not-builds)
- [Functions](#functions)
- [Choosing the device](#choosing-the-device)
- [Backend support matrix](#backend-support-matrix)
- [Precision](#precision)
- [GPU memory](#gpu-memory)
- [Platform limits](#platform-limits)
- [Removed and deprecated](#removed-and-deprecated)

## Plugins, not builds

**A GPU is not a build.** There is one extension build per platform, and the CPU device runs on the
vendored [ggml](https://github.com/ggml-org/ggml) engine; no build links ONNX Runtime. Each accelerator
is a plugin the extension `dlopen`s at runtime:

| device | plugin file | what it is |
|---|---|---|
| `cuda` | `libanofox_tabfm_ggml-cuda_plugin.so` | the **ggml GPU plugin**: the same ggml engine as the CPU, on an NVIDIA GPU. The CUDA runtime and cuBLAS are linked into it statically, so it needs only the NVIDIA driver (`libcuda.so.1`), no CUDA toolkit |
| `rocm` | `libanofox_tabfm_ggml-rocm_plugin.so` | the ggml GPU plugin on an AMD GPU (HIP). It uses the system's ROCm (6.4 or newer) runtime libraries: HIP, hipBLAS and rocBLAS |
| `mlx` | `libanofox_tabfm_mlx_plugin.dylib` | Apple MLX on Apple Silicon, running the shipped ONNX graph itself; links Apple's `libmlx` / `libmlxc` (`brew install mlx mlx-c`) |

`CALL tabfm_accelerate()` fetches the right one; `CALL tabfm_download_runtime('cuda' | 'rocm' | 'mlx')`
fetches one by name. Each plugin is a release asset with a sha256 sidecar (`ggml-cuda_plugin.sha256`,
`ggml-rocm_plugin.sha256`, `mlx_plugin.sha256`); the download is checked against it and a plugin whose
checksum does not match is not used. A plugin is also checked against the extension's plugin ABI version
on load, so a mismatched pair refuses rather than misbehaves. A plugin also lists the graph inputs it binds,
and the extension refuses a graph that declares one it does not (plugin ABI 2: a plugin left over from
an older release is refused under `anofox_tabfm_device = 'auto'`, which then uses the CPU, and `tabfm_backends()` names
the file and `CALL tabfm_accelerate()` as the fix). They land in the `runtime` subdirectory of
`anofox_tabfm_cache_dir`, unless `anofox_tabfm_ep_path` points elsewhere. Building a plugin from source:
[BUILDING.md](../BUILDING.md#gpu-plugins-from-source).

The ggml GPU plugin reads the plain external-data graph of each model (the `ext` graph), so one graph
serves both vendors and every built-in model ships one. Its weights are read from the
downloaded safetensors and uploaded to the device once, when the session is created; BF16 checkpoints
(`tabfm-v1-1`) are upcast to float32 on the way. Ops the GPU backend has no kernel for run on the CPU
inside the same graph, so a model never needs a separate GPU graph.

## Functions

| Call | Returns | Purpose |
|---|---|---|
| `CALL tabfm_accelerate()` | `step, status, detail` | find the accelerator, fetch its plugin, verify it loads; re-running re-verifies without re-downloading |
| `CALL tabfm_download_runtime(backend)` | `file, bytes, status` | fetch one plugin: `'cuda'`, `'rocm'` or `'mlx'` |
| `SELECT * FROM tabfm_devices()` | `device_id, ep, name, arch, vram_total, vram_free, driver, usable` | the devices found: always `cpu`, plus any CUDA, ROCm or MLX device, `usable = false` when present but unsupported. `ep` names the engine that serves the device: `ggml` (cpu), `ggml-cuda`, `ggml-rocm`, `mlx` |
| `SELECT * FROM tabfm_backends()` | `model, task, device, backend, supported, reason` | which device can serve which model, and why not; `supported` is `NULL` while the weights are not downloaded |
| `SELECT model, device FROM tabfm_models()` | | which device serves each loaded model right now |

## Choosing the device

`anofox_tabfm_device` (default `auto`) resolves per model: `auto` picks the best device that model can
actually be served on (its plugin present and a graph available), else the CPU, so it never promises an
accelerator a model cannot use. `SELECT * FROM tabfm_backends()` shows the choice and the reason. Naming a
device explicitly (`cuda`, `rocm`, `mlx`) is a hard request: it errors rather than falling back. An explicit
`cuda` or `rocm` without the plugin in `anofox_tabfm_ep_path` names the missing file and the fix
(`CALL tabfm_download_runtime('cuda')`, `SET anofox_tabfm_ep_path`, or `SET anofox_tabfm_device='cpu'`).

Which cards count as usable in `tabfm_devices()`:

- **CUDA:** NVIDIA cards of compute capability 7.5 (Turing) or newer. The plugin carries kernels for
  `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100` and `sm_120` (plus PTX for newer cards); an older
  card is listed with `usable = false`.
- **ROCm:** a GPU node in the KFD topology, discrete card or APU iGPU, whose architecture the published plugin
  carries kernels for: `gfx1030`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`,
  `gfx90a` and `gfx942`. Any other card is listed with `usable = false`, so `'auto'` passes over it (an
  unsupported iGPU in front of a supported discrete card, say), and `SET anofox_tabfm_device = 'rocm'` with only
  such a card names it and the supported list. The list is compiled into the extension; a plugin you build for
  another arch ([BUILDING.md](../BUILDING.md#gpu-plugins-from-source)) also needs that arch added to
  `GgmlRocmPluginArchs` in `src/tabfm_devices.cpp`.
- **MLX:** Apple Silicon running an arm64 process (under Rosetta the row is `usable = false`).

## Backend support matrix

What is verified, not what might work. `SELECT * FROM tabfm_backends()` is the source of truth for what a
backend serves on your machine.

| backend | platforms | models | verified on |
|---|---|---|---|
| CPU (ggml) | Linux x64/arm64, macOS **arm64**, Windows x64 | every built-in model | CI suites + install-smoke with inference on every platform; every op checked against ONNX Runtime goldens, and all built-in models run on their real weights ([GGML_PLAN.md](../GGML_PLAN.md)) |
| CUDA (ggml GPU plugin) | Linux x64, NVIDIA driver for CUDA 12.8, compute capability ≥ 7.5 | every built-in model, both tasks (each ships an ext graph) | RTX 4090: per-op goldens 444/455 bit-for-bit (the rest float rounding), Nori 1.5 against ONNX Runtime 3.4e-3 on logits of scale 11, end to end through SQL with `tabfm_models()` reporting `cuda:0`. Full catalogue with the release build (`tools/gpu_test/gpu_catalog.sh`): all 44 model×task pairs served by `cuda:0`, 0 label differences against the CPU, regression within 4.9e-6 of the prediction range; the GPU use-case suite passes |
| ROCm (ggml GPU plugin) | Linux x64, ROCm ≥ 6.4 | every built-in model, both tasks (each ships an ext graph) | Radeon 890M iGPU (gfx1150): per-op goldens 451/455, Nori 1.5 end to end through SQL (`rocm:0`, 6.6e-5 against the CPU at scale 7.0), and the GPU use-case suite (`test/sql/tabfm_gpu_use_cases.test`: churn scoring, house prices, device switch, refusals). RX 9070 XT (gfx1201): goldens 451/455, Nori 1.5 against ONNX Runtime 2.1e-3 at 200 rows, warm 200 rows 0.26 s against 0.84 s on the CPU; full catalogue: all 44 model×task pairs served by `rocm:0`, 0 label differences, regression within 8e-6; the GPU use-case suite passes |
| MLX (plugin) | macOS arm64 (Apple Silicon) | every task of every built-in model (all ship an ext graph) | Apple M3, real weights: all 44 model×task pairs through SQL against the CPU (`tools/gpu_test/gpu_catalog.sh`): 0 label differences, regression within 3.4e-6 of the prediction range, identical quality; earlier 100-row run (`test/sql/tabfm_mlx_real_models.test`) 0/100 label differences; 4000-row stress, device/precision alternation |

The first call of a new shape builds its graph, so it is slower than later calls of that shape. To warm a
model, run one small prediction.

## Precision

`anofox_tabfm_gpu_precision` (default `fp32` everywhere):

| value | backends | meaning |
|---|---|---|
| `fp32` | all | strict: a device switch does not change answers beyond float rounding. On CUDA the plugin disables TF32 tensor-core rounding (`NVIDIA_TF32_OVERRIDE=0` before the CUDA runtime starts) and pads small matmuls so they avoid ggml's own small-batch TF32 kernel |
| `tf32` | CUDA (accepted on ROCm, which has no TF32 switch to flip; refused on MLX) | keeps the hardware's TF32 tensor-core matmuls: fp32 storage, faster matmuls, ~1e-3 relative error per matmul |
| `bf16`, `fp16` | MLX | MLX casts the program; validate on your own data. On `cuda` and `rocm` these are **refused** with an error: the ggml GPU plugin has no reduced-precision mode, and a silent fp32 run under another name would be a lie |

The TF32 switch is process-wide and latched by the first CUDA initialisation, so set
`anofox_tabfm_gpu_precision` before the first CUDA predict of the process; an `NVIDIA_TF32_OVERRIDE` you set
in the environment yourself wins. Label flips under a reduced precision against fp32 are model-dependent
and not confined to near-ties, so validate on your own data before using `bf16` / `fp16` on MLX. A mode a
backend cannot honour is an error, never a silent fp32 run.

## GPU memory

The ggml GPU plugin uploads a model's weights when the session is created and checks the device's free
memory first. A model that does not fit fails at load with the byte counts and the ways out: free the device
(`CALL tabfm_unload()` for a model you no longer use, or `SET anofox_tabfm_max_sessions = 1`), use a smaller
model, or `SET anofox_tabfm_device='cpu'`. Sessions are cached per (model, device, precision), so several
models on one card add up.

## Platform limits

macOS is **arm64 only**: `osx_amd64` is not built (it was dropped with ONNX Runtime, and the ggml engine has
not been verified there).

The CUDA and ROCm plugins are published for Linux x64 only; asking `tabfm_download_runtime` for one on
another platform is refused up front, naming what is available there. Windows artifacts are CPU-only (the
plugin builds as `anofox_tabfm_ggml-cuda_plugin.dll` from source, but is not published or verified). On
Apple Silicon the MLX plugin is the accelerator; ggml's Metal backend is not used, because its matmul is
half-precision internally and was 3 logit units off.

Released builds are served from the anofox extension repository
(`SET custom_extension_repository = 'https://get.anofox.com'`, which requires
`duckdb -unsigned`) as well as from the DuckDB community repository (signed, no
flag, but published by a separate submission that can trail a release). See
[Install and load](../guides/01-getting-started.md#1-install-and-load) for both paths and the `httpfs` ordering.

## Removed and deprecated

Removed: the ONNX Runtime CUDA plugin (`libanofoxort_gpu`), the MIGraphX ROCm plugin and its `.mxr` compile cache, and the
in-process `cuda` / `rocm` / `coreml` builds; CoreML was an ONNX Runtime provider, and MLX serves Apple Silicon. What is still
accepted, so existing scripts keep working:

| name | now |
|---|---|
| `SET anofox_tabfm_device = 'coreml'` | accepted; the first prediction errors, pointing to `CALL tabfm_download_runtime('mlx')` / `SET anofox_tabfm_device='cpu'`. `auto` never picks it |
| `SET anofox_tabfm_device = 'migraphx'` | an alias for `rocm` |
| `anofox_tabfm_mxr_source`, `anofox_tabfm_cpu_prepack` | no effect |
| `tabfm_register_model(..., classification_migraphx_graph / regression_migraphx_graph := ...)` | accepted and ignored; GPUs read the `*_ext_graph` |
| `CALL tabfm_gpu_precompile(task [, rows, features], model := '<id>')` | deprecated: loads the session and reports `status = 'not needed'` (`bucket_rows` / `bucket_features` are `NULL`); warm with one small prediction instead |

The design notes of the removed backends are in [archive/](../archive/README.md).
