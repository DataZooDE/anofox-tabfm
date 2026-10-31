# Settings

| Setting | Default | Purpose |
|---|---|---|
| `anofox_tabfm_accept_hf_license` | `false` | license gate — downloads fail without it |
| `anofox_tabfm_cache_dir` | `~/.cache/anofox-tabfm` | weight cache root (pre-seed for air-gapped use) |
| `anofox_tabfm_threads` | cores / 2 | ONNX Runtime intra-op threads |
| `anofox_tabfm_cpu_prepack` | `true` | prepack weights for faster CPU matmuls (~+16% RSS) |
| `anofox_tabfm_context_cache` | `false` | encode the labelled context once and reuse it, for models that ship a prepare/query graph pair — see [docs/KV_CACHE_DESIGN.md](../KV_CACHE_DESIGN.md) |
| `anofox_tabfm_max_rows` | `10000` | guardrail per predict / group |
| `anofox_tabfm_max_features` | `500` | guardrail |
| `anofox_tabfm_device` | `auto` | `auto` / `cpu` / `cuda` / `rocm` / `mlx` / `coreml` (`migraphx` alias for `rocm`). `cuda`, `rocm` and `mlx` are explicit opt-ins — `auto` never selects a device whose plugin is fetched separately |
| `anofox_tabfm_gpu_precision` | `fp32` | GPU numeric mode: `fp32` (strict — same answers as CPU, measured exact on both GPUs) / `tf32` (CUDA tensor-core rounding) / `bf16` / `fp16` (ROCm quantize; flip rates vs fp32 are model-dependent — measured: tabfm-v1 bf16 flipped ~2% including high-confidence rows, mitra bf16 flipped 30% of query rows but only at near-ties (margins ≤ 0.023) while fp16 flipped 10% at margins ≤ 0.002 — validate on your data) |
| `anofox_tabfm_max_sessions` | `4` | loaded (device, precision) sessions kept per model; oldest evicted beyond this |
| `anofox_tabfm_max_memory` | `''` | refuse predicts once resident memory is at/above this (e.g. `'16GB'`); `''` disables |
| `anofox_tabfm_default_model` | `''` | session-wide model when `model :=` is not given |
| `anofox_tabfm_mxr_source` | `''` | directory of precompiled ROCm `.mxr` programs to stage from |
| `anofox_tabfm_ep_path` | `''` | directory holding the GPU backend plugin (and the runtime libraries beside it) |
| `anofox_tabfm_trace_level` | `warn` | log verbosity: `error` / `warn` / `info` / `debug` / `trace` |

## Options

Options (the trailing `opts` MAP, all values VARCHAR): `task`, `n_estimators`
(v1: `1`), `seed`, `output_mode` (`compact` / `detail`), `context_rows`,
`softmax_temperature`, `model`. Unknown keys error.
