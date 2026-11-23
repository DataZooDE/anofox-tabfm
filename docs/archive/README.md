# Archive: the ONNX Runtime and MIGraphX era

These documents describe GPU and platform backends that have been **removed** from the extension: the
in-process ONNX Runtime builds (the `cuda` / `rocm` / `coreml` flavors), the ONNX Runtime CUDA plugin
(`libanofoxort_gpu`), the direct MIGraphX plugin for ROCm (`.mxr` compile cache, shape buckets, masked
graphs, bucket ceilings) and the CoreML execution provider. They are kept for history: the design
reasoning, the measurements and the upstream bugs found along the way.

None of it describes current behaviour. Today the CPU device runs on the vendored ggml engine, CUDA and
ROCm are served by the ggml GPU plugin, and Apple Silicon by the MLX plugin; see
[GPU](../api/04-gpu.md), [BUILDING.md](../BUILDING.md) and [GGML_PLAN.md](../GGML_PLAN.md). Links and file
paths inside these documents (source files, scripts, settings such as `anofox_tabfm_mxr_source`) may name
things that no longer exist.

| Document | What it was |
|---|---|
| [DYNAMIC_BACKENDS.md](DYNAMIC_BACKENDS.md) | One artifact, GPU backends as runtime plugins, the equivalence suite |
| [GPU_DISTRIBUTION.md](GPU_DISTRIBUTION.md) | Shipping the ONNX Runtime / MIGraphX GPU support |
| [GPU_HARDENING_PLAN.md](GPU_HARDENING_PLAN.md) | GPU hardening: root causes and spikes |
| [UPSTREAM_ORT_ISSUES.md](UPSTREAM_ORT_ISSUES.md) | ONNX Runtime bugs found and reported upstream |
| [rocm-build.md](rocm-build.md) | Building an ONNX Runtime with the MIGraphX EP for the old `rocm` flavor |
| [ROCM_SINGLE_EVAL_POS.md](ROCM_SINGLE_EVAL_POS.md) | Why some models could not use MIGraphX's fixed-shape compilation; the masked graphs |
| [ROCM_TABDPT_SPIKE.md](ROCM_TABDPT_SPIKE.md) | Converting tabdpt to run on MIGraphX |
| [ROCM_TABPFN_PLAN.md](ROCM_TABPFN_PLAN.md) | TabPFN on MIGraphX |
| [ROCM_CAUSILO_PLAN.md](ROCM_CAUSILO_PLAN.md) | Causilo on MIGraphX |
| [COREML_PLAN.md](COREML_PLAN.md), [COREML_LOCAL_PLAN.md](COREML_LOCAL_PLAN.md) | CoreML acceleration (dropped in favour of MLX) |
| [WINDOWS_INFERENCE.md](WINDOWS_INFERENCE.md) | Inference on Windows with ONNX Runtime |
| [KV_CACHE_DESIGN.md](KV_CACHE_DESIGN.md) | The first context-cache design (ORT and MIGraphX backends); the current plan is [CONTEXT_CACHE_PLAN.md](../CONTEXT_CACHE_PLAN.md) |
| [GPU_AND_MEMORY_FINDINGS.md](GPU_AND_MEMORY_FINDINGS.md) | ROCm (ORT MIGraphX EP) and weight-loading memory findings (2026-07-04) |
| [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md) | Performance plan (2026-07-04): bf16 on MIGraphX, `.mxr` precompile |
| [PHASE_COMPLETION_PLAN.md](PHASE_COMPLETION_PLAN.md) | Completing the dynamic-backends phases |
| [CODE_REVIEW_2026-07-04.md](CODE_REVIEW_2026-07-04.md) | Code review of the ONNX Runtime engine (2026-07-04) |
| [RUST_CANDLE_FEASIBILITY.md](RUST_CANDLE_FEASIBILITY.md) | Replacing ONNX Runtime with candle: feasibility (superseded by ggml) |
| [MEMORY_ESTIMATOR_PLAN.md](MEMORY_ESTIMATOR_PLAN.md) | A memory estimator for the ORT and MIGraphX backends |
