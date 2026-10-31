# anofox-tabfm documentation

Start with the [README](../README.md) and [Getting Started](guides/01-getting-started.md). The guides walk
through a task; the API reference is the source for every function, setting and error. The Background
documents explain how things came to be; where they disagree with the guides or the API pages, those win.

## Guides

| Guide | Description |
|---|---|
| [Getting Started](guides/01-getting-started.md) | Install, download a model, a first prediction on iris |
| [Choosing a model](guides/02-choosing-a-model.md) | The registry, selecting a model, your own model, how the models compare |
| [Evaluate and compare](guides/03-evaluate-and-compare.md) | A churn walk-through, cross-validation, plotting |
| [GPU acceleration](guides/04-gpu-acceleration.md) | `tabfm_accelerate()`, checking the device, the ROCm first-run compile |
| [Generate and impute](guides/05-generate-and-impute.md) | Synthetic rows and filling NULLs |

## API reference

| Topic | Description |
|---|---|
| [Functions](api/01-functions.md) | `tabfm_classify`, `tabfm_regress`, missing targets, segments, output columns |
| [Models and weights](api/02-models-and-weights.md) | The model table, download / load / unload / remove, the cache, secrets |
| [Generation](api/03-generation.md) | `tabfm_generate`, `tabfm_impute`, options, outputs |
| [GPU](api/04-gpu.md) | Plugins, support matrix, precision, platform limits |
| [Settings](api/05-settings.md) | Every setting and the `opts` keys |
| [Troubleshooting](api/06-troubleshooting.md) | Errors and what to run |
| [Evaluation](api/07-evaluation.md) | Metrics, cross-validation, scoring rules |
| [Plotting](api/08-plotting.md) | The six `_by` macros for anofox-visualization |

## Background

Labels: **Reference** is current and maintained; **Design** explains the architecture, but its status
lines may be out of date; **Report** records a measurement or spike at a date; **History** is a plan or
review that has been carried out, superseded or dropped.

### Reference

| Document | What it is |
|---|---|
| [BUILDING.md](BUILDING.md) | Building and testing from source, flavors, the GPU plugins |
| [REAL_MODELS.md](REAL_MODELS.md) | Each real model: how it was ported and verified, licences, gated Hugging Face repositories |
| [GENERATE.md](GENERATE.md) | How generation and imputation work, what binning costs, what this is not |
| [rocm-build.md](rocm-build.md) | Legacy toolchain: an ONNX Runtime with the MIGraphX EP for the `rocm` flavor (the ROCm plugin does not need it) |

### Design

| Document | What it is |
|---|---|
| [DYNAMIC_BACKENDS.md](DYNAMIC_BACKENDS.md) | One artifact, GPU backends as runtime plugins, the equivalence suite |
| [KV_CACHE_DESIGN.md](KV_CACHE_DESIGN.md) | The context cache behind `anofox_tabfm_context_cache` |
| [GPU_DISTRIBUTION.md](GPU_DISTRIBUTION.md) | Shipping GPU support |

### Reports

| Document | What it is |
|---|---|
| [GPU_AND_MEMORY_FINDINGS.md](GPU_AND_MEMORY_FINDINGS.md) | ROCm and weight-loading memory findings (2026-07-04) |
| [MLX_SPIKE_RESULTS.md](MLX_SPIKE_RESULTS.md) | The Apple MLX backend spike |
| [ROCM_TABDPT_SPIKE.md](ROCM_TABDPT_SPIKE.md) | Converting tabdpt to run on ROCm |
| [ROCM_SINGLE_EVAL_POS.md](ROCM_SINGLE_EVAL_POS.md) | Why some models cannot use ROCm's fixed-shape compilation |
| [RUST_CANDLE_FEASIBILITY.md](RUST_CANDLE_FEASIBILITY.md) | Replacing ONNX Runtime with candle: feasibility |
| [WINDOWS_INFERENCE.md](WINDOWS_INFERENCE.md) | Windows inference (resolved 2026-07-05) |

### History (plans and reviews)

| Document | What it is |
|---|---|
| [ANOFOX_CONTRACT_PLAN.md](ANOFOX_CONTRACT_PLAN.md) | The anofox integration contract behind the `_by` macros |
| [COREML_PLAN.md](COREML_PLAN.md), [COREML_LOCAL_PLAN.md](COREML_LOCAL_PLAN.md) | CoreML acceleration (dropped in favour of MLX) |
| [GENERATE_PLAN.md](GENERATE_PLAN.md) | Synthetic data generation and imputation |
| [GPU_HARDENING_PLAN.md](GPU_HARDENING_PLAN.md) | GPU hardening: root causes and spikes |
| [MLX_PLAN.md](MLX_PLAN.md) | The Apple MLX backend |
| [MULTI_MODEL_PLAN.md](MULTI_MODEL_PLAN.md) | The multi-model registry |
| [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md) | Performance plan (2026-07-04) |
| [PHASE_COMPLETION_PLAN.md](PHASE_COMPLETION_PLAN.md) | Completing the dynamic-backends phases |
| [PURE_SQL_SURFACE_PLAN.md](PURE_SQL_SURFACE_PLAN.md) | The pure-SQL model surface |
| [ROCM_TABPFN_PLAN.md](ROCM_TABPFN_PLAN.md) | TabPFN on ROCm |
| [CODE_REVIEW_2026-07-04.md](CODE_REVIEW_2026-07-04.md) | Consolidated code review (2026-07-04) |
| [UPSTREAM_ORT_ISSUES.md](UPSTREAM_ORT_ISSUES.md) | Upstream ONNX Runtime issue drafts |

[CHANGELOG.md](../CHANGELOG.md) lists every release.
