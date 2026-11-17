# One memory estimator, one implementation per backend

Status: concept and plan (2026-11-14). Nothing here is implemented yet. It follows issue #107 and is ordered
after the ROCm bucket-ceilings PR, which lands first as it is and then becomes the first backend of this design.

## Why

A single `tabfm_classify` / `tabfm_regress` call can grow past the machine's memory, and when it does the
kernel kills the whole DuckDB process. #107: `mitra` on the CPU with 4,000 context rows x 15 features is
OOM-killed on 15 GB, although the default `anofox_tabfm_max_rows` (10,000) and the registry's recommendation
(10k rows) allow it. Three mechanisms touch this today, and none of them predicts a new shape:

| mechanism | where | what it knows | gap |
|---|---|---|---|
| `anofox_tabfm_max_rows`, registry `max_rows` | bind | a row count | rows only, not features or model; 10k is a GPU / large-RAM figure for mitra |
| `anofox_tabfm_max_memory` | `tabfm_engine.cpp` (`ForwardCost` / `RecordForwardCost`) | resident memory now, plus the measured cost of an earlier pass with exactly the same model, device, rows and features | keyed on the exact `T x H`, so every new row count is unguarded; off by default |
| ROCm bucket ceilings | `tabfm_model_spec.hpp` (`RocmBucketCeilingFor`, `RocmTensorTerm`) | per (model, task) the graph's largest tensors as functions of `(T, H)`, and the largest bucket measured to compile and run on a 16 GiB card | ROCm only; a measured limit, not a byte estimate |

The ceilings already hold the hard part: every graph's largest tensors as formulas in the rows `T` and
features `H`, fitted from `migraphx-driver read` and checked over the bucket grid. Those formulas describe
the graph, not ROCm. The CPU, CUDA and MLX backends run the same graphs and so allocate the same tensors.

## The design

Two layers: one describes the graph, the other the backend.

### 1. Graph memory model (backend-neutral)

```
struct TensorTerm { Dim d[4]; int64_t Elements(T, H); }         // today's RocmTensorTerm / RocmDim
struct GraphMemoryModel {
    vector<TensorTerm> terms;     // every tensor that is the graph's largest somewhere on the (T, H) grid
    idx_t weight_bytes;           // from the safetensors header
};
bool GraphMemoryModelFor(model, task, GraphMemoryModel &out);   // false: no model, estimate unavailable
```

- `T` is the rows of **one forward pass**: context + query rows of that pass (`query_batch` bounds the query
  share only; the context is re-run per batch except for `tabicl-v2`'s cached context), plus the context rows
  again when the fitted-value second pass runs (#105, `fitted_second_pass`).
- A registered model (`tabfm_register_model`) has no terms: the estimator falls back to today's measured
  cost, and the docs say so.
- Example, mitra: `[H+1, 4, T, T]` (row attention), `[T, 4, H+1, H+1]` (feature attention),
  `[T, H+1, 2048]` (MLP). At T = 2,500, H = 17 the first is 2.1 GB of the 3.19 GB peak measured on MLX
  (docs/MLX_SPIKE_RESULTS.md).

### 2. Backend estimators

One function per backend turns the graph terms at a shape into peak bytes, and says how much memory there is:

```
struct MemoryEstimate { idx_t peak_bytes; idx_t budget_bytes; string budget_source; int64_t max_t_that_fits; };
MemoryEstimate Estimate(const GraphMemoryModel &, const BackendMemoryProfile &, int64_t t, int64_t h);
```

| backend | shape the graph sees | peak | budget |
|---|---|---|---|
| CPU (ORT) | exact `T` | `base + weights + alpha_cpu x largest_term` | `anofox_tabfm_max_memory` if set, else available RAM, minus resident memory |
| MIGraphX | padded bucket (`PadToShapeBucket`) | all intermediates preallocated per compiled program | device memory; calibrated so the measured bucket is exactly at the limit, which reproduces today's ceilings |
| CUDA (ORT-GPU) | exact `T` | ORT arena: `weights + alpha_cuda x largest_term` | free device memory (NVML) |
| MLX | exact `T` | lazy evaluation; `alpha_mlx x largest_term` | unified memory: system RAM |

`alpha` is fitted per backend from measurements and is a property of the backend, not of the model: how
many largest-size buffers the runtime keeps alive at once. A per-model override exists for graphs that
break that pattern. The measurements already in the repo are the first calibration points:

| backend | model | shape (T = context + query per pass, H) | measured peak | largest term |
|---|---|---|---|---|
| CPU | mitra | 1,000 (500 + 500), 15 | 2.2 GB RSS | 0.26 GB |
| CPU | mitra | 1,100 (1,000 + 100), 15 | 2.5 GB RSS | 0.31 GB |
| CPU | mitra | 1,500 (1,000 + 500), 5 | 1.7 GB RSS | 0.22 GB |
| CPU | mitra | 1,500 (1,000 + 500), 15 | 3.5 GB RSS | 0.58 GB |
| CPU | mitra | 2,500 (2,000 + 500), 15 | 6.8 GB RSS | 1.60 GB |
| MLX | mitra | 2,500, 17 | 3.19 GB | 2.1 GB |
| MIGraphX | every model | the measured buckets | peak VRAM <= 11 GiB | the ceilings table, docs/ROCM_SINGLE_EVAL_POS.md |

The CPU rows (from #107, 4 threads) suggest `peak ~ 1.3 GB + 3.4 x largest_term` for mitra at 15 features (the 5-feature row sits
0.4 GB below it, so the smaller terms matter too). Phase 2 refits
this on a wider grid and more models before anything refuses on it.

### 3. One decision and one error

The engine asks once per forward pass, before running it:

- `peak_bytes > budget_bytes` refuses the call, with an error that names the shape, the estimate, the budget
  and where the budget came from. It also gives the largest context that fits (`max_t_that_fits`, by
  inverting the estimate), so the advice is a number: "sample the context to at most 1,800 rows
  (`USING SAMPLE 1800 ROWS`)". It suggests `query_batch` only when the query rows are a large share of `T`,
  and another device when one is present and has room.
- After a successful pass, the measured cost is recorded as today. It now corrects the estimate (the
  maximum of estimate and measurement for that shape) instead of being the only source.
- ROCm keeps refusing the same buckets as today, through the same path. `tabfm_rocm_ceilings.test` is
  the regression test.

## Plan

| phase | what | depends on | done when |
|---|---|---|---|
| 0 | user docs for #107: mitra's CPU scaling, what `max_memory` guards | nothing | in #108 |
| 1 | move `RocmTensorTerm` / `RocmDim` / the per-model terms into a backend-neutral header (`tabfm_memory_model.hpp`), with `GraphMemoryModelFor`; ROCm ceilings call it. No behaviour change | the ROCm ceilings PR merged | `tabfm_rocm_ceilings.test` and the model-spec C++ tests unchanged and green |
| 2 | CPU estimator: calibration grid (5 models x T 500-4,000 x H 5-100, peak RSS), fit `alpha_cpu`, refusal + error, `max_t_that_fits` | 1 | estimate within +-25% of measured peak on the grid; #107's 4,000-row call refused with a sampling hint instead of killed |
| 3 | CUDA (RunPod) and MLX (M3) estimators | 2 | same accuracy check per backend |
| 4 | the exact-shape `ForwardCost` becomes a correction of the estimate; the registry's `max_rows` is documented as quality guidance, not a memory limit | 2 | docs/api/05 and the skills describe one mechanism |

## Open questions (for the user before phase 2)

1. **Default on?** Refuse on the estimate even when `anofox_tabfm_max_memory` is unset, with available RAM as
   the budget? That turns today's OOM kills into errors by default, but it can refuse a call that would have
   fit (for example, a machine with swap).
2. **Safety margin:** refuse at estimate > 100 % or > 90 % of the budget?
3. **Several connections at once:** the budget is per process, so two concurrent large calls can each pass
   the check. Should a passed check reserve its estimate until the pass ends?
