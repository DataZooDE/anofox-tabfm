# SPIKE: TabPFN v2 Regression Tensor Contract

**Date:** 2026-09-21
**Status:** COMPLETE
**Workstream:** WS-E (predict surface) / WS-A (tools/export_onnx)
**Author:** investigation via Claude Code agent

---

## Feasibility Verdict

**PARTIAL — inference contract fully confirmed, ONNX export blocked by data-dependent preprocessing**

The TabPFN v2 regression output contract (logits shape, bin borders, decode math) is fully characterized and empirically verified. A correct C++ decoder can be written. However, ONNX export of the TabPFNV2 model is blocked by data-dependent branches in the preprocessing pipeline that `torch.export` cannot trace. The existing `tools/export_onnx` directory targets a **completely different model** (Google TabFM from `vendor/tabfm`), not Prior Labs TabPFN v2.

---

## Package Versions Used

| Package | Version |
|---------|---------|
| tabpfn | 9.0.0 (latest PyPI; covers v2, v2.5, v2.6, v3, v3.5 architectures) |
| torch | 2.14.0 |
| onnx | 1.23.0 |
| onnxscript | 0.7.2 |
| Python | 3.11 |

Investigation environment: `/tmp/.../scratchpad/spike_tabpfn/.venv/` (isolated, not in repo).
Model checkpoint downloaded from public `Prior-Labs/TabPFN-v2-reg` HuggingFace repo — scratchpad only, never committed.

---

## Critical Architectural Discovery: Two Model Families, Two Contracts

This repo currently ships tooling and an inference engine for **Google TabFM** (`vendor/tabfm`), not Prior Labs TabPFN v2.

| Property | Google TabFM (current) | Prior Labs TabPFN v2 (roadmap) |
|----------|----------------------|-------------------------------|
| Source | `vendor/tabfm/` (Apache-2.0 submodule) | `Prior-Labs/TabPFN-v2-reg` on HuggingFace |
| ONNX export tool | `tools/export_onnx/` (works, S01–S06 done) | No tool yet; export BLOCKED |
| Regression output tensor shape | `[1, T, 1]` — scalar per row | `[n_test, 1, K]` with K=5000 |
| Output semantics | Point estimate (scalar) | Bar distribution logits |
| Bin borders | Not applicable | Required; stored in checkpoint |

The Phase 2 implementation must decide which model family it targets for the regression distribution surface. **The two families have incompatible output contracts.**

---

## Exact Tensor Contract for TabPFN v2 Regression

### Output Tensors

#### 1. Logits tensor (primary inference output)

| Property | Value |
|----------|-------|
| Tensor name | `output` (when `only_return_standard_out=True`; key `"main"` in the full dict) |
| Shape (single estimator raw) | `[n_test, B=1, K=5000]` — `n_test` test rows, batch dim B=1 |
| Shape (aggregated, post-ensemble) | `[n_test, K=5000]` — squeeze of batch dim after averaging |
| DType | `float32` |
| Semantics | Unnormalized log-probabilities (logits) in z-normalized target space |
| Space | **z-normalized** — `(y - y_train_mean) / y_train_std`; NOT raw target units |

The `predict(output_type="full")` API returns `result["logits"]` as `[n_test, 5000]` after ensemble aggregation.

#### 2. Bin borders tensor (required for decode)

| Property | Value |
|----------|-------|
| Shape | `[K+1] = [5001]` — K+1 borders for K=5000 bins |
| DType | `float32` |
| Origin | Checkpoint state dict (`criterion.borders` for v2/v2.5; `model.regression_borders` buffer for v3+) |
| Space | **z-normalized** — same scale as logits; must be affine-transformed for raw-space metrics |
| Uniformity | **Non-uniform** — highly non-uniform by design (half-normal tails) |
| Min bucket width | ~0.0024 (central bins) |
| Max bucket width | ~66.95 (outermost bins) |
| Width ratio | ~28,000:1 |
| Fixed vs data-dependent | **Fixed at training time** — same borders for every dataset; only the affine transform is dataset-dependent |

Empirical border range (v2 regressor): `borders[0] = -93.087`, `borders[-1] = 86.943`.

#### 3. K (number of bins)

`K = 5000`. Source: `num_buckets: int` field in checkpoint config (ArchitectureConfig). This is hardcoded in the v2 checkpoint; v3/v3.5 may differ.

---

## Border Origin and Uniformity (Roadmap Open Question — RESOLVED)

**Borders are NOT uniform. They are checkpoint-fixed, stored in the checkpoint state dict.**

The roadmap warning was correct: "Assuming uniform bins yields plausible-but-wrong quantiles/CRPS" — the outer bins span 70+ normalized units while central bins are ~0.002 wide. Uniform-bin assumptions undercount extreme-tail probability mass by orders of magnitude.

### Where borders come from per architecture version:

- **v2 / v2.5:** `criterion.borders` in the checkpoint's `state_dict`; loaded via `_resolve_regression_borders()` in `model_loading.py`
- **v3 / v3.5:** `model.regression_borders` registered buffer; initialized by `_spline_based_regression_borders(num_buckets)` which interpolates a fixed spline from v2.5 reference points — still checkpoint-fixed, not data-dependent

### z-normalize vs raw-space

The model operates in z-normalized target space. The `TabPFNRegressor` computes:

```python
raw_borders = znorm_borders * y_train_std_ + y_train_mean_
```

This affine transform is applied in Python **outside** the model's `forward()`. A C++ decoder must apply this transform using the per-dataset scale/shift parameters: `(y_train_std, y_train_mean)`, which must be passed as sidecar data alongside the logits.

---

## Decode Math

### Class: FullSupportBarDistribution

TabPFN v2 uses `FullSupportBarDistribution`, a subclass of `BarDistribution` that extends the outermost bins with half-normal tails to handle extreme values. `num_bars = K = 5000`.

### Data layout

```
borders:         [b_0, b_1, ..., b_K]        shape [K+1]
bucket_widths:   [b_1-b_0, b_2-b_1, ...]     shape [K]
bucket_midpoints: [(b_0+b_1)/2, ...]          shape [K]
probs:           softmax(logits)              shape [n_test, K]
```

### Mean (point estimate)

```
mean = probs @ bucket_midpoints
```

Exception for outermost bins (FullSupportBarDistribution override):

- Left tail (`i=0`): mean = `b_0 - sqrt(pi/2) * sigma_left` where `sigma_left = bucket_widths[0]`
- Right tail (`i=K-1`): mean = `b_K + sqrt(pi/2) * sigma_right` where `sigma_right = bucket_widths[K-1]`

For most practical inputs the extreme-bin probability is negligible and the simple midpoint formula is numerically indistinguishable. The full half-normal correction must be used for conformance.

### Quantile (icdf)

```python
def icdf(logits, q):
    probs = softmax(logits)          # [n_test, K]
    cumprobs = cumsum(probs, axis=1) # [n_test, K]
    # For each row, find bin i where cumprobs[i-1] < q <= cumprobs[i]
    # Linear interpolate within that bin:
    #   x = b_i + (q - cumprobs[i-1]) / probs[i] * bucket_widths[i]
```

FullSupportBarDistribution additionally handles `q < probs[0]` (left half-normal tail) and `q > 1 - probs[K-1]` (right half-normal tail) with inverse-half-normal formulas. For standard quantiles (q in [0.1, 0.9]) this rarely triggers.

### CRPS

CRPS requires explicit bin borders — **cannot be computed with assumed uniform bins**.

The continuous ranked probability score integrates `(CDF(t) - 1{t >= y})^2 dt`. For a bar distribution this decomposes over bins:

```
CRPS = sum over bins i of:
    integral over [b_{i-1}, b_i] of (F_i(t) - 1{t >= y})^2 dt
```

where `F_i(t)` is piecewise linear between `cumprobs[i-1]` and `cumprobs[i]`.

The integral within each bin evaluates analytically using bin widths. The bin-width terms enter the formula as `w_i = borders[i+1] - borders[i]` — the non-uniform values directly affect the per-bin loss. Assuming `w_i = 1/K` (uniform) would give wrong values for all but the central bins.

**Implementation requirement:** borders must be stored alongside logits as a sidecar tensor for all CRPS computations.

### Raw-space decode workflow

```
1. Run model forward: logits [n_test, K], borders [K+1] in z-space
2. Compute softmax(logits) → probs [n_test, K]
3. Affine transform borders to raw space:
       raw_borders = znorm_borders * y_std + y_mean
4. Compute point estimate, quantiles, or CRPS on raw_borders
```

Steps 1–2 are inside ONNX (if/when export is unblocked). Steps 3–4 are in C++.

---

## ONNX Export Status

### Full model export: BLOCKED

Attempted `torch.export.export(model, ...)` on the full `TabPFNV2` architecture. Both `strict=True` and `strict=False` modes fail with `TorchExportError`:

```
Could not guard on data-dependent expression Eq(u0, 1)
```

Root cause locations:

1. `_remove_constant_features` → `select_features`: contains `if torch.all(sel)` — data-dependent control flow that dynamo cannot trace with symbolic shapes
2. `_impute_nan_and_inf_with_mean`: uses `torch_nanmean` — data-dependent reduction
3. `add_column_embeddings`: uses data-dependent indexing

These functions run inside `forward()` and cannot be stripped without changing model semantics.

### Backbone-only wrapper export: BLOCKED

Attempted wrapping only the transformer backbone (skipping preprocessing), passing pre-encoded tensors directly. Still fails:

```
torch.onnx.dynamo_export: symbolic integers u0, u1, u2 unresolved
```

Root cause: `chunked_evaluate_maybe_inplace` inside attention blocks uses conditional chunking based on runtime tensor sizes. The symbolic path creates data-dependent `Eq(u0, 1)` guards that `torch.export` cannot resolve.

### What this means for the tools/export_onnx directory

The existing `tools/export_onnx/` tool exports **Google TabFM** (from `vendor/tabfm`) — a completely separate architecture. It is not affected by the TabPFN v2 export blockers and continues to work for the TabFM model.

If TabPFN v2 support is added to the ONNX export pipeline, a new export tool will be needed — either:
- A custom torch.jit.trace (bypasses dynamo, may produce incorrect graphs for data-dependent ops)
- Using `tabpfn`'s own future ONNX export path (not available as of v9.0.0)
- A C++ inference-only path that calls the tabpfn Python package at runtime (non-ONNX approach)

---

## Concrete Next-Step Guidance

### For Phase 2 planner

1. **Decide model family scope.** The eval/metrics surface (CRPS, quantile score) is model-agnostic, but the regression distribution output contracts differ between TabFM (scalar) and TabPFN v2 (bar distribution). Phase 2 should specify which model family the regression distribution metrics target.

2. **Bar distribution borders as first-class output.** Any C++ decode of TabPFN v2 regression requires borders as a sidecar tensor. The `TabFMState` model cache must store `(logits [n_test, K], borders [K+1], y_mean, y_std)` rather than just logits. If the ONNX graph cannot embed borders, they must be stored separately (e.g., as a companion `.borders` file alongside the ONNX checkpoint).

3. **K is checkpoint-specific.** Do not hardcode `K=5000` in C++ without a version guard. The `ModelManifest` struct should carry `num_buckets` parsed from the model's config.json.

4. **Affine transform is outside the model.** `raw_borders = znorm_borders * y_std + y_mean` requires `y_std` and `y_mean` from the training data. These are runtime, dataset-dependent values — they cannot be baked into the ONNX graph. The C++ preprocessing layer must compute and pass them explicitly.

5. **FullSupportBarDistribution tail correction.** The outer-bin half-normal mean correction is a small but spec-required detail. Implement it; skip it only if the outer-bin probability is negligible (add a trace-level warning if outer bin mass > 1e-3).

### For tools/export_onnx

6. **TabPFN v2 is not exportable via torch.export today.** Do not attempt to extend `tools/export_onnx/` for TabPFN v2 with current torch/onnx tooling. Track upstream `tabpfn` releases for a future `export_onnx()` API or wait for torch.export support for data-dependent preprocessing.

7. **Alternative: tabpfn runtime integration (non-ONNX).** A `tools/tabpfn_predict/` uv project that runs TabPFN v2 natively (Python, CPU) and serializes `(logits, znorm_borders)` to a file or pipe for the C++ engine to consume is technically feasible today and avoids the export blocker entirely. This approach sacrifices cross-platform single-binary packaging but enables regression distribution metrics before ONNX export is unblocked.

8. **Border embedding workaround.** When ONNX export is eventually available, embed borders as a constant tensor in the ONNX graph's initializers (they are fixed per checkpoint). This eliminates the need for a separate sidecar file and makes the ONNX graph self-contained for inference.

---

## Summary Table

| Question | Answer |
|----------|--------|
| K (number of bins) | 5000 |
| Logits shape | `[n_test, K]` post-ensemble, `[n_test, 1, K]` raw |
| Logits dtype | float32 |
| Borders shape | `[K+1] = [5001]` |
| Borders dtype | float32 |
| Borders uniform? | **No** — highly non-uniform (ratio ~28,000:1) |
| Borders origin | Checkpoint state dict (`criterion.borders`) |
| Borders data-dependent? | **No** — fixed at training time |
| Borders space | z-normalized; affine transform needed for raw space |
| ONNX export feasible? | **No** — blocked by data-dependent preprocessing |
| Existing tools/export_onnx targets TabPFN v2? | **No** — targets Google TabFM (different model) |
| Decode math: mean | `softmax(logits) @ bucket_midpoints` + half-normal correction for outer bins |
| Decode math: quantile | cumsum search + linear interpolation within bin |
| Decode math: CRPS | Sum over bins of analytical integral using actual `bucket_widths[i]`; **requires non-uniform borders** |
