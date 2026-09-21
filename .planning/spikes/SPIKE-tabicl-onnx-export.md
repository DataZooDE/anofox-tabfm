# SPIKE: TabICL ONNX Export Feasibility

**Spike date:** 2026-09-21  
**Spike environment:** Isolated venv at `/tmp/.../scratchpad/spike_tabicl/`  
**Authored for:** Phase 2 planner (TabICL onboarding scope decision)

---

## FEASIBILITY VERDICT

**INFEASIBLE** — end-to-end ONNX export of TabICL is not achievable with the current v2.2.0 codebase without upstream changes. Two distinct data-dependent control-flow blockers exist in the `ColEmbedding` stage (Stage 1). Stages 2 (`RowInteraction`) and 3 (`ICLearning`) export individually but the partial exports produce incorrect results in inference mode, making a per-stage glue strategy unviable without deeper surgery.

---

## Package and Environment Versions

| Package | Version |
|---------|---------|
| tabicl | 2.2.0 |
| torch | 2.14.0+cu130 |
| onnx | 1.23.0 |
| onnxruntime | 1.30.0 |
| Python | 3.11.14 |

The project pins ORT 1.23.2. The test used ORT 1.30.0 for checking exported models.

---

## Architecture Summary

TabICL is a three-stage Transformer for tabular in-context learning. The `TabICL` nn.Module chains:

1. **Stage 1 — ColEmbedding** (`col_embedder`): Distribution-aware column-wise embedding via a Set Transformer (induced self-attention). Inputs: `X (B, T, H)`, `y_train (B, train_size)`. Feature groups of size 3 are formed via circular permutation. Each column/group is processed by a shared set transformer. The stage is target-aware: `y_train` labels are embedded and added to training-row projections before the set transformer runs. When `num_classes > max_classes`, mixed-radix ensembling runs the transformer multiple times with different digit-decomposed labels. Output: `(B, T, G+CLS, E)` where G = ceil(H/3) groups + 4 CLS tokens, E = 128.

2. **Stage 2 — RowInteraction** (`row_interactor`): Row-wise self-attention to capture cross-feature interactions within each row. The 4 CLS tokens aggregate feature information per row. RoPE positional encoding is applied. Output: `(B, T, 4*E)` = `(B, T, 512)` via CLS token concatenation.

3. **Stage 3 — ICLearning** (`icl_predictor`): Dataset-wise causal-style transformer (in-context learning). The `y_train` label embedding is added to training-row representations. The transformer attends to all T positions with a causal mask (test rows only attend to training context). Output for classification: `(B, test_size, num_classes)` logits or probabilities.

The forward signature is `forward(X, y_train)` with shapes:
- `X`: float32 `(B, T, H)` — rows 0..train_size-1 are training, rest test
- `y_train`: int64 `(B, train_size)` — class labels 0..C-1
- Output: float32 `(B, test_size, num_classes)` logits (inference) or raw `(B, test_size, max_classes)` (training mode)

Dynamic dimensions: T (number of rows), H (number of features), num_classes are all dynamic at inference time.

---

## What Was Tried

### Step 1: Forward pass verification
`TabICLClassifier(device="cpu").fit(X_train, y_train).predict_proba(X_test)` — **PASS**. The model downloads the `tabicl-classifier-v2-20260212.ckpt` checkpoint (~100 MB) from HuggingFace and runs inference correctly.

### Step 2: ONNX export attempts (6 variants)

All export attempts use `torch.onnx.export` (torch 2.14), which internally uses `torch.export.export` (dynamo-based exporter). The legacy TorchScript tracer is no longer the default in torch 2.14.

**2b: `_train_forward` wrapper, fixed shapes** — FAILED  
**2c: `_train_forward` wrapper, dynamic_axes** — FAILED  
**2d: `_inference_forward` wrapper, fixed shapes** — FAILED  
**2e/2f: `dynamo_export` API** — FAILED (API name changed; `dynamo_export` no longer exported from `torch.onnx` in 2.14)  
**2g: `col_embedder` stage alone** — FAILED  
**2h: `row_interactor` stage alone** — **SUCCESS** (fixed shapes, training mode)  
**2i: `icl_predictor` stage alone** — **SUCCESS** (fixed shapes, training mode)

### Step 3: Targeted diagnosis and workaround attempts

**3b: Isolate `SkippableLinear` as the blocker**  
Plain `nn.Linear` (no skip branch): export succeeded.  
`SkippableLinear` directly: FAILED with `GuardOnDataDependentSymNode`.

Root cause confirmed: `SkippableLinear.forward` at `layers.py:134`:
```python
skip_mask = (src == self.skip_value).all(dim=-1)
if skip_mask.any():          # ← data-dependent branch, untraceable by torch.export
    out[skip_mask] = self.skip_value
```

**3c: Patch `SkippableLinear` with `torch.where()`**  
Replaced the `if skip_mask.any()` branch with:
```python
skip_mask = (src == self.skip_value).all(dim=-1, keepdim=True)
out = torch.where(skip_mask, torch.tensor(self.skip_value, ...), out)
```
After this patch the first `SkippableLinear` blocker is resolved, but export still fails at `embedding.py:380`:

```python
num_classes = int(y_train.max().item()) + 1      # ← SECOND BLOCKER
needs_mixed_radix = self.max_classes > 0 and num_classes > self.max_classes
```

`y_train.max().item()` materializes a data-dependent integer to control execution flow (determines whether to run the standard path or mixed-radix ensembling). This is a structural blocker in the `ColEmbedding._compute_embeddings` method that cannot be removed with a simple `torch.where` patch — it gates an entire alternative code path (the mixed-radix ensemble loop).

**3d: Full patched model**  
FAILED at the same `embedding.py:380` error — the second blocker persists even after the SkippableLinear patch.

### Step 4: ORT validation of partial exports

The `row_interactor` and `icl_predictor` ONNX files pass `onnx.checker.check_model()` but produce incorrect numerical output:

| Stage | Max abs diff vs PyTorch |
|-------|------------------------|
| row_interactor | 0.5833 |
| icl_predictor | 2.217 |

These mismatches stem from the exports being captured in training mode (via `_train_forward` path), while the correct inference behavior requires eval-mode dropout, causal masking for test rows, and target-label conditioning — behaviors that differ between `_train_forward` and `_inference_forward`. The exports do not faithfully reproduce inference semantics.

---

## Verbatim Key Errors

### Blocker 1 — `SkippableLinear` data-dependent branch (layers.py:134)

```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
Could not guard on data-dependent expression Eq(u0, 1) (unhinted: Eq(u0, 1)).
(Size-like symbols: none)

Caused by: (_export/non_strict_utils.py:1377 in __torch_function__)

The following call raised this error:
  File "tabicl/_model/layers.py", line 134, in forward
    if skip_mask.any():
```

### Blocker 2 — `y_train.max().item()` data-dependent integer (embedding.py:380)

```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
Could not extract specialized integer from data-dependent expression u0
(unhinted: u0). (Size-like symbols: none)

Caused by: (tabicl/_model/embedding.py:380 in _compute_embeddings)

# The offending code:
num_classes = int(y_train.max().item()) + 1
needs_mixed_radix = self.max_classes > 0 and num_classes > self.max_classes
```

---

## Which Export Path Got Furthest

**Legacy TorchScript tracer** (the old `torch.onnx.export` pre-torch-2.x): would have been the most promising path, but torch 2.14 routes all `torch.onnx.export` calls through the dynamo exporter by default. The legacy `torch.jit.trace` would likely fail on the same control-flow issues.

**Dynamo-based exporter** (torch 2.14 `torch.onnx.export`): correctly identifies the untraceable data-dependent expressions but cannot proceed.

The `SkippableLinear` fix is one commit. The `y_train.max().item()` blocker in `_compute_embeddings` requires an architectural change: the number of classes needs to be either a model hyperparameter (known at construction time) or passed as a separate integer tensor input with a fixed specialization. This is not a minor patch.

---

## Stages That Export (Individually, Training Mode, Fixed Shapes)

| Stage | ONNX valid | ORT runs | Matches PyTorch inference |
|-------|-----------|---------|--------------------------|
| row_interactor | yes | yes | no (training mode, diff=0.58) |
| icl_predictor | yes | yes | no (training mode, diff=2.22) |
| col_embedder | no | — | — |

None of the individual exports produce correct inference-mode output. A per-stage glue strategy would require:
1. Fixing both blockers in `ColEmbedding` so Stage 1 can be exported.
2. Re-exporting all three stages in eval mode with correct inference semantics.
3. Writing C++ glue code to chain the three ONNX sessions together in the DuckDB extension, passing intermediate tensors between them.

---

## Root Cause Analysis

### Why `ColEmbedding` is fundamentally hard to export

Three interlocking design choices make `ColEmbedding` hostile to static-graph export:

1. **`SkippableLinear`** uses `if skip_mask.any()` — a data-dependent boolean guard on tensor contents. Fixable with `torch.where()`.

2. **`y_train.max().item()`** controls which execution branch runs (standard vs. mixed-radix ensembling). The branching decision depends on the input data. Fixing this requires either: (a) always running the mixed-radix path (wasteful, changes semantics), (b) specializing the export for `num_classes <= max_classes` only (partial solution, brittle), or (c) accepting `num_classes` as a separate fixed input to the graph.

3. **`InferenceManager`** wraps the actual compute in a Python object with memory-tracking state, async copy management, and dynamic batch-splitting loops. The inference forward path runs through `InferenceManager.__call__()` which contains Python loops, OOM-recovery logic, and runtime device queries — none of which are traceable. Only the `_train_forward` path (which bypasses `InferenceManager`) was traceable enough to attempt export.

### Why `RowInteraction` and `ICLearning` export but mismatch

These stages export because their `_train_forward` paths are simpler (no InferenceManager, no data-dependent branching). However, the exported graphs encode training-mode behavior: LayerNorm scale computation, no causal masking for test isolation, and no temperature scaling. The inference path (`_inference_forward`) applies additional conditioning and masking that is not in the exported graph.

---

## Scope Recommendation for Phase 2 Planner

**DEFER TabICL onboarding.** The ONNX export path is currently infeasible without upstream code changes to the `tabicl` Python package.

### What would be needed to unblock it

**Option A — Minimal upstream fix (2–3 PRs to soda-inria/tabicl):**
1. Replace `SkippableLinear`'s `if skip_mask.any()` with a `torch.where()` equivalent (1 line change, semantically equivalent).
2. Remove the `int(y_train.max().item())` branch in `_compute_embeddings`; instead accept `num_classes` as a constructor parameter and fix a single path at export time. This requires the caller to specialize the ONNX graph per `num_classes` value (one export per distinct class count).
3. Provide a dedicated `forward_for_export(X, y_train)` method that bypasses `InferenceManager` entirely and uses fixed shapes.

If all three fixes land upstream, a full end-to-end export with fixed shapes (one ONNX per `(H, T, num_classes)` triple) becomes feasible.

**Option B — Compiled export via `torch.compile` + custom tracing:**  
`torch.compile` with `fullgraph=True` would fail at the same control-flow boundaries. No shortcut here.

**Option C — Per-stage export with C++ glue:**  
Export Stage 2 and 3 only; implement Stage 1 natively in C++ (the set-transformer column embedding is ~500 lines of Python, not trivial). High implementation cost; not recommended without a clear upstream timeline.

**Option D — Alternative: provide a TabICL-compatible ONNX via torchdynamo after fixing upstream:**  
After Option A PRs merge, re-run this spike against the patched package to confirm. Expected timeline to confirm: 1 day of spiking after upstream merge.

### Uncertainty flags

- The `InferenceManager` bypassing (using `_train_forward` vs `_inference_forward`) introduces semantic differences. Even after fixing the two blockers, verifying that an exported `_inference_forward`-equivalent graph produces numerically identical output to the Python runtime requires careful testing.
- tabicl 2.2.0 uses `torch.onnx.export` defaults that route through the dynamo exporter. The legacy TorchScript tracer (via `torch.jit.trace`) was not tested exhaustively; it might succeed on fixed shapes for stages 2 and 3, but is deprecated and not usable for production.
- SSMax (`qassmax-mlp-elementwise`) uses sequence-length-dependent scaling (`log(n)` where `n = src_len.item()`). This would be a third blocker for dynamic-shape export, but is not encountered in the fixed-shape attempts since the sequence length is constant. For any dynamic-shape export, SSMax's `n: int` argument (currently a Python int derived from `.shape[1]`) would need to become an ONNX symbolic value.
- The ORT 1.23.2 version pinned in the extension predates the 1.30.0 used here. If an export were produced, compatibility with 1.23.2 would need to be verified (opset 18 support landed in ORT 1.14, so this is likely fine).
