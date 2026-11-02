# export_tabicl — TabICL v2 → weight-free ONNX (export spike)

Onboards **TabICL v2 (soda-inria, BSD-3-Clause)** into the anofox-tabfm ONNX
runtime contract. Mirrors `tools/export_onnx` (the TabFM shipping pipeline):
dynamo export (opset 18) → strip metadata → force-externalize checkpoint
initializers → tensor-name map → ORT-vs-PyTorch parity → delete `.onnx.data`.

```bash
uv sync
uv run export_tabicl --task classification --config real --out ../../resources
uv run export_tabicl --task regression     --config real --out ../../resources
uv run make_tabicl_fixture ../../test/fixtures/tabicl          # CI fixture
uv run pytest
```

## Optional: `--split-context`, the context-cache pair

`tabfm_classify` re-encodes the labelled context on every call, because the combined graph is one forward
pass over support and query rows together. On `tabicl-v2` that pass is 83-98% of a call's CPU time at a 375-
or 1000-row context, and the support set never changes between calls (DataZooDE/anofox-tabfm#37,
`docs/CONTEXT_CACHE_PLAN.md`).

```bash
# export into a scratch directory, then copy only the pair into resources/: the combined graph and its
# report there stay as shipped
uv run export_tabicl --task classification --config real --out /tmp/kv/classification --split-context
uv run export_tabicl --task regression     --config real --out /tmp/kv/regression     --split-context
# the ship gate: the pair against the SHIPPED combined graph, on the real (pinned) weights
uv run gate_context_cache --pair DIR_WITH_BOTH_PAIRS
```

emits two weight-free graphs per task and their tensor maps (`kv.py`):

| | inputs | outputs |
|---|---|---|
| `graph_prepare_tabicl_<task>.onnx` | `x[1,S,H]`, `y[1,S]` | `hidden_0..2` (column-ISAB state), `icl_k_0`, `icl_v_0`, ... `icl_v_11` (each ICL layer's support keys/values), regression only `y_mean`, `y_std`, and `context_logits[1,S,C]` |
| `graph_query_tabicl_<task>.onnx` | `x[1,Q,H]` and the state by name (no `y`) | `logits[1,Q,C]` |

- **Query rows only.** The query graph runs the query rows through TabICL's own `cached_kv` attention
  path against the cached keys/values; the support rows never pass through any layer again.
- **Every row is exact.** `context_logits` are the support rows' outputs as the combined graph computes
  them (each with its own label), so the engine's fitted values do not change with the cache.
- **Regression matches the shipped regression graph:** the raw targets are z-scored in prepare
  (population std, `0 -> 1`), the model sees `y_z` everywhere, and query returns the mean over the 999
  quantiles, inverse-scaled: `[1,Q,1]` in raw target units. The first-generation pair (`split.py`) fed raw
  `y` and returned raw quantiles, which is what this replaces.
- **Each half has its own tensor map,** mapped by NAME. `kv.finish` refuses a map built by value hash:
  zero-initialised biases are identical at export time and would map to the wrong real tensor (that bug hit
  the first draft of this exporter: a prefixed wrapper mapped ISAB biases to row-interactor biases, and the
  random-weight parity still passed).

**Parity.** At export: chained pair vs one PyTorch forward of the combined wrapper, every row, random
weights, shapes other than the export example (3.4e-07 classification, 1.8e-07 regression, budget 1e-3).
The ship gate (`gate_context_cache`, and `tests/test_real_weights.py` under `TABFM_REAL_WEIGHTS=1`) runs the
pair against the shipped combined graph on the real pinned weights: classification 0 label differences and
|Δlogit| ≤ 1.9e-05 on test and context rows over 11 tables (S 64-1000); regression |Δyhat| ≤ 2.8e-06 x
std(y); re-batching the query rows changes nothing beyond fp32 rounding; and a permuted-label control moves
the answers by 0.5-10 logits (≈2 std for regression), so the identity is not vacuous.

**The extension uses the pair by default** (`anofox_tabfm_context_cache`, default `true`): it looks for the
four files next to the model's combined graph (bundled ones for the built-ins) and serves the pair only when
the prepare graph emits `context_logits`. The pair from older exports (no `context_logits`) is ignored.

### Spike: `spike_context_cache`, the per-layer key/value contract (#37)

```bash
uv run spike_context_cache --graphs DIR --out DIR   # DIR: combined graph + map, and the --split-context pair
```

Exports two more pairs (weight-free, at real and fixture scale) and checks them against the shipped
combined graph on the real `tabicl-classifier-v2` weights, injected through the tensor maps:

- `prepare_fitted`: today's prepare, plus `context_logits`, the context rows' logits as the combined
  graph computes them;
- `prepare_kv` / `query_kv`: prepare also emits each ICL layer's support keys/values; query runs the
  query rows only and takes no `y`.

Results, and what they imply for the engine and the other models: `docs/CONTEXT_CACHE_PLAN.md`.
`test/fixtures/tabicl_kv` holds the fixture-scale pairs, now written by `kv.py` (both tasks).

### Why the split is exact

`InducedSelfAttentionBlock.induced_attention` already separates:

```python
hidden = multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])
out    = multihead_attn2(src, hidden, hidden)
```

`hidden` is derived from the support rows alone and every row's output is an independent attention
against it; block *k*'s `hidden` comes from block *k-1*'s support outputs, so the recursion stays
support-only. See `split.py` for the two traps (the `y_train.max()` path and the skip-mask's
shape-changing boolean index).

## Result: FULL SUCCESS (export), with a changed I/O contract

TabICL is the hard candidate (per-column set-transformer over a variable feature
axis). It exports **cleanly at DYNAMIC H** — no fixed feature bucket needed —
after five small, audited monkeypatches (`tabicl_patches.py`); we never edit or
copy the `tabicl` package. Parity ORT-vs-PyTorch (fp32, random weights, shapes ≠
export example): **classification 1.5e-7, regression 4.2e-7** (budget 1e-3).

### Exported signature (differs from the TabFM fixed contract)

| | |
|---|---|
| input `x` | `[1, T, H]` f32 — all rows; **H is dynamic** (no padding, no bucket) |
| input `y` | `[1, S]` f32 — **training labels only**; `S == train_size` |
| output `logits` | `[1, T, C]` f32 — rows `>= S` are test predictions |
| `train_size` | **implicit** = `len(y)` — no separate input, not baked |
| `cat_mask` | **omitted** — TabICL has no categorical path (all-numeric, z-scored internally) |
| `d` | **omitted** — unsupported when feature grouping is on (the v2 default) |
| `C` | classification: `max_classes` (10). regression: `num_quantiles` (**999 quantile logits**, not 1) |

**Engine integration:** feed input `y` = the training labels only (length
`train_size`), not the full `[1,T]` label column, and do **not** pad features to
a fixed H (feed the true width). These two deltas from the TabFM contract are the
price of TabICL's architecture; both use only names in the allowed set `{x, y}`.

### The five patches (all confirmed necessary)

1. `SkippableLinear.forward` — `if skip_mask.any()` → branchless `where`.
2. `InducedSelfAttentionBlock.forward` — skip-branch → compute-all + `where`.
3. `ColEmbedding._compute_embeddings` — drop `int(y_train.max().item())`
   (mixed-radix many-class path); force the standard target-aware branch
   (engine guarantees `num_classes <= max_classes`).
4. `ColEmbedding.feature_grouping` — `(idxs + 2**i) % H` (symbolic divisor →
   onnxscript `aten_remainder_scalar` crash) rewritten to
   `torch.remainder(idx, H_tensor)` (ONNX `Mod`); bit-exact incl. wrap-around.
5. `ssmax._logn` — `math.log(n)` bakes the ssmax temperature to the export
   train_size (no `Log` in the graph). Rewritten to derive `n` from a dynamic
   tensor (`ones(n).sum()`) so `ReduceSum → Log` stays dynamic — **required for
   real trained weights** (invisible with random init).

The training-mode forward is the exportable path (the inference path has KV
cache, `InferenceManager`, feature-shuffle Python loops, `torch.unique`); the
wrapper puts the model in `train()` and calls the three stages directly.

## License wall

No soda-inria checkpoint bytes anywhere. `resources/graph_tabicl_*.onnx` are
architecture-only (initializers externalized then `.onnx.data` deleted);
`test/fixtures/tabicl/` is seeded random init. BSD-3-Clause is ungated:
`commercial:true, redistributable:true, gate_setting:null`.

## Real-weight onboarding (not done here)

Parity used random weights. To inject the real `jingang/TabICL` v2 checkpoints,
first confirm their config matches the exported dims (the TabICL v2 `__init__`
defaults, see `configs._REAL_KWARGS`); a shape mismatch fails injection. The
tensor map keys are the bare model `state_dict` keys, so they match a checkpoint
loaded via `TabICL.load_state_dict` when the config agrees.

## The ROCm graph (masked contract)

`tabicl_mask.py` computes the same function with the split as a VALUE, so MIGraphX can compile it per (T, H)
shape bucket: `x[1,T,H]`, `y[1,T]` (full length, -100 off the context), `train_size`, `d`, `n_rows`. It runs
upstream's own modules and changes only what read a padded shape: the circular feature grouping is taken
modulo `d` (division-free, then a gather), the target-aware column embedding and the ICL label embedding go to
the context rows only, the column ISAB's inducing points and the ICL transformer attend to the context through
a key mask, QASSMax reads `log(S)` from the value, and the row interactor masks padded groups with its own `d`
mask. While it traces, the packed q/k/v weight is sliced rather than `split` (no SplitToSequence) and RoPE's
outer-product einsum is a broadcast. `mask_parity.gate` passes only when each of five negative controls is
caught; on the real checkpoints the masked forward matches the positional one to 2.6e-5 (both tasks).

```bash
uv run export_tabicl --task classification --config real --contract mask --out ./mask
uv run python ../make_migraphx_graph.py classification \
    --weights ~/.cache/anofox-tabfm/jingang__TabICL@4dcd344ece2c00be9e831fdd35bed57b5ad83e19/classification/model.safetensors \
    --graph ./mask/graph_mask_tabicl_classification.onnx \
    --tensor-map ./mask/tensor_map_mask_tabicl_classification.json \
    --dynamic --out ../../resources/graph_migraphx_tabicl_classification.onnx
uv run python ../check_migraphx_int_div.py ../../resources/graph_migraphx_tabicl_classification.onnx
TABFM_REAL_WEIGHTS=1 TABICL_MASK_DIR=./mask uv run pytest tests/test_real_weights.py   # vs the shipped graph
```

(and the same with `regression`). The masked graph reads the same 391 / 347 initializers as the positional
pair, so the tensor-map keys and the weights header sha are unchanged.
