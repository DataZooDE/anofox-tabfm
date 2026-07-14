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
