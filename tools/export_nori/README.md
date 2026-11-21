# export_nori — Nori-30M (Synthefy) → weight-free ONNX

WS-A exporter for the built-in **`nori-30m`** model (`Synthefy/Nori-30M`, Apache-2.0 code and weights).
Upstream is the PyPI package `synthefy-nori`, pinned exactly in `pyproject.toml` and patched at run time
(`nori_patches.py`): no upstream source is copied into this repo, and no weight bytes are committed.

```bash
cd tools/export_nori && uv sync
uv run export_nori --config real --out ./out          # graph + tensor map, weight-free (regression only)
uv run make_nori_fixture ../../test/fixtures/nori      # the committed random-init CI fixture
uv run pytest                                          # TABFM_REAL_WEIGHTS=1 adds the real-weight tests
```

The shipped copies live in `resources/` as `graph_nori30m_regression.onnx` and
`tensor_map_nori30m_regression.json` (renamed on copy). There is no converter: the engine reads `nori.pt`
(`{model_config, ema_state_dict}`) natively. Status, measurements and the licence are in
`docs/REAL_MODELS.md`; this file is about the exporter.

## What the exporter does

Nori is a LimiX fork, so the wrapper is LimiX-2M's: `(x, y)`-only, the target z-scored in-graph and inverted,
and the context presented a second time as queries so `is_training` rows are real in-context values.
Regression only (upstream's `build_model` refuses any other `task_type`).

| # | upstream | rewrite | why |
|---|---|---|---|
| 1 | `FeaturesTransformer.forward`: the feature-pair pad behind `if feature_to_add > 0`, the label pad `nan * zeros(B, T - S, 1)`, the mask-prediction dict | a regression-only forward that always pads and takes the label pad as the tail of a `T`-long tensor | a Python branch on the symbolic width; a concretised pad length |
| 2 | `QASSMaxScaling.forward`: `math.log(float(key_len))` | `log(ones(S).sum())` | a Python float of the symbolic context length specialises the graph |
| 3 | `ValidFeatureEncoder.forward`: `int(eval_pos)`, `self.valid_feature_num = ...` | removed | same specialisation; a module write during forward |
| 4 | `NormalizationEncoder.forward`: `self.mean = ...` | removed | a module write during forward |
| 5 | the label embedding's `.squeeze(2)` | squeeze only a 4-D output | a torch no-op on the 3-D output exports as an unconditional `Squeeze` that ORT rejects |

Every patch keeps the original as `_upstream_forward`; `nori_patches.unpatched()` restores them, and the tests
compare the patched model with upstream's own forward bit for bit.

The 999-quantile head is reduced to upstream's default point estimate, the mean of the piecewise-linear
quantile function on the uniform grid `tau_k = (k + 1) / 1000`, in closed form without a sort
(`quantile_mean`). `configs.assert_shipped_path` fails loudly if a checkpoint declares a non-uniform grid.

## The fixture

A 2-layer random-init model in the released file's own `torch.save` layout. A default initialisation decodes
every row to nearly one value (DeepNorm 0.19 and std-0.02 output projections barely move the target token, and
the mean over random quantile rows averages out), so `fixture.sensitive` scales the attention-output and
SwiGLU-down projections and gives the quantile head one shared direction. The golden inputs avoid standardised
values at or near zero, which Nori's decimal tokenizer (LimiX's) treats as a separate token.

## The ROCm graph (masked contract) and the one-time converter

`nori_mask.py` computes the same function with the split as a VALUE, so MIGraphX can compile it per (T, H)
shape bucket: `x[1,T,H]`, `y[1,T]` (full length, -100 off the context), `train_size`, `d`, `n_rows`. The
preprocessing statistics, the label imputation and the target-aware embedding read the context only, padded
feature GROUPS are masked out of the feature attention (`2 * g < d`, division-free), the sample attention masks
non-context keys, and QASSMax reads `log(S)` from the value. The tokenizer is re-expressed without the integer
floor-division / modulo and the `torch.split` MIGraphX cannot take (`rbf_tokenizer`, equal to upstream bit for
bit), and the attention projections are GEMMs instead of einsums. `mask_parity.gate` passes only when each of
four negative controls is caught.

MIGraphX reads the weights itself, as external data in a file named `model.safetensors`, so ROCm needs the
checkpoint converted once. `CALL tabfm_download('regression', model := 'nori-30m')` does that natively; the
converter below writes the same bytes and stays as the reference for building the graph (the CPU path keeps
reading `nori.pt`; its answers are identical either way):

```bash
C=~/.cache/anofox-tabfm/Synthefy__Nori-30M@63c9f7facf9fb32c37ce3fc2fba331d524696318
uv run convert_nori_weights $C/model.ckpt $C/model.safetensors        # header sha 49797138...; tabfm_download writes the same file
uv run export_nori --config real --contract mask --out ./mask
uv run python ../make_migraphx_graph.py regression --weights $C/model.safetensors \
    --graph ./mask/graph_mask_nori_regression.onnx --tensor-map ./mask/tensor_map_mask_nori_regression.json \
    --dynamic --inline-ok m.encoder_x.0.numeric_mlp.centers \
    --out ../../resources/graph_migraphx_nori30m_regression.onnx
uv run python ../check_migraphx_int_div.py ../../resources/graph_migraphx_nori30m_regression.onnx
```

`--inline-ok` names the tokenizer's RBF centres: a `linspace` built from the config, not a checkpoint weight.

What MIGraphX (2.15, gfx1201) could not compile, each found by compiling small extracted graphs on the GPU and
each rewritten to the same function: the ONNX `LayerNormalization` op (its scale/bias `broadcast[axis]` broke
`simplify_reshapes`: "BROADCAST : axis 2 is out of range"), the regression-target statistics when the boolean
mask was transposed before masking (same error), and QASSMax's `base_mlp` applied to a `[1, 1]` tensor
("cannot create std::vector larger than max_size()"). All three are written out in `nori_mask.py`.

## Nori 1.5 ROCm masked contract (offline half; NOT yet compiled on a GPU)

`nori15_mask.py` is the analogue of `nori_mask.py` for Model v3 (two-phase encoder), same signature (`x[1,T,H]`,
`y[1,T]` with -100 off the context, `train_size`, `d`, `n_rows` -> `logits[1,T,1]`; `features_per_group = 1`, so
masking is per column). `export_nori --config real15 --contract mask [--gate]` (or `fixture15`) writes the graph;
`mask_parity15.gate` is the gate (padded + masked == positional within `mask_parity.TOL`, and eight negative
controls each caught by more than 10x TOL: statistics over every row, non-context rows in the PLE edges, query rows
in the row attention, query rows in the column attention, no key mask in the column attention, padded columns in
the feature attentions, QASS sample length from the padded shape, QASS feature length from the padded width).
On the real weights: `TABFM_REAL_WEIGHTS=1 NORI15_CKPT=... pytest tests/test_real_weights_v15.py`.

What differs from the 30M contract, and what was rewritten:

| # | upstream | rewrite | why |
|---|---|---|---|
| 1 | `PromptQuantilePLEEncoder._build_reference`: `sort` the valid context values per column, `gather` 2 x 65 order statistics | `ple_edges`: a stable rank of every valid context value (`#smaller + #equal-and-earlier`, a compare matrix reduced by a GEMM), then the k-th order statistic as a one-hot GEMM `sum_j [rank_j == k] v_j`; the interpolation is upstream's float for float | Sort and data-dependent Gather are the likeliest MIGraphX failures, and the sort must ignore non-context rows and padded columns. Equal to upstream **bit for bit** (`torch.equal`) on decades, zero, constant, duplicate, NaN, all-NaN, one-value and inf columns, with junk rows/columns around them. Cost: an O(T^2 * H) compare matrix. |
| 2 | `ValidFeatureEncoder` rescale | dropped | exactly 1 at `num_features = 1` (`sqrt(1 / max(valid, 1))`) |
| 3 | QASSMax `log_only` uses the `key_len` of THAT attention and leaves q alone for `key_len <= 1` | `log(max(n, 2))` or 1, with `n` = S for column / row attention, `d + K` for the cells' feature attention, `d` for the read-outs and the read-back | in 1.5 the feature attentions are QASS-scaled too, so `d` (a value) enters the logits, not only the mask |
| 4 | column attention `columns[:, :, :eval_pos]`, row attentions `normalized[:, :, :eval_pos]` | keys are the context copy's first T rows under a `row < S` mask | S is a value; every query still reads context rows only |
| 5 | per-head q/k `RMSNorm`, `Linear(1, n)` on the label | RMSNorm written out; the `Linear(1, n)` layers (target MLP, boundary label encoder, target-aware embedding) as broadcast multiply-adds | no fused norm op and no K = 1 GEMM (the 30M compile failed on exactly those kinds of op) |
| 6 | tokenizer | the 30M `rbf_tokenizer` (no integer div / modulo / split) plus the PLE channels | same as 30M |

Measured offline: padded-vs-positional worst 5.5e-5 (fixture15), 7.7e-5 (real weights; 6.2e-4 on a real-dims random
model woken up for the CLI `--gate`); ORT on the exported masked graph vs positional 1.0e-4 on the real weights; every
negative control moves the answer by >= 0.27 (fixture) / >= 0.79 (real weights). The `make_migraphx_graph.py` recipe
works on the graph (`--inline-ok m.encoder_x.0.numeric_mlp.centers,m.x_preprocess.3.quantile_levels`, comma
separated; `check_migraphx_int_div.py` clean). The op set has no `SplitToSequence`, `SequenceAt`, `Einsum`, `Unique`,
`NonZero`, `TopK`, `GatherElements`, `GatherND` or `LayerNormalization`; the remaining `Gather`s are embedding
lookups and shape arithmetic.

**Unverified on a real MIGraphX** (the first compile on gfx1201 is a separate step). Constructs at risk, most likely
first:

1. The PLE compare matrix `[H, T, T]` (`Less`/`Equal`/`Cast`/`Mul`, then a batched GEMV): `4 * H * T^2` bytes per
   intermediate (H = 32, T = 2048 is 0.5 GB, several live at once), and MIGraphX may not fuse it. A bucket cap on T,
   or chunking over columns, may be needed.
2. The one-hot selection GEMM `[H, 130, T] @ [H, T, 1]`, and the equality of float ranks against `floor`/`ceil` of
   float positions (exact in fp32; any fp16/TF32 downcast of the rank matmul would break it: keep it fp32).
3. Column attention with `Lq = 2T`, `Lk = T`, batch `N = H`; and `N = 2T` batches of tiny `[G, G + K]` feature
   attentions: very many small matmuls/softmaxes.
4. The additive `-1e4` key mask before Softmax on scores already inflated by the QASS factor: fine in fp32,
   untested in fp16.
5. `Where` on a runtime `d` / `train_size` compare, `Log` of a runtime scalar, `Ceil`/`Floor` on `[H, 65]`.
6. float64 in the tokenizer (inherited from 30M, where it compiled) and ~220 `ReduceMean(x^2)` for the RMSNorms.
7. Size: ~8.4k nodes at the released dims; first-compile time per (T, H) bucket is unknown.
