# export_kumo — NVIDIA Kumo-Tabular → weight-free ONNX

Onboards **Kumo-Tabular** (NVIDIA; code Apache-2.0, weights OpenMDW-1.1) at all three published sizes
as `kumo-tabular-small`, `kumo-tabular-medium` and `kumo-tabular-large`. Same pipeline as
`tools/export_causilo`: dynamo export (opset 18, `optimize=False` to keep the state_dict names) → strip
metadata → externalize every checkpoint initializer → tensor map → ORT-vs-PyTorch parity on every row →
delete `.onnx.data`.

```bash
uv sync
for s in small medium large; do for t in classification regression; do
  uv run export_kumo --task $t --size $s --out ../../resources
done; done
uv run make_kumo_fixture ../../test/fixtures/kumo
uv run convert_kumo_weights --size small        # .pt -> model.safetensors (tabfm_download now writes the same bytes)
uv run pytest                                   # fast exporter gates
TABFM_REAL_WEIGHTS=1 uv run pytest tests/test_real_weights.py -s   # parity with upstream, real weights

# GPU variants, per size and task (W = the cached weights dir):
python ../make_external_graph.py $t --weights $W/model.safetensors --graph ../../resources/graph_kumo_${s}_$t.onnx \
    --tensor-map ../../resources/tensor_map_kumo_${s}_$t.json --out ../../resources/graph_ext_kumo_${s}_$t.onnx
python ../make_migraphx_graph.py $t --weights $W/model.safetensors --graph ../../resources/graph_kumo_${s}_$t.onnx \
    --tensor-map ../../resources/tensor_map_kumo_${s}_$t.json --dynamic --out ../../resources/graph_migraphx_kumo_${s}_$t.onnx
```

## Contract

The tabfm-v1 five inputs, `x[1,T,H]`, `y[1,T]`, `train_size[1]`, `cat_mask[1,H]`, `d[1]` → `logits[1,T,C]`.
The wrapper (`kumo_patches.ExportWrapper`) calls upstream's own padded path,
`_KumoTabular.forward_train(x, y, categorical_mask, num_columns=d, train_size=train_size)`
(NVIDIA/structured-data-models #1204, #1297), so the split and the real width are values and padding is
inert. That is what lets one graph family serve the CPU, CUDA (ext graph) and bucket-padded ROCm
(migraphx graph). `structured-data-models` is pinned to the commit that has that path (PyPI 0.2.1 predates it).

Regression returns the mean of the 999 quantiles (upstream's own benchmark point estimate,
`benchmark/tabular/model.py`). Classification returns the 10 head logits; more than 10 classes is refused
(upstream uses error-correcting output codes, which this does not reproduce).

## What had to change, and why

1. **`index % num_columns` → conditional subtraction** (`kumo_patches._wrap`). The circular feature groups
   wrap at the real width. A data-dependent integer modulo is what MIGraphX's GPU target miscomputes
   (docs/ROCM_TABPFN_PLAN.md); the group shifts are 0, 1, 3, so four subtractions are exact.
2. **Unshared RoPE parameters** (`unshare_parameters`). Upstream hands one `RotaryEmbedding` to every row
   block's query and key transform, so 8 checkpoint keys are one tensor and the exporter emitted one
   initializer. The engine injects every checkpoint tensor by name and refused the graph. Deep-copying the
   shared leaf gives a 1:1 map (328/558/556 initializers for small/medium/large).
3. **`SplitToSequence`/`SequenceAt` → `Slice`** (`graph_passes.py`, a graph rewrite, not a module patch).
   The qkv split, RoPE's half split and the readout split lower to sequence ops, which neither MIGraphX's
   parser nor the MLX interpreter implements (166 in the small graph).
4. **`y` sanitised**: query and padding rows carry the engine's `-100` sentinel, which `Embedding` cannot
   index; upstream zeroes those rows' label embedding anyway, so class 0 there is the identical function.

## Preprocessing: the engine does it

Kumo's single-estimator numeric recipe (`DropConstantColumns`, `Standardize(eps=1e-6)`, `Clip(-100, 100)`,
`ClipSigma(4, "log")`, target `Standardize()`) is exactly the engine's standardising profile, in float64 on
both sides. So the graph takes standardised features (profile `kumo_tabular_v1_minimal`) and
`engine_ref.py` is a float64 replica used for the fixture goldens and the parity test.

## Measured

| size | params (clf / reg) | parity vs upstream, 1 estimator (CPU) |
|---|---|---|
| small | 27.5 M / 28.5 M | iris, wine, breast-cancer labels 100 %, probs ≤ 6.0e-6; diabetes ≤ 2.4e-4 (spread 321) |
| medium | 61.5 M / 62.5 M | labels 100 %, probs ≤ 6.7e-6; diabetes ≤ 3.8e-4 |
| large | 213.7 M / 215.7 M | labels 100 %, probs ≤ 1.5e-5; diabetes ≤ 4.1e-4 |

See docs/REAL_MODELS.md for the engine numbers, ROCm and what is not reproduced.
