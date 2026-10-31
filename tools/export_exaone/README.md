# export_exaone — EXAONE Tabular v1 (LG AI Research) → weight-free ONNX

WS-A exporter for the built-in **`exaone-tabular`** model (`LG-AI-Research/EXAONE-Tabular`). Upstream code is
the `exaonetabular` package (BSD-3-Clause-LG), not on PyPI, so it is pinned to an exact git commit in
`pyproject.toml`. The WEIGHTS are under the **EXAONE AI Model License Agreement 1.2 - NC** (research and
educational use only, no commercial use of the Model, Derivatives or Output); running the real-weight tests
here is evaluation. No upstream source is copied and no weight bytes are committed.

```bash
cd tools/export_exaone && uv sync
uv run export_exaone --task classification --config real --out ./out   # graph + tensor map, weight-free
uv run export_exaone --task regression     --config real --out ./out
uv run make_exaone_fixture ../../test/fixtures/exaone                   # the committed random-init CI fixture
uv run pytest                                                           # TABFM_REAL_WEIGHTS=1 adds the real-weight tests
```

The shipped copies live in `resources/` as `graph_exaone_tabular_<task>.onnx` and
`tensor_map_exaone_tabular_<task>.json`. LG publish safetensors keyed by the module namespace, so there is no
converter. Status, measurements and the licence reasoning are in `docs/REAL_MODELS.md`.

## What the exporter does

Upstream's forward is built for a CUDA runtime: argument validation that a symbolic size fails, attention
weights packed lazily into buffers keyed by `data_ptr()`, support keys/values cached in module state between
the halves of a layer, FlashAttention pinned for part of the stack, `math.log` of a symbolic length for the
SSMax temperature, and `torch.unique` in the classification label encoder. `exaone_patches.py` therefore
re-expresses the INFERENCE forward over upstream's own modules and parameters rather than patching it:

- the plain attention path `softmax(q k^T / sqrt(d)) v` (the same function as the SDPA kernels);
- the queries' multi-query cross-attention onto the first context head, as upstream;
- SSMax as `k * log(context_length)` with the length taken from a tensor;
- a first-occurrence rank table instead of `torch.unique` for class labels, and branchless label imputation;
- every layer computed with its support rows updated (upstream's final layer skips that update, which no
  query reads).

`upstream_forward` calls upstream's own `model.forward` (one member, float32 CPU) and every test compares
against it: 2.9e-6 on the real weights.

The wrapper is `(x, y)`-only, standardises the regression target as `EXAONETabularRegressor.fit` does (float32,
population variance, rounding-level variance counts as constant), reads the mean of the 999 quantiles (the
released regressor's `point_estimate="mean"`), and presents the context a second time as queries so
`is_training` rows are real in-context values.

## The fixture

2 CAST blocks, width 24, random init, one safetensors per task. A default initialisation collapses to one class
(the head's GELU hidden layer is mostly positive, so a random row's sum picks the class for every input) and
its quantile mean averages out, so `fixture.sensitive` centres the class head's rows, gives the quantile head
one shared direction, and scales the label encoder for regression. Even so no seed made a random 2-block model
predict all three classes, so the builder requires two (context rows included) and a top-1/top-2 margin above
0.05 on every row.
