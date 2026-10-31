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
