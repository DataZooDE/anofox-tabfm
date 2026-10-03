# export_limix — LimiX-2M (StableAI) → weight-free ONNX

WS-A exporter for the built-in **`limix-2m`** model (`stable-ai/LimiX-2M`). Upstream ships no packaging
metadata, so it is pinned as the `vendor/limix` git submodule (the convention `vendor/tabfm` uses) and patched
at runtime: no upstream source is copied into this repo, and no weight bytes are committed anywhere.

```bash
git submodule update --init vendor/limix
cd tools/export_limix && uv sync
uv run export_limix --task classification --config real --out ./out     # graph + tensor map, weight-free
uv run export_limix --task regression     --config real --out ./out
uv run convert_limix_weights <LimiX-2M.ckpt> model.safetensors          # optional: enables the ext graphs
uv run make_limix_fixture ../../test/fixtures/limix                     # the committed random-init CI fixture
uv run pytest                                                           # TABFM_REAL_WEIGHTS=1 adds the real-weight tests
```

The shipped copies live in `resources/` as `graph_limix2m_*.onnx`, `graph_ext_limix2m_*.onnx` (made with
`tools/make_external_graph.py`) and `tensor_map_limix2m_*.json`. The stem is `limix2m`, not `limix`, so LimiX-2
(a different, 400 M-parameter model) can never be mistaken for it. Status, measurements and the licence
reasoning are in `docs/REAL_MODELS.md`; this file is about the exporter.

## Licence

Upstream **code** is Apache-2.0 (`vendor/limix/LICENSE.txt`). The **weights** are under the *Stable AI
Technology Co., Ltd. License v1.0* (`LICENSE.txt` in the weights repo, 2026-09): Apache-2.0 plus a Section 10
that permits commercial use but requires the attribution **"Built with StableAI LimiX"** when you distribute or
make available the Work or a product built on it, and a **"LimiX"** prefix on the name of an AI model derived
from the weights; internal research, evaluation, benchmarking and testing are exempt. The older upstream README
says "academic research, commercial with authorization" and the model card is internally inconsistent (front
matter `license: other`, body "Apache 2.0"); the registry follows the newest `LICENSE.txt` (`commercial: true`,
gated by `accept_hf_license`). Whether shipping weight-free graphs counts as "distributing the Work" is **not**
confirmed with StableAI. Running the real-weight tests here is evaluation.

## What the exporter does

Upstream's `forward(x, y, eval_pos, task_type=...)` already takes a row-major table plus a label prefix and a
split point, so the graph maps onto the engine's `(x, y)`-only contract with no engine change. `ExportWrapper`
exposes both tasks (classification → `[1,T,C]`, regression → `[1,T,1]`).

| # | upstream | rewrite | why |
|---|---|---|---|
| 1 | two `if torch.isnan(...).any(): raise` guards in `forward` | proxy whose `.any()` is statically False, installed only inside `model.transformer` | data-dependent branch; pure input validation |
| 2 | `mixed_y_embedding` splits labels by `y_type` with boolean-mask indexing | call the single live encoder directly | `y_type` is uniform per graph, so the split is the identity (proved against upstream by a test) |
| 3 | `y["data"][:, eval_pos:] = torch.nan` | `torch.where` against a row-index mask | an in-place dynamic slice assign bakes `index_put` shapes |
| 4 | `NanEncoder.forward`'s four boolean-mask assignments | `torch.where` | same shape-baking; elementwise-identical |
| 5 | the y decoder's split by `y_type` | call the live decoder | as 2 |
| 6 | `MulticlassTargetEncoder` ranks labels with `torch.unique` | a first-occurrence rank table | `Unique` has a data-dependent shape and is not in the MLX interpreter's op table; equal to upstream on duplicates, gaps, single-class and NaN-pad cases |
| 7 | `nn.ReLU` | `clamp(x, min=0)`, per instance | `Relu` is not in the MLX op table; identical function, exports as `Clip` |

Beyond those patches, the things that were wrong and are now guarded by tests (`tests/test_export.py`):

- **T and S were unified.** The traced reshape baked `T − S` as a constant, so ORT failed at every shape other
  than the trace shape. The pad is now the tail of a `T`-long tensor.
- **Odd feature widths.** `num_features % features_per_group` is a Python branch on a symbolic size; the group
  padding is now unconditional.
- **Export time exploded with depth** (8 layers took more than 480 s; the real model has 12) because the
  per-layer slices create `Min(S, …)` expressions sympy re-simplifies. `torch._check(S <= T)` makes them exact;
  the real config exports in about 25 s.
- **Context rows.** The wrapper presents the context a second time as queries, so `is_training` rows are real
  in-context values (decoding every row from one pass scored 0.111 on a 5-class problem against 1.000).
- **The regression target is not normalised by the model.** The wrapper z-scores it and inverts the output.
- **The engine injects every checkpoint tensor by name**, and ORT rejects one the graph lacks. The checkpoint
  holds both tasks' heads and an imputation head, none of which one task graph reads, so each unread tensor is
  declared as an initializer *and* feeds a `keep_*` Identity whose output nothing uses (ORT prunes an unused
  initializer when it loads the graph, before the engine injects; its dead-code removal runs after).

## Architecture facts (from `LimiX-2M.ckpt`'s embedded `config`)

2,377,837 parameters over 137 tensors, 12 layers, `embed_dim` 96, `layer_arch` `"smf"`. Two flags keep the
traced path narrow and are pinned by `configs.assert_shipped_path`: `mask_prediction = False` (the imputation
head is off, so `forward` returns logits) and `feature_positional_embedding_type = "none"` (avoids the runtime
`torch.randn` positional draw, untraceable at a symbolic column count). Checkpoint keys are in the **bare**
module namespace, so `export.CKPT_KEY_PREFIX` is `""`. The RBF tokenizer's `centers` are a deterministic
`linspace` built from the config (`use_random_kernels: False`), which is why the one small inline constant per
graph is correct for the real model.

## The numeric tokenizer, and why it matters

The tokenizer encodes a standardised value as (sign, decimal exponent, mantissa), and an **exact zero as its own
token**. A value near zero is therefore hypersensitive in proportion to 1/|z|, and an exact zero (a value equal
to its column mean, which discrete or symmetric data produces constantly) flips to a completely different
embedding under any rounding noise. Consequences, all measured and pinned:

- the fixture's golden inputs must avoid it (`fixture.assert_well_conditioned` rejects exact zeros and values
  closer to zero than 0.02; with quarter-multiple inputs 41 standardised values were exactly 0 and the logits
  moved by 0.04–20 under 1e-6 noise);
- the one ORT-vs-PyTorch outlier from the original spike (1.2e-3 at T=33, H=9, S=21) is this: its input has a
  standardised value of 3.2e-6 and is chaotic in eager PyTorch too (`test_the_ort_outlier_is_…`);
- the real model predicts a row sitting exactly on the training mean badly, and saturates when extrapolating,
  identically in upstream's own forward (`test_a_feature_value_exactly_at_the_training_mean_…`).
