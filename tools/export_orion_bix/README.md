# export_orion_bix — Orion-BiX (Lexsi Labs) → weight-free ONNX

WS-A exporter for onboarding **Orion-BiX** (`Lexsi/Orion-BiX`, MIT) into the
anofox-tabfm engine. Mirrors `tools/export_tabicl` — Orion-BiX is a TabICL
descendant, so the upstream package is pinned by **git sha** (it is not on PyPI)
and patched at runtime rather than vendored. No upstream source and no weight
bytes live in this repo.

```bash
uv sync
# shipping graph (real dims) -> resources/
uv run export_orion_bix --config real --out ../../resources
# committed CI fixture (random-init graph + safetensors + golden)
uv run make_orion_bix_fixture ../../test/fixtures/orion_bix
uv run pytest   # export, dynamism, license-wall and fixture-pin guards
# verify the committed tensor map covers the real checkpoint 1:1
uv run python convert_weights.py
```

## Exported contract (engine feeds by NAME)

```
inputs:  x [1,T,H] f32   (preprocessed features, all rows; H dynamic)
         y [1,S]   f32   (TRAIN labels only; train_size is implicit = len(y))
output:  logits [1,T,C] f32   (C = max_classes = 10)
```

Same `(x, y)`-only shape as TabPFN and TabICL, so the engine's y-as-train-prefix
path drives it with **no model-specific C++**. `cat_mask`/`d`/`train_size` are
omitted; the engine reads predictions on rows `>= S`.

**Classification only.** Upstream ships `sklearn/classifier.py` and no regression
head, so the model spec declares `capabilities: ["classify"]` and
`tabfm_regress(..., model := 'orion-bix')` raises the unsupported-task error.

## What the released checkpoint actually is

Read from `Orion-BiX-v1.1.ckpt`'s own embedded `config` (27,051,666 params,
277 tensors), and transcribed into `configs._REAL_KWARGS`:

| flag | value | consequence |
|---|---|---|
| `col/row/icl_attention_type` | **`"standard"`** | despite the model's name, `BiAxialAttention` and `LinearAttentionBlock` are **not instantiated** — the traced path is ISAB → MHA+RoPE → MHA |
| `*_feature_map` | `"identity"` | linear-attention feature maps unused |
| state_dict keys | `_orig_mod.*` | saved from a `torch.compile`-wrapped module |

`configs.assert_shipped_path` fails loudly if a future checkpoint flips an
attention flag, rather than silently exporting a different architecture.

## What is patched (`orion_bix_patches.py`)

Runtime monkeypatches, each mathematically identical to upstream on the
inference path:

| upstream | rewrite | why |
|---|---|---|
| `SkippableLinear.forward` | branchless `torch.where` | `if skip_mask.any(): out[mask]=skip` is a data-dependent branch |
| `InducedSelfAttentionBlock.forward` | compute-all, then mask | same skip pattern; LayerNorm of a constant column is 0, never NaN |
| `RotaryEmbedding.__init__` | `cache_if_possible=False` | the freq cache becomes a **FakeTensor** under tracing (serialization dies), and would bake the export example's row count into the rotary table |

The wrapper also puts the model in `train()`: like TabICL, Orion-BiX dispatches
on `self.training`, and the eval path routes through `InferenceManager`
(chunking, memory probing, Python loops). With `dropout=0.0` in the released
config the two are numerically identical.

Verified: parity **2.24e-07** (budget 1e-3) at shapes different from the export
example, 277/277 initializers mapped, and `convert_weights.py` confirms all 277
map keys are present in the real checkpoint.

## Preprocessing

`orion_bix_v1_minimal` — **not** a `*_raw` profile. Unlike TabPFN/TabICL,
Orion-BiX does no internal normalization; its sklearn wrapper standardizes
externally (`CustomStandardScaler`, `RTDLQuantileTransformer`,
`OutlierRemover`). The engine therefore standardizes features, as it does for
`tabfm-v1` and `mitra`.
