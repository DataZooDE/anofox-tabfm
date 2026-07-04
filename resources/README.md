# resources/ — TabFM weight-free ONNX graphs (embedded into the extension)

These are the **architecture-only** ONNX graphs and tensor-name maps the
extension embeds and ships. They contain **zero bytes of Google's weights**
(BRD FR-2.2, HLD §4.4 / §5 license wall): every checkpoint initializer is an
`EXTERNAL` stub pointing at a `graph_<task>.onnx.data` file that is **never
generated for shipping and never committed**. At runtime the extension
injects weights from a user-downloaded `model.safetensors` via
`AddExternalInitializers` (S02), using the tensor-name map below. Creating an
ORT session on the bare graph fails cleanly (`ONNXRuntimeError ... External
data path validation failed for initializer: m.cls_tokens`) — that is the
"weights not injected" guard, verified here.

## Files

| file | bytes | sha256 |
|---|---|---|
| `graph_classification.onnx` | 1,100,818 | `803611913c9751513c2eba78682c363fc201d94248fef016b6909a5d816395f3` |
| `tensor_map_classification.json` | 101,211 | `db36611f5735b9756295869d2c91a86ec3c0058ae11d8eb74dd388a5f9f4f7cc` |
| `graph_regression.onnx` | 1,098,500 | `e4e8a8b588abc9851498f9230c91f6db9df49dfabee09c47a279d0edd893cbc9` |
| `tensor_map_regression.json` | 101,709 | `0ddc87ee09c60acbb80be2567332c9130e35a904cc7566442df7fe0a4ef93b56` |

Each graph is ~1.1 MB (pure ONNX node/graph structure; ~3.5k+ nodes). The
tensor map is `{ONNX initializer name -> checkpoint-namespace safetensors
key}` (the wrapper's `m.` prefix stripped); `transforms` is empty
(name-matching maps 100% under `optimize=False`).

## Provenance

- **Exporter:** `tools/export_onnx` (WS-A), S01 shipping pipeline.
- **Exporter versions:** torch 2.12.1+cpu, onnx 1.22.0, onnxruntime 1.27.0,
  onnxscript 0.7.1, Python 3.11.15.
- **Opset:** 18 (`ai.onnx` only), IR v10. Dynamic dims: T (rows) min 4 /
  max 100000, H (features) min 2 / max 512; `d` and `train_size` are runtime
  value inputs.
- **Architecture dims** read from the real model's `config.json`, fetched
  **anonymously** from Hugging Face (dims only — no weights ever touched):
  - classification: `https://huggingface.co/google/tabfm-1.0.0-pytorch/resolve/main/classification/config.json`
    sha256 `d61e14960ab879bf9d0d23a10acac6e09f780ea548deac2fe3999eb131377e16`
  - regression: `.../regression/config.json`
    sha256 `169c7a935965b45c4ed9d981201392743e498115ceddb29931eccd37e0c8f6ad`
  - dims: embed_dim 256, max_classes 10, col_num_blocks 3 / col_nhead 4 /
    col_num_inds 256, row_num_blocks 3 / row_nhead 8 / row_num_cls 8,
    icl_num_blocks 24 / icl_nhead 8, ff_factor 4, feature_group_size 3,
    num_freq 32 (≈1.64 B params). The graph is built with **RANDOM** weights
    at these dims — it is weight-free, so no Google bytes are involved.
- **Parity** (ORT vs PyTorch fp32, random weights, test rows, budget 1e-3):
  classification worst **6.5e-06**, regression worst **1.9e-06**.
- **Unmapped inline constants** (Apache-2.0 code, not checkpoint — stay
  inline on purpose): classification has one 8-byte
  `m.icl_predictor.y_encoder.lifted_tensor_0` (the `torch.tensor(num_classes)`
  constant from `OneHotAndLinear`); regression has none.

## Regeneration (command line)

```bash
cd tools/export_onnx
uv run export_onnx --task classification --config real --out ../../resources
uv run export_onnx --task regression   --config real --out ../../resources
# each: fetches config.json anonymously, builds a random-weight ~1.6B model,
# dynamo-exports, strips metadata, force-externalizes all checkpoint
# initializers (threshold 0), parity-checks, then DELETES the .onnx.data.
# Peak RSS ~13 GB during postprocess; ~10 min per task on CPU.
```

The `export_report_<task>.json` files record the exact command, versions,
config source + sha, parity numbers, and sizes for each build.
