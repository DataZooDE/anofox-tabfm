"""Build the random-init CI fixture for test/fixtures/tabpfn/.

Weight-free graph + tiny random safetensors (29 mapped weights, checkpoint
namespace, _pos_base excluded/inline) + tensor map + golden.json + v2 manifest.
Everything is seeded/random-init — NO TabPFN weight bytes.
"""
import hashlib, json, pathlib
import numpy as np
import torch
from safetensors.torch import save_file

from export_tabpfn import configs, export
from export_tabpfn.tabpfn_patched import build_random_model, ExportWrapper

SEED = 0
OUT = pathlib.Path(__file__).resolve().parents[2] / "test" / "fixtures" / "tabpfn"
OUT.mkdir(parents=True, exist_ok=True)
cfg = configs.fixture()
MAXC = cfg.max_classes

kw = dict(cfg.model_kwargs); kw["num_buckets"] = cfg.num_buckets
kw["max_num_classes"] = MAXC
model = build_random_model("classification", kw, seed=SEED)
wrapper = ExportWrapper(model).eval()

graph_path = OUT / "graph_tabpfn_classification.onnx"
map_path = OUT / "tensor_map_tabpfn_classification.json"
export.export_graph(model, graph_path, example=cfg.example, max_classes=MAXC)
tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
export.write_tensor_map(map_path, tensor_map, task="classification",
                        safetensors_rel="model.safetensors")

# safetensors: the mapped weights only, keyed by checkpoint-namespace key.
sd = dict(model.state_dict())
mapped_keys = set(tensor_map["initializers"].values())
weights = {k: sd[k].contiguous() for k in sorted(mapped_keys)}
st_path = OUT / "model.safetensors"
save_file(weights, str(st_path))
st_sha = hashlib.sha256(st_path.read_bytes()).hexdigest()

# golden: fixed input -> PyTorch logits (the contract the C++ engine reproduces).
torch.manual_seed(123)
T, H, N = cfg.example
x = torch.randn(1, T, H)
y = torch.arange(N, dtype=torch.float32).remainder(MAXC)[None, :]
with torch.no_grad():
    logits = wrapper(x, y).numpy()

golden = {
    "_doc": {
        "purpose": "C++ parity: safetensors -> initializer injection -> ORT run "
                   "on graph_tabpfn_classification.onnx reproduces these fp32 logits.",
        "input_contract": "x[1,T,H] raw features (TabPFN scales internally); "
                          "y[1,N] = TRAIN labels only (N=train_size, dense 0..C-1). "
                          "cat_mask/d/train_size are NOT graph inputs.",
        "parity_slice": "predictions occupy logits[:, N:, :]; rows < N are zero pad.",
        "rtol": 1e-3,
    },
    "inputs": {"x": x.numpy().tolist(), "y": y.numpy().tolist(),
               "train_size": N},
    "logits": logits.tolist(),
    "safetensors_sha256": st_sha,
}
(OUT / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")

# verify weight-free + ORT reproduces golden after injecting weights
export.delete_weight_data(graph_path)
export.assert_weight_free(graph_path, tensor_map)

manifest = {
    "schema_version": 2,
    "id": "tabpfn-v2-fixture",
    "display_name": "TabPFN v2 (random-init CI fixture)",
    "family": "icl-transformer",
    "license": {"id": "apache-2.0", "commercial": True, "redistributable": True,
                "attribution": "Random-init fixture; no Prior Labs weights.",
                "gate_setting": None},
    "preprocessing_profile": "tabpfn_v2_raw",
    "weights": {"classification": {"files": [
        {"path": "model.safetensors", "bytes": st_path.stat().st_size}]}},
    "graph": {"classification": "graph_tabpfn_classification.onnx",
              "tensor_map": "tensor_map_tabpfn_classification.json"},
    "capabilities": ["classify"],
    "tensor_contract": {
        "inputs": {
            "features": {"name": "x", "dtype": "f32", "shape": ["1", "T", "H"]},
            "labels": {"name": "y", "dtype": "f32", "shape": ["1", "N"]},
        },
        "outputs": {"logits": {"name": "logits", "dtype": "f32",
                               "shape": ["1", "T", "C"]}},
    },
    "size_regime": {"max_rows": 4096, "max_features": 64, "max_classes": 4},
    "compute": {"cpu": "f32"},
    "_note": "Random-init TabPFN v2 (emsize=32, 2 layers). train_size is conveyed "
             "as len(y) (train-label prefix), NOT a scalar input; cat_mask/d omitted. "
             "See tools/export_tabpfn for the export recipe.",
}
mpath = OUT / "manifest.json"
mpath.write_text(json.dumps(manifest, indent=2) + "\n")

# SHA256SUMS for CI pinning
sums = []
for f in ["golden.json", "graph_tabpfn_classification.onnx", "manifest.json",
          "model.safetensors", "tensor_map_tabpfn_classification.json"]:
    h = hashlib.sha256((OUT / f).read_bytes()).hexdigest()
    sums.append(f"{h}  {f}")
(OUT / "FIXTURE_SHA256").write_text("\n".join(sums) + "\n")

print("fixture written to", OUT)
print(f"  graph {graph_path.stat().st_size} bytes, safetensors {st_path.stat().st_size} bytes")
print(f"  {len(tensor_map['initializers'])} mapped initializers, "
      f"{len(tensor_map['unmatched_small'])} small inline")

# final self-check: inject weights, run ORT, compare to golden
import onnx, onnxruntime as ort
proto = onnx.load(str(graph_path), load_external_data=False)
name_by_init = {i.name: i for i in proto.graph.initializer}
for onnx_name, st_key in tensor_map["initializers"].items():
    arr = weights[st_key].detach().cpu().numpy()
    if tensor_map["transforms"].get(onnx_name) == "transpose":
        arr = np.ascontiguousarray(arr.T)
    new = onnx.numpy_helper.from_array(arr, onnx_name)
    name_by_init[onnx_name].CopyFrom(new)
tmp = OUT / "_injected.onnx"
onnx.save(proto, str(tmp))
sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
(r,) = sess.run(["logits"], {"x": x.numpy(), "y": y.numpy()})
d = float(np.abs(r[:, N:] - logits[:, N:]).max())
tmp.unlink()
print(f"  SELF-CHECK ORT(injected) vs golden test-row maxdiff: {d:.2e}")
assert d < 1e-3, d
