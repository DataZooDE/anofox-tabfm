"""Build the random-init CI fixture for test/fixtures/tabpfn/ (both tasks).

Per task: weight-free graph + tiny random safetensors (checkpoint-namespace
keys, _pos_base excluded/inline) + tensor map + golden_<task>.json parity slice.
Classification: logits[1,T,C] class logits. Regression: logits[1,T,1] RAW-space
point estimate (bar-distribution mean, de-standardized — see tabpfn_patched.
ExportWrapper). Everything is seeded/random-init — NO TabPFN weight bytes (the
regression borders are random-init too; a real export injects criterion.borders).
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
T, H, N = cfg.example


def build_task(task: str):
    """Export one task's weight-free graph + safetensors + golden slice.

    Returns (list_of_written_filenames, manifest_fragment_dict).
    Classification keeps the historical single-task file names (model.safetensors,
    golden.json); regression uses *_regression suffixes (mitra-style)."""
    st_name = "model.safetensors" if task == "classification" else "model_regression.safetensors"
    golden_name = "golden.json" if task == "classification" else "golden_regression.json"
    graph_name = f"graph_tabpfn_{task}.onnx"
    map_name = f"tensor_map_tabpfn_{task}.json"

    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = MAXC
    model = build_random_model(task, kw, seed=SEED)
    wrapper = ExportWrapper(model, task=task).eval()

    graph_path = OUT / graph_name
    map_path = OUT / map_name
    export.export_graph(model, graph_path, example=cfg.example,
                        max_classes=MAXC, task=task)
    tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
    export.write_tensor_map(map_path, tensor_map, task=task,
                            safetensors_rel=st_name)

    # safetensors: the mapped weights only, keyed by checkpoint-namespace key.
    sd = dict(model.state_dict())
    mapped_keys = set(tensor_map["initializers"].values())
    weights = {k: sd[k].contiguous() for k in sorted(mapped_keys)}
    st_path = OUT / st_name
    save_file(weights, str(st_path))
    st_sha = hashlib.sha256(st_path.read_bytes()).hexdigest()

    # golden: fixed input -> PyTorch logits (the contract the C++ engine reproduces).
    torch.manual_seed(123)
    x = torch.randn(1, T, H)
    if task == "classification":
        y = torch.arange(N, dtype=torch.float32).remainder(MAXC)[None, :]
        out_doc = ("logits[1,T,C] class logits; predictions occupy rows >= N.")
        y_doc = ("y[1,N] = TRAIN labels only (N=train_size, dense 0..C-1).")
    else:
        # RAW continuous train targets with a non-trivial mean/scale so the
        # in-graph z-norm + de-standardization is exercised.
        y = (torch.randn(1, N) * 2.5 + 5.0).to(torch.float32)
        out_doc = ("logits[1,T,1] RAW-space point estimate (bar-distribution "
                   "mean, de-standardized); predictions occupy rows >= N.")
        y_doc = ("y[1,N] = RAW TRAIN targets only (N=train_size); the graph "
                 "z-normalizes internally. Engine must NOT standardize the "
                 "target nor inverse-transform the output.")
    with torch.no_grad():
        logits = wrapper(x, y).numpy()

    golden = {
        "_doc": {
            "purpose": (f"C++ parity: {st_name} -> initializer injection -> ORT "
                        f"run on {graph_name} reproduces these fp32 logits."),
            "input_contract": ("x[1,T,H] raw features (TabPFN scales internally); "
                               + y_doc + " cat_mask/d/train_size are NOT graph inputs."),
            "output_contract": out_doc,
            "parity_slice": "predictions occupy logits[:, N:, :]; rows < N are zero pad.",
            "rtol": 1e-3,
        },
        "task": task,
        "inputs": {"x": x.numpy().tolist(), "y": y.numpy().tolist(),
                   "train_size": N},
        "logits": logits.tolist(),
        "safetensors_sha256": st_sha,
    }
    (OUT / golden_name).write_text(json.dumps(golden, indent=2) + "\n")

    export.delete_weight_data(graph_path)
    export.assert_weight_free(graph_path, tensor_map)

    # self-check: inject weights, run ORT, compare to golden test rows
    import onnx, onnxruntime as ort
    proto = onnx.load(str(graph_path), load_external_data=False)
    name_by_init = {i.name: i for i in proto.graph.initializer}
    for onnx_name, st_key in tensor_map["initializers"].items():
        arr = weights[st_key].detach().cpu().numpy()
        if tensor_map["transforms"].get(onnx_name) == "transpose":
            arr = np.ascontiguousarray(arr.T)
        name_by_init[onnx_name].CopyFrom(onnx.numpy_helper.from_array(arr, onnx_name))
    tmp = OUT / f"_injected_{task}.onnx"
    onnx.save(proto, str(tmp))
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    (r,) = sess.run(["logits"], {"x": x.numpy(), "y": y.numpy()})
    d = float(np.abs(r[:, N:] - logits[:, N:]).max())
    tmp.unlink()
    print(f"  [{task}] graph {graph_path.stat().st_size} B, st {st_path.stat().st_size} B, "
          f"{len(tensor_map['initializers'])} mapped, SELF-CHECK maxdiff {d:.2e}")
    assert d < 1e-3, d

    frag = {
        "st_name": st_name, "st_bytes": st_path.stat().st_size,
        "graph_name": graph_name, "map_name": map_name,
        "golden_name": golden_name,
    }
    return [st_name, golden_name, graph_name, map_name], frag


written = ["manifest.json", "FIXTURE_SHA256"]
frags = {}
for task in ("classification", "regression"):
    files, frag = build_task(task)
    written += files
    frags[task] = frag

manifest = {
    "schema_version": 2,
    "id": "tabpfn-v2-fixture",
    "display_name": "TabPFN v2 (random-init CI fixture)",
    "family": "icl-transformer",
    "license": {"id": "apache-2.0", "commercial": True, "redistributable": True,
                "attribution": "Random-init fixture; no Prior Labs weights.",
                "gate_setting": None},
    "preprocessing_profile": "tabpfn_v2_raw",
    "weights": {
        "classification": {"files": [
            {"path": frags["classification"]["st_name"],
             "bytes": frags["classification"]["st_bytes"]}]},
        "regression": {"files": [
            {"path": frags["regression"]["st_name"],
             "bytes": frags["regression"]["st_bytes"]}]},
    },
    "graph": {
        "classification": frags["classification"]["graph_name"],
        "regression": frags["regression"]["graph_name"],
        # Per-task tensor maps: TabPFN's classification and regression graphs are
        # DIFFERENT (129 vs 130 initializers; regression adds `regression_borders`
        # and a 5000-bucket head), so unlike Mitra they cannot share one map. The
        # engine must resolve tensor_map[task] when this object is keyed by task
        # name (vs an inline {onnx -> st} map like Mitra's).
        "tensor_map": {
            "classification": frags["classification"]["map_name"],
            "regression": frags["regression"]["map_name"],
        },
    },
    "capabilities": ["classify", "regress"],
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
    "_note": (
        "Random-init TabPFN v2 (emsize=32, 2 layers). train_size is conveyed as "
        "len(y) (train-target prefix), NOT a scalar input; cat_mask/d omitted. "
        "REGRESSION: y is RAW train targets and logits[1,T,1] is a RAW point "
        "estimate — the graph z-normalizes y and de-standardizes its own output, "
        "so the engine must feed raw targets and neither standardize them nor "
        "inverse-transform the output. See tools/export_tabpfn for the recipe."),
}
mpath = OUT / "manifest.json"
mpath.write_text(json.dumps(manifest, indent=2) + "\n")

# SHA256SUMS for CI pinning
sums = []
for f in sorted(set(written)):
    if f == "FIXTURE_SHA256":
        continue
    h = hashlib.sha256((OUT / f).read_bytes()).hexdigest()
    sums.append(f"{h}  {f}")
(OUT / "FIXTURE_SHA256").write_text("\n".join(sums) + "\n")

print("fixture written to", OUT)
print("  files:", ", ".join(sorted(set(written))))
