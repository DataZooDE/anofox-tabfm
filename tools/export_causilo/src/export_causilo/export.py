"""Causilo -> weight-free ONNX graph + tensor-name map.

Mirrors tools/export_tabicl step for step, so the runtime injection path
(safetensors -> initializer -> ORT) is identical:

  1. dynamo export, ``optimize=False`` (keep dotted-FQN initializer names so the tensor
     map can key ONNX-init -> checkpoint safetensors key), opset 18, ``torch.export.Dim``
     for T (rows), S (train size) and H (features). Signature: x[1,T,H], y[1,S] -> [1,T,C].
  2. strip doc_string + metadata_props (dynamo stack traces).
  3. force-externalize EVERY checkpoint-mapped initializer so no weight bytes ship inline.
  4. tensor map: ONNX initializer name -> checkpoint safetensors key (strip the wrapper's
     ``m.`` prefix).
  5. parity: ORT vs PyTorch fp32 on random weights at shapes DIFFERENT from the export
     example, over EVERY row. (The TabICL exporter this was copied from compared test rows
     only; that cannot see a wrong context row, which is the defect class #51 was about.)
  6. delete the .onnx.data: the shipping artifact is graph + map only.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np
import onnx
import onnx.external_data_helper
import torch

from export_causilo.causilo_patches import ExportWrapper

OPSET = 18
# Budget for ORT vs PyTorch, fp32. Measured worst is ~2e-7, so 1e-4 leaves ~500x of headroom for
# platform noise while still being far below the effect of the bugs this gate exists for (the
# baked `log(key_count)` trap moves logits by ~6e-3 even on a tiny random model). The 1e-3 this
# was copied with would have hidden anything much smaller.
PARITY_TOL = 1e-4


def _pin_dynamic_slice_bounds(model_proto) -> None:
    """Re-apply the ScatterND slice-bound pins before saving (issue #21).

    The CUDA EP reads those bounds from a CPU-side buffer that has already been recycled,
    so the Slice trims nothing and ScatterND rejects the shapes. Naming the bounds as graph
    outputs excludes them from buffer reuse. CI re-checks this
    (.github/workflows/graph_invariants.yml); applying it here makes that a backstop.
    """
    import importlib.util

    root = pathlib.Path(__file__).resolve().parents[4]
    path = root / "tools" / "pin_dynamic_slice_bounds.py"
    spec = importlib.util.spec_from_file_location("pin_dynamic_slice_bounds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    existing = {o.name for o in model_proto.graph.output}
    pins = [(n, r) for n, r in module.targets(model_proto.graph) if n not in existing]
    for name, rank in pins:
        model_proto.graph.output.append(
            module.helper.make_tensor_value_info(
                name, module.TensorProto.INT64, [] if rank == 0 else [1]))
    if pins:
        print(f"  pinned {len(pins)} ScatterND slice bound(s): {[n for n, _ in pins]}")


def example_inputs(t, h, s, classes, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, t, h)
    y = torch.randint(0, classes, (1, s)).float() if classes else torch.randn(1, s)
    return x, y


def export_graph(model, graph_path: pathlib.Path, *, example, classes: int,
                 dim_rows=("rows", 4, 100_000), dim_train=("train", 2, 100_000),
                 dim_features=("features", 2, 512), opset: int = OPSET) -> ExportWrapper:
    """Step 1: dynamo export with dynamic T/S/H. Writes graph + .onnx.data."""
    wrapper = ExportWrapper(model).eval()
    # Dim.AUTO, not named Dims with min/max: Causilo groups features in threes, so the group
    # count is ceil(H / 3), and torch.export's derived-dim solver cannot build a root for
    # it from a bounded H ("Cannot create Dim with inconsistent min=0, max=-170").
    # AUTO lets the solver keep the relation without that bookkeeping (the TabPFN masked
    # exports do the same). The bounds are enforced by the engine, not the graph.
    A = torch.export.Dim.AUTO
    dyn = ({1: A, 2: A}, {1: A})  # x, y
    t, h, s = example
    ex = example_inputs(t, h, s, classes)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, ex, str(graph_path),
            dynamo=True, dynamic_shapes=dyn, opset_version=opset,
            input_names=["x", "y"], output_names=["logits"],
            external_data=True, optimize=False,
        )
    return wrapper


def _norm_name(name: str) -> str:
    # dynamo emits the wrapper attribute ('m.') as prefix; strip to the bare checkpoint key.
    for pre in ("m.", "p_m_", "b_m_", "p_", "b_"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def build_tensor_map(model_proto: onnx.ModelProto, state_dict: dict) -> dict:
    """ONNX initializer name -> checkpoint-namespace safetensors key."""
    sd_by_name = {k: k for k in state_dict}
    sd_by_sig, sd_by_sig_t = {}, {}
    for k, v in state_dict.items():
        arr = v.detach().cpu().numpy()
        sd_by_sig[(tuple(arr.shape), hashlib.sha1(arr.tobytes()).hexdigest())] = k
        if arr.ndim == 2:
            at = np.ascontiguousarray(arr.T)
            sd_by_sig_t[(tuple(at.shape), hashlib.sha1(at.tobytes()).hexdigest())] = k

    mapping, transforms, unmatched_small, unmatched_large = {}, {}, [], []
    for init in model_proto.graph.initializer:
        key = sd_by_name.get(_norm_name(init.name))
        if key is None:
            arr = onnx.numpy_helper.to_array(init)
            sig = (tuple(arr.shape), hashlib.sha1(arr.tobytes()).hexdigest())
            key = sd_by_sig.get(sig)
            if key is None:
                key = sd_by_sig_t.get(sig)
                if key is not None:
                    transforms[init.name] = "transpose"
            if key is None:
                (unmatched_large if arr.nbytes >= 1024 else unmatched_small).append(init.name)
                continue
        mapping[init.name] = key
    if unmatched_large:
        raise RuntimeError(
            f"unmatched large (>=1KB) initializers: {unmatched_large}; the tensor "
            "map must cover 100% of them")
    return {"initializers": mapping, "transforms": transforms,
            "unmatched_small": unmatched_small}


def postprocess(graph_path: pathlib.Path, state_dict: dict) -> dict:
    """Steps 2-4: strip metadata, externalize checkpoint initializers, build map."""
    model_proto = onnx.load(str(graph_path), load_external_data=True)
    del model_proto.metadata_props[:]
    model_proto.doc_string = ""
    model_proto.graph.doc_string = ""

    def _strip_graph(g):
        for node in g.node:
            node.doc_string = ""
            del node.metadata_props[:]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _strip_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sub in attr.graphs:
                        _strip_graph(sub)
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            vi.doc_string = ""
            del vi.metadata_props[:]

    _strip_graph(model_proto.graph)
    for fn in model_proto.functions:
        for node in fn.node:
            node.doc_string = ""
            del node.metadata_props[:]
        del fn.metadata_props[:]

    tensor_map = build_tensor_map(model_proto, state_dict)

    data_name = graph_path.name + ".data"
    data_path = graph_path.with_name(data_name)
    mapped = set(tensor_map["initializers"])
    for init in model_proto.graph.initializer:
        if init.name in mapped:
            onnx.external_data_helper.set_external_data(init, location=data_name)
    data_path.unlink(missing_ok=True)
    _pin_dynamic_slice_bounds(model_proto)
    onnx.save(model_proto, str(graph_path))
    return tensor_map


def write_tensor_map(map_path: pathlib.Path, tensor_map: dict, *, task: str,
                     safetensors_rel: str, opset: int = OPSET) -> None:
    payload = {
        "task": task, "opset": opset, "safetensors": safetensors_rel,
        "initializers": tensor_map["initializers"],
        "transforms": tensor_map["transforms"],
    }
    map_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_feed(t, h, s, classes, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    y = (rng.integers(0, classes, (1, s)) if classes else rng.standard_normal((1, s))).astype(np.float32)
    return {"x": x, "y": y}


def check_parity(graph_path: pathlib.Path, wrapper: ExportWrapper, shapes, classes: int,
                 tol: float = PARITY_TOL) -> dict:
    """ORT vs PyTorch fp32 on random weights, over EVERY row.

    ``shapes`` = ((T, H, S), ...), all different from the export example so the dynamic
    dims are genuinely exercised. Context rows (< S) and query rows (>= S) are reported
    separately so a failure says which half is wrong.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    results, worst = [], 0.0
    for t, h, s in shapes:
        feed = make_feed(t, h, s, classes)
        t0 = time.time()
        (ort_out,) = sess.run(["logits"], feed)
        ort_ms = (time.time() - t0) * 1e3
        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(feed["x"]), torch.from_numpy(feed["y"])).numpy()
        if ort_out.shape != pt_out.shape:
            raise RuntimeError(f"shape mismatch at (T,H,S)=({t},{h},{s}): ORT {ort_out.shape} vs torch {pt_out.shape}")
        ctx = float(np.abs(ort_out[:, :s] - pt_out[:, :s]).max())
        qry = float(np.abs(ort_out[:, s:] - pt_out[:, s:]).max()) if t > s else 0.0
        worst = max(worst, ctx, qry)
        results.append({"T": t, "H": h, "train": s, "max_abs_delta_context_rows": ctx,
                        "max_abs_delta_query_rows": qry, "ort_ms": ort_ms})
    return {"ok": worst < tol, "worst": worst, "tol": tol, "shapes": results}


def delete_weight_data(graph_path: pathlib.Path) -> None:
    graph_path.with_name(graph_path.name + ".data").unlink(missing_ok=True)


def assert_weight_free(graph_path: pathlib.Path, tensor_map: dict) -> None:
    data_path = graph_path.with_name(graph_path.name + ".data")
    if data_path.exists():
        raise RuntimeError(f"{data_path} still exists: weight bytes on disk")
    proto = onnx.load(str(graph_path), load_external_data=False)
    mapped = set(tensor_map["initializers"])
    for init in proto.graph.initializer:
        external = (init.data_location == onnx.TensorProto.EXTERNAL)
        if init.name in mapped and not external:
            raise RuntimeError(f"mapped initializer {init.name} is INLINE: weight bytes in graph")
