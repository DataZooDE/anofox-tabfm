"""TabICL -> weight-free ONNX graph + tensor-name map (export spike pipeline).

Mirrors ``tools/export_onnx`` (the TabFM shipping pipeline) step for step, so the
runtime injection path (safetensors -> initializer -> ORT) is identical:

  1. dynamo export, ``optimize=False`` (keep dotted-FQN initializer names so the
     tensor map can key ONNX-init -> checkpoint safetensors key), opset 18,
     ``torch.export.Dim`` for T (rows), S (train_size) and H (features).
     Signature: x[1,T,H] f32, y[1,S] f32 -> logits[1,T,C].
  2. strip doc_string + metadata_props (dynamo stack traces).
  3. force-externalize EVERY checkpoint-mapped initializer (threshold 0) so no
     weight bytes ship inline; unmapped inline code constants (the two scalar
     ``ssmax_layer.lifted_tensor_*`` = a traced log-of-length seed) stay inline.
  4. tensor map: ONNX initializer name -> checkpoint-namespace safetensors key
     (strip the wrapper's ``m.`` prefix).
  5. parity: ORT vs PyTorch fp32 on random weights at shapes DIFFERENT from the
     export example (dynamic T/S/H genuinely exercised).
  6. delete the .onnx.data — the shipping artifact is graph + map only.
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

from export_tabicl.tabicl_patches import ExportWrapper

OPSET = 18
PARITY_TOL = 1e-3  # budget; the spike measures ~1e-7


def example_inputs(t, h, s, max_classes, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, t, h)
    if max_classes > 0:
        y = torch.randint(0, max_classes, (1, s)).float()
    else:
        y = torch.randn(1, s)
    return x, y


def export_graph(model, graph_path: pathlib.Path, *,
                 dim_rows=("rows", 4, 100_000),
                 dim_train=("train", 2, 100_000),
                 dim_features=("features", 2, 512),
                 example=(20, 5, 12),
                 opset: int = OPSET) -> ExportWrapper:
    """Step 1: dynamo export with dynamic T/S/H. Writes graph + .onnx.data."""
    wrapper = ExportWrapper(model).train()
    T = torch.export.Dim(dim_rows[0], min=dim_rows[1], max=dim_rows[2])
    S = torch.export.Dim(dim_train[0], min=dim_train[1], max=dim_train[2])
    H = torch.export.Dim(dim_features[0], min=dim_features[1], max=dim_features[2])
    dyn = ({1: T, 2: H}, {1: S})  # x, y
    t, h, s = example
    ex = example_inputs(t, h, s, model.max_classes)
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
    # dynamo emits the wrapper attribute ('m.') as prefix; strip to the bare
    # checkpoint key. p_m_/b_m_ manglings kept as fallback.
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
            f"unmatched large (>=1KB) initializers: {unmatched_large} — the tensor "
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


def make_feed(t, h, s, max_classes, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    if max_classes > 0:
        y = rng.integers(0, max_classes, (1, s)).astype(np.float32)
    else:
        y = rng.standard_normal((1, s)).astype(np.float32)
    return {"x": x, "y": y}


def check_parity(graph_path: pathlib.Path, wrapper: ExportWrapper,
                 shapes, tol: float = PARITY_TOL) -> dict:
    """ORT vs PyTorch fp32 on random weights. Compares TEST rows (>= train_size).

    ``shapes`` = ((T, H, S), ...), all different from the export example so the
    dynamic dims are genuinely exercised.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    results, worst = [], 0.0
    for t, h, s in shapes:
        feed = make_feed(t, h, s, wrapper.m.max_classes)
        t0 = time.time()
        (ort_out,) = sess.run(["logits"], feed)
        ort_ms = (time.time() - t0) * 1e3
        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(feed["x"]), torch.from_numpy(feed["y"])).numpy()
        delta = float(np.abs(ort_out[:, s:] - pt_out[:, s:]).max())
        worst = max(worst, delta)
        results.append({"T": t, "H": h, "train": s,
                        "max_abs_delta_test_rows": delta, "ort_ms": ort_ms})
    return {"ok": worst < tol, "worst": worst, "tol": tol, "shapes": results}


def delete_weight_data(graph_path: pathlib.Path) -> None:
    graph_path.with_name(graph_path.name + ".data").unlink(missing_ok=True)


def assert_weight_free(graph_path: pathlib.Path, tensor_map: dict) -> None:
    data_path = graph_path.with_name(graph_path.name + ".data")
    if data_path.exists():
        raise RuntimeError(f"{data_path} still exists — weight bytes on disk")
    proto = onnx.load(str(graph_path), load_external_data=False)
    mapped = set(tensor_map["initializers"])
    for init in proto.graph.initializer:
        external = (init.data_location == onnx.TensorProto.EXTERNAL)
        if init.name in mapped and not external:
            raise RuntimeError(f"mapped initializer {init.name} is INLINE — weight bytes in graph")
