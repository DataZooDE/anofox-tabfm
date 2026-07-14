"""TabPFN v2 -> weight-free ONNX graph + tensor-name map.

Pipeline (mirrors the TabFM S01 shipping pipeline, tools/export_onnx/export.py):
  1. dynamo export, optimize=False (keep dotted-FQN initializer names), opset 18,
     Dim.AUTO for the row/feature/train dims. Signature: x[1,T,H] f32,
     y[1,N] f32 (N = train_size) -> logits[1,T,C]. train_size is the LENGTH of
     y, not a scalar input (TabPFN derives the split from len(y)); cat_mask/d are
     omitted (categoricals are ordinal-encoded upstream; features are
     auto-detected internally). See tabpfn_patched.ExportWrapper.
  2. strip doc_string + metadata_props.
  3. force-externalize EVERY checkpoint-mapped initializer (threshold 0) so no
     weight bytes ship inline. The code-generated positional-embedding base
     (`_pos_base`) is NOT a checkpoint weight: it stays INLINE and is excluded
     from the tensor map.
  4. tensor map: ONNX initializer name -> checkpoint-namespace safetensors key.
  5. parity: ORT vs PyTorch fp32 on random weights at shapes != the export
     example (test rows only).
  6. delete the .onnx.data file — the shipping artifact is graph + map only.
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

from export_tabpfn.tabpfn_patched import ExportWrapper

OPSET = 18
PARITY_TOL = 1e-3  # DoD budget; expect ~1e-6
# The one initializer that is code-generated (seed-derived), not a weight. It
# must stay inline in the weight-free graph and never enters the tensor map.
POS_BASE_NAME = "_pos_base"


def example_inputs(t, h, n, max_classes=4, seed=0, task="classification"):
    torch.manual_seed(seed)
    x = torch.randn(1, t, h)
    if task == "regression":
        # continuous RAW train targets (the wrapper z-normalizes internally)
        y = torch.randn(1, n, dtype=torch.float32)
    else:
        # dense 0..C-1 train labels, every class present (identity densification)
        y = torch.arange(n, dtype=torch.float32).remainder(max_classes)[None, :]
    return x, y


def export_graph(model, graph_path: pathlib.Path, *, example=(12, 5, 8),
                 max_classes=4, opset: int = OPSET,
                 task="classification") -> ExportWrapper:
    """Step 1: dynamo export with Dim.AUTO. Writes graph + .onnx.data."""
    wrapper = ExportWrapper(model, task=task).eval()
    A = torch.export.Dim.AUTO
    S = torch.export.Dim.STATIC
    dyn = ({0: S, 1: A, 2: A}, {0: S, 1: A})
    t, h, n = example
    ex = example_inputs(t, h, n, max_classes=max_classes, task=task)
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
    for pre in ("m.", "p_m_", "b_m_", "p_", "b_"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def build_tensor_map(model_proto: onnx.ModelProto, state_dict: dict) -> dict:
    """ONNX initializer name -> checkpoint-namespace safetensors key.

    `_pos_base` (code-generated) is excluded and kept inline. Returns
    {"initializers": {...}, "transforms": {...}, "unmatched_small": [...]}.
    """
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
        norm = _norm_name(init.name)
        if norm == POS_BASE_NAME:
            continue  # code-generated base stays inline, never mapped
        key = sd_by_name.get(norm)
        if key == POS_BASE_NAME:
            continue
        if key is None:
            arr = onnx.numpy_helper.to_array(init)
            sig = (tuple(arr.shape), hashlib.sha1(arr.tobytes()).hexdigest())
            key = sd_by_sig.get(sig)
            if key is None:
                key = sd_by_sig_t.get(sig)
                if key is not None:
                    transforms[init.name] = "transpose"
            if key == POS_BASE_NAME:
                continue
            if key is None:
                (unmatched_large if arr.nbytes >= 1024 else unmatched_small).append(
                    init.name)
                continue
        mapping[init.name] = key
    if unmatched_large:
        raise RuntimeError(
            f"unmatched large (>=1KB) initializers: {unmatched_large}")
    return {"initializers": mapping, "transforms": transforms,
            "unmatched_small": unmatched_small}


def postprocess(graph_path: pathlib.Path, state_dict: dict) -> dict:
    """Steps 2-4: strip metadata, externalize checkpoint initializers, map."""
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
        "task": task,
        "opset": opset,
        "safetensors": safetensors_rel,
        "initializers": tensor_map["initializers"],
        "transforms": tensor_map["transforms"],
    }
    map_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_feed(t, h, n, classes=4, seed=1, task="classification"):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    if task == "regression":
        # RAW continuous train targets with non-trivial mean/scale, to exercise
        # the in-graph z-norm + de-standardization.
        y = (rng.standard_normal((1, n)) * 3.0 + 7.0).astype(np.float32)
    else:
        y = rng.integers(0, classes, (1, n)).astype(np.float32)
    return {"x": x, "y": y}


def check_parity(graph_path: pathlib.Path, model, shapes, max_classes=4,
                 tol: float = PARITY_TOL, task="classification") -> dict:
    """ORT vs PyTorch fp32 on random weights. Compares TEST rows (>= N).

    classification: also checks argmax agreement over class logits.
    regression:     output is a scalar point estimate [1,T,1]; only the numeric
                    max-abs delta on test rows is meaningful (argmax is trivial).
    """
    import onnxruntime as ort

    wrapper = ExportWrapper(model, task=task).eval()
    sess = ort.InferenceSession(str(graph_path),
                                providers=["CPUExecutionProvider"])
    results, worst, argmax_ok = [], 0.0, True
    for t, h, n in shapes:
        feed = make_feed(t, h, n, classes=max_classes, task=task)
        t0 = time.time()
        (ort_out,) = sess.run(["logits"], feed)
        ort_ms = (time.time() - t0) * 1e3
        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(feed["x"]),
                             torch.from_numpy(feed["y"])).numpy()
        delta = float(np.abs(ort_out[:, n:] - pt_out[:, n:]).max())
        if task == "regression":
            aok = True  # scalar output; argmax not applicable
        else:
            aok = bool(
                (ort_out[:, n:].argmax(-1) == pt_out[:, n:].argmax(-1)).all())
        worst = max(worst, delta)
        argmax_ok = argmax_ok and aok
        results.append({"T": t, "H": h, "train": n,
                        "max_abs_delta_test_rows": delta,
                        "argmax_agree": aok, "ort_ms": ort_ms})
    return {"ok": worst < tol, "worst": worst, "tol": tol,
            "argmax_all_agree": argmax_ok, "shapes": results}


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
            raise RuntimeError(f"mapped initializer {init.name} is INLINE")
