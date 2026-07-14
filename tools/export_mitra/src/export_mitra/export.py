"""Mitra shipping pipeline: Tab2D -> weight-free ONNX graph + tensor-name map.

Mirrors tools/export_onnx (TabFM / S01) step-for-step:
  1. dynamo export, optimize=False (keep dotted-FQN initializer names so the
     name map survives), opset 18, torch.export.Dim for T (rows) and H
     (features). train_size and d are runtime int64 scalars, NOT dims.
     Signature: x[1,T,H] f32, y[1,T] f32, train_size[1] i64, d[1] i64
     -> logits[1,T,C].  cat_mask is OMITTED: Mitra has no categorical-specific
     embedding (categoricals are ordinal-encoded and treated numerically).
  2. strip doc_string + metadata_props (dynamo stack traces).
  3. force-externalize EVERY checkpoint-mapped initializer (threshold 0) so no
     real Mitra weight bytes ship inline; unmapped code constants stay inline.
  4. tensor map: ONNX initializer name -> safetensors key (identity here — the
     released safetensors are keyed by the bare Tab2D state_dict, which is
     exactly the module namespace this graph exports).
  5. parity: ORT vs patched-PyTorch fp32 on random weights at shapes DIFFERENT
     from the export example (test rows only). Runs before .data deletion.
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

from export_mitra.mitra_model_patched import Tab2D

OPSET = 18
PARITY_TOL = 1e-3


class ExportWrapper(torch.nn.Module):
    """Maps the engine contract (x, y, train_size, d) onto Mitra's 6-arg
    support/query forward, with a fixed batch of 1 (one table per call).

      x          [1, T, H] float32   preprocessed features (padded to H)
      y          [1, T]    float32   labels; test rows may hold any value
      train_size [1]       int64     number of context rows (prefix of T)
      d          [1]       int64     active feature count (<= H)
    Output:
      logits     [1, T, C]           C = dim_output (10 cls | 1 reg)

    The whole T-row table is fed as BOTH support and query; train_size splits
    them via masks (no data-dependent slicing), and d builds the feature
    padding mask. The engine reads predictions on the test rows (>= train_size).
    """

    def __init__(self, model: Tab2D):
        super().__init__()
        self.m = model

    def forward(self, x, y, train_size, d):
        b, t, h = x.shape
        ar_t = torch.arange(t, device=x.device)
        ar_h = torch.arange(h, device=x.device)
        padding_obs_support = ar_t[None, :] >= train_size[:, None]   # (1, T)
        padding_obs_query = torch.zeros((b, t), dtype=torch.bool, device=x.device)
        padding_features = ar_h[None, :] >= d[:, None]               # (1, H)
        return self.m(x, y, x, padding_features,
                      padding_obs_support, padding_obs_query)


def build_model(task: str, model_kwargs: dict, seed: int = 0,
                state_dict: dict | None = None) -> Tab2D:
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification|regression, got {task!r}")
    torch.manual_seed(seed)
    model = Tab2D(task=task.upper(), **model_kwargs)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model.eval()


def example_inputs(t, h, d_active, train, n_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g)
    y = torch.randint(0, n_classes, (1, t), generator=g).float()
    train_size = torch.tensor([train], dtype=torch.int64)
    d = torch.tensor([d_active], dtype=torch.int64)
    return x, y, train_size, d


def export_graph(model: Tab2D, graph_path: pathlib.Path, *,
                 dim_rows=("rows", 4, 100_000),
                 dim_features=("features", 2, 100),
                 example=(40, 8, 8, 24),
                 opset: int = OPSET) -> ExportWrapper:
    wrapper = ExportWrapper(model).eval()
    T = torch.export.Dim(dim_rows[0], min=dim_rows[1], max=dim_rows[2])
    H = torch.export.Dim(dim_features[0], min=dim_features[1], max=dim_features[2])
    dyn = (
        {1: T, 2: H},  # x
        {1: T},        # y
        None,          # train_size
        None,          # d
    )
    t, h, d_active, train = example
    n_classes = model.dim_output if model.task == "CLASSIFICATION" else 10
    ex = example_inputs(t, h, d_active, train, n_classes=n_classes)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, ex, str(graph_path),
            dynamo=True, dynamic_shapes=dyn, opset_version=opset,
            input_names=["x", "y", "train_size", "d"],
            output_names=["logits"],
            external_data=True, optimize=False,
        )
    return wrapper


def _norm_name(name: str) -> str:
    for pre in ("m.", "p_m_", "b_m_", "p_", "b_"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def build_tensor_map(model_proto: onnx.ModelProto, state_dict: dict) -> dict:
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
        raise RuntimeError(f"unmatched large (>=1KB) initializers: {unmatched_large}")
    return {"initializers": mapping, "transforms": transforms,
            "unmatched_small": unmatched_small}


def postprocess(graph_path: pathlib.Path, state_dict: dict) -> dict:
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


def make_feed(t, h, d, train, n_classes=10, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, t, h)).astype(np.float32)
    x[:, :, d:] = 0.0
    y = rng.integers(0, n_classes, (1, t)).astype(np.float32)
    return {"x": x, "y": y,
            "train_size": np.array([train], dtype=np.int64),
            "d": np.array([d], dtype=np.int64)}


def check_parity(graph_path: pathlib.Path, model: Tab2D, wrapper: ExportWrapper,
                 shapes, tol: float = PARITY_TOL) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    n_classes = model.dim_output if model.task == "CLASSIFICATION" else 10
    results, worst = [], 0.0
    for t, h, d, train in shapes:
        feed = make_feed(t, h, d, train, n_classes=n_classes)
        t0 = time.time()
        (ort_out,) = sess.run(["logits"], feed)
        ort_ms = (time.time() - t0) * 1e3
        with torch.no_grad():
            pt_out = wrapper(
                torch.from_numpy(feed["x"]), torch.from_numpy(feed["y"]),
                torch.from_numpy(feed["train_size"]), torch.from_numpy(feed["d"]),
            ).numpy()
        delta = float(np.abs(ort_out[:, train:] - pt_out[:, train:]).max())
        worst = max(worst, delta)
        results.append({"T": t, "H": h, "d": d, "train": train,
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
        if init.name in mapped and init.data_location != onnx.TensorProto.EXTERNAL:
            raise RuntimeError(f"mapped initializer {init.name} is INLINE (license wall)")
