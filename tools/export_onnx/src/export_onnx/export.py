"""S01 shipping pipeline: TabFM -> weight-free ONNX graph + tensor-name map.

Pipeline (every step is a run finding from S01/S06 RESULTS.md):
  1. dynamo export, optimize=False (keeps dotted-FQN initializer names —
     the post-export optimizer would constant-fold Linear weights into
     anonymous transposed `val_NN` initializers and kill the name map),
     opset 18, torch.export.Dim for T and H, cat_mask/d pinned as required
     inputs. Signature: x[1,T,H] f32, y[1,T] f32, train_size[1] i64,
     cat_mask[1,H] bool, d[1] i64 -> logits[1,T,C].
  2. strip doc_string + metadata_props (~92% of raw graph bytes are dynamo
     stack traces).
  3. force-externalize EVERY checkpoint-mapped initializer (threshold 0 —
     torch's own external_data saver keeps small tensors inline, which for
     the real model would ship Google weight bytes inside the "weight-free"
     graph). Non-checkpoint lifted constants (e.g. the 8-byte
     `lifted_tensor_0` from OneHotAndLinear) stay INLINE on purpose: they
     come from Apache-2.0 code, not the checkpoint, and the runtime cannot
     inject them from safetensors.
  4. tensor map: ONNX initializer name -> CHECKPOINT-namespace safetensors
     key (bare model state_dict; the wrapper's 'm.' prefix is stripped).
     Match by normalized name first, then (shape, sha1) hash, then
     transposed hash (2-D) recorded as transform "transpose".
  5. parity: ORT vs PyTorch fp32 on random weights at shapes DIFFERENT from
     the export example (budget 1e-3; measured ~5e-8). Runs before .data
     deletion.
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

from export_onnx.tabfm_model_patched import TabFM

OPSET = 18
PARITY_TOL = 1e-3  # DoD budget; expect ~5e-8 (S01)


class ExportWrapper(torch.nn.Module):
  """Pins TabFM's optional args so the exported graph has a fixed signature.

  Inputs (B fixed to 1 — one table per call, matching the sklearn wrapper):
    x          [1, T, H] float32   preprocessed features (padded to H)
    y          [1, T]    float32   labels; test rows may hold any value
    train_size [1]       int64     number of context rows (prefix of T)
    cat_mask   [1, H]    bool      per-column categorical flag
    d          [1]       int64     active feature count (<= H)
  Output:
    logits     [1, T, C]           C = max_classes | 1 (regression)
  """

  def __init__(self, model: TabFM):
    super().__init__()
    self.m = model

  def forward(self, x, y, train_size, cat_mask, d):
    return self.m(x, y, train_size, cat_mask=cat_mask, d=d)


def build_model(task: str, model_kwargs: dict, seed: int = 0) -> TabFM:
  """Random-weight TabFM at the given dims. No checkpoint bytes anywhere."""
  if task not in ("classification", "regression"):
    raise ValueError(f"task must be classification|regression, got {task!r}")
  torch.manual_seed(seed)
  model = TabFM(is_classifier=(task == "classification"), **model_kwargs)
  return model.eval()


def example_inputs(t, h, d_active, train, max_classes=3, seed=0):
  torch.manual_seed(seed)
  x = torch.randn(1, t, h)
  y = torch.randint(0, max_classes, (1, t)).float()
  train_size = torch.tensor([train], dtype=torch.int64)
  cat_mask = torch.zeros(1, h, dtype=torch.bool)
  cat_mask[0, : min(3, d_active)] = True
  d = torch.tensor([d_active], dtype=torch.int64)
  return x, y, train_size, cat_mask, d


def export_graph(model: TabFM, graph_path: pathlib.Path, *,
                 dim_rows=("rows", 4, 100_000),
                 dim_features=("features", 2, 512),
                 example=(64, 12, 10, 48),
                 opset: int = OPSET) -> ExportWrapper:
  """Step 1: dynamo export with dynamic T/H. Writes graph + .onnx.data."""
  wrapper = ExportWrapper(model).eval()
  T = torch.export.Dim(dim_rows[0], min=dim_rows[1], max=dim_rows[2])
  H = torch.export.Dim(dim_features[0], min=dim_features[1],
                       max=dim_features[2])
  dyn = (
      {1: T, 2: H},  # x
      {1: T},        # y
      None,          # train_size (runtime value, not a dim)
      {1: H},        # cat_mask
      None,          # d (runtime value)
  )
  t, h, d_active, train = example
  ex = example_inputs(t, h, d_active, train, max_classes=model.max_classes)
  graph_path.parent.mkdir(parents=True, exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
        wrapper,
        ex,
        str(graph_path),
        dynamo=True,
        dynamic_shapes=dyn,
        opset_version=opset,
        input_names=["x", "y", "train_size", "cat_mask", "d"],
        output_names=["logits"],
        external_data=True,   # weights -> .onnx.data (local only, never ship)
        optimize=False,       # keep FQN initializer names (see module doc)
    )
  return wrapper


def _norm_name(name: str) -> str:
  # dynamo (torch 2.12) emits dotted FQNs with the wrapper's attribute as
  # prefix ('m.cell_embedder...'); strip it to get the exact checkpoint key.
  # p_m_/b_m_ manglings kept as fallback for older exporters.
  for pre in ("m.", "p_m_", "b_m_", "p_", "b_"):
    if name.startswith(pre):
      return name[len(pre):]
  return name


def build_tensor_map(model_proto: onnx.ModelProto, state_dict: dict) -> dict:
  """ONNX initializer name -> checkpoint-namespace safetensors key.

  `state_dict` MUST be the BARE model's state_dict (checkpoint namespace,
  e.g. 'cls_tokens'), not the wrapper's ('m.cls_tokens') — the released
  safetensors are keyed by the bare model (S06 friction #2).

  Returns {"initializers": {onnx_name: st_key},
           "transforms": {onnx_name: "transpose"},   # only hash-fallback hits
           "unmatched_small": [names]}               # inline code constants
  """
  sd_by_name = {}
  for k in state_dict:
    sd_by_name[k] = k
  sd_by_sig = {}
  sd_by_sig_t = {}
  for k, v in state_dict.items():
    arr = v.detach().cpu().numpy()
    sd_by_sig[(tuple(arr.shape),
               hashlib.sha1(arr.tobytes()).hexdigest())] = k
    if arr.ndim == 2:
      at = np.ascontiguousarray(arr.T)
      sd_by_sig_t[(tuple(at.shape),
                   hashlib.sha1(at.tobytes()).hexdigest())] = k

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
        arr_bytes = arr.nbytes
        (unmatched_large if arr_bytes >= 1024 else unmatched_small).append(
            init.name)
        continue
    mapping[init.name] = key
  if unmatched_large:
    raise RuntimeError(
        f"unmatched large (>=1KB) initializers: {unmatched_large} — the "
        "tensor map must cover 100% of them (S01 DoD)")
  return {"initializers": mapping, "transforms": transforms,
          "unmatched_small": unmatched_small}


def postprocess(graph_path: pathlib.Path, state_dict: dict) -> dict:
  """Steps 2-4: strip metadata, force-externalize checkpoint initializers
  (threshold 0), rewrite graph + .data, build the tensor map.

  Returns the tensor-map dict from build_tensor_map().
  """
  model_proto = onnx.load(str(graph_path), load_external_data=True)

  # strip trace metadata (8.4 MB -> ~0.5 MB on tiny; no behavior change).
  # dynamo attaches metadata_props at BOTH the model and the per-node level
  # (stack traces, pkg.torch.*) — strip both, plus all doc_strings.
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

  # Force-externalize every checkpoint-mapped initializer (threshold 0):
  # torch's saver keeps small tensors inline, which would embed weight bytes
  # in the shipping graph. Unmapped code constants stay inline.
  data_name = graph_path.name + ".data"
  data_path = graph_path.with_name(data_name)
  mapped = set(tensor_map["initializers"])
  for init in model_proto.graph.initializer:
    if init.name in mapped:
      onnx.external_data_helper.set_external_data(init, location=data_name)
  data_path.unlink(missing_ok=True)  # onnx appends; never reuse torch's file
  onnx.save(model_proto, str(graph_path))
  return tensor_map


def write_tensor_map(map_path: pathlib.Path, tensor_map: dict, *, task: str,
                     safetensors_rel: str, opset: int = OPSET) -> None:
  payload = {
      "task": task,
      "opset": opset,
      "safetensors": safetensors_rel,
      # ONNX initializer name -> checkpoint-namespace safetensors key
      "initializers": tensor_map["initializers"],
      # onnx_name -> "transpose" where the transposed-hash fallback fired
      # (empty with optimize=False; kept for forward compatibility)
      "transforms": tensor_map["transforms"],
  }
  map_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_feed(t, h, d, train, classes=3, seed=1):
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((1, t, h)).astype(np.float32)
  x[:, :, d:] = 0.0  # padded columns
  y = rng.integers(0, classes, (1, t)).astype(np.float32)
  cat = np.zeros((1, h), dtype=bool)
  cat[0, : min(3, d)] = True
  return {
      "x": x, "y": y,
      "train_size": np.array([train], dtype=np.int64),
      "cat_mask": cat,
      "d": np.array([d], dtype=np.int64),
  }


def check_parity(graph_path: pathlib.Path, model: TabFM,
                 shapes, tol: float = PARITY_TOL) -> dict:
  """ORT vs PyTorch fp32 on random weights. Needs .onnx.data present.

  Compares TEST rows only (rows >= train_size) — the runtime reads
  predictions only there. `shapes` = ((T, H, d, train), ...), all different
  from the export example so the dynamic dims are genuinely exercised.
  """
  import onnxruntime as ort  # local import: keep export usable without ORT

  sess = ort.InferenceSession(str(graph_path),
                              providers=["CPUExecutionProvider"])
  results, worst = [], 0.0
  for t, h, d, train in shapes:
    feed = make_feed(t, h, d, train, classes=model.max_classes)
    t0 = time.time()
    (ort_out,) = sess.run(["logits"], feed)
    ort_ms = (time.time() - t0) * 1e3
    with torch.no_grad():
      pt_out = model(
          torch.from_numpy(feed["x"]),
          torch.from_numpy(feed["y"]),
          torch.from_numpy(feed["train_size"]),
          cat_mask=torch.from_numpy(feed["cat_mask"]),
          d=torch.from_numpy(feed["d"]),
      ).numpy()
    delta = float(np.abs(ort_out[:, train:] - pt_out[:, train:]).max())
    worst = max(worst, delta)
    results.append({"T": t, "H": h, "d": d, "train": train,
                    "max_abs_delta_test_rows": delta, "ort_ms": ort_ms})
  ok = worst < tol
  return {"ok": ok, "worst": worst, "tol": tol, "shapes": results}


def delete_weight_data(graph_path: pathlib.Path) -> None:
  """Step 6: the shipping artifact is weight-free."""
  graph_path.with_name(graph_path.name + ".data").unlink(missing_ok=True)


def assert_weight_free(graph_path: pathlib.Path, tensor_map: dict) -> None:
  """Every mapped initializer must be EXTERNAL with no .data file present."""
  data_path = graph_path.with_name(graph_path.name + ".data")
  if data_path.exists():
    raise RuntimeError(f"{data_path} still exists — weight bytes on disk")
  proto = onnx.load(str(graph_path), load_external_data=False)
  mapped = set(tensor_map["initializers"])
  for init in proto.graph.initializer:
    external = (init.data_location == onnx.TensorProto.EXTERNAL)
    if init.name in mapped and not external:
      raise RuntimeError(f"mapped initializer {init.name} is INLINE — "
                         "weight bytes inside the graph (license wall)")
