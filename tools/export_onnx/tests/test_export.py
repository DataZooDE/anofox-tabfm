"""S01 Definition-of-Done, encoded as pytest asserts (run on `tiny`).

- export writes graph + map; graph < 20 MB and weight-free
- parity max |dLogit| < 1e-3 fp32 at (16,4)/(300,25)/(1200,80) incl. d < H,
  all shapes DIFFERENT from the export example (T=64, H=12)
- same for regression
- tensor_map covers 100% of initializers >= 1 KB, values in checkpoint
  namespace (no wrapper 'm.' prefix)
- weight-free graph fails ORT session creation cleanly (negative test)
"""

import json
import pathlib

import onnx
import pytest

from export_onnx import configs, export

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="session", params=["classification", "regression"])
def exported(request, tmp_path_factory):
  task = request.param
  cfg = configs.tiny()
  out = tmp_path_factory.mktemp(f"export_{task}")
  graph_path = out / f"graph_{task}.onnx"
  model = export.build_model(task, cfg.model_kwargs, seed=0)
  export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                      dim_features=cfg.dim_features, example=cfg.example)
  tensor_map = export.postprocess(graph_path, dict(model.state_dict()))
  map_path = out / f"tensor_map_{task}.json"
  export.write_tensor_map(map_path, tensor_map, task=task,
                          safetensors_rel=f"{task}/model.safetensors")
  parity = export.check_parity(graph_path, model, cfg.parity_shapes)
  export.delete_weight_data(graph_path)
  return dict(task=task, cfg=cfg, model=model, graph_path=graph_path,
              map_path=map_path, tensor_map=tensor_map, parity=parity)


def test_graph_written_and_small(exported):
  p = exported["graph_path"]
  assert p.exists()
  assert p.stat().st_size < 20 * 1024 * 1024  # DoD: < 20 MB
  # S01 measured ~0.5 MB after metadata strip; 2 MB catches regressions.
  assert p.stat().st_size < 2 * 1024 * 1024


def test_graph_is_weight_free(exported):
  export.assert_weight_free(exported["graph_path"], exported["tensor_map"])
  # unmatched inline constants must all be tiny code-derived tensors
  proto = onnx.load(str(exported["graph_path"]), load_external_data=False)
  mapped = set(exported["tensor_map"]["initializers"])
  for init in proto.graph.initializer:
    if init.name not in mapped:
      arr = onnx.numpy_helper.to_array(init)
      assert arr.nbytes < 1024, f"large unmapped initializer {init.name}"


def test_metadata_stripped(exported):
  proto = onnx.load(str(exported["graph_path"]), load_external_data=False)
  assert len(proto.metadata_props) == 0
  assert all(n.doc_string == "" for n in proto.graph.node)
  assert all(len(n.metadata_props) == 0 for n in proto.graph.node)


def test_opset_18_ai_onnx_only(exported):
  proto = onnx.load(str(exported["graph_path"]), load_external_data=False)
  domains = {o.domain: o.version for o in proto.opset_import}
  assert domains.get("") == 18 or domains.get("ai.onnx") == 18
  assert all(d in ("", "ai.onnx") for d in domains), domains


def test_signature(exported):
  proto = onnx.load(str(exported["graph_path"]), load_external_data=False)
  ins = {i.name: i for i in proto.graph.input}
  assert list(ins) == ["x", "y", "train_size", "cat_mask", "d"]
  elem = {n: i.type.tensor_type.elem_type for n, i in ins.items()}
  assert elem["x"] == onnx.TensorProto.FLOAT
  assert elem["y"] == onnx.TensorProto.FLOAT
  assert elem["train_size"] == onnx.TensorProto.INT64
  assert elem["cat_mask"] == onnx.TensorProto.BOOL
  assert elem["d"] == onnx.TensorProto.INT64

  def dims(name):
    return [(dd.dim_param or dd.dim_value)
            for dd in ins[name].type.tensor_type.shape.dim]

  x_dims = dims("x")
  assert x_dims[0] == 1 and isinstance(x_dims[1], str) \
      and isinstance(x_dims[2], str)  # [1, T(sym), H(sym)]
  assert dims("y")[0] == 1 and isinstance(dims("y")[1], str)
  outs = [o.name for o in proto.graph.output]
  assert outs == ["logits"]


def test_tensor_map_full_coverage_and_namespace(exported):
  tm = json.loads(exported["map_path"].read_text())
  sd = dict(exported["model"].state_dict())
  proto = onnx.load(str(exported["graph_path"]), load_external_data=False)
  mapped = tm["initializers"]
  for name, key in mapped.items():
    assert not key.startswith("m."), f"wrapper-namespace key leaked: {key}"
    assert key in sd, f"map value {key} not a checkpoint key"
  # 100% coverage of initializers >= 1KB
  for init in proto.graph.initializer:
    arr_elems = 1
    for d in init.dims:
      arr_elems *= d
    if arr_elems * 4 >= 1024:
      assert init.name in mapped, f"unmapped >=1KB initializer {init.name}"
  # transforms empty with optimize=False (name matching maps everything)
  assert tm["transforms"] == {}


def test_parity_within_budget_at_non_export_shapes(exported):
  parity = exported["parity"]
  assert parity["ok"], parity
  assert parity["worst"] < 1e-3
  t_ex, h_ex, _, _ = exported["cfg"].example
  for r in parity["shapes"]:
    assert (r["T"], r["H"]) != (t_ex, h_ex), (
        "parity shape equals the export example — dynamic dims not proven")


def test_weight_free_graph_fails_ort_cleanly(exported):
  import onnxruntime as ort
  with pytest.raises(Exception) as ei:
    ort.InferenceSession(str(exported["graph_path"]),
                         providers=["CPUExecutionProvider"])
  msg = str(ei.value)
  assert ("External data" in msg or "NO_SUCHFILE" in msg
          or "does not exist" in msg), msg


def test_real_config_fetch_shapes_only():
  """`real` config carries architecture dims only — never weights."""
  scratch = pathlib.Path(__file__).parent / "data"
  local = scratch / "config_classification.json"
  if local.exists():
    cfg = configs.real("classification", str(local))
  else:
    try:
      cfg = configs.real("classification")
    except Exception as e:  # offline CI: skip, resources/ pins provenance
      pytest.skip(f"HF unreachable: {e}")
  assert cfg.model_kwargs["embed_dim"] >= 8
  assert cfg.model_kwargs["max_classes"] == 10
  assert cfg.config_source and cfg.config_source["sha256"]
