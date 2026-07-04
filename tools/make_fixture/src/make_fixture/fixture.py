"""S06: CI fixture model — random weights, tiny dims, deterministic bytes.

The fixture is a TabFM() at its unit-test-sized defaults (embed_dim=8,
max_classes=3, ~17k params) with SEEDED RANDOM weights: architecture is
Apache-2.0, the weights are ours — zero Google bytes (BRD FR-2.2).

S06 RESULTS.md fixes baked in:
  * weights re-drawn from a DEDICATED torch.Generator (seed 1337) over
    SORTED state_dict keys — independent of nn.Module default-init order
    across torch versions (only RNG-stream-stable per torch version; the
    committed artifacts + pinned hashes are the source of truth, CI never
    regenerates).
  * safetensors with EXACTLY ONE __metadata__ key: safetensors-rust
    serializes >=2 __metadata__ keys in random HashMap order ->
    nondeterministic sha256 -> flaky CI hash check.
  * tensor map built against the BARE model state_dict (checkpoint
    namespace), not the wrapper's ('m.'-prefixed) one.
  * golden.json documents that runtime parity compares rows >= train_size.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import torch
from safetensors.torch import save_file

from export_onnx import configs as x_configs
from export_onnx import export as x_export
from export_onnx.tabfm_model_patched import TabFM

SEED_WEIGHTS = 1337
SEED_GOLDEN_INPUTS = 7
WEIGHT_SCALE = 0.05
GOLDEN_SHAPE = dict(t=12, h=5, d=5, train=8)  # S06 golden inputs
PARITY_RTOL = 1e-4  # C++ runtime budget (S06 measured max rel 4.0e-7)

# TabFM.__init__ defaults, spelled out (S06 FIXTURE_CFG).
FIXTURE_CFG = dict(
    embed_dim=8, max_classes=3, col_num_blocks=2, col_nhead=2, col_num_inds=4,
    row_num_blocks=2, row_nhead=2, row_num_cls=2, icl_num_blocks=2,
    icl_nhead=2, ff_factor=2, feature_group_size=3, num_freq=32,
    is_classifier=True,
)

FILES = ["graph_fixture.onnx", "model.safetensors",
         "tensor_map_fixture.json", "golden.json", "manifest.json"]


def seeded_model() -> TabFM:
  model = TabFM(**FIXTURE_CFG)
  gen = torch.Generator().manual_seed(SEED_WEIGHTS)
  with torch.no_grad():
    sd = model.state_dict()
    for key in sorted(sd):  # sorted: init-order independent
      p = sd[key]
      p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype)
              * WEIGHT_SCALE)
  return model.eval()


def safetensors_bytes_digest(model: TabFM, st_path: pathlib.Path) -> str:
  # single metadata key ONLY (see module docstring / S06 friction #1)
  save_file(
      model.state_dict(), str(st_path),
      metadata={"origin": "anofox-tabfm CI fixture, random init, Apache-2.0 "
                          "architecture, license fixture-mit (our weights, "
                          "not Google's)"})
  return hashlib.sha256(st_path.read_bytes()).hexdigest()


def golden_inputs():
  gen = torch.Generator().manual_seed(SEED_GOLDEN_INPUTS)
  t, h, d, train = (GOLDEN_SHAPE[k] for k in ("t", "h", "d", "train"))
  x = torch.randn(1, t, h, generator=gen)
  y = torch.randint(0, FIXTURE_CFG["max_classes"], (1, t), generator=gen
                    ).float()
  ts = torch.tensor([train], dtype=torch.int64)
  cm = torch.zeros(1, h, dtype=torch.bool)
  cm[0, :2] = True
  dd = torch.tensor([d], dtype=torch.int64)
  return x, y, ts, cm, dd


def sha256_file(p: pathlib.Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out: pathlib.Path) -> dict:
  """Generate all fixture artifacts into `out`. Returns {file: sha256}."""
  out.mkdir(parents=True, exist_ok=True)

  # 1. deterministic seeded model; in-process determinism assertion
  model = seeded_model()
  model2 = seeded_model()
  sd, sd2 = model.state_dict(), model2.state_dict()
  assert sorted(sd) == sorted(sd2)
  for k in sd:
    assert torch.equal(sd[k], sd2[k]), f"nondeterministic weight {k}"

  # 2. safetensors — byte-determinism asserted by double save
  st_path = out / "model.safetensors"
  digest = safetensors_bytes_digest(model, st_path)
  tmp = out / "model.safetensors.recheck"
  digest2 = safetensors_bytes_digest(model2, tmp)
  tmp.unlink()
  assert digest == digest2, (
      f"safetensors bytes not deterministic: {digest} != {digest2}")

  # 3. graph + tensor map through the SAME exporter code path as the real
  # model (S01) — the exporter itself is exercised by CI.
  cfg = x_configs.fixture()
  graph_path = out / "graph_fixture.onnx"
  x_export.export_graph(model, graph_path, dim_rows=cfg.dim_rows,
                        dim_features=cfg.dim_features, example=cfg.example)
  tensor_map = x_export.postprocess(graph_path, dict(model.state_dict()))
  x_export.write_tensor_map(out / "tensor_map_fixture.json", tensor_map,
                            task="classification",
                            safetensors_rel="model.safetensors")

  # 4. sanity parity at a shape different from the export example (S06:
  # T=40,H=7,d=6,train=30), while .data is still present
  parity = x_export.check_parity(graph_path, model, cfg.parity_shapes)
  assert parity["ok"], parity

  # 5. fixture ships weight-free too
  x_export.delete_weight_data(graph_path)
  x_export.assert_weight_free(graph_path, tensor_map)

  # 6. golden.json — inputs + PyTorch fp32 logits for the C++ parity test
  x, y, ts, cm, dd = golden_inputs()
  with torch.no_grad():
    logits = model(x, y, ts, cat_mask=cm, d=dd)
  (out / "golden.json").write_text(json.dumps({
      "_doc": {
          "purpose": "C++ parity: safetensors -> initializer injection -> "
                     "ORT run on graph_fixture.onnx must reproduce these "
                     "fp32 logits.",
          "parity_slice": "the runtime reads predictions only for rows >= "
                          "train_size; parity is asserted on logits[:, "
                          "train_size:, :] (all-row comparison also holds "
                          "for this fixture but is not the contract)",
          "rtol": PARITY_RTOL,
          "y_convention": "y rows >= train_size are ignored by the model; "
                          "here they hold valid class ids, the runtime pads "
                          "with -100.0",
      },
      "inputs": {
          "x": x.tolist(),
          "y": y.tolist(),
          "train_size": GOLDEN_SHAPE["train"],
          "cat_mask": cm.tolist(),
          "d": GOLDEN_SHAPE["d"],
      },
      "logits": logits.tolist(),
      "safetensors_sha256": digest,
  }, indent=2) + "\n")

  # 7. manifest.json per HLD D7: repo, revision, files+sizes, graph id,
  # tensor-name map file, preprocessing profile id, license id, engine
  # profile (dtype per EP).
  def file_entry(name):
    p = out / name
    return {"path": name, "bytes": p.stat().st_size,
            "sha256": sha256_file(p)}

  manifest = {
      "manifest_version": 1,
      "model_id": "fixture",
      "task": "classification",
      "license": "fixture-mit",
      "repo": "local:test/fixtures",
      "revision": "fixture-v1",
      "graph": "graph_fixture.onnx",
      "graph_id": "tabfm-v1-fixture-opset18",
      "tensor_map": "tensor_map_fixture.json",
      "preprocessing_profile": "tabfm_v1_minimal",
      "engine_profiles": {"cpu": {"dtype": "f32"}},
      "files": [file_entry("model.safetensors"),
                file_entry("graph_fixture.onnx"),
                file_entry("tensor_map_fixture.json")],
  }
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

  hashes = {name: sha256_file(out / name) for name in FILES}
  total = sum((out / name).stat().st_size for name in FILES)
  assert total < 5 * 1024 * 1024, f"fixture total {total} B >= 5 MB budget"
  return hashes


def roundtrip_check(out: pathlib.Path) -> float:
  """Python proxy for the C++ path: safetensors -> inject -> ORT -> golden.

  Returns max relative delta vs golden logits on rows >= train_size.
  """
  import numpy as np
  import onnx
  import onnxruntime as ort
  from safetensors.numpy import load_file

  tm = json.loads((out / "tensor_map_fixture.json").read_text())
  golden = json.loads((out / "golden.json").read_text())
  tensors = load_file(str(out / "model.safetensors"))

  proto = onnx.load(str(out / "graph_fixture.onnx"),
                    load_external_data=False)
  injected = 0
  for init in proto.graph.initializer:
    key = tm["initializers"].get(init.name)
    if key is None:
      continue
    arr = tensors[key]
    if tm["transforms"].get(init.name) == "transpose":
      arr = arr.T.copy()
    init.CopyFrom(onnx.numpy_helper.from_array(arr, name=init.name))
    injected += 1
  assert injected == len(tm["initializers"])

  sess = ort.InferenceSession(proto.SerializeToString(),
                              providers=["CPUExecutionProvider"])
  gi = golden["inputs"]
  feed = {
      "x": np.asarray(gi["x"], dtype=np.float32),
      "y": np.asarray(gi["y"], dtype=np.float32),
      "train_size": np.asarray([gi["train_size"]], dtype=np.int64),
      "cat_mask": np.asarray(gi["cat_mask"], dtype=bool),
      "d": np.asarray([gi["d"]], dtype=np.int64),
  }
  (got,) = sess.run(["logits"], feed)
  want = np.asarray(golden["logits"], dtype=np.float32)
  ts = gi["train_size"]
  g, w = got[:, ts:], want[:, ts:]
  rel = np.abs(g - w) / np.maximum(np.abs(w), 1e-12)
  return float(rel.max())
