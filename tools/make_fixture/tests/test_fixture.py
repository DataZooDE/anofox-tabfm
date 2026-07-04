"""S06 Definition-of-Done as pytest asserts.

- make_fixture is deterministic (same safetensors bytes on re-run)
- artifacts total < 5 MB
- safetensors has EXACTLY ONE __metadata__ key (>=2 keys serialize in
  random HashMap order -> nondeterministic sha256, S06 friction #1)
- tensor map is keyed against the BARE model state_dict
- golden roundtrip: safetensors -> initializer injection -> ORT reproduces
  golden.json logits within rtol 1e-4 (Python proxy for the C++ S02 path)
- manifest.json carries the HLD D7 fields
"""

import json
import struct

import pytest
import torch

from make_fixture import fixture


@pytest.fixture(scope="session")
def built(tmp_path_factory):
  out = tmp_path_factory.mktemp("fixture_out")
  hashes = fixture.build(out)
  return out, hashes


def test_weights_deterministic_and_sorted_generator():
  m1, m2 = fixture.seeded_model(), fixture.seeded_model()
  for k, v in m1.state_dict().items():
    assert torch.equal(v, m2.state_dict()[k])


def test_total_size_budget(built):
  out, _ = built
  total = sum((out / n).stat().st_size for n in fixture.FILES)
  assert total < 5 * 1024 * 1024


def test_safetensors_single_metadata_key(built):
  out, _ = built
  raw = (out / "model.safetensors").read_bytes()
  (hlen,) = struct.unpack("<Q", raw[:8])
  header = json.loads(raw[8: 8 + hlen])
  assert "__metadata__" in header
  assert len(header["__metadata__"]) == 1, header["__metadata__"]


def test_tensor_map_bare_namespace(built):
  out, _ = built
  tm = json.loads((out / "tensor_map_fixture.json").read_text())
  sd = fixture.seeded_model().state_dict()
  assert tm["initializers"], "empty tensor map"
  for onnx_name, key in tm["initializers"].items():
    assert key in sd, f"{key} not a bare state_dict key"
    assert not key.startswith("m.")
  # every checkpoint tensor is mapped (fixture graph keeps them all)
  assert set(tm["initializers"].values()) == set(sd.keys())


def test_graph_weight_free_negative(built):
  out, _ = built
  import onnxruntime as ort
  with pytest.raises(Exception) as ei:
    ort.InferenceSession(str(out / "graph_fixture.onnx"),
                         providers=["CPUExecutionProvider"])
  msg = str(ei.value)
  assert ("External data" in msg or "NO_SUCHFILE" in msg
          or "does not exist" in msg)


def test_golden_roundtrip_within_budget(built):
  out, _ = built
  rel = fixture.roundtrip_check(out)
  assert rel < fixture.PARITY_RTOL, rel


def test_golden_documents_parity_slice(built):
  out, _ = built
  golden = json.loads((out / "golden.json").read_text())
  assert golden["inputs"]["train_size"] == 8
  assert golden["inputs"]["d"] == 5
  x = golden["inputs"]["x"]
  assert len(x[0]) == 12 and len(x[0][0]) == 5  # T=12, H=5
  assert "train_size" in golden["_doc"]["parity_slice"]
  assert golden["safetensors_sha256"]
  logits = golden["logits"]
  assert len(logits[0]) == 12 and len(logits[0][0]) == 3  # C = max_classes


def test_manifest_hld_d7_fields(built):
  out, _ = built
  m = json.loads((out / "manifest.json").read_text())
  assert m["license"] == "fixture-mit"
  assert m["repo"] and m["revision"]
  assert m["graph"] == "graph_fixture.onnx" and m["graph_id"]
  assert m["tensor_map"] == "tensor_map_fixture.json"
  assert m["preprocessing_profile"] == "tabfm_v1_minimal"
  assert m["engine_profiles"]["cpu"]["dtype"] == "f32"
  by_path = {f["path"]: f for f in m["files"]}
  for name in ("model.safetensors", "graph_fixture.onnx",
               "tensor_map_fixture.json"):
    assert name in by_path
    assert by_path[name]["bytes"] == (out / name).stat().st_size
    assert len(by_path[name]["sha256"]) == 64


def test_committed_fixture_matches_pin():
  """The committed test/fixtures artifacts must match FIXTURE_SHA256."""
  import hashlib
  import pathlib
  fixtures = pathlib.Path(__file__).resolve().parents[3] / "test/fixtures"
  pin_file = fixtures / "FIXTURE_SHA256"
  if not pin_file.exists():
    pytest.skip("fixture not generated into test/fixtures yet")
  for line in pin_file.read_text().splitlines():
    sha, name = line.split()
    p = fixtures / name
    assert p.exists(), f"pinned file missing: {name}"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, name
