"""Spike tests: patched TabICL exports, is weight-free, and round-trips via ORT."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import torch

from export_tabicl import configs, export
from export_tabicl.tabicl_patches import build_model


def _export(tmp_path, task):
    cfg = configs.fixture()
    model = build_model(task, cfg.model_kwargs, seed=0)
    graph = tmp_path / f"graph_{task}.onnx"
    wrapper = export.export_graph(model, graph, dim_rows=cfg.dim_rows,
                                  dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                  example=cfg.example)
    tmap = export.postprocess(graph, dict(model.state_dict()))
    return model, wrapper, graph, tmap


def test_classification_exports_and_parity(tmp_path):
    model, wrapper, graph, tmap = _export(tmp_path, "classification")
    parity = export.check_parity(graph, wrapper, ((40, 7, 30), (80, 15, 50)))
    assert parity["ok"], parity
    assert parity["worst"] < 1e-4


def test_regression_out_dim_is_quantiles(tmp_path):
    model, wrapper, graph, tmap = _export(tmp_path, "regression")
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    feed = export.make_feed(30, 6, 12, max_classes=0)
    (out,) = sess.run(["logits"], feed)
    assert out.shape == (1, 30, model.num_quantiles)  # quantile logits, not 1


def test_dynamic_H(tmp_path):
    # H genuinely dynamic: run at three feature counts (weights still present).
    model, wrapper, graph, tmap = _export(tmp_path, "classification")
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    for h in (3, 9, 20):
        (out,) = sess.run(["logits"], export.make_feed(24, h, 10, max_classes=3))
        assert out.shape == (1, 24, 3)


def test_weight_free_after_delete(tmp_path):
    model, wrapper, graph, tmap = _export(tmp_path, "classification")
    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tmap)  # raises if any mapped init is inline


def test_dynamic_train_size_no_baked_logn(tmp_path):
    """The ssmax log(n) must stay in-graph (Patch 5), not baked to the example."""
    model, wrapper, graph, tmap = _export(tmp_path, "classification")
    proto = onnx.load(str(graph), load_external_data=False)
    ops = [n.op_type for n in proto.graph.node]
    assert "Log" in ops, "ssmax log(n) was baked to a constant (Patch 5 missing)"


def test_tensor_map_covers_all_checkpoint_keys(tmp_path):
    model, wrapper, graph, tmap = _export(tmp_path, "classification")
    sd = set(model.state_dict().keys())
    mapped_targets = set(tmap["initializers"].values())
    assert mapped_targets == sd, sd ^ mapped_targets


def test_committed_fixture_roundtrips():
    """safetensors -> initializer injection -> ORT reproduces golden.json."""
    from safetensors.numpy import load_file
    base = pathlib.Path(__file__).resolve().parents[3] / "test" / "fixtures" / "tabicl"
    if not (base / "golden.json").exists():
        import pytest
        pytest.skip("committed fixture not present")
    tm = json.loads((base / "tensor_map_tabicl_fixture.json").read_text())
    golden = json.loads((base / "golden.json").read_text())
    tensors = load_file(str(base / "model.safetensors"))
    proto = onnx.load(str(base / "graph_tabicl_fixture.onnx"), load_external_data=False)
    for init in proto.graph.initializer:
        key = tm["initializers"].get(init.name)
        if key is None:
            continue
        arr = tensors[key]
        if tm["transforms"].get(init.name) == "transpose":
            arr = arr.T.copy()
        init.CopyFrom(onnx.numpy_helper.from_array(arr, name=init.name))
    sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    gi = golden["inputs"]
    (got,) = sess.run(["logits"], {"x": np.asarray(gi["x"], np.float32),
                                   "y": np.asarray(gi["y"], np.float32)})
    want = np.asarray(golden["logits"], np.float32)
    ts = gi["train_size"]
    rel = np.abs(got[:, ts:] - want[:, ts:]) / np.maximum(np.abs(want[:, ts:]), 1e-12)
    assert rel.max() < golden["_doc"]["rtol"]
