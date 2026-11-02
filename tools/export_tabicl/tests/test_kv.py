"""The context-cache pair (kv.py): prepare once, query rows only, every row equal to the combined graph."""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
import torch

from export_tabicl import configs, export, kv
from export_tabicl.fixture import seeded_model

# Same budget the single-graph export is held to in test_export.py.
BUDGET = 1e-3


@pytest.fixture(scope="module", params=["classification", "regression"])
def exported(request, tmp_path_factory):
    task = request.param
    out = tmp_path_factory.mktemp(task)
    cfg = configs.fixture()
    model = seeded_model(task)
    prep, qry = kv.pair_paths(out, task)
    kv.export_kv(model, prep, qry, cfg)
    return task, model, prep, qry, cfg


def test_every_row_matches_the_combined_forward(exported):
    """Context rows AND query rows, at shapes other than the export example."""
    task, model, prep, qry, _ = exported
    worst = kv.check_against_torch(prep, qry, model, shapes=((24, 9, 7), (31, 5, 7), (18, 13, 11)))
    assert worst < BUDGET, f"{task}: {worst:.2e}"


def test_query_rows_are_independent_of_their_batch(exported):
    """query(a ++ b) == query(a) ++ query(b): what the engine's query_batch relies on."""
    task, model, prep, qry, _ = exported
    p, q = kv._ort_session(prep), kv._ort_session(qry)
    state = [i.name for i in q.get_inputs() if i.name != "x"]
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((1, 26, 7)).astype(np.float32)
    y = (rng.integers(0, 3, (1, 26)) if task == "classification" else rng.standard_normal((1, 26)) * 4 + 2)
    st = dict(zip(state, p.run(state, {"x": xs, "y": y.astype(np.float32)})))
    xq = rng.standard_normal((1, 12, 7)).astype(np.float32)
    whole = q.run(["logits"], {"x": xq, **st})[0]
    parts = np.concatenate([q.run(["logits"], {"x": xq[:, :5], **st})[0],
                            q.run(["logits"], {"x": xq[:, 5:], **st})[0]], axis=1)
    assert np.abs(whole - parts).max() < 1e-6


def test_contract_names(exported):
    """prepare(x, y) -> state + context_logits; query(x, state) -> logits, no y."""
    task, model, prep, qry, _ = exported
    p = onnx.load(str(prep), load_external_data=False).graph
    q = onnx.load(str(qry), load_external_data=False).graph
    hid, kvn, scaler = kv.state_names(model)
    assert [i.name for i in p.input] == ["x", "y"]
    outs = [o.name for o in p.output]
    assert outs[:len(hid + kvn + scaler) + 1] == hid + kvn + scaler + ["context_logits"]
    assert [i.name for i in q.input] == ["x"] + hid + kvn + scaler
    assert q.output[0].name == "logits"
    assert ("y_mean" in scaler) == (task == "regression")


def test_regression_returns_the_raw_point_estimate(exported):
    task, model, prep, qry, _ = exported
    if task != "regression":
        pytest.skip("regression only")
    p, q = kv._ort_session(prep), kv._ort_session(qry)
    state = [i.name for i in q.get_inputs() if i.name != "x"]
    rng = np.random.default_rng(1)
    xs = rng.standard_normal((1, 20, 5)).astype(np.float32)
    y = (rng.standard_normal((1, 20)) * 10 + 500).astype(np.float32)
    out = p.run(state + ["context_logits"], {"x": xs, "y": y})
    st = dict(zip(state, out[:-1]))
    assert out[-1].shape == (1, 20, 1)
    np.testing.assert_allclose(st["y_mean"], y.mean(keepdims=True), rtol=1e-6)
    np.testing.assert_allclose(st["y_std"], y.std(keepdims=True), rtol=1e-5)
    pred = q.run(["logits"], {"x": rng.standard_normal((1, 3, 5)).astype(np.float32), **st})[0]
    assert pred.shape == (1, 3, 1)
    # In raw target space (around 500), not z-scores.
    assert np.all(np.abs(pred - 500) < 100)


def test_ships_weight_free_and_maps_every_initializer_by_name(tmp_path):
    cfg = configs.fixture()
    model = seeded_model("classification")
    report = kv.export_pair(model, tmp_path, "classification", cfg, check=False,
                            safetensors_rel="model_classification.safetensors")
    for name in (report["prepare"], report["query"]):
        path = tmp_path / name
        tmap = json.loads(kv.map_path_for(path).read_text())
        export.assert_weight_free(path, tmap)
        assert not path.with_name(path.name + ".data").exists()
        assert all(export._norm_name(k) == v for k, v in tmap["initializers"].items())


def test_finish_refuses_a_map_built_by_value(tmp_path):
    """A wrapper holding the model under a second prefix gets its initializers mapped by value hash,
    and zero-initialised biases then collide (it mapped ISAB biases to row-interactor biases once)."""
    cfg = configs.fixture()
    model = seeded_model("classification")

    class Prefixed(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.prep = kv.PrepareKV(m)

        def forward(self, x, y):
            return self.prep(x, y)

    xs, _, y = kv._example(model, cfg)
    path = tmp_path / "graph_prepare_tabicl_classification.onnx"
    with torch.no_grad():
        torch.onnx.export(Prefixed(model).train(), (xs, y), str(path), dynamo=True, opset_version=configs.OPSET,
                          input_names=["x", "y"], external_data=True, optimize=False)
    with pytest.raises(RuntimeError, match="by value"):
        kv.finish(model, path, "classification", "model_classification.safetensors")
