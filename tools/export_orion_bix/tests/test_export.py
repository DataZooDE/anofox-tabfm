"""Tests: patched Orion-BiX exports, is weight-free, and round-trips via ORT."""

from __future__ import annotations

import hashlib
import json
import pathlib

import onnxruntime as ort
import pytest

from export_orion_bix import configs, export, fixture
from export_orion_bix.orion_bix_patches import build_model

REPO = pathlib.Path(__file__).resolve().parents[3]  # tests/ -> tool -> tools/ -> repo
COMMITTED_FIXTURE = REPO / "test/fixtures/orion_bix"
RESOURCES = REPO / "resources"


def _export(tmp_path):
    cfg = configs.fixture()
    model = build_model(cfg.model_kwargs, seed=0)
    graph = tmp_path / "graph.onnx"
    wrapper = export.export_graph(model, graph, dim_rows=cfg.dim_rows,
                                  dim_train=cfg.dim_train, dim_features=cfg.dim_features,
                                  example=cfg.example)
    tmap = export.postprocess(graph, dict(model.state_dict()))
    return model, wrapper, graph, tmap


def test_exports_and_parity(tmp_path):
    _model, wrapper, graph, _tmap = _export(tmp_path)
    # Shapes all differ from the export example (20, 5, 12), so T/H/S dynamism
    # is genuinely exercised rather than replayed.
    parity = export.check_parity(graph, wrapper, ((40, 7, 30), (80, 15, 50)))
    assert parity["ok"], parity
    assert parity["worst"] < 1e-4


def test_dynamic_H_and_train_size(tmp_path):
    _model, _wrapper, graph, _tmap = _export(tmp_path)
    sess = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    for h in (3, 9, 20):
        for s in (5, 10):
            (out,) = sess.run(["logits"], export.make_feed(24, h, s, max_classes=3))
            assert out.shape == (1, 24, 3)


def test_weight_free_after_delete(tmp_path):
    _model, _wrapper, graph, tmap = _export(tmp_path)
    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tmap)  # raises if any mapped init is inline


def test_tensor_map_uses_checkpoint_key_namespace(tmp_path):
    """The released ckpt is torch.compile-wrapped: every key has `_orig_mod.`.

    The extension injects by name against the checkpoint AS SHIPPED, so a map
    written in the bare namespace would bind nothing at runtime.
    """
    _model, _wrapper, _graph, tmap = _export(tmp_path)
    assert tmap["initializers"]
    assert all(v.startswith("_orig_mod.") for v in tmap["initializers"].values())


def test_non_standard_attention_is_rejected():
    """Only the released standard-attention path is patched; guard the rest."""
    for key in ("col_attention_type", "row_attention_type", "icl_attention_type"):
        kwargs = dict(configs._REAL_KWARGS)
        kwargs[key] = "linear"
        with pytest.raises(ValueError, match="only 'standard' is on the exported path"):
            configs.assert_shipped_path(kwargs)


# --- license wall -------------------------------------------------------------

def test_shipped_graph_is_weight_free():
    """No Lexsi weight bytes in the committed shipping artifacts."""
    graph = RESOURCES / "graph_orion_bix_classification.onnx"
    tmap = json.loads((RESOURCES / "tensor_map_orion_bix_classification.json").read_text())
    assert graph.exists()
    assert not graph.with_name(graph.name + ".data").exists()
    export.assert_weight_free(graph, tmap)
    # 27M fp32 params would be ~108 MB; the weight-free graph is a fraction of that.
    assert graph.stat().st_size < 5 * 1024 * 1024


def test_committed_fixture_matches_pinned_hashes():
    """The committed fixture must match its FIXTURE_SHA256 pin (CI has no torch)."""
    pin = (COMMITTED_FIXTURE / "FIXTURE_SHA256").read_text().strip().splitlines()
    assert pin, "empty pin file"
    for line in pin:
        digest, name = line.split()
        actual = hashlib.sha256((COMMITTED_FIXTURE / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} drifted from FIXTURE_SHA256"


def test_committed_fixture_is_reproducible(tmp_path):
    """Rebuilding the fixture from seed reproduces the committed bytes exactly."""
    hashes = fixture.build(tmp_path)
    for name, digest in hashes.items():
        committed = hashlib.sha256((COMMITTED_FIXTURE / name).read_bytes()).hexdigest()
        assert committed == digest, f"{name} is not reproducible from seed"
