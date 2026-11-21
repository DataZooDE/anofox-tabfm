"""The committed Nori 1.5 CI fixture's builder: determinism, conditioning, layout, and that the committed files
match it. Written red first: ``export_nori.fixture15`` did not exist.

Random weights at the fixture15 config (every structural flag of the released Model v3, tiny dims): no
Synthefy checkpoint bytes anywhere (CLAUDE.md, "license wall"). The checkpoint is written in the RELEASED
file's own layout, ``{model_config, model_state_dict, inference_config}`` (no EMA copy), so the fixture
exercises the native reader's ``model_state_dict`` path, which the Nori-30M fixture (EMA layout) cannot.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import onnx
import pytest
import torch

from export_nori import configs, fixture, fixture15
from export_nori.nori_patches import ExportWrapper

COMMITTED = pathlib.Path(__file__).resolve().parents[3] / "test" / "fixtures" / "nori15"
MLX_OPS_HEADER = pathlib.Path(__file__).resolve().parents[3] / "src" / "include" / "tabfm_mlx_ops.hpp"


@pytest.fixture(scope="module")
def model():
    return fixture15.seeded_model()


def test_the_seeded_model_is_deterministic():
    a, b = fixture15.seeded_model(), fixture15.seeded_model()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


def test_the_fixture_is_the_two_phase_architecture(model):
    assert model.two_phase_config == configs.fixture15().model_config["two_phase_config"]
    assert len(model.state_dict()) > 100


def test_the_fixture_is_responsive_and_well_conditioned(model):
    x, y = fixture.golden_tensors()
    fixture.assert_well_conditioned(ExportWrapper(model).eval(), model, x, y)


def test_the_default_initialisation_is_input_blind_which_is_why_the_gains_exist():
    """Without the gains a random two-phase stack decodes every row to nearly one value, so a golden test
    could not tell a right wiring from a wrong one."""
    from export_nori.nori_patches import build_model

    plain = build_model(configs.fixture15().model_config, seed=fixture15.SEED_WEIGHTS)
    x, y = fixture.golden_tensors()
    problems = fixture.check_well_conditioned(ExportWrapper(plain).eval(), plain, x, y)
    assert problems, "a default-initialised model passed the conditioning guard: the gains are unnecessary"


def test_the_build_is_deterministic_and_has_the_model_v3_layout(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ha, hb = fixture15.build(a), fixture15.build(b)
    assert ha == hb, "two builds differ"
    obj = torch.load(str(a / fixture15.WEIGHTS), map_location="cpu", weights_only=True)
    assert set(obj) == {"model_config", "model_state_dict", "inference_config"}, "the released file's layout"
    assert "ema_state_dict" not in obj
    assert {t.dtype for t in obj["model_state_dict"].values()} == {torch.float32}
    total = sum((a / n).stat().st_size for n in fixture15.FILES)
    assert total < 5 * 1024 * 1024, total
    assert sorted(p.name for p in a.iterdir()) == sorted(fixture15.FILES)


def test_the_built_graph_is_weight_free_and_inside_the_mlx_op_set(tmp_path):
    fixture15.build(tmp_path)
    proto = onnx.load(str(tmp_path / "graph_nori_regression.onnx"), load_external_data=False)
    ops = {n.op_type for n in proto.graph.node}
    body = MLX_OPS_HEADER.read_text().split("MlxImplementedOps()", 1)[1].split("};", 1)[0]
    assert ops <= set(re.findall(r'"([A-Za-z0-9_]+)"', body)), sorted(ops)
    assert not (tmp_path / "graph_nori_regression.onnx.data").exists()


def test_the_manifest_declares_a_regression_only_model_v3_fixture(tmp_path):
    fixture15.build(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["id"] == "nori15-fixture" and m["capabilities"] == ["regress"]
    assert m["preprocessing_profile"] == "nori_v15_raw"
    assert m["weights"]["regression"]["repo"] == "local:test/fixtures/nori15"
    entry = m["weights"]["regression"]["files"][0]
    assert hashlib.sha256((tmp_path / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_the_committed_fixture_is_this_builders_output():
    """The committed bytes are what the builder makes (CI verifies the pinned sha256 and never regenerates;
    this is the check a human runs after touching the builder or the exporter)."""
    built = pathlib.Path(__import__("tempfile").mkdtemp())
    hashes = fixture15.build(built)
    for name in fixture15.FILES:
        assert hashlib.sha256((COMMITTED / name).read_bytes()).hexdigest() == hashes[name], name
    golden = json.loads((COMMITTED / "golden_regression.json").read_text())
    assert golden["output_shape"] == [1, fixture.GOLDEN_SHAPE["t"], 1]
    assert len(golden["per_row"]["values"]) == fixture.GOLDEN_SHAPE["t"]
