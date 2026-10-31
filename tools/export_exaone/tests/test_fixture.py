"""The committed CI fixture: deterministic, responsive, and the committed files match the builder."""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from export_exaone import configs, fixture
from export_exaone.exaone_patches import ExportWrapper, build_model

COMMITTED = pathlib.Path(__file__).resolve().parents[3] / "test" / "fixtures" / "exaone"


@pytest.mark.parametrize("task", fixture.TASKS)
def test_inputs_are_exact_in_float32(task):
    x, y = fixture.sql_inputs(task)
    for arr in (np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)):
        assert (arr.astype(np.float32).astype(np.float64) == arr).all()


@pytest.mark.parametrize("task", fixture.TASKS)
def test_the_fixture_is_deterministic_and_responsive(task):
    a, b = fixture.seeded_model(task), fixture.seeded_model(task)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    x, y = fixture.golden_tensors(task)
    assert not fixture.check(ExportWrapper(a, task).eval(), x, y, task)


def test_the_default_initialisation_collapses_which_is_why_the_gains_exist():
    cfg = configs.get("fixture", "classification")
    m = build_model("classification", cfg.model_config, seed=fixture.SEED_WEIGHTS["classification"])
    x, y = fixture.golden_tensors("classification")
    assert fixture.check(ExportWrapper(m, "classification").eval(), x, y, "classification")


@pytest.mark.parametrize("task", fixture.TASKS)
def test_the_committed_fixture_matches_its_manifest_and_golden(task):
    manifest = json.loads((COMMITTED / "manifest.json").read_text())
    entry = manifest["weights"][task]["files"][0]
    path = COMMITTED / entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    cfg = configs.get("fixture", task)
    m = build_model(task, cfg.model_config, cfg.regression_config)
    m.load_state_dict(load_file(str(path)), strict=True)
    x, y = fixture.golden_tensors(task)
    golden = json.loads((COMMITTED / f"golden_{task}.json").read_text())
    with torch.no_grad():
        out = ExportWrapper(m.eval(), task)(x, y)
    assert np.allclose(out.numpy(), np.array(golden["outputs"]), atol=1e-4)
