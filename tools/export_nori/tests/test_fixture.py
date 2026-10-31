"""The committed CI fixture's builder: determinism, conditioning, and that the committed files match it.

Nori keeps LimiX's decimal numeric tokenizer, so an exact standardised zero is its own token and values
near zero are hypersensitive; the builder rejects such inputs, and these tests show the guard can fail.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest
import torch

from export_nori import fixture
from export_nori.nori_patches import ExportWrapper

COMMITTED = pathlib.Path(__file__).resolve().parents[3] / "test" / "fixtures" / "nori"


def test_inputs_are_exact_in_float32_and_expressible_in_sql():
    x, y = fixture.sql_inputs()
    for arr in (np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)):
        assert (arr.astype(np.float32).astype(np.float64) == arr).all(), "an input is not exact in float32"
    assert len(x) == fixture.GOLDEN_SHAPE["t"] and len(x[0]) == fixture.GOLDEN_SHAPE["h"]
    assert len(y) == fixture.GOLDEN_SHAPE["s"]


def test_the_seeded_model_is_deterministic():
    a, b = fixture.seeded_model(), fixture.seeded_model()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k


def test_the_fixture_is_responsive_and_well_conditioned():
    model = fixture.seeded_model()
    x, y = fixture.golden_tensors()
    fixture.assert_well_conditioned(ExportWrapper(model).eval(), model, x, y)


def test_the_conditioning_guard_can_fail():
    """Quarter-multiples without the sixteenths term put column means on data values, so standardised
    values are exactly 0: the guard must reject that input."""
    model = fixture.seeded_model()
    t, h = fixture.GOLDEN_SHAPE["t"], fixture.GOLDEN_SHAPE["h"]
    x = torch.tensor([[[((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0 for j in range(h)]
                       for i in range(t)]], dtype=torch.float32)
    _, y = fixture.golden_tensors()
    problems = fixture.check_well_conditioned(ExportWrapper(model).eval(), model, x, y)
    assert any("exact zero" in p or "1/|z|" in p for p in problems), problems


def test_the_default_initialisation_is_input_blind_which_is_why_the_gains_exist():
    """Without sensitive() the random stack decodes every row to nearly one value (measured std ~1e-4 over
    20 rows), so a golden test could not tell a right wiring from a wrong one."""
    from export_nori import configs
    from export_nori.nori_patches import build_model

    model = build_model(configs.fixture().model_config, seed=fixture.SEED_WEIGHTS)
    x, y = fixture.golden_tensors()
    problems = fixture.check_well_conditioned(ExportWrapper(model).eval(), model, x, y)
    assert any("nearly constant" in p for p in problems), problems


def test_the_committed_fixture_matches_its_manifest_and_golden():
    """The committed bytes are what the manifest says, and the golden values are this builder's output
    for the committed weights (re-derived here in eager PyTorch from the committed checkpoint)."""
    manifest = json.loads((COMMITTED / "manifest.json").read_text())
    entry = manifest["weights"]["regression"]["files"][0]
    ckpt = COMMITTED / entry["path"]
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == entry["sha256"]
    golden = json.loads((COMMITTED / "golden_regression.json").read_text())

    from export_nori import configs
    from export_nori.nori_patches import build_model

    obj = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    assert set(obj) == {"model_config", "ema_state_dict"}, "the released file's layout"
    model = build_model(configs.fixture().model_config)
    model.load_state_dict(obj["ema_state_dict"], strict=True)
    x, y = fixture.golden_tensors()
    with torch.no_grad():
        out = ExportWrapper(model.eval())(x, y)[0, :, 0]
    assert np.allclose(out.numpy(), golden["per_row"]["values"], atol=1e-4)
