"""The committed CI fixture's builder: determinism, conditioning, and the guards that keep it honest.

LimiX's numeric tokenizer encodes a value as (sign, decimal exponent, mantissa), and an EXACT zero as a
separate token. A feature value that standardises to exactly 0 (it equals its column mean, which discrete
or symmetric data does all the time) therefore flips to a completely different embedding under any
rounding noise, and a value near zero is hypersensitive in proportion to 1/|z|. A golden-value fixture
built on such inputs is not reproducible across platforms, so the builder asserts against it; these tests
show the assertion is able to fail.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
import torch

from export_limix import fixture
from export_limix.limix_patches import ExportWrapper


def test_inputs_are_exact_in_float32_and_expressible_in_sql():
    for task in ("classification", "regression"):
        x, y = fixture.sql_inputs(task)
        arr = np.array(x, dtype=np.float64)
        assert (arr.astype(np.float32).astype(np.float64) == arr).all(), "an input is not exact in float32"
        assert len(x) == fixture.GOLDEN_SHAPE["t"] and len(x[0]) == fixture.GOLDEN_SHAPE["h"]
        assert len(y) == fixture.GOLDEN_SHAPE["s"]


def test_the_seeded_model_is_deterministic():
    a, b = fixture.seeded_model(), fixture.seeded_model()
    sa, sb = a.state_dict(), b.state_dict()
    assert sorted(sa) == sorted(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_the_fixture_is_responsive_and_well_conditioned(task):
    model = fixture.seeded_model()
    x, y = fixture.golden_tensors(task)
    fixture.assert_well_conditioned(ExportWrapper(model, task=task).eval(), model, x, y, task)


def test_the_conditioning_guard_can_fail():
    """Quarter-multiples with no sub-structure give columns whose mean equals one of their values, so
    some standardised values are exactly 0 and flip tokens under 1e-6 noise (measured: 41 zeros, output
    moves by 0.04 .. 20 on a logit spread of 0.03 .. 12). The guard must reject that input."""
    model = fixture.seeded_model()
    t, h, s = (fixture.GOLDEN_SHAPE[k] for k in ("t", "h", "s"))
    x = torch.tensor([[[((i * 7 + j * 3 + (i * j) % 5) % 11) * 0.25 - 1.0 for j in range(h)]
                       for i in range(t)]], dtype=torch.float32)
    _, y = fixture.golden_tensors("classification")
    with pytest.raises(AssertionError, match="ill-conditioned|exact zero"):
        fixture.assert_well_conditioned(ExportWrapper(model, task="classification").eval(), model, x, y,
                                        "classification")


def test_a_row_with_a_near_tie_is_rejected():
    logits = torch.zeros(1, 4, 3)
    logits[0, :, 0] = 1.0
    logits[0, 2, 1] = 1.0 - 1e-4                 # a near-tie in one row
    with pytest.raises(AssertionError, match="near-tie"):
        fixture.assert_margins(logits)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("limix_fixture")
    hashes = fixture.build(out)
    return out, hashes


def test_the_build_is_complete_small_and_weight_free(built):
    out, hashes = built
    assert set(hashes) == set(fixture.FILES)
    assert sum((out / n).stat().st_size for n in fixture.FILES) < 5 * 1024 * 1024
    for task in ("classification", "regression"):
        assert not (out / f"graph_limix_{task}.onnx.data").exists(), "weight bytes left on disk"


def test_the_build_is_byte_deterministic(built, tmp_path):
    _, hashes = built
    assert fixture.build(tmp_path) == hashes


def test_the_golden_files_cover_every_row_and_all_classes(built):
    out, _ = built
    t, s = fixture.GOLDEN_SHAPE["t"], fixture.GOLDEN_SHAPE["s"]
    cls = json.loads((out / "golden_classification.json").read_text())
    reg = json.loads((out / "golden_regression.json").read_text())
    assert len(cls["per_row"]["labels"]) == t and len(reg["per_row"]["values"]) == t
    assert set(cls["per_row"]["labels"]) == {"c0", "c1", "c2"}, "the fixture must predict every class"
    assert cls["inputs"]["train_size"] == s
    assert len(set(reg["per_row"]["values"])) > t // 2, "regression output is nearly constant"


def test_the_manifest_declares_one_weights_file_for_both_tasks(built):
    out, _ = built
    man = json.loads((out / "manifest.json").read_text())
    assert man["preprocessing_profile"] == "limix_v1_raw"
    files = {t: [f["path"] for f in man["weights"][t]["files"]] for t in ("classification", "regression")}
    assert files["classification"] == files["regression"] == ["model.safetensors"]
    assert man["size_regime"]["max_classes"] == 3
