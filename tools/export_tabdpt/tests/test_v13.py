"""TabDPT v1.3 (`tabdpt-v1-3`): the per-task y-encoders, exported through the pinned 1.2 package.

v1.3 changed one module family: the per-layer target encoder became a per-TASK pair
(cls_y_encoders.* Embedding / reg_y_encoders.* MLP). tabdpt_patches.upgrade_to_v13 reproduces it
on the pinned tabdpt==1.2.0 model; the upstream 1.3.1 package is the oracle for the real-weight
parity recorded in docs/REAL_MODELS.md.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from export_tabdpt import configs, export
from export_tabdpt.cli import build_model
from export_tabdpt.tabdpt_mask_patches import build_mask_wrapper
from export_tabdpt.tabdpt_patches import build_wrapper

REPO = pathlib.Path(__file__).resolve().parents[3]
RESOURCES = REPO / "resources"
FIXTURE = REPO / "test/fixtures/tabdpt13"
WEIGHTS13 = pathlib.Path.home() / ".cache/anofox-tabfm/Layer6__TabDPT@main/tabdpt1_3/model.safetensors"


def test_v13_state_dict_signature():
    """679 tensors: v1.2's 647 minus 128 y_encoders.* plus 32 cls + 128 reg encoder tensors."""
    model = build_model(configs.real13())
    sd = model.state_dict()
    assert len(sd) == 679
    assert not any(k.startswith("y_encoders.") for k in sd)
    assert tuple(sd["cls_y_encoders.0.weight"].shape) == (16, 128)
    assert tuple(sd["reg_y_encoders.0.0.weight"].shape) == (128, 1)
    assert tuple(sd["reg_y_encoders.0.2.weight"].shape) == (128, 128)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_v13_fixture_exports_with_parity_and_records_unused(tmp_path, task):
    cfg = configs.fixture13()
    model = build_model(cfg, seed=0)
    wrapper = build_wrapper(model, task)
    graph = tmp_path / f"graph_{task}.onnx"
    export.export_graph(wrapper, graph, cfg=cfg)
    tmap = export.postprocess(graph, dict(model.state_dict()))
    parity = export.check_parity(graph, wrapper, cfg.parity_shapes)
    assert parity["ok"], parity
    # Each task's graph reads only its own encoder; the other one is listed, so the engine
    # skips it when it injects the shared file.
    other = "reg_y_encoders." if task == "classification" else "cls_y_encoders."
    mine = "cls_y_encoders." if task == "classification" else "reg_y_encoders."
    assert tmap["unused_by_graph"] and all(k.startswith(other) for k in tmap["unused_by_graph"])
    assert any(v.startswith(mine) for v in tmap["initializers"].values())
    assert not any(v.startswith(other) for v in tmap["initializers"].values())


def test_v13_y_encoder_depends_on_task():
    """The classification embedding must not be what regression uses (and vice versa).

    Built through mask_parity.build_model, which wakes the zero-initialised residual projections:
    with upstream's init the attention output is identically zero and y could not move anything.
    """
    from export_tabdpt.mask_parity import build_model as woken

    model = woken(configs.fixture13(), seed=0)
    base = woken(configs.fixture13(), seed=0)
    with torch.no_grad():
        for p in model.cls_y_encoders.parameters():
            p.add_(torch.randn_like(p))  # perturb only the CLASS encoder
    torch.manual_seed(1)
    x = torch.randn(1, 20, 5)
    y = torch.randn(1, 12)
    with torch.no_grad():
        r_pert = build_wrapper(model, "regression")(x, y)
        r_base = build_wrapper(base, "regression")(x, y)
        c_pert = build_wrapper(model, "classification")(x, y.abs().round().clamp(max=3))
        c_base = build_wrapper(base, "classification")(x, y.abs().round().clamp(max=3))
    assert torch.equal(r_pert, r_base)       # regression never reads cls_y_encoders
    assert not torch.allclose(c_pert, c_base)  # classification does


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_v13_masked_matches_positional(task):
    model = build_model(configs.fixture13(), seed=0)
    pos, msk = build_wrapper(model, task), build_mask_wrapper(model, task)
    torch.manual_seed(3)
    x = torch.randn(1, 30, 6)
    y = (torch.randint(0, 4, (1, 30)).float() if task == "classification" else torch.randn(1, 30))
    y[:, 22:] = -999.0  # poison the query labels: the mask must make them irrelevant
    with torch.no_grad():
        a = pos(x, y[:, :22])[:, 22:]
        b = msk(x, y, torch.tensor([22]), torch.tensor([6]))[:, 22:]
    assert (a - b).abs().max() < 1e-4


def test_committed_fixture_matches_pin():
    import hashlib

    pin = (FIXTURE / "FIXTURE_SHA256").read_text().split("\n")
    for line in filter(None, pin):
        digest, name = line.split()
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_committed_v13_map_covers_the_published_safetensors(task):
    if not WEIGHTS13.exists():
        pytest.skip("real TabDPT v1.3 weights not cached")
    import hashlib

    with open(WEIGHTS13, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        raw = f.read(n)
    have = set(json.loads(raw)) - {"__metadata__"}
    tmap = json.loads((RESOURCES / f"tensor_map_tabdpt13_{task}.json").read_text())
    want = set(tmap["initializers"].values())
    # used + unused partition the file exactly
    assert want | set(tmap["unused_by_graph"]) == have
    assert not (want & set(tmap["unused_by_graph"]))
    assert hashlib.sha256(raw).hexdigest() == (
        "8ab85c0bf4ce2fc87a81f040b4170ae7c10b50d75bad937d2f8009558767b5a3")
    configs.assert_matches_checkpoint(str(WEIGHTS13))
    configs.assert_state_dict_matches_checkpoint(build_model(configs.real13()), str(WEIGHTS13))
