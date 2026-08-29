"""Tests: patched TabDPT exports, stays weight-free, and keeps its dims dynamic.

The two regressions pinned hardest here are the ones that did NOT announce
themselves during development: a context length silently baked into a constant,
and a checkpoint key prefix that mapped nothing. Both produced an export that
reported success.
"""

from __future__ import annotations

import json
import pathlib

import onnx
import pytest
import torch

from export_tabdpt import configs, export
from export_tabdpt.cli import build_model
from export_tabdpt.tabdpt_patches import build_wrapper

REPO = pathlib.Path(__file__).resolve().parents[3]  # tests/ -> tool -> tools/ -> repo
COMMITTED_FIXTURE = REPO / "test/fixtures/tabdpt"
RESOURCES = REPO / "resources"


def _export(tmp_path, task="classification"):
    cfg = configs.fixture()
    model = build_model(cfg, seed=0)
    wrapper = build_wrapper(model, task)
    graph = tmp_path / f"graph_{task}.onnx"
    export.export_graph(wrapper, graph, cfg=cfg)
    tmap = export.postprocess(graph, dict(model.state_dict()))
    return model, wrapper, graph, tmap


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_exports_and_parity(tmp_path, task):
    _model, wrapper, graph, _tmap = _export(tmp_path, task)
    # Shapes all differ from the export example (12, 5, 8), so T/H/S dynamism is
    # genuinely exercised rather than replayed.
    parity = export.check_parity(graph, wrapper, ((40, 7, 30), (16, 9, 6)))
    assert parity["ok"], parity
    assert parity["worst"] < 1e-4


def test_train_size_dim_is_not_specialized(tmp_path):
    """`y`'s length must stay symbolic.

    `get_scale_param` materialises the context length to build the attention
    temperature. Upstream does that with `torch.as_tensor(eval_pos)`, which bakes
    a symbolic dim to a constant: the export still SUCCEEDS, and produces a graph
    whose `y` is pinned to the tracing example's length. ORT then rejects the
    first real call with any other context size. Only inspecting the graph
    catches it, so inspect the graph.
    """
    _model, _wrapper, graph, _tmap = _export(tmp_path)
    proto = onnx.load(str(graph), load_external_data=False)
    dims = {i.name: [(d.dim_param or d.dim_value)
                     for d in i.type.tensor_type.shape.dim]
            for i in proto.graph.input}
    assert dims["y"][1] == "train", dims
    assert dims["x"][1] == "rows", dims
    assert dims["x"][2] == "features", dims


def test_committed_tensor_map_keys_match_published_checkpoint_namespace():
    """The map must use BARE keys, not Orion-BiX's `_orig_mod.` prefix.

    This exporter was seeded from tools/export_orion_bix, whose checkpoint was
    saved from a torch.compile-wrapped module. TabDPT's published safetensors use
    the plain module namespace, so inheriting that prefix mapped 0/647 keys — a
    graph that loads fine and injects nothing.
    """
    assert export.CKPT_KEY_PREFIX == ""
    tmap = json.loads((RESOURCES / "tensor_map_tabdpt_classification.json").read_text())
    keys = list(tmap["initializers"].values())
    assert keys, "committed tensor map is empty"
    assert not any(k.startswith("_orig_mod.") for k in keys)
    # Real module paths, e.g. "encoder.weight" / "transformer_encoder.0...".
    assert any(k.startswith("encoder.") for k in keys), keys[:5]


def test_bin_centres_stays_inline_and_unmapped(tmp_path):
    """The regression bar grid is code-generated, not a weight.

    It is a linspace over three scalar config values, so it must stay inline in
    the weight-free graph and never enter the tensor map — the treatment
    export_tabpfn gives `_pos_base`. At the released bin_count of 2048 it is
    8 KB, over the >=1KB "unmatched large initializer" guard.
    """
    _model, _wrapper, graph, tmap = _export(tmp_path, "regression")
    assert not any(k.endswith("bin_centres") for k in tmap["initializers"])
    proto = onnx.load(str(graph), load_external_data=True)
    names = {i.name for i in proto.graph.initializer}
    assert any(n.endswith("bin_centres") for n in names), names


def test_committed_fixture_is_weight_free_and_shared():
    """One safetensors serves BOTH tasks (single head, two output halves)."""
    manifest = json.loads((COMMITTED_FIXTURE / "manifest.json").read_text())
    clf = manifest["weights"]["classification"]["files"][0]["path"]
    reg = manifest["weights"]["regression"]["files"][0]["path"]
    assert clf == reg == "model.safetensors"
    assert manifest["capabilities"] == ["classify", "regress"]
    assert not list(COMMITTED_FIXTURE.glob("*.onnx.data")), "weight bytes on disk"


def test_frozen_scale_patch_is_algebraically_upstream():
    """Folding beta into q must equal scaling the SDPA temperature by beta.

    softmax(1/sqrt(d) * (beta*q) @ k^T) == softmax(beta/sqrt(d) * q @ k^T)
    """
    import math

    import torch.nn.functional as F

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 6, 8) for _ in range(3))
    beta = torch.tensor(1.7)
    d = 1.0 / math.sqrt(8)
    upstream = F.scaled_dot_product_attention(q, k, v, scale=float(d * beta))
    patched = F.scaled_dot_product_attention(q * beta, k, v, scale=d)
    assert torch.allclose(upstream, patched, atol=1e-6), (upstream - patched).abs().max()


WEIGHTS = pathlib.Path.home() / ".cache/anofox-tabfm/Layer6__TabDPT@main/model.safetensors"


def test_real_config_matches_published_checkpoint():
    """Skipped unless the real weights are cached; a dims drift is silent."""
    if not WEIGHTS.exists():
        pytest.skip("real TabDPT weights not cached")
    configs.assert_matches_checkpoint(str(WEIGHTS))  # raises SystemExit on drift


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_committed_map_covers_the_published_safetensors(task):
    """The claim the C++ header-sha constant cannot make on its own.

    `ExpectedWeightsHeaderShaFor("tabdpt", ...)` is a constant compared against a
    constant: it would keep passing if Layer 6 republished the checkpoint with a
    different tensor layout, while the ext graph's baked offsets silently indexed
    the wrong tensors. Checking the real file is what actually pins it.
    """
    import hashlib

    if not WEIGHTS.exists():
        pytest.skip("real TabDPT weights not cached")

    with open(WEIGHTS, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        raw = f.read(n)
    header = json.loads(raw)
    have = set(header) - {"__metadata__"}

    tmap = json.loads((RESOURCES / f"tensor_map_tabdpt_{task}.json").read_text())
    want = set(tmap["initializers"].values())
    assert not (want - have), sorted(want - have)[:5]

    # Both tasks read the SAME file, so both must agree with the one baked sha.
    assert hashlib.sha256(raw).hexdigest() == (
        "0959127002658b64f981ea233be8f1efec3dade6384a4fe637c75400a41a9a78")
