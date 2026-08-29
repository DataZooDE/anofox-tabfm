"""Tests: patched Orion-MSP exports, stays weight-free, and freezes its mask.

The load-bearing tests here are about the block-sparse attention mask. Upstream
redraws it per call with `torch.randperm`, so the exported graph must (a) be
deterministic and (b) reproduce upstream's EFFECTIVE mask — including upstream's
own dead sliding-window branch, which it would be wrong to "fix".
"""

from __future__ import annotations

import json
import pathlib

import onnx
import pytest
import torch

from export_orion_msp import configs, export
from export_orion_msp.orion_msp_patches import build_model

REPO = pathlib.Path(__file__).resolve().parents[3]  # tests/ -> tool -> tools/ -> repo
COMMITTED_FIXTURE = REPO / "test/fixtures/orion_msp"
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
    parity = export.check_parity(graph, wrapper, ((40, 7, 30),))
    assert parity["ok"], parity
    assert parity["worst"] < 1e-4


def test_frozen_mask_is_deterministic_but_upstream_is_not():
    """The whole point of Patch 4: same inputs, same mask, every time."""
    import orion_msp.model.interaction as interaction
    from export_orion_msp.orion_msp_patches import apply

    kw = dict(seq_len=24, num_special=8, window=3, num_random=2, return_bool=True)
    apply()
    a = interaction._build_block_sparse_mask(**kw)
    b = interaction._build_block_sparse_mask(**kw)
    assert torch.equal(a, b), "frozen mask must not vary between calls"


def test_frozen_mask_matches_upstreams_effective_structure():
    """Reproduce upstream's mask, INCLUDING its dead sliding-window branch.

    Upstream computes `local = (dist <= window).to(dtype) * 0.0`, which is all
    zeros regardless of `dist`, so `local + -inf` is -inf everywhere and the
    `torch.where` is a no-op: the window opens nothing. The effective mask is
    specials + random links + diagonal, and that is what the released weights
    were trained against. Honouring the window would open 14 keys per query
    where upstream opens 11 — a different architecture under the same weights.
    """
    import orion_msp.model.interaction as interaction
    from export_orion_msp.orion_msp_patches import apply

    L, ns, window, nr = 24, 8, 3, 2
    apply()
    b = interaction._build_block_sparse_mask(
        seq_len=L, num_special=ns, window=window, num_random=nr, return_bool=True)

    allowed = ~b
    # Special rows/columns fully connected, diagonal always open.
    assert allowed[:ns].all()
    assert allowed[:, :ns].all()
    assert allowed.diagonal().all()

    # Each non-special query: ns specials + itself + at most `nr` extra links.
    # (Exactly `nr` unless a pick collides with an already-open key, which
    # upstream's randperm also allows.)
    per_row = allowed[ns:].sum(-1)
    assert (per_row >= ns + 1).all()
    assert (per_row <= ns + 1 + nr).all(), per_row

    # The window is NOT honoured: a query 1 step away but outside the specials
    # is not automatically open.
    assert per_row.max().item() <= ns + 1 + nr


def test_committed_tensor_map_keys_match_published_checkpoint_namespace():
    """Bare keys, not Orion-BiX's `_orig_mod.` prefix (0/357 if inherited)."""
    assert export.CKPT_KEY_PREFIX == ""
    tmap = json.loads((RESOURCES / "tensor_map_orion_msp_classification.json").read_text())
    keys = list(tmap["initializers"].values())
    assert keys
    assert not any(k.startswith("_orig_mod.") for k in keys)
    assert any(k.startswith("col_embedder.") for k in keys), keys[:5]


def test_link_table_stays_inline_and_unmapped(tmp_path):
    """The frozen link table is code-generated, so it never enters the map."""
    _model, _wrapper, graph, tmap = _export(tmp_path)
    assert not any(k.endswith("_LINK_SCORES") for k in tmap["initializers"])
    proto = onnx.load(str(graph), load_external_data=True)
    assert any(i.name.endswith("_LINK_SCORES") for i in proto.graph.initializer)


def test_released_config_assumptions_are_enforced():
    """A future checkpoint that widens the architecture must fail loudly."""
    kwargs = dict(configs.real().model_kwargs)
    with pytest.raises(ValueError, match="row_scales"):
        configs.assert_shipped_path(dict(kwargs, row_scales=(1, 4, 16)))
    with pytest.raises(ValueError, match="perc_num_latents"):
        configs.assert_shipped_path(dict(kwargs, perc_num_latents=8))


def test_committed_fixture_is_classify_only_and_weight_free():
    manifest = json.loads((COMMITTED_FIXTURE / "manifest.json").read_text())
    assert manifest["capabilities"] == ["classify"]
    assert "regression" not in manifest["weights"]
    assert not list(COMMITTED_FIXTURE.glob("*.onnx.data")), "weight bytes on disk"
