"""Fast exporter checks on random-init models (no checkpoint bytes).

The masked contract is only worth anything if padding is inert, so the core gate here compares the
wrapper on an unpadded call against the same call padded in BOTH axes (rows past T, columns past d) --
which is what the MIGraphX plugin does to every call -- and against upstream's own unmasked inference
forward. Each comparison has a negative control that must be caught first, so a check that cannot
fail is not counted as a pass.
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
import torch

from export_kumo import configs, export
from export_kumo.graph_passes import split_sequences_to_slices
from export_kumo.kumo_patches import ExportWrapper, _wrap, build_model


def _model(task, seed=0):
    cfg = configs.get("fixture")
    return build_model(task, cfg.model, seed=seed, num_classes=cfg.classes,
                       num_quantiles=cfg.quantiles, wake_std=0.5)


def _inputs(t, h, s, classes, seed=3, cats=()):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g)
    y = torch.full((1, t), -100.0)
    y[0, :s] = (torch.randint(0, classes, (s,), generator=g).float() if classes
                else torch.randn(s, generator=g))
    cm = torch.zeros(1, h, dtype=torch.bool)
    for c in cats:
        cm[0, c] = True
    return x, y, cm


def _pad(x, y, cm, tp, hp):
    t, h = x.shape[1], x.shape[2]
    xp = torch.zeros(1, tp, hp)
    xp[:, :t, :h] = x
    yp = torch.full((1, tp), -100.0)
    yp[:, :t] = y
    cmp_ = torch.zeros(1, hp, dtype=torch.bool)
    cmp_[:, :h] = cm
    return xp, yp, cmp_


def test_wrap_is_modulo_without_division():
    for d in range(1, 9):
        idx = torch.arange(d)[:, None] + torch.tensor([0, 1, 3])[None, :]
        assert torch.equal(_wrap(idx, torch.tensor(d)), idx % d)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_padding_is_inert(task):
    classes = 0 if task == "regression" else 3
    w = ExportWrapper(_model(task), task == "regression").eval()
    t, h, s = 23, 5, 15
    x, y, cm = _inputs(t, h, s, classes, cats=(1,))
    with torch.no_grad():
        base = w(x, y, torch.tensor([s]), cm, torch.tensor([h]))
        xp, yp, cmp_ = _pad(x, y, cm, 64, 16)
        padded = w(xp, yp, torch.tensor([s]), cmp_, torch.tensor([h]))[:, :t]
        # negative control: tell the model the padding is real -> must change the answer
        wrong = w(xp, yp, torch.tensor([s]), cmp_, torch.tensor([16]))[:, :t]
    assert (padded - base).abs().max().item() < 1e-5
    assert (wrong - base).abs().max().item() > 1e-3


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_masked_equals_upstream_inference(task):
    """Query rows of the masked path == upstream's own positional inference forward."""
    classes = 0 if task == "regression" else 3
    m = _model(task)
    w = ExportWrapper(m, task == "regression").eval()
    t, h, s = 30, 7, 20
    x, y, cm = _inputs(t, h, s, classes, cats=(0, 4))
    with torch.no_grad():
        ours = w(x, y, torch.tensor([s]), cm, torch.tensor([h]))[0, s:]
        ytrain = y[:, :s] if task == "regression" else y[:, :s].long()
        ref = m(x, ytrain, cm)[0]
        if task == "regression":
            ref = ref.mean(-1, keepdim=True)
        # negative control: shift the split by one row
        off = w(x, y, torch.tensor([s - 1]), cm, torch.tensor([h]))[0, s:]
    assert (ours - ref).abs().max().item() < 1e-5
    assert (off - ref).abs().max().item() > 1e-4


def test_split_pass_leaves_no_sequence_ops(tmp_path):
    m = _model("classification")
    g = tmp_path / "g.onnx"
    export.export_graph(m, g, example=(24, 7, 15), classes=3, regression=False)
    proto = onnx.load(str(g), load_external_data=False)
    assert any(n.op_type == "SplitToSequence" for n in proto.graph.node)
    assert split_sequences_to_slices(proto) > 0
    assert not any(n.op_type in ("SplitToSequence", "SequenceAt", "Mod") for n in proto.graph.node)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_onnx_parity_including_padded_columns(task, tmp_path):
    classes = 0 if task == "regression" else 3
    m = _model(task)
    g = tmp_path / f"graph_{task}.onnx"
    w = export.export_graph(m, g, example=(24, 7, 15), classes=classes, regression=task == "regression")
    export.postprocess(g, dict(m.state_dict()))
    par = export.check_parity(g, w, ((40, 9, 25), (33, 12, 20, 10, (0, 4)), (17, 3, 9, 2)), classes)
    assert par["ok"], par
    assert all(np.isfinite(r["max_abs_delta_query_rows"]) for r in par["shapes"])


def test_gqa_expansion_equals_torch_enable_gqa():
    """The GQA rewrite (slices + Concat instead of SDPA's enable_gqa broadcast, which MIGraphX
    miscompiles) is checked against torch's own enable_gqa, not against itself: the patch is global."""
    from export_kumo.kumo_patches import _GqaFunctional

    g = torch.Generator().manual_seed(0)
    for hq, hkv in ((8, 2), (16, 2), (4, 1)):
        q = torch.randn(1, hq, 9, 16, generator=g)
        k = torch.randn(1, hkv, 13, 16, generator=g)
        v = torch.randn(1, hkv, 13, 16, generator=g)
        mask = torch.rand(1, 1, 9, 13, generator=g) > 0.3
        mask[..., 0] = True
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        ours = _GqaFunctional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        assert (ours - ref).abs().max().item() < 1e-6
        # negative control: a wrong grouping (interleaved instead of blocked) must be caught
        wrong_k = torch.cat([k[:, h % hkv:h % hkv + 1] for h in range(hq)], dim=1)
        wrong_v = torch.cat([v[:, h % hkv:h % hkv + 1] for h in range(hq)], dim=1)
        wrong = torch.nn.functional.scaled_dot_product_attention(q, wrong_k, wrong_v, attn_mask=mask)
        assert hkv == 1 or (wrong - ref).abs().max().item() > 1e-3
