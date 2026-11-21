"""Nori 1.5 masked (ROCm) contract: padded + masked == the positional CPU forward, and every control is caught.

Random weights at the fixture15 config (offline), woken up as for the fixture so masking errors are visible; the same
gate runs on the real weights in test_real_weights_v15.py."""

from __future__ import annotations

import pytest
import torch

from export_nori import configs, export, fixture15
from export_nori.mask_parity15 import NEGATIVE_CONTROLS, TOL, check_onnx, compare, gate
from export_nori.nori15_mask import encode_x, ple_edges


@pytest.fixture(scope="module")
def model():
    return fixture15.seeded_model()


def test_padded_and_masked_equals_positional(model):
    assert compare(model) < TOL


@pytest.mark.parametrize("control", sorted(NEGATIVE_CONTROLS))
def test_every_negative_control_is_caught(model, control):
    with NEGATIVE_CONTROLS[control]():
        assert compare(model) > 10 * TOL, control


def test_the_gate_passes_only_with_every_control_caught(model):
    res = gate(model)
    assert res["ok"] and all(res["controls_caught"].values()), res


def _ple(model):
    return next(m for m in model.x_preprocess if type(m).__name__ == "PromptQuantilePLEEncoder")


def _probe_table():
    """Columns: decades, zeros, a constant, heavy duplicates, NaN, an all-NaN column, a single value, inf."""
    g = torch.Generator().manual_seed(0)
    t = 40
    x = torch.randn(t, 9, generator=g)
    x[:, 0] = torch.logspace(-6, 3, t)                          # decades
    x[:, 1] = 0.0                                               # zero column
    x[:, 2] = 1.5                                               # constant
    x[:, 3] = torch.randint(0, 3, (t,), generator=g).float()    # heavy duplicates
    x[::4, 4] = float("nan")
    x[:, 5] = float("nan")
    x[:, 6] = float("nan")
    x[3, 6] = 2.0                                               # one finite value
    x[::5, 7] = float("inf")
    x[:, 8] = (torch.arange(t) % 2).float()                     # two values
    return x


@pytest.mark.parametrize("s", [2, 7, 25, 40])
def test_the_migraphx_safe_ple_edges_are_upstreams(model, s):
    """No Sort / Gather: the same edges as upstream's sort + gather, with padding rows and columns that must not
    leak (junk rows after the table, junk columns beyond d, rows after the context)."""
    ple = _ple(model)
    x = _probe_table()
    t, h = x.shape
    with torch.no_grad():
        want = ple._build_reference(x.view(1, t, h, 1), torch.isfinite(x).view(1, t, h, 1), s)["edges"][0, :, 0]
        pad = torch.full((t + 3, h + 4), 7.0)
        pad[:t, :h] = x
        ctx = torch.arange(t + 3) < s
        valid = torch.isfinite(pad) & ctx.view(-1, 1)
        valid[:, h:] = False
        got = ple_edges(ple, torch.where(torch.isfinite(pad), pad, torch.zeros_like(pad)), valid)
    assert torch.equal(got[:h], want), (got[:h] - want).abs().max()
    assert torch.equal(got[h:], torch.zeros_like(got[h:]))


def test_the_migraphx_safe_tokenizer_with_ple_is_upstreams(model):
    """MaskEmbEncoder (RBF + PLE channels, no split / integer div): the same embedding at decades, zero, NaN."""
    enc = model.encoder_x[0]
    ple = _ple(model)
    x = _probe_table()
    x = torch.where(torch.isfinite(x), x, torch.full_like(x, float("nan")))   # process_4_x: non-finite -> NaN
    t, h = x.shape
    s = 25
    ctx = torch.arange(t) < s
    valid = torch.isfinite(x) & ctx.view(-1, 1)
    with torch.no_grad():
        edges = ple_edges(ple, torch.where(torch.isfinite(x), x, torch.zeros_like(x)), valid)
        got = encode_x(enc, x, edges)
        data = x.view(1, t, h, 1).clone()
        want = enc({"data": data, "eval_pos": s,
                    "_prompt_quantile_ple_reference": ple._build_reference(data, torch.isfinite(data), s)})["data"][0]
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_the_masked_graph_in_ort_equals_positional(tmp_path, model):
    import onnx

    cfg = configs.fixture15()
    graph = tmp_path / "mask.onnx"
    export.export_mask_graph(model, graph, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features,
                             example=cfg.example)
    res = check_onnx(graph, model)
    assert res["ok"], res
    proto = onnx.load(str(graph), load_external_data=False)
    assert [i.name for i in proto.graph.input] == ["x", "y", "train_size", "d", "n_rows"]
    ops = {n.op_type for n in proto.graph.node}
    assert not ops & {"SplitToSequence", "SequenceAt", "Einsum", "Unique", "NonZero", "TopK", "GatherElements",
                      "GatherND", "LayerNormalization", "RMSNormalization"}, ops
