"""The gate for the masked (ROCm) contract: padded + masked must equal the positional CPU forward.

``compare`` runs the positional ``ExportWrapper`` on the UNPADDED table (y = the context labels only) and the
``MaskExportWrapper`` on the same table padded to a larger (T, H) bucket the way the MIGraphX plugin pads it
(zeros, the -100 label sentinel), and compares every real row. ``CASES`` covers H = 1, S = 2, a single query row
and no padding at all. ``NEGATIVE_CONTROLS`` break one mask each; every one must be CAUGHT, or a pass would mean
nothing.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from export_orion_msp import orion_mask
from export_orion_msp.orion_msp_patches import ExportWrapper
from export_orion_msp.orion_mask import MaskExportWrapper

# (T, H, S, T_pad, H_pad). The 64-wide case puts most of the frozen random-link table's columns in the padding.
CASES = ((50, 7, 30, 128, 16), (31, 1, 30, 64, 16), (12, 5, 2, 128, 16), (20, 16, 12, 20, 16), (40, 10, 25, 64, 64))
# Logits run to ~11 on the real weights; fp32 reassociation alone (the masked forward on the UNPADDED table) moves
# them by up to 6.9e-4 there (the H = 1 case), and the same comparison in float64 agrees to 1.4e-12 -- the masked
# function is the positional one exactly. So the budget is 1e-3 (1e-4 relative); every control misses by > 100x.
TOL = 1e-3


def _inputs(t, h, s, n_classes, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g)
    y = (torch.arange(s) % n_classes).float()[torch.randperm(s, generator=g)].unsqueeze(0)
    return x, y


def _padded(x, y, tp, hp):
    t, h = x.shape[1], x.shape[2]
    xp = torch.zeros(1, tp, hp)
    xp[0, :t, :h] = x[0]
    yp = torch.full((1, tp), -100.0)
    yp[0, : y.shape[1]] = y[0]
    return xp, yp


def compare(model, cases=CASES, *, seed=0) -> float:
    n_classes = min(3, model.max_classes)
    worst = 0.0
    for i, (t, h, s, tp, hp) in enumerate(cases):
        x, y = _inputs(t, h, s, n_classes, seed + i)
        xp, yp = _padded(x, y, tp, hp)
        with torch.no_grad():
            ref = ExportWrapper(model)(x, y)[0]
            got = MaskExportWrapper(model)(xp, yp, torch.tensor([s]), torch.tensor([h]))[0]
        worst = max(worst, float((got[:t] - ref).abs().max()))
    return worst


@contextlib.contextmanager
def _patched(name, replacement):
    original = getattr(orion_mask, name)
    setattr(orion_mask, name, replacement)
    try:
        yield
    finally:
        setattr(orion_mask, name, original)


def _inducing_points_see_every_row():
    orig = orion_mask.isab

    def isab(blk, src, row_bias):
        return orig(blk, src, torch.zeros_like(row_bias))
    return _patched("isab", isab)


def _padded_columns_kept():
    orig = orion_mask.row_representations

    def rows(row, emb, d, table):
        return orig(row, emb, torch.tensor([emb.shape[2]]), table)
    return _patched("row_representations", rows)


def _icl_attends_every_row():
    orig = orion_mask.icl_logits

    def icl(icl_mod, reps, y, ctx):
        # labels still on the context only; only the attention's key mask is opened
        with _patched("key_bias", lambda valid, like: torch.zeros(valid.shape, dtype=like.dtype)):
            return orig(icl_mod, reps, y, ctx)
    return _patched("icl_logits", icl)


def _labels_on_every_row():
    orig = orion_mask.icl_logits

    def icl(icl_mod, reps, y, ctx):
        classes = torch.arange(icl_mod.max_classes).to(y.dtype)
        label = torch.nn.functional.linear((y.unsqueeze(-1) == classes).to(reps.dtype), icl_mod.y_encoder.weight,
                                           icl_mod.y_encoder.bias)
        off = ~ctx.view(1, -1, 1)
        # a query row that gets the label encoder's output: what an unmasked `R + y_encoder(y)` over T rows does
        return orig(icl_mod, reps + torch.where(off, label, torch.zeros_like(label)), y, ctx)
    return _patched("icl_logits", icl)


def _pma_sees_padded_columns():
    orig = orion_mask.pma_group

    def pma(pma_mod, feats, feat_bias):
        return orig(pma_mod, feats, torch.zeros_like(feat_bias))
    return _patched("pma_group", pma)


def _random_links_over_the_padded_length():
    orig = orion_mask.sparse_allow

    def allow(seq_len, num_special, num_random, n_real, table):
        return orig(seq_len, num_special, num_random, torch.tensor([seq_len]), table)
    return _patched("sparse_allow", allow)


NEGATIVE_CONTROLS = {
    "PMA pooling attends to padded columns": _pma_sees_padded_columns,
    "random links drawn over the padded length": _random_links_over_the_padded_length,
    "inducing points attend to every row, not the context": _inducing_points_see_every_row,
    "padded columns kept (d = padded H)": _padded_columns_kept,
    "ICL keys over every row, not the context": _icl_attends_every_row,
    "label embedding added to every row": _labels_on_every_row,
}


def gate(model) -> dict:
    """Pass only if the real comparison is within TOL AND every negative control is caught."""
    worst = compare(model)
    caught = {}
    for name, control in NEGATIVE_CONTROLS.items():
        with control():
            delta = compare(model)
        caught[name] = {"delta": delta, "caught": delta > 10 * TOL}
    ok = worst < TOL and all(c["caught"] for c in caught.values())
    return {"worst": worst, "ok": ok, "controls_caught": caught}


def check_onnx(graph_path, model, cases=CASES) -> dict:
    """ORT on the exported masked graph (padded inputs) vs the positional torch wrapper (unpadded)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    n_classes = min(3, model.max_classes)
    worst = 0.0
    for i, (t, h, s, tp, hp) in enumerate(cases):
        x, y = _inputs(t, h, s, n_classes, i)
        with torch.no_grad():
            ref = ExportWrapper(model)(x, y)[0].numpy()
        xp, yp = _padded(x, y, tp, hp)
        feed = {"x": xp.numpy(), "y": yp.numpy(), "train_size": np.array([s], dtype=np.int64),
                "d": np.array([h], dtype=np.int64)}
        wanted = {i.name for i in sess.get_inputs()}
        (got,) = sess.run(["logits"], {k: v for k, v in feed.items() if k in wanted})
        worst = max(worst, float(np.abs(got[0, :t] - ref).max()))
    return {"worst": worst, "ok": worst < TOL, "cases": [list(c) for c in cases]}
