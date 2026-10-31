"""The gate for the masked (ROCm) contract: padded + masked must equal the positional CPU forward.

``compare`` runs the positional ``ExportWrapper`` on the UNPADDED table and the ``MaskExportWrapper`` on the
same table padded to a larger (T, H) bucket the way the MIGraphX plugin pads it (zeros, the -100 label
sentinel), and compares every real row. ``NEGATIVE_CONTROLS`` break one mask each; every one must be CAUGHT,
or a pass would mean nothing.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from export_nori import nori_mask
from export_nori.nori_mask import MaskExportWrapper
from export_nori.nori_patches import ExportWrapper

# (T, H, S, T_pad, H_pad, constant_column): odd and even widths, one feature, S = 2, an unpadded bucket
CASES = ((50, 7, 30, 128, 16, 2), (31, 1, 30, 64, 16, None), (12, 5, 2, 128, 16, None),
         (20, 16, 12, 20, 16, 0), (40, 8, 25, 128, 16, None))
TOL = 1e-3   # the decimal tokenizer amplifies float32 noise (docs/REAL_MODELS.md); measured worst ~1e-5


def _inputs(t, h, s, constant, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g) * 2 + 0.5
    if constant is not None:
        x[0, :, constant] = 1.5
    return x, torch.randn(1, s, generator=g) * 5 + 3


def compare(model, cases=CASES, *, seed=0) -> float:
    worst = 0.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(t, h, s, constant, seed + i)
        with torch.no_grad():
            ref = ExportWrapper(model)(x, y)[0, :, 0]
            xp = torch.zeros(1, tp, hp)
            xp[0, :t, :h] = x[0]
            yp = torch.full((1, tp), -100.0)
            yp[0, :s] = y[0]
            got = MaskExportWrapper(model)(xp, yp, torch.tensor([s]), torch.tensor([h]), torch.tensor([t]))[0, :t, 0]
        worst = max(worst, float((got - ref).abs().max()))
    return worst


@contextlib.contextmanager
def _patched(name, replacement):
    original = getattr(nori_mask, name)
    setattr(nori_mask, name, replacement)
    try:
        yield
    finally:
        setattr(nori_mask, name, original)


def _wrap_forward(change):
    orig = nori_mask.masked_forward

    def forward(model, x, y_norm, train_size, d):
        return orig(model, x, y_norm, *change(x, train_size, d))
    return _patched("masked_forward", forward)


def _wrap_layer(change):
    orig = nori_mask.masked_layer

    def layer(lyr, x, *, rows, ctx_key_mask, log_s, group_key_mask):
        kw = change(dict(rows=rows, ctx_key_mask=ctx_key_mask, log_s=log_s, group_key_mask=group_key_mask))
        return orig(lyr, x, **kw)
    return _patched("masked_layer", layer)


def _preprocess_over_every_row():
    orig = nori_mask.preprocess
    return _patched("preprocess", lambda model, xg, ctx: orig(model, xg, torch.ones_like(ctx)))


NEGATIVE_CONTROLS = {
    "feature statistics over every row, not the context": _preprocess_over_every_row,
    "padded feature groups kept (d = padded H)":
        lambda: _wrap_forward(lambda x, s, d: (s, torch.tensor([x.shape[2]]))),
    "no key mask in the sample attention":
        lambda: _wrap_layer(lambda kw: {**kw, "ctx_key_mask": torch.ones_like(kw["ctx_key_mask"])}),
    "QASSMax length from the padded shape":
        lambda: _wrap_layer(lambda kw: {**kw, "log_s": torch.log(torch.tensor([float(kw["rows"])]))}),
}


def gate(model) -> dict:
    worst = compare(model)
    caught = {}
    for name, control in NEGATIVE_CONTROLS.items():
        with control():
            caught[name] = compare(model) > 10 * TOL
    return {"worst": worst, "ok": worst < TOL and all(caught.values()), "controls_caught": caught}


def check_onnx(graph_path, model, cases=CASES) -> dict:
    """ORT on the exported masked graph (padded inputs) vs the positional torch wrapper (unpadded)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(t, h, s, constant, i)
        with torch.no_grad():
            ref = ExportWrapper(model)(x, y)[0, :, 0].numpy()
        xp = np.zeros((1, tp, hp), dtype=np.float32)
        xp[0, :t, :h] = x[0].numpy()
        yp = np.full((1, tp), -100.0, dtype=np.float32)
        yp[0, :s] = y[0].numpy()
        (got,) = sess.run(["logits"], {"x": xp, "y": yp, "train_size": np.array([s], dtype=np.int64),
                                       "d": np.array([h], dtype=np.int64), "n_rows": np.array([t], dtype=np.int64)})
        worst = max(worst, float(np.abs(got[0, :t, 0] - ref).max()))
    return {"worst": worst, "ok": worst < TOL, "cases": [list(c) for c in cases]}
