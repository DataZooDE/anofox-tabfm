"""The gate for the masked (ROCm) contract: padded + masked must equal the positional CPU forward.

``compare`` runs the positional ``ExportWrapper`` on the UNPADDED table and the ``MaskExportWrapper`` on the
same table padded to a larger (T, H) bucket, the way the MIGraphX plugin pads it (zeros, the -100 label
sentinel), and compares every real row. ``CASES`` covers a constant column, a single query row, H = 1 and
S = 2. ``NEGATIVE_CONTROLS`` break one mask each; every one must be CAUGHT, or a pass would mean nothing.

There is deliberately no control for ``n_rows``: it only feeds the feature encoder's constant-column test,
and for finite inputs (the engine imputes before the graph) that test is inert -- a column constant over
the real rows normalises to exactly 0 whether or not it is "retained", and its non-finite code is 0 either
way. Measured: forcing n_rows to the padded T changes nothing. It stays in the contract because it is the
correct mask for the test upstream writes, should a non-finite value ever reach the graph.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from export_exaone import exaone_mask
from export_exaone.exaone_mask import MaskExportWrapper
from export_exaone.exaone_patches import ExportWrapper

# (T, H, S, T_pad, H_pad, constant_column)
CASES = ((50, 7, 30, 128, 16, 2), (31, 1, 30, 64, 16, None), (12, 5, 2, 128, 16, None), (20, 16, 12, 20, 16, 0))
TOL = {"classification": 1e-4, "regression": 2e-4}


def _inputs(task, t, h, s, constant, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g) * 2 + 0.5
    if constant is not None:
        x[0, :, constant] = 1.5
    if task == "classification":
        # case 0 uses the ids 1..3 (no class 0): the engine always sends dense 0..k-1, but only a label set
        # without 0 lets the label-statistics control see the off-context rows' zeros
        offset = 1.0 if seed % len(CASES) == 0 else 0.0
        y = ((torch.arange(s) % 3).float() + offset)[torch.randperm(s, generator=g)].unsqueeze(0)
    else:
        y = torch.randn(1, s, generator=g) * 5 + 3
    return x, y


def compare(model, task, cases=CASES, *, seed=0) -> float:
    worst = 0.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(task, t, h, s, constant, seed + i)
        with torch.no_grad():
            ref = ExportWrapper(model, task)(x, y)[0]
            xp = torch.zeros(1, tp, hp)
            xp[0, :t, :h] = x[0]
            yp = torch.full((1, tp), -100.0)
            yp[0, :s] = y[0]
            got = MaskExportWrapper(model, task)(xp, yp, torch.tensor([s]), torch.tensor([h]), torch.tensor([t]))[0]
        c = 4 if task == "classification" else 1
        worst = max(worst, float((got[:t, :c] - ref[:, :c]).abs().max()))
    return worst


@contextlib.contextmanager
def _patched(name, replacement):
    original = getattr(exaone_mask, name)
    setattr(exaone_mask, name, replacement)
    try:
        yield
    finally:
        setattr(exaone_mask, name, original)


@contextlib.contextmanager
def _label_statistics_over_every_row():
    """Both guards off: off-context labels are not NaN-ed AND the label encoder counts every first-copy row
    as support (either guard alone is enough, which is why each is not a control of its own)."""
    orig = exaone_mask.masked_label_encoder

    def encoder(enc, labels, ctx, *, regression):
        first_copy = torch.arange(ctx.shape[0]) < ctx.shape[0] // 2
        return orig(enc, labels, first_copy, regression=regression)
    with _patched("support_only", lambda labels, ctx: labels), _patched("masked_label_encoder", encoder):
        yield


def _no_feature_mask():
    orig = exaone_mask.masked_model_forward

    def forward(model, x, labels, train_size, d, n_rows, *, regression):
        return orig(model, x, labels, train_size, torch.tensor([x.shape[2]]), n_rows, regression=regression)
    return _patched("masked_model_forward", forward)


def _no_key_bias():
    orig = exaone_mask.masked_attention

    def attention(mod, query, context=None, *, key_mask=None, log_len=None, first_context_head=False):
        return orig(mod, query, context, key_mask=None, log_len=log_len, first_context_head=first_context_head)
    return _patched("masked_attention", attention)


def _padded_ssmax_length():
    orig = exaone_mask.masked_attention

    def attention(mod, query, context=None, *, key_mask=None, log_len=None, first_context_head=False):
        return orig(mod, query, context, key_mask=key_mask, log_len=None, first_context_head=first_context_head)
    return _patched("masked_attention", attention)


NEGATIVE_CONTROLS = {
    "label statistics over every row, not the context": _label_statistics_over_every_row,
    "padded columns kept (d = padded H)": _no_feature_mask,
    "no key mask in any attention": _no_key_bias,
    "SSMax length from the padded shape": _padded_ssmax_length,
}


def gate(model, task) -> dict:
    """Pass only if the real comparison is within TOL AND every negative control is caught."""
    worst = compare(model, task)
    caught = {}
    for name, control in NEGATIVE_CONTROLS.items():
        with control():
            caught[name] = compare(model, task) > 10 * TOL[task]
    return {"worst": worst, "ok": worst < TOL[task] and all(caught.values()), "controls_caught": caught}


def check_onnx(graph_path, model, task, cases=CASES) -> dict:
    """ORT on the exported masked graph (padded inputs) vs the positional torch wrapper (unpadded)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(task, t, h, s, constant, i)
        with torch.no_grad():
            ref = ExportWrapper(model, task)(x, y)[0].numpy()
        xp = np.zeros((1, tp, hp), dtype=np.float32)
        xp[0, :t, :h] = x[0].numpy()
        yp = np.full((1, tp), -100.0, dtype=np.float32)
        yp[0, :s] = y[0].numpy()
        (got,) = sess.run(["logits"], {"x": xp, "y": yp, "train_size": np.array([s], dtype=np.int64),
                                       "d": np.array([h], dtype=np.int64), "n_rows": np.array([t], dtype=np.int64)})
        c = 4 if task == "classification" else 1
        worst = max(worst, float(np.abs(got[0, :t, :c] - ref[:, :c]).max()))
    return {"worst": worst, "ok": worst < TOL[task], "cases": [list(c) for c in cases]}
