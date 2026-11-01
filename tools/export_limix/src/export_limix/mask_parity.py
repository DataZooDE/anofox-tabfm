"""The gate for the masked (ROCm) contract: padded + masked must equal the positional CPU forward.

``compare`` runs the positional ``ExportWrapper`` on the UNPADDED table and the ``MaskExportWrapper`` on the
same table padded to a larger (T, H) bucket the way the MIGraphX plugin pads it (zeros, the -100 label
sentinel), and compares every real row -- context (fitted) rows included. ``NEGATIVE_CONTROLS`` break one
mask each; every one must be CAUGHT, or a pass would mean nothing.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from export_limix import limix_mask
from export_limix.limix_mask import MaskExportWrapper
from export_limix.limix_patches import ExportWrapper

# (T, H, S, T_pad, H_pad, constant_column): odd and even widths, one feature, S = 2, an unpadded bucket, a
# constant column (the valid-feature test), and the plugin's smallest real bucket (128 x 16)
CASES = ((50, 7, 30, 128, 16, 2), (31, 1, 30, 64, 16, None), (12, 5, 2, 128, 16, None),
         (20, 16, 12, 20, 16, 0), (40, 8, 25, 128, 16, None), (100, 8, 70, 128, 16, 5))
TOL = 1e-3   # the decimal tokenizer amplifies float32 noise near zero (docs/REAL_MODELS.md)


def _inputs(task, t, h, s, constant, seed):
    """A random table whose context-standardised values all keep 0.1 away from zero. The decimal tokenizer
    gives an exact zero its own token and is chaotic in 1/|z| near it: on a draw with |z| = 2e-4 the
    POSITIONAL forward alone moves by 8e-3 under 1e-6 input noise, which says nothing about the masking
    (fixture.assert_well_conditioned applies the same rule to the golden inputs)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g) * 2 + 0.5
    if constant is not None:
        x[0, :, constant] = 1.5
    for _ in range(50):   # nudge the near-zero cells out to |z| = 0.2; the statistics move a little each pass
        mean, std = x[0, :s].mean(0), x[0, :s].std(0)
        safe = torch.where(std > 0, std, torch.ones_like(std))
        z = (x[0] - mean) / safe
        near = (z.abs() < 0.1) & (std > 0)
        if not near.any():
            break
        x[0] = torch.where(near, mean + torch.where(z < 0, -0.2, 0.2) * safe, x[0])
    else:
        raise RuntimeError("could not condition the random table")
    if task == "classification":
        y = torch.randint(0, 3, (1, s), generator=g).float()
    else:
        y = torch.randn(1, s, generator=g) * 5 + 3
    return x, y


def _padded(x, y, tp, hp):
    t, h, s = x.shape[1], x.shape[2], y.shape[1]
    xp = torch.zeros(1, tp, hp)
    xp[0, :t, :h] = x[0]
    yp = torch.full((1, tp), -100.0)
    yp[0, :s] = y[0]
    return xp, yp, torch.tensor([s]), torch.tensor([h]), torch.tensor([t])


def compare(model, task, cases=CASES, *, seed=0) -> float:
    worst = 0.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(task, t, h, s, constant, seed + i)
        with torch.no_grad():
            ref = ExportWrapper(model, task=task)(x, y)[0]
            got = MaskExportWrapper(model, task=task)(*_padded(x, y, tp, hp))[0, :t]
        worst = max(worst, float((got - ref).abs().max()))
    return worst


@contextlib.contextmanager
def _patched(name, replacement):
    original = getattr(limix_mask, name)
    setattr(limix_mask, name, replacement)
    try:
        yield
    finally:
        setattr(limix_mask, name, original)


def _wrap_forward(change):
    orig = limix_mask.masked_forward

    def forward(model, task, x, y, train_size, d, n_rows):
        return orig(model, task, x, y, *change(x, train_size, d, n_rows))
    return _patched("masked_forward", forward)


def _wrap_layer(change):
    orig = limix_mask.masked_layer

    def layer(lyr, x, *, rows, ctx_key_mask, group_key_mask):
        kw = change(dict(rows=rows, ctx_key_mask=ctx_key_mask, group_key_mask=group_key_mask))
        return orig(lyr, x, **kw)
    return _patched("masked_layer", layer)


def _preprocess(change):
    orig = limix_mask.preprocess
    return _patched("preprocess", lambda model, xg, ctx, real: orig(model, xg, *change(ctx, real)))


NEGATIVE_CONTROLS = {
    "feature statistics over every row, not the context":
        lambda: _preprocess(lambda ctx, real: (torch.ones_like(ctx), real)),
    "valid-feature test over the padded rows too (n_rows = padded T)":
        lambda: _wrap_forward(lambda x, s, d, n: (s, d, torch.tensor([x.shape[1]]))),
    "padded feature groups kept (d = padded H)":
        lambda: _wrap_forward(lambda x, s, d, n: (s, torch.tensor([x.shape[2]]), n)),
    "no key mask in the sample attention":
        lambda: _wrap_layer(lambda kw: {**kw, "ctx_key_mask": torch.ones_like(kw["ctx_key_mask"])}),
}


def gate(model, task) -> dict:
    worst = compare(model, task)
    caught = {}
    for name, control in NEGATIVE_CONTROLS.items():
        with control():
            caught[name] = compare(model, task) > 10 * TOL
    return {"worst": worst, "ok": worst < TOL and all(caught.values()), "controls_caught": caught}


def check_onnx(graph_path, model, task, cases=CASES) -> dict:
    """ORT on the exported masked graph (padded inputs) vs the positional torch wrapper (unpadded)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(graph_path), opts, providers=["CPUExecutionProvider"])
    worst = 0.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(task, t, h, s, constant, i)
        with torch.no_grad():
            ref = ExportWrapper(model, task=task)(x, y)[0].numpy()
        xp, yp, ts, dd, nr = _padded(x, y, tp, hp)
        (got,) = sess.run(["logits"], {"x": xp.numpy(), "y": yp.numpy(), "train_size": ts.numpy(),
                                       "d": dd.numpy(), "n_rows": nr.numpy()})
        worst = max(worst, float(np.abs(got[0, :t] - ref).max()))
    return {"worst": worst, "ok": worst < TOL, "cases": [list(c) for c in cases]}


def ort_session_with_weights(graph_path, tensor_map_path, safetensors_path):
    """An ORT session on a weight-free graph with the checkpoint injected by its tensor map -- what the engine
    does at load time -- so a shipped graph can be compared on real weights without the extension."""
    import json

    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    from safetensors.numpy import load_file

    proto = onnx.load(str(graph_path), load_external_data=False)
    tmap = json.load(open(tensor_map_path))
    inits, transforms = tmap["initializers"], tmap.get("transforms", {})
    weights = load_file(str(safetensors_path))
    for i, init in enumerate(proto.graph.initializer):
        key = inits.get(init.name)
        if key is None:
            continue
        arr = weights[key]
        if transforms.get(init.name) == "transpose":
            arr = np.ascontiguousarray(arr.T)
        proto.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), init.name))
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(proto.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def check_shipped(mask_graph, mask_map, positional_graph, positional_map, safetensors_path, model, task,
                  cases=CASES) -> dict:
    """The masked graph (padded) against the SHIPPED positional graph (unpadded), both in ORT on the real
    weights, plus against the eager positional wrapper on the same weights."""
    mask_sess = ort_session_with_weights(mask_graph, mask_map, safetensors_path)
    pos_sess = ort_session_with_weights(positional_graph, positional_map, safetensors_path)
    worst_graph, worst_eager, scale = 0.0, 0.0, 1.0
    for i, (t, h, s, tp, hp, constant) in enumerate(cases):
        x, y = _inputs(task, t, h, s, constant, i)
        xp, yp, ts, dd, nr = _padded(x, y, tp, hp)
        (got,) = mask_sess.run(["logits"], {"x": xp.numpy(), "y": yp.numpy(), "train_size": ts.numpy(),
                                            "d": dd.numpy(), "n_rows": nr.numpy()})
        (pos,) = pos_sess.run(["logits"], {"x": x.numpy(), "y": y.numpy()})
        with torch.no_grad():
            eager = ExportWrapper(model, task=task)(x, y)[0].numpy()
        worst_graph = max(worst_graph, float(np.abs(got[0, :t] - pos[0]).max()))
        worst_eager = max(worst_eager, float(np.abs(got[0, :t] - eager).max()))
        scale = max(scale, float(np.abs(eager).max()))
    # Regression outputs are in the target's raw units (|y| up to ~15 here), and ORT vs eager on the SAME
    # positional graph already differs by ~1e-3 there (the tokenizer's float32 noise), so the budget scales.
    tol = TOL * (scale if task == "regression" else 1.0)
    return {"worst_vs_shipped_graph": worst_graph, "worst_vs_eager": worst_eager, "tol": tol,
            "ok": worst_graph < tol and worst_eager < tol}
