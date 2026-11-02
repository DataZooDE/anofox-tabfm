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

from export_tabicl import tabicl_mask
from export_tabicl.tabicl_mask import MaskExportWrapper
from export_tabicl.tabicl_patches import ExportWrapper

# (T, H, S, T_pad, H_pad): widths 1..16 (the grouping wraps differently at every d < 2**(size-1)), S = 2,
# an unpadded bucket, and the plugin's smallest real bucket (128 x 16)
CASES = ((50, 7, 30, 128, 16), (31, 1, 30, 64, 16), (12, 5, 2, 128, 16), (20, 16, 12, 20, 16),
         (40, 3, 25, 128, 16), (100, 8, 70, 128, 16), (24, 2, 10, 32, 8))
TOL = 1e-3


def sensitive_model(task: str, seed: int = 0):
    """A random fixture-sized TabICL that RESPONDS to its input. At TabICL's own init the output barely
    depends on the table (QASSMax's query MLP is zero-initialised and the attention is near-uniform), so
    every negative control passes unnoticed: the gate would mean nothing. Weights drawn at 2/sqrt(fan_in)
    make every mask visible (the real weights are, see test_real_weights.py)."""
    from export_tabicl import configs
    from export_tabicl.tabicl_patches import build_model

    model = build_model(task, configs.fixture().model_kwargs, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2 or "query_mlp" in name or "base_mlp" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 2.0 / max(1, p.shape[-1]) ** 0.5)
    return model


def _inputs(model, t, h, s, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, t, h, generator=g)
    if model.max_classes > 0:
        y = torch.randint(0, min(3, model.max_classes), (1, s), generator=g).float()
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


def compare(model, cases=CASES, *, seed=0) -> float:
    model.train()
    worst = 0.0
    for i, (t, h, s, tp, hp) in enumerate(cases):
        x, y = _inputs(model, t, h, s, seed + i)
        with torch.no_grad():
            ref = ExportWrapper(model)(x, y)[0]
            got = MaskExportWrapper(model)(*_padded(x, y, tp, hp))[0, :t]
        worst = max(worst, float((got - ref).abs().max()))
    return worst


@contextlib.contextmanager
def _patched(name, replacement):
    original = getattr(tabicl_mask, name)
    setattr(tabicl_mask, name, replacement)
    try:
        yield
    finally:
        setattr(tabicl_mask, name, original)


def _isab_keys_unmasked():
    orig = tabicl_mask._isab
    return _patched("_isab", lambda block, src, key_bias: orig(block, src, torch.zeros_like(key_bias)))


def _icl_keys_unmasked():
    orig = tabicl_mask.icl
    return _patched("icl", lambda m, reps, y, ctx, key_bias: orig(m, reps, y, ctx, torch.zeros_like(key_bias)))


def _padded_width_as_d():
    orig = tabicl_mask.masked_forward
    return _patched("masked_forward", lambda model, x, y, s, d: orig(model, x, y, s, torch.tensor([x.shape[2]])))


def _ssmax_from_padded_length():
    orig = tabicl_mask.ssmax_length
    return _patched("ssmax_length", lambda log_s: orig(torch.zeros_like(log_s) + float(np.log(128.0))))


def _labels_on_every_row():
    orig = tabicl_mask.icl
    return _patched("icl", lambda m, reps, y, ctx, key_bias: orig(m, reps, y, torch.ones_like(ctx), key_bias))


NEGATIVE_CONTROLS = {
    "inducing points attend to every row, not the context": _isab_keys_unmasked,
    "ICL keys not masked to the context": _icl_keys_unmasked,
    "padded feature columns kept (d = padded H)": _padded_width_as_d,
    "QASSMax length from the padded shape": _ssmax_from_padded_length,
    "label embedding added to every row": _labels_on_every_row,
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

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(graph_path), opts, providers=["CPUExecutionProvider"])
    model.train()
    worst = 0.0
    for i, (t, h, s, tp, hp) in enumerate(cases):
        x, y = _inputs(model, t, h, s, i)
        with torch.no_grad():
            ref = ExportWrapper(model)(x, y)[0].numpy()
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


def check_shipped(mask_graph, mask_map, positional_graph, positional_map, safetensors_path, model,
                  cases=CASES) -> dict:
    """The masked graph (padded) against the SHIPPED positional graph (unpadded), both in ORT on the real
    weights, plus against the eager positional wrapper on the same weights."""
    mask_sess = ort_session_with_weights(mask_graph, mask_map, safetensors_path)
    pos_sess = ort_session_with_weights(positional_graph, positional_map, safetensors_path)
    model.train()
    worst_graph, worst_eager = 0.0, 0.0
    for i, (t, h, s, tp, hp) in enumerate(cases):
        x, y = _inputs(model, t, h, s, i)
        xp, yp, ts, dd, nr = _padded(x, y, tp, hp)
        (got,) = mask_sess.run(["logits"], {"x": xp.numpy(), "y": yp.numpy(), "train_size": ts.numpy(),
                                            "d": dd.numpy(), "n_rows": nr.numpy()})
        (pos,) = pos_sess.run(["logits"], {"x": x.numpy(), "y": y.numpy()})
        with torch.no_grad():
            eager = ExportWrapper(model)(x, y)[0].numpy()
        worst_graph = max(worst_graph, float(np.abs(got[0, :t] - pos[0]).max()))
        worst_eager = max(worst_eager, float(np.abs(got[0, :t] - eager).max()))
    return {"worst_vs_shipped_graph": worst_graph, "worst_vs_eager": worst_eager,
            "ok": worst_graph < TOL and worst_eager < TOL}
