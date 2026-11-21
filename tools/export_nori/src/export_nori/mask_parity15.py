"""The gate for the Nori 1.5 masked (ROCm) contract: padded + masked must equal the positional CPU forward.

Same shape as ``mask_parity`` (Nori-30M): ``compare`` runs the positional ``ExportWrapper`` on the UNPADDED table
and ``MaskExportWrapper15`` on the same table padded to a larger (T, H) bucket (zeros, the -100 label sentinel).
``NEGATIVE_CONTROLS`` break one mask or one statistic each, with the 1.5-specific places a leak can hide (the PLE
edges, the column / row attentions, the feature attentions and their QASS lengths); every one must be CAUGHT, or a
pass would mean nothing.
"""

from __future__ import annotations

import contextlib

import torch

from export_nori import nori15_mask
from export_nori.mask_parity import CASES, TOL, _inputs, check_onnx  # noqa: F401  (check_onnx: same I/O contract)
from export_nori.nori15_mask import MaskExportWrapper15
from export_nori.nori_patches import ExportWrapper


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
            got = MaskExportWrapper15(model)(xp, yp, torch.tensor([s]), torch.tensor([h]),
                                             torch.tensor([t]))[0, :t, 0]
        worst = max(worst, float((got - ref).abs().max()))
    return worst


@contextlib.contextmanager
def _patched(**replacements):
    originals = {name: getattr(nori15_mask, name) for name in replacements}
    for name, fn in replacements.items():
        setattr(nori15_mask, name, fn)
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(nori15_mask, name, fn)


def _layers(change):
    """Patch both layer functions: ``change(kwargs) -> kwargs``."""
    orig_cell, orig_row = nori15_mask.cell_layer, nori15_mask.row_layer
    return _patched(
        cell_layer=lambda layer, cells, targets, **kw: orig_cell(layer, cells, targets, **change(kw)),
        row_layer=lambda layer, rows, memory, **kw: orig_row(layer, rows, memory, **change(kw)))


def _stats_over_every_row():
    orig = nori15_mask.preprocess
    return _patched(preprocess=lambda model, xg, ctx: orig(model, xg, torch.ones_like(ctx)))


def _ple_over_every_row():
    orig = nori15_mask.ple_edges
    return _patched(ple_edges=lambda ple, x, valid: orig(ple, x, torch.isfinite(x)))


def _all_ones(n):
    return torch.ones(n, dtype=torch.bool)


def _row_keys_leak():
    """The row attentions (target stream and phase B) also read the QUERY copy and the padded rows."""
    return _patched(row_keys=lambda nrm, ctx: (nrm, _all_ones(nrm.shape[1])))


def _col_keys_leak():
    return _patched(col_keys=lambda cols, ctx: (cols, _all_ones(cols.shape[1])))


def _col_keys_unmasked():
    return _patched(col_keys=lambda cols, ctx: (cols[:, : ctx.shape[0]], torch.ones_like(ctx)))


def _log(n):
    return torch.log(torch.tensor([float(n)]))


NEGATIVE_CONTROLS = {
    "feature statistics over every row, not the context": _stats_over_every_row,
    "non-context rows enter the PLE edges": _ple_over_every_row,
    "query rows leak into the row attention": _row_keys_leak,
    "query rows leak into the column attention": _col_keys_leak,
    "no key mask in the column attention": _col_keys_unmasked,
    "padded columns kept in the feature attentions":
        lambda: _layers(lambda kw: {**kw, "col_mask": torch.ones_like(kw["col_mask"])}),
    "QASSMax sample length from the padded shape":
        lambda: _layers(lambda kw: {**kw, "s_factor": _log(kw["ctx_mask"].numel())}),
    "QASSMax feature length from the padded width":
        lambda: _layers(lambda kw: {**kw, "f_read": _log(kw["col_mask"].numel()),
                                    **({"f_cells": _log(kw["col_mask"].numel() + 2)} if "f_cells" in kw else {})}),
}


def gate(model) -> dict:
    worst = compare(model)
    caught = {}
    for name, control in NEGATIVE_CONTROLS.items():
        with control():
            caught[name] = compare(model) > 10 * TOL
    return {"worst": worst, "ok": worst < TOL and all(caught.values()), "controls_caught": caught}
