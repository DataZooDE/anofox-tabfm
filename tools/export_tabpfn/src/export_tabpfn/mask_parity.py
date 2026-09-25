"""Does the masked AlongColumnAttention compute what upstream computes?

Run:  uv run python -m export_tabpfn.mask_parity

TabPFN is the hard case for the train_size conversion. Its attention is not
uniform: train rows get full multi-head attention, test rows get multi-query
(the first key-value head only), and nothing attends to a test row. Replacing
that with a mask means computing BOTH and selecting per row, so unlike TabDPT
this is not a pure algebraic rewrite and deserves its own gate.

The gate proves it can fail before it is believed. Upstream zero-initialises
`out_projection`, so a freshly constructed layer returns identically zero and
any comparison passes vacuously -- that has now happened in two different
architectures in this codebase, so the weights are woken explicitly and the
negative controls are mandatory.
"""

from __future__ import annotations

import sys

import torch

import tabpfn.architectures.tabpfn_v2_6 as v26

from export_tabpfn import tabpfn_mask_patches as mp

#: Bound ONCE at import, before anything can install the patch.
#:
#: The gate compares upstream's forward against the masked one. If the patch has
#: already been applied in-process -- which apply_mask_patches does by design --
#: then looking up AlongColumnAttention.forward at comparison time returns the
#: PATCH, and the gate compares the patch against itself: a tautology that
#: reports 0.0 and proves nothing. That is the same class of blindness as the
#: zero-initialised weights below, arriving by a different route.
_UPSTREAM_FORWARD = v26.AlongColumnAttention.forward

E, H, D = 32, 4, 8
SHAPES = ((16, 12), (40, 30), (64, 8))  # last one: mostly test rows


def _layer(seed: int = 0):
    torch.manual_seed(seed)
    layer = v26.AlongColumnAttention(embedding_size=E, num_heads=H, head_dim=D,
                                     device=torch.device("cpu"), dtype=torch.float32).eval()
    # Without this the layer outputs zero for every input and the comparison
    # below cannot distinguish a correct mask from no mask at all.
    if float(layer.out_projection.weight.abs().sum()) != 0.0:
        raise RuntimeError("expected out_projection to be zero-initialised; upstream changed, so re-check "
                           "whether this gate still needs waking — and whether it is still blind without it")
    torch.nn.init.normal_(layer.out_projection.weight, std=0.02)
    return layer


def _relative(layer, rows: int, train_size: int) -> float:
    torch.manual_seed(rows)
    x = torch.randn(1, rows, E)
    with torch.no_grad():
        want, _ = _UPSTREAM_FORWARD(layer, x, train_size)
        mp.set_train_size(torch.tensor([train_size], dtype=torch.int64))
        # Pass the value through as well as stashing it, so the patch's
        # stash-vs-caller consistency check is actually exercised here rather
        # than only in production.
        got, _ = mp._patched_along_column_forward(layer, x, train_size)
    return (want - got).abs().max().item() / max(want.abs().max().item(), 1e-6)


def main() -> int:
    layer = _layer()
    ok = True

    controls = []
    saved = mp.NEG_INF
    mp.NEG_INF = 0.0
    controls.append(("key bias neutralised", _relative(layer, 40, 30)))
    mp.NEG_INF = saved

    saved_mask = mp.context_row_mask
    mp.context_row_mask = lambda r, ts, dev: torch.ones(r, dtype=torch.bool, device=dev)
    controls.append(("row mask all-true (no MQA split)", _relative(layer, 40, 30)))
    mp.context_row_mask = saved_mask

    for name, rel in controls:
        caught = rel > 1e-5
        print(f"control: {name:36} rel={rel:.3e} {'caught' if caught else 'NOT CAUGHT'}")
        ok &= caught
    if not ok:
        print("GATE IS BLIND — it would pass a broken conversion")
        return 2

    for rows, train_size in SHAPES:
        rel = _relative(layer, rows, train_size)
        good = rel < 1e-5
        print(f"rows={rows:3} train_size={train_size:3} rel={rel:.3e} {'PASS' if good else 'FAIL'}")
        ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
