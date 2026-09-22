"""Does the masked TabDPT compute the same thing as the positional one?

This is the gate the whole spike turns on, and it runs in pure PyTorch before
any ONNX is produced. The conversion claims to be an algebraic rewrite — every
slice replaced by a masked reduction over the same elements — so on identical
weights and identical inputs the two must agree to floating-point noise. If
they do not, the conversion is wrong, and a wrong graph that returns
plausible-looking probabilities is the worst failure this repository knows how
to produce (a content-hashed .mxr key and every *_SERVED_BY assertion exist
because of one).

Run:
    uv run python -m export_tabdpt.mask_parity --config fixture
    uv run python -m export_tabdpt.mask_parity --config real --rows 200
"""

from __future__ import annotations

import argparse
import sys

import torch

from export_tabdpt import configs
from export_tabdpt.tabdpt_mask_patches import masked_model_forward
from export_tabdpt.tabdpt_patches import apply as apply_positional


def build_model(cfg, seed: int = 0, wake_residuals: bool = True):
    """Random-init TabDPTModel — with its residual paths woken up.

    Upstream zero-initialises out_proj.weight, q_gate.weight and ff.down.weight
    (model.py, TransformerEncoderLayer.reset_parameters). That is a sound
    training trick, and it makes a random-init model USELESS as a parity
    fixture: every layer's residual is identically zero, so the entire
    attention block cannot influence the output whatever it computes.

    This was not theoretical. With those weights left at zero, deliberately
    neutralising the attention mask (NEG_INF -> 0, i.e. letting every query row
    attend to every other query row) still produced bit-exact parity. The gate
    was silently blind to the single most important part of the conversion, and
    would have "verified" a broken attention rewrite.

    So the zeroed projections are filled with small random values before
    comparing. Both paths see the same weights, so this cannot manufacture
    agreement — it can only expose disagreement that was previously invisible.
    """
    from tabdpt.model import TabDPTModel

    torch.manual_seed(seed)
    model = TabDPTModel(**cfg.model_kwargs).eval()
    if wake_residuals:
        torch.manual_seed(seed + 977)
        woken = 0
        for module in model.modules():
            for attr in ("out_proj", "q_gate"):
                sub = getattr(module, attr, None)
                if sub is not None and hasattr(sub, "weight") and float(sub.weight.abs().sum()) == 0.0:
                    torch.nn.init.normal_(sub.weight, std=0.02)
                    woken += 1
            ff = getattr(module, "ff", None)
            down = getattr(ff, "down", None) if ff is not None else None
            if down is not None and hasattr(down, "weight") and float(down.weight.abs().sum()) == 0.0:
                torch.nn.init.normal_(down.weight, std=0.02)
                woken += 1
        if woken == 0:
            raise RuntimeError("expected zero-initialised residual projections to wake; found none — "
                               "upstream init changed and this gate may be blind again")
    return model


def positional_logits(model, x: torch.Tensor, y_full: torch.Tensor, train_size: int) -> torch.Tensor:
    """What the shipped (single_eval_pos) path computes.

    y is the TRAIN PREFIX only, and the model reads the split from its length.
    """
    y_prefix = y_full[:, :train_size]
    num_features = torch.tensor([model.num_features], dtype=torch.long)
    out = model(x_src=x, y_src=y_prefix, num_features=num_features)  # (T-S, B, O)
    return out.transpose(0, 1)  # (B, T-S, O)


def masked_logits(model, x: torch.Tensor, y_full: torch.Tensor, train_size: int) -> torch.Tensor:
    """What the converted path computes: full y, split as a value."""
    ts = torch.tensor([train_size], dtype=torch.long)
    out = masked_model_forward(model, x, y_full, ts)  # (T, B, O)
    return out.transpose(0, 1)  # (B, T, O)


def compare(cfg_name: str, rows: int, features: int, train_frac: float, seed: int) -> int:
    cfg = getattr(configs, cfg_name)()
    apply_positional()  # the shipped export patches; masked path is separate
    model = build_model(cfg, seed=seed)

    torch.manual_seed(seed + 1)
    train_size = max(2, int(rows * train_frac))
    # Pad features to the model's fixed width, exactly as ExportWrapper does
    # before either path runs. Feeding the raw width instead makes the
    # positional path die inside the encoder, which is a harness bug, not a
    # finding about the conversion.
    x = torch.randn(1, rows, features)
    x = torch.nn.functional.pad(x, (0, model.num_features - x.shape[2]))
    # Dense class ids for the context; the query positions carry a placeholder
    # the mask must make irrelevant — if it does not, this test fails, which is
    # exactly what it is for.
    y_full = torch.randint(0, max(2, cfg.model_kwargs.get("n_out", 2)), (1, rows)).float()
    y_full[:, train_size:] = -999.0  # poison: must never reach the answer

    with torch.no_grad():
        want = positional_logits(model, x, y_full, train_size)   # (1, T-S, O)
        got_all = masked_logits(model, x, y_full, train_size)    # (1, T, O)
    got = got_all[:, train_size:, :]                             # engine's slice

    if want.shape != got.shape:
        print(f"SHAPE MISMATCH positional={tuple(want.shape)} masked={tuple(got.shape)}")
        return 2

    diff = (want - got).abs()
    max_abs = diff.max().item()
    scale = want.abs().max().clamp(min=1e-6).item()
    rel = max_abs / scale

    print(f"config={cfg_name} rows={rows} features={features} train_size={train_size}")
    print(f"  shape          {tuple(want.shape)}")
    print(f"  max |diff|     {max_abs:.3e}")
    print(f"  relative       {rel:.3e}  (vs max |positional| = {scale:.3e})")

    # Tolerance reasoning: the two paths do the SAME arithmetic in a different
    # order (masked reductions over T elements where zeros contribute nothing,
    # versus dense reductions over S), so differences are reassociation noise
    # in fp32, not algorithmic. 1e-4 relative is loose enough for that and far
    # tighter than any real disagreement would be -- a mask that leaked the
    # poisoned query labels would move this by whole units.
    ok = rel < 1e-4
    print(f"  VERDICT        {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mask_parity")
    ap.add_argument("--config", default="fixture", choices=["fixture", "real"])
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return compare(args.config, args.rows, args.features, args.train_frac, args.seed)


if __name__ == "__main__":
    sys.exit(main())


def self_test(argv=None) -> int:
    """Prove the gate can fail, then prove the conversion passes.

    A parity check that cannot fail is worse than none: it converts "untested"
    into "verified" without touching the code in between. This runs three
    negative controls, one per independent part of the conversion, and requires
    each to be caught before reporting the real result.
    """
    import torch as _torch

    from export_tabdpt import tabdpt_mask_patches as mp

    shape = ("fixture", 64, 8, 0.75, 0)
    controls = []

    mp.NEG_INF = 0.0  # queries may attend to queries
    controls.append(("attention mask neutralised", compare(*shape)))
    mp.NEG_INF = -1e30

    original_rows = mp._row_mask_for
    mp._row_mask_for = lambda data, ctx: _torch.ones_like(data, dtype=bool)
    controls.append(("normalisation over all rows", compare(*shape)))
    mp._row_mask_for = original_rows

    original_beta = mp.masked_get_scale_param
    mp.masked_get_scale_param = lambda layer, ts, *, device, dtype: original_beta(
        layer, _torch.tensor([shape[1]]), device=device, dtype=dtype)
    controls.append(("scale param given T, not train_size", compare(*shape)))
    mp.masked_get_scale_param = original_beta

    blind = [name for name, rc in controls if rc == 0]
    for name, rc in controls:
        print(f"control: {name:38} {'caught' if rc else 'NOT CAUGHT'}")
    if blind:
        print(f"GATE IS BLIND to: {', '.join(blind)}")
        return 2
    return compare(*shape)
