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
    # (T, B, O) since the head now runs over every data row (it used to be
    # (T-S, B, O) with the wrapper zero-padding the context rows afterwards).
    out = model(x_src=x, y_src=y_prefix, num_features=num_features)
    return out.transpose(0, 1)[:, train_size:, :]  # (B, T-S, O), the query rows


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
    """The whole gate, by default.

    This used to run `compare()` alone, and the `__main__` block sat ABOVE
    `self_test` and `compare_wrappers` in the file — so `python -m
    export_tabdpt.mask_parity`, the invocation this module's own docstring
    documents, exited before either was defined and silently ran only the
    positive check. A gate whose negative controls are unreachable from its
    documented entry point is the same failure it exists to prevent, one level
    up: it reports PASS without having tried to fail.

    Now: controls first (and a non-zero exit if any is NOT caught), then the
    model-level comparison, then the wrapper-level one for BOTH tasks — the
    only check that the regression y-standardisation does not leak query
    labels into the scale of every prediction.
    """
    ap = argparse.ArgumentParser(prog="mask_parity")
    ap.add_argument("--config", default="fixture", choices=["fixture", "real"])
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-compare", action="store_true",
                    help="just the model-level comparison; skips the controls and the "
                         "wrapper checks (narrowing flag, not the default)")
    args = ap.parse_args(argv)

    if args.only_compare:
        return compare(args.config, args.rows, args.features, args.train_frac, args.seed)

    rc = self_test()
    if rc != 0:
        return rc
    for task in ("classification", "regression"):
        rc |= compare_wrappers(args.config, task, args.rows, args.features,
                               args.train_frac, args.seed)
    return rc




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

    saved_neg_inf = mp.NEG_INF
    original_rows = mp._row_mask_for
    original_beta = mp.masked_get_scale_param

    # try/finally: a control that raises must not leave the module patched for
    # the positive comparison that follows, which would then "pass" against a
    # deliberately broken conversion.
    try:
        mp.NEG_INF = 0.0  # queries may attend to queries
        controls.append(("attention mask neutralised", compare(*shape)))
        mp.NEG_INF = saved_neg_inf

        mp._row_mask_for = lambda data, ctx: _torch.ones_like(data, dtype=bool)
        controls.append(("normalisation over all rows", compare(*shape)))
        mp._row_mask_for = original_rows

        mp.masked_get_scale_param = lambda layer, ts, *, device, dtype: original_beta(
            layer, _torch.tensor([shape[1]]), device=device, dtype=dtype)
        controls.append(("scale param given T, not train_size", compare(*shape)))
    finally:
        mp.NEG_INF = saved_neg_inf
        mp._row_mask_for = original_rows
        mp.masked_get_scale_param = original_beta

    blind = [name for name, rc in controls if rc == 0]
    for name, rc in controls:
        print(f"control: {name:38} {'caught' if rc else 'NOT CAUGHT'}")
    if blind:
        print(f"GATE IS BLIND to: {', '.join(blind)}")
        return 2
    return compare(*shape)


def compare_wrappers(cfg_name: str, task: str, rows: int, features: int,
                     train_frac: float, seed: int) -> int:
    """Wrapper-to-wrapper parity — the level that actually gets exported.

    compare() checks the model forward. This checks the thing torch.onnx.export
    is handed, which additionally owns feature padding, the regression target
    standardisation and the output layout. The regression path is the reason
    this exists separately: the positional wrapper standardises y over its
    train PREFIX, and the masked one must do it over the context rows of a
    full-length y. Getting that from the full y instead would leak query
    labels into the scale of every regression prediction — and would still
    look perfectly reasonable in classification, where y_src is untouched.
    """
    from export_tabdpt.tabdpt_mask_patches import build_mask_wrapper
    from export_tabdpt.tabdpt_patches import build_wrapper

    cfg = getattr(configs, cfg_name)()
    apply_positional()
    model = build_model(cfg, seed=seed)

    positional = build_wrapper(model, task)
    masked = build_mask_wrapper(model, task)  # SAME weights, by construction

    torch.manual_seed(seed + 1)
    train_size = max(2, int(rows * train_frac))
    x = torch.randn(1, rows, features)
    if task == "classification":
        y_full = torch.randint(0, max(2, cfg.max_classes), (1, rows)).float()
    else:
        y_full = torch.randn(1, rows) * 3.0 + 1.0
    y_full[:, train_size:] = -999.0  # poison; the mask must make it irrelevant

    with torch.no_grad():
        want = positional(x, y_full[:, :train_size])           # (1, T, C), head zero-padded
        got = masked(x, y_full,
                     torch.tensor([train_size], dtype=torch.long),
                     torch.tensor([features], dtype=torch.long))

    # The positional wrapper zero-pads rows < train_size; compare where the
    # engine actually reads.
    want_q = want[:, train_size:, :]
    got_q = got[:, train_size:, :]
    if want_q.shape != got_q.shape:
        print(f"[{task}] SHAPE MISMATCH positional={tuple(want_q.shape)} masked={tuple(got_q.shape)}")
        return 2

    max_abs = (want_q - got_q).abs().max().item()
    scale = want_q.abs().max().clamp(min=1e-6).item()
    rel = max_abs / scale
    ok = rel < 1e-4
    print(f"[{task}] rows={rows} train_size={train_size} shape={tuple(want_q.shape)} "
          f"rel={rel:.3e} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    # At EOF on purpose: everything main() calls must already be defined.
    sys.exit(main())
