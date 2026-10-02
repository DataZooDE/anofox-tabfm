"""Does the masked WHOLE model compute what the unpadded model computes, on padded input?

    uv run python -m export_causilo.mask_parity_full

The reference is the shipped positional `ExportWrapper` on the UNPADDED table (itself checked against
upstream's `ModelRunner` and `Normalizer` in tests/test_export.py). The candidate is the masked wrapper on the
same data padded up to a larger (T rows, H features) bucket, with the split, the real row count and the real
feature count passed as VALUES. They must agree on every real row, context and query reported separately.

A gate that cannot fail is worse than none, and this codebase has been blinded by that repeatedly: a model
that ignores its input (Causilo does at default initialisation, measured: logits vary by 0.008 across rows),
a comparison against the patch itself, a check run on synthetic arrays instead of the situation it guards.
So:

  * the model is `fixture.sensitive()`, responsive to its labels and features and well-conditioned;
  * `fixture_r` runs every loop the released model runs, so no branch executes once only;
  * every mechanism has a negative control that must be CAUGHT before a pass counts.
"""

from __future__ import annotations

import sys

import torch

from export_causilo import causilo_mask_forward as cmf
from export_causilo import configs
from export_causilo.causilo_mask_export import MaskExportWrapper
from export_causilo.causilo_patches import ExportWrapper, apply, build_model
from export_causilo.fixture import sensitive

CONFIGS = ("fixture", "fixture_r")
TASKS = ("classification", "regression")
SEED = 1337
TOL = 1e-4

# (real rows n, train rows N, real features d, bucket rows T, bucket features H, data flavour)
#   flavours: None | "nan" (a missing cell) | "const" (a constant real column) | "outlier" (4-sigma tails)
CASES = (
    (20, 12, 5, 32, 16, None),        # d not a multiple of 3, H not either
    (40, 30, 9, 64, 16, None),        # d a multiple of 3
    (24, 10, 4, 48, 32, "const"),     # a real constant column
    (24, 10, 4, 48, 32, "nan"),       # a missing cell: NaN must survive to the missing-value embedding
    (30, 20, 6, 64, 18, "outlier"),   # the 4-sigma bounds and the arcsinh tails
    (16, 15, 3, 32, 16, None),        # ONE query row
    (12, 8, 2, 32, 16, None),         # d = 2
    (10, 6, 1, 32, 16, None),         # d = 1
    (14, 1, 5, 32, 16, None),         # ONE training row: ddof and the std floor
    (9, 2, 7, 16, 24, None),          # two training rows, three real groups in eight bucket groups
    (12, 12, 4, 16, 12, None),        # no query rows at all
)


def _data(task, n, N, d, flavour, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g) * (0.5 + 3 * torch.rand(d, generator=g)) + torch.randn(d, generator=g)
    if flavour == "const":
        x[:, d - 1] = 3.5
    elif flavour == "nan":
        x[3, 0] = float("nan")
        x[min(n - 1, N + 1), min(1, d - 1)] = float("nan")     # a QUERY-row cell when there is one
    elif flavour == "outlier":
        x[2, 0], x[5, 1] = 80.0, -60.0
    y = (torch.randint(0, 3, (N,), generator=g).float() if task == "classification"
         else torch.randn(N, generator=g) * 3.0 + 1.0)
    return x, y


def _models(cfg_name, task):
    apply()
    cfg = configs.get(cfg_name)
    return sensitive(build_model(task, cfg, seed=SEED)).eval()


def _run_pair(model, task, case, seed=0):
    n, N, d, T, H, flavour = case
    x, y = _data(task, n, N, d, flavour, seed)
    with torch.no_grad():
        ref = ExportWrapper(model)(x[None], y[None])[0]                      # [n, C]
        xp = torch.zeros(1, T, H)
        xp[0, :n, :d] = x
        yp = torch.full((1, T), -100.0)
        yp[0, :N] = y
        got = MaskExportWrapper(model)(xp, yp, torch.tensor([N]), torch.tensor([n]),
                                       torch.tensor([d]))[0, :n]
    return ref, got


def _rel(ref, got):
    both_nan = torch.isnan(ref) & torch.isnan(got)
    diff = (ref - got).masked_fill(both_nan, 0.0).abs()
    return (diff.max() / ref.abs().nan_to_num().max().clamp(min=1e-6)).item()


def _all_cases(cfg_name, task):
    model = _models(cfg_name, task)
    worst, rows = 0.0, []
    for case in CASES:
        ref, got = _run_pair(model, task, case)
        if ref.shape != got.shape:
            return 9.9, [(case, f"SHAPE {tuple(ref.shape)} vs {tuple(got.shape)}")]
        N = case[1]
        r = _rel(ref, got)
        rows.append((case, r, _rel(ref[:N], got[:N]), _rel(ref[N:], got[N:]) if ref.shape[0] > N else 0.0))
        worst = max(worst, r)
    return worst, rows


def _controls() -> int:
    """Each must be CAUGHT on every config and task, or the gate is blind to that mechanism."""
    seams = {n: getattr(cmf, n) for n in
             ("_row_keep", "_group_keep", "_n_keys_rows", "_n_keys_groups", "_stat_rows", "_inject_keep")}
    saved_neg = cmf.NEG_INF

    def padded_rows(train):                     # the key count reads the PADDED row count, as upstream's shape would
        return torch.tensor(float(train.shape[0]))

    def padded_groups(group_keep, latents):     # the key count reads the PADDED group count
        return torch.tensor(float(group_keep.shape[0] + latents))

    controls = [
        ("row key mask off (non-train rows are keys)",
         lambda: setattr(cmf, "_row_keep", lambda train: torch.ones_like(train)),
         lambda: setattr(cmf, "_row_keep", seams["_row_keep"])),
        ("group key mask off (padded feature groups are keys)",
         lambda: setattr(cmf, "_group_keep", lambda G, d, size, dev: torch.ones(G, dtype=torch.bool, device=dev)),
         lambda: setattr(cmf, "_group_keep", seams["_group_keep"])),
        ("row key count = padded rows (QueryScale log n)",
         lambda: setattr(cmf, "_n_keys_rows", padded_rows),
         lambda: setattr(cmf, "_n_keys_rows", seams["_n_keys_rows"])),
        ("group key count = padded groups (QueryScale log n)",
         lambda: setattr(cmf, "_n_keys_groups", padded_groups),
         lambda: setattr(cmf, "_n_keys_groups", seams["_n_keys_groups"])),
        ("normaliser statistics over all rows",
         lambda: setattr(cmf, "_stat_rows", lambda train: torch.ones_like(train)),
         lambda: setattr(cmf, "_stat_rows", seams["_stat_rows"])),
        ("target embedding added to non-train rows",
         lambda: setattr(cmf, "_inject_keep", lambda train: torch.ones_like(train)),
         lambda: setattr(cmf, "_inject_keep", seams["_inject_keep"])),
        ("every key bias neutralised",
         lambda: setattr(cmf, "NEG_INF", 0.0),
         lambda: setattr(cmf, "NEG_INF", saved_neg)),
    ]

    blind = []
    for name, install, restore in controls:
        for cfg_name in CONFIGS:
            for task in TASKS:
                install()
                try:
                    worst = _all_cases(cfg_name, task)[0]
                finally:
                    restore()
                caught = worst >= TOL * 10
                print(f"control: {name:54s} {cfg_name:9s} {task[:5]} rel={worst:.3e} "
                      f"{'caught' if caught else 'NOT CAUGHT'}", flush=True)
                if not caught:
                    blind.append(f"{name} [{cfg_name}/{task}]")
    if blind:
        print("GATE IS BLIND to:\n  " + "\n  ".join(blind))
        return 2
    return 0


def main() -> int:
    rc = _controls()
    if rc:
        return rc
    ok = True
    for cfg_name in CONFIGS:
        for task in TASKS:
            worst, rows = _all_cases(cfg_name, task)
            good = worst < TOL
            ok &= good
            print(f"{cfg_name:9s} {task:14s} worst={worst:.3e} {'PASS' if good else 'FAIL'}")
            for row in rows:
                if isinstance(row[1], str):
                    print("    ", row)
                    continue
                case, r, rc_, rq = row
                n, N, d, T, H, fl = case
                print(f"      n={n:3d} N={N:3d} d={d:2d} bucket=({T},{H}) {str(fl):8s} "
                      f"all={r:.2e} ctx={rc_:.2e} qry={rq:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
