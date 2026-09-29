"""Does the masked WHOLE model compute what upstream computes, on padded input?

    uv run python -m export_tabpfn.mask_parity_full

`mask_parity` proved one attention layer. This is the gate for everything the
conversion touches: preprocessing statistics, the group mask, thinking rows, the
fitted-values routing. The reference is upstream's own ExportWrapper on the
UNPADDED data; the candidate is the masked wrapper on the same data padded up to
a larger bucket. They must agree on every real row -- context and query -- to
float noise.

A gate that cannot fail is worse than none, and this codebase has now been
blinded three ways: zero-initialised projections (a random model computes
nothing), a comparison against the patch itself, and a check validated against
synthetic arrays instead of the situation it runs in. So:

  * every zero-initialised parameter is woken;
  * upstream's forwards are captured at IMPORT, before any patch exists, and
    swapped in for the reference -- never looked up at comparison time;
  * each negative control must be caught before a pass counts.
"""

from __future__ import annotations

import sys

import torch

import tabpfn.architectures.tabpfn_v2 as v2
import tabpfn.architectures.tabpfn_v2_5 as v25
import tabpfn.architectures.tabpfn_v2_6 as v26

from tabpfn.preprocessing.torch import ops as opsmod

from export_tabpfn import configs
from export_tabpfn import tabpfn_mask_forward as mf
from export_tabpfn.tabpfn_mask_export import MaskExportWrapper
from export_tabpfn.tabpfn_patched import (ExportWrapper, apply_module_patches,
                                          build_random_model, prepare_model_for_export)

#: Captured before anything installs a patch.
_UPSTREAM = {(m, k): getattr(m, k).forward
             for m in (v2, v25, v26) for k in ("AlongColumnAttention", "AlongRowAttention")}

#: Upstream's own select_features, captured before apply_module_patches replaces
#: it. The exporter's patched one keeps a constant column (moved to the back, value
#: intact) where upstream REMOVES it and pads the group with zeros. The two agree on
#: every input the runtime feeds -- it filters constant columns itself -- so the
#: patched one is fine for shipping, but it is not a reference for "what upstream
#: computes" when a constant column is present. This is.
_TRUE_SELECT = opsmod.select_features

ARCHES = (("v2", "fixture"), ("v2.5", "fixture25"), ("v2.6", "fixture26"))
TASKS = ("classification", "regression")

# (real rows, train rows, real features, bucket rows, bucket features, constant col?)
CASES = (
    (20, 12, 5, 32, 16, False),
    (40, 30, 9, 64, 16, False),
    (24, 10, 4, 48, 32, "const"),      # a real constant column: removed only if constancy is over REAL rows
    (24, 10, 4, 48, 32, "nanconst"),   # constant only after imputation: needs the real row count in normalisation
    (16, 15, 3, 32, 16, False),        # one query row
)
TOL = 1e-4


def _restore_upstream():
    for (mod, name), fwd in _UPSTREAM.items():
        getattr(mod, name).forward = fwd
    opsmod.select_features = v2.select_features = v25.select_features = v26.select_features = _TRUE_SELECT


def _use_patched_select():
    for arch in ("v2", "v2.5", "v2.6"):
        apply_module_patches(arch)


def _wake(model):
    """Upstream zero-initialises several projections; a random model would
    otherwise compute nothing and every comparison would pass vacuously."""
    torch.manual_seed(977)
    woken = 0
    for p in model.parameters():
        if float(p.abs().sum()) == 0.0:
            torch.nn.init.normal_(p, std=0.02)
            woken += 1
    if woken == 0:
        raise RuntimeError("no zero-initialised parameters found; upstream changed and this "
                           "gate may be blind in a way it no longer knows about")
    return woken


def _model(arch, cfgname, task):
    cfg = configs.get(cfgname, task=task)
    apply_module_patches(cfg.arch)
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    m = build_random_model(task, kw, seed=0, arch=cfg.arch)
    m = prepare_model_for_export(m, arch=cfg.arch, allow_synthetic_pos_base=True)
    _wake(m)
    return m, cfg


def _data(task, n, N, d, const, classes, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    if const == "const":
        x[:, d - 1] = 3.5                       # a real, constant, NON-zero column
    elif const == "nanconst":
        # Not constant on its face (a NaN), constant once imputed. It survives the
        # step-1 filter, then reads as constant in normalisation, which changes the
        # group's used-feature count and so every OTHER column's scale.
        #
        # Column 1, NOT the last: with features_per_group = 3 the last real column
        # of a d=4 input sits ALONE in its group, where clip(min=1) makes the used
        # count identical whether or not it is flagged constant -- the case would
        # exist and exercise nothing. Column 1 shares group 0 with two varying ones.
        x[:, 1] = 2.0
        x[5, 1] = float("nan")
    if n > 4:
        x[3, 0] = float("nan")                  # a missing cell
    if task == "classification":
        y = torch.randint(0, classes, (N,), generator=g).float()
    else:
        y = torch.randn(N, generator=g) * 3.0 + 1.0
    return x, y


def _run_pair(model, cfg, task, case, seed=0):
    n, N, d, Tp, Hp, const = case
    x, y = _data(task, n, N, d, const, min(cfg.max_classes, 3), seed)

    # reference: upstream, upstream forwards, unpadded
    _restore_upstream()
    ref_w = ExportWrapper(model, task).eval()
    with torch.no_grad():
        ref = ref_w(x.unsqueeze(0), y.unsqueeze(0))[0]              # [n, C]

    # candidate: masked, patched forwards, padded
    mf.apply_mask_patches()
    _use_patched_select()
    xp = torch.zeros(1, Tp, Hp)
    xp[0, :n, :d] = x
    yp = torch.full((1, Tp), -100.0)
    yp[0, :N] = y
    cand_w = MaskExportWrapper(model, task).eval()
    with torch.no_grad():
        got = cand_w(xp, yp, torch.tensor([N]), torch.tensor([n]), torch.tensor([d]))[0, :n]
    _restore_upstream()
    return ref, got


def _rel(ref, got):
    return ((ref - got).abs().max() / ref.abs().max().clamp(min=1e-6)).item()


def _all_cases(arch, cfgname, task):
    model, cfg = _model(arch, cfgname, task)
    worst, rows = 0.0, []
    for case in CASES:
        ref, got = _run_pair(model, cfg, task, case)
        if ref.shape != got.shape:
            return 9.9, [(case, f"SHAPE {tuple(ref.shape)} vs {tuple(got.shape)}")]
        r = _rel(ref, got)
        n, N = case[0], case[1]
        rows.append((case, r, _rel(ref[:N], got[:N]), _rel(ref[N:], got[N:])))
        worst = max(worst, r)
    return worst, rows


def _controls() -> int:
    """Each must be CAUGHT on every architecture it applies to, or the gate is blind.

    Run per architecture, not once: v2 has its own forward (constant detection per
    group slot, group count from `d`), so a control that only ever ran against
    v2.6 proves nothing about v2's code. This gate passed v2 on the first try,
    which is exactly when to check it could have failed.
    """
    import export_tabpfn.tabpfn_mask_patches as mp

    real_row = mf._patched_along_row_forward
    orig_nc = mf._nonconstant
    orig_keep = mf._keep_mask
    orig_norm = mf._masked_normalize_groups
    saved_neg = mp.NEG_INF

    def no_group_mask(self, x):
        saved = mf._GROUP_KEEP
        mf._GROUP_KEEP = torch.ones_like(saved)
        try:
            return real_row(self, x)
        finally:
            mf._GROUP_KEEP = saved

    def padded_nonconstant(x, real, n_real):
        return orig_nc(x, torch.ones_like(real), torch.tensor(real.shape[0]))

    # name, architectures it applies to, install, restore
    every = ("v2", "v2.5", "v2.6")
    later = ("v2.5", "v2.6")
    controls = [
        ("group mask neutralised", every,
         lambda: setattr(mf, "_patched_along_row_forward", no_group_mask),
         lambda: setattr(mf, "_patched_along_row_forward", real_row)),
        ("column attention mask neutralised", every,
         lambda: setattr(mp, "NEG_INF", 0.0),
         lambda: setattr(mp, "NEG_INF", saved_neg)),
        ("real-row counts ignored (padded height)", every,
         lambda: setattr(mf, "_nonconstant", padded_nonconstant),
         lambda: setattr(mf, "_nonconstant", orig_nc)),
        ("constant detection over padded rows", later,
         lambda: setattr(mf, "_keep_mask",
                         lambda x, real, n_real, d: orig_keep(x, torch.ones_like(real), n_real, d)),
         lambda: setattr(mf, "_keep_mask", orig_keep)),
    ]
    arch_cfg = {"v2": "fixture", "v2.5": "fixture25", "v2.6": "fixture26"}

    blind = []
    for name, archs, install, restore in controls:
        for arch in archs:
            install()
            try:
                worst = _all_cases(arch, arch_cfg[arch], "classification")[0]
            finally:
                restore()
            caught = worst >= TOL * 10
            print(f"control: {name:42} {arch:5s} rel={worst:.3e} {'caught' if caught else 'NOT CAUGHT'}")
            if not caught:
                blind.append(f"{name} [{arch}]")
    if blind:
        print(f"GATE IS BLIND to: {', '.join(blind)}")
        return 2
    return 0


def main() -> int:
    rc = _controls()
    if rc:
        return rc
    ok = True
    for arch, cfgname in ARCHES:
        for task in TASKS:
            worst, rows = _all_cases(arch, cfgname, task)
            good = worst < TOL
            ok &= good
            print(f"{arch:5s} {task:14s} worst={worst:.3e} {'PASS' if good else 'FAIL'}")
            for row in rows:
                if isinstance(row[1], str):
                    print("    ", row)
                    continue
                case, r, rc_, rq = row
                print(f"      n={case[0]:3d} N={case[1]:3d} d={case[2]:2d} bucket=({case[3]},{case[4]}) "
                      f"all={r:.2e} ctx={rc_:.2e} qry={rq:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
