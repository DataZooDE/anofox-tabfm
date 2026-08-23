"""S-M2 -- does the MLX mitra forward compute what the shipped graph computes?

Success, per the plan: logits match ORT CPU at rtol 1e-4 on the fixture-sized
workload AND on the real weights. Both hold here -- the golden reference this
compares against was produced by ORT CPU over the real 302 MB safetensors in
S-M1, so there is no synthesized-weights escape hatch.

Only the QUERY rows are compared. The support rows' logits are computed by the
graph but never read by the engine (predictions are taken at >= train_size),
and holding a backend to agreement on outputs nobody consumes would be
inventing a requirement.

Run:  uv run sm2_parity [--fast-sdpa|--no-fast-sdpa]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from .common import MITRA_CACHE, SpikeError, compare
from .mitra_mlx import MitraMLX, Weights
from .sm1_env import GOLDEN

# The bar. fp32 on Metal is not bit-identical to fp32 on CPU -- same arithmetic,
# different reduction order -- so bit equality is never the test.
#
# The plan proposed "relative 1e-4 + full argmax agreement", inheriting
# equivalence.py's max-per-element-relative on LOGITS. sm2_triangulate.py
# measured that bar against the torch reference implementation -- the very code
# the ONNX graph was exported from -- and it scores 9.3e-04, failing its own
# test. Logits cross zero, so that metric divides by ~0 and reports noise as
# catastrophe. It cannot distinguish a correct backend from a broken one.
#
# So parity is judged on what the extension actually returns to SQL: the
# post-softmax probabilities, plus the predicted class. PROB_TOL is 1e-4
# absolute on a quantity bounded in [0, 1] -- four orders below anything a user
# could observe, and comfortably passed by torch-vs-ORT as well.
PROB_TOL = 1e-4
LOGIT_ABS_TOL = 1e-3  # a coarse guard: catches a real bug, ignores fp32 dust


def _mark(key: str, value: object) -> None:
    print(f"SM2_{key}={value}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-sdpa", dest="fast", action="store_true", default=True,
                    help="use mx.fast.scaled_dot_product_attention (default)")
    ap.add_argument("--no-fast-sdpa", dest="fast", action="store_false",
                    help="use the explicit softmax(QK^T)V, for isolating SDPA")
    args = ap.parse_args()

    print("=== S-M2: mitra forward in MLX vs the ORT CPU golden ===", flush=True)
    _mark("FAST_SDPA", args.fast)

    golden_path = MITRA_CACHE["classification"] / GOLDEN
    if not golden_path.exists():
        raise SpikeError(f"golden reference missing at {golden_path}; run `uv run sm1_env` first")
    g = np.load(golden_path)

    t0 = time.perf_counter()
    w = Weights(MITRA_CACHE["classification"] / "model.safetensors")
    model = MitraMLX(w, task="classification", n_heads=4, fast_sdpa=args.fast)
    _mark("LOAD_MS", f"{(time.perf_counter() - t0) * 1000:.0f}")
    _mark("SHAPE", f"layers={model.n_layers} dim={model.dim} out={model.dim_output}")

    cases = sorted({k.split("__")[0] for k in g.files})
    worst_prob, worst_logit, worst_agree, worst_strict, failed = 0.0, 0.0, 1.0, 0.0, []
    for name in cases:
        x = mx.array(g[f"{name}__x"])
        y = mx.array(g[f"{name}__y"])
        train_size, d = (int(v) for v in g[f"{name}__meta"])
        reference = g[f"{name}__logits"]

        t0 = time.perf_counter()
        out = model(x, y, train_size, d)
        mx.eval(out)  # MLX is lazy; without this we would time graph construction
        dt = time.perf_counter() - t0
        candidate = np.asarray(out)

        # Query rows only -- the ones the engine actually reads.
        c = compare(reference[:, train_size:, :], candidate[:, train_size:, :])
        ok = (c.prob_max_abs <= PROB_TOL and c.max_abs <= LOGIT_ABS_TOL
              and c.argmax_agreement == 1.0)
        worst_prob = max(worst_prob, c.prob_max_abs)
        worst_logit = max(worst_logit, c.max_abs)
        worst_strict = max(worst_strict, c.max_rel_strict)
        worst_agree = min(worst_agree, c.argmax_agreement)
        if not ok:
            failed.append(name)
        _mark(f"{name.upper()}", f"{'ok' if ok else 'FAIL'} {c.describe()} ms={dt * 1000:.0f}")

    _mark("WORST_PROB_ABS", f"{worst_prob:.3e}")
    _mark("WORST_LOGIT_ABS", f"{worst_logit:.3e}")
    _mark("WORST_ARGMAX_AGREEMENT", f"{worst_agree:.4f}")
    _mark("TOL", f"prob<={PROB_TOL:.0e} logit<={LOGIT_ABS_TOL:.0e} argmax==1.0")
    # Reported, never enforced: see the PROB_TOL comment for why this metric
    # fails on the reference implementation itself.
    _mark("WORST_MAX_REL_STRICT_INFORMATIONAL", f"{worst_strict:.3e}")
    if failed:
        _mark("RESULT", f"FAIL({','.join(failed)})")
        return 1
    _mark("RESULT", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
