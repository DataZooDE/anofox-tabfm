"""S-M4 -- performance reality check: MLX on Metal vs ORT on this Mac's CPU.

Run before S-M3 rather than after it. S-M3 costs a day of mlx-c work whose only
justification is that MLX is meaningfully faster here; measuring first means
that day is spent on evidence rather than hope. The plan's own instruction --
"measure before promising", earned by ROCm's 25-minute compile -- points the
same way.

Three questions:
  1. Warm per-predict latency across the row ladder, MLX vs ORT CPU.
  2. First-call cost. MLX is lazy and JIT-compiles Metal kernels, so the first
     call at a new shape pays for compilation. If that is large, the shape
     bucket + precompile machinery the project already has must generalize.
  3. Peak memory. The quantile embedding allocates rows x 999 x features, which
     grows faster than anything in the transformer and sets the real ceiling.

Run:  uv run sm4_bench [--max-rows N] [--ort-budget-s S]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from .common import MITRA_CACHE, compare, make_workload
from .mitra_mlx import MitraMLX, Weights

ROW_LADDER = [100, 250, 500, 1000, 2500]
FEATURES = 20


def _mark(key: str, value: object) -> None:
    print(f"SM4_{key}={value}", flush=True)


def _time_mlx(model: MitraMLX, w, repeats: int) -> tuple[float, float, np.ndarray]:
    """Returns (first_call_s, best_warm_s, logits)."""
    x, y = mx.array(w.x), mx.array(w.y)
    t0 = time.perf_counter()
    out = model(x, y, w.train_size, w.d)
    mx.eval(out)
    first = time.perf_counter() - t0

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = model(x, y, w.train_size, w.d)
        mx.eval(out)  # lazy: without this we time graph construction, not compute
        best = min(best, time.perf_counter() - t0)
    return first, best, np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=2500)
    ap.add_argument("--ort-budget-s", type=float, default=90.0,
                    help="stop timing ORT once a single call exceeds this")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    print("=== S-M4: MLX (Metal) vs ORT (CPU) on this Mac ===", flush=True)
    from .ort_ref import OrtReference

    w_st = Weights(MITRA_CACHE["classification"] / "model.safetensors")
    model = MitraMLX(w_st, task="classification", n_heads=4, fast_sdpa=True)
    ref = OrtReference("classification")
    _mark("ORT_SERVED_BY", ref.served_by)
    _mark("MLX_DEVICE", mx.default_device())

    ort_alive = True
    for rows in ROW_LADDER:
        if rows > args.max_rows:
            break
        train_size = rows // 2
        wl = make_workload(t=rows, h=FEATURES, train_size=train_size, d=FEATURES, seed=rows)

        mx.reset_peak_memory()
        first, warm, mlx_logits = _time_mlx(model, wl, args.repeats)
        peak_gb = mx.get_peak_memory() / 2**30

        if ort_alive:
            t0 = time.perf_counter()
            ort_logits = ref.run(wl)
            ort_s = time.perf_counter() - t0
            c = compare(ort_logits[:, train_size:, :], mlx_logits[:, train_size:, :])
            # Speed is only interesting if the answer is still the same one.
            _mark(f"R{rows}_PARITY", f"{c.describe()}")
            _mark(f"R{rows}", f"mlx_warm={warm * 1000:.0f}ms mlx_first={first * 1000:.0f}ms "
                              f"ort={ort_s * 1000:.0f}ms speedup={ort_s / warm:.1f}x "
                              f"mlx_peak={peak_gb:.2f}GB")
            if ort_s > args.ort_budget_s:
                ort_alive = False
                _mark("ORT_DROPPED_AT_ROWS", rows)
        else:
            _mark(f"R{rows}", f"mlx_warm={warm * 1000:.0f}ms mlx_first={first * 1000:.0f}ms "
                              f"ort=skipped mlx_peak={peak_gb:.2f}GB")

    # --- first-call cost at an unseen shape, isolated ------------------------
    # If MLX's per-shape compile is expensive, tabfm_gpu_precompile has to grow
    # an MLX path. Measured on a shape nothing above touched.
    wl = make_workload(t=333, h=17, train_size=200, d=17, seed=333)
    first, warm, _ = _time_mlx(model, wl, args.repeats)
    _mark("UNSEEN_SHAPE_FIRST_MS", f"{first * 1000:.0f}")
    _mark("UNSEEN_SHAPE_WARM_MS", f"{warm * 1000:.0f}")
    _mark("UNSEEN_SHAPE_COMPILE_OVERHEAD_MS", f"{(first - warm) * 1000:.0f}")
    _mark("RESULT", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
