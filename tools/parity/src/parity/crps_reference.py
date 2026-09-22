"""PSR-01 CRPS reference: compute and print golden CRPS values.

Documents the exact crps_bar() function used to derive the C++ test golden values
in test/cpp/test_tabfm_scoring.cpp and test/sql/tabfm_scoring.test.

crps_bar(y, logits, borders) computes the analytical CRPS over a piecewise-uniform
bar distribution:
  1. Apply softmax to logits → per-bin probabilities
  2. For each bin [b_i, b_{i+1}]: compute the integral of (F(t) - 1{t>=y})^2 dt
     using the closed-form expressions from 03-RESEARCH.md §Exact CRPS Math
     (Case A: y < b_i, Case B: y > b_{i+1}, Case C: y inside the bin).
  3. Sum contributions over all K bins.

Assumption A1 (03-RESEARCH.md): integrates only over [b_0, b_K].
y outside this range contributes via Case A/B of every bin — no half-normal tail extension.

Non-uniform bin widths (b_{i+1} - b_i) are used throughout. NEVER assumes uniform bins.

Usage:
    cd tools/parity
    uv run python -m parity.crps_reference

Outputs two reference values:
  1. Synthetic K=4 uniform case: logits=[0,0,0,0], borders=[0,1,2,3,4], y=1.5
     Hand-verifiable anchor for the C++ unit test.
  2. Fixture K=16 case: logits from golden.json row 0, raw_borders, y = first training target.
     Used by test/sql/tabfm_scoring.test golden assertion.

These values are hard-coded (with a comment referencing this file) in the tests.

Spec ref: PSR-01, 03-RESEARCH.md §Exact CRPS Math, §Golden Testing Strategy
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Sequence

# Repo root relative to this file: tools/parity/src/parity/crps_reference.py
_THIS_DIR = pathlib.Path(__file__).parent
_DEFAULT_FIXTURE_DIR = _THIS_DIR.parents[3] / "test" / "fixtures" / "tabpfn_v2"


def _softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax (subtract max before exp)."""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def crps_bar(y: float, logits: Sequence[float], borders: Sequence[float]) -> float:
    """CRPS over bar distribution with piecewise-uniform density.

    Implements the analytical closed-form from 03-RESEARCH.md §Exact CRPS Math.
    Identical logic to ComputeCRPS() in src/tabfm_scoring.cpp (which operates on
    already-softmaxed probs; this function accepts raw logits and softmaxes them).

    Parameters
    ----------
    y : float
        Observed outcome.
    logits : Sequence[float]
        Pre-softmax logits (K values, one per bin).
    borders : Sequence[float]
        Bin edges (K+1 values, strictly increasing).

    Returns
    -------
    float
        CRPS value (>= 0, lower is better). Returns 0.0 on degenerate input.

    Assumption A1: integrates only over [b_0, b_K] (no half-normal tail extension).
    """
    probs = _softmax(list(logits))
    K = len(probs)
    if K == 0 or len(borders) != K + 1:
        return 0.0

    crps = 0.0
    cum = 0.0  # C_i: cumulative probability up to left edge of bin i

    for i in range(K):
        b_lo = borders[i]
        b_hi = borders[i + 1]
        wi = b_hi - b_lo
        pi = probs[i]

        # T-03-02: skip zero/negative-width bins (no divide-by-zero in Case C)
        if wi <= 0:
            cum += pi
            continue

        if y < b_lo:
            # Case A: y is to the LEFT of this bin; indicator = 1 throughout
            # integral = w_i * ((C_i - 1)^2 + (C_i - 1) * p_i + p_i^2 / 3)
            a = cum - 1.0
            crps += wi * (a * a + a * pi + pi * pi / 3)

        elif y > b_hi:
            # Case B: y is to the RIGHT of this bin; indicator = 0 throughout
            # integral = w_i * (C_i^2 + C_i * p_i + p_i^2 / 3)
            crps += wi * (cum * cum + cum * pi + pi * pi / 3)

        else:
            # Case C: y is INSIDE this bin [b_lo, b_hi]
            d = y - b_lo       # distance from left edge
            e = b_hi - y       # distance to right edge
            c = pi / wi        # density in this bin

            # Left part [b_lo, y]: indicator = 0
            # integral of (C_i + c*s)^2 ds from 0 to d:
            #   = C_i^2*d + C_i*c*d^2 + c^2*d^3/3
            L = cum * cum * d + cum * c * d * d + c * c * d * d * d / 3

            # Right part [y, b_hi]: indicator = 1
            # F(y) = C_i + p_i * d / w_i  (the CDF at y)
            # a2 = F(y) - 1 (negative, since F(y) < 1 inside any bin)
            fy = cum + pi * d / wi
            a2 = fy - 1.0
            R = a2 * a2 * e + a2 * c * e * e + c * c * e * e * e / 3

            crps += L + R

        cum += pi

    return crps


def main() -> None:
    """Compute and print the golden CRPS reference values for the C++ unit test."""

    # ── Synthetic K=4 case (hand-verifiable, uniform bins) ──────────────────
    #
    # logits=[0,0,0,0] → probs=[0.25, 0.25, 0.25, 0.25]
    # borders=[0,1,2,3,4], y=1.5 (falls in bin 1)
    #
    # Expected CRPS = 0.395833... (derived in RESEARCH.md, verified by hand):
    #   bin 0 [0,1]: Case B → 1*(0 + 0 + 0.25^2/3) = 0.020833
    #   bin 1 [1,2]: Case C d=0.5,e=0.5 → L+R = 0.208333
    #   bin 2 [2,3]: Case A → 1*((-0.5)^2 + (-0.5)*0.25 + 0.25^2/3) = 0.145833
    #   bin 3 [3,4]: Case A → 1*((-0.25)^2 + (-0.25)*0.25 + 0.25^2/3) = 0.020833
    #   total = 0.395833...

    logits_k4  = [0.0, 0.0, 0.0, 0.0]
    borders_k4 = [0.0, 1.0, 2.0, 3.0, 4.0]

    val_inside = crps_bar(1.5, logits_k4, borders_k4)
    val_left   = crps_bar(-1.0, logits_k4, borders_k4)

    print("=== Synthetic K=4 reference values ===")
    print(f"crps_bar(y=1.5,  logits=[0,0,0,0], borders=[0,1,2,3,4]) = {val_inside!r}")
    print(f"crps_bar(y=-1.0, logits=[0,0,0,0], borders=[0,1,2,3,4]) = {val_left!r}")

    # ── Fixture K=16 case (golden.json) ─────────────────────────────────────
    #
    # Load golden.json and compute CRPS for:
    #   - Each of the 4 training rows (where actual y is available)
    # using raw_borders (y-space) and the per-row logits.
    #
    # This is the reference for the SQL aggregate test in tabfm_scoring.test,
    # which computes tabfm_crps over the 4 non-NULL rows in dist_data.

    fixture_path = _DEFAULT_FIXTURE_DIR / "golden.json"
    if not fixture_path.exists():
        print(f"\n[SKIP] Fixture not found: {fixture_path}")
        return

    with fixture_path.open() as f:
        golden = json.load(f)

    raw_borders  = golden["raw_borders"]   # K+1=17 borders in y-space
    all_logits   = golden["logits"]         # 6 rows, K=16 per row
    all_y        = golden["inputs"]["y"][0] # 6 rows; first 4 are training targets
    train_size   = golden["inputs"]["train_size"]  # 4

    print(f"\n=== Fixture K=16 reference values (train_size={train_size}) ===")
    crps_vals = []
    for i in range(train_size):
        y_i = all_y[i]
        crps_i = crps_bar(y_i, all_logits[i], raw_borders)
        crps_vals.append(crps_i)
        print(f"  row {i}: y={y_i:.6f}, CRPS={crps_i:.15g}")

    mean_crps = sum(crps_vals) / len(crps_vals)
    print(f"\nMean CRPS over {train_size} training rows: {mean_crps:.15g}")
    print(f"round(mean_crps, 6) = {round(mean_crps, 6)}")
    print(f"\n[tabfm_scoring.test] Use: round(tabfm_crps(actual, dist), 6) == {round(mean_crps, 6)}")


if __name__ == "__main__":
    main()
