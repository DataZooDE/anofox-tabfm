"""PSR-01/02/03 reference: compute and print golden CRPS, log-score, and interval-score values.

Documents the exact crps_bar(), log_score_bar(), and interval_score_bar() functions used to
derive the C++ test golden values in test/cpp/test_tabfm_scoring.cpp and
test/sql/tabfm_scoring.test.

crps_bar(y, logits, borders) computes the analytical CRPS over a piecewise-uniform
bar distribution:
  1. Apply softmax to logits → per-bin probabilities
  2. For each bin [b_i, b_{i+1}]: compute the integral of (F(t) - 1{t>=y})^2 dt
     using the closed-form expressions from 03-RESEARCH.md §Exact CRPS Math
     (Case A: y < b_i, Case B: y > b_{i+1}, Case C: y inside the bin).
  3. Sum contributions over all K bins.

log_score_bar(y, logits, borders) computes the negative log-likelihood:
  NLL = log(w_i) - log(max(clip_eps, p_i))  for y in bin i
  clip_eps = 1e-10 (Assumption A2, 03-RESEARCH.md §Log-Score Math)
  Out-of-support or zero-width bin → -log(clip_eps)

interval_score_bar(y, logits, borders, coverage=0.9) computes the Gneiting & Raftery (2007)
interval score:
  IS = (u - l) + (2/alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]
  alpha = 1 - coverage, l = q_{alpha/2}, u = q_{1-alpha/2}
  Quantiles computed via DistributionQuantile equivalent (CDF cumsum + linear interpolation).

Assumption A1 (03-RESEARCH.md): CRPS integrates only over [b_0, b_K].
Non-uniform bin widths (b_{i+1} - b_i) are used throughout. NEVER assumes uniform bins.

Usage:
    cd tools/parity
    uv run python -m parity.crps_reference

Spec ref: PSR-01, PSR-02, PSR-03, 03-RESEARCH.md §Exact CRPS Math, §Log-Score Math,
          §Interval Score Math, §Golden Testing Strategy
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


_LOG_SCORE_CLIP_EPS = 1e-10  # Assumption A2 (03-RESEARCH.md §Log-Score Math)


def log_score_bar(y: float, logits: Sequence[float], borders: Sequence[float]) -> float:
    """Log-score (negative log-likelihood) over a bar distribution.

    For y in bin i: NLL = log(w_i) - log(max(clip_eps, p_i))
    where clip_eps = 1e-10 (Assumption A2, 03-RESEARCH.md §Log-Score Math).

    Out-of-support (y outside [b_0, b_K]) or zero-width bin (w_i < 1e-300)
    returns the maximum penalty -log(clip_eps) to avoid log(0) or divide-by-zero
    (T-03-04 mitigation: DoS prevention for degenerate inputs).

    Parameters
    ----------
    y : float
        Observed outcome.
    logits : Sequence[float]
        Pre-softmax logits (K values).
    borders : Sequence[float]
        Bin edges (K+1 values, strictly increasing).

    Returns
    -------
    float
        Log-score (>= 0, lower is better — equals -log(clip_eps) at worst).

    Spec ref: PSR-02
    """
    probs = _softmax(list(logits))
    K = len(probs)
    if K == 0 or len(borders) != K + 1:
        return -math.log(_LOG_SCORE_CLIP_EPS)

    b_first = borders[0]
    b_last = borders[K]

    # T-03-04: out-of-support guard (Pitfall 7 in 03-RESEARCH.md)
    if y < b_first or y > b_last:
        return -math.log(_LOG_SCORE_CLIP_EPS)

    # Find the bin containing y
    for i in range(K):
        b_lo = borders[i]
        b_hi = borders[i + 1]
        if b_lo <= y <= b_hi:
            wi = b_hi - b_lo
            # T-03-04: zero-width bin guard (Pitfall 4 in 03-RESEARCH.md)
            if wi < 1e-300:
                return -math.log(_LOG_SCORE_CLIP_EPS)
            p_clipped = max(_LOG_SCORE_CLIP_EPS, probs[i])
            return math.log(wi) - math.log(p_clipped)

    # y exactly at borders[K] hits the last bin; shouldn't reach here normally
    return -math.log(_LOG_SCORE_CLIP_EPS)


def _distribution_quantile(probs: list[float], borders: Sequence[float], q: float) -> float:
    """Python equivalent of DistributionQuantile (tabfm_predict.hpp:206).

    CDF cumsum search + linear interpolation within the found bin.
    Matches the C++ implementation's non-uniform-bin algorithm.

    Parameters
    ----------
    probs : list[float]
        Normalized per-bin probabilities (K values, softmax-ed).
    borders : Sequence[float]
        Bin edges (K+1 values).
    q : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        The quantile value; clamped to [borders[0], borders[-1]].
    """
    K = len(probs)
    if K == 0 or len(borders) != K + 1:
        return float("nan")
    q = max(0.0, min(1.0, q))
    cum = 0.0
    for i in range(K):
        cum_next = cum + probs[i]
        if cum_next >= q:
            wi = borders[i + 1] - borders[i]
            if wi <= 0.0 or probs[i] <= 0.0:
                return borders[i]
            frac = (q - cum) / probs[i]
            return borders[i] + frac * wi
        cum = cum_next
    return borders[K]


def interval_score_bar(
    y: float,
    logits: Sequence[float],
    borders: Sequence[float],
    coverage: float = 0.9,
) -> float:
    """Interval score (Gneiting & Raftery 2007) over a bar distribution.

    IS = (u - l) + (2/alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]
    alpha = 1 - coverage
    l = q_{alpha/2},  u = q_{1-alpha/2}  via DistributionQuantile

    Python equivalent of ComputeIntervalScore in tabfm_scoring.cpp.

    Parameters
    ----------
    y : float
        Observed outcome.
    logits : Sequence[float]
        Pre-softmax logits (K values).
    borders : Sequence[float]
        Bin edges (K+1 values).
    coverage : float
        Nominal coverage level, default 0.9.  Must be in (0, 1).

    Returns
    -------
    float
        Interval score (>= 0, lower is better).

    Spec ref: PSR-03
    """
    assert 0.0 < coverage < 1.0, f"coverage must be in (0,1), got {coverage}"
    probs = _softmax(list(logits))
    alpha = 1.0 - coverage
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    l = _distribution_quantile(probs, borders, q_lo)
    u = _distribution_quantile(probs, borders, q_hi)
    penalty_lo = (l - y) if y < l else 0.0
    penalty_hi = (y - u) if y > u else 0.0
    return (u - l) + (2.0 / alpha) * (penalty_lo + penalty_hi)


def main() -> None:
    """Compute and print the golden CRPS, log-score, and interval-score reference values."""

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

    # PSR-02: log-score synthetic K=4
    ls_inside = log_score_bar(1.5, logits_k4, borders_k4)
    ls_oob    = log_score_bar(-1.0, logits_k4, borders_k4)
    print(f"\nlog_score_bar(y=1.5,  K=4 uniform): {ls_inside!r}")
    print(f"  = ln(4) = {math.log(4)!r}")
    print(f"log_score_bar(y=-1.0, K=4 uniform, out-of-support): {ls_oob!r}")
    print(f"  = -log(clip_eps) = {-math.log(_LOG_SCORE_CLIP_EPS)!r}")

    # PSR-03: interval-score synthetic K=4
    is_09 = interval_score_bar(1.5, logits_k4, borders_k4, 0.9)
    is_05 = interval_score_bar(1.5, logits_k4, borders_k4, 0.5)
    print(f"\ninterval_score_bar(y=1.5, K=4 uniform, coverage=0.9): {is_09!r}")
    print(f"interval_score_bar(y=1.5, K=4 uniform, coverage=0.5): {is_05!r}")

    # ── Fixture K=16 case (golden.json) ─────────────────────────────────────
    #
    # Load golden.json and compute CRPS, log-score, interval-score for:
    #   - Each of the 4 training rows (where actual y is available)
    # using raw_borders (y-space) and the per-row logits.
    #
    # These are the references for the SQL aggregate tests in tabfm_scoring.test.

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

    # PSR-01: CRPS
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

    # PSR-02: log-score
    print(f"\n--- Log-Score (PSR-02) ---")
    ls_vals = []
    for i in range(train_size):
        y_i = all_y[i]
        ls_i = log_score_bar(y_i, all_logits[i], raw_borders)
        ls_vals.append(ls_i)
        print(f"  row {i}: y={y_i:.6f}, log_score={ls_i:.15g}")

    mean_ls = sum(ls_vals) / len(ls_vals)
    print(f"\nMean log_score over {train_size} rows: {mean_ls:.15g}")
    print(f"round(mean_ls, 6) = {round(mean_ls, 6)}")
    print(f"\n[tabfm_scoring.test] Use: round(tabfm_log_score(actual, dist), 6) == {round(mean_ls, 6)}")

    # PSR-03: interval-score at 0.9 and 0.5
    print(f"\n--- Interval-Score coverage=0.9 (PSR-03) ---")
    is_vals_09 = []
    for i in range(train_size):
        y_i = all_y[i]
        is_i = interval_score_bar(y_i, all_logits[i], raw_borders, 0.9)
        is_vals_09.append(is_i)
        print(f"  row {i}: y={y_i:.6f}, interval_score={is_i:.15g}")

    mean_is_09 = sum(is_vals_09) / len(is_vals_09)
    print(f"\nMean interval_score (0.9) over {train_size} rows: {mean_is_09:.15g}")
    print(f"round(mean_is_09, 6) = {round(mean_is_09, 6)}")
    print(f"\n[tabfm_scoring.test] Use: round(tabfm_interval_score(actual, dist, 0.9), 6) == {round(mean_is_09, 6)}")

    print(f"\n--- Interval-Score coverage=0.5 (PSR-03) ---")
    is_vals_05 = []
    for i in range(train_size):
        y_i = all_y[i]
        is_i = interval_score_bar(y_i, all_logits[i], raw_borders, 0.5)
        is_vals_05.append(is_i)
        print(f"  row {i}: y={y_i:.6f}, interval_score={is_i:.15g}")

    mean_is_05 = sum(is_vals_05) / len(is_vals_05)
    print(f"\nMean interval_score (0.5) over {train_size} rows: {mean_is_05:.15g}")
    print(f"round(mean_is_05, 6) = {round(mean_is_05, 6)}")
    print(f"\n[tabfm_scoring.test] Use: round(tabfm_interval_score(actual, dist, 0.5), 6) == {round(mean_is_05, 6)}")


if __name__ == "__main__":
    main()
