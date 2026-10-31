"""A float64 replica of what the engine feeds a `*_minimal` graph (src/tabfm_preprocess.cpp).

Kumo-Tabular's single-estimator numeric recipe is the engine's standardising profile, so the graph
takes engine-preprocessed features. To compute golden values for the fixture, and to compare the
engine with upstream on real weights, the same transform is needed in Python. Mirrors, in order:

  UniqueFeatureFilter   drop columns with <= 1 distinct value over the context rows (kept when
                        the context has <= 1 row)                      == DropConstantColumns()
  CustomStandardScaler  context mean / population std, + 1e-6, clip to [-100, 100]
                                                                       == Standardize(eps=1e-6),
                                                                          Clip(-100, 100)
  OutlierRemover        two-stage 4-sigma bounds on the context (ddof 1), then
                        max(lower - log1p|x|, x), min(upper + log1p|x|, x)
                                                                       == ClipSigma(4, "log")
  target (regression)   context mean / population std (std 0 -> 1)     == Standardize()

Upstream's recipe also draws a random FlipSign, ShuffleColumns and (classification) a class shift
per estimator, and has a round-robin Choice whose first member is Identity. Those are ensemble
diversity, not part of the model's function; the deterministic member is what is reproduced here
and what the engine computes.
"""

from __future__ import annotations

import numpy as np


def keep_columns(x_train: np.ndarray) -> np.ndarray:
    if x_train.shape[0] <= 1:
        return np.ones(x_train.shape[1], dtype=bool)
    return (x_train != x_train[:1]).any(axis=0)


def standardise_features(x: np.ndarray, s: int) -> np.ndarray:
    """x [T, H] float64, first s rows are the context -> [T, H'] float64 (constant columns dropped)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, keep_columns(x[:s])]
    mean = x[:s].mean(axis=0)
    scale = np.sqrt(((x[:s] - mean) ** 2).mean(axis=0)) + 1e-6
    z = np.clip((x - mean) / scale, -100.0, 100.0)

    ddof1 = s > 1

    def mean_std(col, mask):
        vals = col[:s][~mask]
        m = vals.mean() if vals.size else 0.0
        var = ((vals - m) ** 2).sum()
        cnt = vals.size
        denom = (cnt - 1 if cnt > 1 else 1) if ddof1 else cnt
        return m, max(np.sqrt(var / (denom if denom else 1)), 1e-6)

    out = np.empty_like(z)
    for j in range(z.shape[1]):
        col = z[:, j]
        m1, s1 = mean_std(col, np.zeros(s, dtype=bool))
        masked = (col[:s] < m1 - 4 * s1) | (col[:s] > m1 + 4 * s1)
        m2, s2 = mean_std(col, masked)
        lo, hi = m2 - 4 * s2, m2 + 4 * s2
        v = np.maximum(-np.log1p(np.abs(col)) + lo, col)
        out[:, j] = np.minimum(np.log1p(np.abs(v)) + hi, v)
    return out


def standardise_target(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    y = np.asarray(y_train, dtype=np.float64)
    mean = y.mean()
    std = np.sqrt(((y - mean) ** 2).mean())
    if std == 0.0:
        std = 1.0
    return (y - mean) / std, float(mean), float(std)


def engine_inputs(x: np.ndarray, y_train: np.ndarray, *, regression: bool, classes=None):
    """The five graph inputs the engine builds for one call, as float32/int64/bool numpy arrays,
    plus (mean, std) to invert a regression output. `classes` maps a raw label to its code
    (the engine label-encodes alphabetically); omitted = labels already are codes."""
    s = len(y_train)
    z = standardise_features(x, s).astype(np.float32)
    t, h = z.shape
    y = np.full((1, t), -100.0, dtype=np.float32)
    inv = (0.0, 1.0)
    if regression:
        yz, mean, std = standardise_target(y_train)
        y[0, :s] = yz
        inv = (mean, std)
    else:
        y[0, :s] = [classes[v] if classes else v for v in y_train]
    feed = {"x": z[None], "y": y, "train_size": np.array([s], dtype=np.int64),
            "cat_mask": np.zeros((1, h), dtype=bool), "d": np.array([h], dtype=np.int64)}
    return feed, inv
