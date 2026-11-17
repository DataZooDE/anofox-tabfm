"""The REAL regressor, against the contract the engine depends on: the target is min-max scaled.

Skipped unless TABFM_REAL_WEIGHTS is set and the checkpoint is cached (~/.cache/anofox-tabfm, the same file the
engine reads; never downloaded into this repo). Random weights cannot answer this: only the released weights know
what scale of target they were trained on.

Upstream's predictor normalises a regression target as (y - y_min) / (y_max - y_min)
(autogluon.tabular.models.mitra._internal.data.preprocessor: normalize_y / undo_normalize_y), so the released head
was trained on targets in [0, 1]. The engine used to z-score it instead and the model answered with predictions
shrunk toward the mean. Measured here on a noise-free linear target, with this package's Tab2D (byte-identical to
upstream's forward on real weights):

    target scaling   R2      slope(y on yhat)
    z-score          0.61    1.87
    min-max          0.999   1.005

Features are NOT the issue: the quantile embedding makes the model invariant to a monotone feature transform,
which the test also checks, so the engine may keep z-scoring them.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

# the engine's cache keys the directory by repo@revision; the built-in entry pins a commit (tabfm_registry.cpp)
CKPT = next(iter(sorted((pathlib.Path.home() / ".cache/anofox-tabfm").glob(
    "autogluon__mitra-regressor@*/model.safetensors"))), None)
pytestmark = pytest.mark.skipif(
    not (os.environ.get("TABFM_REAL_WEIGHTS") and CKPT is not None),
    reason="needs TABFM_REAL_WEIGHTS=1 and the cached mitra-regressor checkpoint")

from safetensors.torch import load_file  # noqa: E402

from export_mitra.export import ExportWrapper, build_model  # noqa: E402


@pytest.fixture(scope="module")
def wrapper():
    torch.set_num_threads(4)
    model = build_model("regression", dict(dim=512, dim_output=1, n_layers=12, n_heads=4),
                        state_dict=load_file(str(CKPT)))
    return ExportWrapper(model)


def _predict(w, xtr, ytr, xte, *, features, target):
    n, h = xtr.shape
    if features == "z":
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    shift, scale = (ytr.mean(), ytr.std()) if target == "z" else (ytr.min(), ytr.max() - ytr.min())
    xs = np.vstack([xtr, xte]).astype(np.float32)[None]
    ys = np.concatenate([(ytr - shift) / scale, np.zeros(len(xte))]).astype(np.float32)[None]
    with torch.no_grad():
        out = w(torch.from_numpy(xs), torch.from_numpy(ys), torch.tensor([n]), torch.tensor([h]))[0, n:, 0].numpy()
    return out * scale + shift


def _fit(y, p):
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, np.polyfit(p, y, 1)[0]


@pytest.fixture(scope="module")
def linear():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, (450, 3))
    return x[:300], 3 * x[:300, 0] + 2 * x[:300, 1] - x[:300, 2], x[300:], 3 * x[300:, 0] + 2 * x[300:, 1] - x[300:, 2]


def test_minmax_target_is_what_the_model_was_trained_on(wrapper, linear):
    xtr, ytr, xte, yte = linear
    r2, slope = _fit(yte, _predict(wrapper, xtr, ytr, xte, features="raw", target="minmax"))
    assert r2 > 0.99 and abs(slope - 1.0) < 0.05, (r2, slope)


def test_zscored_target_shrinks_predictions_so_the_engine_must_not_use_it(wrapper, linear):
    """The bad case, pinned: if this stops failing the model changed and the engine's choice should be revisited."""
    xtr, ytr, xte, yte = linear
    r2, slope = _fit(yte, _predict(wrapper, xtr, ytr, xte, features="raw", target="z"))
    assert r2 < 0.8 and slope > 1.5, (r2, slope)


def test_feature_scaling_is_irrelevant(wrapper, linear):
    xtr, ytr, xte, yte = linear
    raw = _predict(wrapper, xtr, ytr, xte, features="raw", target="minmax")
    z = _predict(wrapper, xtr, ytr, xte, features="z", target="minmax")
    assert np.abs(raw - z).max() < 0.05 * (yte.max() - yte.min())


def test_minmax_is_scale_and_offset_equivariant(wrapper, linear):
    xtr, ytr, xte, yte = linear
    base = _predict(wrapper, xtr, ytr, xte, features="raw", target="minmax")
    moved = _predict(wrapper, xtr, 1000 * ytr + 7000, xte, features="raw", target="minmax")
    assert np.abs((moved - 7000) / 1000 - base).max() < 1e-3 * (yte.max() - yte.min())
