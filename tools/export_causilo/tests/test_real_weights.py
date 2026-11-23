"""The exported computation against upstream's own public estimators, on the REAL checkpoints.

Skipped unless TABFM_REAL_WEIGHTS is set: it downloads nums-ai/causilo at upstream's pinned
revision into the Hugging Face cache (never into this repo). Weights are under the Causilo
License v1.0 (non-commercial research/evaluation); running this is evaluation.

Random weights cannot answer any of these questions. They pin:

  * the wrapper (normaliser + network + output reduction) equals `CausiloClassifier` /
    `CausiloRegressor` with n_estimators=1, the path the engine can honestly claim;
  * the in-context fitted values for the training rows are real, and the duplicate route is
    why: the cheap alternative is measurably wrong on the real model, and this test keeps
    that number from being forgotten.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("TABFM_REAL_WEIGHTS"),
                                reason="needs the real Causilo checkpoints (set TABFM_REAL_WEIGHTS=1)")

from sklearn.datasets import load_diabetes, load_iris, load_wine, make_classification  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from causilo import CausiloClassifier, CausiloRegressor  # noqa: E402
from causilo.checkpoints import load_pretrained_model  # noqa: E402

from export_causilo.causilo_patches import ExportWrapper, apply, normalise_features, predict  # noqa: E402

apply()


@pytest.fixture(scope="module")
def classifier():
    return load_pretrained_model("classification").eval()


@pytest.fixture(scope="module")
def regressor():
    return load_pretrained_model("regression").eval()


def _classification_sets():
    yield "iris", *load_iris(return_X_y=True)
    yield "wine", *load_wine(return_X_y=True)
    yield "synthetic-5-class", *make_classification(n_samples=400, n_features=20, n_informative=8,
                                                    n_classes=5, random_state=1)


@pytest.mark.parametrize("name,X,y", list(_classification_sets()), ids=lambda v: v if isinstance(v, str) else "")
def test_classifier_matches_upstream_single_estimator(classifier, name, X, y):
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    est = CausiloClassifier(n_estimators=1, device="cpu").fit(Xtr, ytr)
    want = est.predict_proba(Xte)
    n = len(est.classes_)
    x = torch.from_numpy(np.vstack([Xtr, Xte]).astype(np.float32))[None]
    labels = torch.from_numpy(np.searchsorted(est.classes_, ytr).astype(np.float32))[None]
    with torch.no_grad():
        out = ExportWrapper(classifier)(x, labels)[0]
    got = torch.softmax(out[len(ytr):, :n], -1).numpy()
    assert np.abs(got - want).max() < 1e-4
    assert (got.argmax(-1) == want.argmax(-1)).all()


def test_regressor_matches_upstream_single_estimator(regressor):
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    want = CausiloRegressor(n_estimators=1, device="cpu").fit(Xtr, ytr).predict(Xte)
    x = torch.from_numpy(np.vstack([Xtr, Xte]).astype(np.float32))[None]
    with torch.no_grad():
        out = ExportWrapper(regressor)(x, torch.from_numpy(ytr.astype(np.float32))[None])[0, :, 0].numpy()
    # relative to the target scale: float32 graph vs upstream's float64 pre/post-processing
    assert np.abs(out[len(ytr):] - want).max() < 1e-4 * np.abs(want).mean()


def _cheap_route(model, x, y, regression):
    """The route NOT taken: run the last prediction layer over all rows. Context rows then decode
    a representation with their own label already injected, which the model never saw."""
    count = y.shape[1]
    table = normalise_features(x, y)
    tg, mean_y, std_y = y, None, None
    if regression:
        mean_y = y.mean(1, keepdim=True)
        std_y = ((y - mean_y) ** 2).mean(1, keepdim=True).sqrt()
        tg = (y - mean_y) / std_y
    from causilo.execution.runner import inject_targets
    f = model.feature_embedding(table)
    f = inject_targets(f, model.feature_target(tg))
    f = model.columns[0](f.transpose(1, 2), context_rows=count).transpose(1, 2)
    f = model.row(f)
    f = model.columns[1](f.transpose(1, 2), context_rows=count).transpose(1, 2)
    rows = inject_targets(model.pool(f), model.row_target(tg))
    for layer in model.prediction.layers:
        rows = layer.query(rows, layer.prepare(rows[..., :count, :]))
    out = model.head(rows)
    return out.mean(-1) * std_y + mean_y if regression else out


def test_fitted_values_are_in_context_values_and_the_cheap_route_is_not(regressor):
    """Clean linear problem: query R2 ~ 0.999. Duplicate route: context R2 ~ 1.0. Cheap route: ~ 0.5."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] + 0.1 * rng.standard_normal(200)).astype(np.float32)
    x = torch.from_numpy(X)[None]
    s = 140
    yt = torch.from_numpy(y[:s])[None]
    r2 = lambda p, t: 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()

    with torch.no_grad():
        ours = ExportWrapper(regressor)(x, yt)[0, :, 0].numpy()
        cheap = _cheap_route(regressor, x, yt, True)[0].numpy()
    assert r2(ours[s:], y[s:]) > 0.95, "query rows must be a good model"
    assert r2(ours[:s], y[:s]) > 0.95, "context rows must carry real in-context fitted values"
    assert r2(cheap[:s], y[:s]) < 0.8, "if the cheap route is now fine, the duplicate route is dead weight"


def test_duplicate_route_leaves_query_rows_unchanged(classifier):
    X, y = load_wine(return_X_y=True)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    x = torch.from_numpy(np.vstack([Xtr, Xte]).astype(np.float32))[None]
    labels = torch.from_numpy(np.searchsorted(np.unique(ytr), ytr).astype(np.float32))[None]
    s = len(ytr)
    with torch.no_grad():
        single = predict(classifier, normalise_features(x, labels), labels)[0]
        every = ExportWrapper(classifier)(x, labels)[0, s:]
    assert (single - every).abs().max().item() < 1e-3
