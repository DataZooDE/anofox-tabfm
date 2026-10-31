"""The re-driven forward must reproduce upstream TabPFNV3p5.forward on every row.

Random-init fixture dims (woken: no all-zero projection), both tasks, both
contracts. The masked contract is fed the same rows padded to a larger bucket
(extra rows AND extra feature columns) and must agree on every real row. Negative
controls make sure the comparison can fail.
"""

import pytest
import torch

from export_tabpfn35 import configs
from export_tabpfn35.model import build_random_model
from export_tabpfn35.wrappers import MaskWrapper35, Wrapper35, upstream_wrapped

TOL = 2e-5


def _feed(T, H, N, task, seed=1, nan_frac=0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, H, generator=g)
    x[:, :, 0] = torch.randint(0, 3, (1, T), generator=g).float()   # a tied column
    if nan_frac:
        x[torch.rand(x.shape, generator=g) < nan_frac] = float("nan")
    if task == "regression":
        y = torch.randn(1, N, generator=g) * 3.0 + 7.0
    else:
        y = torch.arange(N, dtype=torch.float32).remainder(3)[None]
    return x, y


@pytest.fixture(scope="module")
def model():
    return build_random_model(configs.FIXTURE35, seed=0)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("shape", [(40, 7, 30), (16, 9, 6), (20, 2, 13), (9, 1, 1)])
def test_positional_matches_upstream(model, task, shape):
    T, H, N = shape
    x, y = _feed(T, H, N, task)
    with torch.no_grad():
        got = Wrapper35(model, task=task)(x, y)
        want = upstream_wrapped(model, x, y, task)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=0, atol=TOL)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_positional_matches_upstream_with_nans(model, task):
    x, y = _feed(30, 6, 20, task, nan_frac=0.1)
    with torch.no_grad():
        got = Wrapper35(model, task=task)(x, y)
        want = upstream_wrapped(model, x, y, task)
    torch.testing.assert_close(got, want, rtol=0, atol=TOL)


def _masked(model, task, x, y, Tp, Hp):
    T, H = x.shape[1], x.shape[2]
    N = y.shape[1]
    xp = torch.zeros(1, Tp, Hp)
    xp[:, :T, :H] = x
    yp = torch.full((1, Tp), -100.0)
    yp[:, :N] = y
    t = lambda v: torch.tensor([v], dtype=torch.int64)  # noqa: E731
    with torch.no_grad():
        return MaskWrapper35(model, task=task)(xp, yp, t(N), t(T), t(H))[:, :T]


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("shape", [(40, 7, 30), (16, 9, 6), (20, 2, 13)])
def test_masked_matches_upstream_on_real_rows(model, task, shape):
    T, H, N = shape
    x, y = _feed(T, H, N, task)
    with torch.no_grad():
        want = upstream_wrapped(model, x, y, task)
    got = _masked(model, task, x, y, T + 11, H + 5)
    torch.testing.assert_close(got, want, rtol=0, atol=TOL)


def test_negative_control_masked_d_ignored(model):
    """Telling the masked graph the padded width is the real one must be caught."""
    x, y = _feed(20, 4, 13, "classification")
    with torch.no_grad():
        want = upstream_wrapped(model, x, y, "classification")
    xp = torch.zeros(1, 20, 9)
    xp[:, :, :4] = x
    yp = torch.full((1, 20), -100.0)
    yp[:, :13] = y
    t = lambda v: torch.tensor([v], dtype=torch.int64)  # noqa: E731
    with torch.no_grad():
        bad = MaskWrapper35(model, task="classification")(xp, yp, t(13), t(20), t(9))
    assert (bad - want).abs().max() > 1e-3


def test_negative_control_one_train_row_too_many(model):
    x, y = _feed(20, 4, 13, "regression")
    with torch.no_grad():
        want = upstream_wrapped(model, x, y, "regression")
    got = _masked(model, "regression", x, torch.cat([y, torch.zeros(1, 1)], 1), 25, 6)
    assert (got - want).abs().max() > 1e-3
