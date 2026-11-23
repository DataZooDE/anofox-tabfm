"""The re-driven forward must reproduce upstream TabPFNV3p5.forward on every row.

Random-init fixture dims (woken: no all-zero projection), both tasks.
"""

import pytest
import torch

from export_tabpfn35 import configs
from export_tabpfn35.model import build_random_model
from export_tabpfn35.wrappers import Wrapper35, upstream_wrapped

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


