"""The exportable ECDF must reproduce upstream's bucketed midrank ECDF exactly.

Upstream (`tabpfn_v3_5._build_ecdf_context` + `_in_context_ecdf`) sorts the train
rows per column and ranks every cell with `torch.searchsorted`, which has no ONNX
equivalent. `export_tabpfn35.ecdf` does the same with a fixed-iteration binary
search. These cases cover both of upstream's rulers: the exact one (at most
`num_buckets` distinct train values) and the row-position one above it, which
the real model reaches only past 8192 train rows -- so it is exercised here with a
small bucket count.
"""

import pytest
import torch

import tabpfn.architectures.tabpfn_v3_5 as v35

from export_tabpfn35.ecdf import ecdf_ranks


def _upstream(x_BRiC, num_train, num_buckets):
    ctx = v35._build_ecdf_context(x_BRiC, num_train, num_buckets)
    return v35._in_context_ecdf(x_BRiC, ctx)


def _cases():
    g = torch.Generator().manual_seed(0)
    cont = torch.randn(1, 60, 5, generator=g)
    ties = torch.randint(0, 4, (1, 60, 5), generator=g).float()
    mixed = torch.cat([cont[:, :, :2], ties[:, :, :2],
                       torch.full((1, 60, 1), 3.0)], dim=2)        # + a constant column
    return {"continuous": cont, "ties": ties, "mixed": mixed}


@pytest.mark.parametrize("name", ["continuous", "ties", "mixed"])
@pytest.mark.parametrize("num_train", [1, 2, 7, 40, 59])
@pytest.mark.parametrize("num_buckets", [8192, 16, 5])
def test_positional_matches_upstream(name, num_train, num_buckets):
    x = _cases()[name]
    want = _upstream(x, num_train, num_buckets)
    got = ecdf_ranks(x, torch.tensor(num_train), num_buckets)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ["continuous", "ties", "mixed"])
@pytest.mark.parametrize("num_train", [1, 7, 40])
@pytest.mark.parametrize("num_buckets", [8192, 16, 5])
def test_masked_rows_match_upstream(name, num_train, num_buckets):
    """Padded rows after the real ones must not move any real row's rank.

    The masked contract pads x to a bucket with junk rows; the train set is still
    the first `num_train` rows, but the tensors are longer than upstream's.
    """
    x = _cases()[name]
    want = _upstream(x, num_train, num_buckets)
    pad = torch.full((1, 23, x.shape[2]), 1e6)
    pad[:, ::2] = -1e6
    got = ecdf_ranks(torch.cat([x, pad], dim=1), torch.tensor(num_train), num_buckets)
    torch.testing.assert_close(got[:, : x.shape[1]], want, rtol=0, atol=1e-6)


def test_negative_control_wrong_train_count_is_caught():
    """The comparison above must be able to fail: one train row too many."""
    x = _cases()["continuous"]
    want = _upstream(x, 40, 8192)
    got = ecdf_ranks(x, torch.tensor(41), 8192)
    assert (got - want).abs().max() > 1e-3
