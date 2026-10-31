"""TabPFN-3.5's in-context ECDF feature, in a form ONNX (and MIGraphX) can run.

3.5 gives every cell a third channel next to its scaled value and its NaN
indicator: the cell's midrank ECDF against the TRAIN rows of its column. Upstream
(`tabpfn_v3_5._build_ecdf_context` + `_in_context_ecdf`) builds it with
`torch.sort` and `torch.searchsorted`. Sort exports (TopK); searchsorted has no
ONNX operator at all, which is the one thing that blocks a plain export.

This module computes the same numbers with a binary search of a FIXED number of
steps (`_SEARCH_STEPS`), each a Gather and a compare -- no data-dependent loop,
no integer division (MIGraphX's GPU target miscomputes integer floor-division,
docs/ROCM_TABPFN_PLAN.md), no Python branch on a value.

It follows upstream's algorithm step for step, both rulers included:

* at most `num_buckets` distinct train values in a column: one edge per distinct
  value, and the rank is the exact midrank `#(train < v) + #(train == v) / 2`;
* more than that (the released model: more than 8192 train rows AND distinct
  values): edges are spaced over row positions and ranks between two edges are
  interpolated -- reproduced here rather than replaced by the exact midrank, so
  the graph agrees with upstream at every table size, not only below 8192 rows.

The train rows are the FIRST `n_train` rows of the row axis; anything after them
(test rows, or the masked contract's padding) is ranked but never ranked against.
That one convention serves both graph contracts: the positional one passes
`n_train = len(y)`, the masked one passes its `train_size` input.
"""

from __future__ import annotations

import torch

#: Binary-search steps. 2**24 - 1 rows is far past anything the engine admits
#: (anofox_tabfm_max_rows defaults to 10 000); every step is one Gather.
_SEARCH_STEPS = 24

#: Sort key for rows that are not train rows: they sort after every real value
#: and the bounded searches below never look at them.
_AFTER_EVERYTHING = torch.finfo(torch.float32).max


def _searchsorted(seq, values, length, *, right: bool):
    """`torch.searchsorted(seq[..., :length], values, side=...)` for sorted `seq`.

    seq [..., L] (only the first `length` entries are sorted and valid), values
    [..., M], length broadcastable to [..., 1]. Returns int64 [..., M]: the number
    of valid entries `< v` (left) or `<= v` (right).

    Power-of-two descent: `pos` only ever grows by a step if the entry just below
    the new position still precedes `v` and lies inside `length`. Every step is
    shape-static; L does not have to be known at export time.
    """
    L = seq.shape[-1]
    pos = torch.zeros(values.shape, dtype=torch.int64, device=values.device)
    for p in reversed(range(_SEARCH_STEPS)):
        cand = pos + (1 << p)
        idx = torch.clamp(cand - 1, min=0, max=L - 1)
        s = seq.gather(-1, idx)
        before = (s <= values) if right else (s < values)
        pos = torch.where((cand <= length) & before, cand, pos)
    return pos


def ecdf_ranks(x_BRiC: torch.Tensor, n_train: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """Upstream's `_in_context_ecdf(x, _build_ecdf_context(x, n_train, num_buckets))`.

    x_BRiC [B, Ri, C] finite (imputed) values, n_train a 0-d / [1] int64 tensor
    (>= 1), the train rows being x[:, :n_train]. Returns [B, Ri, C] float32.
    """
    dt = x_BRiC.dtype
    dev = x_BRiC.device
    Ri = x_BRiC.shape[1]
    n = n_train.reshape(1, 1, 1).to(torch.int64)

    vals_BCR = x_BRiC.transpose(1, 2).to(torch.float32)
    row = torch.arange(Ri, device=dev).reshape(1, 1, Ri)
    big = torch.full((), _AFTER_EVERYTHING, dtype=torch.float32, device=dev)
    sorted_BCR = torch.sort(torch.where(row < n, vals_BCR, big), dim=-1).values

    # Index of each sorted position within the column's distinct values.
    is_new = torch.cat([torch.ones_like(sorted_BCR[..., :1], dtype=torch.int64),
                        (sorted_BCR[..., 1:] != sorted_BCR[..., :-1]).to(torch.int64)], dim=-1)
    distinct_idx = torch.cumsum(is_new, dim=-1) - 1                                  # [B,C,Ri]
    num_distinct = distinct_idx.gather(-1, (n - 1).expand(*distinct_idx.shape[:2], 1)) + 1

    # K = min(num_buckets, n) edges. Allocated at min(num_buckets, Ri) -- a slice
    # of a constant bound, so no symbolic min -- and the tail past K (only ever
    # present in the masked contract, where Ri > n) is excluded by `k` below.
    k = torch.clamp(n, max=num_buckets)
    steps = torch.arange(Ri, device=dev, dtype=torch.float32)[:num_buckets]           # [K]
    # Upstream divides by (k - 1) only when k > 1 and uses all-zero targets
    # otherwise; with a numerator of 0 at k == 1, a denominator of 1 is the same.
    denom = torch.clamp(k - 1, min=1)
    distinct_t = steps * ((num_distinct - 1).to(torch.float32) / denom.to(torch.float32))
    # Upstream computes (n - 1) / (k - 1) as a Python float (double) before the
    # fp32 multiply; do the same so the rounding is the same.
    ratio = ((n - 1).to(torch.float64) / denom.to(torch.float64)).to(torch.float32)
    rows_K = torch.round(steps * ratio.reshape(1, 1, 1)).to(torch.int64)
    rows_K = torch.clamp(rows_K, max=Ri - 1).expand(*sorted_BCR.shape[:2], -1)
    targets = torch.where(num_distinct <= k,
                          torch.round(distinct_t).to(torch.int64),
                          distinct_idx.gather(-1, rows_K))

    below = _searchsorted(distinct_idx, targets, n, right=False)
    at_most = _searchsorted(distinct_idx, targets, n, right=True)
    edges = sorted_BCR.gather(-1, torch.clamp(below, max=Ri - 1))
    below_f = below.to(torch.float32)
    at_most_f = at_most.to(torch.float32)

    # ---- _ecdf_midrank_counts ------------------------------------------------
    left = _searchsorted(edges, vals_BCR, k, right=False)
    right = _searchsorted(edges, vals_BCR, k, right=True)
    lo_idx = torch.clamp(left - 1, min=0)
    hi_idx = torch.minimum(left, k - 1)
    edge_lo = edges.gather(-1, lo_idx)
    edge_hi = edges.gather(-1, hi_idx)
    at_most_lo = at_most_f.gather(-1, lo_idx)
    below_hi = below_f.gather(-1, hi_idx)

    width = edge_hi - edge_lo
    inside = width > 0
    one = torch.ones((), dtype=torch.float32, device=dev)
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    weight = torch.where(inside, (vals_BCR - edge_lo) / torch.where(inside, width, one), zero)
    counts = at_most_lo + weight * (below_hi - at_most_lo)
    is_edge = right > left
    exact = 0.5 * (below_hi + at_most_f.gather(-1, hi_idx))
    counts = torch.where(is_edge, exact, counts)
    counts = torch.where((left == 0) & ~is_edge, zero, counts)

    last = (k - 1).expand(*sorted_BCR.shape[:2], 1)
    mid_lo = 0.5 * (below_f[..., :1] + at_most_f[..., :1])
    mid_hi = 0.5 * (below_f.gather(-1, last) + at_most_f.gather(-1, last))
    counts = torch.minimum(torch.maximum(counts, mid_lo), mid_hi)
    # The last edge is the column maximum, so its at-most count is the row count.
    n_rows = at_most_f.gather(-1, last)
    return (counts / n_rows).transpose(1, 2).to(dt)
