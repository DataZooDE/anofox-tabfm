"""The two export wrappers, and the upstream reference both are checked against.

Positional contract (CPU / CUDA, the shipped `graph_tabpfn35*_<task>.onnx`):
    x[1,T,H] f32, y[1,N] f32 (the TRAIN labels; N = train_size) -> logits[1,T,C]
Masked contract (MIGraphX, fixed-shape buckets):
    x[1,T,H], y[1,T], train_size[1], n_rows[1], d[1] (int64) -> logits[1,T,C]

Both are the same model call (`forward35.rows_v35`) and the same output contract
as tools/export_tabpfn's ExportWrapper: EVERY row carries a real value -- rows >=
train_size are the query predictions, rows below are the in-context fitted
values. Classification decodes the train rows' own embeddings (the embedding
route); regression presents the rows a second time as queries (the duplicate
route) -- the split tools/export_tabpfn measured for v3, re-measured for 3.5 on
the released weights (docs/REAL_MODELS.md, TabPFN-3.5 section).

Regression is raw-in / raw-out exactly like v3: the wrapper z-normalises the raw
train targets (population std), decodes the bar distribution to its mean over
the checkpoint's `heads.regression_borders`, and de-standardises.
"""

from __future__ import annotations

import torch

from export_tabpfn.tabpfn_patched import ExportWrapper

from export_tabpfn35.forward35 import rows_v35


def _count(t):
    """The length of a 1-D tensor as an int64 VALUE (not a shape)."""
    return torch.ones_like(t, dtype=torch.int64).sum()


class Wrapper35(ExportWrapper):
    """Positional contract. `self.m` is a TabPFNV3p5; `_bardist_mean` is inherited."""

    def forward(self, x, y):
        xt = x.permute(1, 0, 2)                                     # [T,1,H]
        T = xt.shape[0]
        N = y.shape[1]
        yt = y[0]
        n = _count(yt)
        d = _count(x[0, 0])
        if self.task == "regression":
            ymean = yt.mean()
            ystd = torch.clamp(torch.sqrt(((yt - ymean) ** 2).mean()), min=1e-20)
            yz = (yt - ymean) / ystd
            x2 = torch.cat([xt[:N], xt], dim=0)                     # [N+T,1,H]
            y2 = torch.cat([yz, torch.zeros(T, dtype=yz.dtype, device=yz.device)])
            logits = rows_v35(self.m, x2, y2, n, d, task="regression")[N:]
            full = (self._bardist_mean(logits) * ystd + ymean).unsqueeze(-1)
        else:
            yf = torch.cat([yt, torch.zeros(T - N, dtype=yt.dtype, device=yt.device)])
            full = rows_v35(self.m, xt, yf, n, d, task="classification")
        return full.permute(1, 0, 2)                                # [1,T,C]


class MaskWrapper35(ExportWrapper):
    """Masked contract: every input padded to the (T, H) bucket; the split, the
    real row count and the real feature count arrive as values."""

    def forward(self, x, y, train_size, n_rows, d):
        xt = x.permute(1, 0, 2)
        y_R = torch.nan_to_num(y[0], nan=0.0, posinf=0.0, neginf=0.0)
        T = xt.shape[0]
        # Indexed, never .reshape(()): MIGraphX reads a Reshape to [] as ZERO
        # elements and refuses the graph (docs/ROCM_TABPFN_PLAN.md).
        ts = train_size[0]
        dd = d[0]
        del n_rows  # 3.5 has no statistic over all rows; padded rows are only queries
        if self.task == "regression":
            train = torch.arange(T, device=xt.device) < ts
            nf = ts.to(y_R.dtype)
            zero = torch.zeros_like(y_R)
            ymean = torch.where(train, y_R, zero).sum() / nf
            ystd = torch.clamp(torch.sqrt(torch.where(train, (y_R - ymean) ** 2, zero).sum() / nf),
                               min=1e-20)
            yz = (y_R - ymean) / ystd
            logits = rows_v35(self.m, torch.cat([xt, xt], 0), torch.cat([yz, yz], 0), ts, dd,
                              task="regression")[T:]
            full = (self._bardist_mean(logits) * ystd + ymean).unsqueeze(-1)
        else:
            full = rows_v35(self.m, xt, y_R, ts, dd, task="classification")
        return full.permute(1, 0, 2)


# --------------------------------------------------------------------------- #
# Upstream reference                                                           #
# --------------------------------------------------------------------------- #

def upstream_rows(model, x_RiBC, y_N, task):
    """Upstream's own TabPFNV3p5.forward, arranged as the wrappers arrange it.

    classification: query rows from `forward(...)["standard"]`, train rows by
    decoding upstream's own `train_embeddings` with upstream's own heads.
    regression: the duplicate route, [train ; all] with the train labels.
    """
    N = y_N.shape[0]
    with torch.no_grad():
        if task == "regression":
            x2 = torch.cat([x_RiBC[:N], x_RiBC], 0)
            return model(x2, y_N, "regression")
        res = model(x_RiBC, y_N, "multiclass", only_return_standard_out=False)
        tr = res["train_embeddings"].transpose(0, 1)               # [B,N,D]
        keys = model.heads.project_decoder_keys(tr)
        npc = int(torch.nan_to_num(y_N, nan=0.0).max().item()) + 1
        fitted = model.heads(keys, tr, y_N.reshape(1, N), task_type="multiclass",
                             num_present_classes=npc)
        fitted = torch.nan_to_num(fitted, nan=0.0)
        return torch.cat([fitted, res["standard"]], 0)


def upstream_wrapped(model, x, y, task):
    """`Wrapper35(model, task)(x, y)` computed through upstream's forward instead."""
    w = ExportWrapper(model, task=task)
    xt = x.permute(1, 0, 2)
    yt = y[0]
    if task == "regression":
        ymean = yt.mean()
        ystd = torch.clamp(torch.sqrt(((yt - ymean) ** 2).mean()), min=1e-20)
        logits = upstream_rows(model, xt, (yt - ymean) / ystd, task)
        full = (w._bardist_mean(logits) * ystd + ymean).unsqueeze(-1)
    else:
        full = upstream_rows(model, xt, yt, task)
    return full.permute(1, 0, 2)
