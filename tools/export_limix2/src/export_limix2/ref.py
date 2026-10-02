"""Reference outputs from UNMODIFIED upstream (no apply_export_patches), in a process of their own, so the
class-level patches can never leak into the oracle. The one deliberate difference from upstream: the positional
embedding table is the seeded one (upstream draws it unseeded on every call)."""
from __future__ import annotations

import sys
import types

import numpy as np
import torch

from .limix2_patches import GMAX, load_upstream
from .wrapper import REG_TEMP


def make_case(T, H, S, task, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, T, H)).astype(np.float32)
    w = rng.standard_normal((H, 3))
    z = x[0] @ w
    if task == "classification":
        y = z.argmax(1)[None, :S].astype(np.float32)
    else:
        y = (z[:, 0] * 7.0 + 30.0 + 0.1 * rng.standard_normal(T))[None, :S].astype(np.float32)
    return x, y


def _ref_model(layers, seed=0):
    m, cfg = load_upstream(layers)
    G = int(m.feature_positional_embedding.in_features)
    table = torch.randn((GMAX, G), generator=torch.Generator().manual_seed(seed))

    def add_embeddings(self, x, *, generator=None):
        return x + self.feature_positional_embedding(table[: x.shape[2]].to(x.dtype))[None, None]

    m.add_embeddings = types.MethodType(add_embeddings, m)
    return m


@torch.inference_mode()
def reference(m, x, y, task):
    """All-row reference: queries from (x, y, S) as upstream runs them; fitted rows from the duplicate route."""
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    S = yt.shape[1]
    if task == "regression":
        mu = yt.double().mean(); sd = yt.double().std(unbiased=True)
        sd = sd if sd != 0 else torch.tensor(1.0, dtype=torch.float64)
        yin = ((yt.double() - mu) / sd).float()
    else:
        yin = yt
    tt = "Classification" if task == "classification" else "Regression"

    def run(xx):
        o = m(x=xx, y=yin, eval_pos=S, task_type=tt)
        if task == "classification":
            return o["cls_output"][0]
        lg = o["reg_output"][0][0]
        p = torch.softmax(lg / REG_TEMP, dim=-1)
        c = (m._reg_borders[:-1] + m._reg_borders[1:]) / 2.0
        return ((p * c).sum(-1, keepdim=True).double() * sd + mu).float()

    qry = run(xt)  # [T-S, C]: upstream's own call, queries only
    allrows = run(torch.cat([xt[:, :S], xt], 1))  # [T, C]: duplicate route, rows < S are the in-context fitted values
    return qry.numpy(), allrows.numpy()
