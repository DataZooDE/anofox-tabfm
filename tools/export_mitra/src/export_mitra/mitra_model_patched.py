# PATCHED, ONNX-EXPORTABLE re-implementation of AWS AutoGluon's Mitra (Tab2D).
#
# Upstream (Apache-2.0):
#   autogluon.tabular.models.mitra._internal.models.tab2d.Tab2D
#   autogluon.tabular.models.mitra._internal.models.embedding.*
#
# This file reproduces Mitra's **CPU forward path** (the non-flash-attention
# branch — the one AutoGluon itself runs on machines without a CUDA GPU) with
# the exact same module/parameter NAMES and SHAPES, so the released
# safetensors (autogluon/mitra-classifier | mitra-regressor) load unchanged.
# The parameter-bearing submodules are byte-identical to upstream; only the
# non-exportable glue is rewritten:
#
#   * torch.utils.checkpoint.checkpoint   -> plain layer call (inference only)
#   * einx / einops.pack / einops.unpack  -> plain torch reshape/cat/permute
#   * torch.quantile(x, q=999, dim=1)     -> sort + gather + lerp (ONNX TopK/
#                                            Gather), computed over the VALID
#                                            (non-padded) support rows only
#   * torch.vmap(torch.bucketize)         -> broadcast compare + sum
#   * in-place boolean-mask assignment    -> torch.where
#
# Two SEMANTIC generalizations over the upstream CPU branch (which silently
# ignores its padding masks and is therefore only correct when support/query
# are pre-sliced with no padding):
#   1. The quantile embedding excludes padded support rows from the quantile /
#      mean / std statistics (upstream relies on external slicing to do this).
#   2. Row attention masks out padded support rows as KEYS, and feature
#      attention masks out padded feature columns as KEYS.
# Both reduce to the upstream result when there is no padding, and match the
# GPU flash-attention branch's semantics. They are what let the exported graph
# take a single [1,T,H] table plus train_size/d (the engine contract) instead
# of pre-split support/query tensors.
#
# Copyright the AutoGluon authors (Apache-2.0). Rewrites: anofox-tabfm.

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1.0e9  # additive attention mask sentinel (finite -> ONNX/ORT safe)


# --------------------------------------------------------------------------- #
# Embeddings (parameter modules — names/shapes byte-identical to upstream)
# --------------------------------------------------------------------------- #
class Tab2DEmbeddingX(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.x_embedding = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upstream: einx.rearrange("b s f -> b s f 1", x)
        return self.x_embedding(x.unsqueeze(-1))  # (b, s, f, d)


class Tab2DQuantileEmbeddingX(nn.Module):
    """Rank/quantile feature normalization, computed INSIDE the model.

    Parameter-free. Reproduces upstream Tab2DQuantileEmbeddingX but (a) via
    sort instead of torch.quantile/vmap.bucketize and (b) excluding padded
    support rows from the statistics.
    """

    N_QUANTILES = 999  # upstream: arange(1, 1000) / 1000

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x_support, x_query, padding_mask, feature_mask):
        # x_support/x_query: (b, s, f) / (b, q, f); padding_mask: (b, s)
        # True == padded. feature_mask unused here (features handled in attn).
        b = x_support.shape[0]
        s = x_support.shape[1]
        f = x_support.shape[2]
        dev = x_support.dtype

        valid = ~padding_mask  # (b, s)
        seq_len = valid.to(x_support.dtype).sum(dim=1)  # (b,)
        seq_len_c = seq_len.clamp(min=1.0)

        # --- quantiles over VALID support rows only (sort + gather + lerp) ---
        # push padded rows to +inf so they sort to the top and never enter the
        # interpolation window [0, seq_len-1].
        big = torch.full_like(x_support, float("inf"))
        x_masked = torch.where(valid[:, :, None], x_support, big)
        sorted_x, _ = torch.sort(x_masked, dim=1)  # (b, s, f) ascending

        q = (torch.arange(1, self.N_QUANTILES + 1, device=x_support.device,
                          dtype=torch.float32) / (self.N_QUANTILES + 1))
        pos = q[None, :] * (seq_len_c - 1.0)[:, None]           # (b, Q)
        lo = torch.clamp(torch.floor(pos), min=0.0)
        hi = torch.clamp(lo + 1.0, max=(seq_len_c - 1.0)[:, None])
        frac = pos - lo
        lo_i = lo.to(torch.int64)[:, :, None].expand(b, self.N_QUANTILES, f)
        hi_i = hi.to(torch.int64)[:, :, None].expand(b, self.N_QUANTILES, f)
        q_lo = torch.gather(sorted_x, 1, lo_i)                  # (b, Q, f)
        q_hi = torch.gather(sorted_x, 1, hi_i)
        quantiles = q_lo + frac[:, :, None] * (q_hi - q_lo)     # (b, Q, f)

        # --- bucketize(value, quantiles) == count(quantiles < value) --------
        # matches torch.bucketize(..., right=False).
        def _bucketize(vals):  # vals: (b, n, f) -> (b, n, f)
            cmp = (quantiles[:, None, :, :] < vals[:, :, None, :])  # (b,n,Q,f)
            return cmp.to(x_support.dtype).sum(dim=2)
        xs = _bucketize(x_support) / seq_len_c[:, None, None]
        xq = _bucketize(x_query) / seq_len_c[:, None, None]

        # mean over valid rows, then subtract
        xs = torch.where(valid[:, :, None], xs, torch.zeros_like(xs))
        mean = xs.sum(dim=1, keepdim=True) / seq_len_c[:, None, None]
        xs = xs - mean
        xq = xq - mean

        # variance over valid rows, then divide by std
        xs = torch.where(valid[:, :, None], xs, torch.zeros_like(xs))
        var = (xs * xs).sum(dim=1, keepdim=True) / seq_len_c[:, None, None]
        std = var.sqrt()
        xs = xs / std
        xq = xq / std

        zero_var = (var == 0)
        xs = torch.where(zero_var, torch.zeros_like(xs), xs)
        xq = torch.where(zero_var, torch.zeros_like(xq), xq)
        return xs.to(dev), xq.to(dev)


class Tab2DEmbeddingYClasses(nn.Module):
    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.y_embedding = nn.Embedding(n_classes, dim)
        self.y_mask = nn.Embedding(1, dim)

    def forward(self, y_support, padding_obs_support, n_obs_query):
        b = y_support.shape[0]
        # engine feeds float labels; clamp to a valid class id (test/padded
        # rows carry arbitrary values -> they are zeroed just below anyway).
        y = torch.clamp(y_support.to(torch.int64), 0, self.n_classes - 1)
        y = torch.where(padding_obs_support, torch.zeros_like(y), y)  # (b, s)
        emb = self.y_embedding(y)                                     # (b, s, d)
        emb = torch.where(padding_obs_support[:, :, None],
                          torch.zeros_like(emb), emb)
        y_support_e = emb.unsqueeze(2)                               # (b, s, 1, d)
        q_ids = torch.zeros((b, n_obs_query), dtype=torch.int64,
                            device=y_support.device)
        y_query_e = self.y_mask(q_ids).unsqueeze(2)                  # (b, q, 1, d)
        return y_support_e, y_query_e


class Tab2DEmbeddingYRegression(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.y_embedding = nn.Linear(1, dim)
        self.y_mask = nn.Embedding(1, dim)

    def forward(self, y_support, padding_obs_support, n_obs_query):
        b = y_support.shape[0]
        y = y_support.to(torch.float32).unsqueeze(-1)  # (b, s, 1)
        emb = self.y_embedding(y)                      # (b, s, d)
        emb = torch.where(padding_obs_support[:, :, None],
                          torch.zeros_like(emb), emb)
        y_support_e = emb.unsqueeze(2)                 # (b, s, 1, d)
        q_ids = torch.zeros((b, n_obs_query), dtype=torch.int64,
                            device=y_support.device)
        y_query_e = self.y_mask(q_ids).unsqueeze(2)    # (b, q, 1, d)
        return y_support_e, y_query_e


# --------------------------------------------------------------------------- #
# Attention + transformer layer (CPU / SDPA path, mask-aware)
# --------------------------------------------------------------------------- #
class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_flash_attn: bool = False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, query, key, value, attn_mask=None):
        # (b, t, dim) -> (b, h, t, hd)
        b, tq, _ = query.shape
        tk = key.shape[1]
        h, hd = self.n_heads, self.dim // self.n_heads
        q = self.q(query).view(b, tq, h, hd).permute(0, 2, 1, 3)
        k = self.k(key).view(b, tk, h, hd).permute(0, 2, 1, 3)
        v = self.v(value).view(b, tk, h, hd).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.permute(0, 2, 1, 3).reshape(b, tq, self.dim)
        return self.o(o)


class Layer(nn.Module):
    def __init__(self, dim: int, n_heads: int, use_flash_attn: bool = False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.attention1 = MultiheadAttention(dim, n_heads)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.layer_norm3 = nn.LayerNorm(dim)
        self.attention2 = MultiheadAttention(dim, n_heads)
        self.layer_norm4 = nn.LayerNorm(dim)
        self.linear3 = nn.Linear(dim, dim * 4)
        self.linear4 = nn.Linear(dim * 4, dim)

    def forward(self, support, query, row_key_mask, feat_key_mask):
        # support/query: (b, s|q, f1, d);
        # row_key_mask:  (b, 1, 1, s) additive, masks padded SUPPORT rows.
        # feat_key_mask: (b, 1, 1, f1) additive, masks padded feature cols.
        b, n_s, f1, d = support.shape
        n_q = query.shape[1]

        # --- attention across observations (rows); keys = support ----------
        s_res, q_res = support, query
        support = self.layer_norm1(support)
        query = self.layer_norm1(query)
        s_flat = support.permute(0, 2, 1, 3).reshape(b * f1, n_s, d)
        q_flat = query.permute(0, 2, 1, 3).reshape(b * f1, n_q, d)
        # broadcast row mask (b,1,1,s) -> (b*f1, 1, 1, s)
        rm = None
        if row_key_mask is not None:
            rm = row_key_mask.expand(b, f1, 1, n_s).reshape(b * f1, 1, 1, n_s)
        s_att = self.attention1(s_flat, s_flat, s_flat, attn_mask=rm)
        q_att = self.attention1(q_flat, s_flat, s_flat, attn_mask=rm)
        support = s_res + s_att.reshape(b, f1, n_s, d).permute(0, 2, 1, 3)
        query = q_res + q_att.reshape(b, f1, n_q, d).permute(0, 2, 1, 3)

        # --- MLP block 1 ---------------------------------------------------
        s_res, q_res = support, query
        support = self.layer_norm2(support)
        query = self.layer_norm2(query)
        support = self.linear2(F.gelu(self.linear1(support)))
        query = self.linear2(F.gelu(self.linear1(query)))
        support = s_res + support
        query = q_res + query

        # --- attention across features; keys = features --------------------
        s_res, q_res = support, query
        support = self.layer_norm3(support)
        query = self.layer_norm3(query)
        s_feat = support.reshape(b * n_s, f1, d)
        q_feat = query.reshape(b * n_q, f1, d)
        fm_s = fm_q = None
        if feat_key_mask is not None:
            fm_s = feat_key_mask.expand(b, n_s, 1, f1).reshape(b * n_s, 1, 1, f1)
            fm_q = feat_key_mask.expand(b, n_q, 1, f1).reshape(b * n_q, 1, 1, f1)
        s_fa = self.attention2(s_feat, s_feat, s_feat, attn_mask=fm_s)
        q_fa = self.attention2(q_feat, q_feat, q_feat, attn_mask=fm_q)
        support = s_res + s_fa.reshape(b, n_s, f1, d)
        query = q_res + q_fa.reshape(b, n_q, f1, d)

        # --- MLP block 2 ---------------------------------------------------
        s_res, q_res = support, query
        support = self.layer_norm4(support)
        query = self.layer_norm4(query)
        support = self.linear4(F.gelu(self.linear3(support)))
        query = self.linear4(F.gelu(self.linear3(query)))
        support = s_res + support
        query = q_res + query
        return support, query


# --------------------------------------------------------------------------- #
# Tab2D (top level) — param names identical to upstream
# --------------------------------------------------------------------------- #
CLASSIFICATION = "CLASSIFICATION"
REGRESSION = "REGRESSION"


class Tab2D(nn.Module):
    def __init__(self, dim, dim_output, n_layers, n_heads, task):
        super().__init__()
        self.dim = dim
        self.dim_output = dim_output
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.task = task if isinstance(task, str) else str(task)

        self.x_quantile = Tab2DQuantileEmbeddingX(dim)
        self.x_embedding = Tab2DEmbeddingX(dim)
        if self.task == CLASSIFICATION:
            self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)
        elif self.task == REGRESSION:
            if dim_output == 1:
                self.y_embedding = Tab2DEmbeddingYRegression(dim)
            else:
                self.y_embedding = Tab2DEmbeddingYClasses(dim, dim_output)
        else:
            raise ValueError(f"Task {task} not supported")

        self.layers = nn.ModuleList(
            [Layer(dim, n_heads) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(dim)
        self.final_layer = nn.Linear(dim, dim_output)

    def forward(self, x_support, y_support, x_query,
                padding_features, padding_obs_support, padding_obs_query):
        # x_support: (b, s, f)  y_support: (b, s)  x_query: (b, q, f)
        # padding_features: (b, f) True==pad
        # padding_obs_support: (b, s) True==pad
        b = x_support.shape[0]
        n_q = x_query.shape[1]

        x_support, x_query = self.x_quantile(
            x_support, x_query, padding_obs_support, padding_features)
        x_support = self.x_embedding(x_support)   # (b, s, f, d)
        x_query = self.x_embedding(x_query)       # (b, q, f, d)
        y_support_e, y_query_e = self.y_embedding(
            y_support, padding_obs_support, n_q)  # (b,s,1,d), (b,q,1,d)

        # einops.pack((y, x), "b s * d") -> concat on the feature axis
        support = torch.cat([y_support_e, x_support], dim=2)  # (b, s, f+1, d)
        query = torch.cat([y_query_e, x_query], dim=2)        # (b, q, f+1, d)

        # feature mask: the y column (index 0) is never padded
        pad_y = torch.zeros((b, 1), dtype=torch.bool, device=x_support.device)
        pad_feat_full = torch.cat([pad_y, padding_features], dim=1)  # (b, f+1)
        feat_key_mask = torch.where(
            pad_feat_full[:, None, None, :],
            torch.full_like(pad_feat_full, NEG_INF, dtype=torch.float32)[:, None, None, :],
            torch.zeros_like(pad_feat_full, dtype=torch.float32)[:, None, None, :],
        )
        row_key_mask = torch.where(
            padding_obs_support[:, None, None, :],
            torch.full_like(padding_obs_support, NEG_INF, dtype=torch.float32)[:, None, None, :],
            torch.zeros_like(padding_obs_support, dtype=torch.float32)[:, None, None, :],
        )

        for layer in self.layers:
            support, query = layer(support, query, row_key_mask, feat_key_mask)

        query = self.final_layer_norm(query)
        query = self.final_layer(query)            # (b, q, f+1, C)
        y_query_out = query[:, :, 0, :]            # (b, q, C)

        if self.task == REGRESSION and self.dim_output == 1:
            return y_query_out                     # (b, q, 1)
        return y_query_out                         # (b, q, C)

    @torch.no_grad()
    def load_real_weights(self, state_dict):
        self.load_state_dict(state_dict)
        return self
