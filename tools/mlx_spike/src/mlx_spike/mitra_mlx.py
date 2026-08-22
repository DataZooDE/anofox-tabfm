"""S-M2 -- Mitra (Tab2D) forward, ported to MLX.

A line-for-line port of
`tools/export_mitra/src/export_mitra/mitra_model_patched.py` plus its
`ExportWrapper`, which together define what the shipped ONNX graph computes.
That file is the authority; this one is deliberately shaped like it so the two
can be read side by side. Where MLX forced a deviation it is commented with
WHY, because an unexplained deviation is where parity goes to die.

Weights load straight from the released safetensors (`mx.load` mmaps them) --
no conversion step, exactly as the plan predicted: the tensor map is
transform-free, so ONNX initializer `m.layers.0.attention1.q.weight` is
safetensors key `layers.0.attention1.q.weight`.

Batch is always 1 (one table per call), but the b axis is kept throughout so
the shapes match the torch source rather than a simplified special case.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

NEG_INF = -1.0e9  # must match mitra_model_patched.NEG_INF: a *finite* sentinel
LN_EPS = 1e-5     # torch.nn.LayerNorm default


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def linear(x: mx.array, w: mx.array, b: mx.array | None) -> mx.array:
    """torch.nn.Linear: weight is (out, in) and the op is x @ W.T + b."""
    y = x @ w.T
    return y if b is None else y + b


def layer_norm(x: mx.array, w: mx.array, b: mx.array) -> mx.array:
    return mx.fast.layer_norm(x, w, b, LN_EPS)


def gelu(x: mx.array) -> mx.array:
    """F.gelu with approximate='none' (the torch default) -- the exact erf form.
    The tanh approximation differs by ~1e-3, which is above our parity bar."""
    return x * 0.5 * (1.0 + mx.erf(x * 0.70710678118654752440))


def softmax_attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None,
                      *, fast: bool) -> mx.array:
    """(b, h, tq, hd) x (b, h, tk, hd) -> (b, h, tq, hd), additive mask."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    if fast:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    scores = (q * scale) @ k.swapaxes(-1, -2)
    if mask is not None:
        scores = scores + mask
    return mx.softmax(scores.astype(mx.float32), axis=-1).astype(v.dtype) @ v


# --------------------------------------------------------------------------- #
# Weight access
# --------------------------------------------------------------------------- #
class Weights:
    """The safetensors state dict, addressed by upstream parameter name."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.w = mx.load(str(self.path))

    def __getitem__(self, name: str) -> mx.array:
        try:
            return self.w[name]
        except KeyError as exc:
            raise KeyError(f"{name!r} not in {self.path.name}") from exc

    @property
    def n_layers(self) -> int:
        return 1 + max(int(k.split(".")[1]) for k in self.w if k.startswith("layers."))

    @property
    def dim(self) -> int:
        return int(self["final_layer_norm.weight"].shape[0])

    @property
    def dim_output(self) -> int:
        return int(self["final_layer.weight"].shape[0])


# --------------------------------------------------------------------------- #
# Tab2DQuantileEmbeddingX -- parameter-free rank normalization
# --------------------------------------------------------------------------- #
def quantile_embedding(x_support: mx.array, x_query: mx.array, padding_mask: mx.array,
                       n_quantiles: int = 999) -> tuple[mx.array, mx.array]:
    """x_*: (b, n, f); padding_mask: (b, s) with True == padded.

    Statistics are taken over VALID support rows only -- padded rows are pushed
    to +inf so they sort above the interpolation window and never contribute.
    """
    b, s, f = x_support.shape
    valid = ~padding_mask                                   # (b, s)
    seq_len = valid.astype(mx.float32).sum(axis=1)          # (b,)
    seq_len_c = mx.maximum(seq_len, 1.0)

    big = mx.full(x_support.shape, mx.inf, dtype=x_support.dtype)
    x_masked = mx.where(valid[:, :, None], x_support, big)
    sorted_x = mx.sort(x_masked, axis=1)                    # (b, s, f) ascending

    q = mx.arange(1, n_quantiles + 1, dtype=mx.float32) / (n_quantiles + 1)
    pos = q[None, :] * (seq_len_c - 1.0)[:, None]           # (b, Q)
    lo = mx.maximum(mx.floor(pos), 0.0)
    hi = mx.minimum(lo + 1.0, (seq_len_c - 1.0)[:, None])
    frac = pos - lo
    lo_i = mx.broadcast_to(lo.astype(mx.int32)[:, :, None], (b, n_quantiles, f))
    hi_i = mx.broadcast_to(hi.astype(mx.int32)[:, :, None], (b, n_quantiles, f))
    q_lo = mx.take_along_axis(sorted_x, lo_i, axis=1)       # (b, Q, f)
    q_hi = mx.take_along_axis(sorted_x, hi_i, axis=1)
    quantiles = q_lo + frac[:, :, None] * (q_hi - q_lo)

    def bucketize(vals: mx.array) -> mx.array:
        # torch.bucketize(right=False) == count of quantiles strictly below.
        # (b, n, Q, f) is the peak allocation of the whole forward: it grows as
        # rows x 999 x features, so it dominates memory long before the
        # transformer does. S-M4 measures the ceiling this imposes.
        cmp = quantiles[:, None, :, :] < vals[:, :, None, :]
        return cmp.astype(mx.float32).sum(axis=2)

    xs = bucketize(x_support) / seq_len_c[:, None, None]
    xq = bucketize(x_query) / seq_len_c[:, None, None]

    zeros = mx.zeros_like(xs)
    xs = mx.where(valid[:, :, None], xs, zeros)
    mean = xs.sum(axis=1, keepdims=True) / seq_len_c[:, None, None]
    xs = xs - mean
    xq = xq - mean

    xs = mx.where(valid[:, :, None], xs, mx.zeros_like(xs))
    var = (xs * xs).sum(axis=1, keepdims=True) / seq_len_c[:, None, None]
    std = mx.sqrt(var)
    xs = xs / std
    xq = xq / std

    zero_var = var == 0
    xs = mx.where(zero_var, mx.zeros_like(xs), xs)
    xq = mx.where(zero_var, mx.zeros_like(xq), xq)
    return xs, xq


# --------------------------------------------------------------------------- #
# Attention / layer
# --------------------------------------------------------------------------- #
def multihead_attention(w: Weights, prefix: str, n_heads: int, query: mx.array,
                        key: mx.array, mask: mx.array | None, *, fast: bool) -> mx.array:
    """query/key: (b, t, dim). Value is always the key tensor here, as in the
    source (self-attention, or query-attends-to-support with key == value)."""
    b, tq, dim = query.shape
    tk = key.shape[1]
    hd = dim // n_heads
    q = linear(query, w[f"{prefix}.q.weight"], w[f"{prefix}.q.bias"])
    k = linear(key, w[f"{prefix}.k.weight"], w[f"{prefix}.k.bias"])
    v = linear(key, w[f"{prefix}.v.weight"], w[f"{prefix}.v.bias"])
    q = q.reshape(b, tq, n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(b, tk, n_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(b, tk, n_heads, hd).transpose(0, 2, 1, 3)
    o = softmax_attention(q, k, v, mask, fast=fast)
    o = o.transpose(0, 2, 1, 3).reshape(b, tq, dim)
    return linear(o, w[f"{prefix}.o.weight"], w[f"{prefix}.o.bias"])


def layer_forward(w: Weights, i: int, n_heads: int, support: mx.array, query: mx.array,
                  row_key_mask: mx.array, feat_key_mask: mx.array, *, fast: bool
                  ) -> tuple[mx.array, mx.array]:
    p = f"layers.{i}"
    b, n_s, f1, d = support.shape
    n_q = query.shape[1]

    def ln(name: str, t: mx.array) -> mx.array:
        return layer_norm(t, w[f"{p}.{name}.weight"], w[f"{p}.{name}.bias"])

    def mlp(t: mx.array, a: str, c: str) -> mx.array:
        h = gelu(linear(t, w[f"{p}.{a}.weight"], w[f"{p}.{a}.bias"]))
        return linear(h, w[f"{p}.{c}.weight"], w[f"{p}.{c}.bias"])

    # --- attention across observations (rows); keys are the support rows -----
    s_res, q_res = support, query
    s_n, q_n = ln("layer_norm1", support), ln("layer_norm1", query)
    s_flat = s_n.transpose(0, 2, 1, 3).reshape(b * f1, n_s, d)
    q_flat = q_n.transpose(0, 2, 1, 3).reshape(b * f1, n_q, d)
    rm = mx.broadcast_to(row_key_mask, (b, f1, 1, n_s)).reshape(b * f1, 1, 1, n_s)
    s_att = multihead_attention(w, f"{p}.attention1", n_heads, s_flat, s_flat, rm, fast=fast)
    q_att = multihead_attention(w, f"{p}.attention1", n_heads, q_flat, s_flat, rm, fast=fast)
    support = s_res + s_att.reshape(b, f1, n_s, d).transpose(0, 2, 1, 3)
    query = q_res + q_att.reshape(b, f1, n_q, d).transpose(0, 2, 1, 3)

    # --- MLP block 1 ---------------------------------------------------------
    support = support + mlp(ln("layer_norm2", support), "linear1", "linear2")
    query = query + mlp(ln("layer_norm2", query), "linear1", "linear2")

    # --- attention across features; keys are the feature columns -------------
    s_res, q_res = support, query
    s_feat = ln("layer_norm3", support).reshape(b * n_s, f1, d)
    q_feat = ln("layer_norm3", query).reshape(b * n_q, f1, d)
    fm_s = mx.broadcast_to(feat_key_mask, (b, n_s, 1, f1)).reshape(b * n_s, 1, 1, f1)
    fm_q = mx.broadcast_to(feat_key_mask, (b, n_q, 1, f1)).reshape(b * n_q, 1, 1, f1)
    s_fa = multihead_attention(w, f"{p}.attention2", n_heads, s_feat, s_feat, fm_s, fast=fast)
    q_fa = multihead_attention(w, f"{p}.attention2", n_heads, q_feat, q_feat, fm_q, fast=fast)
    support = s_res + s_fa.reshape(b, n_s, f1, d)
    query = q_res + q_fa.reshape(b, n_q, f1, d)

    # --- MLP block 2 ---------------------------------------------------------
    support = support + mlp(ln("layer_norm4", support), "linear3", "linear4")
    query = query + mlp(ln("layer_norm4", query), "linear3", "linear4")
    return support, query


# --------------------------------------------------------------------------- #
# Top level -- Tab2D.forward composed with ExportWrapper
# --------------------------------------------------------------------------- #
class MitraMLX:
    """The engine's tensor contract, computed on Metal.

    forward(x[1,T,H], y[1,T], train_size, d) -> logits[1,T,C]
    """

    def __init__(self, weights: Weights, *, task: str = "classification",
                 n_heads: int = 4, fast_sdpa: bool = True):
        self.w = weights
        self.task = task
        self.n_heads = n_heads
        self.fast_sdpa = fast_sdpa
        self.n_layers = weights.n_layers
        self.dim = weights.dim
        self.dim_output = weights.dim_output

    def _embed_y(self, y_support: mx.array, padding_obs_support: mx.array,
                 n_obs_query: int) -> tuple[mx.array, mx.array]:
        b = y_support.shape[0]
        if self.task == "classification":
            ids = mx.clip(y_support.astype(mx.int32), 0, self.dim_output - 1)
            ids = mx.where(padding_obs_support, mx.zeros_like(ids), ids)
            emb = mx.take(self.w["y_embedding.y_embedding.weight"], ids, axis=0)
        else:
            emb = linear(y_support.astype(mx.float32)[:, :, None],
                         self.w["y_embedding.y_embedding.weight"],
                         self.w["y_embedding.y_embedding.bias"])
        emb = mx.where(padding_obs_support[:, :, None], mx.zeros_like(emb), emb)
        y_support_e = emb[:, :, None, :]                      # (b, s, 1, d)
        mask_vec = self.w["y_embedding.y_mask.weight"][0]     # nn.Embedding(1, dim)
        y_query_e = mx.broadcast_to(mask_vec, (b, n_obs_query, 1, self.dim))
        return y_support_e, y_query_e

    def __call__(self, x: mx.array, y: mx.array, train_size: int, d: int) -> mx.array:
        b, t, h = x.shape
        # ExportWrapper: the whole table is both support and query; train_size
        # and d become masks, never slices, so no shape is data-dependent.
        ar_t = mx.arange(t)
        ar_h = mx.arange(h)
        padding_obs_support = (ar_t >= train_size)[None, :]   # (1, T) True == pad
        padding_features = (ar_h >= d)[None, :]               # (1, H)

        xs, xq = quantile_embedding(x, x, padding_obs_support)
        xw = self.w["x_embedding.x_embedding.weight"]
        xb = self.w["x_embedding.x_embedding.bias"]
        x_support = linear(xs[:, :, :, None], xw, xb)         # (b, s, f, d)
        x_query = linear(xq[:, :, :, None], xw, xb)
        y_support_e, y_query_e = self._embed_y(y, padding_obs_support, t)

        # einops.pack((y, x), "b s * d") -- the label column is feature 0.
        support = mx.concatenate([y_support_e, x_support], axis=2)  # (b, s, f+1, d)
        query = mx.concatenate([y_query_e, x_query], axis=2)

        pad_y = mx.zeros((b, 1), dtype=mx.bool_)
        pad_feat_full = mx.concatenate([pad_y, padding_features], axis=1)  # (b, f+1)
        feat_key_mask = mx.where(pad_feat_full, NEG_INF, 0.0)[:, None, None, :]
        row_key_mask = mx.where(padding_obs_support, NEG_INF, 0.0)[:, None, None, :]

        for i in range(self.n_layers):
            support, query = layer_forward(self.w, i, self.n_heads, support, query,
                                           row_key_mask, feat_key_mask, fast=self.fast_sdpa)

        query = layer_norm(query, self.w["final_layer_norm.weight"],
                           self.w["final_layer_norm.bias"])
        query = linear(query, self.w["final_layer.weight"], self.w["final_layer.bias"])
        return query[:, :, 0, :]                               # (b, T, C)
