"""The TabPFN-3.5 forward the graphs are exported from.

Why not trace `TabPFNV3p5.forward` itself, the way tools/export_tabpfn traces v3?
Four things in it cannot be exported, and patching them one by one would leave a
graph that only works in the process that installed the patches (the trap the
2.5-line masked path fell into, tools/export_tabpfn tests/test_mask_export_isolated.py):

  1. the in-context ECDF uses `torch.searchsorted` (no ONNX operator) -- see ecdf.py;
  2. `num_present_classes = y.max().item() + 1` sizes the decoder's one-hot, a
     data-dependent shape;
  3. the stage 0-2 row/column chunking and the attention-backend dispatcher are
     Python loops and registries keyed on runtime shapes;
  4. the multiclass target-range guard (`(y < 0).any()`), a data-dependent branch.

So, as for TabPFN-3's masked graph (tabpfn_mask_v3.py), the blocks are re-driven
from their own weights -- `blk.attn.q_projection`, ... -- and nothing upstream is
monkeypatched. The result is checked against upstream's own forward on every row
(tests/test_forward_parity.py, and parity_real.py on the released weights).

ONE forward serves both graph contracts. Train rows are the first `n_train` rows
and every attention whose keys are train rows MASKS the others rather than slicing
them, so `n_train` may be a runtime value (masked contract, for MIGraphX's fixed
shapes) or the length of `y` (positional contract, for ORT). Padded feature
columns (`d` < H) are excluded the same way, and the neighbour wrap that builds
each column's feature group wraps at the real width `d`, not the padded one.

Differences from upstream that are NOT approximations:

* the decoder one-hot is always `max_num_classes` wide. Upstream narrows it to the
  classes present and pads the probabilities back with zeros; an absent class has
  an all-zero value column, so attention returns exactly that zero either way.
* `nn.Linear(1, n)` layers (the softmax-scaling MLPs' first layer, the regression
  target encoders) are computed elementwise. Same arithmetic; MIGraphX's GPU
  compile dies on a K=1 matmul feeding a broadcast Mul (docs/ROCM_TABPFN_PLAN.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import tabpfn.architectures.tabpfn_v3_5 as v35
from tabpfn.preprocessing.torch.ops import torch_nanmean

from export_tabpfn35.ecdf import ecdf_ranks

NEG_INF = -1e30  # additive key bias; finite so a fully masked row is not NaN


def _bias(keep, like):
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    neg = torch.full((), NEG_INF, dtype=like.dtype, device=like.device)
    return torch.where(keep, zero, neg).reshape(1, 1, 1, -1)


def _linear_k1(lin, x):
    """`nn.Linear(1, n)` as an elementwise multiply: x[..., 1] -> [..., n]."""
    y = x * lin.weight.reshape(-1)
    return y if lin.bias is None else y + lin.bias


def _scale_queries(scaling, q_BSHD, n):
    """v35.SoftmaxScalingMLP.forward with `n` (the real key count) as a tensor."""
    l0, act, l2 = scaling.base_mlp[0], scaling.base_mlp[1], scaling.base_mlp[2]
    logn = v35._safe_log_seqlen(n, q_BSHD.device, q_BSHD.dtype).reshape(1, 1)
    base = l2(act(_linear_k1(l0, logn))).view(1, 1, scaling.num_heads, scaling.head_dim)
    modulation = 1 + torch.tanh(scaling.query_mlp(q_BSHD))
    return q_BSHD * (base * modulation)


def _attend(q_BSHD, k_BKJD, v_BKJD, keep=None, scaling=None, n=None):
    """Upstream `_batched_scaled_dot_product_attention`, keys optionally masked."""
    if scaling is not None:
        q_BSHD = _scale_queries(scaling, q_BSHD, n)
    q = q_BSHD.transpose(1, 2)
    k = k_BKJD.transpose(1, 2)
    v = v_BKJD.transpose(1, 2)
    heads, kv = q.shape[1], k.shape[1]
    if kv != heads:
        if kv == 1:
            k = k.expand(-1, heads, -1, -1)
            v = v.expand(-1, heads, -1, -1)
        else:
            k = k.repeat_interleave(heads // kv, dim=1)
            v = v.repeat_interleave(heads // kv, dim=1)
    mask = None if keep is None else _bias(keep, q)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2)


def _rope(rope, t_BSHD):
    return rope.rotate_queries_or_keys(t_BSHD.transpose(1, 2)).transpose(1, 2)


def _cross_block(blk, x_BQE, ctx_BVE, keep=None, n=None):
    """v35.CrossAttentionBlock (q/k RMSNorm before the softmax scaling)."""
    a = blk.attn
    B, Q, _ = x_BQE.shape
    V = ctx_BVE.shape[1]
    xq = blk.layernorm_q(x_BQE)
    xkv = blk.layernorm_kv(ctx_BVE)
    q = a.q_norm(a.q_projection(xq).view(B, Q, -1, a.head_dim))
    k = a.k_norm(a.k_projection(xkv).view(B, V, -1, a.head_dim))
    v = a.v_projection(xkv).view(B, V, -1, a.head_dim)
    out = _attend(q, k, v, keep, a.softmax_scaling_layer, n)
    x = x_BQE + a.out_projection(out.reshape(B, Q, a.head_dim * a.num_heads))
    return x + blk.mlp(blk.layernorm2(x))


def _self_block(blk, x_BSE, rope, keep):
    """v35.TransformerBlock.forward: RoPE first, THEN the q/k RMSNorm."""
    a = blk.attention
    B, S, _ = x_BSE.shape
    h = blk.layernorm(x_BSE)
    q = a.q_projection(h).view(B, S, -1, a.head_dim)
    k = a.k_projection(h).view(B, S, -1, a.head_dim)
    v = a.v_projection(h).view(B, S, -1, a.head_dim)
    q, k = a.q_norm(_rope(rope, q)), a.k_norm(_rope(rope, k))
    out = _attend(q, k, v, keep)
    x = x_BSE + a.out_projection(out.reshape(B, S, a.head_dim * a.num_heads))
    return x + blk.mlp(blk.layernorm_mlp(x))


def _cls_readout(blk, q_BQE, ctx_BVE, rope, keep):
    """v35.TransformerBlock.forward_cross: the CLS tokens read every column."""
    a = blk.attention
    B, Q, _ = q_BQE.shape
    V = ctx_BVE.shape[1]
    q = a.q_projection(blk.layernorm(q_BQE)).view(B, Q, -1, a.head_dim)
    c = blk.layernorm(ctx_BVE)
    k = a.k_projection(c).view(B, V, -1, a.head_dim)
    v = a.v_projection(c).view(B, V, -1, a.head_dim)
    q, k = a.q_norm(_rope(rope, q)), a.k_norm(_rope(rope, k))
    out = _attend(q, k, v, keep)
    x = q_BQE + a.out_projection(out.reshape(B, Q, a.head_dim * a.num_heads))
    return x + blk.mlp(blk.layernorm_mlp(x))


def _neighbour_index(H, d, shift, dev):
    """[H] int64: column j -> (j + shift) mod d, by repeated subtraction (no modulo)."""
    j = torch.arange(H, device=dev) + shift
    for _ in range(shift):
        j = torch.where(j >= d, j - d, j)
    return torch.clamp(j, min=0, max=H - 1)


def _group(model, parts_RiBC, d):
    """[B, Ri, H, 3G]: upstream `_group_features` over (x, nan-indicator, ecdf).

    Gathered in the row-major layout and transposed once after the concat (a
    Concat of Transposes kills MIGraphX's `simplify_reshapes`, see tabpfn_mask_v3).
    """
    H = parts_RiBC[0].shape[-1]
    idx = [_neighbour_index(H, d, 2 ** i, parts_RiBC[0].device)
           for i in range(model.feature_group_size)]
    grouped = torch.cat([torch.stack([t[:, :, i] for i in idx], dim=-1) for t in parts_RiBC],
                        dim=-1)                                                      # [Ri,B,H,3G]
    return grouped.transpose(0, 1)


def _masked_mean(x, valid):
    """nan_to_num(nanmean(x over `valid` & finite rows), 0) -- upstream's imputation mean."""
    nan = torch.full((), float("nan"), dtype=x.dtype, device=x.device)
    m = torch_nanmean(torch.where(valid & torch.isfinite(x), x, nan), axis=0)
    return torch.where(torch.isnan(m), torch.zeros_like(m), m)


def _nan_inf_indicator(x):
    """v35._generate_nan_and_inf_indicator with float operands (bool*float does not export)."""
    dt = x.dtype
    return (torch.isnan(x).to(dt) * v35._NAN_INDICATOR
            + torch.isposinf(x).to(dt) * v35._INFINITY_INDICATOR
            + torch.isneginf(x).to(dt) * v35._NEG_INFINITY_INDICATOR)


def _embed_cells(xe, g_BRCG):
    """FourierPlusMetadataFeatureGroupEmbedder._embed, without its autocast context."""
    G = xe.group_size
    fr = xe.fourier
    proj = g_BRCG[..., :G].unsqueeze(-1) * fr.frequencies                           # [...,G,F]
    fourier = fr.in_linear(torch.cat([proj.sin(), proj.cos()], dim=-1).sum(dim=-2))
    ranks = g_BRCG[..., -G:]
    meta = torch.cat([g_BRCG[..., :-G],
                      v35._ecdf_fourier_features(ranks, xe.ecdf_num_frequencies).flatten(-2)],
                     dim=-1)
    return xe.layernorm(fourier + F.linear(meta, xe.metadata_linear.weight))


def rows_v35(model, x_RiBC, y_Ri, n_train, d, *, task: str):
    """Head output for EVERY row. x [Ri, 1, H], y [Ri] (only the first n_train read).

    Returns [Ri, 1, C]: classification -> `max_num_classes` logits (upstream's
    `log(clamp(p, 1e-5) + 3e-5)`), regression -> `num_buckets` bar logits. Rows
    < n_train are the train rows decoded the same way (the embedding route).
    """
    dev = x_RiBC.device
    Ri, B, H = x_RiBC.shape
    clf = task == "classification"
    n = n_train.reshape(()).to(torch.int64) if n_train.dim() else n_train.to(torch.int64)
    rowi = torch.arange(Ri, device=dev)
    train = rowi < n                                                                # [Ri]
    train_b = train.reshape(-1, 1, 1)
    nan = torch.full((), float("nan"), dtype=x_RiBC.dtype, device=dev)

    # -- _preprocess_raw: indicators, imputation, scaling, ECDF, over TRAIN rows --
    ind = _nan_inf_indicator(x_RiBC)
    finite = torch.isfinite(x_RiBC)
    x = torch.where(finite, x_RiBC, _masked_mean(x_RiBC, train_b).unsqueeze(0).expand_as(x_RiBC))
    fit = model.standard_scaler.fit(torch.where(train_b, x, nan))
    fit["std"] = torch.where(n == 1, torch.ones_like(fit["std"]), fit["std"])
    x = torch.where(finite, x, fit["mean"].unsqueeze(0).expand_as(x))
    ecdf = ecdf_ranks(x.transpose(0, 1), n, model.ecdf_num_buckets).transpose(0, 1)  # [Ri,B,H]
    x = model.standard_scaler.transform(x, fitted_cache=fit)
    g = _group(model, (x, ind, ecdf), d)                                           # [B,Ri,H,3G]

    # -- the target: imputed over the train rows, embedded on train rows only --------
    y = y_Ri.reshape(Ri)
    y_finite = torch.isfinite(y)
    y_mean = _masked_mean(y.reshape(Ri, 1), train.reshape(Ri, 1))[0]
    y_imp = torch.where(y_finite, y, y_mean)
    if clf:
        y_imp = torch.where(y_finite, y_imp, y_imp.ceil())
    # Non-train rows carry the engine's -100 sentinel (or padding): an embedding
    # lookup must stay legal, and nothing reads them (train mask below).
    y_use = torch.where(train, y_imp, torch.zeros_like(y_imp))
    train_row = train.reshape(1, Ri, 1)

    def embed_y(enc, ln):
        if clf:
            e = enc["multiclass"](y_use.reshape(1, Ri))
        else:
            e = _linear_k1(enc["regression"], y_use.reshape(1, Ri, 1))
        return ln(e)

    # -- stage 1: cell embedding + target-aware column embedding ------------------
    x_emb = _embed_cells(model.x_embed, g)                                          # [B,Ri,H,E]
    x_emb = x_emb + torch.where(train_row, embed_y(model.col_y_encoder, model.col_y_layernorm),
                                0.0).unsqueeze(2)
    E = x_emb.shape[-1]

    # -- stage 2a: per-column distribution embedder (inducing points read train rows) -
    x_flat = x_emb.transpose(1, 2).reshape(B * H, Ri, E)
    for blk in model.feature_distribution_embedder.layers:
        ind_v = blk.inducing_vectors.unsqueeze(0).expand(B * H, -1, -1)
        hidden = _cross_block(blk.cross_attn_block1, ind_v, x_flat, train, n)
        x_flat = _cross_block(blk.cross_attn_block2, x_flat, hidden)
    x_emb = x_flat.reshape(B, H, Ri, E).transpose(1, 2)                            # [B,Ri,H,E]

    # -- stage 2b: column aggregation; padded columns masked out as keys ----------
    agg = model.column_aggregator
    ncls = agg.num_cls_tokens
    col_keep = torch.cat([torch.ones(ncls, dtype=torch.bool, device=dev),
                          torch.arange(H, device=dev) < d])
    cls = agg.cls_tokens.expand(B, Ri, ncls, E)
    xs = torch.cat((cls, x_emb), dim=2).reshape(B * Ri, ncls + H, E)
    for blk in agg.blocks[:-1]:
        xs = _self_block(blk, xs, agg.rope, col_keep)
    cls_out = _cls_readout(agg.blocks[-1], xs[:, :ncls], xs, agg.rope, col_keep)
    row = agg.out_ln(cls_out).reshape(B, Ri, ncls * E)                              # [B,Ri,D]

    # -- stage 3: ICL over the train rows -----------------------------------------
    row = row + torch.where(train_row, embed_y(model.icl_y_encoder, model.icl_y_layernorm), 0.0)
    for blk in model.icl_blocks:
        at = blk.icl_attention
        h = blk.layernorm(row)
        q = at.q_norm(at.q_projection(h).view(B, Ri, at.num_heads, at.head_dim))
        k = at.k_norm(at.k_projection(h).view(B, Ri, at.num_kv_heads, at.head_dim))
        v = at.v_projection(h).view(B, Ri, at.num_kv_heads, at.head_dim)
        out = _attend(q, k, v, train, at.softmax_scaling_layer, n)
        if at.num_kv_heads_test is not None:
            nt = at.num_kv_heads_test
            out_test = _attend(q, k[:, :, :nt], v[:, :, :nt], train, at.softmax_scaling_layer, n)
            out = torch.where(train.reshape(1, Ri, 1, 1), out, out_test)
        row = row + at.out_projection(out.reshape(B, Ri, at.head_dim * at.num_heads))
        row = row + blk.mlp(blk.layernorm_mlp(row))
    row = model.output_norm(row)                                                    # [B,Ri,D]

    # -- heads ---------------------------------------------------------------------
    heads = model.heads
    if clf:
        e = heads.mlp_classification(row)
        dec = heads.many_class_decoder
        q = dec.q_projection(e).view(B, Ri, dec.num_heads, dec.head_dim)
        k = dec.k_projection(e).view(B, Ri, dec.num_heads, dec.head_dim)
        # Upstream one-hots the RAW train labels, zeroing non-finite ones.
        y_lab = torch.where(y_finite & train, y, torch.zeros_like(y))
        onehot = F.one_hot(y_lab.long(), num_classes=dec.max_num_classes).to(q.dtype)
        onehot = torch.where((y_finite & train).reshape(Ri, 1), onehot, torch.zeros_like(onehot))
        onehot = onehot.reshape(1, Ri, 1, -1).expand(-1, -1, dec.num_heads, -1)
        probs = _class_attention(q, k, onehot, train, dec.softmax_scaling_layer, n)
        out = torch.log(torch.clamp(probs.mean(2).transpose(0, 1), min=1e-5) + 3e-5)
    else:
        out = heads.output_projection(heads.mlp_regression(row).transpose(0, 1))
    return torch.nan_to_num(out, nan=0.0)


def _class_attention(q_BSHD, k_BJHD, v_BJHT, keep, scaling, n):
    """v35._chunked_class_attention with the keys masked."""
    B, S, H, D = q_BSHD.shape
    T = v_BJHT.shape[-1]
    chunks = math.ceil(T / D)
    pad = chunks * D - T
    if pad > 0:
        v_BJHT = F.pad(v_BJHT, (0, pad))
    J = v_BJHT.shape[1]
    v_f = v_BJHT.reshape(B, J, H, chunks, D).permute(0, 3, 1, 2, 4).reshape(B * chunks, J, H, D)
    q_f = q_BSHD.unsqueeze(1).expand(-1, chunks, -1, -1, -1).reshape(B * chunks, S, H, D)
    k_f = k_BJHD.unsqueeze(1).expand(-1, chunks, -1, -1, -1).reshape(B * chunks, J, H, D)
    out = _attend(q_f, k_f, v_f, keep, scaling, n)
    return out.reshape(B, chunks, S, H, D).permute(0, 2, 3, 1, 4).reshape(B, S, H, chunks * D)[..., :T]
