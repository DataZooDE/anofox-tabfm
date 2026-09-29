"""The masked whole-model forward for TabPFN-3.

v3 is not the 2.5 line with a different name. It has no feature groups to
normalise and no constant-column removal, but it consumes the train/test split in
seven places that all had to become values instead of slice bounds:

  1. imputation means and the standard scaler are fitted on the first N rows;
  2. the target is imputed over the first N rows, and only those rows are embedded;
  3. the per-column inducing points attend to the N train rows only (keys);
  4. the target embedding is added to the train rows only (twice: column stage and
     ICL stage);
  5. ICL attention: keys are train rows, train queries use every KV head and test
     queries use only the first `icl_num_kv_heads_test`;
  6. the many-class decoder attends the queries over the train rows only;
  7. every softmax-scaling MLP takes log(number of KEYS) -- a value, not a shape.

Padded FEATURES need one more thing. v3 builds each column's input from its
neighbours with `torch.roll(x, -2**i, dim=columns)`, which wraps at the PADDED
width; the neighbours of the last real column would be padding instead of column 0.
The wrap is redone with an index gather that wraps at the REAL width `d`, using
subtraction, never division or modulo: MIGraphX's GPU target miscomputes integer
floor-division (see tabpfn_mask_forward), and this graph is meant to run there.
Padded columns become extra tokens in the column aggregator, so they are masked
out as keys.

Nothing upstream is monkeypatched. The blocks are re-driven from their own weights
(`blk.attn.q_projection` ...), so the module never has to be patched in the right
process, which is the trap the 2.5-line path fell into (tests/test_mask_export_isolated.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import tabpfn.architectures.tabpfn_v3 as v3
from tabpfn.preprocessing.torch.ops import torch_nanmean

from export_tabpfn import tabpfn_mask_patches as mp
from export_tabpfn.tabpfn_patched import _patched_nan_inf_indicator


def _bias(keep, like):
    """[1,1,1,K] additive key bias: 0 on real keys, NEG_INF on excluded ones."""
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    neg = torch.full((), mp.NEG_INF, dtype=like.dtype, device=like.device)
    return torch.where(keep, zero, neg).reshape(1, 1, 1, -1)


def _linear_k1(lin, x):
    """`nn.Linear(1, n)` as an elementwise multiply: x[..., 1] -> [..., n].

    Exactly what the layer computes, but without a matmul whose contraction
    dimension is 1. Exported as a matmul it feeding a broadcast Mul kills MIGraphX's
    GPU compile in `simplify_reshapes` ("cannot create std::vector larger than
    max_size()"); ORT and the reference target are fine, so only the GPU sees it.
    Guarded by tests/test_mask_export_isolated.py::test_masked_graph_has_no_k1_matmul.
    """
    y = x * lin.weight.reshape(-1)
    return y if lin.bias is None else y + lin.bias


def _scale_queries(scaling, q_BSHD, n):
    """v3.SoftmaxScalingMLP.forward, with its Linear(1, n) layer done elementwise.

    q * base_mlp(log n) * (1 + tanh(query_mlp(q))), n being the REAL key count as a
    tensor. Upstream's own `_safe_log_seqlen` is used for the log, so the value is the
    same code path.
    """
    l0, act, l2 = scaling.base_mlp[0], scaling.base_mlp[1], scaling.base_mlp[2]
    logn = v3._safe_log_seqlen(n, q_BSHD.device, q_BSHD.dtype).reshape(1, 1)
    base = l2(act(_linear_k1(l0, logn))).view(1, 1, scaling.num_heads, scaling.head_dim)
    modulation = 1 + torch.tanh(scaling.query_mlp(q_BSHD))
    return q_BSHD * (base * modulation)


def _attend(q_BSHD, k_BKJD, v_BKJD, keep, scaling=None, n=None):
    """Upstream `_batched_scaled_dot_product_attention` with the keys masked.

    `n` is the REAL key count, as a tensor: upstream reads it off `k.shape[1]`,
    which here is the padded length. Fewer KV heads than query heads are expanded
    by hand rather than through GQA support, so the graph exports the same
    everywhere.
    """
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


def _cross_block(blk, x_BQE, ctx_BVE, keep, n):
    """v3.CrossAttentionBlock, keys optionally masked."""
    a = blk.attn
    B, Q, _ = x_BQE.shape
    V = ctx_BVE.shape[1]
    xq = blk.layernorm_q(x_BQE)
    xkv = blk.layernorm_kv(ctx_BVE)
    q = a.q_projection(xq).view(B, Q, -1, a.head_dim)
    k = a.k_projection(xkv).view(B, V, -1, a.head_dim)
    v = a.v_projection(xkv).view(B, V, -1, a.head_dim)
    out = _attend(q, k, v, keep, a.softmax_scaling_layer, n)
    x = x_BQE + a.out_projection(out.reshape(B, Q, a.head_dim * a.num_heads))
    return x + blk.mlp(blk.layernorm2(x))


def _rope(rope, t_BSHD):
    return rope.rotate_queries_or_keys(t_BSHD.transpose(1, 2)).transpose(1, 2)


def _transformer_block(blk, x_BSE, rope, keep):
    """v3.TransformerBlock.forward (self-attention over the column tokens)."""
    a = blk.attention
    B, S, _ = x_BSE.shape
    h = blk.layernorm(x_BSE)
    q = a.q_projection(h).view(B, S, -1, a.head_dim)
    k = a.k_projection(h).view(B, S, -1, a.head_dim)
    v = a.v_projection(h).view(B, S, -1, a.head_dim)
    if rope is not None:
        q, k = _rope(rope, q), _rope(rope, k)
    out = _attend(q, k, v, keep)
    x = x_BSE + a.out_projection(out.reshape(B, S, a.head_dim * a.num_heads))
    return x + blk.mlp(blk.layernorm_mlp(x))


def _cls_readout(blk, q_BQE, ctx_BVE, rope, keep):
    """v3.TransformerBlock.forward_cross: the CLS tokens read every column."""
    a = blk.attention
    B, Q, E = q_BQE.shape
    V = ctx_BVE.shape[1]
    q = a.q_projection(blk.layernorm(q_BQE)).view(B, Q, -1, a.head_dim)
    c = blk.layernorm(ctx_BVE)
    k = a.k_projection(c).view(B, V, -1, a.head_dim)
    v = a.v_projection(c).view(B, V, -1, a.head_dim)
    if rope is not None:
        q, k = _rope(rope, q), _rope(rope, k)
    out = _attend(q, k, v, keep)
    x = q_BQE + a.out_projection(out.reshape(B, Q, a.head_dim * a.num_heads))
    return x + blk.mlp(blk.layernorm_mlp(x))


def _neighbour_index(H, d, shift, dev):
    """[H] int64: column j -> (j + shift) mod d, without a modulo.

    `torch.roll(x, -shift)` over the real width. Wrapping is repeated
    subtraction: j < d, so at most `shift` subtractions are ever needed (d == 1).
    Padded columns (j >= d) get an arbitrary in-range index; nothing reads them.
    """
    j = torch.arange(H, device=dev) + shift
    for _ in range(shift):
        j = torch.where(j >= d, j - d, j)
    return torch.clamp(j, min=0, max=H - 1)


def _group(model, x_RiBC, ind_RiBC, d):
    """[B, Ri, H, G]: each column's grouped neighbours, from [Ri, B, H] inputs.

    Gathered in the ORIGINAL row-major layout and transposed ONCE, after the
    concatenation. The obvious order -- transpose to [B, Ri, H] first, gather, then
    stack -- makes a Concat whose inputs are all Transposes, and MIGraphX's GPU
    pipeline rewrites that in `find_concat_transpose`, which asserts
    `s.transposed()` (simplify_reshapes.cpp:845) on this input. Release builds
    surface it as `simplify_reshapes: cannot create std::vector larger than
    max_size()`. The reference target never runs that pass, so only the GPU sees it.
    """
    size = model.feature_group_size
    H = x_RiBC.shape[-1]
    idx = [_neighbour_index(H, d, 2 ** i, x_RiBC.device) for i in range(size)]
    grouped = torch.stack([x_RiBC[:, :, i] for i in idx], dim=-1)                  # [Ri,B,H,G]
    if ind_RiBC is not None:
        grouped = torch.cat([grouped, torch.stack([ind_RiBC[:, :, i] for i in idx], dim=-1)], dim=-1)
    return grouped.transpose(0, 1)


def _masked_impute_mean(x, valid):
    """nan_to_num(nanmean(x over `valid` & finite rows), 0): upstream's imputation mean."""
    nan = torch.full((), float("nan"), dtype=x.dtype, device=x.device)
    m = torch_nanmean(torch.where(valid & torch.isfinite(x), x, nan), axis=0)
    return torch.where(torch.isnan(m), torch.zeros_like(m), m)


def masked_rows_v3(model, x_RiBC, y_Ri, train, real, d):
    """Logits for every row of a padded sequence. Returns [Ri, B, C_out]. B must be 1.

    `real` is accepted for interface parity with the 2.5 path and is not needed: v3
    has no statistic that looks at all rows, and padded rows are only ever queries.
    """
    del real
    dev = x_RiBC.device
    Ri, B, H = x_RiBC.shape
    clf = model.task_type == "multiclass"
    n_train = train.sum()
    train_b = train.reshape(-1, 1, 1)
    nan = torch.full((), float("nan"), dtype=x_RiBC.dtype, device=dev)

    # -- 1. NaN/Inf indicators, imputation and scaling over the TRAIN rows ------
    indicator = _patched_nan_inf_indicator(x_RiBC) if model.use_nan_indicators else None
    x = torch.where(torch.isfinite(x_RiBC), x_RiBC,
                    _masked_impute_mean(x_RiBC, train_b).unsqueeze(0).expand_as(x_RiBC))
    fit = model.standard_scaler.fit(torch.where(train_b, x, nan))
    # Upstream: `if x.shape[0] == 1: std = 1`, on the sliced train rows.
    fit["std"] = torch.where(n_train == 1, torch.ones_like(fit["std"]), fit["std"])
    x = model.standard_scaler.transform(x, fitted_cache=fit)

    x_g = _group(model, x, indicator, d)                                              # [B,Ri,H,G]

    # -- 2. the target, over the train rows only --------------------------------
    y = y_Ri.reshape(Ri)
    finite = torch.isfinite(y)
    # Indexed, NOT .reshape(()): a Reshape to an empty dims list makes MIGraphX read
    # ZERO elements and refuse the graph; ORT accepts it, so only the GPU sees it
    # (tests/test_mask_export_isolated.py::test_masked_graph_has_no_scalar_reshape).
    y_mean = _masked_impute_mean(y.reshape(Ri, 1), train.reshape(Ri, 1))[0]
    y_imp = torch.where(finite, y, y_mean)
    if clf:
        y_imp = torch.where(finite, y_imp, y_imp.ceil())      # classes: ceil the imputed
    # Class ids index an embedding, and the -100 sentinel is out of range: zero every
    # non-train row. Nothing reads them (train mask below) but the lookup must be legal.
    y_use = torch.where(train, y_imp, torch.zeros_like(y_imp))

    def embed_y(enc):
        return enc(y_use.reshape(1, Ri)) if clf else _linear_k1(enc, y_use.reshape(1, Ri, 1))

    train_row = train.reshape(1, Ri, 1)

    # -- 3. per-column distribution embedding -----------------------------------
    x_emb = model.x_embed(x_g)                                                     # [B,Ri,H,E]
    x_emb = x_emb + torch.where(train_row, embed_y(model.col_y_encoder), 0.0).unsqueeze(2)
    E = x_emb.shape[-1]
    x_flat = x_emb.transpose(1, 2).reshape(B * H, Ri, E)
    layers = model.feature_distribution_embedder.layers
    for blk in layers:
        ind = blk.inducing_vectors.unsqueeze(0).expand(B * H, -1, -1)
        hidden = _cross_block(blk.cross_attn_block1, ind, x_flat, train, n_train)
        x_flat = _cross_block(blk.cross_attn_block2, x_flat, hidden, None, None)
    x_emb = x_flat.reshape(B, H, Ri, E).transpose(1, 2)                            # [B,Ri,H,E]

    # -- 4. column aggregation; padded columns are masked out as keys -----------
    agg = model.column_aggregator
    ncls = agg.num_cls_tokens
    col_keep = torch.cat([torch.ones(ncls, dtype=torch.bool, device=dev),
                          torch.arange(H, device=dev) < d])
    cls = agg.cls_tokens.expand(B, Ri, ncls, E).to(dev)
    x = torch.cat((cls, x_emb), dim=2).reshape(B * Ri, ncls + H, E)
    for blk in agg.blocks[:-1]:
        x = _transformer_block(blk, x, agg.rope, col_keep)
    cls_out = _cls_readout(agg.blocks[-1], x[:, :ncls], x, agg.rope, col_keep)
    row = agg.out_ln(cls_out).reshape(B, Ri, ncls * E)                             # [B,Ri,D]

    # -- 5. ICL over the train rows ---------------------------------------------
    row = row + torch.where(train_row, embed_y(model.icl_y_encoder), 0.0)
    for blk in model.icl_blocks:
        at = blk.icl_attention
        h = blk.layernorm(row)
        q = at.q_projection(h).view(B, Ri, at.num_heads, at.head_dim)
        k = at.k_projection(h).view(B, Ri, at.num_kv_heads, at.head_dim)
        v = at.v_projection(h).view(B, Ri, at.num_kv_heads, at.head_dim)
        out = _attend(q, k, v, train, at.softmax_scaling_layer, n_train)
        if at.num_kv_heads_test is not None:
            nt = at.num_kv_heads_test
            out_test = _attend(q, k[:, :, :nt], v[:, :, :nt], train, at.softmax_scaling_layer, n_train)
            out = torch.where(train.reshape(1, Ri, 1, 1), out, out_test)
        row = row + at.out_projection(out.reshape(B, Ri, at.head_dim * at.num_heads))
        row = row + blk.mlp(blk.layernorm_mlp(row))
    row = model.output_norm(row)                                                   # [B,Ri,D]

    # -- 6. decode every row against the train rows -----------------------------
    if clf:
        dec = model.many_class_decoder
        q = dec.q_projection(row).view(B, Ri, dec.num_heads, dec.head_dim)
        k = dec.k_projection(row).view(B, Ri, dec.num_heads, dec.head_dim)
        onehot = F.one_hot(y_use.long(), num_classes=dec.max_num_classes).to(q.dtype)
        onehot = onehot.reshape(1, Ri, 1, -1).expand(-1, -1, dec.num_heads, -1)
        probs = _class_attention(q, k, onehot, train, dec.softmax_scaling_layer, n_train)
        out = torch.log(torch.clamp(probs.mean(2).transpose(0, 1), min=1e-5) + 3e-5)
    else:
        out = model.output_projection(row.transpose(0, 1))
    if model._nan_safe_output:
        out = torch.nan_to_num(out, nan=0.0)
    return out


def _class_attention(q_BSHD, k_BJHD, v_BJHT, keep, scaling, n):
    """v3._chunked_class_attention with the keys masked."""
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
