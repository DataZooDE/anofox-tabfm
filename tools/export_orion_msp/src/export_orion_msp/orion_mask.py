"""Orion-MSP with the train/test split as a VALUE instead of a shape (the ROCm / MIGraphX contract).

The CPU graph (``orion_msp_patches.ExportWrapper``) is ``(x, y)``-only: the split is ``len(y)``, a SHAPE, which a
fixed-shape MIGraphX compile cannot bucket (docs/ROCM_SINGLE_EVAL_POS.md). This module computes the same function
with every shape a function of the padded (T, H) bucket alone -- the contract Orion-BiX, tabdpt, the TabPFN family,
EXAONE Tabular and Nori-30M serve on ROCm:

    x[1, T, H] f32   y[1, T] f32 (FULL length; non-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)
    -> logits[1, T, C]   for every row (C = max_classes = 10)

The column embedder and the ICL transformer are Orion-BiX's (TabICL's), masked the same way: the inducing points and
every ICL query attend to the CONTEXT rows only (key masks ``row < train_size``), and the label embedding is added to
the context rows only. The row stage is MSP's own, and the released checkpoint runs it at a single scale with
``features_per_group = 1`` in effect (``row_scales = (1,)``), so its sequence is the 8 special tokens followed by one
PMA-pooled token per column:

  1. the PMA pooling attends over the columns: padded columns are masked out as KEYS (``col < d``, upstream's own
     ``valid_counts`` mask). Its positional queries are indexed by column, and padding sits at the end, so the real
     columns keep theirs;
  2. the block-sparse row encoder: padded column tokens are masked out as keys (``col < 8 + d``, upstream's own
     ``key_padding_mask``), on top of the frozen BigBird mask (specials both ways, the diagonal, and
     ``row_num_random`` seeded random links per query);
  3. the random links are the top-``num_random`` columns of the frozen score table restricted to the REAL columns
     (``8 <= col < 8 + d``). The CPU graph takes the top-k of ``table[:L, :L]`` with ``L`` the real length; taking
     it over the padded length unrestricted would hand a real query a different link whenever a padded column
     scores higher -- not inert. Restricted, the picks are the CPU graph's picks exactly.

Nothing else mixes rows or columns, so there is no ``n_rows`` input.

MIGraphX-specific choices, as for Orion-BiX (tools/export_orion_bix/src/export_orion_bix/orion_mask.py): attention
is written out with an additive -1e4 bias, RoPE without einsum/einops, the one-hot label encoding as a comparison,
and the random-link scatter as comparisons against the picks.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from export_orion_msp import orion_msp_patches

NEG = -1e4
SKIP = -100.0


def key_bias(valid: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """[..., Lk] bool (True = may be attended) -> additive bias in like's dtype."""
    return torch.where(valid, torch.zeros((), dtype=like.dtype), torch.full((), NEG, dtype=like.dtype))


def _rope(rope, t: torch.Tensor) -> torch.Tensor:
    """``RotaryEmbedding.rotate_queries_or_keys`` (lang freqs, no xpos, seq_dim -2) without einsum/einops."""
    seq_len, dim = t.shape[-2], t.shape[-1]
    pos = torch.arange(seq_len, device=t.device).to(rope.freqs.dtype) / rope.interpolate_factor
    freqs = pos.unsqueeze(-1) * rope.freqs.unsqueeze(0)                              # [L, dim/2]
    freqs = torch.stack((freqs, freqs), dim=-1).reshape(seq_len, dim)                # (n r) with r = 2
    pairs = t.reshape(*t.shape[:-1], dim // 2, 2)
    rotated = torch.cat((-pairs[..., 1:2], pairs[..., 0:1]), dim=-1).reshape(t.shape)  # rotate_half
    return (t * freqs.cos() + rotated * freqs.sin()).to(t.dtype)


def attention(attn, query: torch.Tensor, kv: torch.Tensor, bias=None, rope=None) -> torch.Tensor:
    """``multi_head_attention_forward`` (tensor-mask branch) with an additive bias over [..., Lq, Lk]."""
    e = query.shape[-1]
    h = attn.num_heads
    hd = e // h
    w, b = attn.in_proj_weight, attn.in_proj_bias
    q = F.linear(query, w[:e], b[:e])
    k = F.linear(kv, w[e:2 * e], b[e:2 * e])
    v = F.linear(kv, w[2 * e:], b[2 * e:])
    q = q.reshape(*q.shape[:-1], h, hd).transpose(-3, -2)
    k = k.reshape(*k.shape[:-1], h, hd).transpose(-3, -2)
    v = v.reshape(*v.shape[:-1], h, hd).transpose(-3, -2)
    if rope is not None:
        q = _rope(rope, q)
        k = _rope(rope, k)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(hd))
    if bias is not None:
        scores = scores + bias
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    out = out.transpose(-3, -2).reshape(*query.shape[:-1], e)
    return F.linear(out, attn.out_proj.weight, attn.out_proj.bias)


def block(blk, q: torch.Tensor, kv=None, bias=None, rope=None) -> torch.Tensor:
    """``MultiheadAttentionBlock.forward`` (norm_first) with an additive key bias."""
    kv = q if kv is None else kv
    if not blk.norm_first:
        raise ValueError("only the pre-norm block is on the released path")
    x = q + attention(blk.attn, blk.norm1(q), blk.norm1(kv), bias, rope)
    return x + blk.linear2(blk.activation(blk.linear1(blk.norm2(x))))


def isab(blk, src: torch.Tensor, row_bias: torch.Tensor) -> torch.Tensor:
    """``InducedSelfAttentionBlock`` with the context slice as a key bias over rows; skipped columns stay -100."""
    *batch, _, e = src.shape
    inds = blk.ind_vectors.expand(*batch, blk.num_inds, e)
    hidden = block(blk.multihead_attn1, inds, src, row_bias)
    out = block(blk.multihead_attn2, src, hidden)
    skip = (src == blk.skip_value).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    return torch.where(skip, torch.full_like(out, blk.skip_value), out)


def column_embeddings(col, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """``ColEmbedding._train_forward(d=None, train_size=S)`` -> [1, T, C + H, E]; ctx [T] bool."""
    x = F.pad(x, (col.reserve_cls_tokens, 0), value=SKIP)
    features = x.transpose(1, 2).unsqueeze(-1)                                       # [1, C+H, T, 1]
    src = col.in_linear(features)
    row_bias = key_bias(ctx, src)                                                    # [T]
    for blk in col.tf_col.blocks:
        src = isab(blk, src, row_bias)
    weights = col.ln_w(col.out_w(src))
    biases = col.ln_b(col.out_b(src))
    return (features * weights + biases).transpose(1, 2)



def pma_queries(pma, k: int, device) -> torch.Tensor:
    """``PMAGroup._make_queries`` for one row: seed + sinusoidal position -> [K, E] (no strided writes)."""
    e = pma.embed_dim
    pos = torch.arange(k, device=device).to(torch.float32).unsqueeze(-1)              # [K, 1]
    omega = torch.exp(torch.arange(0, e, 2, device=device).to(torch.float32) * (-math.log(10000.0) / e))
    angle = pos * omega                                                              # [K, E/2]
    pe = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1).reshape(k, e)
    return pma.seed.reshape(1, e) + pe


def pma_group(pma, feats: torch.Tensor, feat_bias: torch.Tensor) -> torch.Tensor:
    """``PMAGroup.forward(feats, valid_counts=d, K=H)`` -> [1, T, H, E]; feat_bias [H] masks padded columns."""
    h, e = feats.shape[2], feats.shape[3]
    q = pma.q_proj(pma_queries(pma, h, feats.device))                                 # [K, E]
    k = pma.k_proj(feats)                                                            # [1, T, H, E]
    v = pma.v_proj(feats)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(e) + feat_bias         # [1, T, K, H]
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def sparse_allow(seq_len: int, num_special: int, num_random: int, n_real, table: torch.Tensor) -> torch.Tensor:
    """The frozen block-sparse mask (orion_msp_patches.frozen_block_sparse_mask) over the padded length, with the
    random links drawn from the REAL columns only -> [L, L] bool, True = may attend. n_real [1] = 8 + d; table is
    the frozen [MAX_L, MAX_L] score table (orion_msp_patches._random_link_scores)."""
    device = table.device
    idx = torch.arange(seq_len, device=device)
    special = idx < num_special
    allow = special.unsqueeze(0) | special.unsqueeze(1) | (idx.unsqueeze(0) == idx.unsqueeze(1))
    if num_random > 0:
        table = table[:seq_len, :seq_len]
        eligible = (idx >= num_special) & (idx < n_real)                              # [L]
        scores = torch.where(eligible.unsqueeze(0), table, torch.full_like(table, -float("inf")))
        k = min(int(num_random), max(seq_len - num_special, 0))
        if k > 0:
            picks = scores.topk(k, dim=-1).indices                                    # [L, k]
            for j in range(k):
                allow = allow | (picks[:, j:j + 1] == idx.unsqueeze(0))
    return allow


def row_representations(row, emb: torch.Tensor, d: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``RowInteraction._train_forward`` (one scale, group size 1, PMA) with padded columns masked -> [1, T, C*E]."""
    if row.scales != (1,) or row.group_mode != "pma":
        raise ValueError(f"only the released single-scale PMA row stage is traced (scales={row.scales}, "
                         f"group_mode={row.group_mode})")
    b, t, hc, e = emb.shape
    ns = row.num_special
    specials = torch.cat((row.cls_tokens.expand(b, t, row.num_cls, e),
                          row.global_tokens.expand(b, t, row.num_global, e)), dim=2)
    feats = emb[:, :, ns:]                                                           # [1, T, H, E]
    h = feats.shape[2]
    grouped = pma_group(row.pma, feats, key_bias(torch.arange(h, device=emb.device) < d, feats))
    seq = torch.cat((specials, grouped), dim=2)                                      # [1, T, ns + H, E]
    length = ns + h
    n_real = d + ns
    allow = sparse_allow(length, ns, row.num_random, n_real, table)
    allow = allow & (torch.arange(length, device=emb.device) < n_real).unsqueeze(0)  # padded columns never keys
    bias = key_bias(allow, seq)                                                      # [L, L]
    encoder = row.scale_encoders[0]
    for blk in encoder.blocks:
        seq = block(blk, seq, bias=bias, rope=encoder.rope)
    return row.out_ln(seq[:, :, : row.num_cls]).flatten(-2)

def icl_logits(icl, reps: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """``ICLearning._icl_predictions`` with the split as a key mask -> [1, T, max_classes]."""
    classes = torch.arange(icl.max_classes, device=y.device).to(y.dtype)
    one_hot = (y.unsqueeze(-1) == classes).to(reps.dtype)                            # [1, T, C]
    label = F.linear(one_hot, icl.y_encoder.weight, icl.y_encoder.bias)
    reps = reps + torch.where(ctx.view(1, -1, 1), label, torch.zeros_like(label))
    row_bias = key_bias(ctx, reps)                                                   # [T]
    for blk in icl.tf_icl.blocks:
        reps = block(blk, reps, bias=row_bias)
    if icl.norm_first:
        reps = icl.ln(reps)
    return icl.decoder(reps)



def masked_forward(model, x, y, train_size, d, table=None):
    rows = x.shape[1]
    ctx = torch.arange(rows, device=x.device) < train_size                           # [T]
    # whatever sits off the context (the -100 sentinel, NaN) must never reach the label encoder
    y = torch.where(ctx.unsqueeze(0), torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(y))
    emb = column_embeddings(model.col_embedder, x, ctx)
    table = orion_msp_patches._random_link_scores() if table is None else table
    reps = row_representations(model.row_interactor, emb, d, table)
    if model.icl_predictor.memory.has_memory():
        raise ValueError("the Perceiver memory is off in the released checkpoint and is not traced")
    return icl_logits(model.icl_predictor, reps, y, ctx)


class MaskExportWrapper(torch.nn.Module):
    """Orion-MSP under the train_size contract: x, y (full length), train_size, d (see the module doc)."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        # a buffer, so the graph names it "..._LINK_SCORES" (export.INLINE_CONSTANT_SUFFIXES keeps it inline, out of
        # the tensor map) rather than an anonymous lifted constant
        self.register_buffer("_LINK_SCORES", orion_msp_patches._random_link_scores(), persistent=False)

    def forward(self, x, y, train_size, d):
        return masked_forward(self.m, x, y, train_size, d, self._LINK_SCORES)
