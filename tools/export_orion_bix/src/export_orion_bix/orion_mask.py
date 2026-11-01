"""Orion-BiX with the train/test split as a VALUE instead of a shape (the ROCm / MIGraphX contract).

The CPU graph (``orion_bix_patches.ExportWrapper``) is ``(x, y)``-only: the split is ``len(y)``, a SHAPE, which a
fixed-shape MIGraphX compile cannot bucket (docs/ROCM_SINGLE_EVAL_POS.md). This module computes the same function
with every shape a function of the padded (T, H) bucket alone, the contract tabdpt, the TabPFN family, EXAONE
Tabular and Nori-30M already serve on ROCm:

    x[1, T, H] f32   y[1, T] f32 (FULL length; non-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)
    -> logits[1, T, C]   for every row (C = max_classes = 10)

The plugin pads rows past the real count and columns past ``d`` with zeros. Orion-BiX touches the split in exactly
two places, and the columns in one:

  1. the column embedder's induced self-attention: the inducing points attend to the CONTEXT rows only
     (upstream slices ``src[..., :train_size, :]``) -> a key mask over rows, ``row < train_size``;
  2. the ICL transformer's split attention (``attn_mask=train_size``): context rows attend to the context, and the
     other rows attend to the context too -- every query sees the same keys, so it is a key mask
     ``row < train_size``; and the label embedding is added to the context rows only;
  3. the row interaction attends over the 4 CLS slots + every column: padded columns are masked out as KEYS
     (``col < 4 + d``). RoPE positions are the column index, and padding sits at the end, so the real columns
     keep their positions.

Nothing else mixes rows or columns: the column embedder runs every column independently, and a padded row is a
query everywhere, never a key. So there is no ``n_rows`` input (nothing would read it).

MIGraphX-specific choices, each learnt by the earlier conversions (docs/ROCM_TABDPT_SPIKE.md,
docs/ROCM_TABPFN_PLAN.md, tools/export_exaone): attention is written out (matmul, softmax, matmul) with an
additive -1e4 key bias, finite in every float format; RoPE is written without einsum/einops; the one-hot label
encoding is a comparison against a constant ``arange``; ``train_size`` / ``d`` stay rank 1.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

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


def row_representations(row, emb: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``RowInteraction._train_forward`` with padded columns masked as keys -> [1, T, num_cls * E]."""
    b, t, hc, e = emb.shape
    cls = row.cls_tokens.expand(b, t, row.num_cls, e)
    seq = torch.cat((cls, emb[:, :, row.num_cls:]), dim=2)
    col_bias = key_bias(torch.arange(hc, device=emb.device) < d + row.num_cls, seq)  # [C + H]
    for blk in row.tf_row.blocks:
        seq = block(blk, seq, bias=col_bias, rope=row.tf_row.rope)
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


def masked_forward(model, x, y, train_size, d):
    rows = x.shape[1]
    ctx = torch.arange(rows, device=x.device) < train_size                           # [T]
    # whatever sits off the context (the -100 sentinel, NaN) must never reach the label encoder
    y = torch.where(ctx.unsqueeze(0), torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(y))
    emb = column_embeddings(model.col_embedder, x, ctx)
    reps = row_representations(model.row_interactor, emb, d)
    return icl_logits(model.icl_predictor, reps, y, ctx)


class MaskExportWrapper(torch.nn.Module):
    """Orion-BiX under the train_size contract: x, y (full length), train_size, d (see the module doc)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y, train_size, d):
        return masked_forward(self.m, x, y, train_size, d)
