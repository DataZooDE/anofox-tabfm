"""Nori 1.5 (Model v3, two-phase encoder) with the train/test split as a VALUE: the ROCm / MIGraphX contract.

The analogue of ``nori_mask.py`` (Nori-30M), same signature:

    x[1, T, H] f32   y[1, T] f32 (FULL length; off-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)   n_rows[1] i64 (declared for the contract, unused)
    -> logits[1, T, 1]: a raw-space point estimate for every row

It re-expresses ``FeaturesTransformer._forward_two_phase`` (regression, the released config) over upstream's own
modules. ``features_per_group`` is 1 and the x tokenizer has one feature per cell, so the padded width is just H and
masking is per COLUMN (``col < d``). Every place that used a shape the plugin's padding inflates becomes a mask:

  1. NaN-imputation mean, normalisation mean / unbiased std: over the CONTEXT rows (row < S). The
     ``ValidFeatureEncoder`` rescale is exactly 1 at ``num_features = 1`` (sqrt(1 / max(valid, 1))) and is dropped.
  2. ``PromptQuantilePLEEncoder``: upstream sorts the valid context values of each column and gathers 2 x 65
     order statistics. ``ple_edges`` gets the same numbers with no Sort, no TopK and no Gather: a stable RANK of every
     valid context value (``#{smaller} + #{equal and earlier}``, a compare matrix reduced by a GEMM) and the k-th
     order statistic as ``sum_j [rank_j == k] * v_j``, a one-hot GEMM. Padded columns and non-context rows are not
     valid, so they have no rank and select nothing. The interpolation arithmetic is upstream's, float for float.
  3. Phase A, per layer: the column attention (every row reads the CONTEXT rows' cells of its column), the target
     stream's row attention (train-only keys), and the decoupled feature attention: cells read [cells, targets],
     targets read cells only. Padded columns are masked out as keys in both feature attentions.
  4. Phase B, per layer: the row attention (train-only keys) and the read-back from the row's own cells (padded
     columns masked).
  5. QASSMax (``log_only``) multiplies q by ``log(key_len)``, and ``key_len`` is the key axis of THAT attention:
     S for column / row attention (a value), but ``d + K`` for the cells' feature attention and ``d`` for the
     read-outs and the read-back (upstream returns q unchanged for ``key_len <= 1``). All read the values.

The context is presented a second time as queries, as in the CPU wrapper (``cat([x[:S], x])``, eval_pos = S):
the first T rows play the context role (only rows < S are keys, and only they carry the label), the second T rows
are every row as a query with no label. The preprocessing runs once and is duplicated, which is exact.

MIGraphX-safe rewrites (the 30M ones, plus): the PLE above; the Linear(1, n) layers on the label as a broadcast
multiply (no K = 1 GEMM); LayerNorm and RMSNorm written out; attention projections as GEMMs; no integer
floor-division / modulo / split in the tokenizer.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from export_nori.nori_mask import NEG, _f, _layer_norm, _masked_mean, _sequential, _sum_rows, rbf_tokenizer


# ---------------------------------------------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------------------------------------------
def _rms(norm, x):
    """``RMSNorm.forward`` written out (the non-native path): x * rsqrt(mean(x^2) + eps) [* weight]."""
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    return y * norm.weight if norm.weight is not None else y


def _lin1(lin, v):
    """``nn.Linear(1, n)`` on ``v[..., 1]`` as a broadcast multiply-add (no K = 1 GEMM)."""
    return v * lin.weight[:, 0] + lin.bias


def _swiglu(mlp, x):
    return mlp.w_down(F.silu(mlp.w_gate(x)) * mlp.w_up(x))


def _log_len(n):
    """QASSMaxScaling 'log_only' factor for a key length ``n`` ([1] float): log(max(n, 2)), and 1 for n <= 1."""
    return torch.where(n > 1.5, torch.log(n.clamp_min(2.0)), torch.ones_like(n))


# ---------------------------------------------------------------------------------------------------------------
# preprocessing: statistics + the prompt-quantile PLE reference
# ---------------------------------------------------------------------------------------------------------------
def ple_edges(ple, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """``PromptQuantilePLEEncoder._build_reference`` for one table: x [T, G] (normalised), valid [T, G] bool
    (finite, original-finite AND context) -> edges [G, n_bins + 1].

    No Sort / TopK / Gather: stable ranks by a compare matrix and one-hot GEMMs (see the module doc)."""
    t, g = x.shape
    v = torch.where(valid, x, torch.zeros_like(x)).transpose(0, 1)                         # [G, T]
    ok = _f(valid, x).transpose(0, 1)                                                      # [G, T]
    vi, vj = v.unsqueeze(1), v.unsqueeze(2)                                                # [G,1,T_i], [G,T_j,1]
    row = torch.arange(t, device=x.device)
    earlier = _f(row.view(1, -1) < row.view(-1, 1), x)                                     # [T_j, T_i]: i < j
    before = _f(vi < vj, x) + _f(vi == vj, x) * earlier                                    # [G, T_j, T_i]
    rank = torch.matmul(before, ok.unsqueeze(-1)).squeeze(-1)                              # [G, T_j] (exact ints)
    rank = torch.where(ok > 0.5, rank, torch.full_like(rank, -1.0))                        # invalid: selects nothing
    count = _sum_rows(ok.transpose(0, 1)).squeeze(0)                                       # [G] valid count
    levels = ple.quantile_levels.to(device=x.device, dtype=x.dtype)                        # [n_bins + 1]
    positions = (count - 1.0).clamp_min(0.0).unsqueeze(-1) * levels                        # [G, n_bins + 1]
    lower_idx, upper_idx = torch.floor(positions), torch.ceil(positions)
    want = torch.cat((lower_idx, upper_idx), dim=-1)                                       # [G, 2 * (n_bins + 1)]
    onehot = _f(rank.unsqueeze(1) == want.unsqueeze(-1), x)                                # [G, 2(n+1), T]
    picked = torch.matmul(onehot, v.unsqueeze(-1)).squeeze(-1)                             # [G, 2(n+1)]
    lower, upper = picked[:, : levels.numel()], picked[:, levels.numel():]
    edges = lower + (positions - lower_idx) * (upper - lower)
    return torch.where(count.unsqueeze(-1) > 0, edges, torch.zeros_like(edges))


def preprocess(model, xg: torch.Tensor, ctx: torch.Tensor):
    """x_preprocess + process_4_x for one table: xg [T, G] (padded) -> (pre [T, G] NaN where the input was
    non-finite, edges [G, n_bins + 1]).

    NanEncoder -> NormalizationEncoder (train only) -> PromptQuantilePLEEncoder, each over the context rows
    (ctx [T, 1]). ValidFeatureEncoder is a no-op at num_features = 1."""
    nonfinite = ~torch.isfinite(xg)
    mean_fill, _ = _masked_mean(xg, ctx)
    x = torch.where(nonfinite, mean_fill.expand_as(xg), xg)
    mean, num = _masked_mean(x, ctx)
    keep = torch.isfinite(x) & ctx
    sq = torch.where(keep, (mean - x) ** 2, torch.zeros_like(x))
    std = torch.sqrt(_sum_rows(sq) / (num - 1).clamp_min(1)) + 1e-20
    x = torch.clip((x - mean) / std, min=-100, max=100)
    ple = next(m for m in model.x_preprocess if type(m).__name__ == "PromptQuantilePLEEncoder")
    edges = ple_edges(ple, x, torch.isfinite(x) & ~nonfinite & ctx)
    return torch.where(nonfinite, torch.full_like(x, float("nan")), x), edges


def encode_x(enc, pre: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """``MaskEmbEncoder.forward`` (RBF tokenizer + PLE channels): pre [R, G] (NaN = missing), edges [G, n+1]
    -> [R, G, E]. ``PromptQuantilePLEEncoder.encode`` is the clamp of (x - left) / (right - left)."""
    is_mask = torch.isnan(pre)
    x = pre.masked_fill(is_mask, 0.0)
    rbf = rbf_tokenizer(enc.numeric_mlp, x.unsqueeze(-1).unsqueeze(-1)).squeeze(-2)         # [R, G, E - bins]
    left, right = edges[:, :-1], edges[:, 1:]
    width = right - left
    nonzero = width > 0
    safe = torch.where(nonzero, width, torch.ones_like(width))
    ple = ((x.unsqueeze(-1) - left) / safe).clamp(0.0, 1.0)
    ple = torch.where(nonzero, ple, torch.zeros_like(ple))                                  # [R, G, bins]
    emb = torch.cat((rbf, ple.to(rbf.dtype)), dim=-1)
    combined = torch.where(is_mask.unsqueeze(-1), enc.mask_embedding.expand_as(emb), emb)
    return _sequential(enc.fusion_network, combined)


# ---------------------------------------------------------------------------------------------------------------
# attention with a key mask and a QASS length that is a value
# ---------------------------------------------------------------------------------------------------------------
def attention(attn, query, key_value, key_mask, q_factor):
    """Model v3 ``MultiheadAttention`` (qkv_combined=False, qk RMS norm, QASSMax log_only) with a key mask.

    query [N, Lq, E], key_value [N, Lk, E], key_mask [Lk] (bool), q_factor [1] = the QASS factor for this key axis."""
    h_, d_, e_ = attn.num_heads, attn.head_dim, attn.embed_dim
    q = F.linear(query, attn.qkv_proj_weight[0].reshape(h_ * d_, e_)).unflatten(-1, (h_, d_))          # [N,Lq,h,d]
    kv = F.linear(key_value, attn.qkv_proj_weight[1:].reshape(2 * h_ * d_, e_)).unflatten(-1, (2, h_, d_))
    k, v = _rms(attn.k_norm, kv[:, :, 0]), kv[:, :, 1]
    q = _rms(attn.q_norm, q) * q_factor.view(1, 1, 1, 1)
    scores = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-1, -2)) * (1.0 / math.sqrt(d_))
    scores = scores + (1.0 - _f(key_mask, scores)) * NEG
    out = torch.matmul(torch.softmax(scores, dim=-1), v.transpose(1, 2)).transpose(1, 2)               # [N,Lq,h,d]
    return F.linear(out.flatten(-2), attn.out_proj_weight.reshape(h_ * d_, e_).transpose(0, 1))


def col_keys(cols, ctx_mask):
    """Column attention keys: the CONTEXT copy's rows (the first T of the doubled 2T), rows < S."""
    return cols[:, : ctx_mask.shape[0]], ctx_mask


def row_keys(rows_norm, ctx_mask):
    """Row attention keys (target stream and phase B): the same context rows."""
    return rows_norm[:, : ctx_mask.shape[0]], ctx_mask


def cell_layer(layer, cells, targets, *, ctx_mask, col_mask, s_factor, f_cells, f_read):
    """``CellLayer2P`` on the doubled state: cells [2T, G, E], targets [2T, K, E] (batch of 1 elided)."""
    alpha = layer.deepnorm_alpha
    # S: column attention for the cells ...
    cols = _rms(layer.cell_norms[0], cells).transpose(0, 1)                                 # [G, 2T, E]
    keys, kmask = col_keys(cols, ctx_mask)
    cells = cells + alpha * attention(layer.column_attention.sequence_attentions[0], cols, keys, kmask,
                                      s_factor).transpose(0, 1)
    # ... and train-only row attention for the target stream
    rows = targets.flatten(-2)                                                              # [2T, W]
    nrm = _rms(layer.target_norms[0], rows).unsqueeze(0)                                    # [1, 2T, W]
    keys, kmask = row_keys(nrm, ctx_mask)
    rows = rows + alpha * attention(layer.sequence_attentions[0], nrm, keys, kmask, s_factor).squeeze(0)
    # M
    cells = cells + alpha * _swiglu(layer.mlp[0], _rms(layer.cell_norms[1], cells))
    rows = rows + alpha * _swiglu(layer.mlp[1], _rms(layer.target_norms[1], rows))
    # F: decoupled feature attention inside each row
    targets = rows.view(rows.shape[0], layer.target_tokens, layer.cell_dim)
    cn, tn = _rms(layer.cell_norms[2], cells), _rms(layer.target_norms[2], targets)
    fa = layer.feature_attention
    both_mask = torch.cat((col_mask, torch.ones(layer.target_tokens, dtype=torch.bool, device=cells.device)))
    cell_update = attention(fa.feature_attentions[0], cn, torch.cat((cn, tn), dim=1), both_mask, f_cells)
    readout = attention(fa.feature_attentions[1], tn, cn, col_mask, f_read)                 # [2T, K, E]
    target_update = fa.target_mix(readout.flatten(-2)).view_as(targets)
    return cells + alpha * cell_update, targets + alpha * target_update


def row_layer(layer, rows, cell_memory, *, ctx_mask, col_mask, s_factor, f_read):
    """``RowLayer2P`` on the doubled rows [2T, W] with the row's own cells cell_memory [2T, G, W]."""
    alpha = layer.deepnorm_alpha
    nrm = _rms(layer.attn_norm, rows).unsqueeze(0)
    keys, kmask = row_keys(nrm, ctx_mask)
    rows = rows + alpha * attention(layer.sequence_attentions[0], nrm, keys, kmask, s_factor).squeeze(0)
    query = _rms(layer.readback_norm, rows).unsqueeze(1)                                    # [2T, 1, W]
    rows = rows + alpha * attention(layer.feature_attentions[0], query, cell_memory, col_mask, f_read).squeeze(1)
    return rows + alpha * _swiglu(layer.mlp[0], _rms(layer.mlp_norm, rows))


def masked_forward(model, x, y_norm, train_size, d):
    """x [1, T, H], y_norm [1, T] (standardised; any value off-context) -> quantiles for EVERY row [1, T, K]."""
    if model.features_per_group != 1 or model.two_phase_config is None:
        raise ValueError("the Nori 1.5 masked contract covers the released two-phase config (features_per_group = 1)")
    rows, width = x.shape[1], x.shape[2]
    enc = model.transformer_encoder
    ctx = torch.arange(rows, device=x.device) < train_size                                  # [T]
    pre, edges = preprocess(model, x[0], ctx.view(-1, 1))                                   # [T, G], [G, n+1]
    x_emb = encode_x(model.encoder_x[0], torch.cat((pre, pre), dim=0), edges)               # [2T, G, E]

    nan = torch.full_like(y_norm[0], float("nan"))
    y_all = torch.cat((torch.where(ctx, y_norm[0], nan), nan), dim=0)                       # [2T]
    doubled_ctx = torch.cat((ctx, torch.zeros_like(ctx)), dim=0)

    # target-aware embedding: context rows get a bias from their label, every other row zero
    lbl = torch.nan_to_num(y_all, nan=0.0).view(-1, 1)
    bias = torch.where(doubled_ctx.view(-1, 1), _lin1(model.reg_target_aware_embedding, lbl),
                       torch.zeros_like(x_emb[:, 0]))                                       # [2T, E]
    scale = float(getattr(model, "target_aware_scale", 1.0))
    if scale != 1.0:
        bias = bias * scale
    rb = bias.unsqueeze(1)
    if model.use_column_specific_y_aware and model.column_y_aware_alpha is not None:
        col_score = (x_emb * rb).sum(dim=-1) * (1.0 / (model.embed_dim ** 0.5))
        gate = torch.sigmoid(col_score).unsqueeze(-1)
        alpha = torch.sigmoid(model.column_y_aware_alpha).to(x_emb.dtype)
        x_emb = x_emb + (1.0 - alpha) * rb + alpha * (rb * gate)
    else:
        x_emb = x_emb + rb

    # target stream: K tokens per row; the label MLP's first layer is Linear(1, n) as a broadcast multiply
    te = enc.target_encoder
    valid_y = torch.isfinite(y_all).view(-1, 1)
    labels = torch.where(torch.isfinite(y_all), y_all, torch.zeros_like(y_all)).view(-1, 1)
    emb_y = F.linear(F.gelu(_lin1(te.mlp[0], labels)), te.mlp[2].weight, te.mlp[2].bias)    # [2T, W]
    emb_y = torch.where(valid_y, emb_y, te.query_embedding.to(emb_y.dtype).view(1, -1))
    targets = emb_y.view(-1, enc.target_tokens, enc.cell_dim)

    col_mask = torch.arange(width, device=x.device) < d                                     # [G]
    s_val = train_size.to(torch.float32)
    d_val = d.to(torch.float32)
    s_factor = _log_len(s_val)
    f_cells = _log_len(d_val + enc.target_tokens)
    f_read = _log_len(d_val)
    cells = x_emb
    for layer in enc.layers[: enc.num_cell_layers]:
        cells, targets = cell_layer(layer, cells, targets, ctx_mask=ctx, col_mask=col_mask, s_factor=s_factor,
                                    f_cells=f_cells, f_read=f_read)
    # boundary: re-inject the context labels
    row_state = targets.flatten(-2)
    ye = enc.row_y_encoder
    bias_y = _layer_norm(ye[1], _lin1(ye[0], lbl))                                          # [2T, W]
    row_state = row_state + torch.where(doubled_ctx.view(-1, 1), bias_y, torch.zeros_like(bias_y))
    cell_memory = enc.cell_to_row(_rms(enc.cell_out_norm, cells))                           # [2T, G, W]
    for layer in enc.layers[enc.num_cell_layers:]:
        row_state = row_layer(layer, row_state, cell_memory, ctx_mask=ctx, col_mask=col_mask, s_factor=s_factor,
                              f_read=f_read)
    out = _rms(enc.row_out_norm, row_state)[rows:]                                          # [T, W]
    return _sequential(model.reg_y_decoder, out).unsqueeze(0)                               # [1, T, K]


class MaskExportWrapper15(torch.nn.Module):
    """Nori 1.5 under the train_size contract: x, y_full, train_size, d, n_rows (see the module doc)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y, train_size, d, n_rows):
        from export_nori.nori_patches import quantile_mean

        rows = x.shape[1]
        ctx = (torch.arange(rows, device=x.device) < train_size).unsqueeze(0)               # [1, T]
        zero = torch.zeros_like(y)
        y = torch.where(ctx, torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), zero)
        # masking first, transposing after (the 30M wrapper's MIGraphX-safe form)
        n = _sum_rows(_f(ctx, y).transpose(0, 1))[0]                                       # [1] == S
        mean_y = _sum_rows(torch.where(ctx, y, zero).transpose(0, 1))[0] / n
        dev = torch.where(ctx, y - mean_y, zero)
        std_y = torch.sqrt(_sum_rows((dev * dev).transpose(0, 1))[0] / n)
        std_y = torch.where(std_y < 1e-12, torch.ones_like(std_y), std_y)
        q = masked_forward(self.m, x, (y - mean_y) / std_y, train_size, d)                 # [1, T, K]
        return quantile_mean(q).unsqueeze(-1) * std_y + mean_y
