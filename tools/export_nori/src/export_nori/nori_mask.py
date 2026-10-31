"""Nori with the train/test split as a VALUE instead of a shape (the ROCm / MIGraphX contract).

The CPU graph (``nori_patches.ExportWrapper``) is ``(x, y)``-only: the split is ``len(y)``, a SHAPE, which a
fixed-shape MIGraphX compile cannot bucket (docs/ROCM_SINGLE_EVAL_POS.md). This module computes the same
function with every shape a function of the padded (T, H) bucket alone:

    x[1, T, H] f32   y[1, T] f32 (FULL length; off-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)   n_rows[1] i64 (declared for the contract, unused)
    -> logits[1, T, 1]: a raw-space point estimate for every row

It re-expresses ``FeaturesTransformer.forward`` (regression, the released config) over upstream's own
modules, and every place that used a shape the plugin's padding now inflates becomes a mask:

  1. the NaN-imputation mean, the normalisation mean / unbiased std and the valid-feature test, over the
     CONTEXT rows (row < S); upstream's `x[:, :eval_pos]`;
  2. the label NaN-imputation mean and the target-aware embedding, over the context rows;
  3. the feature attention per row: padded feature GROUPS are masked out as keys. A group is real when
     2 * g < d (division-free: MIGraphX miscompiles a runtime integer floor-division, docs/ROCM_TABPFN_PLAN.md).
     A padded column inside the last real group (odd d) is exactly upstream's own pad column;
  4. the sample attention: the context self-attention and the queries' cross-attention mask out
     non-context rows as keys, and the QASSMax length is S, the value, not the padded length;
  5. the regression target statistics, over the context rows.

The context is presented a second time as queries, as in the CPU wrapper, as two static T-row copies: the
first copy plays the context role (only its rows < S are keys), the second is every row as a query. The
preprocessing runs once and is duplicated, which is exact (its statistics read the context only).

n_rows is declared because the plugin binds it when a graph has it; nothing here reads every row (upstream's
statistics are all over the context), so padded rows can never reach a statistic and it stays unused.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

NEG = -1e4


def _sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum over the first axis as a GEMM (ones[1, R] @ x[R, rest]) -- chained masked reductions crash
    MIGraphX's reducer codegen (docs/ROCM_TABDPT_SPIKE.md)."""
    rows = x.shape[0]
    flat = x.reshape(rows, -1)
    ones = torch.ones(1, rows, dtype=flat.dtype, device=flat.device)
    return (ones @ flat).reshape(1, *x.shape[1:])


def _f(mask, like):
    return mask.to(like.dtype)


def _masked_mean(x, mask):
    """Mean over axis 0 of the finite entries where mask; upstream calc_mean (count clipped to 1)."""
    keep = torch.isfinite(x) & mask
    num = _sum_rows(_f(keep, x)).clamp_min(1)
    return _sum_rows(torch.where(keep, x, torch.zeros_like(x))) / num, num


def _layer_norm(ln, x):
    """``nn.LayerNorm`` written out (mean, biased variance, affine). The ONNX LayerNormalization op lowers in
    MIGraphX to a ``broadcast[axis=...]`` of the scale and bias, and its GPU pipeline (simplify_reshapes) then
    fails with "BROADCAST : axis 2 is out of range" on the Nori graph; plain elementwise ops broadcast with
    multibroadcast instead. Same function to float rounding."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    y = (x - mean) * torch.rsqrt(var + ln.eps)
    if ln.weight is not None:
        y = y * ln.weight
    if ln.bias is not None:
        y = y + ln.bias
    return y


def _sequential(seq, x):
    """Run an nn.Sequential with every nn.LayerNorm replaced by ``_layer_norm``."""
    for mod in seq:
        x = _layer_norm(mod, x) if isinstance(mod, torch.nn.LayerNorm) else mod(x)
    return x


def rbf_tokenizer(rbf, x: torch.Tensor) -> torch.Tensor:
    """``RBFembedding.forward`` (exponent_digits = 1, as_tokenizer) without the two ops MIGraphX cannot take.

    * the exponent digit is ``(abs_exp // 10**0) % 10``: an integer floor-division and modulo on runtime data,
      which MIGraphX's GPU target miscomputes (docs/ROCM_TABPFN_PLAN.md). With one exponent digit and the
      exponent clamped to [-9, 9] it is ``abs_exp`` itself, taken here in float and cast once for the lookup;
    * ``torch.split`` of the gate output exports as SplitToSequence / SequenceAt, which MIGraphX's parser
      does not implement; slicing is the same halves.
    The float64 working precision is upstream's and is kept: it decides the decimal exponent."""
    if rbf.exponent_digits != 1 or rbf.use_original_features:
        raise ValueError("the masked tokenizer covers the released config only (1 exponent digit, no raw feature)")
    x = x.squeeze(-1)
    x_work = x.to(torch.float64)
    abs_x = torch.abs(x_work)
    is_zero = abs_x == 0
    safe = torch.where(is_zero, torch.ones_like(abs_x), abs_x)
    exp_f = torch.clamp(torch.floor(torch.log10(safe)), -9.0, 9.0)
    x_scaled = abs_x / torch.pow(torch.full_like(exp_f, 10.0), exp_f)
    x_scaled = torch.where(is_zero, torch.zeros_like(x_scaled), x_scaled)
    centers = rbf.centers.to(torch.float64)
    diff = x_scaled.unsqueeze(-1) - centers
    out = torch.exp(-(diff ** 2) / (2 * (rbf.sigma.to(torch.float64) ** 2))).to(rbf.dtype)
    sign_emb = rbf.sign_embedding((x_work < 0).to(torch.long))
    exp_sign_emb = rbf.exp_sign_embedding((exp_f < 0).to(torch.long))
    digit_emb = rbf.exp_digit_embedding(torch.abs(exp_f).to(torch.long))
    gamma_beta = rbf.gate_mlp(torch.cat([sign_emb, exp_sign_emb, digit_emb], dim=-1).to(rbf.dtype))
    k = rbf.n_kernels
    out = out * torch.sigmoid(gamma_beta[..., :k]) + torch.tanh(gamma_beta[..., k:])
    return rbf.out_layer(_layer_norm(rbf.norm, out).to(rbf.dtype))


def encode_x(enc, data: torch.Tensor) -> torch.Tensor:
    """``MaskEmbEncoder.forward`` (RBF tokenizer, no prompt-quantile channels): [B, R, G, 2] -> [B, R, G, E]."""
    b, r, g, fpg = data.shape
    x = data.unsqueeze(-1)
    is_mask = torch.isnan(x)
    x = x.masked_fill(is_mask, 0.0)
    x_emb = rbf_tokenizer(enc.numeric_mlp, x)
    combined = torch.where(is_mask, enc.mask_embedding.expand_as(x_emb), x_emb)
    return _sequential(enc.fusion_network, combined.flatten(3)).view(b, r, g, -1)


def preprocess(model, xg: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """x_preprocess + process_4_x for one table: xg [T, G, 2] (grouped, padded) -> [T, G, 2].

    NanEncoder -> NormalizationEncoder (train only) -> ValidFeatureEncoder -> NaN where the input was
    non-finite, each over the context rows (ctx [T, 1, 1] broadcast)."""
    nonfinite = ~torch.isfinite(xg)
    # NanEncoder: fill non-finite cells with the finite context mean
    mean_fill, _ = _masked_mean(xg, ctx)
    x = torch.where(nonfinite, mean_fill.expand_as(xg), xg)
    # NormalizationEncoder: (x - mean) / (unbiased std + 1e-20), clip to +-100 (S >= 2 by contract)
    mean, num = _masked_mean(x, ctx)
    keep = torch.isfinite(x) & ctx
    sq = torch.where(keep, (mean - x) ** 2, torch.zeros_like(x))
    std = torch.sqrt(_sum_rows(sq) / (num - 1).clamp_min(1)) + 1e-20
    x = torch.clip((x - mean) / std, min=-100, max=100)
    # ValidFeatureEncoder: a feature is valid when the context has > 1 finite value and min != max
    context = torch.where(nonfinite | ~ctx, torch.full_like(x, float("nan")), x)
    finite = torch.isfinite(context)
    finite_count = _sum_rows(_f(finite, x))
    finite_min = torch.where(finite, context, torch.full_like(context, float("inf"))).amin(dim=0, keepdim=True)
    finite_max = torch.where(finite, context, torch.full_like(context, float("-inf"))).amax(dim=0, keepdim=True)
    valid = (finite_count > 1) & (finite_min != finite_max)                      # [1, G, 2]
    valid_num = torch.clip(_f(valid, x).sum(-1, keepdim=True), min=1)            # [1, G, 1]
    x = x * torch.sqrt(2.0 / valid_num)
    # process_4_x: the original non-finite cells become NaN (the MaskEmbEncoder's mask embedding)
    return torch.where(nonfinite, torch.full_like(x, float("nan")), x)


def _seq_attention(attn, query, key_value, key_mask, log_s):
    """Nori MultiheadAttention (qkv_combined=False, QASSMax) with a key mask.

    query [N, Lq, E], key_value [N, Lk, E], key_mask [Lk] bool, log_s [1] = log(S)."""
    # upstream's einsum projections as plain GEMMs (no committed MIGraphX graph uses Einsum; F.linear is safe)
    h_, d_, e_ = attn.num_heads, attn.head_dim, attn.embed_dim
    q = F.linear(query, attn.qkv_proj_weight[0].reshape(h_ * d_, e_)).unflatten(-1, (h_, d_))       # [N, Lq, h, d]
    kv = F.linear(key_value, attn.qkv_proj_weight[1:].reshape(2 * h_ * d_, e_)).unflatten(-1, (2, h_, d_))
    qm = attn.qassmax
    if qm is not None:
        log_n = log_s.to(q.dtype)
        if qm.qass_mode == "log_only":
            q = q * log_n.view(1, 1, 1, 1)
        else:
            # base_mlp (Linear(1, 64) -> GELU -> Linear(64, h*d)) on the scalar log(S), written out: the module
            # applied to log_n.view(1, 1) made MIGraphX's GPU pipeline fail (simplify_reshapes: "cannot create
            # std::vector larger than max_size()"), bisected on gfx1201 to that call. Same function.
            l1, act, l2 = qm.base_mlp[0], qm.base_mlp[1], qm.base_mlp[2]
            hidden = act(l1.weight.view(-1) * log_n + l1.bias)
            base_delta = (hidden.unsqueeze(0) @ l2.weight.transpose(0, 1) + l2.bias).view(1, 1, qm.num_heads,
                                                                                       qm.head_dim)
            base_scale = log_n.view(1, 1, 1, 1) * (1.0 + torch.tanh(base_delta))
            # the gate reads the UNSCALED query (upstream: q * base_scale * (1 + tanh(gate_mlp(q))))
            q = q * base_scale * (1.0 + torch.tanh(qm.gate_mlp(q))) if qm.qass_mode == "full" else q * base_scale
    k, v = kv[:, :, 0], kv[:, :, 1]
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)                   # [N, h, L, d]
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * (1.0 / math.sqrt(attn.head_dim))
    scores = scores + (1.0 - _f(key_mask, scores)) * NEG
    out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2)                  # [N, Lq, h, d]
    return F.linear(out.flatten(-2), attn.out_proj_weight.reshape(h_ * d_, e_).transpose(0, 1))


def _feat_attention(attn, x, key_mask):
    """Nori MultiheadAttention (qkv_combined=True, no QASSMax) over the tokens of each row, with a key mask.
    x [N, F, E], key_mask [F] bool."""
    h_, d_, e_ = attn.num_heads, attn.head_dim, attn.embed_dim
    qkv = F.linear(x, attn.qkv_proj_weight.reshape(3 * h_ * d_, e_)).unflatten(-1, (3, h_, d_))  # [N, F, 3, h, d]
    q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(attn.head_dim))
    scores = scores + (1.0 - _f(key_mask, scores)) * NEG
    out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2)
    return F.linear(out.flatten(-2), attn.out_proj_weight.reshape(h_ * d_, e_).transpose(0, 1))


def _residual(layer, residual, update):
    return residual + layer.deepnorm_alpha * update if layer.deepnorm_alpha is not None else residual + update


def masked_layer(layer, x, *, rows, ctx_key_mask, log_s, group_key_mask):
    """EncoderBaseLayer (smf, pre-norm, seq_attn_isolated, not serial) on the doubled [1, 2T, G+1, E] state."""
    b, r2, f, e = x.shape
    # 1. sample attention
    residual = x
    h = layer.layer_norms[0](x)
    ctx_part = h[:, :rows].transpose(1, 2).reshape(-1, rows, e)                             # [F, T, E]
    qry_part = h[:, rows:].transpose(1, 2).reshape(-1, r2 - rows, e)
    x_train = _seq_attention(layer.sequence_attentions[0], ctx_part, ctx_part, ctx_key_mask, log_s)
    x_test = _seq_attention(layer.sequence_attentions[1], qry_part, ctx_part, ctx_key_mask, log_s)
    upd = torch.cat((x_train.reshape(b, f, rows, e), x_test.reshape(b, f, r2 - rows, e)), dim=2).transpose(1, 2)
    x = _residual(layer, residual, upd)
    # 2. MLP
    residual = x
    x = _residual(layer, residual, layer.mlp[0](layer.layer_norms[1](x)))
    # 3. feature attention (per row, over the G + 1 tokens)
    residual = x
    h = layer.layer_norms[2](x).reshape(-1, f, e)
    upd = _feat_attention(layer.feature_attentions[0], h, group_key_mask).reshape(b, r2, f, e)
    return _residual(layer, residual, upd)


def masked_forward(model, x, y_norm, train_size, d):
    """x [1, T, H], y_norm [1, T] (standardised; any value off-context) -> quantiles for EVERY row [1, T, K]."""
    rows, width = x.shape[1], x.shape[2]
    fpg = model.features_per_group
    grouped_width = ((width + fpg - 1) // fpg) * fpg
    if fpg > 1:
        x = torch.cat((x, torch.zeros_like(x[:, :, :1]).expand(-1, -1, fpg - 1)), dim=-1)[:, :, :grouped_width]
    groups = grouped_width // fpg
    xg = x[0].reshape(rows, groups, fpg)
    row = torch.arange(rows, device=x.device)
    ctx = row < train_size                                                                 # [T]
    pre = preprocess(model, xg, ctx.view(-1, 1, 1))                                       # [T, G, 2]
    pre = torch.cat((pre, pre), dim=0).unsqueeze(0)                                       # [1, 2T, G, 2]
    x_emb = encode_x(model.encoder_x[0], pre)                                             # [1, 2T, G, E]

    nan = torch.full_like(y_norm[0], float("nan"))
    y_ctx = torch.where(ctx, y_norm[0], nan)                                              # [T]
    y_all = torch.cat((y_ctx, nan), dim=0)                                                # [2T]
    doubled_ctx = torch.cat((ctx, torch.zeros_like(ctx)), dim=0)
    # reg_y_encoder = NanEncoder (finite context mean fill + indicator) -> LinearEncoder([data, indicator])
    y_col = y_all.view(-1, 1)
    fill, _ = _masked_mean(y_col, doubled_ctx.view(-1, 1))
    nan_enc = model.reg_y_encoder[0]
    is_nan, is_pos, is_neg = torch.isnan(y_col), torch.isinf(y_col) & (y_col > 0), torch.isinf(y_col) & (y_col < 0)
    indicator = torch.where(is_nan, torch.full_like(y_col, nan_enc.nan_value),
                            torch.where(is_pos, torch.full_like(y_col, nan_enc.inf_value),
                                        torch.where(is_neg, torch.full_like(y_col, nan_enc.neg_info_value),
                                                    torch.zeros_like(y_col))))
    y_filled = torch.where(is_nan | is_pos | is_neg, fill.expand_as(y_col), y_col)
    embedded_y = model.reg_y_encoder[1].layer(torch.cat((y_filled, indicator), dim=-1)).unsqueeze(0)  # [1, 2T, E]

    # target-aware embedding: context rows get a bias from their label, every other row zero
    bias = model.reg_target_aware_embedding(torch.nan_to_num(y_all, nan=0.0).view(-1, 1).to(x_emb.dtype))
    bias = torch.where(doubled_ctx.view(-1, 1), bias, torch.zeros_like(bias)).unsqueeze(0)            # [1, 2T, E]
    scale = float(getattr(model, "target_aware_scale", 1.0))
    if scale != 1.0:
        bias = bias * scale
    rb = bias.unsqueeze(2)
    if model.use_column_specific_y_aware and model.column_y_aware_alpha is not None:
        col_score = (x_emb * rb).sum(dim=-1) * (1.0 / (model.embed_dim ** 0.5))
        gate = torch.sigmoid(col_score).unsqueeze(-1)
        alpha = torch.sigmoid(model.column_y_aware_alpha).to(x_emb.dtype)
        x_emb = x_emb + (1.0 - alpha) * rb + alpha * (rb * gate)
    else:
        x_emb = x_emb + rb

    state = torch.cat((x_emb, embedded_y.unsqueeze(2)), dim=2)                            # [1, 2T, G+1, E]
    ctx_key_mask = ctx
    log_s = torch.log(train_size.to(torch.float32).clamp_min(2.0))
    group_real = torch.arange(groups, device=x.device) * fpg < d                            # division-free
    group_key_mask = torch.cat((group_real, torch.ones(1, dtype=torch.bool, device=x.device)), dim=0)
    for layer in model.transformer_encoder.layers:
        state = masked_layer(layer, state, rows=rows, ctx_key_mask=ctx_key_mask, log_s=log_s,
                             group_key_mask=group_key_mask)
    out = model.encoder_out_norm(state)
    return _sequential(model.reg_y_decoder, out[:, rows:, -1])                            # [1, T, K]


class MaskExportWrapper(torch.nn.Module):
    """Nori under the train_size contract: x, y_full, train_size, d, n_rows (see the module doc)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y, train_size, d, n_rows):
        from export_nori.nori_patches import quantile_mean

        rows = x.shape[1]
        ctx = (torch.arange(rows, device=x.device) < train_size).unsqueeze(0)               # [1, T]
        zero = torch.zeros_like(y)
        y = torch.where(ctx, torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), zero)
        # Written exactly as the EXAONE wrapper's (which MIGraphX compiles): masking first, transposing after.
        # Transposing the boolean mask first and masking the transposed column made MIGraphX's GPU pipeline
        # fail (simplify_reshapes: "BROADCAST : axis 2 is out of range"), bisected on gfx1201 to these lines.
        n = _sum_rows(_f(ctx, y).transpose(0, 1))[0]                                       # [1] == S
        mean_y = _sum_rows(torch.where(ctx, y, zero).transpose(0, 1))[0] / n
        dev = torch.where(ctx, y - mean_y, zero)
        std_y = torch.sqrt(_sum_rows((dev * dev).transpose(0, 1))[0] / n)
        std_y = torch.where(std_y < 1e-12, torch.ones_like(std_y), std_y)
        q = masked_forward(self.m, x, (y - mean_y) / std_y, train_size, d)                 # [1, T, K]
        return quantile_mean(q).unsqueeze(-1) * std_y + mean_y
