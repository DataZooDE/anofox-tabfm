"""LimiX-2M with the train/test split as a VALUE instead of a shape (the ROCm / MIGraphX contract).

The CPU graph (``limix_patches.ExportWrapper``) is ``(x, y)``-only: the split is ``len(y)``, a SHAPE, which a
fixed-shape MIGraphX compile cannot bucket (docs/ROCM_SINGLE_EVAL_POS.md). This module computes the same
function with every shape a function of the padded (T, H) bucket alone:

    x[1, T, H] f32   y[1, T] f32 (FULL length; off-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)   n_rows[1] i64 (real row count)
    -> logits[1, T, C]: class logits (C = 10) or a raw-space point estimate (C = 1) for every row

It re-expresses ``FeaturesTransformer.forward`` (the released config: smf layers, post-norm, isolated sample
attention, RBF tokenizer) over upstream's own modules. Nori-30M is a LimiX fork, so this is
``tools/export_nori/nori_mask.py`` with LimiX's differences (post-norm, no QASSMax, no target-aware
embedding, a classification head, and a valid-feature test over every row). Every place that used a shape
the plugin's padding now inflates becomes a mask:

  1. the NaN-imputation mean and the normalisation mean / unbiased std, over the CONTEXT rows (row < S);
     upstream's ``x[:, :eval_pos]``;
  2. the valid-feature test (``ValidFeatureEncoder``: a column is valid unless every row equals the first),
     over every REAL row (row < n_rows): upstream reads the whole ``[context ; all rows]`` table, whose rows
     are exactly the real rows, and a zero padded row would make a constant column look valid;
  3. the label path: the NaN imputation over the context, the class rank among the DISTINCT context labels
     (a presence table over the 10 class ids instead of an S x S comparison), the label embedding on the
     context rows and the mask embedding everywhere else;
  4. the feature attention per row: padded feature GROUPS are masked out as keys. A group is real when
     2 * g < d (division-free: MIGraphX miscompiles a runtime integer floor-division, docs/ROCM_TABPFN_PLAN.md).
     A padded column inside the last real group (odd d) is exactly upstream's own pad column;
  5. the sample attention: the context self-attention and the queries' cross-attention mask out
     non-context rows as keys;
  6. regression: the target standardisation reads the context rows only.

The context is presented a second time as queries, as in the CPU wrapper, as two static T-row copies: the
first copy plays the context role (only its rows < S are keys), the second is every row as a query. The
preprocessing runs once and is duplicated, which is exact (its statistics read the context or the real rows).
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


def _layer_norm(ln, x):
    """``nn.LayerNorm`` written out (mean, biased variance, optional affine). The ONNX LayerNormalization op
    broke MIGraphX's simplify_reshapes on the Nori graph ("BROADCAST : axis 2 is out of range",
    tools/export_nori/nori_mask.py); plain elementwise ops broadcast with multibroadcast instead."""
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
    """``MaskEmbEncoder.forward`` (RBF tokenizer): [B, R, G, 2] -> [B, R, G, E]."""
    b, r, g, fpg = data.shape
    x = data.unsqueeze(-1)
    is_mask = torch.isnan(x)
    x = x.masked_fill(is_mask, 0.0)
    x_emb = rbf_tokenizer(enc.numeric_mlp, x)
    combined = torch.where(is_mask, enc.mask_embedding.expand_as(x_emb), x_emb)
    return _sequential(enc.fusion_network, combined.flatten(3)).view(b, r, g, -1)


def preprocess(model, xg: torch.Tensor, ctx: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """x_preprocess + process_4_x for one table: xg [T, G, 2] (grouped, padded) -> [T, G, 2].

    NanEncoder (context mean) -> NormalizationEncoder (context mean / unbiased std, clip +-100) ->
    ValidFeatureEncoder (over the real rows) -> NaN where the input was NaN. ctx and real are [T, 1, 1]."""
    is_nan = torch.isnan(xg)
    nonfinite = ~torch.isfinite(xg)
    # NanEncoder (calc_mean counts the non-NaN cells, as upstream): NaN / inf cells take the context mean
    keep = ~is_nan & ctx
    num = _sum_rows(_f(keep, xg)).clamp_min(1)
    mean_fill = _sum_rows(torch.where(keep, xg, torch.zeros_like(xg))) / num
    x = torch.where(nonfinite, mean_fill.expand_as(xg), xg)
    # NormalizationEncoder: (x - mean) / (unbiased std + 1e-20) over the context, clip to +-100 (S >= 2)
    keep = ctx.expand_as(x)
    num = _sum_rows(_f(keep, x)).clamp_min(1)
    mean = _sum_rows(torch.where(keep, x, torch.zeros_like(x))) / num
    sq = torch.where(keep, (mean - x) ** 2, torch.zeros_like(x))
    std = torch.sqrt(_sum_rows(sq) / (num - 1)) + 1e-20
    x = torch.clip((x - mean) / std, min=-100, max=100)
    # ValidFeatureEncoder: a column is valid unless every REAL row equals the first row
    differs = (x != x[0:1]) & real
    valid = _sum_rows(_f(differs, x)) > 0                                            # [1, G, 2]
    valid_num = torch.clip(_f(valid, x).sum(-1, keepdim=True), min=1)                # [1, G, 1]
    x = x * torch.sqrt(x.shape[-1] / valid_num)
    # process_4_x: the original NaN cells become NaN again (the MaskEmbEncoder's mask embedding)
    return torch.where(is_nan, torch.full_like(x, float("nan")), x)


def _seq_attention(attn, query, key_value, key_mask):
    """LimiX MultiheadAttention (qkv_combined=False, no kv-head sharing) with a key mask.

    query [N, Lq, E], key_value [N, Lk, E], key_mask [Lk] bool. Upstream's einsum projections as GEMMs."""
    h_, d_, e_ = attn.num_heads, attn.head_dim, attn.embed_dim
    q = F.linear(query, attn.qkv_proj_weight[0].reshape(h_ * d_, e_)).unflatten(-1, (h_, d_))       # [N, Lq, h, d]
    kv = F.linear(key_value, attn.qkv_proj_weight[1:].reshape(2 * h_ * d_, e_)).unflatten(-1, (2, h_, d_))
    k, v = kv[:, :, 0], kv[:, :, 1]
    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)                   # [N, h, L, d]
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * (1.0 / math.sqrt(attn.head_dim))
    scores = scores + (1.0 - _f(key_mask, scores)) * NEG
    out = torch.matmul(torch.softmax(scores, dim=-1), vh).transpose(1, 2)                  # [N, Lq, h, d]
    return F.linear(out.flatten(-2), attn.out_proj_weight.reshape(h_ * d_, e_).transpose(0, 1))


def _feat_attention(attn, x, key_mask):
    """LimiX MultiheadAttention (qkv_combined=True) over the tokens of each row, with a key mask.
    x [N, F, E], key_mask [F] bool."""
    h_, d_, e_ = attn.num_heads, attn.head_dim, attn.embed_dim
    qkv = F.linear(x, attn.qkv_proj_weight.reshape(3 * h_ * d_, e_)).unflatten(-1, (3, h_, d_))  # [N, F, 3, h, d]
    q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(attn.head_dim))
    scores = scores + (1.0 - _f(key_mask, scores)) * NEG
    out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2)
    return F.linear(out.flatten(-2), attn.out_proj_weight.reshape(h_ * d_, e_).transpose(0, 1))


def masked_layer(layer, x, *, rows, ctx_key_mask, group_key_mask):
    """EncoderBaseLayer (smf, POST-norm, seq_attn_isolated, not serial) on the doubled [1, 2T, G+1, E] state."""
    if layer.pre_norm or layer.layer_arch != "smf" or not layer.seq_attn_isolated or layer.seq_attn_serial:
        raise ValueError("the masked layer covers the released config only (smf, post-norm, isolated, not serial)")
    if layer.self_share_all_kv_heads or layer.cross_share_all_kv_heads:
        raise ValueError("the masked layer covers the released config only (no kv-head sharing)")
    b, r2, f, e = x.shape
    norms = layer.layer_norms
    # 1. sample attention: the first copy's context rows attend the context; the second copy (every row) too
    ctx_part = x[:, :rows].transpose(1, 2).reshape(-1, rows, e)                             # [F, T, E]
    qry_part = x[:, rows:].transpose(1, 2).reshape(-1, r2 - rows, e)
    x_train = _seq_attention(layer.sequence_attentions[0], ctx_part, ctx_part, ctx_key_mask)
    x_test = _seq_attention(layer.sequence_attentions[1], qry_part, ctx_part, ctx_key_mask)
    upd = torch.cat((x_train.reshape(b, f, rows, e), x_test.reshape(b, f, r2 - rows, e)), dim=2).transpose(1, 2)
    x = _layer_norm(norms[0], x + upd)
    # 2. MLP
    x = _layer_norm(norms[1], x + layer.mlp[0](x))
    # 3. feature attention (per row, over the G + 1 tokens)
    upd = _feat_attention(layer.feature_attentions[0], x.reshape(-1, f, e), group_key_mask).reshape(b, r2, f, e)
    return _layer_norm(norms[2], x + upd)


def _nan_encode(enc, y_col, ctx_col):
    """NanEncoder on a label column [R, 1]: non-finite cells take the context mean, plus the indicator."""
    keep = ~torch.isnan(y_col) & ctx_col
    num = _sum_rows(_f(keep, y_col)).clamp_min(1)
    fill = _sum_rows(torch.where(keep, y_col, torch.zeros_like(y_col))) / num
    is_nan, is_pos, is_neg = torch.isnan(y_col), torch.isinf(y_col) & (y_col > 0), torch.isinf(y_col) & (y_col < 0)
    indicator = torch.where(is_nan, torch.full_like(y_col, enc.nan_value),
                            torch.where(is_pos, torch.full_like(y_col, enc.inf_value),
                                        torch.where(is_neg, torch.full_like(y_col, enc.neg_info_value),
                                                    torch.zeros_like(y_col))))
    filled = torch.where(is_nan | is_pos | is_neg, fill.expand_as(y_col), y_col)
    return filled, indicator


def embed_y(model, task, y_ctx, ctx):
    """The y token for the doubled table [2T]: y_ctx [T] holds the labels on the context rows (NaN elsewhere).
    Returns [1, 2T, E] float32 after upstream's float16 round trip."""
    rows = y_ctx.shape[0]
    nan = torch.full_like(y_ctx, float("nan"))
    y_all = torch.cat((y_ctx, nan), dim=0).view(-1, 1)                                # [2T, 1]
    doubled_ctx = torch.cat((ctx, torch.zeros_like(ctx)), dim=0).view(-1, 1)
    if task == "cls":
        steps = model.cls_y_encoder
        y_filled, _ = _nan_encode(steps[0], y_all, doubled_ctx)
        emb_step = steps[-1]
        n_classes = emb_step.y_embedding.num_embeddings
        # MulticlassTargetEncoder: rank among the DISTINCT context labels, as a presence table over the class
        # ids (the engine feeds dense ids 0..C-1, C <= 10): rank_i = #{v present in the context : v < y_i}
        vals = torch.arange(n_classes, device=y_all.device, dtype=y_all.dtype).view(1, -1)          # [1, K]
        present = _sum_rows(_f((y_all == vals) & doubled_ctx, y_all)) > 0                         # [1, K]
        rank = (_f(vals < y_filled, y_all) * _f(present, y_all)).sum(dim=-1)                       # [2T]
        # EmbYEncoderStep: context rows embed their rank, every other row the mask embedding (index 0)
        lab = emb_step.y_embedding(torch.where(doubled_ctx.view(-1), rank, torch.zeros_like(rank)).to(torch.long))
        msk = emb_step.y_mask.weight[0].expand_as(lab)
        emb = torch.where(doubled_ctx, lab, msk)
    else:
        steps = model.reg_y_encoder
        y_filled, indicator = _nan_encode(steps[0], y_all, doubled_ctx)
        emb = steps[1].layer(torch.cat((y_filled, indicator), dim=-1))
    # upstream writes the embedding into a float16 buffer (mixed_y_embedding); keep that rounding
    return emb.to(torch.float16).to(torch.float32).view(1, 2 * rows, -1)


def masked_forward(model, task, x, y_lab, train_size, d, n_rows):
    """x [1, T, H], y_lab [1, T] (labels on the context; anything elsewhere) -> the decoder output [1, T, C]."""
    rows, width = x.shape[1], x.shape[2]
    fpg = model.features_per_group
    grouped_width = ((width + fpg - 1) // fpg) * fpg
    if fpg > 1:
        x = torch.cat((x, torch.zeros_like(x[:, :, :1]).expand(-1, -1, fpg - 1)), dim=-1)[:, :, :grouped_width]
    groups = grouped_width // fpg
    xg = x[0].reshape(rows, groups, fpg)
    row = torch.arange(rows, device=x.device)
    ctx = row < train_size                                                                 # [T]
    real = row < n_rows
    pre = preprocess(model, xg, ctx.view(-1, 1, 1), real.view(-1, 1, 1))                  # [T, G, 2]
    # the two copies as expand + reshape, not torch.cat([t, t]): a Concat listing one tensor twice preceded a
    # fuse_concat assertion on the Causilo graph (docs/ROCM_CAUSILO_PLAN.md)
    pre = pre.unsqueeze(0).expand(2, *pre.shape).reshape(1, 2 * rows, *pre.shape[1:])  # [1, 2T, G, 2]
    x_emb = encode_x(model.encoder_x[0], pre)                                             # [1, 2T, G, E]

    y_ctx = torch.where(ctx, y_lab[0], torch.full_like(y_lab[0], float("nan")))
    y_emb = embed_y(model, task, y_ctx, ctx)                                              # [1, 2T, E]
    state = torch.cat((x_emb, y_emb.unsqueeze(2)), dim=2)                                 # [1, 2T, G+1, E]

    group_real = torch.arange(groups, device=x.device) * fpg < d                            # division-free
    group_key_mask = torch.cat((group_real, torch.ones(1, dtype=torch.bool, device=x.device)), dim=0)
    for layer in model.transformer_encoder.layers:
        state = masked_layer(layer, state, rows=rows, ctx_key_mask=ctx, group_key_mask=group_key_mask)
    if not isinstance(model.encoder_out_norm, torch.nn.Identity):
        raise ValueError("the masked forward covers the released config only (post-norm, no output norm)")
    out = state[:, rows:, -1]                                                             # [1, T, E]
    decoder = model.cls_y_decoder if task == "cls" else model.reg_y_decoder
    return _sequential(decoder, out)


class MaskExportWrapper(torch.nn.Module):
    """LimiX under the train_size contract: x, y (full length), train_size, d, n_rows (see the module doc)."""

    def __init__(self, model, task: str = "classification"):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.m = model
        self.task = task

    def forward(self, x, y, train_size, d, n_rows):
        rows = x.shape[1]
        ctx = (torch.arange(rows, device=x.device) < train_size).unsqueeze(0)               # [1, T]
        zero = torch.zeros_like(y)
        y = torch.where(ctx, torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), zero)
        if self.task == "classification":
            return masked_forward(self.m, "cls", x, y, train_size, d, n_rows)
        # population std over the context, scale at rounding level -> 1 (the positional wrapper's), as masked
        # sums written masking-first (the order MIGraphX compiles, tools/export_nori/nori_mask.py)
        n = _sum_rows(_f(ctx, y).transpose(0, 1))[0]                                       # [1] == S
        mean_y = _sum_rows(torch.where(ctx, y, zero).transpose(0, 1))[0] / n
        dev = torch.where(ctx, y - mean_y, zero)
        std_y = torch.sqrt(_sum_rows((dev * dev).transpose(0, 1))[0] / n)
        std_y = torch.where(std_y < 1e-6, torch.ones_like(std_y), std_y)
        out = masked_forward(self.m, "reg", x, (y - mean_y) / std_y, train_size, d, n_rows)  # [1, T, 1]
        return out * std_y + mean_y
