"""EXAONE Tabular with the train/test split as a VALUE instead of a shape (the ROCm / MIGraphX contract).

The CPU graph (``exaone_patches.ExportWrapper``) is ``(x, y)``-only: the split is ``len(y)``, a SHAPE, which a
fixed-shape MIGraphX compile cannot bucket (docs/ROCM_SINGLE_EVAL_POS.md). This module computes the same
function with every shape a function of the padded (T, H) bucket alone, the contract tabdpt and the TabPFN
family already serve on ROCm:

    x[1, T, H] f32   y[1, T] f32 (FULL length; non-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)   n_rows[1] i64 (real row count before padding)
    -> logits[1, T, C]   (classification) / [1, T, 1] (regression), for every row

The plugin pads rows past ``n_rows`` and columns past ``d`` with zeros, so every place the CPU forward uses a
shape that would now include padding becomes a mask:

  1. feature-encoder statistics (imputation mean, mean, unbiased std) over the CONTEXT rows (row < S);
  2. its constant-column test over the REAL rows (row < n_rows), and only for real columns (col < d): a padded
     zero row would otherwise "un-constant" a constant column, and upstream runs this test over support and
     query rows alike;
  3. the label encoders: support mean and the class ranks over the context rows only. The class rank uses the
     engine's dense ids 0..k-1 (k <= the 10-class head): rank(v) = #{present classes c < v}, a [T, 10] table
     instead of the [T, T] first-occurrence table the CPU graph needs for arbitrary labels;
  4. the regression target statistics over the context rows;
  5. the feature self-attention per row: padded columns are masked out as KEYS, and the SSMax length is the
     REAL token count 3 + d + 1, not the padded one;
  6. the support self-attention and the queries' cross-attention per column: non-context rows are masked out
     as keys, and the SSMax length is 32 + S.

The context is still presented a second time as queries (the doubled table of the CPU wrapper), now as two
static T-row copies: the first copy plays the context role (only its rows < S are keys), the second is every
row as a query. The encoders run once and are duplicated, which is exact: duplicating rows changes neither the
support statistics nor the constant-column test.

MIGraphX-specific choices, each learnt by the earlier conversions (docs/ROCM_TABDPT_SPIKE.md,
docs/ROCM_TABPFN_PLAN.md): masked row sums are GEMMs (``ones @ x``), because chained masked reductions crash
MIGraphX's reducer codegen; ``train_size`` / ``d`` / ``n_rows`` stay rank 1 (a rank-0 Reshape is refused);
there is no integer division on runtime data; the key bias is -1e4, finite in every float format.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from export_exaone.exaone_patches import FLOAT32_EPS, _feedforward, _heads, _nonfinite_codes

NEG = -1e4
CLASS_TABLE = 10  # the released head's width; class ids are the engine's dense 0..k-1


def _sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum over the first axis as a GEMM (ones[1, R] @ x[R, rest]): x.sum(0, keepdim=True)."""
    rows = x.shape[0]
    flat = x.reshape(rows, -1)
    ones = torch.ones(1, rows, dtype=flat.dtype, device=flat.device)
    return (ones @ flat).reshape(1, *x.shape[1:])


def _f(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask.to(like.dtype)


def masked_attention(mod, query, context=None, *, key_mask=None, log_len=None, first_context_head=False):
    """``exaone_patches.attention`` with an additive key bias and an explicit SSMax length.

    key_mask broadcasts against the key axis ([..., 1, Lk] after unsqueeze), log_len against the heads."""
    q = _heads(mod, F.linear(query, mod.query_weight))
    q = q * mod.head_scale.reshape(*((1,) * (q.ndim - 3)), mod.head_count, 1, 1)
    ctx = query if context is None else context
    k = _heads(mod, F.linear(ctx, mod.key_weight))
    v = _heads(mod, F.linear(ctx, mod.value_weight))
    if first_context_head:
        k = k[..., :1, :, :]
        v = v[..., :1, :, :]
    if log_len is None:
        k = k * math.log(ctx.shape[-2]) if isinstance(ctx.shape[-2], int) else k * torch.log(
            torch.ones(ctx.shape[-2], dtype=k.dtype, device=k.device).sum())
    else:
        k = k * log_len
    if first_context_head:
        k = k.expand(*k.shape[:-3], mod.head_count, *k.shape[-2:])
        v = v.expand(*v.shape[:-3], mod.head_count, *v.shape[-2:])
    scale = 1.0 / math.sqrt(mod.head_width) if mod.score_scale is None else float(mod.score_scale)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if key_mask is not None:
        scores = scores + (1.0 - _f(key_mask, scores)) * NEG
    attended = torch.matmul(torch.softmax(scores, dim=-1), v)
    return F.linear(attended.transpose(-3, -2).flatten(-2), mod.output_weight)


def masked_feature_encoder(enc, source: torch.Tensor, ctx: torch.Tensor, real: torch.Tensor,
                           feat: torch.Tensor) -> torch.Tensor:
    """source [T, H] -> [T, H, width]; ctx/real [T] bool, feat [H] bool."""
    ctx_c, real_c = ctx.unsqueeze(1), real.unsqueeze(1)
    changed = (source != source[:1]) & real_c                       # row 0 is real (n_rows >= 1)
    retained = (_sum_rows(_f(changed, source))[0] > 0) & feat       # [H]
    finite = torch.isfinite(source)
    zero = torch.zeros_like(source)
    finite_support = finite & ctx_c
    counts = _sum_rows(_f(finite_support, source))[0]
    sums = _sum_rows(torch.where(finite_support, source, zero))[0]
    replacements = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    replaced = torch.where(finite, source, replacements.unsqueeze(0))
    n_ctx = _sum_rows(_f(ctx_c, source))[0]                         # [1] == S
    means = _sum_rows(torch.where(ctx_c, replaced, zero))[0] / n_ctx
    centred = torch.where(ctx_c, replaced - means, zero)
    deviations = torch.sqrt(_sum_rows(centred * centred)[0] / (n_ctx - 1.0).clamp_min(1.0))
    deviations = torch.where(deviations == 0, torch.ones_like(deviations), deviations)
    normalized = ((replaced - means) / (deviations + 1e-16)).clamp(-100, 100)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=100.0, neginf=-100.0)
    keep = retained.unsqueeze(0)
    values = torch.where(keep, normalized, zero).unsqueeze(-1)
    codes = torch.where(keep, _nonfinite_codes(source), zero).unsqueeze(-1)
    return torch.nan_to_num(F.linear(torch.cat((values, codes), dim=-1), enc.weight, enc.bias))


def masked_label_encoder(enc, labels: torch.Tensor, ctx: torch.Tensor, *, regression: bool) -> torch.Tensor:
    """labels [R] (NaN outside the context) -> [R, width]; ctx [R] bool marks the support rows."""
    codes = _nonfinite_codes(labels)
    finite_support = torch.isfinite(labels) & ctx
    zero = torch.zeros_like(labels)
    total = _sum_rows(torch.where(finite_support, labels, zero).unsqueeze(1))[0, 0]
    count = _sum_rows(_f(finite_support, labels).unsqueeze(1))[0, 0]
    replacement = torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total))
    replaced = torch.where(torch.isfinite(labels), labels, replacement)
    if regression:
        first_channel = replaced
    else:
        classes = torch.arange(CLASS_TABLE, dtype=labels.dtype, device=labels.device)
        present_rows = (replaced.unsqueeze(1) == classes.unsqueeze(0)) & ctx.unsqueeze(1)    # [R, 10]
        present = _sum_rows(_f(present_rows, labels))[0] > 0                                  # [10]
        below = replaced.unsqueeze(1) > classes.unsqueeze(0)                                  # [R, 10]
        first_channel = _sum_rows(_f(below & present.unsqueeze(0), labels).transpose(0, 1))[0]
    channels = torch.stack((first_channel, codes), dim=-1)
    return torch.nan_to_num(F.linear(channels, enc.weight, enc.bias))


def masked_layer_forward(layer, state, feature_summary, item_summary, *, feat_key_mask, feat_log_len,
                         ctx_key_mask, ctx_log_len, rows):
    """``exaone_patches.layer_forward`` on the doubled [1, 2T, G, W] state; rows = T (the copy length)."""
    norm = layer._normalize
    n_item = layer.item_summary_count
    for repeat, attn_mod in enumerate(layer.feature_attentions):
        if layer.feature_summary_count == 1:
            feature_context = state + feature_summary[:, 0].unsqueeze(1)
        else:
            mixed = masked_attention(layer.feature_context_attentions[repeat], state.transpose(1, 2),
                                     feature_summary.transpose(1, 2)).transpose(1, 2)
            feature_context = state + mixed
        joined = torch.cat((item_summary, feature_context), dim=2)
        joined = norm(joined + masked_attention(attn_mod, joined, key_mask=feat_key_mask, log_len=feat_log_len))
        item_summary = joined[:, :, :n_item]
        state = joined[:, :, n_item:]
        feature_summary = norm(feature_summary)

    if n_item == 1:
        item_context = state + item_summary[:, :, 0].unsqueeze(2)
    else:
        item_context = state + masked_attention(layer.item_context_attention, state, item_summary)

    context_sequence = torch.cat((feature_summary, item_context[:, :rows]), dim=1).transpose(1, 2)
    updated_prefix = (context_sequence + masked_attention(layer.item_attention, context_sequence,
                                                          key_mask=ctx_key_mask, log_len=ctx_log_len)
                      ).transpose(1, 2)
    summary_rows = updated_prefix[:, : layer.feature_summary_count]
    updated_support = updated_prefix[:, layer.feature_summary_count:]

    query_rows = item_context[:, rows:]
    attended = masked_attention(layer.item_attention, query_rows.transpose(1, 2), context_sequence,
                                key_mask=ctx_key_mask, log_len=ctx_log_len,
                                first_context_head=True).transpose(1, 2)
    state = torch.cat((updated_support, query_rows + attended), dim=1)
    feature_summary = summary_rows

    state, feature_summary, item_summary = norm(state), norm(feature_summary), norm(item_summary)
    state = _feedforward(layer.state_feedforward, state)
    if not layer.final_layer:
        feature_summary = _feedforward(layer.feature_summary_feedforward, feature_summary)
        item_summary = _feedforward(layer.item_summary_feedforward, item_summary)
    return norm(state), norm(feature_summary), norm(item_summary)


def support_only(labels: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """Non-context labels become NaN, which is what the positional forward hands its label encoder for the
    query placeholders: they then take part in no statistic and get the non-finite code."""
    return torch.where(ctx, labels, torch.full_like(labels, float("nan")))


def masked_model_forward(model, x, labels, train_size, d, n_rows, *, regression: bool):
    """x [1, T, H], labels [1, T] (any value outside the context) -> head output for EVERY row [1, T, C]."""
    transformer = model.transformer
    rows, width = x.shape[1], x.shape[2]
    row = torch.arange(rows, device=x.device)
    ctx = row < train_size
    real = row < n_rows
    feat = torch.arange(width, device=x.device) < d

    feature_state = masked_feature_encoder(model.feature_encoder, x[0], ctx, real, feat)    # [T, H, W]
    nan = torch.full_like(labels[0], float("nan"))
    support_labels = support_only(labels[0], ctx)
    doubled_ctx = torch.cat((ctx, torch.zeros_like(ctx)), dim=0)
    label_state = masked_label_encoder(model.label_encoder, torch.cat((support_labels, nan), dim=0),
                                       doubled_ctx, regression=regression)                  # [2T, W]
    feature_state = torch.cat((feature_state, feature_state), dim=0)                        # [2T, H, W]
    state = torch.cat((feature_state, label_state.unsqueeze(1)), dim=1).unsqueeze(0)        # [1, 2T, G, W]

    n_item, n_fs = transformer.item_summary_tokens.shape[2], transformer.feature_summary_tokens.shape[1]
    ones_item = torch.ones(n_item, dtype=torch.bool, device=x.device)
    ones_fs = torch.ones(n_fs, dtype=torch.bool, device=x.device)
    true1 = torch.ones(1, dtype=torch.bool, device=x.device)
    feat_key_mask = torch.cat((ones_item, feat, true1), dim=0)                              # [3 + H + 1]
    ctx_key_mask = torch.cat((ones_fs, ctx), dim=0)                                         # [32 + T]
    f32 = state.dtype
    feat_log_len = torch.log(d.to(f32) + float(n_item + 1))                                 # [1]
    ctx_log_len = torch.log(train_size.to(f32) + float(n_fs))

    groups = state.shape[2]
    feature_summary = transformer.feature_summary_tokens.expand(1, -1, groups, -1)
    item_summary = transformer.item_summary_tokens.expand(1, 2 * rows, -1, -1)
    for layer in transformer.layers:
        state, feature_summary, item_summary = masked_layer_forward(
            layer, state, feature_summary, item_summary, feat_key_mask=feat_key_mask, feat_log_len=feat_log_len,
            ctx_key_mask=ctx_key_mask, ctx_log_len=ctx_log_len, rows=rows)
    return transformer.classification_heads["standard"](state[:, rows:, -1, :])


class MaskExportWrapper(torch.nn.Module):
    """EXAONE Tabular under the train_size contract: x, y_full, train_size, d, n_rows (see the module doc)."""

    def __init__(self, model, task: str):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.m = model
        self.task = task

    def forward(self, x, y, train_size, d, n_rows):
        regression = self.task == "regression"
        rows = x.shape[1]
        ctx = (torch.arange(rows, device=x.device) < train_size).unsqueeze(0)               # [1, T]
        # whatever sits in non-context positions must never reach the arithmetic (0 * NaN is NaN)
        y = torch.where(ctx, torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(y))
        if regression:
            zero = torch.zeros_like(y)
            n = _sum_rows(_f(ctx, y).transpose(0, 1))[0]                                     # [1] == S
            center = _sum_rows(torch.where(ctx, y, zero).transpose(0, 1))[0] / n
            dev = torch.where(ctx, y - center, zero)
            variance = _sum_rows((dev * dev).transpose(0, 1))[0] / n
            constant = variance <= n * FLOAT32_EPS * variance + (n * center * FLOAT32_EPS) ** 2
            scale = torch.where(constant, torch.ones_like(variance), variance.sqrt())
            y_in = (y - center) / scale
        else:
            y_in = y
        out = masked_model_forward(self.m, x, y_in, train_size, d, n_rows, regression=regression)
        if regression:
            # mean over the quantile bank as a GEMM against a constant 1/K column (no reduction op)
            k = out.shape[-1]
            weights = torch.full((k, 1), 1.0 / k, dtype=out.dtype, device=out.device)
            return (out @ weights) * scale + center
        return out
