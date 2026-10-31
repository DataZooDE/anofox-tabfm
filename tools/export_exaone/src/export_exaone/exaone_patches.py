"""Export-friendly forward for LG AI Research's EXAONE Tabular (upstream code BSD-3-Clause-LG).

We DO NOT copy or edit the upstream source: ``exaonetabular`` is pinned to an exact git commit
(pyproject.toml) and its modules are used as they are; this file re-expresses their INFERENCE forward
over the upstream modules' own parameters, and tests prove it equal to upstream's forward
(``tests/test_export.py``, ``tests/test_real_weights.py``).

Why a re-expression rather than a few patches (the Nori / LimiX approach): upstream's forward is built
for a CUDA runtime, and almost every step of it is hostile to a trace at symbolic shapes:

  * every module validates its arguments with ``isinstance(support_size, int)`` and friends, which a
    symbolic size is not;
  * ``TensorAttention`` packs its projection weights lazily into buffers keyed by ``data_ptr()`` and
    caches the support keys/values in module state between the support and the query halves of a layer;
  * the attention kernel is pinned with ``sdpa_kernel([FLASH_ATTENTION])`` (``float32`` is CPU-only for
    that reason); the export writes the plain attention ``softmax(q k^T / sqrt(d)) v`` instead, which is
    the same function and what every ONNX backend can run;
  * the SSMax temperature multiplies the keys by ``math.log(context_length)``, a Python float of a
    symbolic size;
  * the classification label encoder ranks labels with ``torch.unique`` (a data-dependent shape), and the
    label imputation takes ``values[finite].mean()`` behind ``if not finite.any()``.

What is reproduced, step for step (``CrossAxisSummaryLayer.forward`` with ``output_support=True`` for
every layer): feature-summary cross-attention, the item-summary + cells feature self-attention (two
repeats), item-summary context, the support self-attention with the 32 feature-summary tokens prefixed,
the queries' multi-query cross-attention onto the FIRST context head, LayerNorms and feed-forwards. The
final layer upstream skips the support rows' update (it is never read); computing it changes nothing a
query row sees, so it is computed here and the graph needs no special final layer.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

FLOAT32_EPS = float(torch.finfo(torch.float32).eps)


def _length(n, like: torch.Tensor) -> torch.Tensor:
    """A (possibly symbolic) size as a float32 scalar tensor, without specialising the trace on it."""
    return torch.ones(n, dtype=torch.float32, device=like.device).sum()


def _heads(mod, projected: torch.Tensor) -> torch.Tensor:
    # (..., L, E) -> (..., H, L, hd), upstream's TensorAttention._as_heads
    return projected.unflatten(-1, (mod.head_count, mod.head_width)).transpose(-3, -2)


def attention(mod, query: torch.Tensor, context: torch.Tensor | None = None, *,
              first_context_head: bool = False) -> torch.Tensor:
    """``TensorAttention.forward`` on its inference path: SSMax keys, optional head-shared context."""
    q = _heads(mod, F.linear(query, mod.query_weight))
    q = q * mod.head_scale.reshape(*((1,) * (q.ndim - 3)), mod.head_count, 1, 1)
    ctx = query if context is None else context
    k = _heads(mod, F.linear(ctx, mod.key_weight))
    v = _heads(mod, F.linear(ctx, mod.value_weight))
    if first_context_head:
        k = k[..., :1, :, :]
        v = v[..., :1, :, :]
    k = k * torch.log(_length(ctx.shape[-2], ctx))
    if first_context_head:
        k = k.expand(*k.shape[:-3], mod.head_count, *k.shape[-2:])
        v = v.expand(*v.shape[:-3], mod.head_count, *v.shape[-2:])
    # upstream passes score_scale to SDPA as `scale=`; None is SDPA's default 1/sqrt(d)
    scale = 1.0 / math.sqrt(mod.head_width) if mod.score_scale is None else float(mod.score_scale)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attended = torch.matmul(torch.softmax(scores, dim=-1), v)
    joined = attended.transpose(-3, -2).flatten(-2)
    return F.linear(joined, mod.output_weight)


def _feedforward(mod, value: torch.Tensor) -> torch.Tensor:
    expanded = F.linear(value, mod.expansion_weight)
    expanded = F.gelu(expanded, approximate="none") if mod.activation == "gelu" else F.relu(expanded)
    return value + F.linear(expanded, mod.projection_weight)


def layer_forward(layer, state, feature_summary, item_summary, support_size):
    """``CrossAxisSummaryLayer.forward`` (no cache, every row updated)."""
    norm = layer._normalize
    n_item = layer.item_summary_count
    for repeat, attn_mod in enumerate(layer.feature_attentions):
        if layer.feature_summary_count == 1:
            feature_context = state + feature_summary[:, 0].unsqueeze(1)
        else:
            mixed = attention(layer.feature_context_attentions[repeat], state.transpose(1, 2),
                              feature_summary.transpose(1, 2)).transpose(1, 2)
            feature_context = state + mixed
        joined = torch.cat((item_summary, feature_context), dim=2)
        joined = norm(joined + attention(attn_mod, joined))
        item_summary = joined[:, :, :n_item]
        state = joined[:, :, n_item:]
        feature_summary = norm(feature_summary)

    if n_item == 1:
        item_context = state + item_summary[:, :, 0].unsqueeze(2)
    else:
        item_context = state + attention(layer.item_context_attention, state, item_summary)

    context_sequence = torch.cat((feature_summary, item_context[:, :support_size]), dim=1).transpose(1, 2)
    updated_prefix = (context_sequence + attention(layer.item_attention, context_sequence)).transpose(1, 2)
    summary_rows = updated_prefix[:, : layer.feature_summary_count]
    updated_support = updated_prefix[:, layer.feature_summary_count:]

    query_rows = item_context[:, support_size:]
    attended = attention(layer.item_attention, query_rows.transpose(1, 2), context_sequence,
                         first_context_head=True).transpose(1, 2)
    state = torch.cat((updated_support, query_rows + attended), dim=1)
    feature_summary = summary_rows

    state, feature_summary, item_summary = norm(state), norm(feature_summary), norm(item_summary)
    state = _feedforward(layer.state_feedforward, state)
    if not layer.final_layer:
        feature_summary = _feedforward(layer.feature_summary_feedforward, feature_summary)
        item_summary = _feedforward(layer.item_summary_feedforward, item_summary)
    return norm(state), norm(feature_summary), norm(item_summary)


def _nonfinite_codes(values: torch.Tensor) -> torch.Tensor:
    # upstream encoders._nonfinite_codes: NaN -2, +inf 2, -inf 4, finite 0
    codes = torch.zeros_like(values)
    codes = torch.where(torch.isnan(values), torch.full_like(values, -2.0), codes)
    codes = torch.where(torch.isposinf(values), torch.full_like(values, 2.0), codes)
    return torch.where(torch.isneginf(values), torch.full_like(values, 4.0), codes)


def _finite_mean(values: torch.Tensor, dim: int) -> torch.Tensor:
    """upstream ``_finite_support_mean`` (0 when nothing is finite), without the data-dependent branch."""
    finite = torch.isfinite(values)
    count = finite.sum(dim=dim)
    total = torch.where(finite, values, torch.zeros_like(values)).sum(dim=dim)
    return torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total))


def feature_encoder(enc, features: torch.Tensor, support_size) -> torch.Tensor:
    """``FeatureEncoder._forward_single_coordinate`` (the released config groups ONE column per token).

    features [rows, groups, 1] -> [rows, groups, width]. Note the constant-column test runs over ALL rows,
    support and query: upstream is transductive here, and the export keeps it."""
    source = features[..., 0]
    retained = (source[1:] != source[0]).any(dim=0)
    finite = torch.isfinite(source)
    finite_support = finite[:support_size]
    support_values = source[:support_size]
    counts = finite_support.sum(dim=0)
    sums = torch.where(finite_support, support_values, torch.zeros_like(support_values)).sum(dim=0)
    replacements = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    replaced = torch.where(finite, source, replacements.unsqueeze(0))
    support = replaced[:support_size]
    means = support.mean(dim=0)
    deviations = support.std(dim=0, correction=1)
    deviations = torch.where(deviations == 0, torch.ones_like(deviations), deviations)
    normalized = ((replaced - means) / (deviations + 1e-16)).clamp(-100, 100)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=100.0, neginf=-100.0)
    keep = retained.unsqueeze(0)
    values = torch.where(keep, normalized, torch.zeros_like(normalized)).unsqueeze(-1)
    codes = torch.where(keep, _nonfinite_codes(source), torch.zeros_like(source)).unsqueeze(-1)
    return torch.nan_to_num(F.linear(torch.cat((values, codes), dim=-1), enc.weight, enc.bias))


def label_encoder(enc, labels: torch.Tensor, support_size, *, regression: bool) -> torch.Tensor:
    """``LabelEncoder`` / ``RegressionLabelEncoder`` for one member: labels [rows] -> [rows, width].

    Classification ranks each label among the DISTINCT support labels (upstream: ``torch.unique`` then
    ``(replaced > unique).sum()``); the same count without ``Unique`` is
    sum_j [a_j < x] * [j is the first occurrence of its value] over the support labels a."""
    codes = _nonfinite_codes(labels)
    replacement = _finite_mean(labels[:support_size], dim=0)
    replaced = torch.where(torch.isfinite(labels), labels, replacement)
    if regression:
        first_channel = replaced
    else:
        a = replaced[:support_size]
        idx = torch.arange(a.shape[0], device=a.device)
        earlier = idx.view(1, -1) < idx.view(-1, 1)                      # [j, k]: k < j
        has_earlier_dup = ((a.unsqueeze(1) == a.unsqueeze(0)) & earlier).any(dim=-1)
        first = ~has_earlier_dup
        first_channel = ((replaced.unsqueeze(1) > a.unsqueeze(0)) & first.unsqueeze(0)).sum(dim=-1)
        first_channel = first_channel.to(labels.dtype)
    channels = torch.stack((first_channel, codes), dim=-1)
    return torch.nan_to_num(F.linear(channels, enc.weight, enc.bias))


def model_forward(model, x: torch.Tensor, y: torch.Tensor, support_size, *, regression: bool) -> torch.Tensor:
    """One member of ``ClassificationModel.forward`` / ``RegressionModel.forward`` (columns_per_group 1).

    x [1, T, H] (rows < S are the support), y [1, S] -> head output for the rows >= S [1, T - S, C]."""
    transformer = model.transformer
    rows = x.shape[1]
    feature_state = feature_encoder(model.feature_encoder, x[0].unsqueeze(-1), support_size).unsqueeze(0)
    placeholders = torch.full_like(x[0, :, 0], float("nan"))[support_size:]
    labels = torch.cat((y[0], placeholders), dim=0)
    label_state = label_encoder(model.label_encoder, labels, support_size, regression=regression)
    state = torch.cat((feature_state, label_state.unsqueeze(0).unsqueeze(2)), dim=2)

    batch, _, groups, _ = state.shape
    feature_summary = transformer.feature_summary_tokens.expand(batch, -1, groups, -1)
    item_summary = transformer.item_summary_tokens.expand(batch, rows, -1, -1)
    for layer in transformer.layers:
        state, feature_summary, item_summary = layer_forward(layer, state, feature_summary, item_summary,
                                                             support_size)
    query_embeddings = state[:, support_size:, -1, :]
    return transformer.classification_heads["standard"](query_embeddings)


class ExportWrapper(torch.nn.Module):
    """Pins EXAONE Tabular to the engine's 2-input ONNX signature.

    Inputs (B fixed to 1 -- one table per call):
      x  [1, T, H] float32   features (all rows; H is dynamic)
      y  [1, S]    float32   TRAINING labels only (S = len(y) <= T): dense class ids, or RAW targets
    Output:
      classification [1, T, C] class logits (C = the head's 10 classes)
      regression     [1, T, 1] a RAW-space point estimate: the mean of the 999-quantile head
    EVERY row is answered; rows < S are in-context fitted values (the context presented a second time
    as queries, as for LimiX / Nori / Causilo: queries cross-attend to the support only, so each query
    is independent of the others, and duplicating the support rows leaves the transductive
    constant-column test of the feature encoder unchanged).

    Regression target: upstream's EXAONETabularRegressor.fit standardises y in float32 with the
    population variance, treating a variance at rounding level as constant (scale 1); predictions are
    mapped back. The point estimate is the mean of the quantile bank, which is what the released
    regressor's default ``point_estimate="mean"`` reads out (README: "targets the conditional mean, which
    is what RMSE scores"); for one member upstream's across-member trimming is a no-op.
    """

    def __init__(self, model, task: str):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.m = model
        self.task = task

    def forward(self, x, y):
        s = y.shape[1]
        torch._check(s <= x.shape[1])
        torch._check(s >= 2)
        regression = self.task == "regression"
        if regression:
            n = _length(s, y)
            center = y.mean(dim=1, keepdim=True)
            variance = (y - center).square().mean(dim=1, keepdim=True)
            constant = variance <= n * FLOAT32_EPS * variance + (n * center * FLOAT32_EPS) ** 2
            scale = torch.where(constant, torch.ones_like(variance), variance.sqrt())
            y_in = (y - center) / scale
        else:
            y_in = y
        table = torch.cat([x[:, :s], x], dim=1)
        out = model_forward(self.m, table, y_in, s, regression=regression)
        if regression:
            return (out.mean(dim=-1) * scale + center).unsqueeze(-1)
        return out


def build_model(task: str, model_config: dict, regression_config: dict | None = None, seed: int = 0):
    """Random-weight EXAONE Tabular at the given config. No checkpoint bytes anywhere."""
    from exaonetabular.config import ModelConfig, RegressionConfig
    from exaonetabular.model.heads import ClassificationModel, RegressionModel

    torch.manual_seed(seed)
    cfg = ModelConfig(**model_config)
    if task == "regression":
        model = RegressionModel(cfg, RegressionConfig(**(regression_config or {})))
    else:
        model = ClassificationModel(cfg)
    return model.float().eval()


def upstream_forward(model, x: torch.Tensor, y: torch.Tensor, support_size: int) -> torch.Tensor:
    """Upstream's own model.forward for one member, float32 CPU (the reference every test compares to)."""
    xs, xq = x[:, :support_size].double().float(), x[:, support_size:]
    with torch.inference_mode():
        return model(xs, y, xq, feedforward_token_chunk=1 << 30, trusted_internal_inputs=True)
