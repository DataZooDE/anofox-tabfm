"""The masked whole-model forward for Causilo (docs/ROCM_CAUSILO_PLAN.md has the inventory).

Everything in the forward that consumes the train/test split, the real row count or the real feature
count is rewritten so those arrive as VALUES and every shape is the padded bucket:

    x          [1, T, H]  real data in [:n_rows, :d], zeros elsewhere
    y          [1, T]     train targets in [:train_size], -100 elsewhere
    train_size            rows [0, train_size) are context
    d                     columns [0, d) are real

Why a mask and not just a longer graph: a feature GROUP of 3 columns is one token, and a bucket wider than
the real feature count adds all-zero groups that the row stages would read as real tokens (measured: error
1.1 on logits of scale 1.5 with one extra group). A zero column INSIDE the last real group is inert (upstream
pads with numeric zero too), so only whole groups are masked, as keys.

Nothing upstream is monkeypatched. Each block is re-driven from its own weights (`layer.attention`,
`layer.attention_norm`, ...), so this module never depends on whether `causilo_patches.apply()` ran in the
process, which is the trap the TabPFN 2.5 path fell into.

MIGraphX's GPU target is the reason for several unusual spellings below. All four failures were found on the
TabPFN conversion and none is visible to ONNX Runtime or to MIGraphX's reference target:

  * no integer division or modulo on runtime data: real groups are `arange(G) * size < d`, counted with `sum`;
  * no Reshape to an empty shape: scalars are indexed with `[0]`, never `.reshape(())`;
  * no Concat whose inputs are all Transposes (`simplify_reshapes` asserts);
  * no matmul with contraction dimension 1: `Linear(1, n)` is written elementwise (`_linear_k1`). Causilo has
    it twice in every `QueryScale` and once more in the regression target embedding.

The small module-level functions named `_row_keep`, `_group_keep`, `_n_keys_rows`, `_n_keys_groups`,
`_stat_rows` and `_inject_keep` are seams, not abstractions: tools/mask_parity_full.py swaps each for a wrong
version to prove the gate would notice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from causilo.nn.cache import KVPair
from causilo.nn.layers.scaling import bounded_multiplier

#: Additive key bias for an excluded key. Large enough that exp() underflows to exactly 0 in float32, small
#: enough to stay finite (an actual -inf makes a fully-masked row NaN).
NEG_INF = -1e4

#: float32 machine epsilon; the graph computes in float32 where upstream's normaliser uses float64.
_EPS32 = 1.1920929e-07


# --- seams ----------------------------------------------------------------------------------------------

def _row_keep(train):
    """[R] bool: the rows that may be KEYS of a row-axis attention (the column gather, every prediction layer)."""
    return train


def _group_keep(G, d, size, device):
    """[G] bool: feature groups holding at least one real column. `j * size < d`, not `j < ceil(d / size)`."""
    return torch.arange(G, device=device) * size < d


def _n_keys_rows(train):
    """0-d float: the REAL number of training rows, which `QueryScale` takes the log of."""
    return train.sum().to(torch.float32)


def _n_keys_groups(group_keep, latents):
    """0-d float: keys of a row-stage `LatentUpdate` = the latents plus the REAL groups."""
    return group_keep.sum().to(torch.float32) + latents


def _stat_rows(train):
    """[R] bool: the rows the normaliser's statistics may read (upstream fits on the training rows only)."""
    return train


def _inject_keep(train):
    """[R] bool: the rows that get their embedded target added."""
    return train


def _twice(t):
    """`[t ; t]` along the row axis (axis 1) WITHOUT a Concat: unsqueeze, expand, reshape.

    The values are identical to `torch.cat([t, t], dim=1)`. The Concat spelling puts a node into the graph whose
    two inputs are the SAME tensor, and the real masked graph fails MIGraphX's GPU pipeline in `fuse_concat`
    (`program.cpp:1211 create_module: Assertion 'not contains(impl->modules, name)'`). See
    docs/ROCM_CAUSILO_PLAN.md for how far that is established.
    """
    return t.unsqueeze(1).expand(t.shape[0], 2, *t.shape[1:]).reshape(t.shape[0], -1, *t.shape[2:])


def _twice_concat(t):
    """The original spelling, kept only so a test can show what the graph looked like (and that the guard sees it)."""
    return torch.cat([t, t], dim=1)


# --- small pieces ---------------------------------------------------------------------------------------

def _bias(keep, like):
    """[1,1,1,K] additive key bias: 0 on real keys, NEG_INF on excluded ones."""
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    neg = torch.full((), NEG_INF, dtype=like.dtype, device=like.device)
    return torch.where(keep, zero, neg).reshape(1, 1, 1, -1)


def _linear_k1(lin, x):
    """`nn.Linear(1, n)` as an elementwise multiply: x[..., 1] -> [..., n].

    The layer's value, but without a matmul whose contraction dimension is 1. Exported as a matmul feeding
    a broadcast Mul it kills MIGraphX's GPU compile in `simplify_reshapes`; ORT and the reference target are
    fine, so only the GPU sees it (tests/test_mask_export_isolated.py).
    """
    y = x * lin.weight.reshape(-1)
    return y if lin.bias is None else y + lin.bias


def _sequence_k1(seq, x):
    """A `Sequential(Linear(1, n), act, Linear(n, m))` whose first layer is written elementwise."""
    first, *rest = list(seq)
    x = _linear_k1(first, x)
    for module in rest:
        x = module(x)
    return x


def _log_count(n, like):
    """log(max(n, 1)) as a [1, 1] tensor. `n` is a 0-d tensor (a VALUE) or a Python int (an architectural constant)."""
    length = like.new_ones(n).sum() if isinstance(n, int) else n.to(like.dtype)
    return torch.log(torch.clamp(length, min=1.0)).reshape(1, 1)


def _query_scale(scaling, query, n):
    """`causilo.nn.layers.scaling.QueryScale.forward`, with the key count a value and no K=1 matmul."""
    log_count = _log_count(n, query)
    length_gain = bounded_multiplier(_sequence_k1(scaling.length, log_count), 8.0)
    length_gain = length_gain * bounded_multiplier(_sequence_k1(scaling.correction, log_count), 2.0)
    content_gain = bounded_multiplier(scaling.content(query), 2.0)
    return query * (length_gain.reshape(scaling.heads, 1, scaling.head_width) * content_gain)


def _attend(att, query, context, keep, n):
    """`Attention.attend` with the keys optionally masked and the key count `n` given as a value.

    query is [*batch, heads, Q, head_width]; context is a KVPair over [*batch, heads, K, head_width]; `keep`
    is a [K] bool or None. Leading axes are flattened into one for SDPA exactly as upstream does.
    """
    if att.scaling is not None:
        query = _query_scale(att.scaling, query, n)
    leading = query.shape[:-3]
    q = query.reshape(-1, *query.shape[-3:])
    k = context.key.reshape(-1, *context.key.shape[-3:])
    v = context.value.reshape(-1, *context.value.shape[-3:])
    mask = None if keep is None else _bias(keep, q)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    merged = out.transpose(-3, -2).flatten(-2).reshape(*leading, query.shape[-2], att.width)
    return att.output(merged)


def _prepare_context(att, rows):
    """`Attention.prepare_context` without `chunk`: the K and V halves of the packed projection as two slices.

    Upstream splits the packed projection with `packed.chunk(2, dim=-1)`, which exports as ONNX
    `SplitToSequence` + `SequenceAt`, and MIGraphX's ONNX parser has no such operator ("Unknown operator:
    SplitToSequence": the masked graph cannot even be LOADED on ROCm). Slices are exactly equivalent and
    supported. Head splitting and the optional per-head RMS normalisation are upstream's own `_keys`.
    """
    packed = F.linear(rows, att.projection.weight[att.width:], att.projection.bias[att.width:])
    return att._keys(packed[..., :att.width], packed[..., att.width:])


def _layer_prepare(layer, rows):
    """`AttentionLayer.prepare`: normalise, then project to K/V."""
    return _prepare_context(layer.attention, layer.attention_norm(rows))


def _layer_query(layer, rows, context, keep, n):
    """`AttentionLayer.query`: the full residual layer against prepared K/V, keys masked."""
    projected = layer.attention.prepare_query(layer.attention_norm(rows))
    return layer._finish(rows, _attend(layer.attention, projected, context, keep, n))


def _target_embed(emb, targets):
    """`TargetEmbedding.forward` without `F.one_hot` (ONNX OneHot is not in the MLX op table) and without a
    K=1 matmul for the regression scalar."""
    if emb.classes:
        ids = torch.arange(emb.classes, device=targets.device)
        return emb.projection((targets.long().unsqueeze(-1) == ids).float())
    return _linear_k1(emb.projection, targets.unsqueeze(-1))


def _inject(rows, embedded, keep):
    """`inject_targets` with the prefix as a mask: rows + embedded target, only where `keep`."""
    k = keep.reshape([1, -1] + [1] * (rows.ndim - 2))
    if rows.ndim == 4:
        embedded = embedded.unsqueeze(-2)
    return rows + torch.where(k, embedded, torch.zeros_like(embedded))


# --- the normaliser -------------------------------------------------------------------------------------

def _row_sum(v):
    """`v.sum(1, keepdim=True)` for [1,T,W], written as `ones[1,1,T] @ v` -> [1,1,W].

    Why a matmul: with the statistics written as reductions, MIGraphX's GPU pipeline fuses the chain
    reduce -> pointwise -> reduce (mean, then the variance around it) into one `fused_reduce` kernel that does not
    compile for gfx1201 (`invalid operands to binary expression ('float' and ...inner_storage<float, 1, ...>)`, a
    16-line reproducer is in docs/ROCM_CAUSILO_PLAN.md). A matmul is not a reduction to the fuser, so it breaks the
    chain. The reciprocal form (`* (1/count)`) and a scalar count did NOT help: the chain is the trigger, not the
    division. Same values up to summation order.
    """
    return torch.ones(1, 1, v.shape[1], dtype=v.dtype, device=v.device) @ v


def normalise_masked(x, train):
    """`Normalizer.fit(train rows, "none").transform(x)` in-graph, the train rows being a MASK. [1,T,H] -> [1,T,H].

    Same arithmetic as `causilo_patches.normalise_features` (validated against upstream's numpy normaliser
    there); every sum over "the first S rows" becomes a sum over rows where the mask is true, and every
    count is that sum rather than a shape. Missing (NaN) cells are mean-filled for the statistics and
    restored to NaN afterwards.
    """
    sm = _stat_rows(train).reshape(1, -1, 1)
    zero = torch.zeros_like(x[:, :1])
    missing = torch.isnan(x)

    observed = (~missing) & sm
    count = _row_sum(observed.to(x.dtype))
    fill = torch.where(count > 0,
                       _row_sum(torch.where(observed, x, torch.zeros_like(x))) / count.clamp(min=1.0),
                       zero)
    filled = torch.where(missing, fill, x)

    n_rows = _row_sum(sm.to(x.dtype))                      # number of statistics rows
    center = _row_sum(torch.where(sm, filled, torch.zeros_like(x))) / n_rows
    spread = (_row_sum(torch.where(sm, (filled - center) ** 2, torch.zeros_like(x))) / n_rows).sqrt()
    tolerance = 8.0 * _EPS32 * torch.clamp(center.abs(), min=1.0)
    spread = torch.where(spread <= tolerance, torch.ones_like(spread), spread)
    scaled = (filled - center) / spread

    # outlier_bounds: two passes, ddof = 1 when there is more than one statistics row
    correction = (n_rows > 1).to(x.dtype)
    mean = _row_sum(torch.where(sm, scaled, torch.zeros_like(x))) / n_rows
    preliminary = torch.clamp(
        (_row_sum(torch.where(sm, (scaled - mean) ** 2, torch.zeros_like(x)))
         / (n_rows - correction)).sqrt(), min=1e-6)
    kept = (~((scaled - mean).abs() > 4.0 * preliminary)) & sm
    kept_count = _row_sum(kept.to(x.dtype))
    centre2 = torch.where(kept_count > 0,
                          _row_sum(torch.where(kept, scaled, torch.zeros_like(x)))
                          / kept_count.clamp(min=1.0),
                          zero)
    dof = kept_count - correction
    variance = _row_sum(torch.where(kept, (scaled - centre2) ** 2, torch.zeros_like(x))) / dof.clamp(min=1.0)
    spread2 = torch.clamp(torch.where(dof > 0, variance.sqrt(), torch.ones_like(variance)), min=1e-6)
    lower, upper = centre2 - 4.0 * spread2, centre2 + 4.0 * spread2

    asinh = lambda z: torch.log(z + torch.sqrt(z * z + 1.0))              # noqa: E731 (Asinh is not in the MLX table)
    values = torch.where(scaled < lower, lower - asinh(lower - scaled), scaled)
    values = torch.where(values > upper, upper + asinh(values - upper), values)
    return torch.where(missing, torch.full_like(values, float("nan")), values)


# --- the stages -----------------------------------------------------------------------------------------

def _column_block(block, rows_BGRD, keep, n):
    """`ColumnBlock.forward` with the context rows a key MASK. rows is [B, G, R, D]."""
    for layer in block.layers:
        latents = layer.latents.expand(*rows_BGRD.shape[:-2], *layer.latents.shape)
        gather = layer.gather
        projected = gather.attention.prepare_query(gather.attention_norm(latents))
        context = _prepare_context(gather.attention, gather.attention_norm(rows_BGRD))
        compressed = gather._finish(latents, _attend(gather.attention, projected, context, keep, n))
        # the broadcast attention has no QueryScale and its keys are the fixed latents: unchanged upstream code
        rows_BGRD = layer.broadcast.query(rows_BGRD, _layer_prepare(layer.broadcast, compressed))
    return rows_BGRD


def _feature_update(fu, rows, latents):
    """`FeatureUpdate.forward`: keys are the L latents, an architectural constant, so no mask."""
    projected = fu.read.prepare_query(fu.context_norm(rows))
    context = _prepare_context(fu.read, fu.context_norm(latents))
    return rows + _attend(fu.read, projected, context, None, latents.shape[-2])


def _latent_update(lu, latents, features, group_keep, n_keys):
    """`LatentUpdate.forward`: the latents read themselves AND every feature group; padded groups are masked."""
    count = latents.shape[-2]
    context = lu.context_norm(torch.cat((latents, features), dim=-2))
    projected = lu.read.prepare_query(context[..., :count, :])
    kv = _prepare_context(lu.read, context)
    keep = torch.cat([torch.ones(count, dtype=torch.bool, device=group_keep.device), group_keep])
    latents = latents + _attend(lu.read, projected, kv, keep, n_keys)
    return latents + lu.feedforward(lu.output_norm(latents))


def _row_block(block, features, group_keep, n_keys):
    """`RowBlock.forward`. features is [B, R, G, D]."""
    latents = block.latents.to(features).expand(*features.shape[:-2], *block.latents.shape)
    for layer in block.rounds:
        count = latents.shape[-2]
        combined = _feature_update(layer.features, torch.cat((latents, features), dim=-2), latents)
        latents, features = combined[..., :count, :], combined[..., count:, :]
        latents = _latent_update(layer.update_latents, latents, features, group_keep, n_keys)
    return _feature_update(block.finish, features, latents)


def _row_pool(pool, features, group_keep, n_keys):
    """`RowPool.forward`: [B, R, G, D] -> [B, R, L * D]."""
    latents = pool.latents.to(features).expand(*features.shape[:-2], *pool.latents.shape)
    for update in pool.rounds:
        latents = _latent_update(update, latents, features, group_keep, n_keys)
    return pool.output_norm(latents).flatten(-2)


def _head(head, rows):
    """`Head.forward` without the `torch.autocast(enabled=False)` context (a no-op on CPU, an HOP for dynamo)."""
    return head.output(F.gelu(head.hidden(head.normalize(rows.float()))))


def _predict_rows(model, table, targets, train, d):
    """`ModelRunner.predict` for EVERY row of `table` [1, R, H]; rows where `train` is true are the context.

    Returns [1, R, outputs]. Rows that are not context are queries; rows that are not real are garbage no
    one reads. The answer for the context rows is meaningful only through the duplicate route in
    `masked_forward`.
    """
    R = table.shape[1]
    row_keep = _row_keep(train)
    n_rows = _n_keys_rows(train)
    y_use = torch.where(train, targets, torch.zeros_like(targets))
    inject = _inject_keep(train)

    features = model.feature_embedding(table)                                   # [1, R, G, D]
    G = features.shape[-2]
    group_keep = _group_keep(G, d, model.config.group_size, table.device)
    n_groups = _n_keys_groups(group_keep, model.config.row_latents)

    features = _inject(features, _target_embed(model.feature_target, y_use), inject)
    features = _column_block(model.columns[0], features.transpose(1, 2), row_keep, n_rows).transpose(1, 2)
    features = _row_block(model.row, features, group_keep, n_groups)
    features = _column_block(model.columns[1], features.transpose(1, 2), row_keep, n_rows).transpose(1, 2)
    rows = _row_pool(model.pool, features, group_keep, n_groups)                # [1, R, L*D]
    rows = _inject(rows, _target_embed(model.row_target, y_use), inject)
    for layer in model.prediction.layers:
        rows = _layer_query(layer, rows, _layer_prepare(layer, rows), row_keep, n_rows)
    return _head(model.head, rows)


def masked_forward(model, x, y, train_size, d):
    """[1, T, C] for every row of the padded input: in-context values for rows < train_size, predictions after.

    `x` is [1, T, H] raw (un-normalised) features, `y` is [1, T] with RAW targets in [:train_size];
    `train_size` and `d` are 0-d integer tensors. The context rows are presented a second time as queries
    (`[x ; x]`, only the first copy's first `train_size` rows being context) and the answer is read from the
    second copy: the route that measured right on real weights where running the last prediction layer over
    all rows measured fitted R2 0.50 against a query R2 of 0.999 (see causilo_patches.py).
    """
    T = x.shape[1]
    regression = model.config.task == "regression"
    train = torch.arange(T, device=x.device) < train_size
    y = torch.where(train.reshape(1, -1), y, torch.zeros_like(y))

    table = normalise_masked(x, train)
    if regression:
        # sklearn StandardScaler on the RAW training targets (population std; a scale at rounding level -> 1)
        n_train = train.to(y.dtype).sum()
        mean_y = torch.where(train.reshape(1, -1), y, torch.zeros_like(y)).sum() / n_train
        var_y = torch.where(train.reshape(1, -1), (y - mean_y) ** 2, torch.zeros_like(y)).sum() / n_train
        std_y = var_y.sqrt()
        std_y = torch.where(std_y < 10.0 * _EPS32, torch.ones_like(std_y), std_y)
        y = torch.where(train.reshape(1, -1), (y - mean_y) / std_y, torch.zeros_like(y))

    two_t = torch.arange(2 * T, device=x.device) < train_size
    out = _predict_rows(model, _twice(table), _twice(y), two_t, d)[:, T:]
    if regression:
        return out.mean(dim=-1, keepdim=True) * std_y + mean_y
    return out
