"""TabICL v2 with the train/test split as a VALUE instead of a shape (the ROCm / MIGraphX contract).

The CPU graph (``tabicl_patches.ExportWrapper``) is ``(x, y)``-only: the split is ``len(y)``, a SHAPE, which a
fixed-shape MIGraphX compile cannot bucket (docs/ROCM_SINGLE_EVAL_POS.md). This module computes the same
function with every shape a function of the padded (T, H) bucket alone:

    x[1, T, H] f32   y[1, T] f32 (FULL length; off-context rows carry the -100 sentinel)
    train_size[1] i64   d[1] i64 (real feature count)   n_rows[1] i64 (declared for the contract, unused)
    -> logits[1, T, C]: class logits (C = max_classes) or a raw-space point estimate (C = 1) for every row

It runs upstream's own modules (``tabicl._model``, training-mode path, as the positional wrapper does) and
changes only what read a shape the plugin's padding inflates:

  1. feature grouping. Upstream groups column ``j`` with ``(j + 2**i) % H`` -- a circular permutation of the
     REAL width. Padded columns would change every group's partners, so the index is taken modulo ``d`` (the
     value), division-free (``j + 2**i < d + 2**(size-1)``, so a bounded number of conditional subtractions;
     MIGraphX miscompiles a runtime integer floor-division and modulo, docs/ROCM_TABPFN_PLAN.md) and gathered.
     A padded group (index >= d) gathers its own zero column: finite, and masked out in step 4;
  2. the target-aware column embedding adds the label embedding to the CONTEXT rows (row < S) only;
  3. the column set-transformer's inducing points attend to the context rows only (a key mask, not the
     ``src[..., :train_size]`` slice), and its QASSMax reads log(S) from the value;
  4. the row interactor masks padded feature groups out as keys (its own ``d`` mask, upstream's code path);
  5. the ICL transformer: every row attends to the context rows only (key mask), the label embedding is
     added to the context rows only, and QASSMax reads log(S);
  6. regression: the target standardisation reads the context rows only.

Two ops MIGraphX's ONNX parser cannot take are rewritten while the masked forward traces (``migraphx_safe``),
each to the same function: ``F._in_projection_packed`` splits the packed q/k/v weight with ``Tensor.split``
whenever the keys are not the queries (the column ISAB and the row interactor's last block), which exports as
SplitToSequence / SequenceAt (slices instead, docs/ROCM_TABDPT_SPIKE.md); and RoPE's ``einsum("..., f -> ...
f")`` is an outer product (a broadcast multiply instead; no committed MIGraphX graph uses Einsum). QASSMax's
``base_mlp`` on the scalar log(S) is written out elementwise (the Nori and Causilo graphs failed on it).

Nothing in TabICL reads statistics over every row, so padded rows cannot reach a statistic and ``n_rows``
stays unused (it is declared because the plugin binds it when a graph has it).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

NEG = -1e4

#: log(S) for QASSMax while a masked forward runs. Upstream computes log(src_len) of the key tensor, which
#: under the mask is the PADDED length; every attention that carries an ssmax layer in TabICL v2 (the column
#: ISAB's first stage and the ICL transformer) has the context as its keys, so one value covers both.
_SSMAX_LOGN: torch.Tensor | None = None


@contextlib.contextmanager
def ssmax_length(log_s: torch.Tensor):
    global _SSMAX_LOGN
    previous, _SSMAX_LOGN = _SSMAX_LOGN, log_s
    try:
        yield
    finally:
        _SSMAX_LOGN = previous


def install() -> None:
    """Route ``ssmax._logn`` through the override (after ``tabicl_patches.apply``, which made it dynamic)."""
    import tabicl._model.ssmax as ssmax

    if getattr(ssmax._logn, "_masked", False):
        return
    positional = ssmax._logn

    def _logn(n, device, dtype):
        if _SSMAX_LOGN is not None:
            # [1], not a 0-d tensor: MIGraphX's parser rejects a Reshape to a scalar shape ("reshape has 0
            # elements whereas the input has 1"); every caller broadcasts or reshapes it to (1, 1) anyway
            return _SSMAX_LOGN.to(dtype)
        return positional(n, device, dtype)

    _logn._masked = True
    ssmax._logn = _logn


class _FunctionalShim:
    """``torch.nn.functional`` as ``tabicl._model.attention`` sees it, with one function swapped."""

    def __init__(self, real, **overrides):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_real"), name)


def _in_projection_sliced(q, k, v, w, b=None):
    """``F._in_projection_packed`` with the packed weight SLICED into q / k / v (no ``Tensor.split``): every
    output element is the same dot product, so the result is identical for self- and cross-attention."""
    e = q.size(-1)
    bq, bk, bv = (None, None, None) if b is None else (b[:e], b[e:2 * e], b[2 * e:])
    return F.linear(q, w[:e], bq), F.linear(k, w[e:2 * e], bk), F.linear(v, w[2 * e:], bv)


def _outer_einsum(equation, *operands):
    if equation.replace(" ", "") == "...,f->...f" and len(operands) == 2:
        a, f = operands
        return a.unsqueeze(-1) * f
    return torch.einsum(equation, *operands)


def _qassmax_forward(self, q, n):
    """``QASSMaxMLP.forward`` with ``base_mlp`` (Linear(1, 64) -> GELU -> Linear) written out on the scalar
    log(n): the module applied to a [1, 1] tensor made MIGraphX's GPU pipeline fail on the Nori and Causilo
    graphs (simplify_reshapes, "cannot create std::vector larger than max_size()"). Same function."""
    import tabicl._model.ssmax as ssmax

    logn = ssmax._logn(n, q.device, q.dtype).reshape(-1)[:1]                          # [1]
    l1, act, l2 = self.base_mlp[0], self.base_mlp[1], self.base_mlp[2]
    hidden = act(l1.weight.view(-1) * logn + l1.bias)                                 # [64]
    base = hidden.unsqueeze(0) @ l2.weight.transpose(0, 1) + l2.bias                  # [1, out]
    if self.elementwise:
        base_scales = base.view(1, self.num_heads, 1, self.head_dim)
    else:
        base_scales = base.view(1, self.num_heads, 1, 1)
    modulation = 1 + torch.tanh(self.query_mlp(q))
    return q * (base_scales * modulation)


@contextlib.contextmanager
def migraphx_safe():
    import tabicl._model.attention as attention
    import tabicl._model.rope as rope
    import tabicl._model.ssmax as ssmax

    real_f, real_einsum, real_qass = attention.F, rope.einsum, ssmax.QASSMaxMLP.forward
    attention.F = _FunctionalShim(real_f, _in_projection_packed=_in_projection_sliced)
    rope.einsum = _outer_einsum
    ssmax.QASSMaxMLP.forward = _qassmax_forward
    try:
        yield
    finally:
        attention.F, rope.einsum, ssmax.QASSMaxMLP.forward = real_f, real_einsum, real_qass


def _sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum over the first axis as a GEMM (ones[1, R] @ x[R, rest]) -- chained masked reductions crash
    MIGraphX's reducer codegen (docs/ROCM_TABDPT_SPIKE.md)."""
    rows = x.shape[0]
    flat = x.reshape(rows, -1)
    ones = torch.ones(1, rows, dtype=flat.dtype, device=flat.device)
    return (ones @ flat).reshape(1, *x.shape[1:])


def group_indices(width: int, d: torch.Tensor, size: int, device=None) -> list[torch.Tensor]:
    """For each grouping offset 2**i: the column each group ``j`` reads, ``(j + 2**i) mod d`` for ``j < d``
    and ``j`` itself for a padded group. Division-free."""
    idx = torch.arange(width, dtype=torch.long, device=device)
    real = idx < d
    out = []
    for i in range(size):
        k = idx + 2 ** i
        # k < d + 2**i; each pass subtracts d once where k >= d, so 2**i passes reach k mod d for any d >= 1
        for _ in range(2 ** i):
            k = torch.where(k >= d, k - d, k)
        out.append(torch.where(real, k, idx))
    return out


def feature_grouping(ce, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``ColEmbedding.feature_grouping`` ("same" mode) over the real width d: [1, T, H] -> [1, T, H, size]."""
    if ce.feature_group is not True and ce.feature_group != "same":
        raise ValueError("the masked grouping covers the released config only (feature_group='same')")
    cols = group_indices(x.shape[2], d, ce.feature_group_size, device=x.device)
    return torch.stack([x.index_select(2, c) for c in cols], dim=-1)


def _linear1(linear, v):
    """``nn.Linear(1, n)`` on a trailing size-1 axis, written elementwise: a matmul with contraction dimension 1
    feeding a broadcast Mul kills MIGraphX's simplify_reshapes (tools/export_causilo, docs/ROCM_CAUSILO_PLAN.md).
    v [...] -> [..., n]; the same function."""
    out = v.unsqueeze(-1) * linear.weight.view(-1)
    return out + linear.bias if linear.bias is not None else out


def _isab(block, src, key_bias):
    """InducedSelfAttentionBlock (patched branchless forward) with the context as the inducing points' keys.
    src [1, G, T, E], key_bias [T] (0 on context rows, NEG elsewhere)."""
    *batch, rows, e = src.shape
    ind = block.ind_vectors.expand(*batch, block.num_inds, e)
    mask = key_bias.view(*([1] * len(batch)), rows).expand(*batch, rows)
    hidden = block.multihead_attn1(ind, src, src, key_padding_mask=mask)
    out = block.multihead_attn2(src, hidden, hidden)
    skip = (src == block.skip_value).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
    return torch.where(skip, torch.full_like(out, block.skip_value), out)


def column_embedding(ce, x, y_lab, ctx, key_bias, d):
    """ColEmbedding._train_forward_with_feature_group with the split as a value -> [1, T, G+C, E]."""
    xg = feature_grouping(ce, x, d)                                                   # [1, T, H, size]
    if ce.reserve_cls_tokens > 0:
        xg = F.pad(xg, (0, 0, ce.reserve_cls_tokens, 0), value=-100.0)
    features = xg.transpose(1, 2)                                                     # [1, G+C, T, size]
    src = ce.in_linear(features)
    if ce.target_aware:
        y_exp = y_lab.unsqueeze(1).expand(-1, features.shape[1], -1)                  # [1, G+C, T]
        y_emb = ce.y_encoder(y_exp.float()) if ce.max_classes > 0 else _linear1(ce.y_encoder, y_exp)
        src = src + torch.where(ctx.view(1, 1, -1, 1), y_emb, torch.zeros_like(y_emb))
    for block in ce.tf_col.blocks:
        src = _isab(block, src, key_bias)
    if ce.affine:
        src = features * ce.ln_w(ce.out_w(src)) + ce.ln_b(ce.out_b(src))
    return src.transpose(1, 2)                                                        # [1, T, G+C, E]


def row_interaction(ri, emb, d):
    """RowInteraction._train_forward with the d mask as an additive float key mask (upstream builds the same
    mask as a bool and turns it into -inf; a finite NEG keeps the softmax away from inf arithmetic)."""
    b, t, hc, e = emb.shape
    emb = torch.cat((ri.cls_tokens.expand(b, t, ri.num_cls, e), emb[:, :, ri.num_cls:]), dim=2)
    pos = torch.arange(hc, device=emb.device)
    key_mask = torch.where(pos < d + ri.num_cls, 0.0, NEG).to(emb.dtype).view(1, 1, hc).expand(b, t, hc)
    return ri._aggregate_embeddings(emb, key_mask)


def icl(icl_mod, reps, y_lab, ctx, key_bias):
    """ICLearning._icl_predictions with the context as every row's keys -> [1, T, out_dim]."""
    if icl_mod.max_classes > 0:
        ry = icl_mod.y_encoder(y_lab.float())
    else:
        ry = _linear1(icl_mod.y_encoder, y_lab)
    src = reps + torch.where(ctx.view(1, -1, 1), ry, torch.zeros_like(ry))
    mask = key_bias.view(1, -1)
    for block in icl_mod.tf_icl.blocks:
        src = block(q=src, key_padding_mask=mask, rope=icl_mod.tf_icl.rope)
    if icl_mod.norm_first:
        src = icl_mod.ln(src)
    return icl_mod.decoder(src)


def masked_forward(model, x, y_lab, train_size, d):
    """x [1, T, H] (padded), y_lab [1, T] (labels on the context rows; any finite value elsewhere) ->
    the positional wrapper's model output for every row."""
    rows = x.shape[1]
    ctx = torch.arange(rows, device=x.device) < train_size                            # [T]
    key_bias = torch.where(ctx, 0.0, NEG).to(x.dtype)
    log_s = torch.log(train_size.to(x.dtype).clamp_min(1.0))
    with ssmax_length(log_s), migraphx_safe():
        emb = column_embedding(model.col_embedder, x, y_lab, ctx, key_bias, d)
        reps = row_interaction(model.row_interactor, emb, d)
        return icl(model.icl_predictor, reps, y_lab, ctx, key_bias)


class MaskExportWrapper(torch.nn.Module):
    """TabICL under the train_size contract: x, y (full length), train_size, d, n_rows (see the module doc)."""

    def __init__(self, model):
        super().__init__()
        install()
        self.m = model
        self.regression = model.max_classes == 0

    def forward(self, x, y, train_size, d, n_rows):
        rows = x.shape[1]
        ctx = (torch.arange(rows, device=x.device) < train_size).unsqueeze(0)          # [1, T]
        zero = torch.zeros_like(y)
        # off-context labels are the -100 sentinel (or anything): never read, but one_hot of a negative id is an
        # error and a non-finite value would poison a masked key through 0 * inf
        y = torch.where(ctx, torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), zero)
        if not self.regression:
            return masked_forward(self.m, x, y, train_size, d)
        # StandardScaler(ddof=0) over the context, 0 -> 1 guard: the positional wrapper's, as masked sums.
        n = _sum_rows(ctx.to(y.dtype).transpose(0, 1))[0]                              # [1] == S
        mean_y = _sum_rows(y.transpose(0, 1))[0] / n
        dev = torch.where(ctx, y - mean_y, zero)
        std_y = torch.sqrt(_sum_rows((dev * dev).transpose(0, 1))[0] / n)
        std_y = torch.where(std_y > 0.0, std_y, torch.ones_like(std_y))
        y_z = torch.where(ctx, (y - mean_y) / std_y, zero)
        quantiles = masked_forward(self.m, x, y_z, train_size, d)                      # [1, T, Q]
        return quantiles.mean(dim=-1, keepdim=True) * std_y + mean_y
