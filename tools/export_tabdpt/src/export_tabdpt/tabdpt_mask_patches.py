"""TabDPT with the train/test split as a VALUE instead of a shape.

Why this exists
---------------
TabDPT is in the engine's ``single_eval_pos`` family: the graph declares only
``x`` and ``y``, and the split is positional — ``eval_pos = y.shape[0]``, rows
``[0, S)`` are context and ``[S, T)`` are queries. That makes the train size a
SHAPE, which is exactly what a fixed-shape MIGraphX compile cannot bucket: a
different context length is a different compiled program, so the ``(T, H)``
bucket table would need a third, unbounded axis. Hence nine of eleven catalog
models, TabDPT among them, have no ROCm graph (docs/ROCM_SINGLE_EVAL_POS.md).

This module rewrites the same computation with the split carried as a runtime
value — ``x[1,T,H]``, ``y[1,T]``, ``train_size[1]``, ``d[1]`` — the contract
tabfm-v1 and mitra already use and which the engine has served since day one.
Every tensor's shape then depends only on ``(T, H)``, so the existing buckets
work unchanged.

What it is NOT
--------------
Not an approximation. Every change below is an algebraic rewrite of a slice
into a masked reduction over the same elements, so the masked model must agree
with the positional one to floating-point noise on identical weights and
inputs. ``parity.py`` asserts exactly that, and it runs before any ONNX is
produced — an unverified conversion that yields plausible-looking numbers is
this repository's worst documented failure mode.

The four places the split hides
-------------------------------
The published analysis mentions the attention slice. There are four, and
missing any one silently changes the answer:

1. ``clip_outliers`` / ``normalize_data`` standardise features over the
   CONTEXT ROWS ONLY (``data[:eval_pos]``). Both already delegate to
   ``maskmean``/``maskstd``, which take a mask — so restricting to context
   becomes an extra AND on that mask, and is exact.
2. The attention projects K/V from context rows only (``h[:, :eval_pos]``) and
   sizes them by ``eval_pos``. This becomes K/V over all ``T`` rows plus an
   additive ``-inf`` bias on non-context keys.
3. ``get_scale_param(eval_pos)`` derives a scalar from the context length,
   which becomes a tensor read of ``train_size``.
4. The head slices query rows (``src[eval_pos + n_think:]``). It now runs over
   every row and the engine slices, which it already does for the other family.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NEG_INF = -1e30  # finite so bf16/fp16 quantisation cannot turn it into a NaN


def context_row_mask(total_rows: int, train_size: torch.Tensor, device) -> torch.Tensor:
    """[T] bool: True on context rows. The whole conversion in one line."""
    positions = torch.arange(total_rows, device=device)
    return positions < train_size.reshape(())


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def _masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # Transcribed from upstream maskstd, including its (num - 1) denominator:
    # the unbiased estimator, which is what torch's .std() uses by default.
    num = mask.sum(dim=dim, keepdim=True)
    mean = _masked_mean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, torch.zeros((), dtype=x.dtype, device=x.device))
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def _row_mask_for(data: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
    """Broadcast a [T] context mask over a (T, B, H) feature tensor."""
    return ctx_mask.reshape(-1, *([1] * (data.dim() - 1))).expand_as(data)


def masked_clip_outliers(data: torch.Tensor, ctx_mask: torch.Tensor, n_sigma: float) -> torch.Tensor:
    """upstream clip_outliers, restricted to context rows by mask not slice."""
    rows = _row_mask_for(data, ctx_mask)
    mask = (~torch.isnan(data)) & rows
    mean = _masked_mean(data, mask, dim=0)
    cutoff = n_sigma * _masked_std(data, mask, dim=0)
    mask = mask & (cutoff >= torch.abs(data - mean))
    cutoff = n_sigma * _masked_std(data, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def masked_normalize_data(data: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
    """upstream normalize_data, restricted to context rows by mask not slice."""
    rows = _row_mask_for(data, ctx_mask)
    mask = (~torch.isnan(data)) & rows
    mean = _masked_mean(data, mask, dim=0)
    std = _masked_std(data, mask, dim=0) + 1e-6  # ADDITIVE, as upstream
    return (data - mean) / std


def masked_encoder_layer_forward(layer, x: torch.Tensor, y: torch.Tensor,
                                 ctx_mask: torch.Tensor, train_size: torch.Tensor) -> torch.Tensor:
    """TransformerEncoderLayer.forward with attention masked, not sliced.

    Upstream (and the positional export patch) build K and V from
    ``h[:, :eval_pos]`` and then ``view(B, eval_pos, ...)``, so the context
    length is a dimension. Here K/V cover all ``T`` rows and non-context keys
    are removed by an additive bias, which leaves every shape a function of the
    bucket alone.

    The ``q * beta`` rearrangement is kept from the positional patch: upstream
    passes ``scale=default_scale * beta`` to SDPA, which the exporter cannot
    trace as a dynamic scalar.
    """
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)
    B, L, _ = x.size()

    h = layer.attn_norm(x)
    q = layer.q_proj(h)
    gate = torch.sigmoid(layer.q_gate(q))

    # K/V over EVERY row; the mask decides which are real.
    k = layer.k_proj(h)
    v_in = torch.cat([h, y], dim=-1)
    v = layer.v_proj(v_in)

    q = q.view(B, L, layer.num_heads, layer.head_dim).transpose(1, 2)
    k = k.view(B, L, layer.num_heads, layer.head_dim).transpose(1, 2)
    v = v.view(B, L, layer.num_heads, layer.head_dim).transpose(1, 2)

    q, k = layer.q_norm(q), layer.k_norm(k)
    beta = masked_get_scale_param(layer, train_size, device=q.device, dtype=q.dtype)
    default_scale = 1.0 / math.sqrt(layer.head_dim)

    # [1, 1, 1, L] additive bias: 0 on context keys, -inf elsewhere.
    bias = torch.where(ctx_mask,
                       torch.zeros((), dtype=q.dtype, device=q.device),
                       torch.full((), NEG_INF, dtype=q.dtype, device=q.device))
    bias = bias.reshape(1, 1, 1, L)

    attn = F.scaled_dot_product_attention(q * beta, k, v, attn_mask=bias,
                                          scale=default_scale).transpose(1, 2)
    attn = attn * gate.unsqueeze(-1)
    attn = layer.out_proj(attn.reshape(B, L, layer.num_heads * layer.head_dim))
    residual = attn + layer.ff(layer.ff_norm(x + attn))
    return residual.transpose(0, 1)


def masked_get_scale_param(layer, train_size: torch.Tensor, *, device, dtype) -> torch.Tensor:
    """upstream get_scale_param with the context length read from a tensor.

    The arithmetic below is upstream's, unchanged. Only the source of ``n``
    differs, and that is the point of the whole conversion:

      * upstream does ``torch.as_tensor(eval_pos)``, which bakes a SymInt to a
        constant at export — the graph then silently pins ``y`` to whatever
        length was traced;
      * the positional patch works around that with
        ``torch.ones(eval_pos).sum()``, building the value from a dynamic dim;
      * here ``train_size`` is already a runtime tensor, so it is simply cast.

    The workaround disappearing is a small sign the masked contract is the more
    natural one for an exported graph, not merely a different one.
    """
    if layer.disable_attention_scaling:
        return torch.tensor(1.0, device=device, dtype=dtype)
    n = train_size.reshape(()).to(device=device, dtype=dtype)
    n = torch.minimum(n, layer.max_len_f.to(dtype))
    return 1.0 + layer.kappa.to(dtype) * torch.clamp(torch.log(n / layer.n0.to(dtype)), min=0.0)


def masked_model_forward(model, x_src: torch.Tensor, y_src: torch.Tensor,
                         train_size: torch.Tensor) -> torch.Tensor:
    """TabDPTModel.forward with the split as a value.

    Transcribed from upstream so the diff against it stays readable; the only
    changes are the four listed in the module docstring.
    """
    x_src = x_src.transpose(0, 1)  # (T, B, H)
    y_src = y_src.transpose(0, 1)  # (T, B)  -- FULL length now, not the prefix
    total_rows = x_src.shape[0]
    n_think = model.n_thinking_rows

    ctx_mask = context_row_mask(total_rows, train_size, x_src.device)

    x_src = masked_clip_outliers(x_src, ctx_mask, n_sigma=model.clip_sigma)
    x_src = masked_normalize_data(x_src, ctx_mask)
    x_src = masked_clip_outliers(x_src, ctx_mask, n_sigma=model.clip_sigma)
    x_src = torch.nan_to_num(x_src, nan=0.0, posinf=0.0, neginf=0.0)

    x_src = model.encoder(x_src)
    src = model.enc_norm(x_src)

    # Thinking rows prepend n_think positions that are ALWAYS context.
    layer_mask = ctx_mask
    if n_think > 0:
        B = src.shape[1]
        src = torch.cat([model.thinking_embed.unsqueeze(1).expand(n_think, B, -1), src], dim=0)
        layer_mask = torch.cat([torch.ones(n_think, dtype=ctx_mask.dtype, device=ctx_mask.device), ctx_mask], dim=0)

    for l, layer in enumerate(model.transformer_encoder):
        y_emb = model.y_encoders[l](y_src.unsqueeze(-1))
        if n_think > 0:
            B = y_emb.shape[1]
            y_emb = torch.cat([y_emb.new_zeros(n_think, B, y_emb.shape[-1]), y_emb], dim=0)
        residual = masked_encoder_layer_forward(layer, src, y_emb, layer_mask, train_size + n_think)
        src = src + residual

    # Every DATA row, not just the queries: the engine slices by train_size,
    # exactly as it does for tabfm-v1 and mitra.
    #
    # The n_think prefix is dropped here. Upstream's `src[eval_pos + n_think:]`
    # removes the thinking rows and the context in one slice, so converting
    # only the eval_pos half leaves the scaffolding rows in the output and the
    # graph returns T + n_think rows -- caught by mask_parity as a shape
    # mismatch (20 vs 16) before any ONNX existed, which is the entire reason
    # that gate runs first.
    if n_think > 0:
        src = src[n_think:]
    return model.head(src.float())
