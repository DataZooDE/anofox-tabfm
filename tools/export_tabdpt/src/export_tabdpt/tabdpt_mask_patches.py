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
    """[T] bool: True on context rows. The whole conversion in one line.

    train_size stays at shape [1] and broadcasts against [T]. It must NOT be
    reshaped to rank 0: MIGraphX's Reshape computes the element count of an
    empty dims list as zero and refuses the graph outright --
    "Reshape: reshape has 0 elements whereas the input has 1" -- which is how
    the first compile of this conversion died. PyTorch and ONNX Runtime both
    accept rank-0 happily, so nothing before the GPU compile could have caught
    it.
    """
    positions = torch.arange(total_rows, device=device)
    return positions < train_size


def _sum_rows(x: torch.Tensor) -> torch.Tensor:
    """Sum over the row axis as a GEMM: ones[1, T] @ x[T, rest].

    Mathematically ``x.sum(dim=0, keepdim=True)``. It is written as a matmul to
    route around a MIGraphX code-generation bug, isolated by ablation on this
    model: a SINGLE masked-statistics block compiles, and two chained ones make
    its reducer template emit invalid C++ --

        invalid operands to binary expression
        ('reducer<...>::inner_storage<float, 1, integral_constant<unsigned,1>>'
         and 'float')

    -- four times, followed by an assertion in optional<tuning_config> and a
    core dump. TabDPT's preprocessing chains three such blocks (clip, normalize,
    clip), so it hits this squarely. No MIGRAPHX_DISABLE_* env var avoids it.

    A GEMM never enters that template, and is a shape a GPU is good at anyway,
    so this costs nothing beyond looking indirect. Verified: with reductions the
    graph aborts the compiler; with matmuls it compiles.
    """
    rows = x.shape[0]
    flat = x.reshape(rows, -1)
    ones = torch.ones(1, rows, dtype=flat.dtype, device=flat.device)
    return (ones @ flat).reshape(1, *x.shape[1:])


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    xz = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))
    return _sum_rows(xz) / _sum_rows(mask.to(x.dtype))


def _masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    # Transcribed from upstream maskstd, including its (num - 1) denominator:
    # the unbiased estimator, which is what torch's .std() uses by default.
    num = _sum_rows(mask.to(x.dtype))
    mean = _masked_mean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, torch.zeros((), dtype=x.dtype, device=x.device))
    return (_sum_rows(diffs * diffs) / (num - 1)) ** 0.5


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

    attn = F.scaled_dot_product_attention(q * beta.reshape(1, 1, 1, 1), k, v, attn_mask=bias,
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
    # Shape [1] throughout, never rank 0 (see context_row_mask): beta
    # broadcasts against q's [B, heads, L, dim] just as well at [1].
    n = train_size.to(device=device, dtype=dtype)
    n = torch.minimum(n, layer.max_len_f.to(dtype).reshape(1))
    return 1.0 + layer.kappa.to(dtype).reshape(1) * torch.clamp(
        torch.log(n / layer.n0.to(dtype).reshape(1)), min=0.0)


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


class MaskExportWrapper(torch.nn.Module):
    """TabDPT under the train_size contract: x, y_full, train_size, d.

    The counterpart of tabdpt_patches.ExportWrapper, which declares only
    ``(x, y)`` and reads the split from ``y``'s length. Here ``y`` is full
    length and ``train_size`` carries the split as a value, so nothing about
    the graph's shapes depends on how many rows happen to be context.

    ``d`` (the real feature count) is declared because the family contract has
    it and because it is what lets the H axis be padded to a bucket: without
    it, padded feature columns are indistinguishable from real zeros. It is
    accepted and currently unused by the forward -- the model pads internally
    to its own fixed width -- but declaring it keeps the graph's inputs
    identical to tabfm-v1's and mitra's, which is what makes the existing
    engine path serve this model with no C++ change at all.
    """

    def __init__(self, model, task: str, num_features: int, n_out: int,
                 bin_min: float, bin_max: float, bin_count: int):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.m = model
        self.task = task
        self.num_features = int(num_features)
        self.n_out = int(n_out)
        if task == "regression":
            edges = torch.linspace(float(bin_min), float(bin_max), int(bin_count) + 1)
            self.register_buffer("bin_centres", (0.5 * (edges[:-1] + edges[1:])).float())

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                train_size: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # x: [1, T, H]   y: [1, T] (FULL length)   train_size: [1]   d: [1]
        total_rows = x.shape[1]
        x = torch.nn.functional.pad(x, (0, self.num_features - x.shape[2]))

        ctx = context_row_mask(total_rows, train_size, x.device).reshape(1, -1)

        # Standardise the target over the CONTEXT rows only -- the masked
        # counterpart of the positional wrapper's y[:, :S] statistics. Getting
        # this from the full y would leak the query labels into the scale of
        # every regression prediction.
        y_ctx_mask = ctx.expand_as(y)
        n_ctx = y_ctx_mask.sum(dim=1, keepdim=True)
        y_zeroed = torch.where(y_ctx_mask, y, torch.zeros((), dtype=y.dtype, device=y.device))
        mean_y = y_zeroed.sum(dim=1, keepdim=True) / n_ctx
        diffs = torch.where(y_ctx_mask, y - mean_y, torch.zeros((), dtype=y.dtype, device=y.device))
        std_y = ((diffs**2).sum(dim=1, keepdim=True) / (n_ctx - 1)) ** 0.5 + 1e-6

        y_src = y if self.task == "classification" else (y - mean_y) / std_y

        out = masked_model_forward(self.m, x, y_src, train_size)  # (T, B, O)
        out = out.transpose(0, 1)                                 # (B, T, O)

        if self.task == "classification":
            return out[..., : self.n_out]
        reg_logits = out[..., self.n_out :]
        weights = torch.softmax(reg_logits.float(), dim=-1)
        centres: torch.Tensor = self.get_buffer("bin_centres")
        point = (weights * centres).sum(dim=-1, keepdim=True)
        return point * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)


def build_mask_wrapper(model, task: str) -> MaskExportWrapper:
    """Wrap a TabDPTModel under the masked contract. No patching of upstream."""
    return MaskExportWrapper(
        model,
        task=task,
        num_features=model.num_features,
        n_out=model.n_out,
        bin_min=getattr(model, "regression_bin_min", -5.0),
        bin_max=getattr(model, "regression_bin_max", 5.0),
        bin_count=getattr(model, "regression_bin_count", 1000),
    ).eval()


def _patched_swiglu_forward(self, x: torch.Tensor) -> torch.Tensor:
    """SwiGLU with the gate split by SLICING instead of ``chunk``.

    Upstream is ``u, v = self.up(x).chunk(2, dim=-1)``. ``torch.chunk`` exports
    as ONNX ``SplitToSequence`` + ``SequenceAt``, and MIGraphX's ONNX parser
    implements neither: the compile dies with "Unknown operator:
    SplitToSequence".

    This is a SECOND MIGraphX blocker in TabDPT, entirely independent of the
    positional train/test split, and it is present in the graph tabdpt already
    ships -- 32 SplitToSequence and 64 SequenceAt, exactly one chunk per layer
    across the 32 layers. docs/ROCM_SINGLE_EVAL_POS.md nominated tabdpt as the
    model worth converting on the strength of the split alone; converting the
    split is necessary and was never sufficient. Only compiling it on the
    hardware could reveal that, because ONNX Runtime supports sequence ops
    perfectly well and the CUDA path has always been happy.

    The rewrite is exact: the halves are the same elements in the same order,
    and ``up`` has a statically known output width, so the split point is a
    constant rather than a traced value.
    """
    h = self.up(x)
    half = h.shape[-1] // 2
    u = h[..., :half]
    v = h[..., half:]
    return self.down(F.silu(u) * v)


def apply_migraphx_friendly() -> None:
    """Patch the ops MIGraphX's parser cannot read. Idempotent.

    Kept separate from the masked forward: this is about what the PARSER
    accepts, not about the train/test contract, and a future backend that
    handles sequence ops would want the contract change without this.
    """
    from tabdpt import model as m

    m.SwiGLU.forward = _patched_swiglu_forward
