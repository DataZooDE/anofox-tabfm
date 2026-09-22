"""TabPFN with the train/test split as a value — the hard case.

TabDPT's conversion (tools/export_tabdpt) was an algebraic rewrite: its
attention is uniform, every query row attends to every context row, so
replacing a slice with a mask changes nothing but the shape story.

TabPFN is not that. `AlongColumnAttention` runs **structurally different
attention for train and test rows**::

    N = single_eval_pos
    k = k_proj(x[:, :N])            # context rows only
    v = v_proj(x[:, :N])
    out_train = sdpa(q[:, :N], k, v)                  # ALL heads
    out_test  = sdpa(q[:, N:], k[:, :, :1], v[:, :, :1])   # FIRST head only
    out = cat([out_train, out_test], dim=1)

Its own docstring calls this "multi-query attention for the test rows": the
train rows get full multi-head attention, the test rows get one shared
key-value head, and neither attends to the test rows at all. That is two
different computations selected by row index, not one computation under a mask.

So the masked form has to compute BOTH and select per row. That is the honest
cost of this conversion and it is not small: roughly double the attention work,
paid on every row, on top of the bucket padding the masked contract already
implies. Whether it is worth paying is a question for the measurement, not for
this module — but it should be read before assuming TabDPT's 5.6x transfers.

Two implementation notes:

* `train_size` is stashed module-level rather than threaded through the
  signature. `single_eval_pos` appears at 29 sites in this architecture and
  changing its type would touch all of them; during export the stashed value is
  a real tensor, so it enters the traced graph as an input exactly as if it had
  been a parameter.
* TabPFN's own `scaled_dot_product_attention` takes (B, S, H, D) and accepts no
  attention mask, so the patched path uses torch's, which does — at the cost of
  transposing to (B, H, S, D) and back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NEG_INF = -1e30  # finite, so bf16/fp16 quantisation cannot turn it into a NaN

#: Set by the export wrapper immediately before the forward. A [1] int64 tensor
#: during tracing, so it becomes a graph input rather than a baked constant.
_TRAIN_SIZE: torch.Tensor | None = None


def set_train_size(train_size: torch.Tensor) -> None:
    global _TRAIN_SIZE
    _TRAIN_SIZE = train_size


def context_row_mask(total_rows: int, train_size: torch.Tensor, device) -> torch.Tensor:
    """[R] bool, True on context rows. Shape [1], never rank 0 — MIGraphX's
    Reshape treats an empty dims list as zero elements and rejects the graph
    (learned the hard way in the TabDPT conversion)."""
    return torch.arange(total_rows, device=device) < train_size


def _patched_along_column_forward(self, x_BcRE, single_eval_pos=None, *,
                                  cached_kv=None, return_kv=False):
    """AlongColumnAttention.forward with the split masked instead of sliced.

    Reproduces both halves of upstream's behaviour:

      * context rows attend to context rows with ALL heads;
      * query rows attend to context rows with the FIRST key-value head only;
      * nothing attends to a query row.

    The first and third fall out of one additive key bias. The second is why
    two attentions are computed and selected per row.
    """
    if _TRAIN_SIZE is None:
        raise RuntimeError("tabpfn_mask_patches: set_train_size() must be called before forward")
    if cached_kv is not None or return_kv:
        raise RuntimeError("tabpfn_mask_patches: the KV-cache path is not part of the exported graph")

    train_size = _TRAIN_SIZE
    Bc, R, _ = x_BcRE.shape
    H = self.num_heads
    D = self.head_dim

    q_BcRHD = self.q_projection(x_BcRE).view(Bc, R, -1, D)
    # K/V over EVERY row; the bias decides which are real keys.
    k_BcRHD = self.k_projection(x_BcRE).view(Bc, R, -1, D)
    v_BcRHD = self.v_projection(x_BcRE).view(Bc, R, -1, D)

    ctx = context_row_mask(R, train_size, x_BcRE.device)
    key_bias = torch.where(ctx,
                           torch.zeros((), dtype=q_BcRHD.dtype, device=q_BcRHD.device),
                           torch.full((), NEG_INF, dtype=q_BcRHD.dtype, device=q_BcRHD.device))
    key_bias = key_bias.reshape(1, 1, 1, R)

    # torch's SDPA wants (B, H, S, D); TabPFN's layout is (B, S, H, D).
    q = q_BcRHD.transpose(1, 2)
    k = k_BcRHD.transpose(1, 2)
    v = v_BcRHD.transpose(1, 2)

    # Train-row behaviour: full multi-head.
    out_mh = F.scaled_dot_product_attention(q, k, v, attn_mask=key_bias)

    # Test-row behaviour: every query head attends to key-value head 0. Expanded
    # rather than relying on GQA support, so the graph exports the same way
    # everywhere.
    k0 = k[:, :1].expand(-1, H, -1, -1)
    v0 = v[:, :1].expand(-1, H, -1, -1)
    out_mqa = F.scaled_dot_product_attention(q, k0, v0, attn_mask=key_bias)

    is_ctx_row = ctx.reshape(1, 1, R, 1)
    out = torch.where(is_ctx_row, out_mh, out_mqa)

    output_BcRHD = out.transpose(1, 2)
    output_BcSF = output_BcRHD.reshape(Bc, R, D * H)
    return self.out_projection(output_BcSF), None


def apply_mask_patches(arch_module) -> None:
    """Install the masked attention on one TabPFN architecture module."""
    arch_module.AlongColumnAttention.forward = _patched_along_column_forward
