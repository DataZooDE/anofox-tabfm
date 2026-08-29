"""Export-time wrapper + patches for Layer 6 AI's TabDPT (Apache-2.0).

The upstream ``tabdpt`` package is NEVER edited — it is a normal pinned PyPI
dependency and everything here is applied at RUNTIME, the same approach as
``tools/export_tabicl`` and ``tools/export_orion_bix``.

**Why TabDPT needs no retrieval backend.** ``docs/MULTI_MODEL_PLAN.md`` deferred
TabDPT behind a hypothetical ``RetrievalOnnxBackend``. That premise turned out to
be wrong: retrieval is not part of the model. ``TabDPTEstimator`` defaults to
``context_reduction="subsample"`` and only reaches for FAISS when the caller opts
in; either way the reduction happens in the sklearn wrapper and merely decides
*which context rows to hand over*. ``TabDPTModel.forward`` itself takes the whole
context and derives the split from the label length::

    forward(x_src[B, T, F], y_src[B, n_ctx], num_features) -> (T - n_ctx, B, O)
    eval_pos = y_src.shape[0]

That is exactly the engine's ``single_eval_pos`` family (TabPFN / TabICL /
Orion-BiX): a graph that declares no ``train_size`` input, with ``y`` fed as the
training-label prefix. So TabDPT maps onto the existing contract with no engine
change at all — the extension already feeds it what it wants.

**What the wrapper has to absorb.** Upstream splits the work between the model
and the estimator; the graph has to be self-contained, so the estimator's share
is baked in here:

  1. **Feature padding.** The model wants a fixed ``num_features`` width
     (``pad_x`` upstream). The wrapper takes a dynamic ``H`` and pads to
     ``model.num_features`` internally, so the engine keeps feeding real widths.
  2. **Target standardisation (regression).** ``_predict`` does
     ``normalize_data(train_y)`` before the forward and ``y_hat * std_y + mean_y``
     after it. Both are computed from the TRAIN PREFIX only, so both fold into
     the graph: the wrapper standardises ``y[:S]`` on the way in and undoes it on
     the way out. That makes the regression graph raw-in / raw-out, matching the
     ``*_raw`` preprocessing profile the way TabPFN's does.
  3. **The regression point estimate.** ``_expectation_from_regression_logits``
     is a bar-distribution mean over ``linspace(regression_bin_min,
     regression_bin_max, regression_bin_count + 1)`` midpoints. The three bounds
     are plain scalars on the model, not weights, so the bin centres are baked in
     as an inline constant — they are code-generated and stay OUT of the tensor
     map, exactly like TabPFN's ``_pos_base``.
  4. **Head selection.** One checkpoint serves both tasks: the output's first
     ``n_out`` columns are class logits and the rest are regression bins. The
     wrapper slices the half its task needs so each task gets its own graph, as
     every other model here does.

After ``build_wrapper`` the signature is ``x[1,T,H] f32, y[1,S] f32 ->
logits[1,T,C]`` — C = ``n_out`` for classification, 1 for regression (a raw-space
point estimate).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_APPLIED = False


def _patched_encoder_forward(self, x: torch.Tensor, y: torch.Tensor,
                             eval_pos: int) -> torch.Tensor:
    """`TransformerEncoderLayer.forward` with the attention scale made traceable.

    Upstream computes a context-length-dependent attention temperature and hands
    it to SDPA as the ``scale=`` kwarg::

        beta         = get_scale_param(eval_pos)      # 0-dim TENSOR
        custom_scale = 1/sqrt(head_dim) * beta
        F.scaled_dot_product_attention(q, k, v, scale=custom_scale)

    ``scale=`` must be a concrete Python float, so torch.export tries to
    ``guard_float`` it. ``beta`` depends on ``eval_pos``, which is a genuine
    runtime dimension for us (it is ``y``'s length), so the guard is on an
    unbacked symbol and the export dies with GuardOnDataDependentSymNode.

    SDPA computes ``softmax(scale * q @ k^T) @ v``, so multiplying ``q`` by
    ``beta`` and leaving the scale at its default is the SAME function:

        softmax(1/sqrt(d) * (beta*q) @ k^T)  ==  softmax(beta/sqrt(d) * q @ k^T)

    ``beta`` stays a tensor, nothing is guarded, and the length-dependent
    temperature is preserved exactly — it is computed in the graph from the real
    ``eval_pos`` at run time rather than frozen at export time.

    Everything else is copied from upstream unchanged.
    """
    # --- verbatim upstream, up to the marked line -----------------------------
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)
    B, L, _ = x.size()

    h = self.attn_norm(x)
    q = self.q_proj(h)
    gate = torch.sigmoid(self.q_gate(q))
    k = self.k_proj(h[:, :eval_pos])

    h_ctx = h[:, :eval_pos]
    v_in = torch.cat([h_ctx, y], dim=-1)
    v = self.v_proj(v_in)

    q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)

    q, k = self.q_norm(q), self.k_norm(k)
    beta = self.get_scale_param(eval_pos, device=q.device, dtype=q.dtype)
    default_scale = 1.0 / math.sqrt(self.head_dim)

    # --- the patch ------------------------------------------------------------
    # upstream: attn = F.scaled_dot_product_attention(q, k, v,
    #                      scale=default_scale * beta).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q * beta, k, v,
                                          scale=default_scale).transpose(1, 2)
    # --- verbatim upstream again ----------------------------------------------
    attn = attn * gate.unsqueeze(-1)
    attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
    residual = attn + self.ff(self.ff_norm(x + attn))

    return residual.transpose(0, 1)


def _patched_get_scale_param(self, eval_pos, *, device, dtype) -> torch.Tensor:
    """`get_scale_param` without specializing the context length.

    Upstream materialises the context length with ``torch.as_tensor(eval_pos)``.
    ``eval_pos`` is a symbolic dimension for us (``y``'s length), and
    ``as_tensor`` on a SymInt BAKES IT to a constant — the export then succeeds
    but silently produces a graph whose ``y`` input is fixed at whatever length
    the tracing example happened to use, which ORT rejects at the first real
    call with a different context size.

    ``torch.ones(eval_pos).sum()`` builds the same value as a genuine tensor op
    over a dynamic dim, so nothing is specialized. The sum of ``eval_pos`` ones
    is exact in fp32 well past the 1e6 ceiling the export declares (fp32
    integers are exact to 2^24), and it is immediately clamped by ``max_len_f``
    anyway.

    The rest is upstream unchanged.
    """
    if self.disable_attention_scaling:
        return torch.tensor(1.0, device=device, dtype=dtype)
    n = torch.ones(eval_pos, device=device, dtype=dtype).sum()
    n = torch.minimum(n, self.max_len_f.to(dtype))
    beta = 1.0 + self.kappa.to(dtype) * torch.clamp(
        torch.log(n / self.n0.to(dtype)), min=0.0)
    return beta


def apply() -> None:
    """Idempotently install the export monkeypatches on ``tabdpt``."""
    global _APPLIED
    if _APPLIED:
        return
    import tabdpt.model as m

    m.TransformerEncoderLayer.forward = _patched_encoder_forward
    m.TransformerEncoderLayer.get_scale_param = _patched_get_scale_param
    _APPLIED = True


class ExportWrapper(torch.nn.Module):
    """Wrap TabDPTModel into the engine's single_eval_pos contract.

    ``y`` carries ONLY the training labels (length S); rows ``S..T`` of ``x`` are
    the queries. Output is padded back to ``[1, T, C]`` so predictions land on
    rows >= S, matching the engine's read of ``logits[:, train_size:]``.
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
            # Bar-distribution bin centres, mirroring
            # TabDPTRegressor._expectation_from_regression_logits. Derived from
            # three scalar config values, so this is a code-generated constant,
            # not a checkpoint weight: it stays inline and out of the tensor map.
            edges = torch.linspace(float(bin_min), float(bin_max), int(bin_count) + 1)
            self.register_buffer("bin_centres", (0.5 * (edges[:-1] + edges[1:])).float())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: [1, T, H]  y: [1, S] (train labels only)
        S = y.shape[1]

        # (1) pad features to the model's fixed width (upstream `pad_x`).
        x = torch.nn.functional.pad(x, (0, self.num_features - x.shape[2]))

        # (2) standardise the target on the TRAIN PREFIX, exactly as `_predict`
        # does before the forward. Computed unconditionally so the two branches
        # below cannot read an unbound value; for classification `y` holds dense
        # class ids and y_src is left untouched, so this costs a mean/std that
        # is then discarded.
        #
        # The epsilon is ADDITIVE and must stay that way: upstream's
        # normalize_data is `std = maskstd(...) + 1e-6`. Guarding instead with
        # "if std is tiny, use 1.0" agrees with upstream on ordinary data but
        # diverges badly on a CONSTANT training target, where upstream's std is
        # ~1e-6 and predictions collapse to the mean, while a std of 1.0 lets the
        # bar-distribution output through at full scale — a silent wrong answer
        # on exactly the degenerate input the guard exists for.
        #
        # `maskstd` divides by (num - 1), i.e. the unbiased estimator, which is
        # what torch's .std() computes by default; y reaches us NaN-free (the
        # engine imputes), so the mask in upstream's version is all-true here.
        mean_y = y.mean(dim=1, keepdim=True)
        std_y = y.std(dim=1, keepdim=True) + 1e-6
        y_src = y if self.task == "classification" else (y - mean_y) / std_y

        # num_features is documented upstream as unused and slated for removal;
        # it is passed for signature compatibility only.
        num_features = torch.tensor([self.num_features], dtype=torch.long,
                                    device=x.device)
        out = self.m(x_src=x, y_src=y_src, num_features=num_features)  # (T-S, B, O)
        out = out.transpose(0, 1)  # (B, T-S, O)

        if self.task == "classification":
            pred = out[..., : self.n_out]
        else:
            # (3) bar-distribution mean, then (2) undone — raw-space output.
            reg_logits = out[..., self.n_out :]
            weights = torch.softmax(reg_logits.float(), dim=-1)
            centres: torch.Tensor = self.get_buffer("bin_centres")
            point = (weights * centres).sum(dim=-1, keepdim=True)
            pred = point * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)

        # Pad back to [1, T, C]: the engine slices rows >= S.
        C = pred.shape[-1]
        head = torch.zeros(pred.shape[0], S, C, dtype=pred.dtype, device=pred.device)
        return torch.cat([head, pred], dim=1)


def build_wrapper(model, task: str) -> ExportWrapper:
    """Wrap a loaded/random TabDPTModel, reading the bin bounds off the model."""
    apply()
    return ExportWrapper(
        model,
        task=task,
        num_features=model.num_features,
        n_out=model.n_out,
        bin_min=getattr(model, "regression_bin_min", -5.0),
        bin_max=getattr(model, "regression_bin_max", 5.0),
        bin_count=getattr(model, "regression_bin_count", 1000),
    ).eval()
