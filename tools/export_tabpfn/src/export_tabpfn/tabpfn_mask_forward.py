"""The masked whole-model forward for the 2.5-line TabPFN (v2.5, v2.6).

`tabpfn_mask_patches` proved the ATTENTION layer converts exactly. This is the
rest: everything else in the forward that consumes the train/test split or the
real row count, rewritten so both arrive as VALUES and every shape is the padded
bucket. See docs/ROCM_TABPFN_PLAN.md for the inventory this implements.

Inputs to the graph (all padded to the shape bucket, T rows x H features):

    x          [1, T, H]  real data in [:n_rows, :d], zeros elsewhere
    y          [1, T]     train targets in [:train_size], -100 elsewhere
    train_size [1] int64  rows [0, train_size) are context
    n_rows     [1] int64  rows [0, n_rows) are REAL; the rest is bucket padding
    d          [1] int64  columns [0, d) are REAL

Why n_rows exists at all: TabPFN's constant-feature detection and feature-group
normalisation compare every row against row 0, so a padded zero row changes
which columns look constant. Real query rows and padded rows both carry the -100
label sentinel, so y cannot supply it.

Statistics are masked by setting the excluded rows to NaN and calling upstream's
own NaN-aware `torch_nanmean` / `torch_nanstd` -- the same function, the same
unbiased count -- rather than re-deriving them. That is what makes this an exact
rewrite instead of an approximation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import tabpfn.architectures.tabpfn_v2 as v2
import tabpfn.architectures.tabpfn_v2_5 as v25
import tabpfn.architectures.tabpfn_v2_6 as v26
from tabpfn.preprocessing.torch.ops import torch_nanmean

from export_tabpfn import tabpfn_mask_patches as mp
#: The exporter's ONNX-friendly replacements, imported and called DIRECTLY. Each is
#: also installed on an architecture module by apply_module_patches, but only on the
#: ONE architecture being exported, so reaching them through another architecture's
#: module silently gets upstream's unexportable original. That bit twice in one
#: afternoon (select_features, then the NaN/Inf indicator) and only showed up when an
#: architecture was exported alone -- see tests/test_mask_export_isolated.py.
from export_tabpfn.tabpfn_patched import _patched_nan_inf_indicator, _patched_select_features

#: Set by the forward immediately before the blocks run. [C] bool over the
#: columns the row attention sees (feature groups then the target column);
#: False marks a padded group that must not be attended to.
_GROUP_KEEP: torch.Tensor | None = None


def _patched_along_row_forward(self, x_BrSE):
    """AlongRowAttention with padded feature groups masked out as KEYS.

    The number of feature groups is fixed by the PADDED width, so a bucket wider
    than the real feature count adds all-zero groups. Left alone they would be
    real tokens in the attention. The target column is the last index, so they
    sit between the real groups and the target: not a prefix mask.

    Only keys are masked. A padded group's own output is garbage, but nothing
    reads it -- column attention is per-column and the decode reads the target
    column only.
    """
    if _GROUP_KEEP is None:
        raise RuntimeError("tabpfn_mask_forward: _GROUP_KEEP must be set before the blocks run")
    Br, C, _ = x_BrSE.shape
    q = self.q_projection(x_BrSE).view(Br, C, -1, self.head_dim).transpose(1, 2)
    k = self.k_projection(x_BrSE).view(Br, C, -1, self.head_dim).transpose(1, 2)
    v = self.v_projection(x_BrSE).view(Br, C, -1, self.head_dim).transpose(1, 2)
    zero = torch.zeros((), dtype=q.dtype, device=q.device)
    neg = torch.full((), mp.NEG_INF, dtype=q.dtype, device=q.device)
    bias = torch.where(_GROUP_KEEP, zero, neg).reshape(1, 1, 1, C)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    out = out.transpose(1, 2).reshape(Br, C, self.head_dim * self.num_heads)
    return self.out_projection(out)


def apply_mask_patches() -> None:
    """Install the masked attention (both kinds) on the v2 and 2.5-line modules."""
    for mod in (v2, v25, v26):
        mp.apply_mask_patches(mod)
        mod.AlongRowAttention.forward = _patched_along_row_forward


def _nonconstant(x, real, n_real):
    """True where a column is NOT constant over the REAL rows.

    Upstream compares every row against row 0 and tests the count against Ri - 1.
    With padding, Ri is the bucket height, so a constant column would look
    non-constant (the padded rows differ from row 0) -- hence `n_real`.
    """
    eq = x[1:] == x[0]
    valid = real[1:].reshape(-1, 1, 1)
    return (eq & valid).sum(0) != (n_real - 1)


def _masked_normalize_groups(x_RiBF, features_per_group, real_rows, n_real):
    """`_normalize_feature_groups` counting only REAL rows."""
    non_constant_mask = _nonconstant(x_RiBF, real_rows, n_real)
    used = torch.clip(non_constant_mask.sum(-1).unsqueeze(-1), min=1).to(x_RiBF.device)
    scale = features_per_group / used.to(x_RiBF.dtype)
    x_RiBF = x_RiBF * torch.sqrt(scale)
    return torch.where(non_constant_mask.unsqueeze(0).expand_as(x_RiBF), x_RiBF,
                       torch.zeros_like(x_RiBF))


def _keep_mask(x_RiBC, real, n_real, d):
    """[B, H] bool: columns that are neither constant over the REAL rows nor padding.

    Upstream compares every row against row 0. A padded zero row would make a
    real column that is constant at, say, 3.5 look non-constant, so padded rows
    are counted as equal to row 0.
    """
    H = x_RiBC.shape[-1]
    same = (x_RiBC == x_RiBC[0:1]) | ~real.reshape(-1, 1, 1)
    keep = ~same[1:].all(0)
    # A logical OR, not torch.where: ORT's CPU kernel has no Where for bool, and
    # `where(c, ones, keep)` is exactly `keep | c`.
    keep = keep | (n_real <= 1)
    # Padded columns are never real, whatever the rows say (a one-row input makes
    # every column look non-constant, padded ones included).
    return keep & (torch.arange(H, device=x_RiBC.device) < d).reshape(1, H)


def _masked_rows_v2(model, x_RiBC, y_Ri, train, real, d):
    """TabPFN v2: the same conversion in v2's order of operations.

    v2 differs from the 2.5 line in ways that matter here: it GROUPS features
    first and removes constants afterwards, per group slot; it has no thinking
    rows; and it does not ceil imputed class targets. The number of real feature
    groups therefore comes from `d` directly rather than from a constant count.
    """
    dev = x_RiBC.device
    Ri, B, H = x_RiBC.shape
    fpg = model.features_per_group
    n_real = real.sum()

    x, G = v2._pad_and_reshape_feature_groups(x_RiBC, fpg)          # (Ri, B*G, F)
    # v2._remove_constant_features would do this, but it calls its OWN module's
    # select_features, which is only the fixed-width one if v2 was the patched
    # architecture. Same trap as the 2.5 branch; spelled out so it cannot depend on
    # which module the process happened to patch. (Fixed width, so the trailing
    # zero-pad upstream applies is always zero columns.)
    x = _patched_select_features(x, _nonconstant(x, real, n_real).to(torch.bool))
    indicator = _patched_nan_inf_indicator(x)

    train_b = train.reshape(-1, 1, 1)
    nan = torch.full((), float("nan"), dtype=x.dtype, device=dev)
    means = torch_nanmean(torch.where(train_b, x, nan), axis=0, include_inf=True)
    bad = torch.logical_or(torch.isnan(x), torch.isinf(x))
    x = torch.where(bad, means.unsqueeze(0).expand_as(x), x)

    fit = model.standard_scaler.fit(torch.where(train_b, x, nan))
    fit["std"] = torch.where(train.sum() == 1, torch.ones_like(fit["std"]), fit["std"])
    x = model.standard_scaler.transform(x, fitted_cache=fit)

    nc = _nonconstant(x, real, n_real)
    used = torch.clip(nc.sum(-1, keepdim=True), min=1)
    x = v2._normalize_feature_groups(x, fpg, nc, used)

    x = torch.cat([x, indicator], dim=-1).to(model.feature_group_embedder.weight.dtype)
    emb = model.feature_group_embedder(x).unflatten(1, [B, G]).transpose(0, 1)
    emb = model.add_column_embeddings(emb)

    y = torch.where(train_b, y_Ri.reshape(Ri, 1, 1), nan)
    y_ind = _patched_nan_inf_indicator(y)
    y_mean = torch_nanmean(y, axis=0, include_inf=True)
    y_bad = torch.logical_or(torch.isnan(y), torch.isinf(y))
    y = torch.where(y_bad, y_mean.unsqueeze(0).expand_as(y), y)      # v2: no ceil
    y = torch.cat([y, y_ind], dim=-1).to(model.target_embedder.weight.dtype)
    emb_y = model.target_embedder(y).transpose(0, 1)

    x_BRCD = torch.cat([emb, emb_y[:, :, None]], dim=2)

    global _GROUP_KEEP
    real_groups = torch.arange(G, device=dev) < ((d + fpg - 1) // fpg)
    _GROUP_KEEP = torch.cat([real_groups, torch.ones(1, dtype=torch.bool, device=dev)])
    mp.set_train_size(train.sum().reshape(1))
    try:
        for block in model.blocks:
            x_BRCD, _ = block([x_BRCD], None, None)
    finally:
        _GROUP_KEEP = None
    return model.output_projection(x_BRCD[:, :, -1].transpose(0, 1))


def masked_rows(model, x_RiBC, y_Ri, train, real, d):
    if isinstance(model, v2.TabPFNV2):
        return _masked_rows_v2(model, x_RiBC, y_Ri, train, real, d)
    return _masked_rows_25(model, x_RiBC, y_Ri, train, real, d)


def _masked_rows_25(model, x_RiBC, y_Ri, train, real, d):
    """Logits for every row of a padded sequence. Returns [Ri, B, C_out].

    `train` and `real` are [Ri] bool tensors, so the same code serves the plain
    layout and the doubled one used for fitted values. `d` is the real feature
    count. B must be 1.
    """
    dev = x_RiBC.device
    Ri, B, H = x_RiBC.shape
    fpg = model.features_per_group

    n_real = real.sum()

    # -- 1. constant features, over REAL rows only ------------------------------
    keep = _keep_mask(x_RiBC, real, n_real, d)
    n_sel = keep.sum()

    # The exporter's fixed-width select_features, called DIRECTLY. Reaching it
    # through an architecture module (v26.select_features) only works if THAT
    # module happens to have been patched: apply_module_patches("v2.5") patches
    # v2.5's, so a v2.5 export silently got upstream's data-dependent
    # torch.all(sel) and failed. The gate never saw it because it patches all
    # three modules up front.
    x = _patched_select_features(x_RiBC, keep)                # fixed width: kept first
    # The exporter's select_features moves constants to the BACK but leaves their
    # values. Upstream removes them and pads groups with ZEROS, so a partly-real
    # group must hold zeros past n_sel, not the constant column's value.
    x = torch.where(torch.arange(H, device=dev).reshape(1, 1, H) < n_sel, x, torch.zeros_like(x))

    x_RiBgF, G = v26._pad_and_reshape_feature_groups(x, fpg)
    indicator = _patched_nan_inf_indicator(x_RiBgF)

    # -- 2. imputation + standard scaling over TRAIN rows -----------------------
    train_b = train.reshape(-1, 1, 1)
    nan = torch.full((), float("nan"), dtype=x_RiBgF.dtype, device=dev)
    means = torch_nanmean(torch.where(train_b, x_RiBgF, nan), axis=0, include_inf=True)
    bad = torch.logical_or(torch.isnan(x_RiBgF), torch.isinf(x_RiBgF))
    x_RiBgF = torch.where(bad, means.unsqueeze(0).expand_as(x_RiBgF), x_RiBgF)

    fit = model.standard_scaler.fit(torch.where(train_b, x_RiBgF, nan))
    # Upstream: `if x.shape[0] == 1: std = 1`, on the sliced train rows.
    fit["std"] = torch.where(train.sum() == 1, torch.ones_like(fit["std"]), fit["std"])
    x_RiBgF = model.standard_scaler.transform(x_RiBgF, fitted_cache=fit)

    # -- 3. feature-group normalisation over REAL rows --------------------------
    x_RiBgF = _masked_normalize_groups(x_RiBgF, fpg, real, n_real)

    emb = model.feature_group_embedder(torch.cat([x_RiBgF, indicator], dim=-1))
    emb = emb.unflatten(1, [B, G]).transpose(0, 1)             # [B, Ri, G, X]
    emb = model._add_column_embeddings(emb)

    # -- 4. targets: NaN outside the train rows, exactly as _prepare_targets ----
    y = y_Ri.reshape(Ri, 1, 1)
    y = torch.where(train_b, y, nan)
    y_ind = _patched_nan_inf_indicator(y)
    y_mean = torch_nanmean(y, axis=0, include_inf=True)
    y_bad = torch.logical_or(torch.isnan(y), torch.isinf(y))
    y_imp = torch.where(y_bad, y_mean.unsqueeze(0).expand_as(y), y)
    if model.task_type != "regression":
        y_imp = torch.where(y_bad, y_imp.ceil(), y_imp)        # classes: ceil the imputed
    emb_y = model.target_embedder(torch.cat([y_imp, y_ind], dim=-1)).transpose(0, 1)

    x_BRiCD = torch.cat([emb, emb_y[:, :, None]], dim=2)
    x_BRCD, _ = model.add_thinking_rows(x_BRiCD, single_eval_pos=0)
    n_think = model.add_thinking_rows.num_thinking_rows

    # -- 5. the blocks, with the split and the group mask as values -------------
    global _GROUP_KEEP
    real_groups = torch.arange(G, device=dev) < ((n_sel + fpg - 1) // fpg)
    _GROUP_KEEP = torch.cat([real_groups, torch.ones(1, dtype=torch.bool, device=dev)])
    mp.set_train_size((train.sum() + n_think).reshape(1))
    try:
        for block in model.blocks:
            x_BRCD, _ = block([x_BRCD], None, None)
    finally:
        _GROUP_KEEP = None

    out = model.output_projection(x_BRCD[:, n_think:, -1].transpose(0, 1))
    return out


def masked_forward(model, x_RiBC, y_Ri, train_size, n_rows, d, *, duplicate):
    """[Ri, B, C_out] logits for every row of the padded input.

    duplicate=False: the model's own answer for every row. Rows >= train_size
    are the real query predictions; rows below are the train-embedding decode.

    duplicate=True: present the rows a second time as queries, so the first
    n_rows outputs are in-context predictions for every row -- the route that is
    measurably right where the embedding decode is not (v2, v2.5, and regression
    everywhere; see ExportWrapper._all_row_logits). The doubled sequence is
    [context copy | query copy]; only the first train_size rows of the context
    copy are unmasked, and the answer is read from the query copy.
    """
    Ri = x_RiBC.shape[0]
    dev = x_RiBC.device
    idx = torch.arange(Ri, device=dev)
    if not duplicate:
        return masked_rows(model, x_RiBC, y_Ri, idx < train_size, idx < n_rows, d)

    idx2 = torch.arange(2 * Ri, device=dev)
    train = idx2 < train_size
    real = train | ((idx2 >= Ri) & (idx2 < Ri + n_rows))
    out = masked_rows(model, torch.cat([x_RiBC, x_RiBC], 0), torch.cat([y_Ri, y_Ri], 0),
                      train, real, d)
    return out[Ri:]
