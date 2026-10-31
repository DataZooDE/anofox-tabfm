"""Kumo-Tabular -> ONNX: the wrapper that pins the graph contract, plus export patches.

Graph contract -- the SAME five inputs tabfm-v1 and mitra take, so CPU, CUDA (ext graph) and
ROCm (migraphx graph, bucket-padded) all run one graph family:

    x          [1, T, H] f32   engine-preprocessed features (see "Preprocessing" below)
    y          [1, T]    f32   context labels in rows < train_size; the engine's -100 sentinel in
                               the query rows (and in any bucket padding), which is masked here
    train_size [1]       i64   number of context rows (a VALUE, never a shape)
    cat_mask   [1, H]    bool  categorical columns (Kumo embeds them with their own frequencies)
    d          [1]       i64   number of real feature columns (<= H; the rest is bucket padding)
    -> logits  [1, T, C]       classification: C = 10 class logits (the head width)
                               regression:     C = 1, the mean of the 999 quantiles, in the
                               engine-standardised target space (the engine inverts it)

Why the masked contract, not tabicl-v2's positional (x, y[1,S]) one. Upstream already has
it: `_KumoTabular.forward_train(x, y, categorical_mask, num_columns=, train_size=)` is its own
padded-context / padded-column path (NVIDIA/structured-data-models #1204, #1297), where every
split is a mask (`arange < train_size`, `arange < K + num_columns`) and no tensor shape depends
on the data. That is exactly what a fixed-shape MIGraphX compile needs
(docs/ROCM_SINGLE_EVAL_POS.md), and on the CPU it is the same function (gated by tests/).

Fitted values. Context rows go through the ICL stack as queries attending the context, which
contains themselves with their label -- the in-context consistency value tabfm-v1 and mitra
report for `is_training` rows, not an out-of-sample estimate.

Preprocessing. Kumo's single-estimator recipe for a numerical column is, step for step, the
engine's standardising profile: drop columns constant on the context
(UniqueFeatureFilter / DropConstantColumns), z-score with population std + 1e-6, clip to
[-100, 100], two-stage 4-sigma bounds with the log1p soft clip (ClipSigma(method="log")). The
engine does all of it in float64 (upstream casts to float64 for the same reason), so the graph
takes standardised features and the profile is `kumo_tabular_v1_minimal`, not `*_raw`. The
regression target is z-scored by the engine (population std) and inverted after the graph, which
is Kumo's own `Standardize()` target step; the mean over quantiles commutes with that affine map.

The few things this wrapper adds to `forward_train`:
  * y is sanitised in-graph: query/padding rows carry -100, which `Embedding` cannot index.
    Upstream zeroes their embedding anyway (masked_fill on ~context), so feeding class 0 there
    is the identical function.
  * the group wrap `index % num_columns` (CellEmbedding) is rewritten without integer modulo:
    MIGraphX's GPU target miscomputes data-dependent integer division (docs/ROCM_TABPFN_PLAN.md),
    and the shifts are tiny (0, 1, 3), so a few conditional subtractions are exact.
"""

from __future__ import annotations

import torch

import sdm.models.tabfm.cell_embedding as _tabfm_cell
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular

_APPLIED = False

#: Upstream's own CellEmbedding.forward, kept so the parity gate can compare against it.
ORIGINAL_CELL_FORWARD = _tabfm_cell.CellEmbedding.forward

#: Group shifts are 2**g - 1 for g < group_size; with group_size = 3 the largest is 3, so after
#: at most `_WRAP_STEPS` subtractions of num_columns (>= 1) every real column's index is < d.
_WRAP_STEPS = 4


def _wrap(index: torch.Tensor, num_columns: torch.Tensor) -> torch.Tensor:
    """`index % num_columns` for 0 <= index < C + 3 and num_columns >= 1, division-free.

    Columns c >= num_columns (bucket padding) may still be >= num_columns after the loop; their
    values only ever feed padded cells, which are never attention keys, so they are clamped into
    range to keep the gather valid and are otherwise inert.
    """
    n = num_columns
    for _ in range(_WRAP_STEPS):
        index = torch.where(index >= n, index - n, index)
    return index


def apply() -> None:
    """Install the export patches (idempotent, process-global)."""
    global _APPLIED
    if _APPLIED:
        return

    original = ORIGINAL_CELL_FORWARD

    def cell_forward(self, x, categorical_mask, *, num_columns=None, batch_size_limit=None,
                     out=None, **kwargs):
        if num_columns is None:
            return original(self, x, categorical_mask, num_columns=num_columns,
                            batch_size_limit=batch_size_limit, out=out, **kwargs)
        # Same body as upstream's num_columns branch, with the modulo replaced (see _wrap).
        *B, R, C = x.size()
        index = torch.arange(C, device=x.device)
        shift = 2 ** torch.arange(self.group_size, device=x.device) - 1
        index = index.view(C, 1) + shift.view(1, self.group_size)
        index = _wrap(index, num_columns[..., None, None]).clamp(max=C - 1)
        index = index.flatten(-2)
        row_index = index.unsqueeze(-2).expand(*B, R, -1)
        x = x.gather(-1, row_index).unflatten(-1, (C, self.group_size))
        kwargs = {k: v.gather(-1, row_index).unflatten(-1, (C, self.group_size))
                  for k, v in kwargs.items()}
        categorical_mask = categorical_mask.expand(*B, C).gather(-1, index)
        categorical_mask = categorical_mask.unflatten(-1, (C, self.group_size)).unsqueeze(-3)
        freq = torch.where(categorical_mask.unsqueeze(-1), self.cat_freq.to(torch.float32),
                           self.num_freq.to(torch.float32))
        dtype = self.num_lin.weight.dtype
        weight = torch.where(
            categorical_mask[..., None, None],
            self.cat_lin.weight.to(dtype).view(*(1,) * categorical_mask.dim(),
                                               *self.cat_lin.weight.size()),
            self.num_lin.weight.to(dtype).view(*(1,) * categorical_mask.dim(),
                                               *self.num_lin.weight.size()))
        bias = torch.where(
            categorical_mask[..., None],
            self.cat_lin.bias.view(*(1,) * categorical_mask.dim(), -1),
            self.num_lin.bias.view(*(1,) * categorical_mask.dim(), -1)).sum(dim=-2).to(dtype)
        return self._forward(x, freq, weight, bias, out=None, **kwargs)

    _tabfm_cell.CellEmbedding.forward = cell_forward
    _APPLIED = True


class ExportWrapper(torch.nn.Module):
    """Pins Kumo-Tabular to the five-input contract in the module docstring."""

    def __init__(self, model: _KumoTabular, regression: bool):
        super().__init__()
        self.m = model
        self.regression = regression

    def forward(self, x, y, train_size, cat_mask, d):
        T = x.shape[1]
        context = torch.arange(T, device=x.device).unsqueeze(0) < train_size.unsqueeze(-1)  # [1,T]
        if self.regression:
            labels = torch.where(context, y, torch.zeros_like(y))
        else:
            labels = torch.where(context, y, torch.zeros_like(y)).to(torch.int64)
        # Upstream takes both lengths as int32 (SDPA's `seqused_key_value` contract).
        out = self.m.forward_train(x, labels, cat_mask, num_columns=d.to(torch.int32),
                                   train_size=train_size.to(torch.int32))
        if self.regression:
            # The mean over the 999 quantiles (upstream's TabArena / benchmark point estimate,
            # benchmark/tabular/model.py: out.numerical.mean(dim=-1)); invariant to their order,
            # so upstream's SortQuantiles needs no in-graph sort.
            return out.mean(dim=-1, keepdim=True)
        return out


def unshare_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """Give every state_dict key its own parameter.

    Upstream builds ONE `RotaryEmbedding` per row embedding and hands it to every row block's query
    AND key transform, so `row_blocks.{i}.attn.{query,key}_transform.0.inv_freq` (8 keys in the small
    model) are one tensor. The exporter then emits one initializer for all of them, and the engine,
    which injects every tensor of the checkpoint by name, finds 7 keys with no initializer and
    refuses the graph. The checkpoint stores each key separately (with equal values), so unsharing
    keeps the function identical and makes the tensor map cover the checkpoint 1:1.
    """
    import copy

    seen: set[int] = set()
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if id(child) in seen and not list(child.children()):
                setattr(parent, name, copy.deepcopy(child))
            seen.add(id(child))
    return model


def build_model(task: str, dims: dict, seed: int = 0, num_classes: int = 10,
                num_quantiles: int = 999, wake_std: float = 0.05) -> _KumoTabular:
    """Random-init Kumo-Tabular at the given dims. No checkpoint bytes."""
    apply()
    torch.manual_seed(seed)
    model = _KumoTabular(
        num_classes=num_classes if task == "classification" else 0,
        num_quantiles=num_quantiles if task == "regression" else 0,
        **dims)
    unshare_parameters(model)
    # Upstream zero-initialises every attention output projection and MLP output layer, so a
    # random-init model is the identity on its residual stream and blind to attention defects.
    # Wake them (the tabdpt / tabpfn mask gates do the same).
    with torch.no_grad():
        for _, p in model.named_parameters():
            if torch.count_nonzero(p) == 0:
                torch.nn.init.normal_(p, std=wake_std)
    return model.eval()


def load_real(task: str, size: str, path: str) -> _KumoTabular:
    """The published checkpoint, loaded the way upstream does (weights_only state_dict)."""
    apply()
    model = _KumoTabular(
        num_classes=10 if task == "classification" else 0,
        num_quantiles=999 if task == "regression" else 0,
        **MODEL_KWARGS[size])
    unshare_parameters(model)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.eval()
