"""Export-time patches for Prior Labs TabPFN v2 (Apache-2.0 / PriorLabs-1.1).

The upstream `tabpfn` package (architectures/tabpfn_v2.py) is NEVER edited; this
module monkey-patches, at export time only, the handful of ops whose *native*
form blocks a general `torch.export` (dynamo) -> ONNX graph. Each patch is
mathematically identical to upstream for the inputs our runtime feeds (dense
0..C-1 class ids, B=1). Analogous to the TabFM `repeat_interleave` patch
(tools/export_onnx/tabfm_model_patched.py), but TabPFN needs five:

  1. select_features           — upstream has a data-dependent `torch.all(sel)`
     early-return and a `B==1` boolean-index fast path that produce a
     data-dependent output width; both are unexportable. We always take the
     ONNX-friendly scatter/gather path (fixed feature width) and cast the bool
     cumsum operands to int64 (ORT has no CumSum(bool)).
  2. _generate_nan_and_inf_indicator — `torch.isnan(x) * -2.0` lowers to
     aten.mul.Scalar on a bool, which the ONNX exporter cannot translate; we
     cast the indicators to float first.
  3. add_column_embeddings     — upstream draws the per-column positional base
     with `torch.randn((num_cols, E//4))` at runtime (symbolic num_cols ->
     "SymIntArrayRef expected concrete integers"). For v2 these draws are
     DETERMINISTIC (fixed seed), so we precompute a fixed [MAX_COLS, E//4]
     buffer once and slice it. The buffer is code-generated (seed-derived), not
     a checkpoint weight — it stays INLINE in the weight-free graph and is
     EXCLUDED from the tensor map (see export.POS_BASE_NAME).
  4. _do_encoder_nan_check=False — skips the `if isnan(x).any():` guard (a
     data-dependent bool -> Python branch). The check only raises on bad input.
  5. use_multiclass_target_encoding=False — the multiclass target densification
     uses `torch.unique` (data-dependent length). It is the IDENTITY when the
     training labels are already the dense set 0..C-1, which is exactly what the
     C++ ordinal encoder feeds. Disabling it makes the graph general; the
     documented assumption is dense class ids.

train_size is NOT a graph input: TabPFN derives the train/test split from
`len(y)` (single_eval_pos = y.shape[0]). The wrapper therefore takes y as the
train-label prefix [1, N]; N is a genuine runtime dimension. Output is padded to
[1, T, C] so predictions land on rows >= N (matches the engine's read of
logits[:, train_size:]).
"""

from __future__ import annotations

import math
import types

import torch

import tabpfn.architectures.tabpfn_v2 as v2mod
from tabpfn.architectures.tabpfn_v2 import TabPFNV2Config, get_architecture
from tabpfn.preprocessing.torch import ops as opsmod

# Code-generated positional-embedding base: one row per (feature-group) column.
# Cover the largest supported feature count / smallest group size with margin.
MAX_COLS = 1024

# FullSupportBarDistribution tail constants (see architectures/shared/
# bar_distribution.py). HalfNormal(1).icdf(0.5) and E[HalfNormal(1)] = sqrt(2/pi).
# The two outer buckets are half-normal tails; their centre is offset from the
# outer border by E[HalfNormal(scale)] with scale = bucket_width / icdf(0.5).
_HN_ICDF_HALF = float(
    torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(torch.tensor(0.5))
)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _random_regression_borders(num_buckets: int, seed: int) -> torch.Tensor:
    """Deterministic, sorted, strictly-increasing borders for a random-init
    fixture. Length num_buckets+1; positive outer bucket widths (required by
    FullSupportBarDistribution). NOT a checkpoint weight — the fixture is
    weight-free/random-init; real exports inject `criterion.borders`."""
    g = torch.Generator().manual_seed(int(seed) + 777)
    widths = torch.rand(num_buckets, generator=g) + 0.5  # all >= 0.5 > 0
    edges = torch.cat([torch.zeros(1), torch.cumsum(widths, dim=0)])
    # Normalize to a fixed modest span (z-normalized-target scale, ~[-4, 4]),
    # independent of num_buckets. Real checkpoint borders concentrate their
    # mass on a similar range; keeping the fixture on the same scale keeps the
    # point-estimate reduction (a weighted sum over all buckets) numerically
    # stable in fp32 (avoids ORT-vs-PyTorch accumulation blow-up at 5000 bars).
    span = 8.0
    edges = (edges - edges.min()) / (edges.max() - edges.min()) * span - span / 2
    return edges.float().contiguous()


def _patched_select_features(x, sel):
    """ONNX-friendly, fixed-width feature selection (see module docstring #1)."""
    B, total_features = sel.shape
    sel_l = sel.to(torch.long)
    sel_cumsum_BF = sel_l.cumsum(dim=-1)
    not_sel_cumsum_BF = (1 - sel_l).cumsum(dim=-1)
    num_selected_B1 = sel_l.sum(dim=-1, keepdim=True)
    dest_indices_BF = torch.where(
        sel, sel_cumsum_BF - 1, num_selected_B1 + not_sel_cumsum_BF - 1
    )
    source_positions_BF = torch.arange(total_features, device=x.device).expand(B, -1)
    src_indices_BF = torch.zeros(B, total_features, dtype=torch.long, device=x.device)
    src_indices_BF.scatter_(dim=-1, index=dest_indices_BF, src=source_positions_BF)
    num_rows = x.shape[0]
    src_indices_RBF = src_indices_BF.unsqueeze(0).expand(num_rows, -1, -1)
    new_x_RBF = torch.gather(x, dim=2, index=src_indices_RBF)
    position_indices_F = torch.arange(total_features, device=x.device)
    padding_mask_BF = position_indices_F >= num_selected_B1
    return new_x_RBF.masked_fill(padding_mask_BF.unsqueeze(0), 0)


def _patched_nan_inf_indicator(x):
    """float-cast NaN/Inf indicator (see module docstring #2)."""
    dt = x.dtype
    isnan = torch.isnan(x).to(dt)
    posinf = torch.logical_and(torch.isinf(x), torch.sign(x) == 1).to(dt)
    neginf = torch.logical_and(torch.isinf(x), torch.sign(x) == -1).to(dt)
    return (
        isnan * v2mod.NAN_INDICATOR
        + posinf * v2mod.INFINITY_INDICATOR
        + neginf * v2mod.NEG_INFINITY_INDICATOR
    ).to(dt)


def _patched_add_column_embeddings(self, x_BRCX):
    """Slice a precomputed deterministic base instead of runtime randn (#3)."""
    num_cols = x_BRCX.shape[2]
    base = self._pos_base[:num_cols].to(x_BRCX.dtype)
    embs = self.feature_positional_embedding_embeddings(base)
    return x_BRCX + embs[None, None]


def apply_module_patches() -> None:
    """Install the two module-level op patches (idempotent, global)."""
    opsmod.select_features = _patched_select_features
    v2mod.select_features = _patched_select_features
    v2mod._generate_nan_and_inf_indicator = _patched_nan_inf_indicator


def prepare_model_for_export(model, *, max_cols: int = MAX_COLS):
    """Apply the per-instance export patches (#3, #4, #5) to a built model."""
    model.use_multiclass_target_encoding = False
    model._do_encoder_nan_check = False
    e_quarter = model.emsize // 4
    g = torch.Generator().manual_seed(int(model.seed))
    model.register_buffer("_pos_base", torch.randn((max_cols, e_quarter), generator=g))
    model.add_column_embeddings = types.MethodType(_patched_add_column_embeddings, model)
    return model.eval()


def build_random_model(task: str, model_kwargs: dict, seed: int = 0):
    """Random-init TabPFNV2 at the given dims. No checkpoint bytes anywhere."""
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification|regression, got {task!r}")
    apply_module_patches()
    torch.manual_seed(seed)
    kw = dict(model_kwargs)
    if task == "classification":
        kw.setdefault("max_num_classes", 10)
        kw["num_buckets"] = kw.get("num_buckets", -1)
    else:
        # n_out = (max_num_classes or num_buckets); -1 is truthy, so use 0 to
        # fall through to num_buckets and keep multiclass encoding off (0<2).
        kw["max_num_classes"] = 0
        kw.setdefault("num_buckets", 64)
    cfg = TabPFNV2Config(**{k: v for k, v in kw.items()
                            if k in TabPFNV2Config.__dataclass_fields__})
    model = get_architecture(cfg)
    prepare_model_for_export(model)
    if task == "regression":
        # Random-init (weight-free) bar-distribution borders for the fixture.
        # A real export overwrites this buffer with the checkpoint's
        # `criterion.borders` (see load_real_model); the buffer is mapped +
        # externalized like any other weight.
        model.register_buffer(
            "regression_borders",
            _random_regression_borders(model.n_out, seed),
        )
    return model.eval()


def load_real_model(task: str, ckpt_path: str):
    """Load a real TabPFN v2 checkpoint into a patched model (parity only).

    Weights are used transiently for parity; never committed. Returns
    (model, state_dict) where state_dict is the checkpoint-namespace mapping.
    """
    from pathlib import Path

    from tabpfn.model_loading import load_model

    apply_module_patches()
    model, criterion, _cfg, _inf = load_model(path=Path(ckpt_path))
    prepare_model_for_export(model)
    if task == "regression":
        # The bar-distribution borders live in the checkpoint criterion
        # (`criterion.borders`, in the z-normalized target space). Attach them
        # as a model buffer so they enter state_dict() and are mapped +
        # externalized like any weight (convert_weights.py writes them to the
        # injected safetensors under key `regression_borders`).
        if criterion is None or not hasattr(criterion, "borders"):
            raise RuntimeError(
                "regression checkpoint has no FullSupportBarDistribution "
                "criterion.borders — cannot build the point-estimate head")
        model.register_buffer(
            "regression_borders", criterion.borders.detach().float().contiguous())
    return model


class ExportWrapper(torch.nn.Module):
    """Fixed I/O signature around TabPFNV2 (see module docstring).

    Inputs (batch fixed to 1):
      x [1, T, H] float32   preprocessed features (raw-ish; TabPFN scales
                            internally — do NOT z-score on the C++ side)
      y [1, N] float32      TRAIN targets only (N = train_size, a runtime dim).
                            classification: dense class ids 0..C-1.
                            regression:     RAW target values (the wrapper
                            z-normalizes them internally — see below).
    Output:
      logits [1, T, C]      classification: C = max_classes (class logits).
                            regression:     C = 1, a RAW-space POINT ESTIMATE
                            (the bar-distribution mean, de-standardized).
                            Predictions occupy rows >= N; rows < N are zero pad.

    Regression contract (Option A — self-contained, raw-in / raw-out):
      TabPFN's regressor standardizes the target on the TRAIN rows
      (y' = (y - mean) / std, population std) and its logits describe a
      FullSupportBarDistribution over `criterion.borders` in that z-normalized
      space. This wrapper reproduces that end to end: it z-normalizes the RAW
      train targets it is fed, runs the transformer, reduces the bucket logits
      to the distribution MEAN over the (z-space) borders, then maps that mean
      back to raw space via `mean * std + mean`. The engine therefore feeds RAW
      train targets and must NOT standardize the target or inverse-transform the
      output — logits[:, train_size:, 0] is the final prediction.
    """

    def __init__(self, model, task: str = "classification"):
        super().__init__()
        self.m = model
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.task = task

    def _bardist_mean(self, logits):
        """Distribution mean of a FullSupportBarDistribution over the model's
        `regression_borders`. Matches FullSupportBarDistribution.mean exactly
        (inner buckets: centre = midpoint; outer buckets: half-normal tails)."""
        b = self.m.regression_borders.to(logits.dtype)
        bw = b[1:] - b[:-1]  # bucket widths, [num_bars]
        mids = b[:-1] + bw / 2.0
        mean0 = (-bw[0] / _HN_ICDF_HALF * _SQRT_2_OVER_PI + b[1]).reshape(1)
        meanN = (bw[-1] / _HN_ICDF_HALF * _SQRT_2_OVER_PI + b[-2]).reshape(1)
        bucket_means = torch.cat([mean0, mids[1:-1], meanN])  # [num_bars]
        p = torch.softmax(logits, dim=-1)  # [T-N, 1, num_bars]
        return torch.matmul(p, bucket_means)  # [T-N, 1]

    def forward(self, x, y):
        xt = x.permute(1, 0, 2)  # [T,1,H] seq-first
        if self.task == "regression":
            yt = y[0]  # [N] RAW train targets
            ymean = yt.mean()
            # population std (correction=0), matching TabPFN's fit path.
            ystd = torch.clamp(torch.sqrt(((yt - ymean) ** 2).mean()), min=1e-20)
            ynorm = (yt - ymean) / ystd
            logits = self.m(xt, ynorm)  # [T-N, 1, num_buckets]
            znorm_mean = self._bardist_mean(logits)  # [T-N, 1] (z-space)
            raw = znorm_mean * ystd + ymean  # [T-N, 1] (raw space)
            out = raw.unsqueeze(-1)  # [T-N, 1, 1]
        else:
            out = self.m(xt, y[0])  # [T-N, 1, C]
        pad_rows = xt.shape[0] - out.shape[0]
        pad = torch.zeros(pad_rows, out.shape[1], out.shape[2], dtype=out.dtype)
        full = torch.cat([pad, out], dim=0)  # [T,1,C]
        return full.permute(1, 0, 2)  # [1,T,C]
