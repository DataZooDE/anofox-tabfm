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
train-label prefix [1, N]; N is a genuine runtime dimension. Output is [1, T, C]:
rows >= N are the query predictions (the engine reads logits[:, train_size:]),
and rows < N are the in-context fitted values for the training rows. Those rows
used to be a zero pad, which the engine surfaced as `is_training` fitted values
-- see ExportWrapper._all_row_logits.
"""

from __future__ import annotations

import math
import types

import torch

import tabpfn.architectures.tabpfn_v2 as v2mod
import tabpfn.architectures.tabpfn_v2_5 as v25mod
import tabpfn.architectures.tabpfn_v2_6 as v26mod
import tabpfn.architectures.tabpfn_v3 as v3mod
from tabpfn.architectures.tabpfn_v2 import TabPFNV2Config, get_architecture
from tabpfn.preprocessing.torch import ops as opsmod

# --- Architecture selection ---------------------------------------------------
#
# `tabpfn` ships each generation as a standalone module with the same symbol
# names (`get_architecture`, `_generate_nan_and_inf_indicator`, a Config
# dataclass), so the patch surface is shared and only the module differs.
# "v2" reproduces the original behaviour exactly — the v2 export path is
# unchanged by the 2.5 work (guarded by tests/test_export.py).
ARCHES = {
    "v2": {
        "module": v2mod,
        "config": TabPFNV2Config,
        "get_architecture": get_architecture,
    },
    "v2.5": {
        "module": v25mod,
        "config": v25mod.TabPFNV2p5Config,
        "get_architecture": v25mod.get_architecture,
    },
    # TabPFN-3 (Prior Labs, released 2026-05-12). Exports with the SAME patch
    # surface as 2.5 minus the column-embedding patch: v3 uses RoPE
    # (`use_rope` / `feat_agg_rope_base`) instead of a pre-generated column
    # embedding table, so there is no randn-at-runtime to freeze. The one thing
    # it does share is the multiclass target-range guard — byte-for-byte the same
    # `(y > self.n_out - 1).any()` branch as 2.5 — which `_freeze_in_train_mode`
    # neutralizes. Verified: a fixture-dims v3 model dynamo-exports to ONNX.
    # TabPFN-2.6 (Prior Labs). A 2.5-LINE architecture, not a v3 one: same
    # emsize/nhead/features_per_group/thinking-row layout, and — the part that
    # matters for export — it still carries `pre_generated_column_embeddings`
    # (2000 x emsize//4), so `prepare_model_for_export` takes the same non-v2
    # branch as 2.5 and needs no new patch. What actually changed is
    # `layernorm_type="rmsnorm"` and a deeper per-layer parameterisation (322
    # clf / 325 reg tensors vs 2.5's 154 / 121), both of which are ordinary
    # weights and trace without help.
    "v2.6": {
        "module": v26mod,
        "config": v26mod.TabPFNV2p6Config,
        "get_architecture": v26mod.get_architecture,
    },
    "v3": {
        "module": v3mod,
        "config": v3mod.TabPFNV3Config,
        "get_architecture": v3mod.get_architecture,
    },
}


def _arch(name: str) -> dict:
    try:
        return ARCHES[name]
    except KeyError:
        raise ValueError(f"unknown arch {name!r} (choose from {sorted(ARCHES)})") from None

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


def _patched_v25_add_column_embeddings(self, x_BRGX):
    """TabPFN-2.5 column embeddings without the symbolic-shaped randn (#3').

    Upstream 2.5 draws ``torch.randn((num_cols, encoding_size // 4))`` at runtime
    (symbolic ``num_cols`` -> "SymIntArrayRef expected concrete integers") and
    then OVERWRITES the first 2000 rows with ``pre_generated_column_embeddings``
    — a fixed table shipped as `tabpfn` package data specifically so the values
    are identical across platforms. Our engine caps features at 512, so at
    ``features_per_group >= 1`` the column count can never reach 2000 and the
    random draw is dead code: slicing the pre-generated table is EXACT, not an
    approximation.

    Like v2's ``_pos_base``, this table is code/data-generated rather than a
    checkpoint weight (it is not in the checkpoint's state_dict at all), so it
    stays INLINE in the weight-free graph and never enters the tensor map.
    """
    num_cols = x_BRGX.shape[2]
    base = self._pos_base[:num_cols].to(x_BRGX.dtype)
    embs = self.feature_positional_embedding_embeddings(base)
    return x_BRGX + embs[None, None]


def apply_module_patches(arch: str = "v2") -> None:
    """Install the two module-level op patches (idempotent, global)."""
    mod = _arch(arch)["module"]
    opsmod.select_features = _patched_select_features
    mod.select_features = _patched_select_features
    mod._generate_nan_and_inf_indicator = _patched_nan_inf_indicator


def prepare_model_for_export(model, *, max_cols: int = MAX_COLS, arch: str = "v2",
                             allow_synthetic_pos_base: bool = False):
    """Apply the per-instance export patches (#3, #4, #5) to a built model."""
    # 2.5 dropped the multiclass target encoding; setting the attribute is inert.
    model.use_multiclass_target_encoding = False
    model._do_encoder_nan_check = False
    if arch == "v3":
        # v3 positions features with RoPE, so there is no pre-generated column
        # embedding table and no runtime randn to patch away. Only the shared
        # multiclass target-range guard needs neutralizing (see
        # _freeze_in_train_mode); everything else exports as-is.
        _freeze_in_train_mode(model)
        return model.eval()
    e_quarter = model.emsize // 4
    if arch == "v2":
        g = torch.Generator().manual_seed(int(model.seed))
        model.register_buffer("_pos_base", torch.randn((max_cols, e_quarter), generator=g))
        model.add_column_embeddings = types.MethodType(_patched_add_column_embeddings, model)
    else:
        # 2.5: the fixed, cross-platform table from `tabpfn` package data.
        pre = model.pre_generated_column_embeddings
        if pre.shape[1] == e_quarter:
            base = pre[:max_cols].detach().clone().float()
        elif allow_synthetic_pos_base:
            # Only reachable for the tiny CI fixture, whose emsize is not the
            # released 192 — upstream would take its runtime-randn branch here,
            # so there is no "real" table to reproduce. A seeded draw keeps the
            # fixture deterministic and exportable; it is still code-generated,
            # stays inline and never enters the tensor map. A REAL checkpoint
            # never lands here (load_real_model forbids it).
            g = torch.Generator().manual_seed(int(getattr(model, 'seed', 42)))
            base = torch.randn((max_cols, e_quarter), generator=g)
        else:
            raise RuntimeError(
                f"pre_generated_column_embeddings has width {pre.shape[1]}, expected "
                f"{e_quarter} (emsize // 4) — upstream would fall back to its runtime "
                "randn, which cannot be exported. Check the architecture config."
            )
        model.register_buffer("_pos_base", base)
        model._add_column_embeddings = types.MethodType(
            _patched_v25_add_column_embeddings, model)
        _freeze_in_train_mode(model)
    return model.eval()


def _freeze_in_train_mode(model) -> None:
    """Pin a 2.5 model in train mode so its target-range guard is skipped (#6).

    TabPFNV2p5.forward has an input-validation branch::

        if (not self.training and self.task_type == "multiclass"
                and (y > self.n_out - 1).any()):
            raise ValueError("Target is out of range. ...")

    ``.any()`` on a symbolic tensor is a data-dependent Python branch and aborts
    the trace with ``GuardOnDataDependentSymNode: Could not guard on
    data-dependent expression Eq(u0, 1)``. Like patch #4 it is a pure guard: it
    only raises on bad input, and the C++ ordinal encoder always feeds dense
    0..C-1 class ids.

    ``self.training`` appears EXACTLY ONCE in the whole ``tabpfn_v2_5`` module —
    this guard — and the released configs set ``dropout = 0.0``, so train mode is
    numerically inert. We therefore pin ``training = True`` rather than trying to
    excise a branch from the middle of a 300-line ``forward``.

    The instance's ``train`` is neutralized because ``export.export_graph``
    wraps the model and calls ``.eval()``, which would otherwise propagate
    ``train(False)`` back down into it.
    """
    model.training = True
    model.train = types.MethodType(lambda self, mode=True: self, model)


def build_random_model(task: str, model_kwargs: dict, seed: int = 0, arch: str = "v2"):
    """Random-init TabPFN at the given dims. No checkpoint bytes anywhere."""
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification|regression, got {task!r}")
    apply_module_patches(arch)
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
    spec = _arch(arch)
    config_cls = spec["config"]
    cfg = config_cls(**{k: v for k, v in kw.items()
                        if k in config_cls.__dataclass_fields__})
    model = spec["get_architecture"](cfg)
    prepare_model_for_export(model, arch=arch, allow_synthetic_pos_base=True)
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


def load_real_model(task: str, ckpt_path: str, arch: str = "v2"):
    """Load a real TabPFN checkpoint into a patched model (parity only).

    Weights are used transiently for parity; never committed. Returns
    (model, state_dict) where state_dict is the checkpoint-namespace mapping.
    """
    from pathlib import Path

    from tabpfn.model_loading import load_model

    apply_module_patches(arch)
    model, criterion, _cfg, _inf = load_model(path=Path(ckpt_path))
    prepare_model_for_export(model, arch=arch)
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
                            EVERY row carries a real value: rows >= N are the
                            query predictions, rows < N the in-context fitted
                            values for the training rows (they used to be a
                            zero pad -- see _all_row_logits).

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

    def _all_row_logits(self, xt, y_used):
        """Model logits for EVERY data row, context rows included.

        Upstream returns logits for the query rows only -- its decoder slices
        the row embeddings at the train/test boundary before projecting -- and
        this wrapper used to zero-pad the context rows back to [T,1,C]. The
        engine surfaces those rows as `is_training` "in-context fitted values",
        so what it actually surfaced was the pad: argmax of zeros is one class
        for every row, and the bar-distribution mean of zeros is one number.
        Measured across the family on separable data where mitra and tabicl-v2
        score 1.0, every TabPFN generation scored ~1/3, i.e. chance.

        Nothing upstream is patched to fix it. Asking for the non-standard
        output already returns `train_embeddings` -- the same per-row embeddings
        the decoder projects, with any thinking-row prefix already stripped --
        so the context rows can be projected with the model's OWN head and
        concatenated.

        That the head is the right one is checked rather than assumed: for all
        four architectures, re-projecting `test_embeddings` this way reproduces
        upstream's `standard` output BIT-EXACTLY (max abs diff 0.0). v3
        multiclass is the one that would have been got wrong by pattern-matching
        the others -- it has no `output_projection` at all and decodes through
        `many_class_decoder`, attending the queries against the train rows, so
        its fitted values come from passing the train embeddings as both.

        Reproducing `standard` proves the HEAD is right. It does not prove the
        TRAIN embeddings are in the same space as the test ones, and on v2 they
        are not -- see the branch below.
        """
        needs_duplicate = (isinstance(self.m, (v2mod.TabPFNV2, v25mod.TabPFNV2p5))
                           or self.task == "regression")
        if needs_duplicate:
            # v2 and v2.5 need a different route, and which architectures those
            # are was established by MEASURING each one on real weights. The
            # embedding route below looks like it should work for them too.
            #
            # It does not. Fitted accuracy on three separable classes, real
            # weights, where every architecture's query rows score 1.00:
            #
            #        embedding route   duplicate route
            #   v2        0.35              1.00
            #   v2.5      0.68              1.00
            #   v2.6      1.00              1.00
            #   v3        1.00              1.00
            #
            # For v2 and v2.5 the target-column embedding is not a decodable
            # posterior, so projecting it yields confident nonsense -- worse
            # than the zero pad it replaces, because it looks like an answer.
            # v2's 0.35 is chance; v2.5's 0.68 is the more dangerous number,
            # since it is high enough to look like a working model.
            #
            # REGRESSION always takes the duplicate route, whatever the
            # architecture. Classification only reads an argmax, which absorbs
            # small logit error -- on v2.6 the two routes agree to 0.0018 of
            # probability and pick the same class every time. Regression decodes
            # the bar distribution to a MEAN, which does not absorb it: v2.6
            # fitted values correlate 0.79 with the target through the cheap
            # route and 1.00 through the duplicate one.
            #
            # So the cheap route survives exactly where it is measurably right:
            # v2.6 and v3 CLASSIFICATION -- which is the hot path. The duplicate
            # route costs ~1.5x wall-clock (measured, v2.6 real dims: 198->318 ms
            # at T=500, 489->738 ms at T=1000) and there is no reason to charge
            # that to the most-used models for values already correct.
            # test_fitted_values_route pins the split so it cannot rot silently.
            #
            # Instead, ask the model the actual question: present the training
            # rows a SECOND time, as queries. Rows >= N are the query section,
            # so [train ; train ; test] returns predictions for the train rows
            # evaluated in context and then the real test rows -- exactly [T,1,C].
            # Measured on real weights: fitted 1.00, query 1.00, and the query
            # half is BIT-IDENTICAL to the ordinary single-pass call, so this
            # cannot move a prediction anyone already relies on.
            #
            # The cost is a sequence of N+T instead of T, paid only by the
            # architectures whose cheap route is wrong.
            n_train = y_used.shape[0]
            return self.m(torch.cat([xt[:n_train], xt], dim=0), y_used)

        res = self.m(xt, y_used, only_return_standard_out=False)
        test_out = res["standard"]                 # [T-N, 1, C]
        train_emb_NBD = res["train_embeddings"]    # [N, 1, D]

        if getattr(self.m, "task_type", None) == "multiclass" and hasattr(
                self.m, "many_class_decoder"):
            # (B, R, D) layout, and the train labels the decoder conditions on.
            tr_BND = train_emb_NBD.transpose(0, 1)
            fitted = self.m.many_class_decoder(tr_BND, tr_BND, y_used.unsqueeze(0))
        else:
            fitted = self.m.output_projection(train_emb_NBD)

        # Mirror upstream's own output guard, so context and query rows are
        # sanitised the same way rather than only the half it computed.
        if getattr(self.m, "_nan_safe_output", False):
            fitted = torch.nan_to_num(fitted, nan=0.0)

        return torch.cat([fitted, test_out], dim=0)  # [T,1,C]

    def forward(self, x, y):
        xt = x.permute(1, 0, 2)  # [T,1,H] seq-first
        if self.task == "regression":
            yt = y[0]  # [N] RAW train targets
            ymean = yt.mean()
            # population std (correction=0), matching TabPFN's fit path.
            ystd = torch.clamp(torch.sqrt(((yt - ymean) ** 2).mean()), min=1e-20)
            ynorm = (yt - ymean) / ystd
            logits = self._all_row_logits(xt, ynorm)  # [T, 1, num_buckets]
            # Over every row now, so context rows decode to real point estimates
            # instead of the bar mean of a zero vector.
            znorm_mean = self._bardist_mean(logits)  # [T, 1] (z-space)
            raw = znorm_mean * ystd + ymean  # [T, 1] (raw space)
            full = raw.unsqueeze(-1)  # [T, 1, 1]
        else:
            full = self._all_row_logits(xt, y[0])  # [T, 1, C]
        return full.permute(1, 0, 2)  # [1,T,C]
