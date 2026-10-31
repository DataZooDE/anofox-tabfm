"""Export-friendly patches for Synthefy's Nori (upstream code and weights Apache-2.0).

We DO NOT copy or edit the upstream source: ``synthefy-nori`` is pinned from PyPI (pyproject.toml) and
patched at RUNTIME, as tools/export_tabicl and tools/export_causilo do. Every patch keeps the upstream
function as ``_upstream_*`` so a test can prove the replacement equal to it on the inference path.

Upstream's ``FeaturesTransformer.forward(x, y, eval_pos, task_type="reg")`` already takes a row-major
table plus a label prefix and a split point, so the graph maps onto the engine's ``(x, y)``-only
contract (the split is ``len(y)``) with no engine change, like LimiX-2M, whose fork it is.

Patches (each the identical function on the inference path):

  1. ``FeaturesTransformer.forward`` -> a regression-only forward without three trace hazards:
     the feature-group pad is a Python branch on the symbolic width (``if feature_to_add > 0``); the
     label pad is ``nan * zeros(B, T - S, 1)``, whose length torch.export concretises; and the
     mask-prediction dict branch (off for inference). Same operations, same order, otherwise.
  2. ``QASSMaxScaling.forward`` takes ``math.log(float(key_len))`` of the context length. ``key_len`` is
     the symbolic S, and a Python float of it specialises the graph to the trace's S. The same value
     comes from ``log(ones(S).sum())`` (exact for S < 2**24), which stays symbolic.
  3. ``ValidFeatureEncoder.forward`` calls ``int(eval_pos)``, which specialises S for the same reason,
     and records ``self.valid_feature_num`` (a module attribute written during forward). Both removed;
     the arithmetic is untouched.
  4. ``NormalizationEncoder.forward`` records ``self.mean`` / ``self.std`` the same way; removed.

Found by the export, not by reading: the label embedding's ``.squeeze(2)`` is a torch no-op on the
3-D regression encoder output, but exports as an unconditional ``Squeeze`` ORT rejects (patch 1).
"""

from __future__ import annotations

import torch

_APPLIED = False


def apply() -> None:
    """Idempotently install the export patches on the installed synthefy_nori."""
    global _APPLIED
    if _APPLIED:
        return
    import synthefy_nori.model.encoders as emod
    import synthefy_nori.model.layer as lmod
    import synthefy_nori.model.transformer as tmod

    # --- Patch 1: regression forward ----------------------------------------------------------------
    tmod.FeaturesTransformer._upstream_forward = tmod.FeaturesTransformer.forward

    def _forward(self, x, y, eval_pos, task_type="reg", **kwargs):
        if task_type != "reg":
            raise ValueError("the Nori export path is regression-only")
        batch_size, seq_len, num_feature = x.shape
        fpg = self.features_per_group
        x_dict = {"data": x, "mask": (~torch.isfinite(x)).to(torch.int32).to(x.device)}
        # Upstream: `feature_to_add = (-H) % fpg; if feature_to_add > 0: cat zeros`. Decided at trace time
        # on a symbolic width, so a graph traced at even H fails every odd-width table. Always append the
        # maximum pad (fpg - 1 zero columns, a static width) and slice to ceil(H / fpg) * fpg: the same
        # values as upstream for every H.
        grouped_width = ((num_feature + fpg - 1) // fpg) * fpg
        for k in ("data", "mask"):
            value = x_dict[k]
            if fpg > 1:
                zeros = torch.zeros_like(value[:, :, :1]).expand(-1, -1, fpg - 1)
                value = torch.cat((value, zeros), dim=-1)[:, :, :grouped_width]
            x_dict[k] = value.reshape(batch_size, seq_len, grouped_width // fpg, fpg)
        x_dict["eval_pos"] = eval_pos

        preprocessed_x = self.x_preprocess(x_dict)
        preprocessed_x = self.process_4_x(preprocessed_x)
        x_emb_result = self.encoder_x(preprocessed_x)["data"]

        # Upstream pads y out to T rows with `nan * zeros(B, T - S, 1)`; the pad is taken instead as the
        # TAIL of a T-long NaN tensor, so its length is T - S by construction (LimiX-2M's export hit the
        # concretised pad length; see tools/export_limix).
        y_col = y.unsqueeze(-1)
        nan_rows = torch.full_like(x[:, :, :1], float("nan")).to(y_col.dtype)
        y_col = torch.cat((y_col, nan_rows[:, y_col.shape[1]:, :]), dim=1)
        target_aware_y = y_col.squeeze(-1).clone()
        seq_positions = torch.arange(y_col.shape[1], device=y_col.device)
        y_data_masked = torch.where(
            seq_positions.view(1, -1, 1) >= eval_pos,
            torch.tensor(float("nan"), device=y_col.device, dtype=y_col.dtype),
            y_col,
        )
        embedded_y = self.reg_y_encoder({"data": y_data_masked, "eval_pos": eval_pos})["data"]
        # Upstream `.squeeze(2)`s this for a 4-D [B, T, 1, E] encoder output. The regression encoder's
        # output is 3-D [B, T, E], where torch's squeeze of a non-unit dim is a no-op but the ONNX
        # exporter emits an unconditional Squeeze(axis=2) that ORT rejects. Squeeze only what upstream's
        # squeeze would actually change.
        if embedded_y.dim() == 4:
            embedded_y = embedded_y.squeeze(2)

        embedded_x = self.add_embeddings(x_emb_result)
        embedded_x = self.apply_target_aware_embedding(embedded_x, target_aware_y, task_type="reg",
                                                       eval_pos=eval_pos)
        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)
        encoder_out = self.transformer_encoder(embedded_all, feature_atten_mask=None, eval_pos=eval_pos)[0]
        encoder_out = self.encoder_out_norm(encoder_out)
        return self.reg_y_decoder(encoder_out[:, eval_pos:, -1])

    tmod.FeaturesTransformer.forward = _forward

    # --- Patch 2: QASSMax temperature from a symbolic context length --------------------------------
    lmod.QASSMaxScaling._upstream_forward = lmod.QASSMaxScaling.forward

    def _qass_forward(self, q, key_len):
        # Upstream returns q unchanged for key_len <= 1; the export's Dim keeps S >= 2.
        n = torch.ones(key_len, device=q.device, dtype=torch.float32).sum().clamp(min=2.0)
        log_n = torch.log(n).to(q.dtype)
        if self.qass_mode == "log_only":
            return q * log_n.view(1, 1, 1, 1)
        base_delta = self.base_mlp(log_n.view(1, 1)).view(1, 1, self.num_heads, self.head_dim)
        base_scale = log_n.view(1, 1, 1, 1) * (1.0 + torch.tanh(base_delta))
        if self.qass_mode == "base_only":
            return q * base_scale
        gate_scale = 1.0 + torch.tanh(self.gate_mlp(q))
        return q * base_scale * gate_scale

    lmod.QASSMaxScaling.forward = _qass_forward

    # --- Patch 3: ValidFeatureEncoder without int(eval_pos) ------------------------------------------
    emod.ValidFeatureEncoder._upstream_forward = emod.ValidFeatureEncoder.forward

    def _valid_forward(self, input):
        x = input[self.in_keys[0]]
        eval_pos = input["eval_pos"]
        context = x[:, :eval_pos]
        original_mask = input.get("mask")
        if self.in_keys[0] == "data" and isinstance(original_mask, torch.Tensor) and original_mask.shape == x.shape:
            context = torch.where(original_mask[:, :eval_pos].to(torch.bool),
                                  torch.full_like(context, float("nan")), context)
        finite = torch.isfinite(context)
        finite_count = finite.sum(dim=1)
        finite_min = torch.where(finite, context, torch.full_like(context, float("inf"))).amin(dim=1)
        finite_max = torch.where(finite, context, torch.full_like(context, float("-inf"))).amax(dim=1)
        valid_feature = (finite_count > 1) & (finite_min != finite_max)
        valid_feature_num = torch.clip(valid_feature.sum(-1).unsqueeze(-1), min=1)
        input["_valid_feature_num"] = valid_feature_num
        if self.nan_normalize:
            if self.sqrt_normalize:
                x = x * torch.sqrt(self.num_features / valid_feature_num).unsqueeze(1).expand_as(x)
            else:
                x = x * (self.num_features / valid_feature_num)
        zeros = torch.zeros(*x.shape[:-1], self.num_features - x.shape[-1], device=x.device, dtype=x.dtype)
        input[self.out_key] = torch.cat([x, zeros], -1)
        return input

    emod.ValidFeatureEncoder.forward = _valid_forward

    # --- Patch 4: NormalizationEncoder without module-attribute writes -------------------------------
    emod.NormalizationEncoder._upstream_forward = emod.NormalizationEncoder.forward

    def _norm_forward(self, input):
        x = input[self.in_keys[0]]
        eval_pos = input["eval_pos"]
        pos = eval_pos if self.train_only else x.shape[1]
        if self.remove_outliers:
            x, _, _ = emod.drop_outliers(x, eval_pos=pos, std_sigma=self.std_sigma)
        if self.normalize_x:
            x, mean, std = emod.normalize_mean0_std1(x, eval_pos=pos)
            input["_norm_mean"] = mean
            input["_norm_std"] = std
        input[self.out_key] = x
        return input

    emod.NormalizationEncoder.forward = _norm_forward

    _APPLIED = True


def quantile_mean(q: torch.Tensor) -> torch.Tensor:
    """Upstream's default point estimate (``quantile_collapse="mean"``), closed form, no sort.

    Upstream sorts the K quantiles and integrates the piecewise-linear quantile function on the grid
    tau_k = (k + 1) / (K + 1), extending the outer quantiles as constants to 0 and 1
    (``quantile_dist_mean_simple``). With that uniform grid (spacing d = 1 / (K + 1)):

        left  = tau_0 * q_(0)                     = d * min(q)
        right = (1 - tau_{K-1}) * q_(K-1)         = d * max(q)
        mid   = sum_k d * (q_(k) + q_(k+1)) / 2   = d * (sum(q) - min(q)/2 - max(q)/2)
        mean  = d * (sum(q) + min(q)/2 + max(q)/2)

    Every term is permutation-invariant except through min/max, so the sort is unnecessary and the
    result is upstream's to float rounding. (Sorting 999 values per row would be a TopK in the graph.)
    The MEAN, not the median, because it is upstream's default for ``NoriRegressor.predict`` and the
    number its benchmark R2 figures are computed from, and because the engine scores regression by
    squared error, for which the mean is the optimal point estimate.
    """
    k = q.shape[-1]
    d = 1.0 / (k + 1)
    return d * (q.sum(dim=-1) + 0.5 * q.amin(dim=-1) + 0.5 * q.amax(dim=-1))


class ExportWrapper(torch.nn.Module):
    """Pins Nori to a fixed 2-input ONNX signature (regression only).

    Inputs (B fixed to 1 -- one table per call):
      x  [1, T, H] float32   features (all rows; H is dynamic)
      y  [1, S]    float32   RAW training targets only (S = eval_pos <= T)
    Output:
      [1, T, 1]              a RAW-space point estimate for EVERY row; rows < S are in-context fitted
                             values (the context presented a second time as queries, as for LimiX-2M).

    Regression target: upstream's NoriRegressor z-scores y with numpy's population std (a std below
    1e-12 becomes 1), predicts, and inverts (api.py ``_predict_point``); the model never sees raw
    targets. The engine feeds raw targets and reads the output as raw, so both live in the graph.

    The 999-quantile head is reduced in-graph to the quantile-function mean (``quantile_mean``).
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y):
        s = y.shape[1]
        # S <= T keeps every per-layer slice exact (without it each layer's Min(S, T) is re-simplified by
        # sympy and export time explodes with depth; tools/export_limix measured it). S >= 2 is upstream's
        # QASS regime (key_len <= 1 is a separate branch) and the minimum a std needs.
        torch._check(s <= x.shape[1])
        torch._check(s >= 2)
        mean_y = y.mean(dim=1, keepdim=True)
        std_y = ((y - mean_y) ** 2).mean(dim=1, keepdim=True).sqrt()
        std_y = torch.where(std_y < 1e-12, torch.ones_like(std_y), std_y)
        y_in = (y - mean_y) / std_y
        # [context ; all rows], eval_pos = S: rows >= S are every row of the table. Sample attention is
        # isolated (queries attend to the context only), so each query is independent of the others.
        table = torch.cat([x[:, :s], x], dim=1)
        q = self.m(table, y_in, eval_pos=s, task_type="reg")      # [1, T, K]
        point = quantile_mean(q).unsqueeze(-1)                   # [1, T, 1]
        return point * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)


def build_model(config: dict, seed: int = 0):
    """Random-weight Nori at the given config. No checkpoint bytes anywhere."""
    apply()
    from synthefy_nori.utils.loading import build_model as _build

    torch.manual_seed(seed)
    return _build(dict(config)).eval()


def load_real(ckpt_path: str):
    """The released checkpoint, loaded the way upstream's load_model does (weights_only, EMA weights,
    mask_prediction off), but with the decomposed RMSNorm the graph exports (``native_rms_norm=False``:
    upstream's own switch back to its historical path)."""
    from synthefy_nori.utils.loading import load_model

    apply()
    return load_model(ckpt_path, mask_prediction=False, native_rms_norm=False).eval()


class unpatched:
    """Context manager: run upstream's own functions (the ``_upstream_*`` originals) inside the block.

    Tests use it to compare the patched model against upstream's eager forward on the same weights."""

    def __enter__(self):
        import synthefy_nori.model.encoders as emod
        import synthefy_nori.model.layer as lmod
        import synthefy_nori.model.transformer as tmod

        apply()
        self._targets = [tmod.FeaturesTransformer, lmod.QASSMaxScaling, emod.ValidFeatureEncoder,
                         emod.NormalizationEncoder]
        self._saved = [cls.forward for cls in self._targets]
        for cls in self._targets:
            cls.forward = cls._upstream_forward
        return self

    def __exit__(self, *exc):
        for cls, fn in zip(self._targets, self._saved):
            cls.forward = fn
        return False
