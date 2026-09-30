"""Causilo -> ONNX: export patches and the wrapper that pins the graph contract.

Upstream has no `forward`: `Model` only registers blocks, and
`causilo.execution.runner.ModelRunner.predict` defines the order. The wrapper below
re-implements that order (embed -> column -> row -> column -> pool -> prediction ->
head) and calls the modules directly, exactly as the TabICL exporter does with its
three stages. Nothing upstream is copied; four small monkeypatches make the modules
traceable.

Graph contract (same family as tabicl-v2: the split is the LENGTH of y):

    x  [1, T, H] f32   engine-preprocessed features: encoded, finite, UN-normalised
    y  [1, S]    f32   training targets only (class ids, or RAW regression targets)
    -> [1, T, C]       classification: class logits (C = the head width, 10)
                       regression:     C = 1, a RAW-space point estimate
                       Every row is a real prediction; rows < S are IN-CONTEXT fitted
                       values (see "Fitted values" below).

Fitted values. The engine surfaces the S context rows as `is_training` values, and
upstream only predicts the rows after the context. The cheap way to get values for them
(run the last prediction layer over all rows) is measurably WRONG on real weights: fitted
R2 0.50 on a clean linear problem where the query R2 is 0.999, fitted accuracy 0.457 on a
5-class problem where the query accuracy is 0.717 (spike, 2026-09-30). Presenting the
context rows a SECOND time as queries, i.e. predicting `[context ; all rows]`, gives 1.000
on both, and leaves the query rows unchanged to float noise (<= 2e-5). That is the route
#51 measured for TabPFN v2/v2.5 and for regression everywhere, for the same reason. It
costs S extra rows per call. The values are in-context consistency checks (a row's own
label is in its context), not out-of-sample estimates, as for every other model here.

Everything Causilo does outside the network for its single-estimator ("none") path is
done in-graph, so the engine's `*_raw` profile applies: the train-prefix z-score, the
two-stage 4-sigma tail bounds and the arcsinh tail compression
(`causilo.data.normalization.Normalizer`), and, for regression, the target
StandardScaler and its inverse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from causilo.execution.runner import inject_targets
from causilo.model import Model
from causilo.nn import embeddings as _embeddings
from causilo.nn import prediction as _prediction
from causilo.nn.layers import scaling as _scaling

_APPLIED = False

#: np.finfo(float32).eps. Upstream computes in float64; the graph computes in float32,
#: so the "is this column constant" tolerance has to be float32-sized or rounding noise
#: in the mean makes a constant column look like it varies.
_EPS32 = 1.1920929e-07


def apply() -> None:
    """Install the export patches (idempotent, process-global)."""
    global _APPLIED
    if _APPLIED:
        return

    # --- Patch 1: QueryScale takes log(key_count) with Python `math.log` ---------
    # `key_count` is key.shape[-2]. Under dynamic shapes that is a symbolic int, and
    # `math.log(max(n, 1))` either specialises to the export example's length or
    # fails. With RANDOM weights the baked constant is invisible to a parity check at
    # the example shape and wrong at every other one, so tests/ also assert a Log node
    # fed from a Shape. The length is materialised as a tensor instead (the same move
    # as tools/export_tabicl's ssmax patch).
    def query_scale_forward(self, query, key_count):
        length = query.new_ones(key_count).sum()
        log_count = torch.log(torch.clamp(length, min=1.0)).reshape(1, 1)
        length_gain = _scaling.bounded_multiplier(self.length(log_count), 8.0)
        length_gain = length_gain * _scaling.bounded_multiplier(self.correction(log_count), 2.0)
        content_gain = _scaling.bounded_multiplier(self.content(query), 2.0)
        return query * (length_gain.reshape(self.heads, 1, self.head_width) * content_gain)

    _scaling.QueryScale.forward = query_scale_forward

    # --- Patch 2: Head.forward wraps the output head in torch.autocast(enabled=False) ---
    # A no-op on CPU, but dynamo would emit an autocast higher-order op.
    def head_forward(self, rows):
        return self.output(F.gelu(self.hidden(self.normalize(rows.float()))))

    _prediction.Head.forward = head_forward

    # --- Patch 3: TargetEmbedding builds one-hot with F.one_hot -------------------------
    # ONNX OneHot is not in the MLX interpreter's op table (src/tabfm_mlx_graph_ops.inc),
    # and an equality against arange is the same tensor.
    def target_forward(self, targets):
        if self.classes:
            ids = torch.arange(self.classes, device=targets.device)
            values = (targets.long().unsqueeze(-1) == ids).float()
        else:
            values = targets.unsqueeze(-1)
        return self.projection(values)

    _embeddings.TargetEmbedding.forward = target_forward

    _APPLIED = True


def _asinh(z: torch.Tensor) -> torch.Tensor:
    """asinh spelled out: the MLX interpreter has no Asinh. Callers only pass z >= 0."""
    return torch.log(z + torch.sqrt(z * z + 1.0))


def normalise_features(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`Normalizer.fit(train, "none").transform(x)`, in-graph. [1,T,H] -> [1,T,H].

    Statistics come from the first S = y.shape[1] rows only. Missing (NaN) cells are
    mean-filled for the statistics and restored to NaN afterwards, which is what
    upstream does for the network's missing-value embedding. The engine imputes NaN
    before the graph, so on engine input this is the finite path; the NaN path is kept
    so the graph is checkable against upstream on tables that do contain NaN.
    """
    S = y.shape[1]
    zero = torch.zeros_like(x[:, :1])                       # [1,1,H]
    missing = torch.isnan(x)
    train_missing = missing[:, :S]
    train = x[:, :S]

    # fill = train column mean over observed cells (0 when a column has none)
    observed = (~train_missing).to(x.dtype)
    count = observed.sum(1, keepdim=True)
    fill = torch.where(count > 0,
                       torch.where(train_missing, zero, train).sum(1, keepdim=True) / count.clamp(min=1.0),
                       zero)
    filled = torch.where(missing, fill, x)

    # z-score (np.mean / np.std, ddof=0); a column whose spread is at rounding level -> 1
    ftrain = filled[:, :S]
    center = ftrain.mean(1, keepdim=True)
    spread = ((ftrain - center) ** 2).mean(1, keepdim=True).sqrt()
    tolerance = 8.0 * _EPS32 * torch.clamp(center.abs(), min=1.0)
    spread = torch.where(spread <= tolerance, torch.ones_like(spread), spread)
    scaled = (filled - center) / spread

    # outlier_bounds: two passes, ddof = 1 when there is more than one training row
    strain = scaled[:, :S]
    n_train = torch.ones_like(y).sum()                      # S as a 0-d float tensor
    correction = (n_train > 1).to(x.dtype)
    mean = strain.mean(1, keepdim=True)
    preliminary = torch.clamp(
        (((strain - mean) ** 2).sum(1, keepdim=True) / (n_train - correction)).sqrt(), min=1e-6)
    extreme = (strain - mean).abs() > 4.0 * preliminary
    kept = (~extreme).to(x.dtype)
    kept_count = kept.sum(1, keepdim=True)
    centre2 = torch.where(kept_count > 0,
                          torch.where(extreme, zero, strain).sum(1, keepdim=True) / kept_count.clamp(min=1.0),
                          zero)
    dof = kept_count - correction
    variance = torch.where(extreme, zero, (strain - centre2) ** 2).sum(1, keepdim=True) / dof.clamp(min=1.0)
    spread2 = torch.clamp(torch.where(dof > 0, variance.sqrt(), torch.ones_like(variance)), min=1e-6)
    lower, upper = centre2 - 4.0 * spread2, centre2 + 4.0 * spread2

    # compress_tails: smooth arcsinh beyond the bounds, centre untouched
    values = torch.where(scaled < lower, lower - _asinh(lower - scaled), scaled)
    values = torch.where(values > upper, upper + _asinh(values - upper), values)
    return torch.where(missing, torch.full_like(values, float("nan")), values)


def _column(component, features: torch.Tensor, count) -> torch.Tensor:
    """ModelRunner._column without chunking/autocast: attend over rows per feature group."""
    return component(features.transpose(1, 2), context_rows=count).transpose(1, 2)


def predict(model: Model, table: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """`ModelRunner.predict`: raw [1, Q, outputs] scores for the rows AFTER the context.

    table is [1, S + Q, H] with the S training rows first; targets is [1, S]. Stage order,
    target injection and the query-only last prediction layer are upstream's own
    (causilo/execution/runner.py); chunking, autocast and caching are left out.
    """
    count = targets.shape[1]
    features = model.feature_embedding(table)
    features = inject_targets(features, model.feature_target(targets))
    features = _column(model.columns[0], features, count)
    features = model.row(features)
    features = _column(model.columns[1], features, count)
    rows = model.pool(features)
    rows = inject_targets(rows, model.row_target(targets))
    return model.head(model.prediction(rows, context_rows=count))


def predict_all_rows(model: Model, table: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[1, T, outputs] for EVERY row of `table`: the context rows are presented a second time
    as queries (see "Fitted values" in the module docstring)."""
    count = targets.shape[1]
    return predict(model, torch.cat([table[:, :count], table], dim=1), targets)


class ExportWrapper(torch.nn.Module):
    """Pins Causilo to the two-input ONNX signature described in the module docstring."""

    def __init__(self, model: Model):
        super().__init__()
        self.m = model
        self.regression = model.config.task == "regression"

    def forward(self, x, y):
        table = normalise_features(x, y)
        if not self.regression:
            return predict_all_rows(self.m, table, y)

        # Regression: sklearn StandardScaler on the RAW training targets (population std;
        # a scale at rounding level -> 1), the network sees z-scored targets, the point
        # estimate is the mean over the output channels (invariant to upstream's sort),
        # then the inverse scaler returns RAW units.
        mean_y = y.mean(dim=1, keepdim=True)
        std_y = ((y - mean_y) ** 2).mean(dim=1, keepdim=True).sqrt()
        std_y = torch.where(std_y < 10.0 * _EPS32, torch.ones_like(std_y), std_y)
        channels = predict_all_rows(self.m, table, (y - mean_y) / std_y)    # [1,T,channels]
        point = channels.mean(dim=-1, keepdim=True)
        return point * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)            # [1,T,1]


def build_model(task: str, config, seed: int = 0) -> Model:
    """Random-init Causilo at the given dims. No checkpoint bytes.

    Upstream leaves the latent / frequency / missing-value parameters as `torch.empty`
    ("load a checkpoint before executing"), so they are filled explicitly: an empty
    parameter would be arbitrary memory, not a model.
    """
    apply()
    torch.manual_seed(seed)
    model = Model(config.model_config(task))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("latents", "frequencies", "missing")):
                torch.nn.init.normal_(param, std=1.0)
    return model.eval()
