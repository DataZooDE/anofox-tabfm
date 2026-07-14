"""Export-friendly monkeypatches for soda-inria TabICL (BSD-3-Clause).

We DO NOT copy or edit the ``tabicl`` package. Instead we apply a small,
audited set of *runtime* monkeypatches to the installed ``tabicl`` module before
tracing (analogous to ``tools/export_onnx/tabfm_model_patched.py``, but done as
targeted method replacements so the patch surface stays tiny and reviewable).

Each patch is mathematically identical to upstream on the inference path; it only
removes a data-dependent Python branch / a non-exportable op so the model can go
through ``torch.onnx.export(dynamo=True, opset=18)``. TabICL's own forward
dispatches on ``self.training`` — the *training-mode* code path is the clean,
single-forward one (no KV cache / InferenceManager / feature-shuffle Python
loops), so the export wrapper puts the model in ``train()`` and calls the three
stage modules directly.

Patches (all confirmed necessary by the export spike, see FEASIBILITY notes):

  1. ``SkippableLinear.forward`` — ``if skip_mask.any(): out[mask]=skip`` is a
     data-dependent branch. Rewritten branchless with ``torch.where``.
  2. ``InducedSelfAttentionBlock.forward`` — same skip-branch pattern around the
     ISAB; rewritten compute-all-then-mask (identical output, skipped CLS
     columns are overwritten with the skip sentinel afterwards).
  3. ``ColEmbedding._compute_embeddings`` — upstream computes
     ``int(y_train.max().item())`` to decide mixed-radix ensembling for
     many-class problems (> max_classes). ``.item()`` is a data-dependent host
     sync that bakes a value. The engine guarantees ``num_classes <=
     max_classes`` (it pads/limits classes), and regression has
     ``max_classes == 0`` so mixed-radix never triggers; we force the standard
     target-aware branch, removing the ``.item()``.
  4. ``ColEmbedding.feature_grouping`` — the "same" grouping indexes columns with
     ``(idxs + 2**i) % H`` where H is the *dynamic* feature axis. onnxscript's
     ``aten_remainder_scalar`` calls ``int()`` on the symbolic divisor and fails
     at ONNX translation. Rewritten with ``torch.remainder(idx, H_tensor)``
     (ONNX ``Mod``), bit-exact for all H including wrap-around (verified H=2,3,7).
  5. ``ssmax._logn`` — QASSMax computes ``log(n)`` where ``n`` is the (dynamic)
     key length; upstream does ``math.log(n)`` on the Python int, which torch
     *specializes to the export-example train_size* and bakes as a constant
     (no ``Log`` op in the graph). With random init this is invisible, but with
     TRAINED weights it would freeze the scalable-softmax temperature to one
     train_size. Rewritten to derive the length from a dynamic tensor
     (``ones(n).sum()``) so an in-graph ``ReduceSum -> Log`` stays dynamic.

After ``apply()`` the model exports with input signature ``x[1,T,H] f32,
y[1,S] f32`` (S = train_size; y holds ONLY the training labels) and output
``logits[1,T,C]``. H is dynamic; train_size is the runtime length of ``y``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_APPLIED = False


def apply() -> None:
    """Idempotently install the five export monkeypatches on ``tabicl``."""
    global _APPLIED
    if _APPLIED:
        return

    import tabicl._model.layers as layers
    import tabicl._model.ssmax as ssmax
    from tabicl._model.embedding import ColEmbedding

    # --- Patch 1: SkippableLinear.forward (branchless) ---------------------
    def skippable_linear_forward(self, src):
        out = F.linear(src, self.weight, self.bias)
        skip = (src == self.skip_value).all(dim=-1, keepdim=True)
        return torch.where(skip, torch.full_like(out, self.skip_value), out)

    layers.SkippableLinear.forward = skippable_linear_forward

    # --- Patch 2: InducedSelfAttentionBlock.forward (compute-all + mask) ----
    def isab_forward(self, src, train_size=None):
        out = self.induced_attention(src, train_size)
        skip = (src == self.skip_value).all(dim=(-2, -1), keepdim=True)
        return torch.where(skip, torch.full_like(out, self.skip_value), out)

    layers.InducedSelfAttentionBlock.forward = isab_forward

    # --- Patch 3: ColEmbedding._compute_embeddings (no mixed-radix .item) ---
    def compute_embeddings(self, features, train_size, y_train=None, embed_with_test=False):
        src = self.in_linear(features)
        if not self.target_aware:
            src = self.tf_col(src, train_size=None if embed_with_test else train_size)
        else:
            assert y_train is not None, "y_train must be provided when target_aware=True."
            if self.max_classes > 0:
                y_emb = self.y_encoder(y_train.float())
            else:
                y_emb = self.y_encoder(y_train.unsqueeze(-1))
            src[..., :train_size, :] = src[..., :train_size, :] + y_emb
            src = self.tf_col(src, train_size=None if embed_with_test else train_size)
        if self.affine:
            weights = self.ln_w(self.out_w(src))
            biases = self.ln_b(self.out_b(src))
            embeddings = features * weights + biases
        else:
            embeddings = src
        return embeddings

    ColEmbedding._compute_embeddings = compute_embeddings

    # --- Patch 4: feature_grouping modulo -> tensor remainder (dynamic H) ---
    def feature_grouping(self, X):
        if not self.feature_group:
            return X.unsqueeze(-1)
        B, T, H = X.shape
        size = self.feature_group_size
        mode = "same" if self.feature_group is True else self.feature_group
        if mode == "same":
            idxs = torch.arange(H, dtype=torch.long, device=X.device)
            H_t = idxs.new_full((), H)  # 0-d tensor holding the dynamic H
            X = torch.stack(
                [X[:, :, torch.remainder(idxs + 2**i, H_t)] for i in range(size)],
                dim=-1,
            )
        else:
            x_pad_cols = (size - H % size) % size
            if x_pad_cols > 0:
                X = F.pad(X, (0, x_pad_cols), value=0)
            X = X.reshape(B, T, -1, size)
        return X

    ColEmbedding.feature_grouping = feature_grouping

    # --- Patch 5: ssmax._logn dynamic (keep Log in-graph) ------------------
    def _logn_dynamic(n, device, dtype):
        length = torch.ones(n, device=device, dtype=dtype).sum()  # n as a tensor
        return torch.log(torch.clamp(length, min=1.0))

    ssmax._logn = _logn_dynamic

    _APPLIED = True


class ExportWrapper(torch.nn.Module):
    """Pins TabICL's training-path stages to a fixed 2-input ONNX signature.

    Inputs (B fixed to 1 — one table per call):
      x  [1, T, H] float32   preprocessed features (all rows; H is dynamic)
      y  [1, S]    float32   TRAINING labels only (S = train_size <= T)
    Output:
      logits [1, T, C]       Rows < S (train rows) carry the model's own values
                             and are ignored by the engine; rows >= S are the
                             test predictions.
      - CLASSIFICATION: C = max_classes (class logits).
      - REGRESSION:     C = 1 — a single real-valued POINT ESTIMATE per row.

    train_size is implicit as ``S = y.shape[1]``; there is no train_size / cat_mask
    / d input (TabICL has no categorical path and, with feature grouping, does not
    accept ``d``). See FEASIBILITY notes for the engine integration contract.

    Regression target-space contract (self-contained graph)
    -------------------------------------------------------
    The engine feeds RAW training targets in ``y`` and reads the [1, T, 1] output
    directly as the real-valued prediction — the engine does NOT z-score the
    target and does NOT inverse-transform the output. This mirrors TabICL's own
    sklearn ``TabICLRegressor`` pipeline, which is reproduced INSIDE the graph:

      1. StandardScaler on the train targets:  y_z = (y - mean_y) / std_y
         (population std, ddof=0, with a zero-variance guard std=0 -> 1), exactly
         matching ``sklearn.preprocessing.StandardScaler``.
      2. The whole model sees ``y_z`` (target-aware column embedding AND the ICL
         y-encoder), so quantiles come out in z-scored space.
      3. Point estimate = MEAN over the 999 quantiles. ``TabICLRegressor.predict``
         defaults to ``output_type="mean"`` == ``dist.quantiles.mean(-1)``; the
         monotonic ``sort`` inside ``QuantileToDistribution`` is a permutation, so
         the mean is invariant and no in-graph sort is needed.
      4. Inverse StandardScaler:  yhat = point_z * std_y + mean_y  → RAW space.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model
        self.regression = (model.max_classes == 0)

    def forward(self, x, y):
        if not self.regression:
            emb = self.m.col_embedder(x, y_train=y, d=None, embed_with_test=False)
            reps = self.m.row_interactor(emb, d=None)
            # _icl_predictions returns full [1, T, C]; the public icl_predictor()
            # training branch would slice to test rows only — we keep all T so the
            # graph output matches the engine's logits[1,T,C] contract.
            return self.m.icl_predictor._icl_predictions(reps, y)

        # --- Regression: self-contained RAW-in / RAW-out point estimate. --------
        # StandardScaler(ddof=0) on the raw train targets, baked into the graph.
        mean_y = y.mean(dim=1, keepdim=True)
        var_y = ((y - mean_y) ** 2).mean(dim=1, keepdim=True)
        std_y = torch.sqrt(var_y)
        std_y = torch.where(std_y > 0.0, std_y, torch.ones_like(std_y))  # 0 -> 1 guard
        y_z = (y - mean_y) / std_y

        emb = self.m.col_embedder(x, y_train=y_z, d=None, embed_with_test=False)
        reps = self.m.row_interactor(emb, d=None)
        quantiles = self.m.icl_predictor._icl_predictions(reps, y_z)  # [1, T, num_quantiles]

        # Point estimate = mean over quantiles (sklearn output_type="mean"),
        # invariant to the monotonic sort, then inverse StandardScaler -> RAW.
        point_z = quantiles.mean(dim=-1, keepdim=True)  # [1, T, 1]
        return point_z * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)  # [1, T, 1]


def build_model(task: str, model_kwargs: dict, seed: int = 0):
    """Random-weight TabICL at the given dims (train mode). No checkpoint bytes."""
    from tabicl._model.tabicl import TabICL

    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification|regression, got {task!r}")
    apply()
    torch.manual_seed(seed)
    kwargs = dict(model_kwargs)
    if task == "regression":
        kwargs["max_classes"] = 0
    model = TabICL(**kwargs)
    model.train()  # training-path branches are the exportable ones
    return model
