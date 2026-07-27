"""Export-friendly monkeypatches for Lexsi Labs' Orion-BiX (MIT).

We DO NOT copy or edit the ``orion_bix`` package — it is pinned by git sha in
``pyproject.toml`` and patched at RUNTIME, the same approach as
``tools/export_tabicl``. Orion-BiX is a TabICL descendant (same
``SkippableLinear`` / ``ColEmbedding`` / ``InducedSelfAttentionBlock`` /
``_icl_predictions`` names and the same ``skip_value`` convention), so the patch
surface is a subset of the TabICL one.

Each patch is mathematically identical to upstream on the inference path; it
only removes a data-dependent Python branch so the model can go through
``torch.onnx.export(dynamo=True, opset=18)``.

**Which modules are actually on the shipped path.** Despite the model's name,
the released ``Orion-BiX-v1.1.ckpt`` embeds
``col/row/icl_attention_type = "standard"`` and ``*_feature_map = "identity"``
(read from the checkpoint's own ``config``). The traced path is therefore

    ColEmbedding -> SetTransformer(InducedSelfAttentionBlock)
                 -> RowInteraction(Encoder(MultiheadAttentionBlock) + RoPE)
                 -> ICLearning(Encoder(MultiheadAttentionBlock))

``BiAxialAttention`` and ``LinearAttentionBlock`` are NOT instantiated by the
released weights, so their skip-branches need no patch. ``configs.real()``
asserts this so a future checkpoint that flips the flag fails loudly instead of
silently exporting a different architecture.

Like TabICL, Orion-BiX's modules dispatch on ``self.training``: the
*training-mode* path is the clean single-forward one, while the eval path routes
through ``InferenceManager`` (chunking, memory probing, Python loops). The
wrapper therefore puts the model in ``train()`` — with dropout 0.0 in the
released config, that is numerically the same computation — and calls the stage
modules directly.

Patches:
  1. ``SkippableLinear.forward`` — ``if skip_mask.any(): out[mask] = skip`` is a
     data-dependent branch. Rewritten branchless with ``torch.where``.
  2. ``InducedSelfAttentionBlock.forward`` — same skip-branch pattern around the
     ISAB; rewritten compute-all-then-mask. Skipped (all-``-100``) CLS columns
     are overwritten with the sentinel afterwards, so the result is identical;
     running attention over a constant column is wasted work, not wrong work
     (LayerNorm of a constant vector is 0, never NaN).

After ``apply()`` the model exports with signature ``x[1,T,H] f32, y[1,S] f32``
(S = train_size; y holds ONLY the training labels) and output
``logits[1,T,C]`` with C = max_classes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_APPLIED = False


def apply() -> None:
    """Idempotently install the two export monkeypatches on ``orion_bix``."""
    global _APPLIED
    if _APPLIED:
        return

    import orion_bix.model.layers as layers

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

    # --- Patch 3: RotaryEmbedding — never cache freqs ----------------------
    # ``RotaryEmbedding.forward`` memoizes its frequency table into a
    # non-persistent buffer (``tmp_store("cached_freqs", ...)``) on first call.
    # Under fake-tensor tracing that buffer becomes a FakeTensor with no
    # storage, and serialization dies with "Cannot take content out from the
    # FakeTensor ('...rope.cached_freqs')".
    #
    # It is also a *correctness* trap: the cache is keyed on sequence length, so
    # a graph traced at the export example would bake that example's row count
    # into the rotary table. Forcing recomputation keeps the table a function of
    # the dynamic row dim. Recomputing is cheap (an outer product) and is
    # numerically identical — upstream stores exactly what it recomputes.
    from orion_bix.model.rope import RotaryEmbedding

    _orig_rope_init = RotaryEmbedding.__init__

    def rope_init(self, *args, **kwargs):
        kwargs["cache_if_possible"] = False
        _orig_rope_init(self, *args, **kwargs)

    RotaryEmbedding.__init__ = rope_init

    _APPLIED = True


class ExportWrapper(torch.nn.Module):
    """Pins Orion-BiX's training-path stages to a fixed 2-input ONNX signature.

    Inputs (B fixed to 1 — one table per call):
      x  [1, T, H] float32   preprocessed features (all rows; H is dynamic)
      y  [1, S]    float32   TRAINING labels only (S = train_size <= T)
    Output:
      logits [1, T, C]       C = max_classes. Rows < S (train rows) carry the
                             model's own values and are ignored by the engine;
                             rows >= S are the test predictions.

    ``train_size`` is implicit as ``S = y.shape[1]`` — there is no train_size /
    cat_mask / d input. This is exactly the ``(x, y)``-only contract the engine
    already drives for TabPFN and TabICL (y-as-train-prefix inference), so the
    built-in manifest needs no new tensor-contract entries.

    **Classification only.** Orion-BiX ships no regressor (upstream has
    ``sklearn/classifier.py`` and no regression head), so the model spec
    declares ``capabilities: ["classify"]`` and there is no regression branch
    here.

    Raw logits, not probabilities: upstream ``_predict_standard`` slices to
    ``[:, train_size:, :num_classes]`` using ``len(torch.unique(y_train[0]))`` —
    a data-dependent host sync — and then applies a temperature softmax. Both
    are engine responsibilities (the engine knows the class count and consumes
    logits), so the graph stops at ``_icl_predictions``, which returns the full
    ``[1, T, max_classes]`` tensor.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y):
        # train_size is symbolic (y's dynamic dim), so the slice bounds inside
        # the stages stay dynamic rather than baking the export example.
        emb = self.m.col_embedder(x, d=None, train_size=y.shape[1])
        reps = self.m.row_interactor(emb, d=None)
        return self.m.icl_predictor._icl_predictions(reps, y)


def build_model(model_kwargs: dict, seed: int = 0):
    """Random-weight OrionBix at the given dims (train mode). No checkpoint bytes."""
    from orion_bix.model.orion_bix import OrionBix

    apply()
    torch.manual_seed(seed)
    model = OrionBix(**model_kwargs)
    model.train()  # training-path branches are the exportable ones
    return model
