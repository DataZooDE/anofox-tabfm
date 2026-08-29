"""Export-friendly monkeypatches for Lexsi Labs' Orion-MSP (MIT).

We DO NOT copy or edit the ``orion_msp`` package — it is pinned by git sha in
``pyproject.toml`` and patched at RUNTIME, the same approach as
``tools/export_tabicl``. Orion-MSP is a TabICL descendant (same
``SkippableLinear`` / ``ColEmbedding`` / ``InducedSelfAttentionBlock`` /
``_icl_predictions`` names and the same ``skip_value`` convention), so the patch
surface is a subset of the TabICL one.

Each patch is mathematically identical to upstream on the inference path; it
only removes a data-dependent Python branch so the model can go through
``torch.onnx.export(dynamo=True, opset=18)``.

**Which modules are actually on the shipped path.** Despite the model's name,
the released ``Orion-MSP-v1.1.ckpt`` embeds
``col/row/icl_attention_type = "standard"`` and ``*_feature_map = "identity"``
(read from the checkpoint's own ``config``). The traced path is therefore

    ColEmbedding -> SetTransformer(InducedSelfAttentionBlock)
                 -> RowInteraction(Encoder(MultiheadAttentionBlock) + RoPE)
                 -> ICLearning(Encoder(MultiheadAttentionBlock))

``BiAxialAttention`` and ``LinearAttentionBlock`` are NOT instantiated by the
released weights, so their skip-branches need no patch. ``configs.real()``
asserts this so a future checkpoint that flips the flag fails loudly instead of
silently exporting a different architecture.

Like TabICL, Orion-MSP's modules dispatch on ``self.training``: the
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

# Frozen random-link scores for the block-sparse row mask (see Patch 4 in
# `apply`). MAX_L bounds the row-stage sequence length: num_special + the number
# of feature GROUPS, so it is a feature-axis bound, not a row-count one. 1024
# covers far more grouped features than the 512-feature size regime allows.
# The row stage's sequence length is num_special + the number of feature GROUPS,
# so this bounds the FEATURE axis, not the row count: at the 512-feature size
# regime and features_per_group=2 that is 256 groups + 8 specials = 264. 512
# leaves headroom and keeps the baked table at 1 MB (f32); 1024 would be 4 MB in
# a graph that is otherwise ~1.5 MB.
MAX_L = 512
RANDOM_LINK_SEED = 20260829
_LINK_SCORES = None


def _random_link_scores(device=None, dtype=torch.float32) -> torch.Tensor:
    """[MAX_L, MAX_L] fixed random scores; the top-k per row are the links.

    Drawn once from a fixed seed on CPU and cached, so every export and every
    inference on every machine sees the same mask. Storing SCORES rather than a
    boolean table lets the caller pick exactly `num_random` links per row with a
    traceable `topk`, matching upstream's "num_random distinct keys per
    non-special query" without upstream's per-call `randperm`.
    """
    global _LINK_SCORES
    if _LINK_SCORES is None:
        g = torch.Generator().manual_seed(RANDOM_LINK_SEED)
        _LINK_SCORES = torch.rand((MAX_L, MAX_L), generator=g)
    scores = _LINK_SCORES.to(dtype)
    return scores.to(device) if device is not None else scores


def apply() -> None:
    """Idempotently install the two export monkeypatches on ``orion_msp``."""
    global _APPLIED
    if _APPLIED:
        return

    import orion_msp.model.layers as layers

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
    from orion_msp.model.rope import RotaryEmbedding

    _orig_rope_init = RotaryEmbedding.__init__

    def rope_init(self, *args, **kwargs):
        kwargs["cache_if_possible"] = False
        _orig_rope_init(self, *args, **kwargs)

    RotaryEmbedding.__init__ = rope_init

    # --- Patch 4: block-sparse row mask — frozen, seeded, sliced ------------
    # Specific to Orion-MSP, and the only patch here that is not a pure refactor
    # of upstream. `_build_block_sparse_mask` draws the BigBird-style random
    # links with a Python loop over a tensor and a `torch.randperm` PER CALL:
    #
    #     for i in rng_idx:
    #         choices = torch.randperm(L - num_special)[:num_random] + num_special
    #         mask[i, choices] = 0.0
    #
    # Two problems. The loop is untraceable once L is symbolic, and the randperm
    # makes upstream inference NON-DETERMINISTIC — the same table scored twice
    # gets two different attention masks. The released checkpoint sets
    # `row_num_random = 2`, so this branch is live and cannot just be dropped.
    #
    # The fix mirrors what tools/export_tabpfn does for TabPFN's runtime `randn`
    # column-embedding table: draw ONCE from a fixed seed into a [MAX_L, MAX_L]
    # score table, then take the top-`num_random` per row and slice by the
    # runtime L. Traceable, deterministic across runs and machines, and
    # structurally faithful — each non-special query keeps exactly `num_random`
    # extra links, a fixed sample rather than a fresh one.
    #
    # Consequence, stated plainly: the graph reproduces ONE particular upstream
    # draw rather than any specific one. Parity is therefore measured against
    # upstream WITH this patch installed; comparing against unpatched upstream
    # would be comparing two different random masks and would mean nothing.
    # See docs/REAL_MODELS.md.
    import orion_msp.model.interaction as interaction

    def frozen_block_sparse_mask(seq_len, num_special, window, num_random=0,
                                 device=None, dtype=torch.float32,
                                 return_bool=False):
        L = seq_len
        neg_inf = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
        zeros = torch.zeros((L, L), device=device, dtype=dtype)
        allow = torch.zeros((L, L), device=device, dtype=torch.bool)

        idx = torch.arange(L, device=device)
        if num_special > 0:
            # CLS + GLOBAL tokens are fully connected both ways.
            special = idx < num_special
            allow = allow | special.unsqueeze(0) | special.unsqueeze(1)

        # NOTE: `window` is deliberately NOT applied, because upstream does not
        # apply it either. Its sliding-window branch reads:
        #
        #     local = (dist <= window).to(mask.dtype) * 0.0
        #     mask  = torch.where(mask.isfinite(), mask, local + float("-inf"))
        #
        # `x * 0.0` makes `local` all zeros regardless of `dist`, so
        # `local + -inf` is -inf everywhere and the `where` keeps `mask`
        # unchanged: the window opens nothing. The effective upstream mask is
        # specials + random links + diagonal, which is what the released weights
        # were trained and published against.
        #
        # Honouring the window here would be "fixing" the architecture out from
        # under the checkpoint -- measured at L=24/num_special=8/window=3 it
        # opens 14 keys per query where upstream opens 11. So the parameter is
        # accepted and ignored, exactly as upstream effectively ignores it.
        # (Pinned against the unpatched function in tests/test_export.py.)

        if num_random > 0:
            scores = _random_link_scores(device=device, dtype=dtype)[:L, :L].clone()
            # Never spend a random link on a special column: those are already
            # connected, which is what upstream's `+ num_special` offset ensures.
            scores[:, :num_special] = -float("inf")
            k = min(int(num_random), max(L - int(num_special), 0))
            if k > 0:
                picks = scores.topk(k, dim=-1).indices
                allow = allow.scatter(1, picks, True)
        allow = allow | torch.eye(L, device=device, dtype=torch.bool)

        if return_bool:
            return ~allow          # True == DISALLOWED, matching upstream
        return torch.where(allow, zeros, neg_inf)

    interaction._build_block_sparse_mask = frozen_block_sparse_mask

    # Materialise the table NOW, outside any trace. It is cached lazily on first
    # use, and first use would otherwise land inside torch.export -- where the
    # `generator=` argument to torch.rand is not a traceable value and the export
    # dies with "argument of type: <class 'torch._C.Generator'>".
    _random_link_scores()

    _APPLIED = True


class ExportWrapper(torch.nn.Module):
    """Pins Orion-MSP's training-path stages to a fixed 2-input ONNX signature.

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

    **Classification only.** Orion-MSP ships no regressor (upstream has
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
    """Random-weight OrionMSP at the given dims (train mode). No checkpoint bytes."""
    import inspect

    from orion_msp.model.orion_msp import OrionMSP

    apply()
    torch.manual_seed(seed)
    # The checkpoint's `config` block carries more keys than the constructor
    # accepts (features_per_group, feature_pos_emb, scale_combine_method,
    # num_memory_heads, num_thinking_tokens, use_memory_gating are consumed
    # elsewhere or defaulted). configs.py records the config VERBATIM so it can
    # be diffed against the checkpoint; the filtering happens here, the same way
    # tools/export_tabpfn filters against its Config dataclass fields.
    accepted = set(inspect.signature(OrionMSP.__init__).parameters) - {"self"}
    model = OrionMSP(**{k: v for k, v in model_kwargs.items() if k in accepted})
    model.train()  # training-path branches are the exportable ones
    return model
