"""TabICL split at the support/query boundary: export the context once, reuse it per batch.

**Why.** `tabfm_classify` re-encodes the labelled context on every call, because the exported graph
is one forward pass over support and query concatenated. Measured on `tabicl-v2`, CPU, that pass is
71-80% of a call: 1.398 s fixed + 9.0 ms/query-row at a 64-row context, 5.053 s + 9.7 ms at 375.
Nothing about the support set changes between calls -- it is the training data -- so the cost is an
artifact of where the graph was cut. See DataZooDE/anofox-tabfm#37.

**Where the cut goes.** TabICL's own `InducedSelfAttentionBlock`:

    hidden = multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])
    out    = multihead_attn2(src, hidden, hidden)

`hidden` is derived from the support rows alone, and every row's output is an independent attention
against it. That stacks: block *k*'s `hidden` comes from block *k-1*'s support outputs. So the whole
support-derived state is one `hidden` per column-ISAB block, plus the support representations the
ICL stage reads.

    prepare(x[1,S,H], y[1,S]) -> hidden_0..hidden_{K-1} [1,G+C,num_inds,E], support_reps [1,S,R]
    query(x[1,Q,H], y[1,S], support_reps, hiddens)      -> logits [1,Q,C]

**Two things that are easy to get wrong**, both found by getting them wrong:

* The query half must not go through `ColEmbedding.forward`. That path takes train_size from
  `y_train.shape[1]` and calls `y_train.max()`, which is meaningless for a batch supplying no
  labels, and empty `y` raises.
* `ISAB.forward` skips a column group whose values all equal `skip_value`, via
  `out[~skip_mask] = induced_attention(src[~skip_mask], ...)`. That boolean index flattens the batch
  dims, so a skipped group changes the SHAPE the cached `hidden` must match. The reserved cls-token
  columns are padded with exactly `skip_value` and survive the combined pass only because the
  target-aware step adds `y_emb` to the support rows. A query-only batch adds no `y_emb`, so those
  columns look skippable and G+C silently shrinks -- 104 to 100 on a 100-feature table -- against a
  cache recorded at the larger size. Calling `multihead_attn2` directly inherits the combined pass's
  decision instead of re-deriving it per batch.
"""

from __future__ import annotations

import pathlib

import torch
import torch.nn.functional as F

from .configs import OPSET


def _grouped_features(ce, x: torch.Tensor) -> torch.Tensor:
    """`feature_grouping` + reserved-cls padding + transpose, as `_train_forward_with_feature_group`."""
    g = ce.feature_grouping(x)
    if ce.reserve_cls_tokens > 0:
        g = F.pad(g, (0, 0, ce.reserve_cls_tokens, 0), value=-100.0)
    return g.transpose(1, 2)  # (B, G+C, T, group_size)


def _affine(ce, features: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if ce.affine:
        return features * ce.ln_w(ce.out_w(src)) + ce.ln_b(ce.out_b(src))
    return src


class PrepareWrapper(torch.nn.Module):
    """Everything derivable from the labelled context, computed once per support set."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y):
        ce = self.m.col_embedder
        features = _grouped_features(ce, x)
        src = ce.in_linear(features)
        if ce.target_aware:
            y_exp = y.unsqueeze(1).expand(-1, features.shape[1], -1)
            src = src + (ce.y_encoder(y_exp.float()) if ce.max_classes > 0
                         else ce.y_encoder(y_exp.unsqueeze(-1)))

        hiddens = []
        out = src
        for block in ce.tf_col.blocks:
            *batch_shape, _, d_model = out.shape
            ind = block.ind_vectors.expand(*batch_shape, block.num_inds, d_model)
            # Every row in this graph is a support row, so the whole tensor is the key/value set --
            # which is what `train_size` selects in the combined pass.
            hidden = block.multihead_attn1(ind, out, out)
            hiddens.append(hidden)
            out = block.multihead_attn2(out, hidden, hidden)

        emb = _affine(ce, features, out).transpose(1, 2)
        return (*hiddens, self.m.row_interactor(emb, d=None))


class QueryWrapper(torch.nn.Module):
    """A query batch against a prepared context. The support rows are never revisited."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y, support_reps, hiddens):
        # `hiddens` is one tuple rather than varargs: torch.export flattens *args into a single
        # entry and then rejects a dynamic_shapes spec written per tensor.
        ce = self.m.col_embedder
        features = _grouped_features(ce, x)
        out = ce.in_linear(features)  # no target-aware term: a query batch carries no labels
        for block, hidden in zip(ce.tf_col.blocks, hiddens):
            out = block.multihead_attn2(out, hidden, hidden)
        emb = _affine(ce, features, out).transpose(1, 2)
        joined = torch.cat([support_reps, self.m.row_interactor(emb, d=None)], dim=1)
        return self.m.icl_predictor(joined, y, softmax_temperature=0.9, return_logits=True)


def n_context_tensors(model) -> int:
    """How many `hidden` tensors the prepare graph emits (one per column-ISAB block)."""
    return len(model.col_embedder.tf_col.blocks)


def export_split(model, prepare_path: pathlib.Path, query_path: pathlib.Path, *,
                 dim_train: tuple, dim_features: tuple, example: tuple,
                 opset: int = OPSET) -> tuple[PrepareWrapper, QueryWrapper]:
    """Export the pair. `example` is (T, H, S) as elsewhere; the query half gets T - S rows.

    The cached tensors' group axis (G+C) is a function of H, so it is declared dynamic and tied to
    the query graph's own H: a context prepared for a 100-feature table has to be usable by a query
    graph that recomputes G+C from the same H, and pinning either one bakes the export example's
    width into the pair.
    """
    prep, qry = PrepareWrapper(model).train(), QueryWrapper(model).train()
    t, h, s = example
    q = max(2, t - s)
    xs = torch.randn(1, s, h)
    xq = torch.randn(1, q, h)
    y = (torch.randint(0, model.max_classes, (1, s)).float() if model.max_classes > 0
         else torch.randn(1, s))

    S = torch.export.Dim(dim_train[0], min=dim_train[1], max=dim_train[2])
    Q = torch.export.Dim("query", min=2, max=dim_train[2])
    H = torch.export.Dim(dim_features[0], min=dim_features[1], max=dim_features[2])
    # G+C, the column-group axis the cached tensors carry. Declared in its own right rather than
    # derived: torch.export cannot see that it is a function of H through `feature_grouping`.
    G = torch.export.Dim("groups", min=2, max=4 * dim_features[2])

    prepare_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        prepared = prep(xs, y)
        torch.onnx.export(
            prep, (xs, y), str(prepare_path),
            dynamo=True, dynamic_shapes=({1: S, 2: H}, {1: S}), opset_version=opset,
            input_names=["x", "y"],
            output_names=[f"hidden_{i}" for i in range(len(prepared) - 1)] + ["support_reps"],
            external_data=True, optimize=False,
        )
        torch.onnx.export(
            qry, (xq, y, prepared[-1], tuple(prepared[:-1])), str(query_path),
            dynamo=True,
            dynamic_shapes=({1: Q, 2: H}, {1: S}, {1: S},
                            tuple({1: G} for _ in prepared[:-1])),
            opset_version=opset,
            input_names=(["x", "y", "support_reps"]
                         + [f"hidden_{i}" for i in range(len(prepared) - 1)]),
            output_names=["logits"], external_data=True, optimize=False,
        )
    return prep, qry


def check_split_parity(prepare_path: pathlib.Path, query_path: pathlib.Path, model,
                       shapes) -> float:
    """Worst |logit| difference between the chained graphs and one PyTorch forward.

    `shapes` are (S, Q, H) triples and should differ from the export example, so a baked dimension
    shows up as a failure rather than as a pass.
    """
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    s_prep = ort.InferenceSession(str(prepare_path), opts, providers=["CPUExecutionProvider"])
    s_qry = ort.InferenceSession(str(query_path), opts, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in s_prep.get_outputs()]

    worst = 0.0
    for s, q, h in shapes:
        xs, xq = torch.randn(1, s, h), torch.randn(1, q, h)
        y = (torch.randint(0, model.max_classes, (1, s)).float() if model.max_classes > 0
             else torch.randn(1, s))
        with torch.no_grad():
            base = model(torch.cat([xs, xq], 1), y_train=y, d=None,
                         embed_with_test=False).numpy()
        prepared = s_prep.run(None, {"x": xs.numpy(), "y": y.numpy()})
        feed = {"x": xq.numpy(), "y": y.numpy()}
        feed.update(dict(zip(out_names, prepared)))
        got = s_qry.run(None, feed)[0]
        worst = max(worst, float(np.abs(base - got).max()))
    return worst
