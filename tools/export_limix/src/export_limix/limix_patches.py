"""Export-friendly patches for StableAI's LimiX (upstream code Apache-2.0).

We DO NOT copy or edit the upstream source. LimiX ships no packaging metadata,
so it is pinned as the ``vendor/limix`` git submodule (the convention
``vendor/tabfm`` already uses) and put on ``sys.path`` here; patches are applied
at RUNTIME, as in ``tools/export_tabicl`` / ``tools/export_orion_bix``.

Upstream's ``FeaturesTransformer.forward(x, y, eval_pos, task_type=...)`` already
takes the shape the engine feeds — ``x[B, T, F]`` plus a label prefix and a split
point — and does its own dict-wrapping internally, so the ``(x, y)``-only ONNX
contract used by TabPFN / TabICL / Orion-BiX applies unchanged and no engine work
is needed.

Patches (each mathematically identical to upstream on the inference path):

  1. The two ``if torch.isnan(...).any(): raise`` guards in
     ``FeaturesTransformer.forward`` — data-dependent Python branches that abort
     the trace. Both are pure input validation ("please add a NanEncoder in the
     encoder"), and the released config enables ``nan_handling_enabled`` /
     ``nan_handling_y_encoder`` so the encoders already replace NaNs. Same class
     as TabPFN's ``_do_encoder_nan_check`` patch. Implemented by swapping
     ``torch.isnan`` *inside model.transformer only*, so the surrounding
     100-line ``forward`` is untouched.

  2. ``FeaturesTransformer.mixed_y_embedding`` — upstream supports a MIXED batch
     of classification and regression targets, splitting the flattened labels by
     ``y_type`` with boolean-mask indexing (``idx[y_type_flat == 0]``) and then
     branching on ``len(idx_cls) > 0``. Both the split widths and the branches
     are data-dependent, and the trace dies with
     ``GuardOnDataDependentSymNode: Could not extract specialized integer``.

     ``forward`` itself builds ``y_type`` as ``zeros_like`` (cls) or
     ``ones_like`` (reg) from its ``task_type`` argument, so within one exported
     graph the tensor is UNIFORM and exactly one branch is ever live — the split
     is the identity. The replacement calls the single relevant encoder directly.
     It keeps upstream's float16 round-trip on the embedding (upstream allocates
     the output buffer as ``torch.empty(..., dtype=torch.float16)``), so the
     numerics match rather than silently gaining precision.
"""

from __future__ import annotations

import pathlib
import sys

import torch

_APPLIED = False

#: vendor/limix — the pinned upstream submodule (repo root / vendor / limix).
# limix_patches.py -> export_limix -> src -> tool -> tools -> repo root
VENDOR = pathlib.Path(__file__).resolve().parents[4] / "vendor" / "limix"


def _ensure_path() -> None:
    """Put the pinned upstream submodule on sys.path.

    LimiX imports its own modules as top-level packages (``from model.transformer
    import ...``), so the submodule ROOT is what goes on the path.
    """
    if not (VENDOR / "model" / "transformer.py").exists():
        raise RuntimeError(
            f"vendor/limix is missing or empty at {VENDOR}. Run:\n"
            "    git submodule update --init vendor/limix"
        )
    p = str(VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)


class _IsNaNResult:
    """Proxy for a ``torch.isnan(t)`` result whose ``.any()`` is statically False.

    ``forward`` uses ``torch.isnan`` for two different purposes: building the
    input mask (``torch.isnan(x).to(torch.int32)``) and the two validation
    guards (``torch.isnan(embedded_y).any()``). Every attribute except ``any``
    forwards to the real tensor, so the mask keeps its exact upstream value
    while the guards stop being data-dependent branches.
    """

    __slots__ = ("_t",)

    def __init__(self, t):
        object.__setattr__(self, "_t", t)

    def any(self):  # noqa: A003 - mirrors the tensor API
        return False

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_t"), name)


class _TorchShim:
    """Proxy for the ``torch`` module that swaps out a single function.

    Attribute access falls through to the real module, so only ``torch.isnan``
    as referenced from ``model.transformer`` changes behaviour; every other call
    (``torch.cat``, ``torch.nan``, ``torch.zeros_like``, ...) is untouched.
    """

    def __init__(self, real, isnan):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_isnan", isnan)

    def __getattr__(self, name):
        if name == "isnan":
            return object.__getattribute__(self, "_isnan")
        return getattr(object.__getattribute__(self, "_real"), name)


def apply() -> None:
    """Idempotently install the export patches on the vendored LimiX."""
    global _APPLIED
    if _APPLIED:
        return
    _ensure_path()

    import model.transformer as tmod

    # --- Patch 1: neutralize the two NaN validation guards -----------------
    # NOTE: transformer.forward also calls torch.isnan to BUILD the input mask
    # (`{'mask': torch.isnan(x)...}`), so the shim must only affect the guards.
    # It does: the mask call is `torch.isnan(x).to(...)`, which would fail on the
    # stand-in — so we keep the real isnan and only special-case `.any()` usage
    # by returning a real tensor that reports no NaNs for the guard operands.
    real_isnan = torch.isnan

    def _isnan_for_guards(t):
        return _IsNaNResult(real_isnan(t))

    tmod.torch = _TorchShim(tmod.torch, _isnan_for_guards)

    # --- Patch 2: single-task y embedding (no data-dependent split) --------
    def _mixed_y_embedding(self, y, y_type, eval_pos):
        y_data = y["data"]
        batch_size, seq_len, _y_num = y_data.shape
        task = getattr(self, "_export_task", "cls")
        encoder = self.cls_y_encoder if task == "cls" else self.reg_y_encoder
        emb = encoder({"data": y_data, "eval_pos": eval_pos})["data"]
        # Upstream writes into a float16 buffer; preserve that exactly.
        flat = emb.reshape(-1, self.embed_dim).to(torch.float16)
        return flat.reshape(batch_size, seq_len, self.embed_dim)

    tmod.FeaturesTransformer.mixed_y_embedding = _mixed_y_embedding

    # --- Patch 3: forward with a functional test-label mask ----------------
    def _forward(self, x, y, eval_pos, y_type=None, task_type="cls",
                 calculate_sample_attention=False, calculate_feature_attention=False,
                 **kwargs):
        batch_size, seq_len, num_feature = x.shape
        x = {"data": x, "mask": torch.isnan(x).to(torch.int32).to(x.device)}
        y = {"data": y}

        # Upstream: `feature_to_add = num_feature % fpg; if feature_to_add > 0: cat zeros`. A Python
        # branch on a symbolic width is decided AT TRACE TIME: traced at an even H it records "no
        # padding" and every odd-width table then fails the grouping reshape (a blocker the README
        # never listed; its three shapes all have even H). Instead: always append the maximum
        # padding (fpg - 1 zero columns, a static width) and slice to the grouped width
        # ceil(H / fpg) * fpg, which is H itself when H divides evenly. Same values as upstream.
        fpg = self.features_per_group
        if fpg > 1:
            grouped_width = ((num_feature + fpg - 1) // fpg) * fpg
            for k in x:
                zeros = torch.zeros_like(x[k][:, :, :1]).expand(-1, -1, fpg - 1)
                x[k] = torch.cat((x[k], zeros), dim=-1)[:, :, :grouped_width]
        for k in x:
            x[k] = x[k].reshape(batch_size, seq_len,
                                x[k].shape[2] // self.features_per_group,
                                self.features_per_group)
        x["eval_pos"] = eval_pos
        preprocessed_x = self.x_preprocess(x)
        preprocessed_x = self.process_4_x(preprocessed_x)
        x_emb_result = self.encoder_x(preprocessed_x)["data"]

        for k in y:
            y[k] = y[k].unsqueeze(-1)
            # Upstream pads y out to T rows with `nan * zeros(B, T - S, 1)` under a Python
            # `if S < T`. Traced, the pad length T - S is concretised at the example (20 - 12 = 8)
            # and baked in as a CONSTANT; the later equality "padded y has T rows" then makes
            # torch.export rewrite T as S + 8 everywhere, including the feature-grouping reshape
            # (ORT: "cannot reshape {1,20,6} to {1,23,3,2}", 23 = 20 + (15 - 12)). The pad is
            # taken instead as the TAIL of a T-long tensor, so its length is T - S by
            # construction, and S == T gives an empty tail rather than a different branch.
            nan_rows = torch.full_like(x["data"][:, :, :1, 0], float("nan")).to(y[k].dtype)
            y[k] = torch.cat((y[k], nan_rows[:, y[k].shape[1]:, :]), dim=1)
        # Upstream: y["data"][:, eval_pos:] = torch.nan  (in-place, shape-baking).
        rows = torch.arange(y["data"].shape[1], device=y["data"].device)
        test_rows = (rows >= eval_pos).reshape(1, -1, 1)
        y["data"] = torch.where(test_rows, torch.full_like(y["data"], float("nan")),
                                y["data"])

        if task_type == "cls":
            y_type = torch.zeros_like(y["data"], device=y["data"].device)
        else:
            y_type = torch.ones_like(y["data"], device=y["data"].device)

        embedded_y = self.mixed_y_embedding(y, y_type=y_type, eval_pos=eval_pos)
        embedded_x = self.add_embeddings(x_emb_result)
        embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

        encoder_out = self.transformer_encoder(
            embedded_all, feature_atten_mask=None, eval_pos=eval_pos, **kwargs)[0]
        encoder_out = self.encoder_out_norm(encoder_out)

        # `_decode_from_row` is an export-time switch, default eval_pos (upstream's behaviour: decode the
        # test rows only). 0 decodes EVERY row, which is how the fitted-value routes are measured.
        first = getattr(self, "_decode_from_row", None)
        first = eval_pos if first is None else first
        test_encoder_out = encoder_out[:, first:, -1]
        test_y_type = y_type[:, first:]
        cls_output, reg_output = self.y_decoder(test_encoder_out, test_y_type)
        return cls_output if task_type == "cls" else reg_output

    tmod.FeaturesTransformer.forward = _forward

    # --- Patch 5: y_decoder without the data-dependent y_type split --------
    # Same family as patch 2. Upstream flattens the test rows, selects the classification and
    # regression rows with boolean-mask indexing (`idx[flat_test_y_type == 0]`) and reshapes each
    # group back to (-1, seq_len, emb). Traced, the selection becomes GatherND with the row count
    # baked in (ORT: "invalid index found, index = 8", 8 = T - S at the example), so the graph
    # only works at the traced T - S. `forward` builds y_type uniform, so the split is the
    # identity and only one decoder is live; the other output is never read. The upstream function
    # is kept as `_upstream_y_decoder` so a test can prove the replacement exact.
    tmod.FeaturesTransformer._upstream_y_decoder = tmod.FeaturesTransformer.y_decoder

    def _y_decoder(self, test_encoder_out, test_y_type):
        if getattr(self, "_export_task", "cls") == "cls":
            return self.cls_y_decoder(test_encoder_out), None
        return None, self.reg_y_decoder(test_encoder_out)

    tmod.FeaturesTransformer.y_decoder = _y_decoder

    # --- Patch 4: branchless NaN/Inf imputation ----------------------------
    import model.encoders as emod

    def _nan_encoder_forward(self, input):
        x = input[self.in_keys[0]]
        eval_pos = input["eval_pos"]
        mean_value, _ = emod.calc_mean(x[:, :eval_pos, :], dim=1)

        is_nan = torch.isnan(x)
        is_inf = torch.isinf(x)
        pos_inf = is_inf & (torch.sign(x) == 1)
        neg_inf = is_inf & (torch.sign(x) == -1)

        zeros = torch.zeros_like(x)
        nans_indicator = torch.where(is_nan, torch.full_like(x, self.nan_value), zeros)
        nans_indicator = torch.where(pos_inf, torch.full_like(x, self.inf_value),
                                     nans_indicator)
        nans_indicator = torch.where(neg_inf, torch.full_like(x, self.neg_info_value),
                                     nans_indicator)

        nan_mask = torch.logical_or(is_nan, is_inf)
        x = torch.where(nan_mask, mean_value.unsqueeze(1).expand_as(x), x)

        input[self.in_keys[0]] = x
        input[self.out_key] = nans_indicator
        return input

    emod.NanEncoder.forward = _nan_encoder_forward

    _APPLIED = True


class ExportWrapper(torch.nn.Module):
    """Pins LimiX to a fixed 2-input ONNX signature.

    Inputs (B fixed to 1 -- one table per call):
      x  [1, T, H] float32   features (all rows; H is dynamic)
      y  [1, S]    float32   TRAINING targets only (S = eval_pos <= T): dense class ids, or RAW
                             regression targets
    Output:
      [1, T, C]              classification: class logits (C = num_classes, 10). Regression: C = 1,
                             a RAW-space point estimate. EVERY row is a prediction; rows < S are
                             in-context fitted values.

    The split is positional (S = len(y)), the same (x, y)-only contract as TabPFN / TabICL.

    Fitted values. The engine decodes every row and surfaces the context rows as `is_training`
    values. Upstream predicts only the rows after the context, and this wrapper used to pad zeros for
    the rest, so every context row decoded to one constant. Two routes to real values were MEASURED on
    the real LimiX-2M weights (spike, 2026-10-02):

        route                        5-class context acc   linear-regression context R2
        decode all rows, one pass          0.111                     0.500
        context presented again            1.000                     0.993

    so the context is presented a second time as queries: the sequence `[context ; all rows]` with
    eval_pos = S, whose rows >= S are every row of the table. Query rows are unchanged by it (single
    pass vs this route: identical to 5e-7 on every dataset tried, `seq_attn_isolated` keeps queries
    independent). It costs S extra rows per call. The values are in-context consistency checks, not
    out-of-sample estimates, as for every other model here.

    Regression target. The model does NOT normalise its target: every upstream example z-scores y,
    predicts, and inverts (README, examples/demo_regression.py), and on a target with a scale of 79 the
    un-normalised call scores R2 -4.2 where a scale-3 target scores 0.99. The engine feeds RAW targets
    and reads the output as raw, so the standardisation (population std, numpy's default, as in the
    upstream examples; a scale at rounding level -> 1) and its inverse live in the graph.
    """

    def __init__(self, model, task: str = "classification"):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.m = model
        self.task = task
        # Patches 2 and 5 read this to pick the single live y encoder / decoder.
        model._export_task = "cls" if task == "classification" else "reg"

    def forward(self, x, y):
        s = y.shape[1]
        # Tell the exporter S <= T. Without it every per-layer slice `x[:, :eval_pos]` / `x[:, eval_pos:]`
        # yields a Min(S, T) expression that sympy re-simplifies in every layer, and export time explodes
        # with depth: on the small config 1/2/4 layers took 30/20/32 s, 6 layers 79-126 s, 8 layers more
        # than 480 s, and the real model has 12 (the original code did not finish 4 layers in 15 minutes).
        # With the assertion the slices are exact and 4 and 6 layers cost the same (28 s).
        torch._check(s <= x.shape[1])
        if self.task == "regression":
            mean_y = y.mean(dim=1, keepdim=True)
            std_y = ((y - mean_y) ** 2).mean(dim=1, keepdim=True).sqrt()
            std_y = torch.where(std_y < 1e-6, torch.ones_like(std_y), std_y)
            y_in = (y - mean_y) / std_y
        else:
            y_in = y
        # [context ; all rows], eval_pos = S: rows >= S are every row of the table.
        table = torch.cat([x[:, :s], x], dim=1)
        out = self.m(table, y_in, eval_pos=s,
                     task_type="cls" if self.task == "classification" else "reg")
        if out.dim() == 2:  # regression point estimate [1, T] -> [1, T, 1]
            out = out.unsqueeze(-1)
        if self.task == "regression":
            out = out * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)
        return out


def build_model(config: dict, seed: int = 0):
    """Random-weight LimiX at the given config. No checkpoint bytes anywhere."""
    apply()
    from utils.loading import build_model as _build

    torch.manual_seed(seed)
    model = _build(dict(config))
    return model.eval()


def load_real_config(ckpt_path: str) -> dict:
    """Read the architecture config embedded in a released LimiX checkpoint."""
    apply()
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return obj["config"]
