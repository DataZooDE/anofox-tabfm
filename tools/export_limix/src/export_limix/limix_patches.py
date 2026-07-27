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

        feature_to_add = num_feature % self.features_per_group
        if feature_to_add > 0:
            for k in x:
                x[k] = torch.cat(
                    (x[k], torch.zeros(batch_size, seq_len, feature_to_add,
                                       device=x[k].device, dtype=x[k].dtype)),
                    dim=-1,
                )
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
            if y[k].shape[1] < x["data"].shape[1]:
                y[k] = torch.cat(
                    (y[k],
                     torch.nan * torch.zeros(
                         y[k].shape[0], x["data"].shape[1] - y[k].shape[1],
                         y[k].shape[2], device=y[k].device, dtype=y[k].dtype)),
                    dim=1,
                )
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

        test_encoder_out = encoder_out[:, eval_pos:, -1]
        test_y_type = y_type[:, eval_pos:]
        cls_output, reg_output = self.y_decoder(test_encoder_out, test_y_type)
        return cls_output if task_type == "cls" else reg_output

    tmod.FeaturesTransformer.forward = _forward

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

    Inputs (B fixed to 1 — one table per call):
      x  [1, T, H] float32   preprocessed features (all rows; H is dynamic)
      y  [1, S]    float32   TRAINING labels/targets only (S = eval_pos <= T)
    Output:
      logits [1, T, C]       classification: C = num_classes (class logits).
                             regression:     C = 1, a point estimate.
                             Predictions occupy rows >= S; rows < S are zero pad.

    ``eval_pos`` is implicit as ``S = y.shape[1]`` — upstream pads the label
    tensor with NaN out to T and masks the test positions itself, so feeding the
    train prefix is exactly upstream's own inference call. That makes this the
    same ``(x, y)``-only contract the engine already drives for TabPFN, TabICL
    and Orion-BiX.
    """

    def __init__(self, model, task: str = "classification"):
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be classification|regression, got {task!r}")
        self.m = model
        self.task = task
        # Patch 2 reads this to pick the single live y encoder.
        model._export_task = "cls" if task == "classification" else "reg"

    def forward(self, x, y):
        eval_pos = y.shape[1]
        out = self.m(x, y, eval_pos=eval_pos,
                     task_type="cls" if self.task == "classification" else "reg")
        if out.dim() == 2:  # regression point estimate [1, T-S] -> [1, T-S, 1]
            out = out.unsqueeze(-1)
        pad = torch.zeros(out.shape[0], eval_pos, out.shape[2], dtype=out.dtype)
        return torch.cat([pad, out], dim=1)


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
