"""Export-time patches for upstream LimiX-2 (model/v2_0). SPIKE code.

Upstream is cloned OUTSIDE the repo (LIMIX2_SRC); nothing of it, and no weight, is committed.
Every patch is a runtime monkeypatch of a bound method, never an edit of upstream's tree.
"""
from __future__ import annotations

import glob
import os
import sys
import types

import torch

LIMIX2_SRC = os.environ.get("LIMIX2_SRC", os.path.expanduser("~/.cache/limix2-spike/src"))
CKPT_GLOB = os.path.expanduser("~/.cache/huggingface/hub/models--stable-ai--LimiX-2/snapshots/*/LimiX-2.ckpt")

# Upstream draws the feature-positional embedding with an UNSEEDED torch.randn((G, E//4)) on every call
# (transformer.py:768). randn has the prefix property (verified: randn((3,64)) == randn((8,64))[:3] under one
# seed), so one seeded table of GMAX groups reproduces upstream's draw for ANY G <= GMAX under that seed.
GMAX = 512  # 1024 features


class _NvtxStub(types.ModuleType):
    """upstream does `import nvtx` at module top and wraps every stage in `with nvtx.annotate(...)` / decorators;
    Dynamo cannot trace nvtx's cython Domain.__call__. A no-op stand-in (context manager AND decorator) removes
    both the dependency and the graph break."""

    class annotate:  # noqa: N801 - mirrors nvtx.annotate
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *e): return False
        def __call__(self, fn): return fn


def _ensure_path():
    if not isinstance(sys.modules.get("nvtx"), _NvtxStub):
        sys.modules["nvtx"] = _NvtxStub("nvtx")
    if LIMIX2_SRC not in sys.path:
        sys.path.insert(0, LIMIX2_SRC)


def load_upstream(num_layers: int | None = None, seed: int = 0):
    """Build the v2_0 model from the checkpoint's own embedded config. Optionally truncated (small-dims stage)."""
    import copy
    _ensure_path()
    obj = torch.load(glob.glob(CKPT_GLOB)[0], map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(obj["config"])
    cfg["mask_prediction"] = False
    sd = obj["state_dict"]
    if num_layers is not None:
        msc = cfg["model_structure_config"]
        msc["layers"] = msc["layers"][:num_layers]
        msc["nlayers"] = num_layers
        cfg["nlayers"] = num_layers
        for k in ("feature_emb_layer", "reg_y_emb_layer", "cls_y_emb_layer"):
            cfg[k] = num_layers - 1
    from model.v2_0.autobatch import AutobatchConfig
    AutobatchConfig.ENABLE_AUTOBATCH = False
    from model.v2_0.loading import build_model
    m = build_model(cfg).eval()
    if num_layers is None:
        m.load_state_dict(sd, strict=True)
    else:  # layers.{i}.* of the first num_layers + all non-layer tensors
        keep = {k: v for k, v in sd.items() if not _is_dropped_layer(k, num_layers)}
        m.load_state_dict(keep, strict=False)
    return m, cfg


def _is_dropped_layer(key: str, n: int) -> bool:
    parts = key.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1]) >= n
    return False


_CLASSES_PATCHED = False


def _patch_classes():
    """Class-level patches (DStI attention, softmax scaling). Idempotent."""
    global _CLASSES_PATCHED
    if _CLASSES_PATCHED:
        return
    import torch.nn.functional as F
    from model.v2_0 import decoupled_structural_task_attention as dsti
    from model.v2_0 import softmax_scaling_mlp as ssm

    def ssm_forward(self, q_BSHD, n):
        # upstream: torch.tensor(n / 1.0). n is the feature-GROUP count + y tokens: a shape-dependent temperature,
        # so it must stay symbolic. scalar_tensor of the SymInt keeps it in the graph (parity at a 2nd H proves it).
        logn = torch.log(torch.scalar_tensor(n, dtype=torch.float32, device=q_BSHD.device)).reshape(1, 1).to(q_BSHD.dtype)
        w = ssm._soft_clamp_range(self.scale_linear.weight, self.temp_lower_bound, self.temp_upper_bound)
        logn_scale = (w @ logn.T).squeeze() + 1.0
        bias_scale = self._clamped_direction(self.scale_linear.bias).to(q_BSHD.dtype)
        return q_BSHD * (logn_scale * bias_scale).reshape(1, 1, self.num_heads, 1)

    def dsti_forward(self, x, x_kv=None, *, feature_padding_mask=None, calculate_feature_attention=False, **kw):
        # upstream: shape validation + `if not torch.any(feature_padding_mask): mask = None` (data-dependent branch,
        # only an optimisation for FlashAttention; an all-False mask gives identical numbers).
        fc = x.shape[2] - self.num_y_tokens
        x_tokens, y_tokens = x[:, :, :fc], x[:, :, fc:]
        mask = feature_padding_mask.bool() if feature_padding_mask is not None else None
        x_update = self._sample_split_x_route(x_tokens, y_tokens, mask)
        y_update = self._sample_split_yx_route(y_tokens, x_tokens, mask)
        return torch.cat((x_update, y_update), dim=2), None, None

    def sdpa_passthrough(self, q, k, v, **kw):
        return F.scaled_dot_product_attention(q, k, v, **kw)

    def sdpa_attention(self, query, key, value, attention_mask):
        # upstream wraps in sdpa_context(deterministic) + a 65535-row chunker (memory only, plants a T<=65535 guard)
        o = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask)
        return o.transpose(1, 2)

    ssm.SoftmaxScalingMLP.forward = ssm_forward
    dsti.DecoupledStructuralTaskAttention.forward = dsti_forward
    dsti.DecoupledStructuralTaskAttention._sdpa_attention = sdpa_attention
    _CLASSES_PATCHED = True


def apply_export_patches(m, seed: int = 0):
    """Make one task's forward traceable. Returns the model (patched in place)."""
    G = int(m.feature_positional_embedding.in_features)  # E // 4
    gen = torch.Generator().manual_seed(seed)
    table = torch.randn((GMAX, G), generator=gen)
    m.register_buffer("_pos_table", table, persistent=False)
    fpg = int(m.features_per_group)

    def add_embeddings(self, x, *, generator=None):
        # upstream: randn((x.shape[2], x.shape[3]//4)) unseeded. Fixed seeded table, prefix-sliced to the group count.
        embs = self.feature_positional_embedding(self._pos_table[: x.shape[2]].to(x.dtype))
        return x + embs[None, None]

    def padding_xy(self, x, y):
        # upstream: Python `%` branch on the width + zeros(T - S) with S baked from y's length.
        # Branchless: always append fpg-1 zero columns, slice to the next multiple of fpg. y NaN-pad is a tail slice.
        B, T, H = x["data"].shape
        wide = ((H + fpg - 1) // fpg) * fpg
        for k in x:
            x[k] = torch.cat([x[k], torch.zeros(B, T, fpg - 1, dtype=x[k].dtype, device=x[k].device)], dim=-1)[..., :wide]
        for k in y:
            yk = y[k].unsqueeze(-1)
            nan_rows = torch.full((yk.shape[0], T, yk.shape[2]), float("nan"), dtype=yk.dtype, device=yk.device)
            y[k] = torch.cat([yk, nan_rows[:, yk.shape[1]:, :]], dim=1)
        return x, y, wide - H

    def y_encode(self, y, seq_len, batch_size, eval_pos, y_type, enable_classification_y_pred_temp=None,
                 enable_regression_y_pred_temp=None):
        cls_on = self.enable_classification_y_pred if enable_classification_y_pred_temp is None else enable_classification_y_pred_temp
        reg_on = self.enable_regression_y_pred if enable_regression_y_pred_temp is None else enable_regression_y_pred_temp
        assert cls_on != reg_on, "export is single-task: one graph per task"
        if cls_on:
            d = {k: v for k, v in y.items()}
            d["eval_pos"] = eval_pos
            d["data"] = torch.where((y_type == 0).unsqueeze(2), y["data"], 0)
            emb = self.cls_y_encoder(d)["data"]
        else:
            d = {k: v for k, v in y.items()}
            d["eval_pos"] = eval_pos
            d["data"] = torch.where((y_type == 1).unsqueeze(2), y["data"], 0)
            emb = self.reg_y_encoder(d)["data"]
        return emb.reshape(batch_size, seq_len, self.y_token_k, self.embed_dim)  # NaN raise removed (data-dependent)

    def mask_process_4_x(self, data, x_categorical_mask=None):
        # upstream evaluates randn_like for mask code 4 (never produced at inference). Codes are 0/1 only here.
        data["data"] = torch.where(data["mask"] == 1, float("nan"), data["data"])
        data["mask"] = data["mask"].to(torch.bool)
        return data

    def _encoder(self, x, y, eval_pos, y_type=None, x_categorical_mask=None, real_feature_nums=None,
                 task_type="Classification", feature_positional_embedding_generator=None):
        """upstream _encoder, single task / no categorical mask / no feature imputation, minus: the in-place
        `y['data'][:, eval_pos:] = nan` (functional where), the raising `isnan(embedded_all).any()`, the Python
        list `real_feature_nums` -> torch.tensor, and the autocast branch."""
        B, T, H = x["data"].shape
        x, y, feature_to_add = self.padding_xy(x, y)
        wide = x["data"].shape[2]
        # padding mask: True for real features. Upstream builds it from a Python list; arange < H is identical.
        feature_padding_mask = (torch.arange(wide, device=x["data"].device).unsqueeze(0) < H).expand(B, -1)
        for k in x:
            x[k] = x[k].reshape(B, T, wide // fpg, fpg)
        x["eval_pos"] = eval_pos
        x_emb, real_x = self.x_encoder(x, x_categorical_mask=None, feature_padding_mask=feature_padding_mask)

        rows = torch.arange(T, device=y["data"].device).view(1, T, 1)
        y["data"] = torch.where(rows < eval_pos, y["data"], torch.nan)  # test labels hidden, functionally
        cls = task_type == "Classification"
        y_type = (torch.zeros_like(y["data"]) if cls else torch.ones_like(y["data"])).squeeze(-1)
        embedded_y = self.y_encode(y, T, B, eval_pos, y_type=y_type,
                                   enable_classification_y_pred_temp=cls, enable_regression_y_pred_temp=not cls)
        embedded_x = self.add_embeddings(x_emb)
        embedded_all = torch.cat((embedded_x, embedded_y), dim=2)

        feature_mask = None
        if self.uses_decoupled_feature_attention:
            valid = feature_padding_mask.reshape(B, -1, fpg).any(dim=-1)
            padded_group_mask = (~valid).unsqueeze(1).expand(-1, T, -1)
            x_feature_mask = padded_group_mask
            if self.config_dict.get("enable_feature_attention_mask", False):
                fm = x["mask"].to(torch.bool)
                x_feature_mask = (fm.any(dim=-1) if fm.dim() == 4 else fm) | padded_group_mask
            y_feature_mask = torch.zeros(*x_feature_mask.shape[:-1], self.y_token_k, device=x_feature_mask.device, dtype=torch.bool)
            feature_mask = torch.cat([x_feature_mask, y_feature_mask], dim=-1)
        elif self.config_dict.get("enable_feature_attention_mask", False):
            raise NotImplementedError("non-decoupled feature mask path not exported")
        if self.enable_add_task_info:
            embedded_all = self.add_task_info(embedded_all, y_type=y_type)
        return (embedded_all.to(torch.float32), feature_mask, y_type, real_x, feature_to_add,
                cls, not cls, False)

    _patch_classes()
    m._encoder = types.MethodType(_encoder, m)
    m.add_embeddings = types.MethodType(add_embeddings, m)
    m.padding_xy = types.MethodType(padding_xy, m)
    m.y_encode = types.MethodType(y_encode, m)
    m.mask_process_4_x = types.MethodType(mask_process_4_x, m)
    return m
