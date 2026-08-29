"""Export configurations for TabDPT (Layer 6 AI, Apache-2.0).

``real`` carries the dims of the published checkpoint. Unusually for this repo
they are not transcribed by hand from a downloaded file: Layer 6 ship the whole
training config inside the safetensors METADATA (``cfg``), so ``real()`` is a
plain record of what that metadata says, and ``assert_matches_checkpoint`` below
re-reads it and fails loudly if a future release changes shape.

``fixture`` is a tiny random-init model for the committed CI fixture. Only dims
shrink — every structural switch (thinking rows, column-attention layers, the
regression bar distribution) is kept on, so the fixture graph exercises the same
code paths as the real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ExportConfig:
    name: str
    model_kwargs: dict
    # (T, H, S) for the traced example; parity runs at DIFFERENT shapes so the
    # dynamic dims are genuinely exercised.
    example: tuple
    parity_shapes: tuple
    max_classes: int
    parity_tol: float = 1e-3


# Published dims, from `Layer6/TabDPT :: tabdpt1_2.safetensors` metadata["cfg"].
# 647 state_dict tensors. dropout = 0.0 in the release, which is what makes the
# eval/train-mode distinction numerically inert.
_REAL_KWARGS = dict(
    ninp=512, nlayers=32, nhead=8, nhid=512,
    num_features=128, n_out=16,
    regression_bin_count=2048, regression_bin_min=-10, regression_bin_max=10,
    y_encoder_dim=128, num_col_attn_layers=2, n_thinking_rows=64,
    enc_cell_dim=-1, base_len=64, max_len=1048576,
    dropout=0.0, use_flash=False, clip_sigma=8.0,
)

_FIXTURE_KWARGS = dict(
    ninp=32, nlayers=2, nhead=2, nhid=32,
    num_features=16, n_out=4,
    regression_bin_count=32, regression_bin_min=-10, regression_bin_max=10,
    y_encoder_dim=16, num_col_attn_layers=1, n_thinking_rows=4,
    enc_cell_dim=-1, base_len=8, max_len=4096,
    dropout=0.0, use_flash=False, clip_sigma=8.0,
)


def real() -> ExportConfig:
    return ExportConfig(
        name="real", model_kwargs=dict(_REAL_KWARGS),
        max_classes=16,
        example=(16, 6, 10), parity_shapes=((32, 12, 20),),
    )


def fixture() -> ExportConfig:
    return ExportConfig(
        name="fixture", model_kwargs=dict(_FIXTURE_KWARGS),
        max_classes=4,
        example=(12, 5, 8), parity_shapes=((40, 7, 30), (16, 9, 6)),
    )


def get(name: str) -> ExportConfig:
    return {"real": real, "fixture": fixture}[name]()


# Maps our TabDPTModel kwarg -> the key under cfg["model"] in the checkpoint
# metadata. `dropout` lives under cfg["training"] and is checked separately.
_CFG_KEYS = {
    "ninp": "emsize", "nlayers": "nlayers", "nhead": "nhead", "nhid": "ff_dim",
    "num_features": "max_num_features", "n_out": "max_num_classes",
    "regression_bin_count": "regression_bin_count",
    "regression_bin_min": "regression_bin_min",
    "regression_bin_max": "regression_bin_max",
    "y_encoder_dim": "y_encoder_dim",
    "num_col_attn_layers": "num_col_attn_layers",
    "n_thinking_rows": "n_thinking_rows",
    "enc_cell_dim": "enc_cell_dim",
    "base_len": "min_eval_context", "max_len": "max_eval_context",
}


def assert_matches_checkpoint(weights_path: str) -> None:
    """Fail loudly if `real()` has drifted from the published checkpoint.

    The graph is architecture-only, so a dims mismatch does not error at export
    time — it produces a graph whose initializers cannot be populated from the
    real weights, which would surface much later as a confusing injection
    failure. Cheap to check here, so check here.
    """
    from safetensors import safe_open

    with safe_open(weights_path, framework="pt", device="cpu") as f:
        cfg = json.loads(f.metadata()["cfg"])

    mismatches = []
    for kwarg, cfg_key in _CFG_KEYS.items():
        want = _REAL_KWARGS[kwarg]
        got = cfg["model"].get(cfg_key)
        if got != want:
            mismatches.append(f"{kwarg} (cfg.model.{cfg_key}): config={want!r} checkpoint={got!r}")
    got_dropout = cfg.get("training", {}).get("dropout")
    if got_dropout != _REAL_KWARGS["dropout"]:
        mismatches.append(
            f"dropout (cfg.training.dropout): config={_REAL_KWARGS['dropout']!r} "
            f"checkpoint={got_dropout!r}")
    if mismatches:
        raise SystemExit(
            "export_tabdpt: configs.real() no longer matches the published "
            "checkpoint:\n  " + "\n  ".join(mismatches))
