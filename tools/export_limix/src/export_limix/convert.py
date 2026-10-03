"""LimiX-2M checkpoint -> safetensors, in the bare key namespace the tensor map points at.

The released ``LimiX-2M.ckpt`` is a ``torch.save`` pickle ``{"config": ..., "state_dict": ...}`` (hence
``weights_only=False``: it is loaded from the user's own Hugging Face cache, never from this repo). The
state_dict is already in the BARE module namespace (``encoder_x.0.mask_embedding``, ...), so no prefix is
re-applied and the converter is a straight re-encode: float32, contiguous, same keys. One file serves both
tasks, because the task is a property of the export wrapper, not of the weights.

    uv run convert_limix_weights ~/.cache/huggingface/.../LimiX-2M.ckpt model.safetensors

Nothing is written into the repository by this module; the caller names the output path. Weights are
under the Stable AI Technology Co., Ltd. License v1.0 (see docs/REAL_MODELS.md); converting is use of them.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from safetensors.torch import save_file


def convert(ckpt_path, out_path) -> int:
    """Re-encode the checkpoint's state_dict as safetensors. Returns the tensor count."""
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"{ckpt_path}: expected a dict with a 'state_dict' entry, got {type(obj).__name__}")
    state = {}
    for key, value in obj["state_dict"].items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{ckpt_path}: state_dict[{key!r}] is {type(value).__name__}, not a tensor")
        # float32 is what the exported graph's initializers are; a half/bf16 checkpoint tensor would be
        # silently widened by the engine's injection otherwise
        state[key] = value.detach().to(torch.float32).contiguous()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out))
    return len(state)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="convert_limix_weights")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    n = convert(args.ckpt, args.out)
    print(f"[convert_limix_weights] {n} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
