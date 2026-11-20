"""nori.pt -> model.safetensors, in the bare key namespace the tensor map points at (ROCm and CUDA).

The CPU path needs no conversion: the engine reads nori.pt (``{model_config, ema_state_dict}`` for Nori-30M,
``{model_config, model_state_dict, inference_config}`` for Nori 1.5) natively. The ROCm and CUDA graphs are
different: MIGraphX reads the weights itself, as external data at fixed byte offsets in a file
named ``model.safetensors`` beside the checkpoint, and the engine checks that file's header against the sha
the graph was built for (``ExpectedWeightsHeaderShaFor``). So ROCm users run this once:

    uv run convert_nori_weights ~/.cache/anofox-tabfm/Synthefy__Nori-30M@63c9f7fa.../model.ckpt \
        ~/.cache/anofox-tabfm/Synthefy__Nori-30M@63c9f7fa.../model.safetensors

The output is deterministic (sorted keys, float32, contiguous, no metadata), which is what makes the header
sha a constant. The engine then prefers the converted file on every device; its answers are the same.
The checkpoint is loaded with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import pathlib

import torch
from safetensors.torch import save_file


def convert(ckpt_path, out_path) -> int:
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    # The same precedence as the engine's reader (src/tabfm_ckpt.cpp): the EMA weights when the file has them
    # (Nori-30M: {model_config, ema_state_dict}), else model_state_dict (Nori 1.5: {model_config,
    # model_state_dict, inference_config}, no EMA copy).
    entry = next((k for k in ("ema_state_dict", "model_state_dict") if isinstance(obj, dict) and k in obj), None)
    if entry is None:
        raise ValueError(f"{ckpt_path}: expected a dict with an 'ema_state_dict' or 'model_state_dict' entry")
    state = {}
    for key in sorted(obj[entry]):
        value = obj[entry][key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{ckpt_path}: {entry}[{key!r}] is {type(value).__name__}, not a tensor")
        state[key] = value.detach().to(torch.float32).contiguous()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out))
    return len(state)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="convert_nori_weights")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    args = ap.parse_args(argv)
    n = convert(args.ckpt, args.out)
    print(f"[convert_nori_weights] {n} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
