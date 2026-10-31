"""One-time convert: Kumo-Tabular `<size>/{classifier,regressor}.pt` -> `model.safetensors` beside it.

The CPU path does NOT need this: the published `.pt` is a plain `torch.save` state_dict of float32
tensors, which the engine's native checkpoint reader (`src/tabfm_ckpt.cpp`) injects directly
against the committed tensor map. What the conversion buys is the GPU and low-memory paths: the
bundled `graph_ext_kumo_*` (CUDA, CPU low-memory) and `graph_migraphx_kumo_*` (ROCm) graphs
reference the weights as ONNX external data at fixed offsets in a safetensors file, and the
engine only uses them when that file's JSON header hashes to the value pinned in
`ExpectedWeightsHeaderShaFor` (src/include/tabfm_model_spec.hpp). This writes exactly that file:
every state_dict key, sorted, as contiguous float32, no metadata -- byte-deterministic, so the
header sha is a property of the checkpoint alone.

    uv run convert_kumo_weights --size small            # both tasks
    uv run convert_kumo_weights --size large --task classification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

import torch
from safetensors.torch import save_file

REVISION = "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f"  # HF tag v1.0.1, the one upstream code pins
CACHE = pathlib.Path("~/.cache/anofox-tabfm").expanduser() / f"nvidia__Kumo-Tabular@{REVISION}"


def header_sha(path: pathlib.Path) -> str:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return hashlib.sha256(f.read(n)).hexdigest()


def convert(ckpt: pathlib.Path, out: pathlib.Path, tensor_map: pathlib.Path | None = None) -> str:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise SystemExit(f"{ckpt}: expected a flat state_dict of tensors")
    # clone(): upstream SHARES one RotaryEmbedding between a block's query and key transforms, so
    # several keys alias one storage, which safetensors refuses. The graph maps one of them.
    tensors = {k: sd[k].detach().to(torch.float32).contiguous().clone() for k in sorted(sd)}
    if tensor_map is not None:
        wanted = set(json.loads(tensor_map.read_text())["initializers"].values())
        missing = sorted(wanted - set(tensors))
        if missing:
            raise SystemExit(f"{ckpt}: {len(missing)} tensor-map keys absent, e.g. {missing[:3]}")
        print(f"  {len(wanted)}/{len(wanted)} tensor-map keys present ({len(tensors)} in checkpoint)")
    tmp = out.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp))
    tmp.replace(out)
    return header_sha(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="convert_kumo_weights")
    ap.add_argument("--size", required=True, choices=["small", "medium", "large"])
    ap.add_argument("--task", choices=["classification", "regression"])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--resources", default=str(pathlib.Path(__file__).resolve().parents[4] / "resources"))
    args = ap.parse_args(argv)
    for task in [args.task] if args.task else ["classification", "regression"]:
        d = pathlib.Path(args.cache) / args.size / task
        ckpt = d / "model.ckpt"
        if not ckpt.exists():
            print(f"{ckpt}: not downloaded -- CALL tabfm_download('{task}', model := "
                  f"'kumo-tabular-{args.size}') first", file=sys.stderr)
            return 1
        tmap = pathlib.Path(args.resources) / f"tensor_map_kumo_{args.size}_{task}.json"
        sha = convert(ckpt, d / "model.safetensors", tmap if tmap.exists() else None)
        print(f"{args.size}/{task}: {d / 'model.safetensors'}  header sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
