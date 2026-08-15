#!/usr/bin/env python3
"""Cross-backend equivalence: a device switch must not change the answer.

The contract behind docs/DYNAMIC_BACKENDS.md. Every backend runs the same
graph on the same inputs and is compared against the CPU result, which is the
reference by definition — it is the one every user gets.

Tolerances are per-comparison and stated rather than assumed, because they mean
different things:

    CPU vs CPU        bit-identical   same kernels, same reduction order; any
                                      difference is a bug in a graph rewrite
    CPU vs GPU fp32   relative 1e-4   different kernels and reduction order
    GPU bf16/fp16     class agreement reduced precision is the point of the mode

Runs with synthesized initializers by default (no licensed weights), or against
a real cached checkpoint with --weights, which is the stronger statement.

    equivalence.py resources/graph_tabicl_classification.onnx --providers cpu
    equivalence.py resources/graph_tabicl_classification.onnx \\
        --providers cpu,cuda --weights ~/.cache/.../model.safetensors \\
        --tensor-map resources/tensor_map_tabicl_classification.json

Exit codes: 0 equivalent, 1 a backend failed to run, 2 answers diverged,
3 a requested provider was not available (never silently skipped — a GPU run
that quietly measures CPU is the failure this suite exists to prevent).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

SAFETENSORS_DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "BOOL": np.bool_}
ONNX_DTYPE = {1: np.float32, 7: np.int64, 9: np.bool_, 11: np.float64}


def read_safetensors(path: Path):
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        base = 8 + header_len
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = SAFETENSORS_DTYPE.get(meta["dtype"])
            if dtype is None:
                raise SystemExit(f"unhandled safetensors dtype {meta['dtype']}")
            begin, end = meta["data_offsets"]
            fh.seek(base + begin)
            arr = np.frombuffer(fh.read(end - begin), dtype=dtype).reshape(meta["shape"] or [1])
            if meta["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[name] = arr
    return out


def initializers(graph: Path, seed: int, weights: Path | None, tensor_map: Path | None):
    """Injected initializers: the real checkpoint when given, else synthesized."""
    model = onnx.load(str(graph), load_external_data=False)
    tensors = read_safetensors(weights) if weights else None
    mapping = json.loads(tensor_map.read_text())["initializers"] if tensor_map else {}
    rng = np.random.default_rng(seed)

    names, values, missing = [], [], []
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        dims = list(init.dims)
        if tensors is not None:
            arr = tensors.get(mapping.get(init.name, init.name))
            if arr is None:
                missing.append(init.name)
                continue
            arr = arr.reshape(dims).astype(np.float32)
        else:
            dtype = ONNX_DTYPE.get(init.data_type)
            if dtype is None:
                raise SystemExit(f"unhandled onnx dtype {init.data_type} for {init.name}")
            arr = ((rng.standard_normal(dims) * 0.02).astype(dtype) if dtype in (np.float32, np.float64)
                   else rng.integers(0, 2, dims).astype(dtype))
        names.append(init.name)
        values.append(ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr)))
    if missing:
        raise SystemExit(f"{len(missing)} initializers unmapped, first: {missing[:3]}")
    return names, values


def run(graph: Path, provider: str, feeds, seed, weights, tensor_map):
    ep = PROVIDERS[provider]
    names, values = initializers(graph, seed, weights, tensor_map)
    so = ort.SessionOptions()
    so.add_external_initializers(names, values)
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(graph), so, providers=[ep])
    if ep not in sess.get_providers():
        # Never fall through to CPU: a GPU comparison that silently ran on CPU
        # would "pass" while testing nothing.
        raise LookupError(f"{ep} did not instantiate (got {sess.get_providers()})")
    return sess.run(["logits"], feeds)[0]


def compare(reference, other, rel_tol):
    a = reference.astype(np.float64)
    b = other.astype(np.float64)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if np.array_equal(reference, other):
        return True, "bit-identical"
    denom = np.maximum(np.abs(a), 1e-12)
    rel = float(np.max(np.abs(a - b) / denom))
    ok = rel <= rel_tol
    detail = f"max relative {rel:.3e} (tolerance {rel_tol:.0e})"
    if reference.ndim == 3 and reference.shape[2] > 1:
        agree = float(np.mean(a.argmax(axis=2) == b.argmax(axis=2)))
        detail += f", argmax agreement {agree:.4f}"
        ok = ok and agree == 1.0
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", type=Path)
    ap.add_argument("--providers", default="cpu", help="comma-separated: cpu,cuda,rocm,migraphx,coreml")
    ap.add_argument("--shapes", default="70x3x60,128x8x100")
    ap.add_argument("--weights", type=Path, help="real checkpoint (.safetensors) instead of synthesized values")
    ap.add_argument("--tensor-map", type=Path, help="onnx-name -> safetensors-key map for --weights")
    ap.add_argument("--rel-tol", type=float, default=1e-4, help="CPU-vs-accelerator tolerance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wanted = [p.strip() for p in args.providers.split(",") if p.strip()]
    if "cpu" not in wanted:
        wanted.insert(0, "cpu")  # CPU is the reference by definition
    print(f"onnxruntime {ort.__version__}  graph {args.graph.name}")
    print(f"reference=cpu  compare={[p for p in wanted if p != 'cpu'] or '(none — cpu self-check only)'}")
    print(f"weights={'real checkpoint' if args.weights else 'synthesized'}")

    failures = divergences = unavailable = 0
    for chunk in args.shapes.split(","):
        T, H, S = (int(v) for v in chunk.lower().split("x"))
        rng = np.random.default_rng(1234)
        feeds = {"x": rng.standard_normal((1, T, H)).astype(np.float32),
                 "y": rng.integers(0, 3, (1, S)).astype(np.float32)}
        print(f"\n=== T={T} H={H} S={S}")

        results = {}
        for provider in wanted:
            try:
                results[provider] = run(args.graph, provider, feeds, args.seed, args.weights, args.tensor_map)
                print(f"  {provider:9} ran, logits {tuple(results[provider].shape)}")
            except LookupError as exc:
                print(f"  {provider:9} UNAVAILABLE: {exc}")
                unavailable += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {provider:9} FAILED: {str(exc).strip().splitlines()[-1][:140]}")
                failures += 1

        reference = results.get("cpu")
        if reference is None:
            continue
        for provider, out in results.items():
            if provider == "cpu":
                continue
            ok, detail = compare(reference, out, args.rel_tol)
            print(f"  cpu vs {provider:9} {'OK  ' if ok else 'DIVERGED'} {detail}")
            if not ok:
                divergences += 1

    print(f"\nfailures={failures} divergences={divergences} unavailable={unavailable}")
    if failures:
        return 1
    if divergences:
        return 2
    return 3 if unavailable else 0


if __name__ == "__main__":
    sys.exit(main())
