#!/usr/bin/env python3
"""Run a weight-free graph through an ONNX Runtime execution provider.

The graphs in resources/ carry no weights (the license wall), so this
synthesizes deterministic random initializers and injects them exactly as the
C++ engine does — enough to exercise the graph's *structure* on a provider,
which is where EP-specific bugs live (issue #21: ScatterND fails on CUDA while
CPU is fine).

Deterministic in the seed, so two graphs fed the same seed and inputs must
produce byte-identical outputs if they are semantically equivalent.

    ort_ep_check.py graph.onnx --provider cuda --shapes 70x3x60,128x8x100
    ort_ep_check.py a.onnx b.onnx --provider cuda --compare

Exit codes: 0 all ran (and matched, with --compare), 1 a run failed,
2 outputs differed.
"""

from __future__ import annotations

import argparse
import hashlib
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
}

NUMPY_DTYPE = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.BOOL: np.bool_,
}


def synth_initializers(path: Path, seed: int = 0):
    """Deterministic stand-ins for every external-data stub in the graph."""
    model = onnx.load(str(path), load_external_data=False)
    rng = np.random.default_rng(seed)
    names, values = [], []
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        dtype = NUMPY_DTYPE.get(init.data_type)
        if dtype is None:
            raise SystemExit(f"unhandled dtype {init.data_type} for {init.name}")
        dims = list(init.dims)
        if dtype in (np.float32, np.float64):
            # Small values: the point is to exercise the graph, not to overflow it.
            arr = (rng.standard_normal(size=dims) * 0.02).astype(dtype)
        else:
            arr = rng.integers(0, 2, size=dims).astype(dtype)
        names.append(init.name)
        values.append(ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr)))
    return names, values


OPT_LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}


def run_one(path: Path, provider: str, T: int, H: int, S: int, seed: int = 0, opt: str = "all",
            mem_pattern: bool = True):
    """(ok, detail, {output: sha256}) for one graph on one provider."""
    names, values = synth_initializers(path, seed)
    so = ort.SessionOptions()
    so.add_external_initializers(names, values)
    so.log_severity_level = 3
    so.graph_optimization_level = OPT_LEVELS[opt]
    # The memory-pattern planner pre-plans buffer reuse from the first run's
    # shapes; with dynamic shapes that has historically handed kernels stale
    # buffers, which looks like a shape bug rather than a memory bug.
    so.enable_mem_pattern = mem_pattern

    ep = PROVIDERS[provider]
    try:
        sess = ort.InferenceSession(str(path), so, providers=[ep])
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        return False, f"session creation failed: {exc}", {}
    if ep not in sess.get_providers():
        return False, f"{ep} not available (got {sess.get_providers()})", {}

    rng = np.random.default_rng(42)
    feeds = {
        "x": rng.standard_normal(size=(1, T, H)).astype(np.float32),
        "y": rng.integers(0, 2, size=(1, S)).astype(np.float32),
    }
    out_names = [o.name for o in sess.get_outputs()]
    try:
        outs = sess.run(out_names, feeds)
    except Exception as exc:  # noqa: BLE001
        return False, f"inference failed: {exc}", {}

    digests = {}
    for name, arr in zip(out_names, outs):
        digests[name] = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    return True, f"shapes {[tuple(o.shape) for o in outs]}", digests


def parse_shapes(text: str):
    out = []
    for chunk in text.split(","):
        T, H, S = (int(v) for v in chunk.lower().split("x"))
        out.append((T, H, S))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graphs", nargs="+", type=Path)
    ap.add_argument("--provider", default="cpu", choices=sorted(PROVIDERS))
    ap.add_argument("--shapes", default="70x3x60", help="TxHxS list, e.g. 70x3x60,128x8x100")
    ap.add_argument("--compare", action="store_true", help="require all graphs to agree bit-for-bit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt-level", default="all", choices=sorted(OPT_LEVELS),
                    help="ORT graph optimization level; 'disabled' isolates optimizer bugs")
    ap.add_argument("--no-mem-pattern", action="store_true",
                    help="disable the memory-pattern planner (suspect under dynamic shapes)")
    args = ap.parse_args()

    print(f"onnxruntime {ort.__version__}  provider={PROVIDERS[args.provider]}  opt={args.opt_level}  "
          f"mem_pattern={not args.no_mem_pattern}")
    print(f"available: {ort.get_available_providers()}")

    failures, mismatches = 0, 0
    for T, H, S in parse_shapes(args.shapes):
        print(f"\n=== T={T} H={H} S={S} (context={S}, test={T - S})")
        digests = {}
        for graph in args.graphs:
            ok, detail, digest = run_one(graph, args.provider, T, H, S, args.seed, args.opt_level,
                                         mem_pattern=not args.no_mem_pattern)
            status = "ok  " if ok else "FAIL"
            print(f"  [{status}] {graph.name}: {detail}")
            if not ok:
                failures += 1
            digests[graph.name] = digest
        if args.compare:
            # Compare only outputs the graphs have in COMMON: a workaround may
            # legitimately add outputs (pinning buffers, issue #21), and that is
            # not a semantic difference.
            populated = [d for d in digests.values() if d]
            if len(populated) > 1:
                shared = set.intersection(*(set(d) for d in populated))
                if not shared:
                    print("  MISMATCH: graphs share no output names")
                    mismatches += 1
                elif len({tuple(sorted((k, d[k]) for k in shared)) for d in populated}) > 1:
                    print(f"  MISMATCH on shared outputs {sorted(shared)}")
                    for name, d in digests.items():
                        print(f"    {name}: { {k: d.get(k) for k in sorted(shared)} }")
                    mismatches += 1
                else:
                    extra = {n: sorted(set(d) - shared) for n, d in digests.items() if set(d) - shared}
                    note = f" (extra outputs ignored: {extra})" if extra else ""
                    print(f"  match on {sorted(shared)}{note}")

    print(f"\nfailures={failures} mismatches={mismatches}")
    if failures:
        return 1
    return 2 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
