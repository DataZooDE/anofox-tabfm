#!/usr/bin/env python3
"""Standalone reproducer: the CUDA EP hands a live CPU shape tensor's buffer to another.

Fetches a public weight-free ONNX graph (TabPFN-3.5-Fast regression, as released
in v2026.11.01, before the workaround), synthesizes its initializers (the graph
ships without weights by design) and runs it on the CPU and CUDA execution
providers. The CPU runs it; CUDA fails at an Expand whose shape operand, computed
on the CPU by a Shape -> Squeeze -> Add/Min -> Reshape -> Concat subgraph,
arrives as {T+S, 1, T+S} instead of {T+S, 1, H}: the feature count's buffer
holds the row-count expression. With the CPU arena disabled it holds garbage.

    pip install onnx numpy
    pip install --no-deps onnxruntime-gpu==1.29.0 \\
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
    python ort_repro_tabpfn35.py              # CPU ok, CUDA fails at node_expand
    python ort_repro_tabpfn35.py --no-arena   # CUDA reads freed memory instead
    python ort_repro_tabpfn35.py --pinned     # shape tensors as graph outputs: both ok

The CUDA-12 index matters: PyPI's onnxruntime-gpu 1.29.0 is the CUDA-13 build
(which fails the same way, as does 1.31.1). The cuDNN/cuBLAS libraries must be on
the loader path (e.g. the nvidia-*-cu12 wheels a CUDA 12 PyTorch install brings);
`onnxruntime.preload_dlls()` is called for that.
"""
import argparse
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

RAW = ("https://raw.githubusercontent.com/DataZooDE/anofox-tabfm/v2026.11.01/"
       "resources/graph_ext_tabpfn35fast_regression.onnx")

# Inputs that ORT's CUDA kernels read on the host.
SHAPE_OPERANDS = {"Expand": (1,), "Reshape": (1,), "Range": (0, 1, 2), "Slice": (1, 2, 3, 4),
                  "Unsqueeze": (1,), "Squeeze": (1,), "ConstantOfShape": (0,), "Tile": (1,),
                  "Pad": (1, 2, 3), "TopK": (1,), "ReduceSum": (1,), "ReduceMean": (1,), "CumSum": (1,)}


def fetch() -> str:
    path = "/tmp/graph_ext_tabpfn35fast_regression_v2026.11.01.onnx"
    try:
        open(path, "rb").close()
    except OSError:
        urllib.request.urlretrieve(RAW, path)
    return path


def pin_shape_tensors(model):
    """The workaround: every tensor in the closure of a host-read operand becomes a graph output."""
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    leaves = {i.name for i in g.initializer} | {i.name for i in g.input}
    stack = [n.input[p] for n in g.node for p in SHAPE_OPERANDS.get(n.op_type, ())
             if p < len(n.input) and n.input[p]]
    types = {v.name: v.type.tensor_type.elem_type
             for v in onnx.shape_inference.infer_shapes(model).graph.value_info}
    seen, have = set(), {o.name for o in g.output}
    while stack:
        t = stack.pop()
        if t in seen or t in leaves:
            continue
        seen.add(t)
        n = producer.get(t)
        if n is None or n.op_type == "Constant":
            continue
        if t not in have:
            g.output.append(helper.make_tensor_value_info(t, types.get(t) or TensorProto.INT64, None))
        if n.op_type != "Shape":
            stack.extend(i for i in n.input if i)
    return model


def session(model_bytes, init_model, provider, arena):
    rng = np.random.default_rng(0)
    so = ort.SessionOptions()
    names, values = [], []
    for init in init_model.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        arr = (rng.standard_normal(list(init.dims)) * 0.02).astype(np.float32)
        names.append(init.name)
        values.append(ort.OrtValue.ortvalue_from_numpy(arr))
    so.add_external_initializers(names, values)
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = arena
    # fp32 so the CPU/CUDA comparison is not dominated by TF32 rounding
    providers = [("CUDAExecutionProvider", {"use_tf32": "0"})] if provider == "cuda" else ["CPUExecutionProvider"]
    s = ort.InferenceSession(model_bytes, so, providers=providers)
    want = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    if want not in s.get_providers():
        raise SystemExit(f"{want} did not instantiate (got {s.get_providers()})")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", action="store_true", help="apply the workaround first")
    ap.add_argument("--no-arena", action="store_true", help="enable_cpu_mem_arena = False")
    args = ap.parse_args()
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    model = onnx.load(fetch(), load_external_data=False)
    if args.pinned:
        model = pin_shape_tensors(model)
    blob = model.SerializeToString()
    print(f"onnxruntime {ort.__version__}  graph v2026.11.01 ({len(model.graph.node)} nodes, "
          f"{len(model.graph.output)} outputs){'  [shape tensors pinned]' if args.pinned else ''}"
          f"  cpu_mem_arena={not args.no_arena}")
    sessions = {p: session(blob, model, p, not args.no_arena) for p in ("cpu", "cuda")}
    rng = np.random.default_rng(42)
    for T, H, S in [(100, 5, 70), (128, 8, 100)]:
        feeds = {"x": rng.standard_normal((1, T, H)).astype(np.float32),
                 "y": rng.standard_normal((1, S)).astype(np.float32)}
        print(f"  x=[1,{T},{H}] y=[1,{S}]")
        ref = None
        for p, s in sessions.items():
            try:
                out = s.run(["logits"], feeds)[0]  # only logits, as an application would
                extra = "" if ref is None else f"  max|cuda-cpu|={float(np.abs(out - ref).max()):.1e}"
                ref = out if ref is None else ref
                print(f"    {p:4} OK   logits {out.shape}{extra}")
            except Exception as exc:  # noqa: BLE001
                print(f"    {p:4} FAIL {' '.join(str(exc).split())[:220]}")


if __name__ == "__main__":
    main()
