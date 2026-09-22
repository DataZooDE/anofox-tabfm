#!/usr/bin/env python3
"""Prepare a TabFM graph so AMD MIGraphX can parse and compile it (GPU path).

Background: ORT's MIGraphX EP is unusable for this >2 GB model — it re-inlines
the weights into an onnx.ModelProto and hits protobuf's 2 GB limit (see
docs/GPU_AND_MEMORY_FINDINGS.md). MIGraphX's *own* file-based parser handles the
model fine, so the GPU route is to feed MIGraphX directly. Two graph transforms
are needed to get it through migraphx's ONNX parser + compiler:

  1. external-data  — reference the cached safetensors on disk (migraphx reads
     external data; keeps the model out of the 2 GB proto). Same as
     tools/make_external_graph.py.
  2. Shape rewrite  — migraphx's parser rejects degenerate `Shape` ops
     (start==end); rewrite every attributed `Shape(x,start,end)` into plain
     `Shape(x)` + `Slice` (both migraphx-friendly).
  3. (optional) static shapes — pin the dynamic `rows`/`features` dims to a fixed
     bucket for a fast first compile (dynamic-shape compiles are very slow).

Output goes next to the cached weights so migraphx resolves "model.safetensors".
Then:  migraphx-driver perf  <out>.onnx --gpu     (compile + run on gfx1201)

Usage: make_migraphx_graph.py <task> [--weights PATH] [--graph PATH]
       [--tensor-map PATH] [--rows N --features M] [--dynamic] [--out PATH]

Defaults target tabfm-v1; pass the three paths for any other model (the mitra
variants in resources/ are produced this way).
"""
import argparse, json, os, struct, sys
import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

DEF_CACHE = os.path.expanduser("~/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main")
INTMAX = 2**63 - 1


def externalize(m, weights, tmap, inline_ok=()):
    """Point initializers at the cached safetensors.

    `inline_ok` names initializers that are allowed to stay embedded because
    they are code-generated constants rather than checkpoint weights --
    tabdpt's `bin_centres` (the bar-distribution bin midpoints, derived from
    three config scalars) is the only one today, and the positional exporter
    documents it as deliberately out of the tensor map.

    Named, not blanket. The refusal below exists because an initializer the
    tensor map missed keeps its STUB bytes and the graph then predicts garbage
    while looking healthy; tolerating every leftover would give that failure
    back. An explicit allowlist keeps the guarantee for everything else.
    """
    missing = []
    inlined = []
    with open(weights, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    off = {k: v["data_offsets"] for k, v in header.items() if k != "__metadata__"}
    onnx2st = json.load(open(tmap))["initializers"]
    for init in m.graph.initializer:
        if init.name in inline_ok:
            # Materialise it INLINE rather than merely skipping it. The graph is
            # loaded without external data, so this initializer still points at
            # the exporter's .onnx.data sidecar -- which does not travel with the
            # converted graph, and MIGraphX then dies at load with
            # "Failure opening file: ....onnx.data". Reading the bytes and
            # embedding them makes the output self-contained apart from the
            # safetensors it is meant to reference.
            inlined.append(init.name)
            continue
        st = onnx2st.get(init.name) or (init.name[2:] if init.name.startswith("m.") else init.name)
        if st not in off:
            missing.append(init.name)
            continue
        b, e = off[st]
        init.ClearField("raw_data")
        for fld in ("float_data", "int32_data", "int64_data", "double_data"):
            init.ClearField(fld)
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", "model.safetensors"), ("offset", str(base + b)), ("length", str(e - b))):
            en = init.external_data.add()
            en.key, en.value = k, v
    if missing:
        # A stub initializer the tensor map missed keeps its stub bytes and the
        # graph silently predicts garbage -- the one failure mode this tool
        # must never allow (review finding, 2026-08-22).
        raise SystemExit(f"REFUSING: {len(missing)} initializers not in the tensor map: {missing[:5]}")
    return inlined


def rewrite_shapes(g):
    consts, new, n = [], [], 0

    def const(name, arr):
        consts.append(helper.make_node("Constant", [], [name],
                                       value=numpy_helper.from_array(np.array(arr, dtype=np.int64), name + "_v")))

    for nd in g.node:
        if nd.op_type == "Shape" and any(a.name in ("start", "end") for a in nd.attribute):
            at = {a.name: a.i for a in nd.attribute}
            s, e = at.get("start", 0), at.get("end", INTMAX)
            full = nd.output[0] + "_full"
            new.append(helper.make_node("Shape", [nd.input[0]], [full], name=nd.name + "_full"))
            sc, ec, ac = nd.name + "_s", nd.name + "_e", nd.name + "_a"
            const(sc, [s]); const(ec, [e]); const(ac, [0])
            new.append(helper.make_node("Slice", [full, sc, ec, ac], [nd.output[0]], name=nd.name + "_slice"))
            n += 1
        else:
            new.append(nd)
    del g.node[:]
    g.node.extend(consts + new)
    return n


def pin_shapes(g, rows, feats):
    fix = {"rows": rows, "features": feats}
    for inp in g.input:
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_param") and d.dim_param in fix:
                v = fix[d.dim_param]; d.ClearField("dim_param"); d.dim_value = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["classification", "regression"])
    ap.add_argument("--weights")
    ap.add_argument("--graph")
    ap.add_argument("--tensor-map")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--dynamic", action="store_true", help="keep dynamic shapes (backend sets per-bucket)")
    ap.add_argument("--out")
    ap.add_argument("--inline-ok", default="",
                    help="comma-separated initializers allowed to stay embedded (code-generated "
                         "constants, e.g. tabdpt's bin_centres) instead of being externalized")
    a = ap.parse_args()
    weights = a.weights or os.path.join(DEF_CACHE, a.task, "model.safetensors")
    inline_ok = {n for n in a.inline_ok.split(",") if n}
    graph_path = a.graph or f"resources/graph_{a.task}.onnx"
    # Load external data only when something has to be embedded: the whole point
    # of the external-data form is to keep multi-GB weights out of the proto, so
    # this reads them only to write a handful of constant bytes back in.
    m = onnx.load(graph_path, load_external_data=bool(inline_ok))
    inlined = externalize(m, weights, a.tensor_map or f"resources/tensor_map_{a.task}.json",
                          inline_ok=inline_ok)
    if inlined:
        print(f"  inlined (code-generated constants): {inlined}")
    n = rewrite_shapes(m.graph)
    if not a.dynamic:
        pin_shapes(m.graph, a.rows, a.features)
    out = a.out or f"resources/graph_migraphx_{a.task}.onnx"
    onnx.save(m, out)
    dims = [(i.name, [dd.dim_value if dd.HasField("dim_value") else dd.dim_param
                      for dd in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    print(f"task={a.task}  shape-rewrites={n}  inputs={dims}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
