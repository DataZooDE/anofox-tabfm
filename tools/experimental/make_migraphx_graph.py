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

Usage: make_migraphx_graph.py <task> [--rows N --features M] [--dynamic]
"""
import argparse, json, os, struct, sys
import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

DEF_CACHE = os.path.expanduser("~/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main")
INTMAX = 2**63 - 1


def externalize(m, weights, tmap):
    with open(weights, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    base = 8 + hlen
    off = {k: v["data_offsets"] for k, v in header.items() if k != "__metadata__"}
    onnx2st = json.load(open(tmap))["initializers"]
    for init in m.graph.initializer:
        st = onnx2st.get(init.name) or (init.name[2:] if init.name.startswith("m.") else init.name)
        if st not in off:
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
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--features", type=int, default=8)
    ap.add_argument("--dynamic", action="store_true", help="keep dynamic shapes (backend sets per-bucket)")
    ap.add_argument("--out")
    a = ap.parse_args()
    weights = os.path.join(DEF_CACHE, a.task, "model.safetensors")
    m = onnx.load(f"resources/graph_{a.task}.onnx", load_external_data=False)
    externalize(m, weights, f"resources/tensor_map_{a.task}.json")
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
