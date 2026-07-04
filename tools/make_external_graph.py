#!/usr/bin/env python3
"""Generate a weight-free external-data ONNX graph for a TabFM task.

The shipped weight-free graph (resources/graph_<task>.onnx) has its initializers
as external-data STUBS that the C++ engine overrides in memory
(AddExternalInitializers). That in-memory injection makes ORT hold its own copy
of the ~6.6 GB weights (peak RSS ~18.6 GB). This tool rewrites the graph so each
initializer references the *cached safetensors file directly* as ONNX
external-data (location "model.safetensors", absolute offset/length). Loading
that graph from a path lets ORT read the weights straight off disk — no
injection, no copy — dropping peak RSS to ~7.3 GB (validated).

The safetensors data section is raw, contiguous, row-major tensor bytes, so the
ONNX external offset is simply 8 + header_len + st_data_offset.

Also prints the SHA-256 of the safetensors JSON header. The engine bakes this in
and validates it at load: a byte-identical header guarantees the offsets are
correct; any mismatch falls back to the (layout-independent) injection path.

Usage:
  tools/make_external_graph.py <task> [--weights PATH] [--out resources/graph_ext_<task>.onnx]
  task = classification | regression
"""
import argparse, hashlib, json, os, struct, sys
import onnx
from onnx import TensorProto

DEF_CACHE = os.path.expanduser("~/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["classification", "regression"])
    ap.add_argument("--weights")
    ap.add_argument("--graph")
    ap.add_argument("--tensor-map")
    ap.add_argument("--out")
    a = ap.parse_args()
    task = a.task
    weights = a.weights or os.path.join(DEF_CACHE, task, "model.safetensors")
    graph = a.graph or f"resources/graph_{task}.onnx"
    tmap = a.tensor_map or f"resources/tensor_map_{task}.json"
    out = a.out or f"resources/graph_ext_{task}.onnx"

    with open(weights, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(hlen)
    header = json.loads(header_bytes)
    header_sha = hashlib.sha256(header_bytes).hexdigest()
    data_base = 8 + hlen
    st_off = {k: v["data_offsets"] for k, v in header.items() if k != "__metadata__"}
    onnx2st = json.load(open(tmap))["initializers"]

    m = onnx.load(graph, load_external_data=False)
    n, missing = 0, []
    for init in m.graph.initializer:
        st = onnx2st.get(init.name) or (init.name[2:] if init.name.startswith("m.") else init.name)
        if st not in st_off:
            missing.append(init.name)
            continue
        b, e = st_off[st]
        init.ClearField("raw_data")
        for fld in ("float_data", "int32_data", "int64_data", "double_data"):
            init.ClearField(fld)
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", "model.safetensors"), ("offset", str(data_base + b)), ("length", str(e - b))):
            en = init.external_data.add()
            en.key, en.value = k, v
        n += 1
    onnx.save(m, out)
    print(f"task={task}  initializers={len(m.graph.initializer)}  externalized={n}  inline_left={len(missing)}")
    print(f"header_sha256={header_sha}")
    print(f"wrote {out} ({os.path.getsize(out)} bytes, weight-free)")


if __name__ == "__main__":
    main()
