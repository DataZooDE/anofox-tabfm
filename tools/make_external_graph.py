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
                               [--pin-cuda-shapes]
  task = classification | regression

--pin-cuda-shapes names every shape-computation tensor as a graph output
(tools/pin_cuda_shape_operands.py), the workaround for the ONNX Runtime CUDA
buffer-reuse bug in docs/archive/UPSTREAM_ORT_ISSUES.md (Draft 3). The TabPFN-3.5 /
3.5-Fast ext graphs need it to run on CUDA; graph_invariants.yml checks it.
"""
import argparse, hashlib, json, os, struct, sys
import onnx
from onnx import TensorProto, helper

DEF_CACHE = os.path.expanduser("~/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83")


#: safetensors dtype -> (ONNX TensorProto type, bytes per element)
ST_DTYPES = {"F32": (TensorProto.FLOAT, 4), "BF16": (TensorProto.BFLOAT16, 2), "F16": (TensorProto.FLOAT16, 2),
             "I64": (TensorProto.INT64, 8), "I32": (TensorProto.INT32, 4)}


def retype_external(m, header, st_for):
    """Make every externalized initializer's TYPE match the bytes it now points at.

    The exported graph declares its initializers FLOAT, and until TabFM 1.1.0 every checkpoint fed
    to these tools was F32, so pointing an initializer at safetensors bytes never had to look at
    the dtype. TabFM 1.1.0 ships BF16: a FLOAT initializer over half as many bytes is not a
    smaller tensor, it is garbage (or a length error, depending on the reader). So:

      * F32 bytes, FLOAT initializer   -> unchanged;
      * BF16/F16 bytes, FLOAT initializer -> the initializer is retyped to the stored dtype and
        renamed `<name>__stored`, and a `Cast(to=FLOAT)` restores `<name>` for every consumer.
        Numerically identical to the CPU injection path, which upcasts BF16 -> F32 exactly;
      * anything else, or a byte length that does not equal numel * itemsize -> refuse.

    Returns the number of initializers that got a Cast.
    """
    casts = []
    for init in m.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        st = st_for(init.name)
        if st not in header:
            continue
        h = header[st]
        if h["dtype"] not in ST_DTYPES:
            raise SystemExit(f"REFUSING: '{init.name}' is stored as {h['dtype']}, which this tool cannot type")
        st_type, itemsize = ST_DTYPES[h["dtype"]]
        numel = 1
        for dim in h["shape"]:
            numel *= dim
        b, e = h["data_offsets"]
        if e - b != numel * itemsize:
            raise SystemExit(f"REFUSING: '{init.name}' spans {e - b} bytes but {h['dtype']}{h['shape']} needs "
                             f"{numel * itemsize}")
        if st_type == init.data_type:
            continue
        if init.data_type == TensorProto.FLOAT and st_type in (TensorProto.BFLOAT16, TensorProto.FLOAT16):
            orig = init.name
            init.name = orig + "__stored"
            init.data_type = st_type
            casts.append(helper.make_node("Cast", [init.name], [orig], to=TensorProto.FLOAT,
                                          name=orig + "__upcast"))
            continue
        raise SystemExit(f"REFUSING: '{init.name}' is declared {TensorProto.DataType.Name(init.data_type)} in the "
                         f"graph but stored as {h['dtype']}; only a FLOAT <- BF16/F16 upcast is supported")
    if casts:
        nodes = list(m.graph.node)
        del m.graph.node[:]
        m.graph.node.extend(casts + nodes)
    return len(casts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["classification", "regression"])
    ap.add_argument("--weights")
    ap.add_argument("--graph")
    ap.add_argument("--tensor-map")
    ap.add_argument("--out")
    ap.add_argument("--pin-cuda-shapes", action="store_true",
                    help="pin shape tensors as graph outputs (CUDA EP workaround, TabPFN-3.5)")
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
    # BF16 checkpoints (TabFM 1.1.0): retype + Cast.
    st_for = lambda name: onnx2st.get(name) or (name[2:] if name.startswith("m.") else name)
    n_cast = retype_external(m, {k: v for k, v in header.items() if k != "__metadata__"}, st_for)
    n_pins = 0
    if a.pin_cuda_shapes:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pin_cuda_shape_operands import apply as pin_shapes
        n_pins = pin_shapes(m)
    onnx.save(m, out)
    print(f"task={task}  initializers={len(m.graph.initializer)}  externalized={n}  inline_left={len(missing)}"
          f"  bf16_upcasts={n_cast}  cuda_shape_pins={n_pins}")
    print(f"header_sha256={header_sha}")
    print(f"wrote {out} ({os.path.getsize(out)} bytes, weight-free)")


if __name__ == "__main__":
    main()
