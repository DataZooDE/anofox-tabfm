#!/usr/bin/env python3
"""Fail if a MIGraphX graph does integer division on runtime data.

MIGraphX's GPU target miscomputes an integer floor-division: for n=8, f=3,
`j < (n + f - 1) // f` marks FOUR groups real where there are three. ORT and
MIGraphX's own reference (CPU) target both get it right, and materialising the
quotient as a separate graph output also gives the right value -- which is what
made it so hard to see. It cost a first TabPFN-on-ROCm attempt a 1-3% relative
error in the final logits while every label still agreed, and it was found only
by bisecting every intermediate tensor. Nothing that runs without a GPU can see
it, so this is the structural guard: an integer `Div` reachable from runtime data
is the pattern, whatever it feeds.

Shape arithmetic is exempt. `padded_columns // features_per_group` is an integer
Div too, but its inputs come from Shape ops, which MIGraphX folds to constants
once the bucket shape is pinned; it never reaches the GPU as a computation. Only a
Div reachable from a graph input through non-Shape ops is evaluated on the device.

    tools/check_migraphx_int_div.py resources/graph_migraphx_*.onnx
"""

import argparse
import pathlib
import sys

import onnx
from onnx import TensorProto as TP
from onnx import shape_inference

INTS = {TP.INT64, TP.INT32, TP.INT16, TP.INT8, TP.UINT8, TP.UINT16, TP.UINT32, TP.UINT64}


def int_divs(path):
    """[(node name, output)] for integer Divs that depend on runtime data."""
    m = onnx.load(str(path), load_external_data=False)
    inferred = shape_inference.infer_shapes(m)
    ty = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input):
        if vi.type.HasField("tensor_type"):
            ty[vi.name] = vi.type.tensor_type.elem_type
    for init in m.graph.initializer:
        ty[init.name] = init.data_type
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    ty[n.output[0]] = a.t.data_type

    tainted = {i.name for i in m.graph.input}
    for n in m.graph.node:  # topological order
        if n.op_type in ("Shape", "Size"):
            continue
        if any(i in tainted for i in n.input):
            tainted.update(n.output)
    return [(n.name, n.output[0]) for n in m.graph.node
            if n.op_type == "Div"
            and any(i in tainted for i in n.input)
            and any(ty.get(i) in INTS for i in n.input)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graphs", nargs="+")
    args = ap.parse_args(argv)
    bad = 0
    for g in map(pathlib.Path, args.graphs):
        found = int_divs(g)
        if found:
            bad += 1
            print(f"FAIL {g}: {len(found)} integer Div on runtime data: {found[:4]}")
        else:
            print(f"ok   {g}")
    if bad:
        print("\nMIGraphX's GPU target miscomputes integer floor-division; use a multiply "
              "or float form (j*f < n rather than j < (n+f-1)//f). See this file's docstring.",
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
