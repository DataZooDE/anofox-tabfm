"""Pin every shape-computation tensor of a graph as a graph output (CUDA EP workaround).

Workaround for an ONNX Runtime CUDA execution-provider bug that broke TabPFN-3.5 and
3.5-Fast on CUDA (ORT 1.29.0, CUDA-12 build; docs/UPSTREAM_ORT_ISSUES.md, Draft 3).
In a CUDA session the shape subgraph (Shape -> Squeeze -> Add/Min -> Reshape -> Concat)
runs on the CPU and feeds host-side operands of CUDA kernels (Expand's `shape`, Range,
Reshape, ...). ORT's allocation planner hands the CPU buffer of one of those small int64
tensors to another while it is still live, so `Expand` receives e.g. {170, 1, 170}
instead of {170, 1, 5} -- the feature count is overwritten with the row count -- and with
the CPU arena disabled it reads freed memory. The CPU EP runs the same graph correctly.

Naming the tensors as graph outputs excludes them from buffer reuse (the same lever as
tools/pin_dynamic_slice_bounds.py for issue #21). The engine requests only `logits`, so
the extra outputs are never copied out; the CPU result is unchanged byte for byte.

Structural, not name-based: the seeds are the host-read operands of every op below
(SHAPE_OPERANDS), and the pinned set is their backward closure up to Shape / Constant /
initializer / graph-input boundaries.

Idempotent. --check exits non-zero if any pin is missing (CI: graph_invariants.yml).
"""

import argparse
import pathlib
import sys

import onnx
from onnx import TensorProto, helper

# Inputs that ORT's CUDA kernels read on the host (shape-like operands).
SHAPE_OPERANDS = {
    "Expand": (1,),
    "Reshape": (1,),
    "Range": (0, 1, 2),
    "Slice": (1, 2, 3, 4),
    "Unsqueeze": (1,),
    "Squeeze": (1,),
    "ConstantOfShape": (0,),
    "Tile": (1,),
    "Pad": (1, 2, 3),
    "TopK": (1,),
    "ReduceSum": (1,),
    "ReduceMean": (1,),
    "ReduceMax": (1,),
    "ReduceMin": (1,),
    "CumSum": (1,),
    "Split": (1,),
    "Resize": (1, 2, 3),
}


def targets(model):
    """Tensors to pin, in a stable order: the closure of every shape operand."""
    graph = model.graph
    producer = {o: n for n in graph.node for o in n.output}
    leaves = {i.name for i in graph.initializer} | {i.name for i in graph.input}
    stack = []
    for node in graph.node:
        for pos in SHAPE_OPERANDS.get(node.op_type, ()):
            if pos < len(node.input) and node.input[pos]:
                stack.append(node.input[pos])
    stack.reverse()
    seen, out = set(), []
    while stack:
        name = stack.pop()
        if name in seen or name in leaves:
            continue
        seen.add(name)
        node = producer.get(name)
        if node is None or node.op_type == "Constant":
            continue
        out.append(name)
        if node.op_type != "Shape":
            stack.extend(i for i in reversed(node.input) if i)
    return out


def elem_types(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    return {v.name: v.type.tensor_type.elem_type
            for v in list(inferred.graph.value_info) + list(inferred.graph.output)}


def apply(model):
    """Add the missing pins to `model` in place; returns how many were added."""
    have = {o.name for o in model.graph.output}
    missing = [t for t in targets(model) if t not in have]
    if not missing:
        return 0
    types = elem_types(model)
    for name in missing:
        model.graph.output.append(
            helper.make_tensor_value_info(name, types.get(name) or TensorProto.INT64, None))
    return len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+", type=pathlib.Path)
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 if a pin is missing")
    args = ap.parse_args()
    rc = 0
    for path in args.graphs:
        model = onnx.load(str(path), load_external_data=False)
        have = {o.name for o in model.graph.output}
        wanted = targets(model)
        missing = [t for t in wanted if t not in have]
        print(f"{path.name}: {len(model.graph.node)} nodes, {len(wanted)} shape tensors, "
              f"{len(missing)} unpinned")
        if args.check:
            rc |= 1 if missing else 0
        elif missing:
            apply(model)
            onnx.save(model, str(path))
            print(f"    wrote {path.name} with {len(missing)} new pinned output(s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
