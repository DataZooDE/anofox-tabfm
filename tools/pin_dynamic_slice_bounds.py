"""Pin the dynamic `ends` of every ScatterND-feeding Slice as a graph output.

Workaround for an ONNX Runtime CUDA bug (DataZooDE/anofox-tabfm#21): the CUDA
`Slice` that trims a `Range` down to `train_size` returns its input untrimmed,
because the CPU-side buffer holding that bound is recycled before the kernel
reads it. ScatterND then sees `indices` of length T against `updates` of length
S and rejects the shapes. Naming the bound tensors as graph outputs excludes
them from buffer reuse, and the graph runs.

Structural, not name-based: the classification and regression graphs number
their nodes differently, and a re-export renumbers them again. So find the
targets by walking the dataflow:

    ScatterND.indices <- Unsqueeze <- Slice <- (ends, and the scalar behind it)

Idempotent: re-running adds nothing. Verify with --check, which exits non-zero
if a graph is missing its pins (use it in CI so a re-export cannot silently drop
the workaround).
"""

import argparse
import pathlib
import sys

import onnx
from onnx import TensorProto, helper


def targets(graph):
    """Tensors that must survive buffer reuse: each ScatterND-feeding Slice's
    `ends`, plus whatever scalar produces it."""
    producer = {o: n for n in graph.node for o in n.output}
    found = []
    for node in graph.node:
        if node.op_type != "ScatterND" or len(node.input) < 2:
            continue
        cur = producer.get(node.input[1])
        # Skip the reshaping ops between the Slice and ScatterND's `indices`.
        while cur is not None and cur.op_type in ("Unsqueeze", "Squeeze", "Reshape", "Cast"):
            cur = producer.get(cur.input[0])
        if cur is None or cur.op_type != "Slice" or len(cur.input) < 3:
            continue
        ends = cur.input[2]                       # Slice(data, starts, ends, ...)
        found.append((ends, 1))
        upstream = producer.get(ends)
        if upstream is not None and upstream.input:
            # The scalar behind `ends` (Squeeze of a Shape, in these graphs).
            src = upstream.input[0]
            if src in producer:
                found.append((src, 0))
    # Deduplicate, keep order.
    seen, out = set(), []
    for name, rank in found:
        if name not in seen:
            seen.add(name)
            out.append((name, rank))
    return out


def apply(path, check_only):
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    have = {o.name for o in graph.output}
    wanted = targets(graph)
    missing = [(n, r) for n, r in wanted if n not in have]

    print(f"{path.name}: {len(graph.node)} nodes, outputs={[o.name for o in graph.output]}")
    for name, rank in wanted:
        print(f"    pin {name:22s} rank={rank}  {'present' if name in have else 'MISSING'}")

    if check_only:
        return 0 if not missing else 1
    if not missing:
        print("    already pinned; nothing to do")
        return 0
    for name, rank in missing:
        graph.output.append(
            helper.make_tensor_value_info(name, TensorProto.INT64, [] if rank == 0 else [1]))
    onnx.save(model, str(path))
    print(f"    wrote {path.name} with {len(missing)} pinned output(s)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+", type=pathlib.Path)
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 if a pin is missing")
    args = ap.parse_args()
    rc = 0
    for g in args.graphs:
        rc |= apply(g, args.check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
