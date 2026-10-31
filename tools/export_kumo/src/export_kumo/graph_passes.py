"""Graph-level rewrites applied to every exported Kumo-Tabular graph (CPU, ext and migraphx alike).

`split_sequences_to_slices`: torch's dynamo exporter lowers `Tensor.split` / `chunk` to
`SplitToSequence` + `SequenceAt`. MIGraphX's ONNX parser implements neither (the same wall
docs/ROCM_TABDPT_SPIKE.md hit as "blocker 2"), and the MLX interpreter has neither in its op table.
Kumo splits in three places -- the fused QKV projection, RoPE's half-split and the readout/column
split in the row embedding -- 83 times in the small graph. Rewriting each `SequenceAt(SplitToSequence(X,
split, axis), p)` into `Slice(X, start_p, end_p, axis)` is the identical tensor and runs everywhere.
Done on the graph rather than by patching upstream's modules, so no upstream forward is copied.
"""

from __future__ import annotations

import collections

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _const_value(name, producers, inits):
    if name in inits:
        return numpy_helper.to_array(inits[name])
    node = producers.get(name)
    if node is not None and node.op_type == "Constant":
        for a in node.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    return None


def split_sequences_to_slices(model: onnx.ModelProto) -> int:
    """Rewrite in place; returns the number of SequenceAt nodes replaced. Raises if a sequence
    op survives (an unhandled form must fail the export, not ship)."""
    g = model.graph
    producers = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    consumers = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)

    new_nodes: list[onnx.NodeProto] = []
    replaced = 0
    counter = 0

    def fresh(prefix):
        nonlocal counter
        counter += 1
        return f"_kumo_s2s_{prefix}_{counter}"

    def const_i64(values):
        name = fresh("c")
        new_nodes.append(helper.make_node(
            "Constant", [], [name],
            value=numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)))
        return name

    replacement: dict[int, list[onnx.NodeProto]] = {}
    for node in g.node:
        if node.op_type != "SequenceAt":
            continue
        split_node = producers.get(node.input[0])
        if split_node is None or split_node.op_type != "SplitToSequence":
            raise RuntimeError(f"SequenceAt {node.name} does not read a SplitToSequence")
        attrs = {a.name: helper.get_attribute_value(a) for a in split_node.attribute}
        if attrs.get("keepdims", 1) != 1:
            raise RuntimeError(f"{split_node.name}: keepdims=0 is not handled")
        axis = int(attrs.get("axis", 0))
        if len(split_node.input) < 2 or not split_node.input[1]:
            raise RuntimeError(f"{split_node.name}: a split without sizes is not handled")
        p = _const_value(node.input[1], producers, inits)
        if p is None:
            raise RuntimeError(f"{node.name}: non-constant sequence position")
        p = int(np.asarray(p).reshape(()))
        split_name = split_node.input[1]
        split_val = _const_value(split_name, producers, inits)

        new_nodes.clear()
        axes = const_i64([axis])
        if split_val is not None:
            if np.asarray(split_val).ndim == 0:
                # Equal chunks of size s: chunk p is [p*s, (p+1)*s); Slice clamps a short last one.
                chunk = int(split_val)
                if p < 0:
                    raise RuntimeError(f"{split_node.name}: negative position into equal chunks")
                starts = const_i64([p * chunk])
                ends = const_i64([(p + 1) * chunk])
                new_nodes.append(helper.make_node("Slice", [split_node.input[0], starts, ends, axes],
                                                  [node.output[0]], name=fresh("slice")))
                replacement[id(node)] = list(new_nodes)
                replaced += 1
                continue
            split_val = np.asarray(split_val).reshape(-1)
            if p < 0:
                p += split_val.size
            start = int(split_val[:p].sum())
            starts = const_i64([start])
            ends = const_i64([start + int(split_val[p])])
        else:
            # Runtime split sizes (shape-derived, e.g. [K, C] in the row embedding): start is the
            # sum of the sizes before p, end adds size p. No CumSum, no division.
            if p < 0:
                raise RuntimeError(f"{split_node.name}: negative position with runtime sizes")
            size_p = fresh("size")
            new_nodes.append(helper.make_node("Slice", [split_name, const_i64([p]), const_i64([p + 1]),
                                                        const_i64([0])], [size_p]))
            if p == 0:
                starts = const_i64([0])
            else:
                head = fresh("head")
                new_nodes.append(helper.make_node("Slice", [split_name, const_i64([0]), const_i64([p]),
                                                            const_i64([0])], [head]))
                starts = fresh("start")
                new_nodes.append(helper.make_node("ReduceSum", [head], [starts], keepdims=1))
            ends = fresh("end")
            new_nodes.append(helper.make_node("Add", [starts, size_p], [ends]))
        new_nodes.append(helper.make_node("Slice", [split_node.input[0], starts, ends, axes],
                                          [node.output[0]], name=fresh("slice")))
        replacement[id(node)] = list(new_nodes)
        replaced += 1

    out_nodes = []
    for node in g.node:
        if id(node) in replacement:
            out_nodes.extend(replacement[id(node)])
        elif node.op_type == "SplitToSequence":
            if any(c.op_type != "SequenceAt" for c in consumers[node.output[0]]):
                raise RuntimeError(f"{node.name}: sequence consumed by something other than SequenceAt")
            continue  # every consumer was rewritten
        else:
            out_nodes.append(node)
    del g.node[:]
    g.node.extend(out_nodes)
    left = [n.op_type for n in g.node if n.op_type in ("SplitToSequence", "SequenceAt")]
    if left:
        raise RuntimeError(f"sequence ops survived the rewrite: {left[:5]}")
    return replaced
