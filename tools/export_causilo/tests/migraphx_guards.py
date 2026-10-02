"""Structural guards for the four failures MIGraphX's GPU target has on graphs ORT runs happily.

None is visible to ONNX Runtime or to MIGraphX's own reference (CPU) target, so no parity check, gate or ORT run
can see them; they surface only when the GPU backend compiles the graph, after minutes. They were all found on
the TabPFN conversion (docs/ROCM_TABPFN_PLAN.md); this is the Causilo counterpart, written BEFORE the first GPU
run. Each guard is exercised against a deliberately bad graph in tests/test_mask_export_isolated.py, because a
guard that has never failed is not known to work: the first reshape guard written for TabPFN passed on the very
graph MIGraphX rejected (the empty shape is a Constant with `value_ints = []`, not a tensor `value`).

  1. integer division on runtime data: `(n + f - 1) // f` is miscomputed;
  2. `Reshape` to an empty shape (`t.reshape(())`): read as ZERO elements, refused at parse time;
  3. a `Concat` whose inputs are all `Transpose`s: `simplify_reshapes` asserts `s.transposed()`;
  4. a matmul with contraction dimension 1 (`Linear(1, n)` on a `[1, 1]` input) feeding a broadcast Mul:
     `simplify_reshapes` dies, at every feature width tried.

  5. ONNX Sequence operators (`SplitToSequence`, `SequenceAt`, ...): MIGraphX's ONNX parser has none, so the
     graph cannot even be LOADED ("Unknown operator: SplitToSequence"). `Tensor.chunk` exports to exactly that
     pair, and upstream Causilo's `Attention.prepare_context` uses it. Found by parsing the real masked graph
     with migraphx-driver, not by anything that runs on ORT; unlike 1-4 it fails at parse time, so even
     `--ref` sees it.

All of them take a path and return a list of offending node names (empty = clean).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import onnx


def int_divs(path):
    """Integer `Div`s reachable from runtime data. Delegates to the shared tool CI also runs."""
    tool = pathlib.Path(__file__).resolve().parents[2] / "check_migraphx_int_div.py"
    spec = importlib.util.spec_from_file_location("check_migraphx_int_div", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [name or out for name, out in module.int_divs(path)]


def scalar_reshapes(path):
    """`Reshape` whose target shape is empty.

    A shape can be an initializer or a Constant node, and a Constant carries its value as a tensor (`value`) OR
    as a list (`value_ints`); the empty target of `reshape(())` arrives as `value_ints = []`. Sizes come from
    the tensor's dims, not its values: the weights are external data and may be deleted.
    """
    m = onnx.load(str(path), load_external_data=False)
    sizes = {i.name: math.prod(i.dims) for i in m.graph.initializer}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    sizes[n.output[0]] = math.prod(a.t.dims)
                elif a.name in ("value_ints", "value_floats", "value_strings"):
                    sizes[n.output[0]] = len(getattr(a, a.name[len("value_"):]))
                elif a.name in ("value_int", "value_float", "value_string"):
                    sizes[n.output[0]] = 1
    return [n.name or n.output[0] for n in m.graph.node
            if n.op_type == "Reshape" and sizes.get(n.input[1]) == 0]


def concat_of_transposes(path):
    """`Concat` nodes with at least two inputs, ALL produced directly by `Transpose` nodes."""
    m = onnx.load(str(path), load_external_data=False)
    transposed = {o for n in m.graph.node if n.op_type == "Transpose" for o in n.output}
    return [n.name or n.output[0] for n in m.graph.node
            if n.op_type == "Concat" and len(n.input) >= 2 and all(i in transposed for i in n.input)]


def k1_matmuls(path):
    """Gemm/MatMul (directly or through a Transpose) applied to a `[n, 1]` weight: a contraction dim of 1."""
    m = onnx.load(str(path), load_external_data=False)
    dims = {i.name: list(i.dims) for i in m.graph.initializer}
    k1 = {n for n, d in dims.items() if len(d) == 2 and d[1] == 1}
    via = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Transpose" and n.input[0] in k1}
    return [n.name or n.output[0] for n in m.graph.node
            if n.op_type in ("Gemm", "MatMul") and any(i in k1 or i in via for i in n.input)]


SEQUENCE_OPS = {"SplitToSequence", "SequenceAt", "SequenceConstruct", "SequenceInsert", "SequenceErase",
                "SequenceLength", "SequenceEmpty", "ConcatFromSequence"}


def sequence_ops(path):
    """Nodes using an ONNX Sequence operator."""
    m = onnx.load(str(path), load_external_data=False)
    return [n.name or n.output[0] for n in m.graph.node if n.op_type in SEQUENCE_OPS]


GUARDS = {
    "integer division": int_divs,
    "reshape to []": scalar_reshapes,
    "concat of transposes": concat_of_transposes,
    "K=1 matmul": k1_matmuls,
    "sequence ops": sequence_ops,
}
