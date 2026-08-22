"""Route 3 — an ONNX interpreter over MLX ops.

Why this exists, when a working mitra hand-port already ships: the requirement
became *every* registered model on MLX. Hand-porting is one second
implementation per architecture — seven of them, each able to drift from the
graph the other backends run. This executes the shipped graph itself, so
coverage is "which ops are implemented" rather than "which models did somebody
transcribe", and a new model costs zero new math.

Measured scope (`resources/graph_ext_*.onnx`): 13 graphs, 4.5k–12.4k nodes,
33–48 distinct ops each, **64 in union**. Most are elementwise and map 1:1.

Design notes that matter for correctness:

- **Shapes are dynamic.** These graphs came from `torch.export` with dynamic T
  and H, so they compute shapes at runtime via Shape/Slice/Concat and feed them
  to Reshape/Expand. That means the interpreter must carry integer tensors
  faithfully, not just floats — a Shape result silently cast to fp32 breaks
  Reshape for any dimension above 2^24.
- **MLX is lazy.** Nothing is evaluated until asked. Intermediates are kept as
  unevaluated arrays so the whole graph fuses; only the outputs are forced.
- **Failure is loud and specific.** An unimplemented op names itself and the
  node, because the whole point is to extend coverage op by op against a
  parity harness.

This is the Python prototype: it settles op semantics and proves parity per
model before any of it is written in C++ against mlx-c.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import onnx
from onnx import numpy_helper

# ONNX TensorProto dtype -> mlx dtype. Integers stay integers: see the note above.
ONNX_TO_MLX = {
    onnx.TensorProto.FLOAT: mx.float32,
    onnx.TensorProto.DOUBLE: mx.float32,  # MLX has no float64; graphs use it only for constants
    onnx.TensorProto.FLOAT16: mx.float16,
    onnx.TensorProto.BFLOAT16: mx.bfloat16,
    onnx.TensorProto.INT64: mx.int64,
    onnx.TensorProto.INT32: mx.int32,
    onnx.TensorProto.INT8: mx.int8,
    onnx.TensorProto.UINT8: mx.uint8,
    onnx.TensorProto.BOOL: mx.bool_,
}


class UnsupportedOp(NotImplementedError):
    """An op kind the interpreter does not implement yet. Names the node."""


@dataclass
class Node:
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str = ""


def _attr_value(a) -> Any:
    if a.type == onnx.AttributeProto.INT:
        return a.i
    if a.type == onnx.AttributeProto.INTS:
        return list(a.ints)
    if a.type == onnx.AttributeProto.FLOAT:
        return a.f
    if a.type == onnx.AttributeProto.FLOATS:
        return list(a.floats)
    if a.type == onnx.AttributeProto.STRING:
        return a.s.decode()
    if a.type == onnx.AttributeProto.STRINGS:
        return [v.decode() for v in a.strings]
    if a.type == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(a.t)
    raise UnsupportedOp(f"attribute type {a.type} for '{a.name}'")


def _scalar(a) -> Any:
    """ONNX scalars arrive as rank-0 or rank-1 tensors depending on the
    exporter -- torch.export emits rank-1 for K, axes, and friends. Ravel
    first so both shapes read the same."""
    return np.asarray(a).ravel()[0]


def _to_mx(arr: np.ndarray) -> mx.array:
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return mx.array(arr)


class OnnxMlxGraph:
    """A loaded graph, ready to run. Initializers are converted once."""

    def __init__(self, model_path: Path, weights_path: Path | None = None):
        self.path = Path(model_path)
        model = onnx.load(str(self.path), load_external_data=False)
        self.model = model
        self.graph = model.graph
        self.inputs = [i.name for i in self.graph.input]
        self.outputs = [o.name for o in self.graph.output]

        # External initializers reference the safetensors by absolute offset;
        # load them through onnx's own resolver so the offsets are honoured
        # exactly the way ORT honours them.
        base = Path(weights_path).parent if weights_path else self.path.parent
        onnx.load_external_data_for_model(model, str(base))
        self.initializers = {t.name: _to_mx(numpy_helper.to_array(t)) for t in self.graph.initializer}

        self.nodes = [
            Node(op=n.op_type, inputs=list(n.input), outputs=list(n.output),
                 attrs={a.name: _attr_value(a) for a in n.attribute}, name=n.name)
            for n in self.graph.node
        ]

    def op_kinds(self) -> set[str]:
        return {n.op for n in self.nodes}

    def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        env: dict[str, Any] = dict(self.initializers)
        for name, value in feeds.items():
            env[name] = _to_mx(np.asarray(value))

        for node in self.nodes:
            handler = HANDLERS.get(node.op)
            if handler is None:
                raise UnsupportedOp(
                    f"op '{node.op}' (node '{node.name}') is not implemented. "
                    f"Graph: {self.path.name}"
                )
            args = [env.get(i) if i else None for i in node.inputs]
            try:
                results = handler(node, args, env)
            except UnsupportedOp:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raise with the node named
                raise RuntimeError(f"{node.op} node '{node.name}' failed: {exc}") from exc
            if not isinstance(results, (list, tuple)):
                results = [results]
            for out_name, value in zip(node.outputs, results):
                if out_name:
                    env[out_name] = value

        out = {}
        for name in self.outputs:
            value = env[name]
            mx.eval(value)
            out[name] = np.asarray(value)
        return out


# --------------------------------------------------------------------------- #
# Op handlers
#
# Signature: (node, inputs, env) -> array | list[array]. `env` is available for
# the few ops whose semantics depend on more than their inputs.
# --------------------------------------------------------------------------- #
def _binary(fn: Callable) -> Callable:
    def handler(node, ins, env):
        return fn(ins[0], ins[1])
    return handler


def _unary(fn: Callable) -> Callable:
    def handler(node, ins, env):
        return fn(ins[0])
    return handler


def _op_constant(node, ins, env):
    if "value" in node.attrs:
        return _to_mx(np.asarray(node.attrs["value"]))
    for key, cast in (("value_int", np.int64), ("value_ints", np.int64),
                      ("value_float", np.float32), ("value_floats", np.float32)):
        if key in node.attrs:
            return _to_mx(np.asarray(node.attrs[key], dtype=cast))
    raise UnsupportedOp(f"Constant node '{node.name}' has no value attribute")


def _op_cast(node, ins, env):
    return ins[0].astype(ONNX_TO_MLX[node.attrs["to"]])


def _op_cast_like(node, ins, env):
    return ins[0].astype(ins[1].dtype)


def _op_shape(node, ins, env):
    shape = np.array(ins[0].shape, dtype=np.int64)
    start = node.attrs.get("start", 0)
    end = node.attrs.get("end", len(shape))
    return mx.array(shape[start:end])


def _op_reshape(node, ins, env):
    shape = [int(v) for v in np.asarray(ins[1])]
    if node.attrs.get("allowzero", 0) == 0:
        # ONNX: a 0 means "copy the corresponding input dim"; MLX has no such rule.
        shape = [ins[0].shape[i] if v == 0 else v for i, v in enumerate(shape)]
    return mx.reshape(ins[0], shape)


def _op_transpose(node, ins, env):
    perm = node.attrs.get("perm")
    if perm is None:
        perm = list(range(ins[0].ndim))[::-1]
    return mx.transpose(ins[0], list(perm))


def _op_concat(node, ins, env):
    return mx.concatenate([i for i in ins if i is not None], axis=node.attrs["axis"])


def _op_unsqueeze(node, ins, env):
    axes = [int(v) for v in np.asarray(ins[1])] if len(ins) > 1 else node.attrs["axes"]
    out = ins[0]
    for ax in sorted(int(a) for a in axes):
        out = mx.expand_dims(out, ax)
    return out


def _op_squeeze(node, ins, env):
    if len(ins) > 1 and ins[1] is not None:
        axes = [int(v) for v in np.asarray(ins[1])]
    else:
        axes = node.attrs.get("axes")
    if axes is None:
        return mx.squeeze(ins[0])
    return mx.squeeze(ins[0], [int(a) for a in axes])


def _op_gather(node, ins, env):
    return mx.take(ins[0], ins[1].astype(mx.int32), axis=node.attrs.get("axis", 0))


def _op_gather_elements(node, ins, env):
    return mx.take_along_axis(ins[0], ins[1].astype(mx.int32), axis=node.attrs.get("axis", 0))


def _op_slice(node, ins, env):
    data = ins[0]
    starts = [int(v) for v in np.asarray(ins[1])]
    ends = [int(v) for v in np.asarray(ins[2])]
    axes = [int(v) for v in np.asarray(ins[3])] if len(ins) > 3 and ins[3] is not None \
        else list(range(len(starts)))
    steps = [int(v) for v in np.asarray(ins[4])] if len(ins) > 4 and ins[4] is not None \
        else [1] * len(starts)
    out = data
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = axis if axis >= 0 else axis + data.ndim
        dim = out.shape[axis]
        # ONNX clamps out-of-range bounds rather than erroring, and exporters
        # lean on that hard (INT64_MAX means "to the end").
        start = max(-dim, min(start, dim)) if step > 0 else max(-dim - 1, min(start, dim - 1))
        end = max(-dim - 1, min(end, dim))
        idx = [slice(None)] * out.ndim
        idx[axis] = slice(start, end, step)
        out = out[tuple(idx)]
    return out


def _op_expand(node, ins, env):
    target = [int(v) for v in np.asarray(ins[1])]
    # ONNX Expand is bidirectional broadcast, not a plain broadcast_to: the
    # result rank is max(rank(a), len(shape)) and 1s on EITHER side stretch.
    return ins[0] * mx.ones(target, dtype=ins[0].dtype)


def _op_range(node, ins, env):
    start = float(_scalar(ins[0]))
    limit = float(_scalar(ins[1]))
    delta = float(_scalar(ins[2]))
    dtype = ins[0].dtype
    return mx.arange(start, limit, delta, dtype=dtype)


def _op_reduce(fn: Callable, keep_default: int = 1) -> Callable:
    def handler(node, ins, env):
        keepdims = bool(node.attrs.get("keepdims", keep_default))
        if len(ins) > 1 and ins[1] is not None:
            axes = [int(v) for v in np.asarray(ins[1])]
        else:
            axes = node.attrs.get("axes")
        if axes is None:
            if node.attrs.get("noop_with_empty_axes", 0):
                return ins[0]
            return fn(ins[0], keepdims=keepdims)
        return fn(ins[0], axis=[int(a) for a in axes], keepdims=keepdims)
    return handler


def _op_softmax(node, ins, env):
    return mx.softmax(ins[0], axis=node.attrs.get("axis", -1))


def _op_layer_norm(node, ins, env):
    x, scale = ins[0], ins[1]
    bias = ins[2] if len(ins) > 2 else None
    eps = float(node.attrs.get("epsilon", 1e-5))
    axis = int(node.attrs.get("axis", -1))
    if axis not in (-1, x.ndim - 1):
        raise UnsupportedOp(f"LayerNormalization over axis {axis} (only the last axis is implemented)")
    return mx.fast.layer_norm(x, scale, bias, eps)


def _op_matmul(node, ins, env):
    return ins[0] @ ins[1]


def _op_gemm(node, ins, env):
    a, b = ins[0], ins[1]
    if node.attrs.get("transA", 0):
        a = a.T
    if node.attrs.get("transB", 0):
        b = b.T
    out = (a @ b) * float(node.attrs.get("alpha", 1.0))
    if len(ins) > 2 and ins[2] is not None:
        out = out + ins[2] * float(node.attrs.get("beta", 1.0))
    return out


def _op_clip(node, ins, env):
    lo = ins[1] if len(ins) > 1 and ins[1] is not None else None
    hi = ins[2] if len(ins) > 2 and ins[2] is not None else None
    out = ins[0]
    if lo is not None:
        out = mx.maximum(out, lo.astype(out.dtype))
    if hi is not None:
        out = mx.minimum(out, hi.astype(out.dtype))
    return out


def _op_where(node, ins, env):
    return mx.where(ins[0], ins[1], ins[2])


def _op_topk(node, ins, env):
    k = int(_scalar(ins[1]))
    axis = int(node.attrs.get("axis", -1))
    largest = bool(node.attrs.get("largest", 1))
    data = ins[0] if largest else -ins[0]
    idx = mx.argsort(-data, axis=axis)
    take = [slice(None)] * data.ndim
    take[axis] = slice(0, k)
    idx_k = idx[tuple(take)]
    values = mx.take_along_axis(ins[0], idx_k, axis=axis)
    return [values, idx_k.astype(mx.int64)]


def _op_pow(node, ins, env):
    return mx.power(ins[0], ins[1].astype(ins[0].dtype))


def _op_size(node, ins, env):
    return mx.array(np.int64(int(np.prod(ins[0].shape))))


def _op_identity(node, ins, env):
    return ins[0]


def _op_cumsum(node, ins, env):
    axis = int(_scalar(ins[1]))
    return mx.cumsum(ins[0], axis=axis,
                     reverse=bool(node.attrs.get("reverse", 0)),
                     inclusive=not bool(node.attrs.get("exclusive", 0)))



# --------------------------------------------------------------------------- #
# The ops the plan flagged as "more work than it looks". They are the entire
# gap between mitra/tabfm-v1 (covered by the handlers above) and the rest of
# the catalogue, so each is written against its ONNX spec rather than against
# the one call site that happens to need it today.
# --------------------------------------------------------------------------- #
def _op_pad(node, ins, env):
    data = ins[0]
    pads = [int(v) for v in np.asarray(ins[1])]
    value = ins[2] if len(ins) > 2 and ins[2] is not None else None
    axes = [int(v) for v in np.asarray(ins[3])] if len(ins) > 3 and ins[3] is not None else None
    mode = node.attrs.get("mode", "constant")
    if mode != "constant":
        raise UnsupportedOp(f"Pad mode '{mode}' (only 'constant' is implemented)")
    # ONNX packs pads as [start_0..start_n, end_0..end_n] over `axes`
    # (all axes when absent), NOT as per-axis pairs.
    rank = data.ndim
    half = len(pads) // 2
    widths = [[0, 0] for _ in range(rank)]
    target_axes = axes if axes is not None else list(range(rank))
    for i, axis in enumerate(target_axes):
        axis = axis if axis >= 0 else axis + rank
        widths[axis] = [pads[i], pads[half + i]]
    pad_value = mx.array(0, dtype=data.dtype) if value is None else value.astype(data.dtype)
    return mx.pad(data, widths, constant_values=pad_value)


def _op_scatter_elements(node, ins, env):
    data, indices, updates = ins[0], ins[1], ins[2]
    axis = int(node.attrs.get("axis", 0))
    reduction = node.attrs.get("reduction", "none")
    if reduction != "none":
        raise UnsupportedOp(f"ScatterElements reduction '{reduction}' (only 'none' is implemented)")
    idx = indices.astype(mx.int32)
    # ONNX allows negative indices; put_along_axis does not.
    dim = data.shape[axis if axis >= 0 else axis + data.ndim]
    idx = mx.where(idx < 0, idx + dim, idx)
    return mx.put_along_axis(data, idx, updates.astype(data.dtype), axis=axis)


def _op_scatter_nd(node, ins, env):
    data, indices, updates = ins[0], ins[1], ins[2]
    reduction = node.attrs.get("reduction", "none")
    if reduction != "none":
        raise UnsupportedOp(f"ScatterND reduction '{reduction}' (only 'none' is implemented)")
    # indices[..., k] addresses the first k axes; the remaining axes come along
    # whole. Flatten those k axes into one so a single put_along_axis does it.
    idx_np = np.asarray(indices).astype(np.int64)
    k = idx_np.shape[-1]
    lead = list(data.shape[:k])
    tail = list(data.shape[k:])
    flat_lead = int(np.prod(lead)) if lead else 1
    strides = np.ones(k, dtype=np.int64)
    for i in range(k - 2, -1, -1):
        strides[i] = strides[i + 1] * lead[i + 1]
    flat_idx = (idx_np.reshape(-1, k) % np.array(lead, dtype=np.int64) * strides).sum(-1)

    flat_data = mx.reshape(data, [flat_lead] + tail)
    flat_upd = mx.reshape(updates, [flat_idx.shape[0]] + tail).astype(data.dtype)
    idx_arr = mx.array(flat_idx.astype(np.int32))
    for _ in tail:
        idx_arr = mx.expand_dims(idx_arr, -1)
    idx_arr = mx.broadcast_to(idx_arr, [flat_idx.shape[0]] + tail)
    out = mx.put_along_axis(flat_data, idx_arr, flat_upd, axis=0)
    return mx.reshape(out, list(data.shape))


def _op_gather_nd(node, ins, env):
    data, indices = ins[0], ins[1]
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise UnsupportedOp(f"GatherND batch_dims={batch_dims} (only 0 is implemented)")
    idx_np = np.asarray(indices).astype(np.int64)
    k = idx_np.shape[-1]
    lead = list(data.shape[:k])
    tail = list(data.shape[k:])
    strides = np.ones(k, dtype=np.int64)
    for i in range(k - 2, -1, -1):
        strides[i] = strides[i + 1] * lead[i + 1]
    flat_idx = (idx_np.reshape(-1, k) % np.array(lead, dtype=np.int64) * strides).sum(-1)
    flat_data = mx.reshape(data, [int(np.prod(lead))] + tail)
    picked = mx.take(flat_data, mx.array(flat_idx.astype(np.int32)), axis=0)
    return mx.reshape(picked, list(idx_np.shape[:-1]) + tail)


def _op_einsum(node, ins, env):
    return mx.einsum(node.attrs["equation"], *[i for i in ins if i is not None])


def _op_split_to_sequence(node, ins, env):
    """Returns a PYTHON LIST into env -- ONNX sequences are values too, and the
    interpreter's env is untyped, so a list is the natural representation."""
    data = ins[0]
    axis = int(node.attrs.get("axis", 0))
    keepdims = int(node.attrs.get("keepdims", 1))
    if len(ins) > 1 and ins[1] is not None:
        split = np.asarray(ins[1]).astype(np.int64)
        if split.ndim == 0:  # scalar: chunk size
            size = int(split)
            dim = data.shape[axis]
            bounds = list(range(size, dim, size))
            parts = mx.split(data, bounds, axis=axis)
        else:
            bounds = list(np.cumsum(split)[:-1])
            parts = mx.split(data, [int(b) for b in bounds], axis=axis)
    else:
        parts = mx.split(data, data.shape[axis], axis=axis)
        if not keepdims:
            parts = [mx.squeeze(p, axis) for p in parts]
    # A single-output node returning a list would be unpacked as multiple
    # outputs by run(); wrap so the list lands intact.
    return [list(parts)]


def _op_sequence_at(node, ins, env):
    seq = ins[0]
    pos = int(_scalar(ins[1]))
    return seq[pos]


HANDLERS: dict[str, Callable] = {
    # elementwise / arithmetic
    "Add": _binary(lambda a, b: a + b),
    "Sub": _binary(lambda a, b: a - b),
    "Mul": _binary(lambda a, b: a * b),
    "Div": _binary(lambda a, b: a / b),
    "Pow": _op_pow,
    "Neg": _unary(lambda a: -a),
    "Abs": _unary(mx.abs),
    "Sqrt": _unary(mx.sqrt),
    "Reciprocal": _unary(lambda a: 1.0 / a),
    "Exp": _unary(mx.exp),
    "Log": _unary(mx.log),
    "Erf": _unary(mx.erf),
    "Sin": _unary(mx.sin),
    "Cos": _unary(mx.cos),
    "Tanh": _unary(mx.tanh),
    "Sigmoid": _unary(mx.sigmoid),
    "Softplus": _unary(lambda a: mx.log1p(mx.exp(a))),
    "Sign": _unary(mx.sign),
    "Floor": _unary(mx.floor),
    "Ceil": _unary(mx.ceil),
    "IsNaN": _unary(mx.isnan),
    "IsInf": _unary(mx.isinf),
    "Max": lambda node, ins, env: _fold(mx.maximum, ins),
    "Min": lambda node, ins, env: _fold(mx.minimum, ins),
    "Mod": _binary(lambda a, b: a % b),
    # comparison / logic
    "Equal": _binary(mx.equal),
    "Greater": _binary(mx.greater),
    "GreaterOrEqual": _binary(mx.greater_equal),
    "Less": _binary(mx.less),
    "LessOrEqual": _binary(mx.less_equal),
    "And": _binary(mx.logical_and),
    "Or": _binary(mx.logical_or),
    "Not": _unary(mx.logical_not),
    "Where": _op_where,
    # shape / structure
    "Constant": _op_constant,
    "Cast": _op_cast,
    "CastLike": _op_cast_like,
    "Shape": _op_shape,
    "Size": _op_size,
    "Reshape": _op_reshape,
    "Transpose": _op_transpose,
    "Concat": _op_concat,
    "Unsqueeze": _op_unsqueeze,
    "Squeeze": _op_squeeze,
    "Expand": _op_expand,
    "Slice": _op_slice,
    "Gather": _op_gather,
    "GatherElements": _op_gather_elements,
    "Range": _op_range,
    "Identity": _op_identity,
    "Clip": _op_clip,
    # reductions
    "ReduceSum": _op_reduce(mx.sum),
    "ReduceMean": _op_reduce(mx.mean),
    "ReduceMin": _op_reduce(mx.min),
    "ReduceMax": _op_reduce(mx.max),
    "CumSum": _op_cumsum,
    "TopK": _op_topk,
    # nn
    "MatMul": _op_matmul,
    "Gemm": _op_gemm,
    "Softmax": _op_softmax,
    "LayerNormalization": _op_layer_norm,
    # the catalogue's remaining ops
    "Pad": _op_pad,
    "ScatterElements": _op_scatter_elements,
    "ScatterND": _op_scatter_nd,
    "GatherND": _op_gather_nd,
    "Einsum": _op_einsum,
    "SplitToSequence": _op_split_to_sequence,
    "SequenceAt": _op_sequence_at,
}


def _fold(fn, ins):
    out = ins[0]
    for other in ins[1:]:
        if other is not None:
            out = fn(out, other)
    return out
