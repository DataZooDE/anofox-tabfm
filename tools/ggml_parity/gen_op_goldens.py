# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "onnxruntime", "numpy"]
# ///
"""Per-op goldens for the ggml CPU engine (docs/GGML_PLAN.md, P3), from ONNX Runtime (the oracle).

    uv run tools/ggml_parity/gen_op_goldens.py

Writes test/fixtures/ggml_ops/<case>.onnx (a tiny graph, no weights) and <case>.golden (inputs and ORT's outputs).
CI verifies these files, it never regenerates them (like test/fixtures). Golden format, one tensor per line:

    in|out <name> <f32|i64|bool> <d0,d1,...|-> <value> <value> ...

`-` is a rank-0 tensor. f32 values are the IEEE bit pattern in 8 hex digits (so NaN and +-inf survive exactly);
i64 values are decimal; bool values are 0/1. Every case carries adversarial inputs (NaN, +-inf, zeros, mixed
broadcast), because the masks and the normalisation of the bundled graphs live on those.
"""
import os
import struct
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "test", "fixtures", "ggml_ops")
rng = np.random.default_rng(20260721)
NAN, INF = np.float32("nan"), np.float32("inf")


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def spiked(*shape):
    """Random floats with NaN, +-inf, +-0 and large values scattered in at distinct positions: every operand, however
    small, carries specials (a size-3 broadcast operand gets NaN, +inf, -inf), so no case can pass on clean data."""
    a = rand(*shape)
    flat = a.reshape(-1)
    specials = [NAN, INF, -INF, 0.0, -0.0, 1e30, -1e30, 1e-30]
    count = min(len(specials), max(1, flat.size // 2 if flat.size < 8 else len(specials)))
    positions = rng.permutation(flat.size)[:count]
    for pos, v in zip(positions, specials):
        flat[pos] = v
    return flat.reshape(shape)


def positive(*shape):
    return (np.abs(rand(*shape)) + 0.1).astype(np.float32)


def vi(name, arr):
    dt = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("int32"): TensorProto.INT32, np.dtype("bool"): TensorProto.BOOL}[arr.dtype]
    return helper.make_tensor_value_info(name, dt, list(arr.shape))


def fmt(kind, name, arr):
    arr = np.asarray(arr)
    dims = ",".join(str(d) for d in arr.shape) if arr.ndim else "-"
    if arr.dtype == np.float32:
        vals = " ".join("%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in arr.reshape(-1))
        dt = "f32"
    elif arr.dtype in (np.int64, np.int32):  # int32 is written as i64: the runner compares integers as exact floats
        vals, dt = " ".join(str(int(v)) for v in arr.reshape(-1)), "i64"
    else:
        vals, dt = " ".join("1" if v else "0" for v in arr.reshape(-1)), "bool"
    return f"{kind} {name} {dt} {dims} {vals}".rstrip()


def case(name, op, inputs, nodes=None, inits=None, **attrs):
    """One case: `op` over float `inputs` {name: array}; `inits` are baked into the graph as initializers."""
    inits = inits or {}
    inputs = {n: np.asarray(a) for n, a in inputs.items()}  # np.float32 scalars -> 0-d arrays
    if nodes is None:
        nodes = [helper.make_node(op, list(inputs) + list(inits), ["y"], **attrs)]
    feeds = dict(inputs)
    graph_inputs = [vi(n, a) for n, a in inputs.items()]
    initializers = [numpy_helper.from_array(a, n) for n, a in inits.items()]
    probe = helper.make_graph(nodes, name, graph_inputs, [helper.make_empty_tensor_value_info("y")], initializer=initializers)
    model = helper.make_model(probe, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # the plain kernels, not fused/folded ones
    sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    (result,) = sess.run(["y"], feeds)
    dt = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("int32"): TensorProto.INT32, np.dtype("bool"): TensorProto.BOOL}[result.dtype]
    graph = helper.make_graph(nodes, name, graph_inputs, [helper.make_tensor_value_info("y", dt, list(result.shape))], initializer=initializers)
    final = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    final.ir_version = 10
    onnx.checker.check_model(final)
    with open(os.path.join(OUT, name + ".onnx"), "wb") as f:
        f.write(final.SerializeToString())
    with open(os.path.join(OUT, name + ".golden"), "w") as f:
        f.write(f"# op {op}\n")
        for n, a in inputs.items():
            f.write(fmt("in", n, a) + "\n")
        f.write(fmt("out", "y", result) + "\n")


os.makedirs(OUT, exist_ok=True)
for stale in os.listdir(OUT):
    os.remove(os.path.join(OUT, stale))

# --- plumbing ---
case("identity_2d", "Identity", {"x": spiked(3, 4)})
case("identity_scalar", "Identity", {"x": np.float32(2.5)})
for tag, shape, target in [("flat", (2, 3, 4), [24]), ("split", (24,), [2, 3, 4]), ("infer", (2, 3, 4), [4, -1]), ("keep0", (2, 3, 4), [0, -1]),
                           ("from_scalar", (), [1, 1])]:
    case(f"reshape_{tag}", "Reshape", {"x": spiked(*shape) if shape else np.float32(1.5)}, inits={"shape": np.array(target, dtype=np.int64)})
case("squeeze_axes", "Squeeze", {"x": spiked(1, 3, 1, 4)}, inits={"axes": np.array([0, 2], dtype=np.int64)})
case("squeeze_negative", "Squeeze", {"x": spiked(3, 1, 4)}, inits={"axes": np.array([-2], dtype=np.int64)})
case("squeeze_all", "Squeeze", {"x": spiked(1, 3, 1, 1)})
case("unsqueeze_front", "Unsqueeze", {"x": spiked(3, 4)}, inits={"axes": np.array([0], dtype=np.int64)})
case("unsqueeze_mixed", "Unsqueeze", {"x": spiked(3, 4)}, inits={"axes": np.array([0, -1], dtype=np.int64)})

# --- unary ---
for op in ["Neg", "Abs", "Sqrt", "Exp", "Log", "Tanh"]:
    case(f"{op.lower()}_spiked", op, {"x": spiked(5, 7)})
case("sqrt_positive", "Sqrt", {"x": positive(4, 4)})
case("log_positive", "Log", {"x": positive(4, 4)})

# --- binary arithmetic, numpy broadcasting in both directions ---
BROADCASTS = [("same", (3, 4), (3, 4)), ("b_row", (3, 4), (4,)), ("b_col", (3, 4), (3, 1)), ("a_row", (4,), (3, 4)),
              ("a_col", (3, 1), (3, 4)), ("both", (3, 1), (1, 4)), ("scalar_b", (3, 4), ()), ("scalar_a", (), (3, 4)),
              ("rank3", (2, 3, 4), (3, 4)), ("rank4_mid", (2, 1, 3, 4), (2, 5, 1, 4))]
for op in ["Add", "Sub", "Mul", "Div"]:
    for tag, sa, sb in BROADCASTS:
        a = spiked(*sa) if sa else np.float32(1.5)
        b = spiked(*sb) if sb else np.float32(-0.5)
        case(f"{op.lower()}_{tag}", op, {"a": a, "b": b})
case("div_by_zero", "Div", {"a": np.array([1, -1, 0, NAN, INF], dtype=np.float32), "b": np.array([0, 0, 0, 0, 0], dtype=np.float32)})
case("mul_zero_inf", "Mul", {"a": np.array([0, INF, NAN, 1], dtype=np.float32), "b": np.array([INF, 0, 0, NAN], dtype=np.float32)})

# --- compute ---
case("transpose_2d", "Transpose", {"x": spiked(3, 4)}, perm=[1, 0])
case("transpose_3d", "Transpose", {"x": spiked(2, 3, 4)}, perm=[2, 0, 1])
case("transpose_4d", "Transpose", {"x": spiked(2, 3, 4, 5)}, perm=[0, 2, 1, 3])
case("transpose_default", "Transpose", {"x": spiked(2, 3, 4)})
case("matmul_2d", "MatMul", {"a": rand(3, 4), "b": rand(4, 5)})
case("matmul_batched", "MatMul", {"a": rand(2, 3, 4), "b": rand(2, 4, 5)})
case("matmul_bcast_b", "MatMul", {"a": rand(2, 3, 4), "b": rand(4, 5)})
case("matmul_bcast_a", "MatMul", {"a": rand(3, 4), "b": rand(2, 4, 5)})
case("matmul_rank4", "MatMul", {"a": rand(2, 3, 4, 5), "b": rand(2, 3, 5, 6)})
case("matmul_rank4_bcast", "MatMul", {"a": rand(2, 1, 4, 5), "b": rand(1, 3, 5, 6)})
case("matmul_vec", "MatMul", {"a": rand(4), "b": rand(4, 5)})
case("matmul_nan", "MatMul", {"a": spiked(3, 4), "b": rand(4, 5)})
case("softmax_last", "Softmax", {"x": rand(3, 5)}, axis=-1)
case("softmax_first", "Softmax", {"x": rand(3, 5)}, axis=0)
case("softmax_rank3_mid", "Softmax", {"x": rand(2, 3, 4)}, axis=1)
case("softmax_nan_row", "Softmax", {"x": np.array([[1, NAN, 2], [1, 2, 3]], dtype=np.float32)}, axis=-1)
case("softmax_inf", "Softmax", {"x": np.array([[INF, 1, 2], [-INF, -INF, 1], [-INF, -INF, -INF]], dtype=np.float32)}, axis=-1)

# --- reductions, normalisation, structure ---
def ax(*a):
    return np.array(a, dtype=np.int64)


for op in ["ReduceMean", "ReduceSum"]:
    tag = op.lower()
    case(f"{tag}_last_keep", op, {"x": spiked(3, 5)}, inits={"axes": ax(-1)})
    case(f"{tag}_last_drop", op, {"x": spiked(3, 5)}, inits={"axes": ax(-1)}, keepdims=0)
    case(f"{tag}_first", op, {"x": spiked(3, 5)}, inits={"axes": ax(0)})
    case(f"{tag}_mid", op, {"x": rand(2, 3, 4)}, inits={"axes": ax(1)})
    case(f"{tag}_two_axes", op, {"x": rand(2, 3, 4)}, inits={"axes": ax(0, 2)}, keepdims=0)
    case(f"{tag}_all", op, {"x": rand(2, 3, 4)})
    case(f"{tag}_rank4", op, {"x": rand(2, 3, 4, 5)}, inits={"axes": ax(2)})
case("layernorm_last", "LayerNormalization", {"x": rand(3, 6)}, inits={"scale": rand(6), "bias": rand(6)}, axis=-1, epsilon=1e-5)
case("layernorm_noaffine_bias", "LayerNormalization", {"x": rand(4, 8)}, inits={"scale": positive(8)}, axis=-1, epsilon=1e-3)
case("layernorm_rank3", "LayerNormalization", {"x": rand(2, 3, 8)}, inits={"scale": rand(8), "bias": rand(8)}, axis=-1)
case("layernorm_two_trailing", "LayerNormalization", {"x": rand(2, 3, 4)}, inits={"scale": rand(3, 4), "bias": rand(3, 4)}, axis=1)
case("layernorm_nan_row", "LayerNormalization", {"x": np.array([[1, NAN, 3, 4], [1, 2, 3, 4]], dtype=np.float32)}, inits={"scale": np.ones(4, np.float32), "bias": np.zeros(4, np.float32)}, axis=-1)
case("layernorm_constant_row", "LayerNormalization", {"x": np.full((2, 4), 3.0, dtype=np.float32)}, inits={"scale": np.ones(4, np.float32), "bias": np.zeros(4, np.float32)}, axis=-1)
case("expand_row", "Expand", {"x": spiked(1, 4)}, inits={"shape": np.array([3, 4], dtype=np.int64)})
case("expand_rank_up", "Expand", {"x": spiked(4)}, inits={"shape": np.array([2, 3, 4], dtype=np.int64)})
case("expand_one_in_target", "Expand", {"x": spiked(3, 1)}, inits={"shape": np.array([1, 4], dtype=np.int64)})
case("concat_axis0", "Concat", {"a": spiked(2, 3), "b": spiked(4, 3)}, axis=0)
case("concat_axis_last", "Concat", {"a": spiked(2, 3), "b": spiked(2, 2), "c": spiked(2, 1)}, axis=-1)
case("concat_rank4_mid", "Concat", {"a": rand(2, 1, 3, 2), "b": rand(2, 4, 3, 2)}, axis=1)
case("slice_basic", "Slice", {"x": spiked(4, 6)}, inits={"starts": ax(1), "ends": ax(3), "axes": ax(1)})
case("slice_negative", "Slice", {"x": spiked(4, 6)}, inits={"starts": ax(-3), "ends": ax(-1), "axes": ax(1)})
case("slice_to_end", "Slice", {"x": spiked(4, 6)}, inits={"starts": ax(2), "ends": ax(np.iinfo(np.int64).max), "axes": ax(0)})
case("slice_two_axes", "Slice", {"x": rand(3, 4, 5)}, inits={"starts": ax(1, 0), "ends": ax(3, 2), "axes": ax(0, 2)})
case("slice_step2", "Slice", {"x": rand(8, 3)}, inits={"starts": ax(1), "ends": ax(8), "axes": ax(0), "steps": ax(2)})
case("slice_clamped", "Slice", {"x": rand(3, 4)}, inits={"starts": ax(-100), "ends": ax(100), "axes": ax(1)})
case("slice_empty_axes_default", "Slice", {"x": rand(3, 4)}, inits={"starts": ax(1, 1), "ends": ax(3, 3)})

# --- the mask chain: comparisons, logic, Where, IsNaN/IsInf (bool tensors; NaN must never leak through a mask) ---
def mask(*shape):
    return rng.random(shape) > 0.5


CMP = [("same", (3, 4), (3, 4)), ("bcast", (3, 1), (1, 4)), ("scalar", (3, 4), ())]
for op in ["Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"]:
    for tag, sa, sb in CMP:
        a = spiked(*sa) if sa else np.float32(0.0)
        b = spiked(*sb) if sb else np.float32(0.0)
        case(f"{op.lower()}_{tag}", op, {"a": a, "b": b})
    case(f"{op.lower()}_ties", op, {"a": np.array([0, -0.0, NAN, INF, -INF, 1], np.float32), "b": np.array([-0.0, 0, NAN, INF, INF, 1], np.float32)})
case("and_same", "And", {"a": mask(3, 4), "b": mask(3, 4)})
case("and_bcast", "And", {"a": mask(3, 1), "b": mask(1, 4)})
case("or_same", "Or", {"a": mask(3, 4), "b": mask(3, 4)})
case("or_bcast", "Or", {"a": mask(4), "b": mask(3, 4)})
case("not_2d", "Not", {"x": mask(3, 4)})
case("isnan_spiked", "IsNaN", {"x": spiked(5, 7)})
case("isinf_spiked", "IsInf", {"x": spiked(5, 7)})
case("isinf_positive_only", "IsInf", {"x": spiked(5, 7)}, detect_negative=0)
case("isinf_negative_only", "IsInf", {"x": spiked(5, 7)}, detect_positive=0)
case("where_same", "Where", {"c": mask(3, 4), "a": spiked(3, 4), "b": spiked(3, 4)})
case("where_bcast", "Where", {"c": mask(3, 1), "a": spiked(3, 4), "b": np.float32(-7.0)})
case("where_cond_row", "Where", {"c": mask(4), "a": spiked(2, 3, 4), "b": spiked(3, 4)})
case("where_nan_in_unselected", "Where", {"c": np.array([True, False, True, False]), "a": np.array([1, NAN, 3, NAN], np.float32), "b": np.array([NAN, 2, NAN, 4], np.float32)})

# --- more unary, Pow/Max/Min/Mod/Clip ---
ties = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999997, 2.0, -0.0, 1e10], dtype=np.float32)
for op in ["Ceil", "Floor", "Round", "Sign", "Relu", "Sigmoid", "Sin", "Cos", "Softplus", "Reciprocal", "Erf"]:
    case(f"{op.lower()}_spiked", op, {"x": spiked(5, 7)})
    case(f"{op.lower()}_ties", op, {"x": np.concatenate([ties, np.array([NAN, INF, -INF, 100.0, -100.0], np.float32)])})
for tag, sa, sb in [("same", (3, 4), (3, 4)), ("bcast", (3, 1), (1, 4)), ("scalar_exp", (3, 4), ())]:
    a = spiked(*sa)
    b = spiked(*sb) if sb else np.float32(2.0)
    case(f"pow_{tag}", "Pow", {"a": a, "b": b})
    case(f"max_{tag}", "Max", {"a": a, "b": b})
    case(f"min_{tag}", "Min", {"a": a, "b": b})
case("pow_edges", "Pow", {"a": np.array([0, 0, -8, -8, 2, NAN, 1, INF, -INF, 0.5], np.float32), "b": np.array([0, -1, 1 / 3, 3, NAN, 0, NAN, 0.5, 3, -INF], np.float32)})
case("max_three", "Max", {"a": spiked(3, 4), "b": spiked(3, 4), "c": spiked(4)})
case("min_three", "Min", {"a": spiked(3, 4), "b": spiked(3, 4), "c": spiked(4)})
case("mod_fmod", "Mod", {"a": spiked(3, 4), "b": positive(3, 4)}, fmod=1)
case("mod_fmod_negative", "Mod", {"a": np.array([5.5, -5.5, 5.5, -5.5, 0, INF, 3], np.float32), "b": np.array([3, 3, -3, -3, 2, 2, 0], np.float32)}, fmod=1)
case("clip_both", "Clip", {"x": spiked(4, 5)}, inits={"lo": np.float32(-0.5), "hi": np.float32(0.7)})
case("clip_min_only", "Clip", {"x": spiked(4, 5)}, nodes=[helper.make_node("Clip", ["x", "lo"], ["y"])], inits={"lo": np.float32(-0.5)})
case("clip_max_only", "Clip", {"x": spiked(4, 5)}, nodes=[helper.make_node("Clip", ["x", "", "hi"], ["y"])], inits={"hi": np.float32(0.7)})

# --- Gemm, Cast, ReduceMax/Min/Prod, host-folded ops ---
case("gemm_plain", "Gemm", {"a": rand(3, 4), "b": rand(4, 5)})
case("gemm_full", "Gemm", {"a": rand(4, 3), "b": rand(5, 4)}, inits={"c": rand(5)}, alpha=0.5, beta=2.0, transA=1, transB=1)
case("gemm_c_matrix", "Gemm", {"a": rand(3, 4), "b": rand(4, 5)}, inits={"c": rand(3, 5)}, beta=0.25)
case("gemm_c_col", "Gemm", {"a": rand(3, 4), "b": rand(4, 5)}, inits={"c": rand(3, 1)})
case("cast_float_float", "Cast", {"x": spiked(3, 4)}, to=TensorProto.FLOAT)
case("cast_float_bool", "Cast", {"x": np.array([0, -0.0, 1, NAN, INF, 1e-30, -2], np.float32)}, to=TensorProto.BOOL)
case("cast_bool_float", "Cast", {"x": mask(3, 4)}, to=TensorProto.FLOAT)
for op in ["ReduceMax", "ReduceMin", "ReduceProd"]:
    tag = op.lower()
    case(f"{tag}_last", op, {"x": rand(3, 5)}, inits={"axes": ax(-1)})
    case(f"{tag}_first_drop", op, {"x": rand(3, 5)}, inits={"axes": ax(0)}, keepdims=0)
    case(f"{tag}_rank4_two", op, {"x": rand(2, 3, 4, 5)}, inits={"axes": ax(1, 3)})
    case(f"{tag}_all", op, {"x": rand(2, 3, 4)})
    case(f"{tag}_nan", op, {"x": np.array([[1, NAN, 3], [4, 5, 6], [INF, -INF, 1]], np.float32)}, inits={"axes": ax(-1)})
long_nan = rand(6, 41)
long_nan[0, 0] = NAN
long_nan[1, 17] = NAN
long_nan[2, 40] = NAN
long_nan[3, 5] = NAN
long_nan[3, 6] = NAN
long_nan[4, :] = NAN
for op in ["ReduceMax", "ReduceMin"]:
    case(f"{op.lower()}_long_nan", op, {"x": long_nan}, inits={"axes": ax(-1)})
case("shape_basic", "Shape", {"x": rand(2, 3, 4)})
case("shape_slice", "Shape", {"x": rand(2, 3, 4, 5)}, start=1, end=-1)
case("shape_scalar", "Shape", {"x": np.float32(1.0)})
case("size_basic", "Size", {"x": rand(2, 3, 4)})
case("constant_float", "Constant", {"x": rand(3)}, nodes=[helper.make_node("Constant", [], ["k"], value=numpy_helper.from_array(rand(2, 3), "k")), helper.make_node("Mul", ["x", "k"], ["y"])])
case("constant_scalar_float", "Constant", {"x": rand(2, 3)}, nodes=[helper.make_node("Constant", [], ["k"], value_float=2.5), helper.make_node("Mul", ["x", "k"], ["y"])])
case("constantofshape_float", "ConstantOfShape", {"x": rand(3)}, nodes=[helper.make_node("Constant", [], ["s"], value=numpy_helper.from_array(np.array([2, 3], np.int64), "s")), helper.make_node("ConstantOfShape", ["s"], ["z"], value=numpy_helper.from_array(np.array([1.5], np.float32), "v")), helper.make_node("Add", ["x", "z"], ["y"])])
case("constantofshape_int", "ConstantOfShape", {"x": rand(2)}, nodes=[helper.make_node("Constant", [], ["s"], value=numpy_helper.from_array(np.array([3], np.int64), "s")), helper.make_node("ConstantOfShape", ["s"], ["y"], value=numpy_helper.from_array(np.array([7], np.int64), "v"))])
case("range_int", "Range", {"x": rand(2)}, nodes=[helper.make_node("Range", ["s", "l", "d"], ["y"])], inits={"s": np.array(2, np.int64), "l": np.array(11, np.int64), "d": np.array(3, np.int64)})
case("range_float", "Range", {"x": rand(2)}, nodes=[helper.make_node("Range", ["s", "l", "d"], ["y"])], inits={"s": np.array(0.5, np.float32), "l": np.array(3.0, np.float32), "d": np.array(0.75, np.float32)})

# --- Gather (constant indices), CastLike ---
def ix(*a):
    return np.array(a, dtype=np.int64)


case("gather_axis0", "Gather", {"x": spiked(5, 3)}, inits={"i": ix(4, 0, 2, 2)}, axis=0)
case("gather_axis1", "Gather", {"x": spiked(3, 5)}, inits={"i": ix(1, 3)}, axis=1)
case("gather_negative", "Gather", {"x": spiked(5, 3)}, inits={"i": ix(-1, -5)}, axis=0)
case("gather_scalar_index", "Gather", {"x": spiked(5, 3)}, inits={"i": np.array(2, np.int64)}, axis=0)
case("gather_scalar_index_axis1", "Gather", {"x": spiked(4, 5, 2)}, inits={"i": np.array(-2, np.int64)}, axis=1)
case("gather_2d_indices", "Gather", {"x": spiked(6, 3)}, inits={"i": np.array([[0, 5], [3, 3]], np.int64)}, axis=0)
case("gather_rank3_last", "Gather", {"x": rand(2, 3, 4)}, inits={"i": ix(3, 0)}, axis=2)
case("gather_default_axis", "Gather", {"x": rand(4, 2)}, inits={"i": ix(1)})
case("castlike_float", "CastLike", {"x": spiked(3, 4)}, inits={"like": rand(2)})
case("castlike_bool", "CastLike", {"x": np.array([0, -0.0, 1, NAN, 2], np.float32)}, inits={"like": np.array([True])})
case("castlike_mask_to_float", "CastLike", {"x": mask(3, 4)}, inits={"like": rand(1)})

# --- real-graph usage patterns found by the coverage review ---
I64MIN, I64MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max
case("mod_floor_semantics", "Mod", {"a": np.array([5.5, -5.5, 5.5, -5.5, 7, -7], np.float32), "b": np.array([3, 3, -3, -3, 2.5, 2.5], np.float32)}, fmod=0) if False else None
case("pow_half_negative_base", "Pow", {"a": spiked(3, 5)}, inits={"b": np.float32(0.5)})
case("pow_cube_negative_base", "Pow", {"a": spiked(3, 5)}, inits={"b": np.float32(3.0)})
case("pow_square", "Pow", {"a": spiked(3, 5)}, inits={"b": np.float32(2.0)})
case("add_scalar_scalar", "Add", {"a": np.float32(1.5), "b": np.float32(-2.25)})
case("div_scalar_by_vector", "Div", {"a": np.float32(1.0), "b": spiked(5)})
case("div_vector_by_scalar", "Div", {"a": spiked(5), "b": np.float32(0.0)})
case("max_scalar_scalar", "Max", {"a": np.float32(1.5), "b": np.float32(NAN)})
case("max_single_input", "Max", {"a": spiked(4)})
case("expand_scalar_to_vec", "Expand", {"x": np.float32(3.5)}, inits={"shape": np.array([5], np.int64)})
case("expand_scalar_to_rank3", "Expand", {"x": np.float32(3.5)}, inits={"shape": np.array([2, 3, 4], np.int64)})
case("expand_identity", "Expand", {"x": spiked(2, 3, 4)}, inits={"shape": np.array([2, 3, 4], np.int64)})
case("expand_mid", "Expand", {"x": spiked(1, 1, 5)}, inits={"shape": np.array([1, 4, 5], np.int64)})
case("expand_last", "Expand", {"x": spiked(3, 2, 1)}, inits={"shape": np.array([3, 2, 2], np.int64)})
case("reducesum_all_to_scalar", "ReduceSum", {"x": rand(5)}, keepdims=0)
case("reducemean_computed_axes", "ReduceMean", {"x": rand(2, 3, 130)}, nodes=[
    helper.make_node("Constant", [], ["ac"], value=numpy_helper.from_array(np.array(-1, np.int64), "ac")),
    helper.make_node("Reshape", ["ac", "one"], ["axes"]),
    helper.make_node("ReduceMean", ["x", "axes"], ["y"])], inits={"one": np.array([1], np.int64)})
case("slice_int64_min_start", "Slice", {"x": spiked(3, 6)}, inits={"starts": ax(I64MIN), "ends": ax(-2), "axes": ax(1)})
case("slice_end_minus_one", "Slice", {"x": spiked(3, 6)}, inits={"starts": ax(0), "ends": ax(-1), "axes": ax(-1)})
case("slice_end_65535", "Slice", {"x": spiked(7, 3)}, inits={"starts": ax(2), "ends": ax(65535), "axes": ax(0)})
case("slice_axis_minus3_rank4", "Slice", {"x": rand(2, 3, 4, 5)}, inits={"starts": ax(1), "ends": ax(3), "axes": ax(-3)})
case("where_two_scalars", "Where", {"c": mask(1, 1, 3, 4)}, nodes=[
    helper.make_node("Constant", [], ["t"], value=numpy_helper.from_array(np.array(0.0, np.float32), "t")),
    helper.make_node("Constant", [], ["f"], value=numpy_helper.from_array(np.array(-np.inf, np.float32), "f")),
    helper.make_node("Where", ["c", "t", "f"], ["y"])])
case("where_scalar_left", "Where", {"c": mask(3, 1), "b": spiked(3, 4)}, nodes=[
    helper.make_node("Constant", [], ["t"], value_float=7.0), helper.make_node("Where", ["c", "t", "b"], ["y"])])
case("where_computed_isnan_cond", "Where", {"x": spiked(3, 4), "b": spiked(3, 4)}, nodes=[
    helper.make_node("IsNaN", ["x"], ["m"]), helper.make_node("Where", ["m", "b", "x"], ["y"])])
case("matmul_b_batch1", "MatMul", {"a": rand(3, 8, 4, 5), "b": rand(1, 8, 5, 6)})
case("matmul_a_batch1", "MatMul", {"a": rand(1, 8, 4, 5), "b": rand(3, 8, 5, 6)})
case("matmul_2d_by_batch1", "MatMul", {"a": rand(2, 4), "b": rand(1, 4, 6)})
case("matmul_rank3_by_batch1", "MatMul", {"a": rand(3, 4, 5), "b": rand(1, 5, 6)})
case("matmul_wide_k", "MatMul", {"a": rand(4, 512), "b": rand(512, 64)})
case("matmul_wide_rank3", "MatMul", {"a": rand(1, 130, 512), "b": rand(512, 96)})
case("matmul_nan_inf_b", "MatMul", {"a": rand(3, 4), "b": spiked(4, 5)})
case("matmul_weight_initializer", "MatMul", {"a": rand(6, 8)}, inits={"b": rand(8, 5)})
case("transpose_r3_021", "Transpose", {"x": spiked(2, 3, 4)}, perm=[0, 2, 1])
case("transpose_r3_102", "Transpose", {"x": spiked(2, 3, 4)}, perm=[1, 0, 2])
case("transpose_r4_2103", "Transpose", {"x": spiked(2, 3, 4, 5)}, perm=[2, 1, 0, 3])
case("transpose_r4_0132", "Transpose", {"x": spiked(2, 3, 4, 5)}, perm=[0, 1, 3, 2])
case("transpose_r4_0231", "Transpose", {"x": spiked(2, 3, 4, 5)}, perm=[0, 2, 3, 1])
case("concat_axis_minus2_rank4", "Concat", {"a": rand(2, 3, 2, 4), "b": rand(2, 3, 5, 4)}, axis=-2)
case("concat_axis2_rank4", "Concat", {"a": rand(2, 3, 2, 4), "b": rand(2, 3, 1, 4)}, axis=2)
case("concat_single_input", "Concat", {"a": spiked(2, 3)}, axis=1)
case("concat_rank3_axis0", "Concat", {"a": rand(1, 3, 4), "b": rand(2, 3, 4)}, axis=0)
wide = rand(2, 3, 130)
wide[0, 0, 5:] = -np.inf
wide[1, 1, :] = -np.inf
case("softmax_rank3_wide_masked", "Softmax", {"x": wide}, axis=-1)
case("softmax_rank4_513", "Softmax", {"x": rand(2, 2, 3, 513) * 5}, axis=-1)
case("layernorm_rank4_wide", "LayerNormalization", {"x": rand(2, 3, 4, 512) * 3 + 1000}, inits={"scale": rand(512), "bias": rand(512)}, axis=-1)
case("layernorm_scale_from_expand", "LayerNormalization", {"x": rand(3, 8)}, nodes=[
    helper.make_node("Constant", [], ["one"], value=numpy_helper.from_array(np.array(1.0, np.float32), "one")),
    helper.make_node("Constant", [], ["sh"], value=numpy_helper.from_array(np.array([8], np.int64), "sh")),
    helper.make_node("Expand", ["one", "sh"], ["scale"]),
    helper.make_node("LayerNormalization", ["x", "scale"], ["y"], axis=-1, epsilon=1e-5)])
case("gemm_transb_no_c", "Gemm", {"a": rand(3, 4), "b": rand(5, 4)}, transB=1)
case("gemm_transb_wide", "Gemm", {"a": rand(6, 64)}, inits={"b": rand(224, 64), "c": rand(224)}, transB=1)
case("unsqueeze_minus3_rank3", "Unsqueeze", {"x": spiked(2, 3, 4)}, inits={"axes": ax(-3)})
case("unsqueeze_last_of_r3", "Unsqueeze", {"x": spiked(2, 3, 4)}, inits={"axes": ax(3)})
case("squeeze_minus1_rank3", "Squeeze", {"x": spiked(2, 3, 1)}, inits={"axes": ax(-1)})
case("constantofshape_default", "ConstantOfShape", {"x": rand(3)}, nodes=[
    helper.make_node("Constant", [], ["s"], value=numpy_helper.from_array(np.array([2, 3], np.int64), "s")),
    helper.make_node("ConstantOfShape", ["s"], ["z"]), helper.make_node("Add", ["x", "z"], ["y"])])
case("constant_value_ints_reshape", "Constant", {"x": spiked(2, 3, 4)}, nodes=[
    helper.make_node("Constant", [], ["t"], value_ints=[6, -1]), helper.make_node("Reshape", ["x", "t"], ["y"])])
case("gather_scalar_rank4", "Gather", {"x": spiked(3, 4, 2, 5)}, inits={"i": np.array(-1, np.int64)}, axis=0)
case("shape_start_neg1_end_default", "Shape", {"x": rand(2, 3, 4)}, start=-1)
case("shape_end_neg3_rank4", "Shape", {"x": rand(2, 3, 4, 5)}, start=0, end=-3)
case("dynamic_reshape_chain", "Reshape", {"x": spiked(2, 3, 4)}, nodes=[
    helper.make_node("Shape", ["x"], ["s"]),
    helper.make_node("Gather", ["s", "i0"], ["d0"]),
    helper.make_node("Unsqueeze", ["d0", "ax0"], ["d0v"]),
    helper.make_node("Concat", ["d0v", "m1"], ["tgt"], axis=0),
    helper.make_node("Reshape", ["x", "tgt"], ["y"])],
    inits={"i0": np.array(0, np.int64), "ax0": np.array([0], np.int64), "m1": np.array([-1], np.int64)})
case("dynamic_expand_chain", "Expand", {"x": spiked(3, 1), "r": rand(3, 4)}, nodes=[
    helper.make_node("Shape", ["r"], ["s"]), helper.make_node("Expand", ["x", "s"], ["y"])])
case("range_cast_mul", "Range", {"x": rand(4)}, nodes=[
    helper.make_node("Shape", ["x"], ["s"]), helper.make_node("Gather", ["s", "z"], ["n"]),
    helper.make_node("Range", ["zero", "n", "one"], ["r"]), helper.make_node("Cast", ["r"], ["rf"], to=TensorProto.FLOAT),
    helper.make_node("Mul", ["x", "rf"], ["y"])],
    inits={"z": np.array(0, np.int64), "zero": np.array(0, np.int64), "one": np.array(1, np.int64)})

# --- review probes: NaN placement in multi-axis reductions, Gemm beta=0, Clip lo>hi, host-known bool operands, chains ---
nan_mid = np.array([[[1, 2], [NAN, 4]], [[5, NAN], [7, 8]], [[9, 10], [11, 12]]], np.float32)  # (3,2,2), NaN not first anywhere
for op in ["ReduceMax", "ReduceMin"]:
    tag = op.lower()
    case(f"{tag}_nan_axis0", op, {"x": nan_mid}, inits={"axes": ax(0)})
    case(f"{tag}_nan_axis_mid", op, {"x": nan_mid}, inits={"axes": ax(1)})
    case(f"{tag}_nan_two_axes", op, {"x": nan_mid}, inits={"axes": ax(0, 1)})
    # NOT a golden: ORT's ReduceMax/Min over a short merged row with a NaN in the middle ([5,NaN,7,8] -> 7) depends on its SIMD
    # lanes, so the result is implementation-defined. We match the scalar-order rule (NaN only wins as the first element).
    case(f"{tag}_nan_all_axes", op, {"x": nan_mid})
case("gemm_beta_zero_inf_c", "Gemm", {"a": rand(3, 4), "b": rand(4, 5)}, inits={"c": np.array([INF, 1, NAN, 2, 3], np.float32)}, beta=0.0)
case("gemm_alpha_zero", "Gemm", {"a": spiked(3, 4), "b": rand(4, 5)}, alpha=0.0)
case("clip_lo_greater_than_hi", "Clip", {"x": np.array([0.5, -3, 3, NAN, 1, 0], np.float32)}, inits={"lo": np.float32(1.0), "hi": np.float32(0.0)})
# a bool operand the host folds (from shapes), consumed by a float op
case("where_known_bool_cond", "Where", {"x": spiked(3, 4)}, nodes=[
    helper.make_node("Shape", ["x"], ["s"]), helper.make_node("Gather", ["s", "i1"], ["n"]),
    helper.make_node("Range", ["z", "n", "o"], ["r"]), helper.make_node("Less", ["r", "lim"], ["m"]),
    helper.make_node("Constant", [], ["ninf"], value_float=float("-inf")),
    helper.make_node("Where", ["m", "x", "ninf"], ["y"])],
    inits={"i1": np.array(1, np.int64), "z": np.array(0, np.int64), "o": np.array(1, np.int64), "lim": np.array(2, np.int64)})
case("and_known_mask", "And", {"m": mask(2, 4)}, nodes=[
    helper.make_node("Shape", ["m"], ["s"]), helper.make_node("Gather", ["s", "i1"], ["n"]),
    helper.make_node("Range", ["z", "n", "o"], ["r"]), helper.make_node("Less", ["r", "lim"], ["k"]),
    helper.make_node("And", ["m", "k"], ["y"])],
    inits={"i1": np.array(1, np.int64), "z": np.array(0, np.int64), "o": np.array(1, np.int64), "lim": np.array(3, np.int64)})
case("pow_int64_exponent", "Pow", {"x": spiked(3, 4)}, inits={"e": np.array(2, np.int64)})
# chains: views feeding ops, aliasing, the same tensor twice, in-place reuse, custom op mid-chain
case("chain_slice_view_ops", "Softmax", {"x": spiked(6, 8)}, nodes=[
    helper.make_node("Slice", ["x", "s0", "e0", "a0", "st"], ["v"]),
    helper.make_node("Neg", ["v"], ["n"]), helper.make_node("Exp", ["n"], ["ex"]),
    helper.make_node("Softmax", ["ex"], ["sm"], axis=-1),
    helper.make_node("Add", ["sm", "v"], ["y"])],
    inits={"s0": ax(1), "e0": ax(6), "a0": ax(0), "st": ax(2)})
case("chain_slice_matmul_transpose", "MatMul", {"x": rand(5, 4)}, nodes=[
    helper.make_node("Slice", ["x", "s0", "e0", "a0"], ["v"]),
    helper.make_node("Transpose", ["v"], ["vt"], perm=[1, 0]),
    helper.make_node("MatMul", ["vt", "v"], ["y"])], inits={"s0": ax(1), "e0": ax(4), "a0": ax(0)})
case("same_tensor_twice_add", "Add", {"x": spiked(3, 4)}, nodes=[helper.make_node("Add", ["x", "x"], ["y"])])
case("same_tensor_twice_equal", "Equal", {"x": spiked(3, 4)}, nodes=[helper.make_node("Equal", ["x", "x"], ["y"])])
case("same_tensor_twice_concat", "Concat", {"x": spiked(2, 3)}, nodes=[helper.make_node("Concat", ["x", "x"], ["y"], axis=1)])
case("same_tensor_twice_matmul", "MatMul", {"x": rand(4, 4)}, nodes=[helper.make_node("MatMul", ["x", "x"], ["y"])])
case("chain_inplace_neg_exp_add", "Exp", {"x": spiked(4, 5), "w": spiked(4, 5)}, nodes=[
    helper.make_node("Neg", ["x"], ["a"]), helper.make_node("Exp", ["a"], ["b"]), helper.make_node("Add", ["b", "w"], ["y"])])
case("chain_custom_mid", "Where", {"x": spiked(4, 5)}, nodes=[
    helper.make_node("IsNaN", ["x"], ["m"]), helper.make_node("Neg", ["x"], ["n"]),
    helper.make_node("Where", ["m", "n", "x"], ["w"]), helper.make_node("Mul", ["w", "w"], ["y"])])
case("output_is_identity_of_input", "Identity", {"x": spiked(3, 4)}, nodes=[helper.make_node("Identity", ["x"], ["i"]), helper.make_node("Identity", ["i"], ["y"])])

# --- rank 5 and 6 (the census: ~12k nodes in the real graphs; ggml is 4-D at most) ---
for tag, perm in [("0132", [0, 1, 3, 2, 4]), ("3120", [3, 1, 2, 0, 4]), ("4123", [4, 1, 2, 3, 0]), ("0213", [0, 2, 1, 3, 4]), ("01243", [0, 1, 2, 4, 3])]:
    case(f"transpose_r5_{tag}", "Transpose", {"x": spiked(2, 3, 2, 3, 4)}, perm=perm)
case("transpose_r5_cycle", "Transpose", {"x": spiked(2, 3, 2, 3, 4)}, perm=[2, 3, 0, 4, 1])  # groups permute as a 4-cycle, not an involution
case("transpose_r6", "Transpose", {"x": spiked(2, 3, 2, 3, 2, 2)}, perm=[4, 1, 2, 3, 0, 5])
case("transpose_r5_with_ones", "Transpose", {"x": spiked(1, 3, 1, 4, 2)}, perm=[4, 3, 2, 1, 0])
case("mul_r5_bcast_both", "Mul", {"a": spiked(2, 1, 3, 1, 4), "b": spiked(2, 2, 3, 5, 4)})
case("add_r5_vs_r4", "Add", {"a": spiked(2, 3, 2, 3, 4), "b": spiked(3, 2, 3, 4)})
case("sub_r5_scalar_left", "Sub", {"a": np.float32(1.5), "b": spiked(2, 2, 3, 2, 2)})
case("div_r6", "Div", {"a": spiked(2, 1, 3, 1, 2, 2), "b": spiked(1, 3, 3, 2, 1, 2)})
case("softmax_r5_last", "Softmax", {"x": rand(2, 3, 2, 3, 5) * 4}, axis=-1)
case("softmax_r5_mid", "Softmax", {"x": rand(2, 3, 2, 3, 5)}, axis=2)
case("neg_r5", "Neg", {"x": spiked(2, 3, 2, 3, 4)})
case("sqrt_r5", "Sqrt", {"x": spiked(2, 3, 2, 3, 4)})
case("erf_r5", "Erf", {"x": spiked(2, 3, 2, 3, 4)})
case("isnan_r5", "IsNaN", {"x": spiked(2, 3, 2, 3, 4)})
case("where_r5", "Where", {"c": mask(2, 1, 3, 1, 4), "a": spiked(2, 2, 3, 2, 4), "b": np.float32(-9.0)})
case("reshape_r4_r5_r6", "Reshape", {"x": spiked(2, 3, 4, 6)}, nodes=[
    helper.make_node("Reshape", ["x", "s5"], ["a"]), helper.make_node("Reshape", ["a", "s6"], ["y"])],
    inits={"s5": np.array([2, 3, 4, 2, 3], np.int64), "s6": np.array([2, 3, 2, 2, 2, 3], np.int64)})
case("squeeze_r5_minus3", "Squeeze", {"x": spiked(2, 3, 1, 4, 5)}, inits={"axes": ax(-3)})
case("squeeze_r6_last", "Squeeze", {"x": spiked(2, 3, 2, 2, 3, 1)}, inits={"axes": ax(-1)})
case("unsqueeze_r4_to_r5", "Unsqueeze", {"x": spiked(2, 3, 4, 5)}, inits={"axes": ax(4)})
case("unsqueeze_r5_to_r6", "Unsqueeze", {"x": spiked(2, 3, 2, 3, 4)}, inits={"axes": ax(5)})
case("slice_r5_axis_minus2", "Slice", {"x": spiked(2, 3, 4, 5, 6)}, inits={"starts": ax(1), "ends": ax(3), "axes": ax(-2)})
case("slice_r5_two_axes", "Slice", {"x": rand(2, 3, 4, 5, 6)}, inits={"starts": ax(1, 2), "ends": ax(3, 5), "axes": ax(0, 3)})
case("slice_r6_outer_step", "Slice", {"x": rand(4, 3, 2, 2, 2, 3)}, inits={"starts": ax(0), "ends": ax(4), "axes": ax(0), "steps": ax(2)})
case("concat_r5_last", "Concat", {"a": rand(2, 3, 2, 2, 2), "b": rand(2, 3, 2, 2, 3), "c": rand(2, 3, 2, 2, 1)}, axis=-1)
case("concat_r6_mid", "Concat", {"a": rand(2, 2, 2, 1, 2, 2), "b": rand(2, 2, 2, 3, 2, 2)}, axis=3)
case("expand_r5", "Expand", {"x": spiked(2, 1, 1, 3, 4)}, inits={"shape": np.array([2, 1, 3, 3, 4], np.int64)})
case("matmul_r5", "MatMul", {"a": rand(1, 3, 6, 5, 4), "b": rand(1, 3, 6, 4, 4)})
case("matmul_r5_bcast", "MatMul", {"a": rand(2, 3, 2, 5, 4), "b": rand(2, 3, 1, 4, 4)})
case("matmul_r5_by_weight", "MatMul", {"a": rand(2, 3, 2, 5, 4)}, inits={"b": rand(4, 6)})
case("matmul_r6", "MatMul", {"a": rand(1, 2, 2, 3, 5, 4), "b": rand(1, 2, 2, 3, 4, 3)})
case("gather_r5_scalar_axis2", "Gather", {"x": spiked(2, 3, 4, 3, 2)}, inits={"i": np.array(-1, np.int64)}, axis=2)
case("gather_r5_vector", "Gather", {"x": spiked(2, 3, 4, 3, 2)}, inits={"i": ix(2, 0)}, axis=1)
for op in ["ReduceMean", "ReduceSum", "ReduceMax"]:
    case(f"{op.lower()}_r5_last", op, {"x": rand(2, 3, 2, 3, 5)}, inits={"axes": ax(-1)})
    case(f"{op.lower()}_r5_mid_drop", op, {"x": rand(2, 3, 2, 3, 5)}, inits={"axes": ax(1, 3)}, keepdims=0)
    case(f"{op.lower()}_r6_first", op, {"x": rand(2, 2, 2, 3, 2, 3)}, inits={"axes": ax(0)})
case("layernorm_r5", "LayerNormalization", {"x": rand(2, 3, 2, 3, 8)}, inits={"scale": rand(8), "bias": rand(8)}, axis=-1)
case("layernorm_r5_two_axes", "LayerNormalization", {"x": rand(2, 3, 2, 3, 4)}, inits={"scale": rand(3, 4), "bias": rand(3, 4)}, axis=3)
case("pow_r5", "Pow", {"a": spiked(2, 3, 2, 3, 4)}, inits={"b": np.float32(2.0)})
case("max_r5", "Max", {"a": spiked(2, 1, 3, 2, 4), "b": spiked(2, 2, 3, 1, 4)})
case("cast_r5_bool", "Cast", {"x": spiked(2, 3, 2, 3, 4)}, to=TensorProto.BOOL)
case("clip_r5", "Clip", {"x": spiked(2, 3, 2, 3, 4)}, inits={"lo": np.float32(-0.5), "hi": np.float32(0.7)})
case("fused_projection_pattern", "MatMul", {"x": rand(6, 5, 2 * 3 * 4)}, nodes=[
    helper.make_node("Reshape", ["x", "tgt"], ["r"]),
    helper.make_node("Slice", ["r", "s0", "e0", "ax2"], ["k"]),
    helper.make_node("Slice", ["r", "s1", "e1", "ax2"], ["v"]),
    helper.make_node("Squeeze", ["k", "ax2"], ["k4"]), helper.make_node("Squeeze", ["v", "ax2"], ["v4"]),
    helper.make_node("Transpose", ["k4"], ["kt"], perm=[0, 1, 3, 2]),
    helper.make_node("MatMul", ["v4", "kt"], ["y"])],
    inits={"tgt": np.array([6, 5, 2, 3, 4], np.int64), "s0": ax(0), "e0": ax(1), "s1": ax(1), "e1": ax(2), "ax2": ax(2)})

_proj = {"tgt": np.array([6, 5, 2, 3, 4], np.int64), "s0": ax(0), "e0": ax(1), "s1": ax(1), "e1": ax(2), "ax2": ax(2)}
case("fused_stage1_slice_k", "Slice", {"x": rand(6, 5, 24)}, nodes=[
    helper.make_node("Reshape", ["x", "tgt"], ["r"]), helper.make_node("Slice", ["r", "s0", "e0", "ax2"], ["y"])], inits=_proj)
case("fused_stage2_slice_v", "Slice", {"x": rand(6, 5, 24)}, nodes=[
    helper.make_node("Reshape", ["x", "tgt"], ["r"]), helper.make_node("Slice", ["r", "s1", "e1", "ax2"], ["y"])], inits=_proj)
case("fused_stage3_squeeze", "Squeeze", {"x": rand(6, 5, 24)}, nodes=[
    helper.make_node("Reshape", ["x", "tgt"], ["r"]), helper.make_node("Slice", ["r", "s1", "e1", "ax2"], ["v"]),
    helper.make_node("Squeeze", ["v", "ax2"], ["y"])], inits=_proj)
case("fused_stage4_transpose", "Transpose", {"x": rand(6, 5, 24)}, nodes=[
    helper.make_node("Reshape", ["x", "tgt"], ["r"]), helper.make_node("Slice", ["r", "s0", "e0", "ax2"], ["k"]),
    helper.make_node("Squeeze", ["k", "ax2"], ["k4"]), helper.make_node("Transpose", ["k4"], ["y"], perm=[0, 1, 3, 2])], inits=_proj)

# --- Nori 1.5's remaining ops: TopK, GatherElements (data-dependent indices), Cast float->int, SplitToSequence/SequenceAt ---
def tied(*shape):
    return np.round(rand(*shape) * 2).astype(np.float32) / 2  # many equal values: tie order matters
case("cast_float_int64_trunc", "Cast", {"x": np.array([1.7, -1.7, 0.2, -0.2, 3.0, -3.0, 100.99, -100.99], np.float32)}, to=TensorProto.INT64)
case("cast_float_int32_trunc", "Cast", {"x": np.array([1.7, -1.7, 0.2, -0.2, 3.0, -3.0, 2147483.5], np.float32)}, to=TensorProto.INT32)
case("topk_smallest_values", "TopK", {"x": tied(3, 8)}, nodes=[helper.make_node("TopK", ["x", "k"], ["y", "i"], axis=-1, largest=0, sorted=1)], inits={"k": ix(3)})
case("topk_smallest_indices_ties", "TopK", {"x": tied(3, 8)}, nodes=[helper.make_node("TopK", ["x", "k"], ["v", "y"], axis=-1, largest=0, sorted=1)], inits={"k": ix(5)})
case("topk_largest_values", "TopK", {"x": tied(3, 8)}, nodes=[helper.make_node("TopK", ["x", "k"], ["y", "i"], axis=-1, largest=1, sorted=1)], inits={"k": ix(4)})
case("topk_largest_indices_ties", "TopK", {"x": tied(3, 8)}, nodes=[helper.make_node("TopK", ["x", "k"], ["v", "y"], axis=-1, largest=1, sorted=1)], inits={"k": ix(8)})
case("topk_wide_ties_smallest_indices", "TopK", {"x": tied(4, 200)}, nodes=[helper.make_node("TopK", ["x", "k"], ["v", "y"], axis=-1, largest=0, sorted=1)], inits={"k": ix(150)})
case("topk_wide_ties_largest_indices", "TopK", {"x": tied(4, 200)}, nodes=[helper.make_node("TopK", ["x", "k"], ["v", "y"], axis=-1, largest=1, sorted=1)], inits={"k": ix(200)})
case("topk_axis0", "TopK", {"x": tied(6, 3)}, nodes=[helper.make_node("TopK", ["x", "k"], ["y", "i"], axis=0, largest=1, sorted=1)], inits={"k": ix(2)})
case("topk_rank3_mid_indices", "TopK", {"x": tied(2, 5, 3)}, nodes=[helper.make_node("TopK", ["x", "k"], ["v", "y"], axis=1, largest=0, sorted=1)], inits={"k": ix(3)})
case("topk_rank4_last", "TopK", {"x": tied(2, 3, 2, 9)}, nodes=[helper.make_node("TopK", ["x", "k"], ["y", "i"], axis=-1, largest=0, sorted=1)], inits={"k": ix(1)})
case("topk_k_from_shape", "TopK", {"x": tied(3, 8), "r": rand(2, 4)}, nodes=[
    helper.make_node("Shape", ["r"], ["s"]), helper.make_node("Gather", ["s", "one"], ["kk"]),
    helper.make_node("Reshape", ["kk", "r1"], ["k"]), helper.make_node("TopK", ["x", "k"], ["y", "i"], axis=-1, largest=0, sorted=1)],
    inits={"one": np.array(1, np.int64), "r1": np.array([1], np.int64)})
case("gatherelements_axis1", "GatherElements", {"d": spiked(3, 5)}, inits={"i": np.array([[0, 4], [2, 2], [1, 3]], np.int64)}, axis=1)
case("gatherelements_axis0", "GatherElements", {"d": spiked(4, 3)}, inits={"i": np.array([[0, 3, 1], [2, 2, 0]], np.int64)}, axis=0)
case("gatherelements_negative_idx", "GatherElements", {"d": spiked(3, 5)}, inits={"i": np.array([[-1, -5], [-2, 0], [4, -3]], np.int64)}, axis=-1)
case("gatherelements_rank3", "GatherElements", {"d": spiked(2, 3, 4)}, inits={"i": np.array([[[0, 3]] * 3, [[1, 1]] * 3], np.int64)}, axis=2)
case("gatherelements_rank4_last", "GatherElements", {"d": spiked(2, 2, 3, 5)}, inits={"i": rng.integers(0, 5, (2, 2, 3, 2)).astype(np.int64)}, axis=-1)
case("gatherelements_dynamic_index", "GatherElements", {"d": spiked(3, 5), "u": (rng.random((3, 2)) * 0.999).astype(np.float32)}, nodes=[
    helper.make_node("Mul", ["u", "five"], ["m"]), helper.make_node("Floor", ["m"], ["f"]),
    helper.make_node("Cast", ["f"], ["fi"], to=TensorProto.INT64), helper.make_node("GatherElements", ["d", "fi"], ["y"], axis=-1)],
    inits={"five": np.float32(5.0)})
case("quantile_pattern", "GatherElements", {"x": tied(2, 9), "q": (rng.random((2, 1)) * 0.999).astype(np.float32)}, nodes=[
    helper.make_node("TopK", ["x", "k"], ["sorted", "idx"], axis=-1, largest=0, sorted=1),
    helper.make_node("Mul", ["q", "eight"], ["pos"]), helper.make_node("Floor", ["pos"], ["lo"]), helper.make_node("Ceil", ["pos"], ["hi"]),
    helper.make_node("Cast", ["lo"], ["loi"], to=TensorProto.INT64), helper.make_node("Cast", ["hi"], ["hii"], to=TensorProto.INT64),
    helper.make_node("GatherElements", ["sorted", "loi"], ["a"], axis=-1), helper.make_node("GatherElements", ["sorted", "hii"], ["b"], axis=-1),
    helper.make_node("Add", ["a", "b"], ["y"])], inits={"k": ix(9), "eight": np.float32(8.0)})
for tag, sp, kd, ix_ in [("scalar_split", np.array(2, np.int64), 1, 1), ("list_split", ix(1, 3, 2), 1, 2), ("scalar_split_neg_index", np.array(2, np.int64), 1, -1)]:
    case(f"seq_{tag}", "SplitToSequence" if tag == "list_split" else "SequenceAt", {"x": spiked(3, 6)}, nodes=[
        helper.make_node("SplitToSequence", ["x", "sp"], ["s"], axis=-1, keepdims=kd),
        helper.make_node("SequenceAt", ["s", "ia"], ["y"])], inits={"sp": sp, "ia": np.array(ix_, np.int64)})
case("seq_no_split_keepdims0", "SequenceAt", {"x": spiked(3, 4)}, nodes=[
    helper.make_node("SplitToSequence", ["x"], ["s"], axis=0, keepdims=0), helper.make_node("SequenceAt", ["s", "ia"], ["y"])], inits={"ia": np.array(2, np.int64)})
case("seq_no_split_keepdims1", "SequenceAt", {"x": spiked(3, 4)}, nodes=[
    helper.make_node("SplitToSequence", ["x"], ["s"], axis=1, keepdims=1), helper.make_node("SequenceAt", ["s", "ia"], ["y"])], inits={"ia": np.array(3, np.int64)})
case("seq_rank3_mid_axis", "SequenceAt", {"x": spiked(2, 6, 3)}, nodes=[
    helper.make_node("SplitToSequence", ["x", "sp"], ["s"], axis=1), helper.make_node("SequenceAt", ["s", "ia"], ["y"])], inits={"sp": np.array(3, np.int64), "ia": np.array(1, np.int64)})

# --- Gather with indices the host cannot fold (computed from float data) ---
def dyn_idx(shape, hi):
    return (rng.random(shape) * 0.999 * hi).astype(np.float32)


for tag, dshape, ishape, axis in [("embedding_axis0", (7, 4), (5,), 0), ("matrix_idx", (6, 3), (2, 4), 0), ("axis1", (3, 6), (4,), 1), ("rank3_mid", (2, 5, 3), (3, 2), 1), ("scalar_like_single", (6, 2), (1,), 0)]:
    hi = dshape[axis]
    case(f"gather_dynamic_{tag}", "Gather", {"d": spiked(*dshape), "u": dyn_idx(ishape, hi)}, nodes=[
        helper.make_node("Floor", ["u"], ["f"]), helper.make_node("Cast", ["f"], ["fi"], to=TensorProto.INT64),
        helper.make_node("Gather", ["d", "fi"], ["y"], axis=axis)])
case("gather_dynamic_negative", "Gather", {"d": spiked(5, 3), "u": np.array([-1.5, -4.2, 0.3, 2.9], np.float32)}, nodes=[
    helper.make_node("Cast", ["u"], ["fi"], to=TensorProto.INT64), helper.make_node("Gather", ["d", "fi"], ["y"], axis=0)])

# --- integer Mod (fmod=0, sign of the divisor) on integers computed from float data ---
case("mod_int_floor_semantics", "Mod", {"x": np.array([5, -5, 5, -5, 7, -7, 0, 6, -6, 9], np.float32)}, nodes=[
    helper.make_node("Cast", ["x"], ["xi"], to=TensorProto.INT64), helper.make_node("Mod", ["xi", "d"], ["y"], fmod=0)],
    inits={"d": np.array([3, 3, -3, -3, 2, 2, 5, -4, -4, 3], np.int64)})
case("mod_int_by_scalar", "Mod", {"x": np.array([[10, -10, 7], [-7, 0, 13]], np.float32)}, nodes=[
    helper.make_node("Cast", ["x"], ["xi"], to=TensorProto.INT64), helper.make_node("Mod", ["xi", "d"], ["y"], fmod=0)], inits={"d": np.array(4, np.int64)})

# --- float64 islands: the graphs cast to DOUBLE, compute, and cast back (exponent digits via floor(log(x)/log(10)) are exact only in double) ---
C = lambda v: helper.make_node("Constant", [], [v[0]], value=numpy_helper.from_array(np.asarray(v[1]), v[0]))
pw10 = np.array([1, 10, 100, 1000, 0.1, 0.01, 5, 12345, 1e10, 0.001, 1e-5, 99999], np.float32)
case("double_floor_log10", "Floor", {"x": pw10}, nodes=[
    helper.make_node("Cast", ["x"], ["xd"], to=TensorProto.DOUBLE), helper.make_node("Log", ["xd"], ["lx"]),
    # log(10) is a FLOAT32 constant's log cast up to double, exactly as in the Nori graph: exact powers of ten then floor one lower
    C(("ten", np.float32(10.0))), helper.make_node("Log", ["ten"], ["l10f"]), helper.make_node("CastLike", ["l10f", "xd"], ["l10"]),
    helper.make_node("Div", ["lx", "l10"], ["q"]),
    helper.make_node("Floor", ["q"], ["f"]), helper.make_node("Cast", ["f"], ["y"], to=TensorProto.FLOAT)])
# NOT exact powers of ten here: ORT's double Log is within 1-2 ulp of libm but not identical (log(10.0) is one ulp low), so a floor() exactly at
# an integer is implementation-defined and cannot be matched.
mant = np.array([3, 12.5, 250, 0.35, 7777, 0.0123, 5.5e8, 42, 0.5, 9.99, 123456], np.float32)
case("double_mantissa_digits", "Pow", {"x": mant}, nodes=[
    helper.make_node("Cast", ["x"], ["xd"], to=TensorProto.DOUBLE), helper.make_node("Log", ["xd"], ["lx"]),
    C(("ten", np.float64(10.0))), C(("lten", np.float64(2.302585092994046))), helper.make_node("Div", ["lx", "lten"], ["q"]),
    helper.make_node("Floor", ["q"], ["e"]), helper.make_node("Pow", ["ten", "e"], ["p"]), helper.make_node("Div", ["xd", "p"], ["m"]),
    helper.make_node("Cast", ["m"], ["y"], to=TensorProto.FLOAT)])
case("double_rbf", "Exp", {"x": (rng.random((3, 5)) * 6 - 3).astype(np.float32)}, nodes=[
    helper.make_node("Cast", ["x"], ["xd"], to=TensorProto.DOUBLE), helper.make_node("Unsqueeze", ["xd", "ax"], ["xu"]),
    helper.make_node("Sub", ["xu", "centers"], ["d"]), helper.make_node("Pow", ["d", "two"], ["d2"]),
    helper.make_node("Neg", ["d2"], ["nd"]), helper.make_node("Div", ["nd", "sig"], ["z"]), helper.make_node("Exp", ["z"], ["e"]),
    helper.make_node("Cast", ["e"], ["y"], to=TensorProto.FLOAT)],
    inits={"ax": np.array([-1], np.int64), "centers": np.linspace(-3, 3, 7).astype(np.float64), "two": np.float64(2.0), "sig": np.float64(0.37)})
case("double_where_max_min_abs", "Where", {"x": np.array([0, 1.5, -2.5, 0, 7, -0.0, 1e-3], np.float32)}, nodes=[
    helper.make_node("Cast", ["x"], ["xd"], to=TensorProto.DOUBLE), helper.make_node("Abs", ["xd"], ["a"]),
    helper.make_node("Equal", ["a", "zero"], ["isz"]), helper.make_node("Max", ["a", "lo"], ["m1"]), helper.make_node("Min", ["m1", "hi"], ["m2"]),
    helper.make_node("Div", ["one", "m2"], ["inv"]), helper.make_node("Where", ["isz", "zero", "inv"], ["w"]),
    helper.make_node("Cast", ["w"], ["y"], to=TensorProto.FLOAT)],
    inits={"zero": np.float64(0.0), "one": np.float64(1.0), "lo": np.float64(1e-3), "hi": np.float64(5.0)})
case("double_expand_reshape", "Expand", {"x": spiked(3, 1)}, nodes=[
    helper.make_node("Cast", ["x"], ["xd"], to=TensorProto.DOUBLE), helper.make_node("Expand", ["xd", "shp"], ["e"]),
    helper.make_node("Reshape", ["e", "flat"], ["r"]), helper.make_node("Cast", ["r"], ["y"], to=TensorProto.FLOAT)],
    inits={"shp": np.array([3, 4], np.int64), "flat": np.array([12], np.int64)})
case("castlike_double_roundtrip", "CastLike", {"x": spiked(4, 3)}, nodes=[
    helper.make_node("CastLike", ["x", "likeD"], ["xd"]), helper.make_node("Mul", ["xd", "k"], ["m"]),
    helper.make_node("CastLike", ["m", "x"], ["y"])], inits={"likeD": np.array([1.0], np.float64), "k": np.float64(1.0000000001)})

# --- the last six ops (census of the 35 other bundled graphs): Pad, CumSum, Einsum, GatherND, ScatterND, ScatterElements ---
def padv(*a):
    return np.array(a, dtype=np.int64)


case("pad_rank2_value", "Pad", {"x": spiked(3, 4)}, nodes=[helper.make_node("Pad", ["x", "pads", "v"], ["y"], mode="constant")], inits={"pads": padv(1, 2, 3, 0), "v": np.float32(-100.0)})
case("pad_rank2_default_zero", "Pad", {"x": spiked(3, 4)}, nodes=[helper.make_node("Pad", ["x", "pads"], ["y"], mode="constant")], inits={"pads": padv(0, 1, 2, 3)})
case("pad_nan_value", "Pad", {"x": spiked(2, 3)}, nodes=[helper.make_node("Pad", ["x", "pads", "v"], ["y"], mode="constant")], inits={"pads": padv(0, 2, 1, 0), "v": np.float32(NAN)})
case("pad_rank4_axis2", "Pad", {"x": spiked(2, 1, 3, 2)}, nodes=[helper.make_node("Pad", ["x", "pads", "v"], ["y"], mode="constant")], inits={"pads": padv(0, 0, 4, 0, 0, 0, 0, 0), "v": np.float32(-100.0)})
case("pad_rank4_last_after", "Pad", {"x": spiked(2, 3, 2, 3)}, nodes=[helper.make_node("Pad", ["x", "pads"], ["y"], mode="constant")], inits={"pads": padv(0, 0, 0, 0, 0, 0, 0, 5)})
case("pad_rank3_both_sides", "Pad", {"x": spiked(2, 3, 4)}, nodes=[helper.make_node("Pad", ["x", "pads", "v"], ["y"], mode="constant")], inits={"pads": padv(1, 1, 1, 1, 1, 1), "v": np.float32(7.5)})
case("pad_rank5_one_axis", "Pad", {"x": spiked(2, 3, 2, 2, 2)}, nodes=[helper.make_node("Pad", ["x", "pads", "v"], ["y"], mode="constant")], inits={"pads": padv(0, 0, 0, 3, 0, 0, 0, 0, 0, 0), "v": np.float32(0.0)})
case("pad_dynamic_pads", "Pad", {"x": spiked(3, 4), "r": rand(5, 2)}, nodes=[
    helper.make_node("Shape", ["r"], ["s"]), helper.make_node("Gather", ["s", "z"], ["n"]), helper.make_node("Unsqueeze", ["n", "u0"], ["n1"]),
    helper.make_node("Concat", ["zero4", "n1", "zero1"], ["pads"], axis=0), helper.make_node("Pad", ["x", "pads"], ["y"], mode="constant")],
    inits={"z": np.array(0, np.int64), "u0": padv(0), "zero4": padv(0, 0, 0)[:2], "zero1": padv(0)})
for tag, shape, axis, excl, rev in [("last", (3, 5), -1, 0, 0), ("first", (4, 3), 0, 0, 0), ("mid_rank3", (2, 4, 3), 1, 0, 0), ("exclusive", (3, 5), -1, 1, 0), ("reverse", (3, 5), -1, 0, 1), ("exclusive_reverse", (3, 5), 1, 1, 1)]:
    case(f"cumsum_{tag}", "CumSum", {"x": spiked(*shape)}, inits={"axis": np.array(axis, np.int64)}, exclusive=excl, reverse=rev)
case("cumsum_long", "CumSum", {"x": rand(3, 200)}, inits={"axis": np.array(-1, np.int64)})
case("einsum_rows_by_jhd", "Einsum", {"a": rand(2, 3, 5), "b": rand(2, 2, 4, 5)}, equation="... s, j h d s -> ... j h d")
case("einsum_hd_by_hds", "Einsum", {"a": rand(2, 3, 2, 4), "b": rand(2, 4, 5)}, equation="... h d, h d s -> ... s")
case("einsum_s_by_hds", "Einsum", {"a": rand(3, 2, 5), "b": rand(2, 4, 5)}, equation="... s, h d s -> ... h d")
case("einsum_outer", "Einsum", {"a": spiked(3, 2), "b": spiked(4)}, equation="..., f -> ... f")
case("einsum_outer_scalar_a", "Einsum", {"a": np.float32(2.5), "b": spiked(4)}, equation="..., f -> ... f")
case("einsum_rank4_rows", "Einsum", {"a": rand(2, 2, 3, 5), "b": rand(3, 2, 4, 5)}, equation="... s, j h d s -> ... j h d")
case("gathernd_rows", "GatherND", {"d": spiked(4, 3, 2)}, inits={"i": np.array([[0, 2], [3, 1], [1, 0]], np.int64)}, batch_dims=0)
case("gathernd_full_index", "GatherND", {"d": spiked(4, 3)}, inits={"i": np.array([[0, 2], [3, 1]], np.int64)}, batch_dims=0)
case("gathernd_first_axis", "GatherND", {"d": spiked(4, 3)}, inits={"i": np.array([[2], [0], [3]], np.int64)}, batch_dims=0)
case("gathernd_negative", "GatherND", {"d": spiked(4, 3)}, inits={"i": np.array([[-1, -3], [0, 0]], np.int64)}, batch_dims=0)
case("gathernd_batch_of_indices", "GatherND", {"d": spiked(5, 4)}, inits={"i": np.array([[[0], [4]], [[2], [2]]], np.int64)}, batch_dims=0)
case("gathernd_dynamic", "GatherND", {"d": spiked(4, 3), "u": (rng.random((3, 2)) * np.array([3.99, 2.99])).astype(np.float32)}, nodes=[
    helper.make_node("Cast", ["u"], ["ui"], to=TensorProto.INT64), helper.make_node("GatherND", ["d", "ui"], ["y"], batch_dims=0)])
case("scatternd_rows", "ScatterND", {"d": spiked(4, 3), "u": spiked(2, 3)}, nodes=[helper.make_node("ScatterND", ["d", "i", "u"], ["y"])], inits={"i": np.array([[1], [3]], np.int64)})
case("scatternd_elements", "ScatterND", {"d": spiked(3, 3, 2), "u": spiked(2, 2)}, nodes=[helper.make_node("ScatterND", ["d", "i", "u"], ["y"])], inits={"i": np.array([[0, 1], [2, 2]], np.int64)})
case("scatternd_duplicate_last_wins", "ScatterND", {"d": spiked(4), "u": spiked(3)}, nodes=[helper.make_node("ScatterND", ["d", "i", "u"], ["y"])], inits={"i": np.array([[1], [1], [2]], np.int64)})
case("scatternd_negative", "ScatterND", {"d": spiked(4, 2), "u": spiked(2, 2)}, nodes=[helper.make_node("ScatterND", ["d", "i", "u"], ["y"])], inits={"i": np.array([[-1], [0]], np.int64)})
case("scatternd_dynamic_index", "ScatterND", {"d": spiked(5, 2), "u": spiked(2, 2), "f": np.array([[1.7], [3.2]], np.float32)}, nodes=[
    helper.make_node("Cast", ["f"], ["fi"], to=TensorProto.INT64), helper.make_node("ScatterND", ["d", "fi", "u"], ["y"])])
case("scatterelements_last", "ScatterElements", {"d": spiked(3, 4), "u": spiked(3, 2)}, nodes=[helper.make_node("ScatterElements", ["d", "i", "u"], ["y"], axis=-1)], inits={"i": np.array([[0, 3], [1, 2], [3, 0]], np.int64)})
case("scatterelements_axis0", "ScatterElements", {"d": spiked(4, 3), "u": spiked(2, 3)}, nodes=[helper.make_node("ScatterElements", ["d", "i", "u"], ["y"], axis=0)], inits={"i": np.array([[0, 3, 1], [2, 0, 3]], np.int64)})
case("scatterelements_axis1_rank3", "ScatterElements", {"d": spiked(2, 3, 2), "u": spiked(2, 2, 2)}, nodes=[helper.make_node("ScatterElements", ["d", "i", "u"], ["y"], axis=1)], inits={"i": np.array([[[0, 1], [2, 0]], [[1, 1], [0, 2]]], np.int64)})
case("scatterelements_negative", "ScatterElements", {"d": spiked(2, 4), "u": spiked(2, 2)}, nodes=[helper.make_node("ScatterElements", ["d", "i", "u"], ["y"], axis=1)], inits={"i": np.array([[-1, -4], [0, -2]], np.int64)})

# --- review round 3: integer semantics on integers computed from float data, folded Sign/Relu on ints, folded Clip default bounds ---
ints6 = np.array([7, -7, 9, -9, 5, -5, 0, 1], np.float32)
case("int_div_truncates", "Div", {"x": ints6}, nodes=[
    helper.make_node("Cast", ["x"], ["xi"], to=TensorProto.INT64), helper.make_node("Div", ["xi", "two"], ["q"]),
    helper.make_node("Mul", ["q", "two"], ["m"]), helper.make_node("Cast", ["m"], ["y"], to=TensorProto.FLOAT)], inits={"two": np.array(2, np.int64)})
case("int_div_two_runtime_ints", "Div", {"x": ints6, "d": np.array([2, 2, -2, -2, 3, 3, 4, -1], np.float32)}, nodes=[
    helper.make_node("Cast", ["x"], ["xi"], to=TensorProto.INT64), helper.make_node("Cast", ["d"], ["di"], to=TensorProto.INT64),
    helper.make_node("Div", ["xi", "di"], ["q"]), helper.make_node("Cast", ["q"], ["y"], to=TensorProto.FLOAT)])
case("int_reducemean_truncates", "ReduceMean", {"x": np.array([[1, 2, 2], [-1, -2, -2], [5, 5, 6]], np.float32)}, nodes=[
    helper.make_node("Cast", ["x"], ["xi"], to=TensorProto.INT64), helper.make_node("ReduceMean", ["xi", "ax"], ["m"], keepdims=0),
    helper.make_node("Cast", ["m"], ["y"], to=TensorProto.FLOAT)], inits={"ax": ax(-1)})
case("fold_sign_of_int", "Sign", {"x": rand(3, 4)}, nodes=[
    helper.make_node("Shape", ["x"], ["s"]), helper.make_node("Gather", ["s", "i0"], ["n"]), helper.make_node("Sub", ["n", "five"], ["d"]),
    helper.make_node("Sign", ["d"], ["sg"]), helper.make_node("Cast", ["sg"], ["y"], to=TensorProto.FLOAT)], inits={"i0": np.array(0, np.int64), "five": np.array(5, np.int64)})
case("fold_relu_of_int", "Relu", {"x": rand(3, 4)}, nodes=[
    helper.make_node("Shape", ["x"], ["s"]), helper.make_node("Gather", ["s", "i0"], ["n"]), helper.make_node("Sub", ["n", "five"], ["d"]),
    helper.make_node("Relu", ["d"], ["r"]), helper.make_node("Cast", ["r"], ["y"], to=TensorProto.FLOAT)], inits={"i0": np.array(0, np.int64), "five": np.array(5, np.int64)})
case("fold_clip_default_bound_inf", "Clip", {"x": rand(2)}, nodes=[
    helper.make_node("Constant", [], ["big"], value=numpy_helper.from_array(np.array([np.inf, -np.inf, 3.0, -3.0], np.float32), "big")),
    helper.make_node("Clip", ["big", "lo"], ["c"]), helper.make_node("Add", ["x", "z2"], ["xx"]), helper.make_node("Identity", ["c"], ["y"])],
    inits={"lo": np.float32(-1.0), "z2": np.float32(0.0)})

# --- P4 round: strided Slice (innermost axis, negative steps), float16 round trips, ScatterND/ScatterElements with empty indices ---
case("slice_inner_step2", "Slice", {"x": spiked(3, 2, 2, 8)}, inits={"starts": ax(0), "ends": ax(I64MAX), "axes": ax(3), "steps": ax(2)})
case("slice_inner_step2_from1", "Slice", {"x": spiked(3, 2, 2, 8)}, inits={"starts": ax(1), "ends": ax(I64MAX), "axes": ax(3), "steps": ax(2)})
case("slice_inner_step3_partial", "Slice", {"x": spiked(4, 10)}, inits={"starts": ax(1), "ends": ax(9), "axes": ax(-1), "steps": ax(3)})
case("slice_negative_step_reverse", "Slice", {"x": spiked(3, 5)}, inits={"starts": ax(-1), "ends": ax(I64MIN), "axes": ax(1), "steps": ax(-1)})
case("slice_negative_step_outer", "Slice", {"x": spiked(6, 3)}, inits={"starts": ax(4), "ends": ax(0), "axes": ax(0), "steps": ax(-2)})
case("slice_negative_step_two_axes", "Slice", {"x": spiked(4, 5, 2)}, inits={"starts": ax(-1, 3), "ends": ax(I64MIN, 0), "axes": ax(0, 1), "steps": ax(-1, -2)})
case("slice_inner_step_rank6", "Slice", {"x": spiked(2, 2, 2, 2, 2, 6)}, inits={"starts": ax(0), "ends": ax(I64MAX), "axes": ax(5), "steps": ax(2)})
f16 = np.array([0.1, 1.0, 65504.0, 65519.0, 65520.0, 70000.0, 6.0e-8, 3.0e-8, 1e-5, -2.5, 1.0009765625, 1.00048828125, NAN, INF, -INF, 0.0, -0.0, 1234.5678, 2049.0, 2050.0, 2051.0], np.float32)
case("cast_float16_roundtrip", "Cast", {"x": f16}, nodes=[
    helper.make_node("Cast", ["x"], ["h"], to=TensorProto.FLOAT16), helper.make_node("Cast", ["h"], ["y"], to=TensorProto.FLOAT)])
case("scatternd_empty_indices", "ScatterND", {"d": spiked(4, 3), "u": np.zeros((0, 3), np.float32)}, nodes=[helper.make_node("ScatterND", ["d", "i", "u"], ["y"])],
     inits={"i": np.zeros((0, 1), np.int64)})
case("scatterelements_empty_indices", "ScatterElements", {"d": spiked(3, 4), "u": np.zeros((3, 0), np.float32)}, nodes=[helper.make_node("ScatterElements", ["d", "i", "u"], ["y"], axis=1)],
     inits={"i": np.zeros((3, 0), np.int64)})

print(len(os.listdir(OUT)) // 2, "cases in", os.path.normpath(OUT))
