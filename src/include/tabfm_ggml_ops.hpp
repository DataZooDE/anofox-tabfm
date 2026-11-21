#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ggml_ops.hpp — the ONNX ops the ggml CPU engine can run, known without building a graph.
//
// Same role as tabfm_mlx_ops.hpp for the MLX plugin: the engine refuses a graph that uses an op its lowering
// table lacks, and says which. The table starts nearly empty on purpose (docs/GGML_PLAN.md, P0) and grows op by
// op in P3, each with an ORT-generated golden. P1 replaces this seed with the full native / composite / custom
// coverage matrix.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

//! How an ONNX op reaches ggml (docs/GGML_PLAN.md, P1; the matrix with reasons is in docs/GGML_LOWERING.md).
//!   NATIVE     one ggml op (plus views/permutes for layout)
//!   COMPOSITE  a few ggml ops, or a native op with a restriction we work around (broadcast side, axis, constants)
//!   CUSTOM     no ggml op: a host kernel run inside the graph via ggml_custom_4d / ggml_map_custom*
//!   HOST       never reaches ggml: evaluated on the host while the shape pass folds the graph for an input shape
enum class GgmlLowerKind { NATIVE, COMPOSITE, CUSTOM, HOST };

struct GgmlLowering {
	GgmlLowerKind kind;
	const char *how;
};

//! P1: every op in every bundled CPU graph, classified, with the reason. The reasons were checked against the
//! vendored ggml.h (no pow, select, comparison, predicate, scatter or min/max-reduce op exists there).
inline const std::map<std::string, GgmlLowering> &GgmlLoweringTable() {
	static const std::map<std::string, GgmlLowering> table = {
	    {"Abs", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_ABS"}},
	    {"Add", {GgmlLowerKind::NATIVE, "ggml_add; src1 broadcasts into src0, so the lowering swaps operands when the larger one is on the right. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel."}},
	    {"And", {GgmlLowerKind::CUSTOM, "no ggml logical ops; host kernel over I32 masks (NaN-safe masks are the point of the graphs)"}},
	    {"Cast", {GgmlLowerKind::COMPOSITE, "ggml_cast between F32/F16/I32; int64 and bool values live on the host or as I32 masks"}},
	    {"CastLike", {GgmlLowerKind::COMPOSITE, "same as Cast with the target dtype taken from the second operand"}},
	    {"Ceil", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_CEIL"}},
	    {"Clip", {GgmlLowerKind::CUSTOM, "ggml_clamp is MAX(MIN(x,hi),lo), which turns NaN into hi, while ONNX Clip passes NaN through and the graphs mask with NaN; the bounds are folded constants, the kernel keeps NaN"}},
	    {"Concat", {GgmlLowerKind::NATIVE, "ggml_concat along the (permuted) axis"}},
	    {"Constant", {GgmlLowerKind::HOST, "folded into a constant tensor by the shape pass"}},
	    {"ConstantOfShape", {GgmlLowerKind::HOST, "folded: the shape operand is known per input shape"}},
	    {"Cos", {GgmlLowerKind::NATIVE, "ggml_cos"}},
	    {"CumSum", {GgmlLowerKind::NATIVE, "ggml_cumsum along dim 0; other axes permute first"}},
	    {"Div", {GgmlLowerKind::COMPOSITE, "ggml_div; ggml_repeat of the left operand when the right one is the larger. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel."}},
	    {"Einsum", {GgmlLowerKind::COMPOSITE, "permutes + ggml_mul_mat for the equations the bundled graphs use (verified in P3)"}},
	    {"Equal", {GgmlLowerKind::CUSTOM, "no ggml comparison ops; host kernel producing an I32 mask"}},
	    {"Erf", {GgmlLowerKind::CUSTOM, "only GGML_UNARY_OP_GELU_ERF exists (x*0.5*(1+erf(x/sqrt2)), singular to invert at 0)"}},
	    {"Exp", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_EXP"}},
	    {"Expand", {GgmlLowerKind::NATIVE, "ggml_repeat to the broadcast shape"}},
	    {"Floor", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_FLOOR"}},
	    {"Gather", {GgmlLowerKind::COMPOSITE, "ggml_get_rows with I32 indices on dim 0 (int64 indices are cast first); other axes permute first"}},
	    {"GatherElements", {GgmlLowerKind::CUSTOM, "no per-element gather along an axis in ggml; host kernel"}},
	    {"GatherND", {GgmlLowerKind::CUSTOM, "no n-d index gather in ggml; host kernel"}},
	    {"Gemm", {GgmlLowerKind::COMPOSITE, "ggml_mul_mat + ggml_add bias, with alpha/beta/transA/transB handled by layout"}},
	    {"Greater", {GgmlLowerKind::CUSTOM, "no ggml comparison ops; host kernel producing an I32 mask"}},
	    {"GreaterOrEqual", {GgmlLowerKind::CUSTOM, "no ggml comparison ops; host kernel producing an I32 mask"}},
	    {"Identity", {GgmlLowerKind::HOST, "an alias, resolved while lowering"}},
	    {"IsInf", {GgmlLowerKind::CUSTOM, "no ggml predicate ops; host kernel producing an I32 mask"}},
	    {"IsNaN", {GgmlLowerKind::CUSTOM, "no ggml predicate ops; host kernel producing an I32 mask"}},
	    {"LayerNormalization", {GgmlLowerKind::COMPOSITE, "ggml_norm, then ggml_mul by scale and ggml_add of bias"}},
	    {"Less", {GgmlLowerKind::CUSTOM, "no ggml comparison ops; host kernel producing an I32 mask"}},
	    {"LessOrEqual", {GgmlLowerKind::CUSTOM, "no ggml comparison ops; host kernel producing an I32 mask"}},
	    {"Log", {GgmlLowerKind::NATIVE, "ggml_log"}},
	    {"MatMul", {GgmlLowerKind::COMPOSITE, "ggml_mul_mat on a transposed right operand; batch broadcast only where src0's batch divides src1's"}},
	    {"Max", {GgmlLowerKind::CUSTOM, "tensor-tensor maximum has no ggml op; the scalar-constant form cannot use ggml_clamp either (NaN becomes the bound)"}},
	    {"Min", {GgmlLowerKind::CUSTOM, "tensor-tensor minimum has no ggml op; the scalar-constant form cannot use ggml_clamp either (NaN becomes the bound)"}},
	    {"Mod", {GgmlLowerKind::CUSTOM, "no ggml modulo; ONNX integer Mod takes the sign of the divisor, host kernel"}},
	    {"Mul", {GgmlLowerKind::NATIVE, "ggml_mul; src1 broadcasts into src0, operands swapped when needed. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel."}},
	    {"Neg", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_NEG on float tensors; integer negation is HOST or custom"}},
	    {"Not", {GgmlLowerKind::CUSTOM, "no ggml logical ops; host kernel over I32 masks"}},
	    {"Or", {GgmlLowerKind::CUSTOM, "no ggml logical ops; host kernel over I32 masks"}},
	    {"Pad", {GgmlLowerKind::COMPOSITE, "ggml_pad_ext zero-fills only: a non-zero constant_value needs an added mask, and reflect/edge modes have no ggml op (modes in use checked in P3)"}},
	    {"Pow", {GgmlLowerKind::CUSTOM, "no ggml pow: constant exponents 2, 0.5 and 3 lower to sqr/sqrt/sqr*x (nearly all nodes), only the dynamic-exponent nodes need the kernel"}},
	    {"Range", {GgmlLowerKind::HOST, "folded: its bounds come from Shape values the shape pass knows"}},
	    {"Reciprocal", {GgmlLowerKind::COMPOSITE, "ggml_div of a constant ones tensor by x"}},
	    {"ReduceMax", {GgmlLowerKind::CUSTOM, "ggml has argmax but no max reduction; host kernel"}},
	    {"ReduceMean", {GgmlLowerKind::COMPOSITE, "ggml_mean over dim 0 after permuting or merging the reduced axes"}},
	    {"ReduceMin", {GgmlLowerKind::CUSTOM, "ggml has no min reduction; host kernel"}},
	    {"ReduceProd", {GgmlLowerKind::CUSTOM, "ggml has no product reduction; host kernel"}},
	    {"ReduceSum", {GgmlLowerKind::COMPOSITE, "ggml_sum_rows over dim 0 after permuting or merging the reduced axes"}},
	    {"Relu", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_RELU"}},
	    {"Reshape", {GgmlLowerKind::NATIVE, "ggml_reshape_1d..4d; five- and six-dimensional shapes go through the rank-reduction pass first"}},
	    {"Round", {GgmlLowerKind::CUSTOM, "ggml_round rounds half away from zero, ONNX rounds half to even"}},
	    {"ScatterElements", {GgmlLowerKind::CUSTOM, "no scatter along an axis in ggml (ggml_set_rows covers rows only); host kernel"}},
	    {"ScatterND", {GgmlLowerKind::CUSTOM, "no n-d scatter in ggml; host kernel"}},
	    {"SequenceAt", {GgmlLowerKind::HOST, "sequences are resolved to views at lowering time"}},
	    {"Shape", {GgmlLowerKind::HOST, "evaluated by the shape pass"}},
	    {"Sigmoid", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_SIGMOID"}},
	    {"Sign", {GgmlLowerKind::NATIVE, "ggml_sgn"}},
	    {"Sin", {GgmlLowerKind::NATIVE, "ggml_sin"}},
	    {"Size", {GgmlLowerKind::HOST, "evaluated by the shape pass"}},
	    {"Slice", {GgmlLowerKind::COMPOSITE, "ggml_view_4d with strides, then ggml_cont; negative steps need a copy through a custom kernel"}},
	    {"Softmax", {GgmlLowerKind::COMPOSITE, "ggml_soft_max works on dim 0; other axes permute first"}},
	    {"Softplus", {GgmlLowerKind::NATIVE, "ggml_softplus"}},
	    {"SplitToSequence", {GgmlLowerKind::HOST, "resolved to views at lowering time"}},
	    {"Sqrt", {GgmlLowerKind::NATIVE, "ggml_sqrt"}},
	    {"Squeeze", {GgmlLowerKind::NATIVE, "a reshape view"}},
	    {"Sub", {GgmlLowerKind::COMPOSITE, "ggml_sub; ggml_repeat of the left operand when the right one is the larger. Float tensors only: the ggml CPU kernels abort on I32/I64, so integer arithmetic is HOST (folded) or a custom kernel."}},
	    {"Tanh", {GgmlLowerKind::NATIVE, "GGML_UNARY_OP_TANH"}},
	    {"TopK", {GgmlLowerKind::COMPOSITE, "ggml_argsort_top_k plus ggml_get_rows for the values: ggml_top_k itself returns indices in no particular order (its CPU kernel swaps the first two) where ONNX TopK is sorted; tie order checked in P3"}},
	    {"Transpose", {GgmlLowerKind::NATIVE, "ggml_permute followed by ggml_cont where a contiguous layout is needed"}},
	    {"Unsqueeze", {GgmlLowerKind::NATIVE, "a reshape view"}},
	    {"Where", {GgmlLowerKind::CUSTOM, "no ggml select; arithmetic emulation is wrong because masked values are NaN (NaN*0 = NaN)"}},
	};
	return table;
}

//! Implemented ops (grows in P3). Always a subset of GgmlLoweringTable().
inline const std::set<std::string> &GgmlSupportedOps() {
	static const std::set<std::string> ops = {"Identity", "Reshape", "Squeeze", "Unsqueeze", "Neg", "Abs", "Sqrt", "Exp", "Log", "Tanh",
	                                          "Add",      "Sub",     "Mul",     "Div",       "Transpose", "MatMul", "Softmax",
	                                          "ReduceMean", "ReduceSum", "LayerNormalization", "Expand", "Concat", "Slice", "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Not", "IsNaN", "IsInf", "Where",
	                                          "Ceil", "Floor", "Round", "Sign", "Relu", "Sigmoid", "Sin", "Cos", "Softplus", "Reciprocal", "Erf", "Pow", "Max",
	                                          "Min", "Mod", "Clip", "Gemm", "Cast", "ReduceMax", "ReduceMin", "ReduceProd", "Shape", "Size", "Constant", "ConstantOfShape",
	                                          "Range", "Gather", "CastLike", "TopK", "GatherElements", "SplitToSequence", "SequenceAt", "Pad", "CumSum", "Einsum", "GatherND", "ScatterND", "ScatterElements"};
	return ops;
}

//! The ops among `op_types` (one entry per node, repeats fine) the ggml engine does not implement, sorted and
//! unique. Empty means the graph can run on it.
inline std::vector<std::string> GgmlMissingOps(const std::vector<std::string> &op_types) {
	std::set<std::string> missing;
	const auto &ops = GgmlSupportedOps();
	for (const auto &op : op_types) {
		if (ops.find(op) == ops.end()) {
			missing.insert(op);
		}
	}
	return std::vector<std::string>(missing.begin(), missing.end());
}

} // namespace anofox
} // namespace duckdb
