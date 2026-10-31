#pragma once

//===----------------------------------------------------------------------===//
// tabfm_mlx_ops.hpp — the ONNX ops the MLX plugin can execute, known to the
// extension without loading the plugin.
//
// The plugin runs a model's ext graph through its own interpreter
// (tabfm_mlx_graph_ops.inc) and refuses a graph that uses an op its Handlers()
// table lacks. tabfm_backends() and 'auto' need that answer earlier, and on
// machines where the plugin does not exist, so the set is mirrored here.
// test/cpp/test_tabfm_mlx_ops.cpp reads the plugin source and fails when the
// two differ: add an op there, add it here.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

//! Every op the interpreter runs: its handler table plus Constant, SplitToSequence and SequenceAt,
//! which the executor handles inline.
inline const std::set<std::string> &MlxImplementedOps() {
	static const std::set<std::string> ops = {
	    "Abs", "Add", "And", "Cast", "CastLike", "Ceil", "Clip", "Concat", "Constant", "Cos", "CumSum", "Div",
	    "Einsum", "Equal", "Erf", "Exp", "Expand", "Floor", "Gather", "GatherElements", "GatherND", "Gemm",
	    "Greater", "GreaterOrEqual", "Identity", "IsInf", "IsNaN", "LayerNormalization", "Less", "LessOrEqual",
	    "Log", "MatMul", "Max", "Min", "Mod", "Mul", "Neg", "Not", "Or", "Pad", "Pow", "Range", "Reciprocal",
	    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "Reshape", "Round", "ScatterElements", "ScatterND",
	    "SequenceAt", "Shape", "Sigmoid", "Sign", "Sin", "Size", "Slice", "Softmax", "Softplus",
	    "SplitToSequence", "Sqrt", "Squeeze", "Sub", "Tanh", "TopK", "Transpose", "Unsqueeze", "Where"};
	return ops;
}

//! The ops among `op_types` (one entry per node, repeats fine) that MLX does not implement, sorted
//! and unique. Empty means the graph can run on 'mlx'.
inline std::vector<std::string> MlxMissingOps(const std::vector<std::string> &op_types) {
	std::set<std::string> missing;
	const auto &ops = MlxImplementedOps();
	for (const auto &op : op_types) {
		if (ops.find(op) == ops.end()) {
			missing.insert(op);
		}
	}
	return std::vector<std::string>(missing.begin(), missing.end());
}

} // namespace anofox
} // namespace duckdb
