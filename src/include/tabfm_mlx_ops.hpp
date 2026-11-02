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
	    "Abs", "Add", "And", "Cast", "CastLike", "Ceil", "Clip", "Concat", "Constant", "ConstantOfShape", "Cos", "CumSum", "Div",
	    "Einsum", "Equal", "Erf", "Exp", "Expand", "Floor", "Gather", "GatherElements", "GatherND", "Gemm",
	    "Greater", "GreaterOrEqual", "Identity", "IsInf", "IsNaN", "LayerNormalization", "Less", "LessOrEqual",
	    "Log", "MatMul", "Max", "Min", "Mod", "Mul", "Neg", "Not", "Or", "Pad", "Pow", "Range", "Reciprocal",
	    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "Reshape", "Round", "ScatterElements", "ScatterND",
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

//! Why a graph that MLX can EXECUTE is still refused there: it runs, and disagrees with the CPU. Empty = no
//! known problem, which is every bundled ext graph today (Apple M3, real weights, 100 rows against ONNX
//! Runtime on the CPU: labels 0/100, regression within 1e-4). Add an entry only with a run of
//! test/sql/tabfm_mlx_real_models.test that shows the model disagreeing; remove one only with a run that
//! shows it agreeing.
//!
//! The last three entries -- tabdpt and tabdpt-v1-3 classification (63-66 of 100 labels) and orion-msp
//! (9 of 100) -- were no misinterpreted op: the interpreter read strided VIEWS to host in storage order
//! (Contiguous() in tabfm_mlx_graph.cpp).
inline std::string MlxKnownWrong(const std::string &model, const std::string &task) {
	(void)model;
	(void)task;
	return std::string();
}

} // namespace anofox
} // namespace duckdb
