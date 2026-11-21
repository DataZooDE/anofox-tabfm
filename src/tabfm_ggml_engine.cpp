#include "tabfm_ggml_engine.hpp"

#include "tabfm_ggml_ops.hpp"

namespace duckdb {
namespace anofox {

std::string GgmlRefusal(const std::vector<std::string> &op_types) {
	const auto missing = GgmlMissingOps(op_types);
	if (missing.empty()) {
		return std::string();
	}
	std::string list;
	for (const auto &op : missing) {
		list += (list.empty() ? "" : ", ") + op;
	}
	return "this graph needs ONNX ops the ggml CPU engine does not implement (" + list +
	       "). Rebuild with TABFM_CPU_ENGINE=ort to run it through ONNX Runtime, and please report these op names.";
}

} // namespace anofox
} // namespace duckdb
