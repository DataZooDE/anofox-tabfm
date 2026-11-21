#pragma once

//===----------------------------------------------------------------------===//
// tabfm_shape_eval.hpp — the host-side shape and constant pass of the ggml CPU engine (docs/GGML_PLAN.md, P2).
//
// The bundled graphs were exported with dynamic rows, train rows and features: they compute shapes at run time
// (Shape -> Gather -> Concat -> Reshape) and feed them to Reshape, Expand, Slice and Range. A ggml graph has
// static shapes, so before one is built the host walks the ONNX graph for a concrete input shape and works out
//   - the shape of every tensor, and
//   - the VALUE of every integer or bool tensor that is derived from shapes and constants (about a third of all
//     nodes: the HOST class of docs/GGML_LOWERING.md), so they never reach ggml.
// Float tensors carry a shape only, unless they are small constants. Integer values live in int64 (not double:
// INT64_MAX, used as "to the end" in Slice, does not survive a round trip through a double).
//
//===----------------------------------------------------------------------===//

#include "tabfm_onnx_reader.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {
namespace anofox {

//! F64 is a tensor the ONNX graph computes in double precision (Cast to DOUBLE): its arithmetic is NOT rounded to float32.
enum class ShapeDtype { F32, F64, I64, BOOL };

struct ShapeValue {
	std::vector<int64_t> shape;
	ShapeDtype dtype = ShapeDtype::F32;
	//! True when the element values are known to the host. Always true for I64 and BOOL tensors the pass derived.
	bool known = false;
	std::vector<int64_t> ints; // I64 and BOOL values, when known
	std::vector<double> floats; // F32 values, when known (small constants only)
};

struct ShapeEvalInput {
	//! Shapes of the graph's float inputs, e.g. x -> [1, T, H], y -> [1, S].
	std::map<std::string, std::vector<int64_t>> shapes;
	//! Values of the graph's integer inputs, e.g. train_size, d (rank-0 or rank-1).
	std::map<std::string, std::vector<int64_t>> ints;
	//! Shapes of integer inputs where it matters: a single value is a rank-0 tensor unless its shape is given here (train_size is rank 1: [1]).
	std::map<std::string, std::vector<int64_t>> int_shapes;
	//! Element types of float-held inputs that are not float32 data (a bool mask such as cat_mask): the values stay unknown, the type is known.
	std::map<std::string, ShapeDtype> dtypes;
};

struct ShapeEvalResult {
	std::unordered_map<std::string, ShapeValue> values;
};

//! Walk `graph` for the given input shapes. Throws std::runtime_error naming the node when an op is not
//! supported by the pass or a value it needs (a Reshape target, a Slice bound) depends on tensor data.
ShapeEvalResult EvaluateShapes(const ::anofox::onnxread::Graph &graph, const ShapeEvalInput &input);

//! `f` rounded to the nearest IEEE half-precision value (ties to even; overflow to infinity), returned as a float: what a Cast to FLOAT16
//! and back does. NaN and infinities pass through.
float RoundToHalf(float f);

//! Evaluate ONE node on fully known inputs, with no limit on the element count (the engine runs this at inference time for the
//! nodes ggml cannot hold, the double-precision ones). `inputs` follow node.inputs (null for an omitted optional input); every
//! non-null input must have known values. Returns one value per node output.
std::vector<ShapeValue> EvaluateNode(const ::anofox::onnxread::Node &node, const std::vector<const ShapeValue *> &inputs);

} // namespace anofox
} // namespace duckdb
