/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_onnx_reader.hpp — just enough ONNX to execute a graph.
 *
 * The MLX plugin needs the node list, the attributes and the initializers'
 * external offsets. It does NOT need onnx's checker, shape inference, version
 * conversion, or protobuf's reflection — and linking libonnx + libprotobuf to
 * get them would tie a small, separately-distributed .dylib to protobuf's ABI
 * and to whatever abseil the extension happened to build against. (Measured:
 * the vcpkg onnx gencode does not compile standalone against its own installed
 * protobuf headers without abseil's full include environment.)
 *
 * So this decodes the protobuf WIRE FORMAT directly. That format is stable by
 * design — field numbers and wire types are the compatibility contract, which
 * is exactly the part of protobuf that never changes — and the subset below is
 * small enough to read in one sitting:
 *
 *   ModelProto      graph=7
 *   GraphProto      node=1, initializer=5, input=11, output=12
 *   NodeProto       input=1, output=2, name=3, op_type=4, attribute=5
 *   AttributeProto  name=1, f=2, i=3, s=4, t=5, floats=7, ints=8, type=20
 *   TensorProto     dims=1, data_type=2, float_data=4, int32_data=5,
 *                   int64_data=7, name=8, raw_data=9, external_data=13,
 *                   data_location=14
 *   StringStringEntry key=1, value=2
 *   ValueInfoProto  name=1
 *
 * Unknown fields are skipped by wire type, so a graph carrying anything else
 * parses fine rather than failing on a field we do not care about.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace anofox {
namespace onnxread {

enum class DataLocation { DEFAULT = 0, EXTERNAL = 1 };

struct Tensor {
	std::string name;
	std::vector<int64_t> dims;
	int32_t data_type = 0;
	DataLocation location = DataLocation::DEFAULT;
	std::map<std::string, std::string> external; // location / offset / length
	std::string raw;                             // inline bytes, when not external
	std::vector<float> floats;
	std::vector<int64_t> int64s;
	std::vector<int32_t> int32s;
};

struct Attribute {
	std::string name;
	int32_t type = 0;
	float f = 0.0f;
	int64_t i = 0;
	std::string s;
	Tensor t;
	bool has_tensor = false;
	std::vector<int64_t> ints;
	std::vector<float> float_list;
};

struct Node {
	std::string op_type;
	std::string name;
	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
	std::vector<Attribute> attributes;

	const Attribute *Attr(const std::string &n) const {
		for (const auto &a : attributes) {
			if (a.name == n) {
				return &a;
			}
		}
		return nullptr;
	}
};

struct Graph {
	std::vector<Node> nodes;
	std::vector<Tensor> initializers;
	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
};

//! Parse an ONNX model file. Throws std::runtime_error naming the file.
Graph ParseModelFile(const std::string &path);

} // namespace onnxread
} // namespace anofox
