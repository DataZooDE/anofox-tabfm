//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P2b (docs/GGML_PLAN.md): the shape and constant pass, checked against ONNX Runtime.
//
// The goldens are the shapes ORT produced for a sample of the real Nori 1.5 graph's intermediate tensors at
// several (rows, train, features) combinations (tools/ggml_parity/gen_nori15_shape_goldens.py; shapes only, no
// weights). The plain bundled graph has the same tensor names as the ext graph the goldens came from.
// Written red first against a stub that threw "not implemented".
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ggml.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace duckdb::anofox;
namespace fs = std::filesystem;

namespace {

struct Golden {
	std::map<std::string, std::vector<int64_t>> shapes;
	std::map<std::string, std::vector<int64_t>> ints; // integer/bool tensors of at most 256 elements
};

Golden ReadShapes(const std::string &path) {
	std::ifstream f(path);
	REQUIRE(f.good());
	Golden golden;
	auto &shapes = golden.shapes;
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty()) {
			continue;
		}
		const auto space = line.find(' ');
		const auto eq = line.find(" = ");
		std::vector<int64_t> dims;
		if (space != std::string::npos) {
			std::stringstream ss(line.substr(space + 1, eq == std::string::npos ? std::string::npos : eq - space - 1));
			std::string tok;
			while (std::getline(ss, tok, ',')) {
				dims.push_back(std::stoll(tok));
			}
		}
		shapes[line.substr(0, space)] = dims;
		if (eq != std::string::npos) {
			std::vector<int64_t> values;
			std::stringstream vs(line.substr(eq + 3));
			std::string tok;
			while (std::getline(vs, tok, ',')) {
				values.push_back(std::stoll(tok));
			}
			golden.ints[line.substr(0, space)] = values;
		}
	}
	return golden;
}

std::string Str(const std::vector<int64_t> &s) {
	std::string t = "[";
	for (size_t i = 0; i < s.size(); i++) {
		t += (i ? "," : "") + std::to_string(s[i]);
	}
	return t + "]";
}

} // namespace

TEST_CASE("shape_eval: every sampled tensor of the Nori 1.5 graph has ORT's shape, and every integer one its values", "[tabfm][shape_eval]") {
	// Tests run from the repository root (as test_tabfm_ckpt.cpp and test_tabfm_mlx_ops.cpp assume).
	REQUIRE(fs::exists("test/fixtures/ggml_shapes"));
	const auto graph = ::anofox::onnxread::ParseModelFile("resources/graph_nori15_regression.onnx");
	std::regex name_re(R"(nori15_regression_T(\d+)_S(\d+)_H(\d+)\.shapes)");
	int configs = 0;
	for (const auto &entry : fs::directory_iterator("test/fixtures/ggml_shapes")) {
		std::smatch m;
		const std::string fname = entry.path().filename().string();
		if (!std::regex_match(fname, m, name_re)) {
			continue;
		}
		const int64_t rows = std::stoll(m[1]), train = std::stoll(m[2]), feats = std::stoll(m[3]);
		INFO("rows " << rows << " train " << train << " features " << feats);
		ShapeEvalInput in;
		in.shapes["x"] = {1, rows, feats};
		in.shapes["y"] = {1, train};
		const auto result = EvaluateShapes(graph, in);
		const auto golden_file = ReadShapes(entry.path().string());
		const auto &golden = golden_file.shapes;
		REQUIRE(golden.size() > 5000);
		REQUIRE(golden_file.ints.size() > 3000);
		std::vector<std::string> wrong;
		size_t compared = 0;
		for (const auto &kv : golden_file.ints) {
			auto it = result.values.find(kv.first);
			if (it == result.values.end()) {
				wrong.push_back(kv.first + ": missing");
				continue;
			}
			if (!it->second.known) {
				continue; // data-dependent (masks computed from y and x): not the pass's to know
			}
			compared++;
			if (it->second.ints != kv.second) {
				wrong.push_back(kv.first + ": folded integer values differ from ORT's");
			}
		}
		INFO("integer tensors compared by value: " << compared);
		CHECK(compared > 1500);
		for (const auto &kv : golden) {
			auto it = result.values.find(kv.first);
			if (it == result.values.end()) {
				wrong.push_back(kv.first + ": missing, want " + Str(kv.second));
			} else if (it->second.shape != kv.second) {
				wrong.push_back(kv.first + ": got " + Str(it->second.shape) + ", want " + Str(kv.second));
			}
		}
		INFO("first mismatches: " << (wrong.empty() ? std::string("none") : wrong[0] + (wrong.size() > 1 ? " | " + wrong[1] : "")));
		CHECK(wrong.size() == 0);
		configs++;
	}
	REQUIRE(configs >= 6);
}

// ---------------------------------------------------------------------------------------------------------------
// Semantics the real graph leans on, pinned on hand-built graphs. The first two are bugs a throwaway spike hit
// on the real Nori 1.5 graph: INT64_MAX ("to the end") passed through a double became INT64_MIN and emptied a
// Slice, and a rank-0 index promoted to rank 1 gives Gather an extra axis.
// ---------------------------------------------------------------------------------------------------------------
namespace {

using ::anofox::onnxread::Attribute;
using ::anofox::onnxread::Graph;
using ::anofox::onnxread::Node;
using ::anofox::onnxread::Tensor;

Tensor Int64Tensor(std::vector<int64_t> dims, std::vector<int64_t> values) {
	Tensor t;
	t.dims = std::move(dims);
	t.data_type = 7;
	t.raw.assign(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(int64_t));
	return t;
}

Tensor FloatTensor(std::vector<int64_t> dims, std::vector<float> values) {
	Tensor t;
	t.dims = std::move(dims);
	t.data_type = 1;
	t.raw.assign(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
	return t;
}

Node Const(const std::string &out, Tensor t) {
	Node n;
	n.op_type = "Constant";
	n.name = "const_" + out;
	n.outputs = {out};
	Attribute a;
	a.name = "value";
	a.has_tensor = true;
	a.t = std::move(t);
	n.attributes.push_back(std::move(a));
	return n;
}

Node Op(const std::string &op, std::vector<std::string> in, std::vector<std::string> out) {
	Node n;
	n.op_type = op;
	n.name = op + "_" + out[0];
	n.inputs = std::move(in);
	n.outputs = std::move(out);
	return n;
}

Node WithInt(Node n, const std::string &name, int64_t v) {
	Attribute a;
	a.name = name;
	a.i = v;
	n.attributes.push_back(a);
	return n;
}

Node WithInts(Node n, const std::string &name, std::vector<int64_t> v) {
	Attribute a;
	a.name = name;
	a.ints = std::move(v);
	n.attributes.push_back(a);
	return n;
}

ShapeEvalResult Eval(Graph g, ShapeEvalInput in = ShapeEvalInput()) {
	return EvaluateShapes(g, in);
}

} // namespace

// The census the rank-reduction pass (P2a) is for: how many of the real graph's float tensors still have more than
// four dimensions of size > 1. Those are the ones ggml cannot hold without a custom op or a smarter merge.
TEST_CASE("shape_eval: census of Nori 1.5 tensors above four dimensions", "[tabfm][shape_eval][census]") {
	const auto graph = ::anofox::onnxread::ParseModelFile("resources/graph_nori15_regression.onnx");
	ShapeEvalInput in;
	in.shapes["x"] = {1, 130, 10};
	in.shapes["y"] = {1, 100};
	const auto r = EvaluateShapes(graph, in);
	size_t floats = 0, raw_above_4 = 0, effective_above_4 = 0, max_raw = 0, max_effective = 0;
	for (const auto &kv : r.values) {
		if (kv.second.dtype != ShapeDtype::F32 && kv.second.dtype != ShapeDtype::F64) { // float tensors, whatever their precision
			continue;
		}
		floats++;
		const auto &s = kv.second.shape;
		size_t effective = 0;
		for (auto d : s) {
			effective += d != 1 ? 1 : 0;
		}
		raw_above_4 += s.size() > 4 ? 1 : 0;
		effective_above_4 += effective > 4 ? 1 : 0;
		max_raw = std::max(max_raw, s.size());
		max_effective = std::max(max_effective, effective);
	}
	WARN("float tensors " << floats << ", rank>4: " << raw_above_4 << " (max rank " << max_raw << "), more than 4 non-1 dims: "
	                      << effective_above_4 << " (max " << max_effective << ")");
	REQUIRE(floats == 6433);
	REQUIRE(raw_above_4 == 407);
	REQUIRE(effective_above_4 == 78);
	REQUIRE(max_raw == 6);
	REQUIRE(max_effective == 5); // never more than five real dimensions: the P3 rank-5 problem in one number
}

// ---------------------------------------------------------------------------------------------------------------
// Review round 1 (P2): float32 arithmetic, exact integer semantics, refusals instead of undefined behaviour.
// ---------------------------------------------------------------------------------------------------------------
