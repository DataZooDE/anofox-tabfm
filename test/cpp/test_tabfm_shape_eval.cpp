//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P2b (docs/GGML_PLAN.md): the shape and constant pass, checked against ONNX Runtime.
//
// The goldens are the shapes ORT produced for a sample of the real Nori 1.5 graph's intermediate tensors at
// several (rows, train, features) combinations (tools/ggml_parity/gen_nori15_shape_goldens.py; shapes only, no
// weights). The plain bundled graph has the same tensor names as the ext graph the goldens came from.
// Written red first against a stub that threw "not implemented".
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_shape_eval.hpp"

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

TEST_CASE("shape_eval: Slice to INT64_MAX keeps the whole axis", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("data", Int64Tensor({5}, {10, 11, 12, 13, 14})),
	           Const("starts", Int64Tensor({1}, {1})),
	           Const("ends", Int64Tensor({1}, {INT64_MAX})),
	           Op("Slice", {"data", "starts", "ends"}, {"out"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("out").shape == std::vector<int64_t> {4});
	REQUIRE(r.values.at("out").ints == std::vector<int64_t> {11, 12, 13, 14});
}

TEST_CASE("shape_eval: Slice with a negative step reverses", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("data", Int64Tensor({5}, {10, 11, 12, 13, 14})),
	           Const("starts", Int64Tensor({1}, {-1})),
	           Const("ends", Int64Tensor({1}, {INT64_MIN})),
	           Const("axes", Int64Tensor({1}, {0})),
	           Const("steps", Int64Tensor({1}, {-2})),
	           Op("Slice", {"data", "starts", "ends", "axes", "steps"}, {"out"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("out").ints == std::vector<int64_t> {14, 12, 10});
}

TEST_CASE("shape_eval: a scalar index removes the gathered axis, a vector index keeps it", "[tabfm][shape_eval]") {
	Graph g;
	g.inputs = {"x"};
	g.nodes = {Op("Shape", {"x"}, {"dims"}),
	           Const("scalar_idx", Int64Tensor({}, {1})),
	           Const("vector_idx", Int64Tensor({1}, {1})),
	           Op("Gather", {"dims", "scalar_idx"}, {"rows"}),
	           Op("Gather", {"dims", "vector_idx"}, {"rows_vec"})};
	ShapeEvalInput in;
	in.shapes["x"] = {1, 7, 3};
	const auto r = Eval(g, in);
	REQUIRE(r.values.at("rows").shape.empty());
	REQUIRE(r.values.at("rows").ints == std::vector<int64_t> {7});
	REQUIRE(r.values.at("rows_vec").shape == std::vector<int64_t> {1});
}

TEST_CASE("shape_eval: Reshape takes its target from Shape arithmetic", "[tabfm][shape_eval]") {
	Graph g;
	g.inputs = {"x"};
	g.nodes = {Op("Shape", {"x"}, {"dims"}),
	           Const("lo", Int64Tensor({1}, {0})),
	           Const("hi", Int64Tensor({1}, {2})),
	           Const("axes", Int64Tensor({1}, {0})),
	           Op("Slice", {"dims", "lo", "hi", "axes"}, {"head"}),   // [1, 7]
	           Const("tail", Int64Tensor({1}, {-1})),
	           WithInt(Op("Concat", {"head", "tail"}, {"target"}), "axis", 0), // [1, 7, -1]
	           Op("Reshape", {"x", "target"}, {"y"})};
	ShapeEvalInput in;
	in.shapes["x"] = {1, 7, 3};
	const auto r = Eval(g, in);
	REQUIRE(r.values.at("y").shape == std::vector<int64_t> {1, 7, 3});
}

TEST_CASE("shape_eval: Cast from float saturates instead of overflowing", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("big", FloatTensor({3}, {1e30f, -1e30f, 2.9f})), WithInt(Op("Cast", {"big"}, {"as_int"}), "to", 7)};
	const auto r = Eval(g);
	REQUIRE(r.values.at("as_int").ints == std::vector<int64_t> {INT64_MAX, INT64_MIN, 2});
}

TEST_CASE("shape_eval: integer Mod follows the divisor's sign, Div truncates", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("a", Int64Tensor({4}, {-7, 7, -7, 7})), Const("b", Int64Tensor({4}, {3, -3, -3, 3})),
	           Op("Mod", {"a", "b"}, {"m"}), Op("Div", {"a", "b"}, {"d"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("m").ints == std::vector<int64_t> {2, -2, -1, 1});
	REQUIRE(r.values.at("d").ints == std::vector<int64_t> {-2, -2, 2, 2});
}

TEST_CASE("shape_eval: unsupported ops and data-dependent shapes are refused by name", "[tabfm][shape_eval]") {
	SECTION("an op the pass does not know") {
		Graph g;
		g.inputs = {"x"};
		g.nodes = {Op("Zebra", {"x"}, {"y"})};
		ShapeEvalInput in;
		in.shapes["x"] = {2};
		try {
			Eval(g, in);
			FAIL("expected an exception");
		} catch (const std::runtime_error &e) {
			REQUIRE(std::string(e.what()).find("Zebra") != std::string::npos);
		}
	}
	SECTION("a Reshape target computed from tensor data") {
		Graph g;
		g.inputs = {"x"};
		g.nodes = {WithInt(Op("Cast", {"x"}, {"target"}), "to", 7), Op("Reshape", {"x", "target"}, {"y"})};
		ShapeEvalInput in;
		in.shapes["x"] = {2};
		try {
			Eval(g, in);
			FAIL("expected an exception");
		} catch (const std::runtime_error &e) {
			const std::string message = e.what();
			REQUIRE(message.find("Reshape") != std::string::npos);
			REQUIRE(message.find("depends on tensor data") != std::string::npos);
		}
	}
}

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

TEST_CASE("shape_eval: float arithmetic rounds to float32 like ORT, so size computations agree", "[tabfm][shape_eval]") {
	// int(float32(10) * float32(0.7)) is 7 in float32 (the product rounds to exactly 7.0) but 6 in double.
	Graph g;
	g.nodes = {Const("n", FloatTensor({}, {10.0f})), Const("f", FloatTensor({}, {0.7f})), Op("Mul", {"n", "f"}, {"p"}),
	           WithInt(Op("Cast", {"p"}, {"size"}), "to", 7)};
	const auto r = Eval(g);
	REQUIRE(r.values.at("size").ints == std::vector<int64_t> {7});
}

TEST_CASE("shape_eval: float64 tensors keep double precision and EvaluateNode evaluates at run time without the size cap", "[tabfm][shape_eval]") {
	// Cast(0.1f -> double) * 3.0 stays 0.10000000149011612 * 3 in double; rounding it to float32 would give 0.30000001192092896.
	Graph g;
	g.nodes = {Const("a", FloatTensor({}, {0.1f})), WithInt(Op("Cast", {"a"}, {"ad"}), "to", 11), Const("three", FloatTensor({}, {3.0f})),
	           WithInt(Op("Cast", {"three"}, {"threed"}), "to", 11), Op("Mul", {"ad", "threed"}, {"p"}), WithInt(Op("Cast", {"p"}, {"pf"}), "to", 1)};
	const auto r = Eval(g);
	REQUIRE(r.values.at("ad").dtype == ShapeDtype::F64);
	REQUIRE(r.values.at("p").dtype == ShapeDtype::F64);
	REQUIRE(r.values.at("p").floats.at(0) == Approx(0.30000000447034836).epsilon(1e-15));
	REQUIRE(r.values.at("p").floats.at(0) != static_cast<double>(static_cast<float>(r.values.at("p").floats.at(0))));
	REQUIRE(r.values.at("pf").dtype == ShapeDtype::F32);
	REQUIRE(r.values.at("pf").floats.at(0) == static_cast<double>(0.30000000447034836f)); // the cast back rounds to float32
	// run-time evaluation: more elements than the host-folding cap (1 << 20) must still be computed, not turned into "unknown"
	ShapeValue big;
	big.shape = {1 << 21};
	big.dtype = ShapeDtype::F64;
	big.known = true;
	big.floats.assign(static_cast<size_t>(1 << 21), 2.0);
	const auto neg = EvaluateNode(Op("Neg", {"x"}, {"y"}), {&big});
	REQUIRE(neg.at(0).known);
	REQUIRE(neg.at(0).floats.size() == static_cast<size_t>(1 << 21));
	REQUIRE(neg.at(0).floats.back() == -2.0);
}

TEST_CASE("shape_eval: integer Pow is exact and truncates negative exponents", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("a", Int64Tensor({4}, {2, 2, -1, 3})), Const("b", Int64Tensor({4}, {10, -1, -3, 0})), Op("Pow", {"a", "b"}, {"p"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("p").dtype == ShapeDtype::I64);
	REQUIRE(r.values.at("p").ints == std::vector<int64_t> {1024, 0, -1, 1});
}

TEST_CASE("shape_eval: Cast to a narrow integer type wraps", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("v", Int64Tensor({3}, {-1, 4294967301LL, 300})), WithInt(Op("Cast", {"v"}, {"u8"}), "to", 2),
	           WithInt(Op("Cast", {"v"}, {"i32"}), "to", 6), WithInt(Op("Cast", {"v"}, {"i8"}), "to", 3)};
	const auto r = Eval(g);
	REQUIRE(r.values.at("u8").ints == std::vector<int64_t> {255, 5, 44});
	REQUIRE(r.values.at("i32").ints == std::vector<int64_t> {-1, 5, 300});
	REQUIRE(r.values.at("i8").ints == std::vector<int64_t> {-1, 5, 44});
}

TEST_CASE("shape_eval: reductions over known integers", "[tabfm][shape_eval]") {
	SECTION("a zero-size reduced axis gives the identity per output element") {
		Graph g;
		g.nodes = {Const("x", Int64Tensor({3, 0}, {})), Const("axes", Int64Tensor({1}, {1})),
		           WithInt(Op("ReduceProd", {"x", "axes"}, {"p"}), "keepdims", 0),
		           WithInt(Op("ReduceSum", {"x", "axes"}, {"s"}), "keepdims", 0)};
		const auto r = Eval(g);
		REQUIRE(r.values.at("p").shape == std::vector<int64_t> {3});
		REQUIRE(r.values.at("p").ints == std::vector<int64_t> {1, 1, 1});
		REQUIRE(r.values.at("s").ints == std::vector<int64_t> {0, 0, 0});
	}
	SECTION("the size of a Shape is a ReduceProd") {
		Graph g;
		g.inputs = {"x"};
		g.nodes = {Op("Shape", {"x"}, {"dims"}), WithInt(Op("ReduceProd", {"dims"}, {"numel"}), "keepdims", 0)};
		ShapeEvalInput in;
		in.shapes["x"] = {2, 3, 7};
		REQUIRE(Eval(g, in).values.at("numel").ints == std::vector<int64_t> {42});
	}
	SECTION("a reduction the pass cannot evaluate is unknown, not wrong") {
		Graph g;
		g.nodes = {Const("x", Int64Tensor({3}, {1, -2, 3})), Op("ReduceL1", {"x"}, {"l1"})};
		const auto r = Eval(g);
		REQUIRE(!r.values.at("l1").known);
	}
}

TEST_CASE("shape_eval: IsInf honours detect_positive and detect_negative", "[tabfm][shape_eval]") {
	const float inf = std::numeric_limits<float>::infinity();
	Graph g;
	g.nodes = {Const("x", FloatTensor({3}, {inf, -inf, 1.0f})), WithInt(Op("IsInf", {"x"}, {"pos"}), "detect_negative", 0),
	           WithInt(Op("IsInf", {"x"}, {"neg"}), "detect_positive", 0), Op("IsInf", {"x"}, {"both"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("pos").ints == std::vector<int64_t> {1, 0, 0});
	REQUIRE(r.values.at("neg").ints == std::vector<int64_t> {0, 1, 0});
	REQUIRE(r.values.at("both").ints == std::vector<int64_t> {1, 1, 0});
}

TEST_CASE("shape_eval: Max and Min propagate NaN", "[tabfm][shape_eval]") {
	const float nan = std::numeric_limits<float>::quiet_NaN();
	Graph g;
	g.nodes = {Const("a", FloatTensor({2}, {nan, 1.0f})), Const("b", FloatTensor({2}, {0.0f, 5.0f})), Op("Max", {"a", "b"}, {"mx"}),
	           Op("Min", {"b", "a"}, {"mn"})};
	const auto r = Eval(g);
	REQUIRE(std::isnan(r.values.at("mx").floats[0]));
	REQUIRE(r.values.at("mx").floats[1] == 5.0);
	REQUIRE(std::isnan(r.values.at("mn").floats[0]));
	REQUIRE(r.values.at("mn").floats[1] == 1.0);
}

TEST_CASE("shape_eval: integer arithmetic wraps instead of overflowing", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("a", Int64Tensor({2}, {INT64_MAX, INT64_MIN})), Const("two", Int64Tensor({}, {2})), Op("Mul", {"a", "two"}, {"m"}),
	           Op("Neg", {"a"}, {"n"}), Op("Abs", {"a"}, {"ab"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("m").ints == std::vector<int64_t> {-2, 0});
	REQUIRE(r.values.at("n").ints == std::vector<int64_t> {-INT64_MAX, INT64_MIN});
	REQUIRE(r.values.at("ab").ints == std::vector<int64_t> {INT64_MAX, INT64_MIN});
}

TEST_CASE("shape_eval: invalid graphs are refused, not read out of bounds", "[tabfm][shape_eval]") {
	auto refused = [](Graph g, const std::string &needle) {
		try {
			Eval(g);
			FAIL("expected an exception mentioning " << needle);
		} catch (const std::runtime_error &e) {
			REQUIRE(std::string(e.what()).find(needle) != std::string::npos);
		}
	};
	Graph g;
	g.nodes = {Const("a", Int64Tensor({2}, {5, 6})), Const("zero", Int64Tensor({2}, {0, 1})), Op("Div", {"a", "zero"}, {"d"})};
	refused(g, "zero");
	g.nodes = {Const("a", Int64Tensor({2}, {5, 6})), Const("zero", Int64Tensor({2}, {0, 1})), Op("Mod", {"a", "zero"}, {"d"})};
	refused(g, "zero");
	g.nodes = {Const("d", Int64Tensor({3}, {1, 2, 3})), Const("s", Int64Tensor({1}, {0})), Const("e", Int64Tensor({1}, {2})),
	           Const("ax", Int64Tensor({1}, {0})), Const("st", Int64Tensor({1}, {0})), Op("Slice", {"d", "s", "e", "ax", "st"}, {"o"})};
	refused(g, "step");
	g.nodes = {Const("a", Int64Tensor({}, {0})), Const("b", Int64Tensor({}, {5})), Const("c", Int64Tensor({}, {0})), Op("Range", {"a", "b", "c"}, {"r"})};
	refused(g, "delta");
	g.nodes = {Const("x", Int64Tensor({2}, {1, 2})), Const("axes", Int64Tensor({2}, {0, 0})), Op("Unsqueeze", {"x", "axes"}, {"y"})};
	refused(g, "axes");
	g.nodes = {Const("x", Int64Tensor({2, 2}, {1, 2, 3, 4})), Const("y", Int64Tensor({3, 2}, {1, 2, 3, 4, 5, 6})),
	           WithInt(Op("Concat", {"x", "y"}, {"c"}), "axis", 1)};
	refused(g, "Concat");
	g.nodes = {Const("x", Int64Tensor({4}, {1, 2, 3, 4})), Const("t", Int64Tensor({2}, {-1, -1})), Op("Reshape", {"x", "t"}, {"r"})};
	refused(g, "-1");
	g.nodes = {Const("x", Int64Tensor({2}, {1, 2})), Op("Cast", {"x"}, {"y"})};
	refused(g, "to");
}

TEST_CASE("shape_eval: huge tensors are not materialised", "[tabfm][shape_eval]") {
	Graph g;
	g.nodes = {Const("shape", Int64Tensor({2}, {1000000, 1000000})), Op("ConstantOfShape", {"shape"}, {"big"})};
	const auto r = Eval(g);
	REQUIRE(r.values.at("big").shape == std::vector<int64_t> {1000000, 1000000});
	REQUIRE(!r.values.at("big").known);
}

TEST_CASE("shape_eval: the shape-feeding ops the real graph relies on", "[tabfm][shape_eval]") {
	Graph g;
	g.inputs = {"x"};
	g.nodes = {Const("a", Int64Tensor({}, {2})), Const("b", Int64Tensor({}, {11})), Const("c", Int64Tensor({}, {3})), Op("Range", {"a", "b", "c"}, {"rng"}),
	           Const("sh", Int64Tensor({2}, {2, 3})), Op("ConstantOfShape", {"sh"}, {"zeros"}),
	           Const("one", Int64Tensor({3}, {1, 2, 3})), Op("Expand", {"one", "sh"}, {"exp"}),
	           Const("p", Int64Tensor({2}, {1, 2})), Const("q", Int64Tensor({1}, {9})), WithInt(Op("Concat", {"p", "q"}, {"cat"}), "axis", -1),
	           Const("neg", Int64Tensor({1}, {-1})), Op("Gather", {"cat", "neg"}, {"last"}),
	           Const("cond", Int64Tensor({3}, {1, 0, 1})), Op("Where", {"cond", "cat", "cat"}, {"w"}),
	           Const("k", Int64Tensor({1}, {2})), WithInt(Op("TopK", {"x", "k"}, {"vals", "idx"}), "axis", 0)};
	ShapeEvalInput in;
	in.shapes["x"] = {5, 4};
	const auto r = Eval(g, in);
	REQUIRE(r.values.at("rng").ints == std::vector<int64_t> {2, 5, 8});
	REQUIRE(r.values.at("zeros").shape == std::vector<int64_t> {2, 3});
	REQUIRE(r.values.at("zeros").floats == std::vector<double>(6, 0.0));
	REQUIRE(r.values.at("exp").shape == std::vector<int64_t> {2, 3});
	REQUIRE(r.values.at("exp").ints == std::vector<int64_t> {1, 2, 3, 1, 2, 3});
	REQUIRE(r.values.at("cat").ints == std::vector<int64_t> {1, 2, 9});
	REQUIRE(r.values.at("last").ints == std::vector<int64_t> {9});
	REQUIRE(r.values.at("w").ints == std::vector<int64_t> {1, 2, 9});
	REQUIRE(r.values.at("vals").shape == std::vector<int64_t> {2, 4});
	REQUIRE(r.values.at("idx").dtype == ShapeDtype::I64);
}
