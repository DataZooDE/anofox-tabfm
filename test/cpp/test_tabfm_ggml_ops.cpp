//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P3 (docs/GGML_PLAN.md): every implemented op against ONNX Runtime's output.
//
// test/fixtures/ggml_ops/<case>.onnx + <case>.golden come from tools/ggml_parity/gen_op_goldens.py (ORT is the
// oracle; CI never regenerates them). Each case carries adversarial inputs (NaN, +-inf, zeros, broadcasts both
// ways). A case runs when its op is in GgmlSupportedOps(); the lowering table says which ops come next.
// Written red first against a stub that threw "not implemented".
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ggml_lower.hpp"
#include "tabfm_ggml_ops.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

using namespace duckdb::anofox;
namespace fs = std::filesystem;

namespace {

struct Case {
	std::string name, op;
	HostTensors inputs, outputs;
};

HostTensor ParseTensor(std::istringstream &line) {
	std::string dtype, dims;
	line >> dtype >> dims;
	REQUIRE((dtype == "f32" || dtype == "bool" || dtype == "i64")); // bool and (small) int tensors are compared as floats
	HostTensor t;
	if (dims != "-") {
		std::stringstream ds(dims);
		std::string tok;
		while (std::getline(ds, tok, ',')) {
			t.shape.push_back(std::stoll(tok));
		}
	}
	std::string hex;
	while (line >> hex) {
		if (dtype == "bool") {
			t.data.push_back(hex == "1" ? 1.0f : 0.0f);
			continue;
		}
		if (dtype == "i64") {
			t.data.push_back(static_cast<float>(std::stoll(hex)));
			continue;
		}
		const uint32_t bits = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
		float f;
		std::memcpy(&f, &bits, 4);
		t.data.push_back(f);
	}
	return t;
}

Case ReadCase(const std::string &stem) {
	Case c;
	c.name = fs::path(stem).filename().string();
	std::ifstream f(stem + ".golden");
	REQUIRE(f.good());
	std::string text;
	while (std::getline(f, text)) {
		std::istringstream line(text);
		std::string kind, name;
		line >> kind;
		if (kind == "#") {
			std::string tag;
			line >> tag >> c.op;
			continue;
		}
		line >> name;
		(kind == "in" ? c.inputs : c.outputs)[name] = ParseTensor(line);
	}
	return c;
}

::anofox::onnxread::Graph ReadGraph(const std::string &stem) {
	return ::anofox::onnxread::ParseModelFile(stem + ".onnx");
}

//! Ops whose result is exactly determined (IEEE arithmetic, selection, copies, comparisons, integer-valued): compared bit for
//! bit, so +0 vs -0 and any rounding slop is a failure. Only the transcendental and accumulating ops get a tolerance.
const std::set<std::string> &ExactOps() {
	static const std::set<std::string> ops = {
	    "Identity", "Reshape", "Squeeze", "Unsqueeze", "Neg", "Abs", "Sqrt", "Add", "Sub", "Mul", "Div", "Transpose", "Expand",
	    "Concat", "Slice", "Gather", "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Not", "IsNaN",
	    "IsInf", "Where", "Ceil", "Floor", "Round", "Sign", "Relu", "Reciprocal", "Max", "Min", "Clip", "Cast", "CastLike",
	    "ReduceMax", "ReduceMin", "Shape", "Size", "Constant", "ConstantOfShape", "Range", "Mod"};
	return ops;
}

//! NaN equals NaN (any payload); everything else bit for bit.
bool SameBits(float got, float want) {
	if (std::isnan(want)) {
		return std::isnan(got);
	}
	uint32_t a, b;
	std::memcpy(&a, &got, 4);
	std::memcpy(&b, &want, 4);
	return a == b;
}

//! NaN equals NaN, infinities must match in sign, everything else within a small relative+absolute band.
//! `wide` loosens the band for ops that accumulate over hundreds of elements (a different summation order is not a bug).
bool SameFloat(float got, float want, bool wide = false) {
	if (std::isnan(want)) {
		return std::isnan(got);
	}
	if (std::isinf(want) || std::isinf(got)) {
		return got == want;
	}
	return wide ? std::fabs(got - want) <= 2e-4 + 2e-4 * std::fabs(want) : std::fabs(got - want) <= 1e-6 + 2e-5 * std::fabs(want);
}

} // namespace

TEST_CASE("ggml_ops: every implemented op matches the ORT golden", "[tabfm][ggml][ggml_ops]") {
	REQUIRE(fs::exists("test/fixtures/ggml_ops"));
	std::set<std::string> covered, skipped;
	int ran = 0;
	for (const auto &entry : fs::directory_iterator("test/fixtures/ggml_ops")) {
		if (entry.path().extension() != ".golden") {
			continue;
		}
		const std::string stem = (entry.path().parent_path() / entry.path().stem()).string();
		const Case c = ReadCase(stem);
		const auto graph = ReadGraph(stem);
		// The label in the golden is only a label: the graph itself must contain the op it claims to test.
		bool has_op = false;
		for (const auto &node : graph.nodes) {
			has_op = has_op || node.op_type == c.op;
		}
		INFO("case " << c.name << " claims op " << c.op << " but the graph does not contain it");
		REQUIRE(has_op);
		if (GgmlSupportedOps().count(c.op) == 0) {
			skipped.insert(c.op);
			continue;
		}
		INFO("case " << c.name << " (" << c.op << ")");
		covered.insert(c.op);
		const bool exact = ExactOps().count(c.op) != 0;
		const bool accumulating = c.op == "MatMul" || c.op == "Gemm" || c.op == "ReduceSum" || c.op == "ReduceMean" || c.op == "ReduceProd";
		// The plain CPU path at three thread counts (fewer rows than threads, and more, must not matter), then the scheduler path that a GPU
		// uses (weights in executor-owned buffers, the scheduler splitting custom ops from native ones), on the CPU backend.
		const std::vector<std::pair<int, std::string>> variants = {{1, "cpu"}, {2, "cpu"}, {7, "cpu"}, {3, "cpu-sched"}};
		for (const auto &variant : variants) {
			const int threads = variant.first;
			INFO("threads " << threads << ", device " << variant.second);
			GgmlRunOptions run_options;
			run_options.threads = threads;
			run_options.device = variant.second;
			const HostTensors got = GgmlRunGraph(graph, c.inputs, run_options);
			REQUIRE(got.size() == c.outputs.size());
			for (const auto &want : c.outputs) {
				const auto it = got.find(want.first);
				REQUIRE(it != got.end());
				INFO("output " << want.first);
				REQUIRE(it->second.shape == want.second.shape);
				REQUIRE(it->second.data.size() == want.second.data.size());
				for (size_t i = 0; i < want.second.data.size(); i++) {
					INFO("element " << i << ": got " << it->second.data[i] << ", want " << want.second.data[i]);
					// ORT's Where turns a selected -0 into +0 (checked: both branches -0): ignore the sign of zero there only.
					const bool zeros = c.op == "Where" && it->second.data[i] == 0.0f && want.second.data[i] == 0.0f;
					REQUIRE((zeros || (exact ? SameBits(it->second.data[i], want.second.data[i])
					               : SameFloat(it->second.data[i], want.second.data[i], accumulating))));
				}
			}
		}
		ran++;
	}
	REQUIRE(ran > 0);
	// A skipped case must belong to an op the lowering table knows is still to come, never to a typo or a dead op.
	for (const auto &op : skipped) {
		INFO("golden for an op that is neither supported nor in the lowering table: " << op);
		REQUIRE(GgmlLoweringTable().count(op) == 1);
	}
	// No op may be declared supported without a golden case: the table must not claim more than is tested.
	for (const auto &op : GgmlSupportedOps()) {
		INFO("supported op without a golden: " << op);
		REQUIRE(covered.count(op) == 1);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Review fixes (P3 crew review): robustness of the runner itself, not of one op.
// ---------------------------------------------------------------------------------------------------------------------

#include <pthread.h>

namespace {

::anofox::onnxread::Node MakeNode(const std::string &op, std::vector<std::string> in, std::string out) {
	::anofox::onnxread::Node n;
	n.op_type = op;
	n.name = out;
	n.inputs = std::move(in);
	n.outputs = {std::move(out)};
	return n;
}

struct DeepChainJob {
	int length = 0;
	bool ok = false;
	std::string error;
	float result = 0;
};

void *RunDeepChain(void *arg) {
	auto *job = static_cast<DeepChainJob *>(arg);
	try {
		::anofox::onnxread::Graph g;
		g.inputs = {"x"};
		std::string prev = "x";
		for (int i = 0; i < job->length; i++) {
			const std::string out = i + 1 == job->length ? "y" : "t" + std::to_string(i);
			g.nodes.push_back(MakeNode("Neg", {prev}, out));
			prev = out;
		}
		g.outputs = {"y"};
		HostTensors in;
		in["x"] = {{2}, {1.5f, -2.0f}};
		const auto out = GgmlRunGraph(g, in, 1);
		job->result = out.at("y").data.at(0);
		job->ok = true;
	} catch (const std::exception &e) {
		job->error = e.what();
	}
	return nullptr;
}

} // namespace

TEST_CASE("ggml_runner: a 20000-node chain builds on a 256 KB stack (no recursion per dependency level)", "[tabfm][ggml][ggml_ops]") {
	DeepChainJob job;
	job.length = 20000; // even: Neg^20000 is the identity
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, 256 * 1024);
	pthread_t t;
	REQUIRE(pthread_create(&t, &attr, RunDeepChain, &job) == 0);
	pthread_join(t, nullptr);
	INFO(job.error);
	REQUIRE(job.ok);
	REQUIRE(job.result == 1.5f);
}

TEST_CASE("ggml_runner: input data that does not match its shape is refused, not read out of bounds", "[tabfm][ggml][ggml_ops]") {
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	g.nodes.push_back(MakeNode("Neg", {"x"}, "y"));
	g.outputs = {"y"};
	HostTensors in;
	in["x"] = {{3, 4}, std::vector<float>(5, 1.0f)};
	REQUIRE_THROWS_WITH(GgmlRunGraph(g, in, 1), Catch::Contains("graph input 'x'"));
}

TEST_CASE("ggml_runner: a graph too large for the context budget throws instead of aborting the process", "[tabfm][ggml][ggml_ops]") {
	// 300 three-axis reductions in a row: each lowers to several tensors; the budget must be derived from what the ops create.
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	std::string prev = "x";
	for (int i = 0; i < 300; i++) {
		const std::string out = i + 1 == 300 ? "y" : "t" + std::to_string(i);
		auto n = MakeNode("ReduceMax", {prev}, out);
		::anofox::onnxread::Attribute keep;
		keep.name = "keepdims";
		keep.i = 1;
		n.attributes.push_back(keep);
		g.nodes.push_back(n);
		prev = out;
	}
	g.outputs = {"y"};
	HostTensors in;
	in["x"] = {{2, 3, 4, 5}, std::vector<float>(120, 1.0f)};
	// Must either run or throw a std::exception; an abort() would take this whole binary down.
	try {
		const auto out = GgmlRunGraph(g, in, 1);
		REQUIRE(out.at("y").data.size() == 1);
	} catch (const std::exception &) {
		SUCCEED();
	}
}

TEST_CASE("ggml_runner: Mod on floats refuses fmod=0 (ONNX defines it for integers only)", "[tabfm][ggml][ggml_ops]") {
	::anofox::onnxread::Graph g;
	g.inputs = {"a", "b"};
	auto n = MakeNode("Mod", {"a", "b"}, "y");
	::anofox::onnxread::Attribute fm;
	fm.name = "fmod";
	fm.i = 0;
	n.attributes.push_back(fm);
	g.nodes.push_back(n);
	g.outputs = {"y"};
	HostTensors in;
	in["a"] = {{2}, {5.5f, -5.5f}};
	in["b"] = {{2}, {3.0f, 3.0f}};
	REQUIRE_THROWS_WITH(GgmlRunGraph(g, in, 1), Catch::Contains("fmod"));
}

TEST_CASE("ggml_runner: an unused initializer of unsupported rank does not stop the graph", "[tabfm][ggml][ggml_ops]") {
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	::anofox::onnxread::Tensor big;
	big.name = "unused";
	big.dims = {1, 1, 1, 1, 2};
	big.data_type = 1;
	big.floats = {1.0f, 2.0f};
	g.initializers.push_back(big);
	g.nodes.push_back(MakeNode("Neg", {"x"}, "y"));
	g.outputs = {"y"};
	HostTensors in;
	in["x"] = {{2}, {1.0f, 2.0f}};
	const auto out = GgmlRunGraph(g, in, 1);
	REQUIRE(out.at("y").data == std::vector<float> {-1.0f, -2.0f});
}

TEST_CASE("ggml_runner: an op the interpreter cannot evaluate is refused on a double-precision tensor, by name", "[tabfm][ggml][ggml_ops]") {
	// Cast(x -> DOUBLE) -> ReduceSum -> Cast(-> FLOAT): the ggml engine has no float64 reduction, and the interpreter would be handed an unknown
	// value at inference time (an exception on a ggml worker thread). It must be refused when the graph is lowered.
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	auto cast_up = MakeNode("Cast", {"x"}, "xd");
	::anofox::onnxread::Attribute to;
	to.name = "to";
	to.i = 11;
	cast_up.attributes.push_back(to);
	auto cast_down = MakeNode("Cast", {"r"}, "y");
	to.i = 1;
	cast_down.attributes = {to};
	g.nodes = {cast_up, MakeNode("ReduceSum", {"xd"}, "r"), cast_down};
	g.outputs = {"y"};
	HostTensors in;
	in["x"] = {{2, 3}, std::vector<float>(6, 1.0f)};
	REQUIRE_THROWS_WITH(GgmlRunGraph(g, in, 1), Catch::Contains("ReduceSum") && Catch::Contains("double-precision tensor is not implemented"));
}

TEST_CASE("ggml_runner: external weights are length-checked and may not be missing", "[tabfm][ggml][ggml_ops]") {
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	::anofox::onnxread::Tensor w;
	w.name = "w";
	w.dims = {2};
	w.data_type = 1;
	w.location = ::anofox::onnxread::DataLocation::EXTERNAL;
	g.initializers.push_back(w);
	g.nodes.push_back(MakeNode("Add", {"x", "w"}, "y"));
	g.outputs = {"y"};
	HostTensors in;
	in["x"] = {{2}, {1.0f, 2.0f}};
	const std::vector<float> weights = {10.0f, 20.0f};
	SECTION("correct") {
		const auto out = GgmlRunGraph(g, in, 1, [&](const std::string &, const std::vector<int64_t> &) { return ExternalWeight {weights.data(), 2}; });
		REQUIRE(out.at("y").data == std::vector<float> {11.0f, 22.0f});
	}
	SECTION("too short") {
		REQUIRE_THROWS_WITH(GgmlRunGraph(g, in, 1, [&](const std::string &, const std::vector<int64_t> &) { return ExternalWeight {weights.data(), 1}; }),
		                    Catch::Contains("1 values for a shape of 2"));
	}
	SECTION("unknown weight") {
		REQUIRE_THROWS_WITH(GgmlRunGraph(g, in, 1, [&](const std::string &, const std::vector<int64_t> &) { return ExternalWeight(); }),
		                    Catch::Contains("no weight was supplied"));
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The persistent executor (P5/P7): compile once per input-shape signature, run many times.
// ---------------------------------------------------------------------------------------------------------------------

TEST_CASE("ggml_executor: a repeated signature runs on the cached graph and sees each call's own inputs", "[tabfm][ggml][ggml_ops]") {
	// y = Exp(Neg(x)) + w with an external weight: cheap, and every input value shows up in the output.
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	::anofox::onnxread::Tensor w;
	w.name = "w";
	w.dims = {3};
	w.data_type = 1;
	w.location = ::anofox::onnxread::DataLocation::EXTERNAL;
	g.initializers.push_back(w);
	g.nodes = {MakeNode("Neg", {"x"}, "n"), MakeNode("Exp", {"n"}, "e"), MakeNode("Add", {"e", "w"}, "y")};
	g.outputs = {"y"};
	const std::vector<float> weights = {10.0f, 20.0f, 30.0f};

	GgmlExecutorConfig config;
	config.threads = 4;
	config.weights = [&](const std::string &, const std::vector<int64_t> &) { return ExternalWeight {weights.data(), 3}; };
	GgmlExecutor executor(g, config);

	auto run = [&](std::vector<float> x, int64_t rows) {
		HostTensors in;
		in["x"] = {{rows, 3}, std::move(x)};
		return executor.Run(in, GgmlRunOptions());
	};
	const auto a = run({0.0f, 1.0f, 2.0f}, 1);
	REQUIRE(executor.CompileCount() == 1);
	// same shape, different data, different buffer: no recompile, and the NEW values are used
	const auto b = run({3.0f, 4.0f, 5.0f}, 1);
	REQUIRE(executor.CompileCount() == 1);
	for (int i = 0; i < 3; i++) {
		REQUIRE(a.at("y").data[i] == Approx(std::exp(-static_cast<float>(i)) + weights[i]));
		REQUIRE(b.at("y").data[i] == Approx(std::exp(-static_cast<float>(3 + i)) + weights[i]));
	}
	// a different shape compiles a second graph; going back to the first reuses it
	const auto c = run({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 2);
	REQUIRE(executor.CompileCount() == 2);
	REQUIRE(c.at("y").shape == std::vector<int64_t> {2, 3});
	run({1.0f, 1.0f, 1.0f}, 1);
	REQUIRE(executor.CompileCount() == 2); // the 1-row graph was still cached (two entries by default)
	executor.Trim();
	run({1.0f, 1.0f, 1.0f}, 1);
	REQUIRE(executor.CompileCount() == 3); // trimmed: compiled again
}

namespace {
::anofox::onnxread::Graph ExpNegAddGraph() {
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	::anofox::onnxread::Tensor w;
	w.name = "w";
	w.dims = {3};
	w.data_type = 1;
	w.location = ::anofox::onnxread::DataLocation::EXTERNAL;
	g.initializers.push_back(w);
	g.nodes = {MakeNode("Neg", {"x"}, "n"), MakeNode("Exp", {"n"}, "e"), MakeNode("Add", {"e", "w"}, "y")};
	g.outputs = {"y"};
	return g;
}

int ProcessThreadCount() {
	int count = 0;
	std::ifstream status("/proc/self/status");
	std::string line;
	while (std::getline(status, line)) {
		if (line.rfind("Threads:", 0) == 0) {
			count = std::atoi(line.c_str() + 8);
		}
	}
	return count;
}
} // namespace

TEST_CASE("ggml_executor: a bad device or device number throws, names the problem and leaks no worker threads", "[tabfm][ggml][ggml_ops]") {
	const auto g = ExpNegAddGraph();
	const std::vector<float> weights = {1.0f, 2.0f, 3.0f};
	const int before = ProcessThreadCount();
	for (const char *device : {"bogus", "gpu:abc", "gpu:-1", "gpu: 1", "gpu:1x", "gpu:"}) {
		GgmlExecutorConfig config;
		config.threads = 4; // a pool is created before the device is validated
		config.device = device;
		config.weights = [&](const std::string &, const std::vector<int64_t> &) { return ExternalWeight {weights.data(), 3}; };
		for (int repeat = 0; repeat < 20; repeat++) {
			REQUIRE_THROWS_WITH(GgmlExecutor(g, config), Catch::Matchers::Contains("device"));
		}
	}
	if (before > 0) { // /proc is Linux-only; elsewhere the throws above are the test
		REQUIRE(ProcessThreadCount() <= before + 1);
	}
}

TEST_CASE("ggml_executor: a failed run does not poison the cached graph, and a missing input is named", "[tabfm][ggml][ggml_ops]") {
	const auto g = ExpNegAddGraph();
	const std::vector<float> weights = {10.0f, 20.0f, 30.0f};
	for (const char *device : {"cpu", "cpu-sched"}) {
		GgmlExecutorConfig config;
		config.threads = 2;
		config.device = device;
		config.weights = [&](const std::string &, const std::vector<int64_t> &) { return ExternalWeight {weights.data(), 3}; };
		GgmlExecutor executor(g, config);
		HostTensors good;
		good["x"] = {{1, 3}, {0.0f, 1.0f, 2.0f}};
		// no input at all: the message names it (it used to be an opaque map::at)
		REQUIRE_THROWS_WITH(executor.Run(HostTensors(), GgmlRunOptions()), Catch::Matchers::Contains("'x'"));
		const auto first = executor.Run(good, GgmlRunOptions());
		HostTensors wrong_size = good; // same signature (shape), data of the wrong length: refused before any mutation
		wrong_size["x"].data = {1.0f};
		REQUIRE_THROWS(executor.Run(wrong_size, GgmlRunOptions()));
		const auto again = executor.Run(good, GgmlRunOptions());
		for (int i = 0; i < 3; i++) {
			REQUIRE(again.at("y").data[i] == first.at("y").data[i]);
			REQUIRE(again.at("y").data[i] == Approx(std::exp(-static_cast<float>(i)) + weights[i]));
		}
	}
}

TEST_CASE("ggml_executor: bounded cache evicts the least recently used signature", "[tabfm][ggml][ggml_ops]") {
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	g.nodes = {MakeNode("Neg", {"x"}, "y")};
	g.outputs = {"y"};
	GgmlExecutorConfig config;
	config.cache_entries = 2;
	GgmlExecutor executor(g, config);
	auto run = [&](int64_t n) {
		HostTensors in;
		in["x"] = {{n}, std::vector<float>(static_cast<size_t>(n), 1.0f)};
		executor.Run(in, GgmlRunOptions());
	};
	run(1);
	run(2);
	run(3); // evicts 1
	REQUIRE(executor.CompileCount() == 3);
	run(2); // still cached
	REQUIRE(executor.CompileCount() == 3);
	run(1); // was evicted
	REQUIRE(executor.CompileCount() == 4);
}

TEST_CASE("ggml_executor: views of an input follow the input to each call's buffer", "[tabfm][ggml][ggml_ops]") {
	// Reshape and Slice of a graph input are ggml views holding an absolute pointer made at compile time; a cached graph must refresh them, or the
	// second call computes on the first call's (freed) buffer. y = Neg(Slice(Reshape(x))).
	::anofox::onnxread::Graph g;
	g.inputs = {"x"};
	auto reshape = MakeNode("Reshape", {"x", "shape"}, "r");
	auto slice = MakeNode("Slice", {"r", "starts", "ends", "axes"}, "s");
	auto make_const = [](const std::string &name, std::vector<int64_t> dims, std::vector<int64_t> values) {
		::anofox::onnxread::Tensor t;
		t.name = name;
		t.dims = std::move(dims);
		t.data_type = 7;
		t.int64s = std::move(values);
		return t;
	};
	g.initializers = {make_const("shape", {2}, {3, 4}), make_const("starts", {1}, {1}), make_const("ends", {1}, {3}), make_const("axes", {1}, {1})};
	g.nodes = {reshape, slice, MakeNode("Neg", {"s"}, "y")};
	g.outputs = {"y"};
	GgmlExecutor executor(g, GgmlExecutorConfig());
	for (int round = 0; round < 3; round++) {
		HostTensors in;
		std::vector<float> data(12);
		for (int i = 0; i < 12; i++) {
			data[i] = static_cast<float>(round * 100 + i);
		}
		in["x"] = {{12}, data};
		const auto out = executor.Run(in, GgmlRunOptions());
		REQUIRE(out.at("y").shape == std::vector<int64_t> {3, 2});
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 2; col++) {
				REQUIRE(out.at("y").data[row * 2 + col] == -data[row * 4 + 1 + col]);
			}
		}
	}
	REQUIRE(executor.CompileCount() == 1);
}
