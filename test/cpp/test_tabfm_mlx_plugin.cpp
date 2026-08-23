//===----------------------------------------------------------------------===//
// CPU (ORT) vs the Apple MLX backend plugin, against the SAME real weights.
//
// docs/MLX_PLAN.md; spike verdicts and the tolerance argument in
// docs/MLX_SPIKE_RESULTS.md. Sibling of test_tabfm_migraphx_plugin.cpp, with
// one difference that matters: the MLX plugin carries no ONNX graph. MLX has
// no ONNX importer (S-M1 established none exists anywhere), so the mitra
// forward is TRANSCRIBED from mitra_model_patched.py into
// src/tabfm_mlx_plugin.cpp. That makes this test the thing standing between a
// hand-written second implementation and silent drift from the graph every
// other backend runs — it is load-bearing in a way the MIGraphX equivalent,
// which executes the shipped graph, is not.
//
// Two cases with deliberately different reach:
//
//   1. the refusal contract — needs only the plugin, so it runs on any machine
//      that built it, CI included.
//   2. CPU/MLX equivalence — needs Apple Silicon AND the developer's model
//      cache, so it skips itself loudly rather than failing a machine that
//      cannot run it (CLAUDE.md's license wall: real weights are never
//      committed).
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ort_engine.hpp"
#include "tabfm_plugin_backend.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

#ifdef TABFM_MLX_PLUGIN_PATH

namespace {

bool FileExists(const string &path) {
	std::ifstream stream(path);
	return stream.good();
}

string HomeDir() {
	const char *h = std::getenv("HOME");
	return h ? string(h) : string();
}

//! The registry's cache slug for mitra classification (repo '/'->'__', '@rev').
string MitraCacheDir() {
	return HomeDir() + "/.cache/anofox-tabfm/autogluon__mitra-classifier@main";
}

//! Softmax over the last axis, in double, matching what the extension hands
//! back to SQL after the aggregate normalises logits.
vector<double> Softmax(const float *row, int64_t c) {
	double max_v = row[0];
	for (int64_t i = 1; i < c; i++) {
		max_v = std::max(max_v, static_cast<double>(row[i]));
	}
	vector<double> out(static_cast<size_t>(c));
	double sum = 0.0;
	for (int64_t i = 0; i < c; i++) {
		out[static_cast<size_t>(i)] = std::exp(static_cast<double>(row[i]) - max_v);
		sum += out[static_cast<size_t>(i)];
	}
	for (auto &v : out) {
		v /= sum;
	}
	return out;
}

TabFMPluginCreateParams MlxParams(const string &weights_dir, const char *arch) {
	TabFMPluginCreateParams params {};
	params.graph_path = ""; // deliberately unused: the forward is transcribed
	params.weights_dir = weights_dir.c_str();
	params.cache_dir = "";
	params.arch = arch;
	params.precision = "fp32";
	params.mxr_source = "";
	params.device_ordinal = 0;
	return params;
}

} // namespace

TEST_CASE("mlx_plugin: refuses a model it has no way to run", "[tabfm][plugin][mlx]") {
	// This contract MOVED when the backend gained the graph interpreter, and
	// the move is the point. It used to refuse by ARCHITECTURE, because the
	// only forward it had was mitra's hand-port. It now executes any shipped
	// ONNX graph, so architecture is no longer the question -- what it cannot
	// do is run a model for which it was given neither a graph nor a hand-port.
	//
	// That is still a refusal, and still for the original reason: serving a
	// wrong answer is worse than erroring. Needs no weights, since the check
	// precedes the load.
	const string weights_dir = MitraCacheDir();
	auto params = MlxParams(weights_dir, "tabpfn-v2"); // MlxParams supplies no graph_path
	REQUIRE_THROWS_AS(LoadPluginBackend(TABFM_MLX_PLUGIN_PATH, params), InvalidInputException);

	// And the message has to name the fix, per CLAUDE.md rule 5.
	try {
		LoadPluginBackend(TABFM_MLX_PLUGIN_PATH, params);
		FAIL("expected the mlx plugin to refuse a graphless, non-mitra model");
	} catch (const InvalidInputException &e) {
		const string msg = e.what();
		REQUIRE(msg.find("tabpfn-v2") != string::npos);
		REQUIRE(msg.find("anofox_tabfm_device='cpu'") != string::npos);
	}

	// mitra WITHOUT a graph must still work: its hand-port is the fast path and
	// the interpreter's independent oracle, so losing it would go unnoticed
	// until the two silently agreed on nothing.
	if (FileExists(weights_dir + "/model.safetensors")) {
		auto mitra = MlxParams(weights_dir, "mitra-classification");
		REQUIRE_NOTHROW(LoadPluginBackend(TABFM_MLX_PLUGIN_PATH, mitra));
	}
}

TEST_CASE("mlx_plugin: CPU and MLX agree on the same real weights", "[tabfm][plugin][mlx]") {
	const string weights_dir = MitraCacheDir();
	const string weights = weights_dir + "/model.safetensors";
	// ORT resolves the ext graph's external data relative to the graph file, so
	// the graph has to sit beside the weights it indexes.
	const string cpu_graph = weights_dir + "/graph_ext_mitra_classification.onnx";

	if (!FileExists(weights) || !FileExists(cpu_graph)) {
		// Name the fix rather than gesture at a document: a skip nobody knows
		// how to un-skip is indistinguishable from no test at all, and this is
		// the only case that compares the two implementations.
		WARN("skipping CPU/MLX equivalence: no mitra cache at "
		     << weights_dir
		     << "\n  Set it up with (Apache-2.0 weights, ~303 MB):\n"
		        "    mkdir -p \""
		     << weights_dir
		     << "\"\n"
		        "    curl -L -o \""
		     << weights << "\" \\\n"
		        "      https://huggingface.co/autogluon/mitra-classifier/resolve/main/model.safetensors\n"
		        "    ln -s \"$PWD/resources/graph_ext_mitra_classification.onnx\" \""
		     << cpu_graph
		     << "\"\n"
		        "  The graph must sit beside the weights: ORT resolves the ext graph's\n"
		        "  external data relative to the graph file. See docs/MLX_SPIKE_RESULTS.md.");
		SUCCEED("skipped -- see the warning above for the one-time setup");
		return;
	}

	// A small deterministic problem: 3 context rows, 2 query rows, 8 active
	// features. This checks that routing through a dlopen'd, hand-written
	// forward returns the same answer — not model accuracy, which the golden
	// fixtures cover.
	const int64_t t = 5, h = 8, train_size = 3, d = 8;
	vector<float> x(static_cast<size_t>(t * h));
	for (int64_t i = 0; i < t; i++) {
		for (int64_t j = 0; j < h; j++) {
			x[static_cast<size_t>(i * h + j)] = 0.1f * static_cast<float>(i + 1) - 0.05f * static_cast<float>(j);
		}
	}
	vector<float> y {0.0f, 1.0f, 0.0f, -100.0f, -100.0f};
	vector<uint8_t> cat_mask(static_cast<size_t>(h), 0);

	TabFMRunInput input;
	input.x = x.data();
	input.y = y.data();
	input.cat_mask = reinterpret_cast<const bool *>(cat_mask.data());
	input.t = t;
	input.h = h;
	input.train_size = train_size;
	input.d = d;

	TabFMSessionConfig config;
	config.intra_op_threads = 2;
	config.device_id = "cpu";
	config.model_tag = "mitra-classification";
	auto session = CreateSessionFromPath(cpu_graph, {}, config);
	auto cpu_output = Run(*session, input);

	auto params = MlxParams(weights_dir, "mitra");
	auto backend = LoadPluginBackend(TABFM_MLX_PLUGIN_PATH, params);
	REQUIRE(backend != nullptr);

	// precompile is a documented no-op for MLX (S-M4 measured per-shape compile
	// at 217 ms), but it must still answer successfully so the engine's warm
	// path stays uniform across backends.
	REQUIRE_NOTHROW(backend->Precompile(t, h));
	auto mlx_output = backend->Run(input);

	REQUIRE(mlx_output.shape.size() == 3);
	REQUIRE(cpu_output.shape == mlx_output.shape);
	REQUIRE(mlx_output.logits.size() == cpu_output.logits.size());
	const int64_t c = cpu_output.shape[2];

	// Only the QUERY rows are compared. The graph computes logits for the
	// context rows too, but the engine reads predictions at >= train_size, and
	// holding a backend to agreement on output nobody consumes would be
	// inventing a requirement.
	//
	// Tolerances follow docs/MLX_SPIKE_RESULTS.md rather than equivalence.py's
	// max-relative-on-logits, which that document shows is failed by the torch
	// reference implementation against its own ONNX export. What is asserted is
	// what reaches SQL: the post-softmax probabilities and the predicted class.
	constexpr double kProbTol = 1e-4;
	constexpr double kLogitAbsTol = 1e-3;

	double worst_logit = 0.0, worst_prob = 0.0;
	for (int64_t row = train_size; row < t; row++) {
		const auto offset = static_cast<size_t>(row * c);
		auto cpu_p = Softmax(cpu_output.logits.data() + offset, c);
		auto mlx_p = Softmax(mlx_output.logits.data() + offset, c);

		int64_t cpu_argmax = 0, mlx_argmax = 0;
		for (int64_t cls = 1; cls < c; cls++) {
			if (cpu_output.logits[offset + cls] > cpu_output.logits[offset + cpu_argmax]) {
				cpu_argmax = cls;
			}
			if (mlx_output.logits[offset + cls] > mlx_output.logits[offset + mlx_argmax]) {
				mlx_argmax = cls;
			}
		}
		INFO("row " << row << ": cpu argmax " << cpu_argmax << ", mlx argmax " << mlx_argmax);
		REQUIRE(cpu_argmax == mlx_argmax);

		for (int64_t cls = 0; cls < c; cls++) {
			worst_logit = std::max(worst_logit,
			                       std::fabs(static_cast<double>(cpu_output.logits[offset + cls]) -
			                                 static_cast<double>(mlx_output.logits[offset + cls])));
			worst_prob = std::max(worst_prob, std::fabs(cpu_p[static_cast<size_t>(cls)] -
			                                            mlx_p[static_cast<size_t>(cls)]));
		}
	}
	INFO("worst |cpu - mlx| logit " << worst_logit << ", probability " << worst_prob);
	REQUIRE(worst_logit < kLogitAbsTol);
	REQUIRE(worst_prob < kProbTol);

	// A comparison against undifferentiated logits would pass for an equally
	// broken backend. Assert the reference actually decided something.
	double spread = 0.0;
	for (int64_t row = train_size; row < t; row++) {
		auto p = Softmax(cpu_output.logits.data() + static_cast<size_t>(row * c), c);
		double best = 0.0;
		for (auto v : p) {
			best = std::max(best, v);
		}
		spread = std::max(spread, best - 1.0 / static_cast<double>(c));
	}
	INFO("reference confidence above uniform: " << spread);
	REQUIRE(spread > 1e-3);
}

#endif // TABFM_MLX_PLUGIN_PATH
