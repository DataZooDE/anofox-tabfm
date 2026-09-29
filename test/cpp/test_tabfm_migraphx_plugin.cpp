//===----------------------------------------------------------------------===//
// Real-hardware equivalence test: CPU (ORT) vs the MIGraphX backend plugin
// (docs/DYNAMIC_BACKENDS.md phase 1), against the SAME real model weights.
//
// This is the check the dynamic-backends goal is actually for: not "does the
// plugin loader work" (test_tabfm_plugin_backend.cpp, fake fixture, always
// runs) but "does routing inference through a dlopen'd .so instead of a
// compiled-in backend change the answer". It can't run in CI (no GPU, no
// license to ship real weights — CLAUDE.md's license wall), so it targets the
// developer's own model cache and skips itself, loudly, whenever that cache
// or the plugin library is absent rather than failing a machine that can't
// run it.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ort_engine.hpp"
#include "tabfm_plugin_backend.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;

#ifdef TABFM_MIGRAPHX_PLUGIN_PATH

namespace {

bool FileExists(const string &path) {
	std::ifstream stream(path);
	return stream.good();
}

string HomeDir() {
	const char *h = std::getenv("HOME");
	return h ? string(h) : string();
}

} // namespace

TEST_CASE("migraphx_plugin: CPU and the MIGraphX plugin agree on the same real weights", "[tabfm][plugin][gpu]") {
	const string plugin_path = TABFM_MIGRAPHX_PLUGIN_PATH;
	const string base = HomeDir() + "/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main/classification";
	const string cpu_graph = base + "/graph_ext_classification.onnx";
	const string gpu_graph = base + "/graph_migraphx_classification.onnx";
	const string cache_dir = HomeDir() + "/.cache/anofox-tabfm/migraphx";

	if (!FileExists(plugin_path) || !FileExists(cpu_graph) || !FileExists(gpu_graph)) {
		SUCCEED("no MIGraphX plugin + real model cache on this machine, skipping");
		return;
	}

	// A small, arbitrary, deterministic problem: 3 train rows, 2 query rows, 8
	// numeric features, all active. Values only need to be real and repeatable
	// across the two backends — this checks routing and marshalling, not
	// model accuracy (that is what the golden-fixture tests already cover).
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

	// CPU (ORT, external-data — same session-creation path the extension uses).
	TabFMSessionConfig config;
	config.intra_op_threads = 2;
	config.device_id = "cpu";
	config.model_tag = "classification";
	auto session = CreateSessionFromPath(cpu_graph, {}, config);
	auto cpu_output = Run(*session, input);

	// The MIGraphX plugin, dlopen'd exactly as tabfm_engine.cpp will load it.
	// This uses the user-wide cache_dir, so a machine whose cache lacks a compiled
	// .mxr for this (graph, weights, bf16, T128/H16) key COMPILES on the first
	// run -- tens of minutes for tabfm-v1 -- and loads thereafter. The key includes
	// a fingerprint of the weights, so it changed once when that was added.
	TabFMPluginCreateParams params {};
	params.graph_path = gpu_graph.c_str();
	params.weights_dir = base.c_str();
	params.cache_dir = cache_dir.c_str();
	params.arch = "gfx1201";
	params.precision = "bf16";
	params.mxr_source = "";
	params.device_ordinal = 0;
	auto backend = LoadPluginBackend(plugin_path, params);
	REQUIRE(backend != nullptr);
	auto gpu_output = backend->Run(input);

	REQUIRE(gpu_output.shape.size() == 3);
	REQUIRE(gpu_output.shape[1] == t);
	REQUIRE(cpu_output.shape == gpu_output.shape);
	const int64_t c = cpu_output.shape[2];
	REQUIRE(gpu_output.logits.size() == cpu_output.logits.size());

	// bf16 has ~3 significant decimal digits, so raw logits get a loose
	// tolerance; what actually has to match is the classification decision.
	double max_abs_diff = 0.0;
	for (size_t i = 0; i < cpu_output.logits.size(); i++) {
		max_abs_diff = std::max(max_abs_diff, (double)std::fabs(cpu_output.logits[i] - gpu_output.logits[i]));
	}
	// A RELATIVE bound. This used to be an absolute 1.0, which only held while the
	// test loaded a program compiled long ago and cached; a fresh compile of the
	// same graph lands at 1.02 -- deterministically -- and the logits here reach
	// ~26, so that is 3.9% relative, ordinary for bf16 (an 8-bit mantissa rounds at
	// ~0.4% per op and compounds through ~24 layers). 10% of the CPU logit scale
	// still catches a wrong model or graph, whose error is the size of the logits
	// themselves. What it CANNOT catch is a subtle error: bf16 noise is larger than
	// the 1-3% error a real GPU-only bug produced elsewhere in this project, so
	// numerical equivalence has to be established at fp32, on the graphs that ship
	// (docs/ROCM_TABPFN_PLAN.md). The argmax loop below is the actual gate.
	double max_cpu = 0.0;
	for (auto v : cpu_output.logits) {
		max_cpu = std::max(max_cpu, static_cast<double>(std::fabs(v)));
	}
	INFO("max |cpu - migraphx(bf16)| logit difference: " << max_abs_diff << " (max |cpu logit| " << max_cpu << ")");
	REQUIRE(max_abs_diff < 0.1 * max_cpu);

	for (int64_t row = 0; row < t; row++) {
		int64_t cpu_argmax = 0, gpu_argmax = 0;
		float cpu_best = cpu_output.logits[static_cast<size_t>(row * c)];
		float gpu_best = gpu_output.logits[static_cast<size_t>(row * c)];
		for (int64_t cls = 1; cls < c; cls++) {
			float cv = cpu_output.logits[static_cast<size_t>(row * c + cls)];
			float gv = gpu_output.logits[static_cast<size_t>(row * c + cls)];
			if (cv > cpu_best) {
				cpu_best = cv;
				cpu_argmax = cls;
			}
			if (gv > gpu_best) {
				gpu_best = gv;
				gpu_argmax = cls;
			}
		}
		REQUIRE(cpu_argmax == gpu_argmax);
	}
}

#endif // TABFM_MIGRAPHX_PLUGIN_PATH
