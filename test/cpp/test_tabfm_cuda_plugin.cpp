//===----------------------------------------------------------------------===//
// Real-hardware equivalence test: CPU (ORT) vs the CUDA backend plugin
// (docs/DYNAMIC_BACKENDS.md phase 3), against the SAME real model weights.
//
// The CUDA counterpart of test_tabfm_migraphx_plugin.cpp, and it exists for
// the same reason: not "does the plugin loader work" (test_tabfm_plugin_
// backend.cpp, fake fixture, always runs) but "does routing inference through
// a dlopen'd .so with its OWN shared ORT-GPU runtime change the answer".
//
// That separate runtime is the whole point of the CUDA plugin. Registering
// ORT's CUDA provider into this binary's ORT was tried first and cannot work:
// the release build links ORT statically, and a static core interposes the
// provider's own symbols, corrupting the heap partway through Run(). This test
// is what proves the replacement is sound end to end.
//
// It can't run in CI (no NVIDIA GPU, and no license to ship real weights —
// CLAUDE.md's license wall), so it targets the developer's own model cache and
// skips itself, loudly, whenever that cache or the plugin library is absent
// rather than failing a machine that can't run it.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ort_engine.hpp"
#include "tabfm_plugin_backend.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;

#ifdef TABFM_CUDA_PLUGIN_PATH

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

TEST_CASE("cuda_plugin: CPU and the CUDA plugin agree on the same real weights", "[tabfm][plugin][gpu]") {
	const string plugin_path = TABFM_CUDA_PLUGIN_PATH;
	const string base = HomeDir() + "/.cache/anofox-tabfm/google__tabfm-1.0.0-pytorch@main/classification";
	// Both backends read the same external-data graph: ORT resolves the weights
	// itself on either side, so nothing but the execution provider differs.
	const string graph = base + "/graph_ext_classification.onnx";

	if (!FileExists(plugin_path) || !FileExists(graph)) {
		SUCCEED("no CUDA plugin + real model cache on this machine, skipping");
		return;
	}

	// A small, arbitrary, deterministic problem: 3 train rows, 2 query rows, 8
	// numeric features, all active. Values only need to be real and repeatable
	// across the two backends — this checks routing and marshalling, not model
	// accuracy (that is what the golden-fixture tests already cover).
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
	auto session = CreateSessionFromPath(graph, {}, config);
	auto cpu_output = Run(*session, input);

	// The CUDA plugin, dlopen'd exactly as tabfm_engine.cpp's TryCudaBackend
	// loads it.
	TabFMPluginCreateParams params {};
	params.graph_path = graph.c_str();
	params.weights_dir = base.c_str();
	params.cache_dir = "";
	params.arch = "";
	params.precision = "fp32";
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

	// Unlike MIGraphX (bf16, ~3 significant digits) this path is fp32 on both
	// sides, so the bound is tight — the design spike measured 7.9e-07 on the
	// committed fixture. Anything materially looser means the GPU is not
	// running the same graph.
	double max_rel_diff = 0.0;
	for (size_t i = 0; i < cpu_output.logits.size(); i++) {
		const double denom = std::max(1e-6, (double)std::fabs(cpu_output.logits[i]));
		max_rel_diff = std::max(max_rel_diff, (double)std::fabs(cpu_output.logits[i] - gpu_output.logits[i]) / denom);
	}
	INFO("max relative |cpu - cuda| logit difference: " << max_rel_diff);
	REQUIRE(max_rel_diff < 1e-3);

	// The decision itself must be identical regardless of tolerance.
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
		INFO("row " << row << ": cpu argmax " << cpu_argmax << " vs cuda argmax " << gpu_argmax);
		REQUIRE(cpu_argmax == gpu_argmax);
	}
}

#endif // TABFM_CUDA_PLUGIN_PATH
