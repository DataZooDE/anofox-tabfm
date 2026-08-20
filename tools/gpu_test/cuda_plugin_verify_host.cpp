// Pod verification host for the CUDA backend plugin.
//
// Exercises the plugin exactly the way src/tabfm_plugin_backend.cpp does —
// dlopen, TabFMGetPluginApi, abi_version check, create/run/free_output/destroy
// — and compares its logits against a plain CPU ORT run of the same graph.
//
// This is the first real exercise of src/tabfm_cuda_plugin.cpp: it proves the
// plugin compiles against a real ORT-GPU distribution, that its external-data
// session config resolves the weights, that CUDA actually runs, and that the
// answer matches CPU.
#include "tabfm_plugin_abi.h"

#include "onnxruntime_cxx_api.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

using namespace std;

static const char *kGraph = "/workspace/fixture/graph_fixture.onnx";
static const char *kWeightsDir = "/workspace/fixture";
static const char *kPlugin = "/workspace/libanofox_tabfm_cuda_plugin.so";

int main(int argc, char **argv) {
	// The same small deterministic problem the C++ equivalence tests use.
	const int64_t t = 5, h = 8, train_size = 3, d = 8;
	vector<float> x(static_cast<size_t>(t * h));
	for (int64_t i = 0; i < t; i++) {
		for (int64_t j = 0; j < h; j++) {
			x[static_cast<size_t>(i * h + j)] = 0.1f * float(i + 1) - 0.05f * float(j);
		}
	}
	vector<float> y {0.0f, 1.0f, 0.0f, -100.0f, -100.0f};
	vector<uint8_t> cat_mask(static_cast<size_t>(h), 0);

	// ---- CPU baseline, straight ORT ----
	vector<float> cpu_logits;
	vector<int64_t> cpu_shape;
	{
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "verify_cpu");
		Ort::SessionOptions opts;
		opts.SetIntraOpNumThreads(2);
		opts.AddConfigEntry("session.model_external_initializers_file_folder_path", kWeightsDir);
		Ort::Session session(env, kGraph, opts);

		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		vector<int64_t> xs = {1, t, h}, ys = {1, t}, ms = {1, h}, ss = {1};
		int64_t ts = train_size, dv = d;
		vector<Ort::Value> in;
		in.push_back(Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), xs.data(), xs.size()));
		in.push_back(Ort::Value::CreateTensor<float>(mem, y.data(), y.size(), ys.data(), ys.size()));
		in.push_back(Ort::Value::CreateTensor<bool>(mem, reinterpret_cast<bool *>(cat_mask.data()), cat_mask.size(),
		                                            ms.data(), ms.size()));
		in.push_back(Ort::Value::CreateTensor<int64_t>(mem, &ts, 1, ss.data(), ss.size()));
		in.push_back(Ort::Value::CreateTensor<int64_t>(mem, &dv, 1, ss.data(), ss.size()));
		const char *inames[] = {"x", "y", "cat_mask", "train_size", "d"};
		const char *onames[] = {"logits"};
		auto out = session.Run(Ort::RunOptions {nullptr}, inames, in.data(), in.size(), onames, 1);
		cpu_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
		size_t n = 1;
		for (auto s : cpu_shape) n *= size_t(s);
		const float *p = out[0].GetTensorData<float>();
		cpu_logits.assign(p, p + n);
		printf("[CPU] ok, %zu logits, first: %f %f %f\n", cpu_logits.size(), cpu_logits[0], cpu_logits[1],
		       cpu_logits[2]);
	}

	// ---- The plugin, loaded exactly as tabfm_plugin_backend.cpp loads it ----
	void *lib = dlopen(kPlugin, RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		fprintf(stderr, "FAIL: dlopen(%s): %s\n", kPlugin, dlerror());
		return 1;
	}
	auto entry = reinterpret_cast<TabFMGetPluginApiFn>(dlsym(lib, TABFM_PLUGIN_ENTRY_SYMBOL));
	if (!entry) {
		fprintf(stderr, "FAIL: plugin exports no %s\n", TABFM_PLUGIN_ENTRY_SYMBOL);
		return 1;
	}
	const TabFMPluginApi *api = entry();
	if (!api) {
		fprintf(stderr, "FAIL: entry returned NULL\n");
		return 1;
	}
	printf("[plugin] name=%s abi_version=%d (expected %d)\n", api->name ? api->name() : "?", api->abi_version,
	       TABFM_PLUGIN_ABI_VERSION);
	if (api->abi_version != TABFM_PLUGIN_ABI_VERSION) {
		fprintf(stderr, "FAIL: ABI mismatch\n");
		return 1;
	}

	char err[1024] = {0};
	TabFMPluginCreateParams params {};
	params.graph_path = kGraph;
	params.weights_dir = kWeightsDir;
	params.cache_dir = "";
	params.arch = "";
	// Track A (docs/GPU_HARDENING_PLAN.md P1/P2): the mode under test comes
	// from argv. fp32 = strict (use_tf32=0), tf32 = tensor-core rounding, and
	// bf16/fp16 MUST make create fail with the CUDA-unsupported message — a
	// mode either happens or errors, never a silent fp32 run.
	const char *precision = argc > 1 ? argv[1] : "fp32";
	params.precision = precision;
	params.mxr_source = "";
	params.device_ordinal = 0;
	void *handle = api->create(&params, err, sizeof(err));
	if (std::string(precision) == "bf16" || std::string(precision) == "fp16") {
		if (handle) {
			fprintf(stderr, "FAIL: create accepted '%s' on CUDA — it must be rejected\n", precision);
			api->destroy(handle);
			return 1;
		}
		if (std::string(err).find("not supported on the CUDA backend") == std::string::npos) {
			fprintf(stderr, "FAIL: '%s' was rejected but with the wrong message: %s\n", precision, err);
			return 1;
		}
		printf("REJECTED_AS_EXPECTED: %s -> %s\n", precision, err);
		printf("PLUGIN VERIFY PASSED\n");
		return 0;
	}
	if (!handle) {
		fprintf(stderr, "FAIL: create(%s): %s\n", precision, err);
		return 1;
	}
	printf("[plugin] created (CUDA session up, precision=%s)\n", precision);

	TabFMPluginRunInput input {};
	input.x = x.data();
	input.y = y.data();
	input.cat_mask = cat_mask.data();
	input.t = t;
	input.h = h;
	input.train_size = train_size;
	input.d = d;

	TabFMPluginRunOutput output {};
	if (api->run(handle, &input, &output, err, sizeof(err)) != TABFM_PLUGIN_OK) {
		fprintf(stderr, "FAIL: run: %s\n", err);
		api->destroy(handle);
		return 1;
	}
	printf("[plugin] ran, %lld logits, first: %f %f %f\n", (long long)output.logits_len, output.logits[0],
	       output.logits[1], output.logits[2]);

	// precompile is a documented no-op for CUDA; make sure it says so rather than failing.
	if (api->precompile(handle, t, h, err, sizeof(err)) != TABFM_PLUGIN_OK) {
		fprintf(stderr, "FAIL: precompile reported an error: %s\n", err);
		api->destroy(handle);
		return 1;
	}

	int rc = 0;
	if (size_t(output.logits_len) != cpu_logits.size()) {
		fprintf(stderr, "FAIL: logit count %lld != cpu %zu\n", (long long)output.logits_len, cpu_logits.size());
		rc = 2;
	} else {
		double max_rel = 0.0;
		for (size_t i = 0; i < cpu_logits.size(); i++) {
			const double denom = fmax(1e-6, fabs(double(cpu_logits[i])));
			max_rel = fmax(max_rel, fabs(double(cpu_logits[i]) - double(output.logits[i])) / denom);
		}
		printf("MODE_AGREEMENT: precision=%s max_rel=%g\n", precision, max_rel);
		if (max_rel > 1e-3) {
			fprintf(stderr, "FAIL: logits diverge beyond 1e-3\n");
			rc = 2;
		}
	}

	api->free_output(&output);

	// S1's cost numbers: steady-state latency of this precision mode. 20 runs
	// after one warmup; the mean is what the plan's cost table wants.
	if (rc == 0) {
		TabFMPluginRunOutput timing_output {};
		struct timespec t0, t1;
		double total_ms = 0.0;
		int timed_runs = 20;
		for (int i = 0; i < timed_runs; i++) {
			clock_gettime(CLOCK_MONOTONIC, &t0);
			if (api->run(handle, &input, &timing_output, err, sizeof(err)) != TABFM_PLUGIN_OK) {
				fprintf(stderr, "FAIL: timed run %d: %s\n", i, err);
				rc = 1;
				break;
			}
			clock_gettime(CLOCK_MONOTONIC, &t1);
			total_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
			api->free_output(&timing_output);
		}
		if (rc == 0) {
			printf("MODE_TIMING: precision=%s mean_ms=%.3f over %d runs\n", precision, total_ms / timed_runs,
			       timed_runs);
		}
	}

	api->destroy(handle);
	dlclose(lib);

	if (rc == 0) {
		printf("PLUGIN VERIFY PASSED\n");
	}
	return rc;
}
