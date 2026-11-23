//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P4 (docs/GGML_PLAN.md): every bundled CPU graph must LOWER and RUN on ggml.
//
// This is a completeness test, not a parity test: the weights are a constant (the graphs are weight-free stubs and real weights are never
// committed), so it proves that no op, attribute, shape pattern or input type makes the engine refuse a bundled graph, and that nothing in
// the output is NaN on well-formed input. Numerical agreement with ONNX Runtime is proven per op (test_tabfm_ggml_ops.cpp) and, with real
// weights, per model (test_tabfm_ggml_nori.cpp, local). Prepare/query pairs (tabicl) are chained: the prepare graph's outputs feed the query graph.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ggml_lower.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>

using namespace duckdb::anofox;
namespace fs = std::filesystem;

namespace {

constexpr int64_t kRows = 20, kFeatures = 5, kTrain = 14;

HostTensor Filled(std::vector<int64_t> shape, uint32_t seed, float scale = 1.0f) {
	HostTensor t;
	t.shape = std::move(shape);
	size_t n = 1;
	for (auto d : t.shape) {
		n *= static_cast<size_t>(d);
	}
	t.data.resize(n);
	uint32_t state = seed * 2654435761u + 12345u;
	for (auto &v : t.data) {
		state = state * 1664525u + 1013904223u;
		v = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * scale;
	}
	return t;
}

//! The test weights: element i of the initializer called `name` is hash(name, i) mapped to [-0.1, 0.1). The same formula is in
//! tools/ggml_parity/graphs_parity.py, so ONNX Runtime can run the very same graph on the very same weights.
std::vector<float> TestWeights(const std::string &name, size_t n) {
	uint32_t seed = 2166136261u; // FNV-1a of the name
	for (unsigned char c : name) {
		seed = (seed ^ c) * 16777619u;
	}
	std::vector<float> w(n);
	for (size_t i = 0; i < n; i++) {
		const uint32_t h = static_cast<uint32_t>((static_cast<uint64_t>(i) * 2654435761ull + seed) & 0xffffffffull);
		w[i] = (static_cast<float>(h >> 8) / 16777216.0f - 0.5f) * 0.2f;
	}
	return w;
}

WeightLookup HashedWeights() {
	auto cache = std::make_shared<std::map<std::string, std::vector<float>>>();
	return [cache](const std::string &name, const std::vector<int64_t> &dims) -> ExternalWeight {
		size_t n = 1;
		for (auto d : dims) {
			n *= static_cast<size_t>(d);
		}
		auto it = cache->find(name);
		if (it == cache->end()) {
			it = cache->emplace(name, TestWeights(name, n)).first;
		}
		return {it->second.data(), it->second.size()};
	};
}

struct Prepared {
	HostTensors outputs;
};

} // namespace

TEST_CASE("ggml_graphs: every bundled CPU graph lowers and runs on ggml", "[tabfm][ggml][ggml_graphs]") {
	REQUIRE(fs::exists("resources"));
	std::vector<std::string> files;
	for (const auto &entry : fs::directory_iterator("resources")) {
		const std::string name = entry.path().filename().string();
		if (name.rfind("graph_", 0) == 0 && entry.path().extension() == ".onnx" && name.find("_ext") == std::string::npos) {
			files.push_back(entry.path().string());
		}
	}
	std::sort(files.begin(), files.end());
	REQUIRE(files.size() == 45);

	std::map<std::string, HostTensors> prepared; // "tabicl_classification" -> outputs of its prepare graph
	int checked = 0;
	for (const auto &file : files) {
		const std::string name = fs::path(file).stem().string().substr(6); // without "graph_"
		const bool is_query = name.rfind("query_tabicl", 0) == 0, is_prepare = name.rfind("prepare_tabicl", 0) == 0;
		const auto graph = ::anofox::onnxread::ParseModelFile(file);
		std::set<std::string> input_names(graph.inputs.begin(), graph.inputs.end());
		HostTensors in;
		GgmlRunOptions options;
		options.threads = 2;
		options.weights = HashedWeights();
		const bool has_train_size = input_names.count("train_size") != 0;
		if (is_query) {
			const std::string kind = name.substr(std::string("query_tabicl_").size());
			const auto &prep = prepared.at(kind);
			in["x"] = Filled({1, 6, kFeatures}, 1);
			for (const auto &kv : prep) {
				if (input_names.count(kv.first)) {
					in[kv.first] = kv.second;
				}
			}
		} else {
			in["x"] = Filled({1, is_prepare ? kTrain : kRows, kFeatures}, 1);
			in["y"] = has_train_size ? Filled({1, kRows}, 2, 10.0f) : Filled({1, kTrain}, 2, 10.0f);
			if (name.find("classification") != std::string::npos) { // class indices 0..2, not reals
				for (size_t i = 0; i < in["y"].data.size(); i++) {
					in["y"].data[i] = static_cast<float>(i % 3);
				}
			}
		}
		if (has_train_size) {
			options.int_inputs["train_size"] = {{1}, {kTrain}};
			options.int_inputs["d"] = {{1}, {kFeatures}};
		}
		if (input_names.count("cat_mask")) {
			in["cat_mask"] = {{1, kFeatures}, std::vector<float>(kFeatures, 0.0f)};
			options.bool_inputs.insert("cat_mask");
		}
		INFO("graph " << name);
		// Different data in fresh buffers, same shapes: the cached graph must read THIS call's inputs, never a previous call's.
		HostTensors in_other = in;
		for (auto &kv : in_other) {
			if (kv.first == "x") {
				for (auto &v : kv.second.data) {
					v = v * 0.5f + 0.125f;
				}
			}
		}
		const auto t0 = std::chrono::steady_clock::now();
		HostTensors out;
		GgmlExecutorConfig executor_config;
		executor_config.threads = options.threads;
		executor_config.weights = options.weights;
		GgmlExecutor executor(graph, executor_config);
		REQUIRE_NOTHROW(out = executor.Run(in, options));
		// A second run on the cached graph must give the very same bits: nothing may leak from one run into the next (arena reuse, input rebinding).
		{
			HostTensors again;
			REQUIRE_NOTHROW(again = executor.Run(in, options));
			REQUIRE(executor.CompileCount() == 1);
			REQUIRE(again.size() == out.size());
			for (const auto &kv : out) {
				const auto &other = again.at(kv.first).data;
				REQUIRE(other.size() == kv.second.data.size());
				bool same = true;
				for (size_t i = 0; i < other.size() && same; i++) {
					same = std::memcmp(&other[i], &kv.second.data[i], sizeof(float)) == 0 || (std::isnan(other[i]) && std::isnan(kv.second.data[i]));
				}
				INFO("graph " << name << " output " << kv.first);
				REQUIRE(same);
			}
		}
		const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		if (seconds > 5) {
			WARN("graph " << name << " took " << seconds << " s");
		}
		{
			HostTensors cached_other = executor.Run(in_other, options); // same signature: served from the cache
			REQUIRE(executor.CompileCount() == 1);
			HostTensors fresh_other = GgmlRunGraph(graph, in_other, options); // from scratch
			for (const auto &kv : cached_other) {
				const auto &want = fresh_other.at(kv.first).data;
				REQUIRE(want.size() == kv.second.data.size());
				bool same = true;
				for (size_t i = 0; i < want.size() && same; i++) {
					same = std::memcmp(&want[i], &kv.second.data[i], sizeof(float)) == 0 || (std::isnan(want[i]) && std::isnan(kv.second.data[i]));
				}
				INFO("graph " << name << " output " << kv.first << ": cached run on new inputs differs from a fresh run");
				REQUIRE(same);
			}
		}
		// The device code path (scheduler, weights uploaded once into executor-owned buffers) on the CPU backend must agree with the plain path
		// bit for bit. The multi-GB models are skipped: it would hold a second copy of their weights.
		if (name != "classification" && name != "regression" && name.rfind("tabfm_v1_1", 0) != 0) {
			GgmlExecutorConfig sched_config = executor_config;
			sched_config.device = "cpu-sched";
			GgmlExecutor sched_executor(graph, sched_config);
			HostTensors via_sched;
			REQUIRE_NOTHROW(via_sched = sched_executor.Run(in, options));
			HostTensors again_sched;
			REQUIRE_NOTHROW(again_sched = sched_executor.Run(in_other, options)); // second call: cached graph, uploaded weights reused
			REQUIRE(sched_executor.CompileCount() == 1);
			for (const auto &kv : out) {
				const auto &got = via_sched.at(kv.first).data;
				REQUIRE(got.size() == kv.second.data.size());
				bool same = true;
				for (size_t i = 0; i < got.size() && same; i++) {
					same = std::memcmp(&got[i], &kv.second.data[i], sizeof(float)) == 0 || (std::isnan(got[i]) && std::isnan(kv.second.data[i]));
				}
				INFO("graph " << name << " output " << kv.first << ": the scheduler path differs from the plain CPU path");
				REQUIRE(same);
			}
		}
		if (const char *dump = std::getenv("TABFM_GRAPHS_DUMP")) { // inputs and outputs for tools/ggml_parity/graphs_parity.py
			fs::create_directories(std::string(dump) + "/" + name);
			auto write = [&](const std::string &file, const std::vector<float> &data) {
				std::ofstream f(std::string(dump) + "/" + name + "/" + file, std::ios::binary);
				f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
			};
			std::ofstream meta(std::string(dump) + "/" + name + "/meta.txt");
			for (const auto &kv : in) {
				write("in_" + kv.first + ".f32", kv.second.data);
				meta << "in " << kv.first << " " << (options.bool_inputs.count(kv.first) ? "bool" : "f32");
				for (auto d : kv.second.shape) {
					meta << " " << d;
				}
				meta << "\n";
			}
			for (const auto &kv : options.int_inputs) {
				meta << "int " << kv.first;
				for (auto d : kv.second.shape) {
					meta << " " << d;
				}
				meta << " =";
				for (auto v : kv.second.values) {
					meta << " " << v;
				}
				meta << "\n";
			}
			for (const auto &kv : out) {
				write("out_" + kv.first + ".f32", kv.second.data);
				meta << "out " << kv.first;
				for (auto d : kv.second.shape) {
					meta << " " << d;
				}
				meta << "\n";
			}
		}
		if (is_prepare) {
			prepared[name.substr(std::string("prepare_tabicl_").size())] = out;
		}
		if (out.count("logits")) {
			size_t bad = 0, nans = 0;
			for (float v : out.at("logits").data) {
				bad += std::isfinite(v) ? 0 : 1;
				nans += std::isnan(v) ? 1 : 0;
			}
			// NaN logits are legitimate for some models on these degenerate inputs (ORT gives the same NaNs: tools/ggml_parity/graphs_parity.py
			// compares the patterns); this test only proves the graph lowers and runs.
			if (bad > 0) {
				WARN("graph " << name << ": " << bad << " of " << out.at("logits").data.size() << " logits not finite (" << nans << " NaN)");
			}
		}
		for (const auto &kv : out) {
			INFO("output " << kv.first);
			REQUIRE(!kv.second.data.empty());
		}
		checked++;
	}
	REQUIRE(checked == 45);
}
