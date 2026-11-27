//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P4 (docs/GGML_PLAN.md): the whole Nori 1.5 graph, ggml against ONNX Runtime.
//
// Needs the REAL graph and weights (never committed: license wall), so it only runs when TABFM_NORI15_DIR points at a directory with
//   graph_ext_nori15_regression.onnx, model.safetensors and the references written by the scratchpad's nori_ref.py
//   (ref_T<rows>_S<train>_H<features>.txt/_x.f32/_y.f32/_logits.f32 from ONNX Runtime). Otherwise the test reports that and passes.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ggml.hpp"
#include "tabfm_safetensors.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace duckdb;
using namespace duckdb::anofox;
namespace fs = std::filesystem;

namespace {

std::vector<char> ReadFile(const std::string &path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	REQUIRE(f.good());
	std::vector<char> bytes(static_cast<size_t>(f.tellg()));
	f.seekg(0);
	f.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
	return bytes;
}

std::vector<float> ReadFloats(const std::string &path) {
	const auto bytes = ReadFile(path);
	std::vector<float> v(bytes.size() / 4);
	std::memcpy(v.data(), bytes.data(), v.size() * 4);
	return v;
}

//! The safetensors weight for an initializer name. The ONNX graph wraps the model as "m.<safetensors key>" (the engine's fallback mapping,
//! tabfm_engine.cpp); only float32 tensors of exactly the initializer's shape are returned.
WeightLookup MakeLookup(const F32Arena &arena) {
	return [&arena](const std::string &name, const std::vector<int64_t> &dims) -> ExternalWeight {
		const auto it = arena.tensors.find(name.rfind("m.", 0) == 0 ? name.substr(2) : name);
		if (it == arena.tensors.end() || it->second.dtype != SafetensorsDtype::F32 || it->second.shape != dims) {
			return ExternalWeight();
		}
		return {reinterpret_cast<const float *>(it->second.data), static_cast<size_t>(it->second.nbytes / sizeof(float))};
	};
}

} // namespace

TEST_CASE("ggml_nori: the whole Nori 1.5 graph matches ONNX Runtime", "[.][ggml_nori]") {
	const char *dir_env = std::getenv("TABFM_NORI15_DIR");
	if (!dir_env) {
		WARN("TABFM_NORI15_DIR is not set: the real-weights Nori 1.5 parity test is skipped");
		return;
	}
	const std::string dir = dir_env;
	const auto graph = ::anofox::onnxread::ParseModelFile(dir + "/graph_ext_nori15_regression.onnx");
	const auto weights_bytes = ReadFile(dir + "/model.safetensors");
	const auto view = ParseSafetensors(reinterpret_cast<const_data_ptr_t>(weights_bytes.data()), weights_bytes.size(), "nori15");
	const auto arena = MaterializeF32Arena(view);
	const WeightLookup lookup = MakeLookup(arena);

	int ran = 0;
	for (const auto &entry : fs::directory_iterator(dir)) {
		const std::string file = entry.path().filename().string();
		if (file.rfind("ref_", 0) != 0 || entry.path().extension() != ".txt") {
			continue;
		}
		const std::string tag = file.substr(4, file.size() - 8);
		std::ifstream meta(entry.path());
		int64_t T, S, H, l0, l1, l2;
		meta >> T >> S >> H >> l0 >> l1 >> l2;
		INFO("config " << tag);
		HostTensors in;
		in["x"] = {{1, T, H}, ReadFloats(dir + "/ref_" + tag + "_x.f32")};
		in["y"] = {{1, S}, ReadFloats(dir + "/ref_" + tag + "_y.f32")};
		const auto want = ReadFloats(dir + "/ref_" + tag + "_logits.f32");

		const char *threads_env = std::getenv("TABFM_NORI15_THREADS"), *repeat_env = std::getenv("TABFM_NORI15_REPEAT");
		const int repeats = repeat_env ? std::max(1, std::atoi(repeat_env)) : 1; // best of N: a warm, fair comparison with ORT's steady state
		HostTensors got;
		double seconds = 1e30, first = 0;
		GgmlExecutorConfig executor_config; // one executor per configuration: the first run compiles, the repeats reuse the graph (the steady state)
		executor_config.threads = threads_env ? std::atoi(threads_env) : 4;
		executor_config.weights = lookup;
		GgmlExecutor executor(graph, executor_config);
		for (int rep = 0; rep < repeats; rep++) {
			const auto t0 = std::chrono::steady_clock::now();
			got = executor.Run(in, GgmlRunOptions());
			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			if (rep == 0) {
				first = elapsed;
			}
			seconds = std::min(seconds, elapsed);
		}
		WARN(tag << ": first run (compile + run) " << first << " s");
		REQUIRE(got.count("logits") == 1);
		const auto &logits = got.at("logits");
		REQUIRE(logits.data.size() == want.size());
		double max_diff = 0, sum_sq = 0;
		for (size_t i = 0; i < want.size(); i++) {
			REQUIRE(std::isfinite(logits.data[i]) == std::isfinite(want[i]));
			const double d = std::fabs(static_cast<double>(logits.data[i]) - want[i]);
			max_diff = std::max(max_diff, d);
			sum_sq += static_cast<double>(want[i]) * want[i];
		}
		const double rms = std::sqrt(sum_sq / want.size());
		WARN(tag << ": ggml " << seconds << " s, max |diff| " << max_diff << ", logits rms " << rms);
		// Measured against ORT: 4e-7 .. 5e-3 over these configurations, and ORT vs ORT (other threads / optimisation) differs by up to 2e-3. The gate
		// is about 4x the worst observed difference: a wrong normalisation, scale or head would be orders of magnitude outside it.
		REQUIRE(max_diff < 0.01 + 0.001 * rms);
		ran++;
	}
	REQUIRE(ran > 0);
}

// Local debugging aid, not a regression test: with TABFM_NORI15_PROBE=<tag> it compares every intermediate tensor ONNX Runtime dumped
// (probe_ref.py, in <dir>/probe_<tag>/) against ggml and prints the first tensors, in graph order, that disagree.
TEST_CASE("ggml_nori: probe the first intermediate tensor that differs from ONNX Runtime", "[.][ggml_nori_probe]") {
	const char *dir_env = std::getenv("TABFM_NORI15_DIR"), *tag_env = std::getenv("TABFM_NORI15_PROBE");
	if (!dir_env || !tag_env) {
		return;
	}
	const std::string dir = dir_env, tag = tag_env, pdir = dir + "/probe_" + tag;
	auto graph = ::anofox::onnxread::ParseModelFile(dir + "/graph_ext_nori15_regression.onnx");
	const auto weights_bytes = ReadFile(dir + "/model.safetensors");
	const auto view = ParseSafetensors(reinterpret_cast<const_data_ptr_t>(weights_bytes.data()), weights_bytes.size(), "nori15");
	const auto arena = MaterializeF32Arena(view);
	const WeightLookup lookup = MakeLookup(arena);
	std::vector<std::string> names;
	{
		std::ifstream order(pdir + "/order.txt");
		std::string line;
		while (std::getline(order, line)) {
			const std::string name = line.substr(0, line.find(' '));
			if (fs::file_size(pdir + "/" + name + ".f32") > 0) { // empty tensors cannot be graph outputs
				names.push_back(name);
			}
		}
	}
	int64_t T, S, H, l0, l1, l2;
	std::ifstream meta(dir + "/ref_" + tag + ".txt");
	meta >> T >> S >> H >> l0 >> l1 >> l2;
	HostTensors in;
	in["x"] = {{1, T, H}, ReadFloats(dir + "/ref_" + tag + "_x.f32")};
	in["y"] = {{1, S}, ReadFloats(dir + "/ref_" + tag + "_y.f32")};
	graph.outputs.clear();
	for (const auto &name : names) {
		graph.outputs.push_back(name);
	}
	const auto got = GgmlRunGraph(graph, in, 4, lookup);
	int reported = 0;
	for (const auto &name : names) {
		const auto want = ReadFloats(pdir + "/" + name + ".f32");
		const auto it = got.find(name);
		if (it == got.end() || it->second.data.size() != want.size()) {
			WARN("probe " << name << ": missing or different size");
			if (++reported >= 5) {
				break;
			}
			continue;
		}
		double worst = 0;
		size_t first_bad = want.size();
		for (size_t i = 0; i < want.size(); i++) {
			const float a = it->second.data[i], b = want[i];
			if (first_bad == want.size() && (std::isnan(a) != std::isnan(b) || std::isinf(a) != std::isinf(b) ||
			                                 (std::isfinite(a) && std::fabs(a - b) > 1e-3 + 1e-3 * std::fabs(b)))) {
				first_bad = i;
			}
			if (std::isnan(a) != std::isnan(b) || (std::isinf(a) != std::isinf(b))) {
				worst = INFINITY;
				break;
			}
			if (std::isfinite(a)) {
				worst = std::max<double>(worst, std::fabs(a - b) / (1e-3 + 1e-3 * std::fabs(b)));
			}
		}
		if (worst > 1.0 && first_bad < want.size()) {
			WARN("probe " << name << ": first mismatch (scaled error " << worst << ") at element " << first_bad << " of " << want.size() << ": ggml "
			                << it->second.data[first_bad] << ", ORT " << want[first_bad] << " (neighbours ggml " << it->second.data[first_bad ? first_bad - 1 : 0]
			                << " / ORT " << want[first_bad ? first_bad - 1 : 0] << ")");
			if (++reported >= 5) {
				break;
			}
		}
	}
	WARN("probe done, " << reported << " mismatching tensors reported (the first in graph order is the culprit)");
}
