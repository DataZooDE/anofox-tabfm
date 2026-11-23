// A standalone driver for the ggml engine: runs one ONNX graph with real weights on a chosen device and compares with an ONNX Runtime reference.
// No DuckDB: it links only src/tabfm_ggml_lower.cpp, src/tabfm_shape_eval.cpp and src/tabfm_onnx_reader.cpp with ggml, so it builds anywhere ggml does
// (a CUDA/HIP/Metal/Vulkan build of ggml, a RunPod machine, a Mac).
//
//   ggml_driver --dir <dir> --tag T200_S150_H17 [--device cpu|cpu-sched|gpu|gpu:N] [--threads 4] [--repeat 5] [--graph graph_ext_nori15_regression.onnx]
//
// <dir> holds the graph (external initializers), model.safetensors (float32), and ref_<tag>.txt / ref_<tag>_{x,y,logits}.f32 written by the scratchpad's
// nori_ref.py (ONNX Runtime's output on the same inputs). Weights are mapped by the engine's rule: the ONNX name "m.<key>" is the safetensors key "<key>".

#include "tabfm_ggml_lower.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace duckdb::anofox;

namespace {

std::vector<char> ReadFile(const std::string &path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f.good()) {
		std::fprintf(stderr, "cannot read %s\n", path.c_str());
		std::exit(2);
	}
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

struct StTensor {
	std::vector<int64_t> shape;
	size_t begin = 0, end = 0;
};

// The safetensors header is a flat JSON object {name: {"dtype": "F32", "shape": [..], "data_offsets": [a, b]}, "__metadata__": {...}}.
std::map<std::string, StTensor> ParseHeader(const std::string &json) {
	std::map<std::string, StTensor> tensors;
	size_t i = 0;
	auto skip_ws = [&] {
		while (i < json.size() && std::isspace(static_cast<unsigned char>(json[i]))) {
			i++;
		}
	};
	auto read_string = [&] {
		std::string out;
		i++; // opening quote
		while (i < json.size() && json[i] != '"') {
			out += json[i++];
		}
		i++;
		return out;
	};
	auto read_ints = [&] {
		std::vector<int64_t> v;
		i++; // [
		while (i < json.size() && json[i] != ']') {
			skip_ws();
			if (json[i] == ',') {
				i++;
				continue;
			}
			size_t used = 0;
			v.push_back(std::stoll(json.substr(i), &used));
			i += used;
		}
		i++;
		return v;
	};
	skip_ws();
	i++; // {
	while (true) {
		skip_ws();
		if (json[i] == '}') {
			break;
		}
		if (json[i] == ',') {
			i++;
			continue;
		}
		const std::string name = read_string();
		skip_ws();
		i++; // :
		skip_ws();
		if (name == "__metadata__") { // a flat string map: skip it
			int depth = 0;
			do {
				depth += json[i] == '{' ? 1 : json[i] == '}' ? -1 : 0;
				i++;
			} while (depth > 0);
			continue;
		}
		StTensor t;
		i++; // {
		while (true) {
			skip_ws();
			if (json[i] == '}') {
				i++;
				break;
			}
			if (json[i] == ',') {
				i++;
				continue;
			}
			const std::string key = read_string();
			skip_ws();
			i++; // :
			skip_ws();
			if (key == "dtype") {
				const std::string dtype = read_string();
				if (dtype != "F32") {
					std::fprintf(stderr, "tensor %s has dtype %s: this driver reads F32 only\n", name.c_str(), dtype.c_str());
					std::exit(2);
				}
			} else if (key == "shape") {
				t.shape = read_ints();
			} else if (key == "data_offsets") {
				const auto off = read_ints();
				t.begin = static_cast<size_t>(off.at(0));
				t.end = static_cast<size_t>(off.at(1));
			}
		}
		tensors[name] = t;
	}
	return tensors;
}

} // namespace

// --goldens <dir> [--only name]: run the per-op ORT goldens (test/fixtures/ggml_ops) on the chosen device. A GPU is not bit-exact with the CPU, so the
// comparison is a tolerance (NaN equals NaN, infinities match in sign); the point is to find the ops a device gets wrong.
#include <dirent.h>

static bool g_strict_fp32 = false;

static HostTensor ParseGoldenTensor(std::istringstream &line) {
	std::string dtype, dims;
	line >> dtype >> dims;
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
		} else if (dtype == "i64") {
			t.data.push_back(static_cast<float>(std::stoll(hex)));
		} else {
			const uint32_t bits = static_cast<uint32_t>(std::stoul(hex, nullptr, 16));
			float f;
			std::memcpy(&f, &bits, 4);
			t.data.push_back(f);
		}
	}
	return t;
}

static int RunGoldens(const std::string &dir, const std::string &device, int threads, const std::string &only) {
	std::vector<std::string> stems;
	if (DIR *d = opendir(dir.c_str())) {
		while (auto *e = readdir(d)) {
			const std::string n = e->d_name;
			if (n.size() > 7 && n.substr(n.size() - 7) == ".golden") {
				stems.push_back(n.substr(0, n.size() - 7));
			}
		}
		closedir(d);
	}
	std::sort(stems.begin(), stems.end());
	int run = 0, failed = 0, refused = 0;
	for (const auto &stem : stems) {
		if (!only.empty() && stem.find(only) == std::string::npos) {
			continue;
		}
		HostTensors inputs, expected;
		std::ifstream f(dir + "/" + stem + ".golden");
		std::string text;
		while (std::getline(f, text)) {
			std::istringstream line(text);
			std::string kind, name;
			line >> kind;
			if (kind == "#") {
				continue;
			}
			line >> name;
			(kind == "in" ? inputs : expected)[name] = ParseGoldenTensor(line);
		}
		run++;
		try {
			const auto graph = ::anofox::onnxread::ParseModelFile(dir + "/" + stem + ".onnx");
			GgmlRunOptions options;
			options.threads = threads;
			options.device = device;
			options.strict_fp32 = g_strict_fp32;
			const auto got = GgmlRunGraph(graph, inputs, options);
			bool ok = true;
			for (const auto &want : expected) {
				const auto &g = got.at(want.first);
				ok = ok && g.shape == want.second.shape && g.data.size() == want.second.data.size();
				for (size_t i = 0; ok && i < want.second.data.size(); i++) {
					const float a = g.data[i], b = want.second.data[i];
					if (std::isnan(b)) {
						ok = std::isnan(a);
					} else if (std::isinf(b) || std::isinf(a)) {
						ok = a == b;
					} else {
						ok = std::fabs(a - b) <= 1e-5 + 1e-4 * std::fabs(b);
					}
				}
			}
			if (!ok) {
				failed++;
				std::printf("MISMATCH %s\n", stem.c_str());
				if (std::getenv("TABFM_DRIVER_VERBOSE")) {
					for (const auto &want : expected) {
						std::printf("   %s got:", want.first.c_str());
						for (size_t i = 0; i < got.at(want.first).data.size() && i < 6; i++) {
							std::printf(" %g", got.at(want.first).data[i]);
						}
						std::printf("   want:");
						for (size_t i = 0; i < want.second.data.size() && i < 6; i++) {
							std::printf(" %g", want.second.data[i]);
						}
						std::printf("\n");
					}
				}
			}
		} catch (const std::exception &e) {
			refused++;
			std::printf("ERROR    %s: %s\n", stem.c_str(), e.what());
		}
	}
	std::printf("device %s: %d golden cases, %d mismatches, %d errors\n", device.c_str(), run, failed, refused);
	return failed + refused == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
	std::string dir, tag, device = "cpu", graph_file = "graph_ext_nori15_regression.onnx", goldens, only, probe;
	int threads = 4, repeat = 3;
	for (int a = 1; a + 1 < argc; a += 2) {
		const std::string k = argv[a], v = argv[a + 1];
		if (k == "--dir") {
			dir = v;
		} else if (k == "--tag") {
			tag = v;
		} else if (k == "--strict-fp32") {
			g_strict_fp32 = v != "0"; // `--strict-fp32 1`: pad small matmuls past the CUDA small-batch kernel (the plugin does this for precision=fp32); also set NVIDIA_TF32_OVERRIDE=0
		} else if (k == "--device") {
			device = v;
		} else if (k == "--threads") {
			threads = std::atoi(v.c_str());
		} else if (k == "--repeat") {
			repeat = std::atoi(v.c_str());
		} else if (k == "--graph") {
			graph_file = v;
		} else if (k == "--goldens") {
			goldens = v;
		} else if (k == "--only") {
			only = v;
		} else if (k == "--probe") {
			probe = v; // run only up to this intermediate tensor and print a checksum (a bisection aid)
		}
	}
	if (!goldens.empty()) {
		std::printf("ggml devices in this build:");
		for (const auto &n : GgmlDeviceNames()) {
			std::printf(" %s", n.c_str());
		}
		std::printf("\n");
		return RunGoldens(goldens, device, threads, only);
	}
	if (dir.empty() || tag.empty()) {
		std::fprintf(stderr, "usage: ggml_driver --dir <dir> --tag T200_S150_H17 [--device cpu|cpu-sched|gpu|gpu:N] [--threads N] [--repeat N] [--graph file]\n");
		return 2;
	}
	std::printf("ggml devices in this build:");
	for (const auto &n : GgmlDeviceNames()) {
		std::printf(" %s", n.c_str());
	}
	std::printf("\n");

	const auto graph = ::anofox::onnxread::ParseModelFile(dir + "/" + graph_file);
	const auto st = ReadFile(dir + "/model.safetensors");
	uint64_t header_len = 0;
	std::memcpy(&header_len, st.data(), 8);
	const auto header = ParseHeader(std::string(st.data() + 8, static_cast<size_t>(header_len)));
	const char *data_begin = st.data() + 8 + header_len;
	GgmlExecutorConfig config;
	config.strict_fp32 = g_strict_fp32;
	config.threads = threads;
	config.device = device;
	config.weights = [&](const std::string &name, const std::vector<int64_t> &dims) -> ExternalWeight {
		const auto it = header.find(name.rfind("m.", 0) == 0 ? name.substr(2) : name);
		if (it == header.end() || it->second.shape != dims) {
			return ExternalWeight();
		}
		return {reinterpret_cast<const float *>(data_begin + it->second.begin), (it->second.end - it->second.begin) / 4};
	};

	std::ifstream meta(dir + "/ref_" + tag + ".txt");
	int64_t T = 0, S = 0, H = 0;
	meta >> T >> S >> H;
	HostTensors in;
	in["x"] = {{1, T, H}, ReadFloats(dir + "/ref_" + tag + "_x.f32")};
	in["y"] = {{1, S}, ReadFloats(dir + "/ref_" + tag + "_y.f32")};
	const auto want = ReadFloats(dir + "/ref_" + tag + "_logits.f32");

	try {
		GgmlExecutor executor(graph, config);
		HostTensors out;
		double best = 1e30, first = 0;
		for (int r = 0; r < repeat; r++) {
			const auto t0 = std::chrono::steady_clock::now();
			GgmlRunOptions run_options;
			if (!probe.empty()) {
				run_options.outputs = {probe};
			}
			out = executor.Run(in, run_options);
			const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
			if (r == 0) {
				first = s;
			}
			best = std::min(best, s);
		}
		if (!probe.empty()) {
			double sum = 0;
			size_t nans = 0, infs = 0;
			for (float v : out.at(probe).data) {
				nans += std::isnan(v);
				infs += std::isinf(v);
				sum += std::isfinite(v) ? v : 0.0;
			}
			std::printf("probe %s ok (%zu values, nan %zu inf %zu, sum %.6g)\n", probe.c_str(), out.at(probe).data.size(), nans, infs, sum);
			return 0;
		}
		const auto &logits = out.at("logits").data;
		if (logits.size() <= 8) {
			std::printf("logits:");
			for (float v : logits) {
				std::printf(" %g", v);
			}
			std::printf("   (ORT:");
			for (float v : want) {
				std::printf(" %g", v);
			}
			std::printf(")\n");
		}
		double max_diff = 0, rms = 0;
		for (size_t i = 0; i < want.size() && i < logits.size(); i++) {
			const double d = static_cast<double>(std::fabs(logits[i] - want[i]));
			max_diff = std::isnan(d) ? INFINITY : std::max(max_diff, d); // a NaN logit is a failure, not "no difference"
			rms += static_cast<double>(want[i]) * want[i];
		}
		rms = std::sqrt(rms / std::max<size_t>(1, want.size()));
		std::printf("device %-9s %s: first run %.3f s (compile + weight upload + run), best of %d %.3f s, max |diff| vs ORT %.3g (logits rms %.3g)%s\n",
		            device.c_str(), tag.c_str(), first, repeat, best, max_diff, rms, max_diff < 0.01 + 0.001 * rms ? "  OK" : "  MISMATCH");
		return max_diff < 0.01 + 0.001 * rms ? 0 : 1;
	} catch (const std::exception &e) {
		std::printf("device %s: FAILED: %s\n", device.c_str(), e.what());
		return 1;
	}
}
