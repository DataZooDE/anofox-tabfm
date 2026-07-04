//===----------------------------------------------------------------------===//
// tabfm_migraphx.cpp — direct AMD MIGraphX inference backend (rocm flavor).
//
// ORT's MIGraphX EP re-inlines initializers into a >2 GB ModelProto and fails
// (docs/GPU_AND_MEMORY_FINDINGS.md), so ROCm drives MIGraphX directly. Per shape
// bucket: parse the migraphx-ready graph (external-data -> model.safetensors,
// degenerate Shape ops rewritten) with the bucket's input shapes, compile for
// the GPU, cache the compiled program to a .mxr on disk (compile is minutes;
// cached loads are fast), and run. Inputs are padded to the bucket; the extra
// rows/features are inert (train_size/d mask them — S01).
//===----------------------------------------------------------------------===//

#include "tabfm_migraphx.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#ifdef TABFM_EP_MIGRAPHX
#include <migraphx/migraphx.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <utility>
#endif

namespace duckdb {
namespace anofox {

#ifdef TABFM_EP_MIGRAPHX

namespace {

class MIGraphXBackend : public TabFMBackend {
public:
	MIGraphXBackend(string graph_path, string weights_dir, string cache_dir, string arch, int device_ordinal)
	    : graph_path(std::move(graph_path)), weights_dir(std::move(weights_dir)), cache_dir(std::move(cache_dir)),
	      arch(std::move(arch)), device_ordinal(device_ordinal) {
		auto slash = this->graph_path.find_last_of("/\\");
		auto dot = this->graph_path.find_last_of('.');
		model_tag = this->graph_path.substr(slash == string::npos ? 0 : slash + 1,
		                                    (dot == string::npos ? this->graph_path.size() : dot) -
		                                        (slash == string::npos ? 0 : slash + 1));
	}

	TabFMRunOutput Run(const TabFMRunInput &in) override {
		auto bucket = MIGraphXShapeBucket(in.t, in.h);
		const int64_t tp = bucket.padded_t;
		const int64_t hp = bucket.padded_h;
		auto &prog = GetProgram(tp, hp);

		// Pad the inputs to the bucket. Real data in [0:t, 0:h]; padded query rows
		// carry the -100 label sentinel; padded features are masked by `d`.
		vector<float> x(static_cast<size_t>(tp * hp), 0.0f);
		for (int64_t i = 0; i < in.t; i++) {
			for (int64_t j = 0; j < in.h; j++) {
				x[static_cast<size_t>(i * hp + j)] = in.x[static_cast<size_t>(i * in.h + j)];
			}
		}
		vector<float> y(static_cast<size_t>(tp), -100.0f);
		for (int64_t i = 0; i < in.t; i++) {
			y[static_cast<size_t>(i)] = in.y[static_cast<size_t>(i)];
		}
		vector<uint8_t> cat(static_cast<size_t>(hp), 0);
		for (int64_t j = 0; j < in.h; j++) {
			cat[static_cast<size_t>(j)] = in.cat_mask[static_cast<size_t>(j)] ? 1 : 0;
		}
		int64_t train_size = in.train_size;
		int64_t d = in.d;

		migraphx::program_parameters params;
		auto pshapes = prog.get_parameter_shapes();
		for (auto *name : pshapes.names()) {
			migraphx::shape s = pshapes[name];
			const string n = name;
			void *ptr = nullptr;
			if (n == "x") {
				ptr = x.data();
			} else if (n == "y") {
				ptr = y.data();
			} else if (n == "cat_mask") {
				ptr = cat.data();
			} else if (n == "train_size") {
				ptr = &train_size;
			} else if (n == "d") {
				ptr = &d;
			} else {
				continue; // e.g. a scratch/output parameter migraphx allocates itself
			}
			params.add(name, migraphx::argument(s, ptr));
		}

		try {
			auto results = prog.eval(params);
			auto out_arg = results[0];
			auto os = out_arg.get_shape();
			auto lens = os.lengths(); // [1, tp, C]
			const int64_t C = lens.empty() ? 0 : static_cast<int64_t>(lens.back());
			const auto *logits = reinterpret_cast<const float *>(out_arg.data());

			// Return only the real rows [0:t] in the engine's [1, T, C] contract.
			TabFMRunOutput result;
			result.shape = {1, in.t, C};
			result.logits.resize(static_cast<size_t>(in.t * C));
			for (int64_t i = 0; i < in.t; i++) {
				for (int64_t c = 0; c < C; c++) {
					result.logits[static_cast<size_t>(i * C + c)] = logits[static_cast<size_t>(i * C + c)];
				}
			}
			return result;
		} catch (const std::exception &e) {
			throw InvalidInputException("anofox_tabfm: MIGraphX inference failed on %s: %s", arch, e.what());
		}
	}

private:
	migraphx::program &GetProgram(int64_t tp, int64_t hp) {
		std::lock_guard<std::mutex> guard(mutex);
		auto key = std::make_pair(tp, hp);
		auto it = programs.find(key);
		if (it != programs.end()) {
			return it->second;
		}
		std::filesystem::create_directories(cache_dir);
		const string mxr = cache_dir + "/" + model_tag + "_" + arch + "_T" + std::to_string(tp) + "_H" +
		                   std::to_string(hp) + ".mxr";
		migraphx::program prog;
		std::ifstream probe(mxr, std::ios::binary);
		if (probe.good()) {
			probe.close();
			prog = migraphx::load(mxr.c_str());
		} else {
			try {
				migraphx::onnx_options opts;
				opts.set_input_parameter_shape("x", {1, static_cast<size_t>(tp), static_cast<size_t>(hp)});
				opts.set_input_parameter_shape("y", {1, static_cast<size_t>(tp)});
				opts.set_input_parameter_shape("cat_mask", {1, static_cast<size_t>(hp)});
				opts.set_input_parameter_shape("train_size", {1});
				opts.set_input_parameter_shape("d", {1});
				opts.set_external_data_path(weights_dir.c_str());
				prog = migraphx::parse_onnx(graph_path.c_str(), opts);
				migraphx::compile_options co;
				co.set_offload_copy(true); // pass/return host buffers; migraphx moves them to/from VRAM
				prog.compile(migraphx::target("gpu"), co);
				migraphx::save(prog, mxr.c_str());
			} catch (const std::exception &e) {
				throw InvalidInputException(
				    "anofox_tabfm: MIGraphX compile failed for %s (bucket T=%lld,H=%lld): %s. Set "
				    "anofox_tabfm_device='cpu' to fall back.",
				    arch, static_cast<long long>(tp), static_cast<long long>(hp), e.what());
			}
		}
		return programs.emplace(key, std::move(prog)).first->second;
	}

	string graph_path;
	string weights_dir;
	string cache_dir;
	string arch;
	string model_tag;
	int device_ordinal;
	std::mutex mutex;
	std::map<std::pair<int64_t, int64_t>, migraphx::program> programs;
};

} // namespace

unique_ptr<TabFMBackend> MakeMIGraphXBackend(const string &graph_path, const string &weights_dir,
                                             const string &cache_dir, const string &arch, int device_ordinal) {
	return make_uniq<MIGraphXBackend>(graph_path, weights_dir, cache_dir, arch, device_ordinal);
}

#else // !TABFM_EP_MIGRAPHX

unique_ptr<TabFMBackend> MakeMIGraphXBackend(const string &, const string &, const string &, const string &, int) {
	throw InvalidInputException("anofox_tabfm: this build has no MIGraphX backend. Install the 'rocm' flavor to run "
	                            "on an AMD GPU, or SET anofox_tabfm_device='cpu'.");
}

#endif

} // namespace anofox
} // namespace duckdb
