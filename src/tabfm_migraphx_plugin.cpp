/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_migraphx_plugin.cpp — the MIGraphX backend as a loadable plugin.
 *
 * Phase 1 of docs/DYNAMIC_BACKENDS.md. This is the same inference logic as
 * the compile-time backend in tabfm_migraphx.cpp (ORT's MIGraphX EP re-inlines
 * initializers into a >2 GB ModelProto and fails, so ROCm drives MIGraphX
 * directly), rebuilt as a standalone shared library against
 * tabfm_plugin_abi.h instead of linked into the extension binary.
 *
 * Deliberately independent of duckdb and of the rest of this codebase: this
 * file, tabfm_plugin_abi.h and tabfm_shape_bucket.hpp are everything it needs,
 * so the cpu/cuda flavors never see a MIGraphX header and this .so can be
 * built, shipped and versioned on its own (tabfm_download_runtime('rocm')).
 * Every std::exception at the API boundary is caught and reported through the
 * ABI's err buffer, never thrown across it.
 *===----------------------------------------------------------------------===*/

#include "tabfm_plugin_abi.h"
#include "tabfm_shape_bucket.hpp"

#include <migraphx/migraphx.hpp>

#include <dlfcn.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

using duckdb::anofox::PadToShapeBucket;

void SetError(char *err, size_t err_len, const std::string &message) {
	if (!err || !err_len) {
		return;
	}
	std::strncpy(err, message.c_str(), err_len - 1);
	err[err_len - 1] = '\0';
}

// migraphx::target("gpu") dlopen's libmigraphx_gpu.so by bare name from inside
// libmigraphx_c — a manual dlopen call, not a DT_NEEDED entry, so it does NOT
// inherit libmigraphx_c's own RUNPATH and fails on any install where that
// library isn't already on LD_LIBRARY_PATH or the loader cache. Every ROCm
// package layout puts it at <libmigraphx_c's dir>/migraphx/lib/, so resolve
// libmigraphx_c's OWN runtime location (dladdr on one of its exports) and
// preload the GPU library from the path relative to that — this works
// wherever the plugin's dependency was actually found, not just at the
// TABFM_MIGRAPHX_DIR this plugin happened to be built against. Best-effort:
// if this doesn't find it, the later migraphx::target("gpu") call still
// produces its own (less specific) error.
void PreloadMigraphxGpuLibrary() {
	static bool attempted = false;
	if (attempted) {
		return;
	}
	attempted = true;
	Dl_info info {};
	if (!dladdr(reinterpret_cast<void *>(&migraphx_target_create), &info) || !info.dli_fname) {
		return;
	}
	std::string c_lib_path = info.dli_fname;
	auto slash = c_lib_path.find_last_of('/');
	if (slash == std::string::npos) {
		return;
	}
	const std::string gpu_lib = c_lib_path.substr(0, slash) + "/migraphx/lib/libmigraphx_gpu.so";
	dlopen(gpu_lib.c_str(), RTLD_NOW | RTLD_GLOBAL); // ignore failure — see comment above
}

// bf16 (upper 16 bits of an fp32) -> fp32.
inline float Bf16ToFloat(uint16_t b) {
	uint32_t bits = static_cast<uint32_t>(b) << 16;
	float out;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

// IEEE half (fp16) -> fp32.
inline float HalfToFloat(uint16_t h) {
	uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1Fu;
	uint32_t mant = h & 0x3FFu;
	uint32_t bits;
	if (exp == 0) {
		if (mant == 0) {
			bits = sign; // +/- 0
		} else {
			exp = 127 - 15 + 1; // subnormal -> normalize
			while ((mant & 0x400u) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3FFu;
			bits = sign | (exp << 23) | (mant << 13);
		}
	} else if (exp == 0x1Fu) {
		bits = sign | 0x7F800000u | (mant << 13); // inf / nan
	} else {
		bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
	}
	float out;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

struct MigraphxPluginBackend {
	std::string graph_path;
	std::string weights_dir;
	std::string cache_dir;
	std::string arch;
	std::string precision;
	std::string mxr_source;
	std::string model_tag;
	int device_ordinal = 0;
	std::mutex mutex;
	std::map<std::pair<int64_t, int64_t>, migraphx::program> programs;

	migraphx::program &GetProgram(int64_t tp, int64_t hp) {
		std::lock_guard<std::mutex> guard(mutex);
		auto key = std::make_pair(tp, hp);
		auto it = programs.find(key);
		if (it != programs.end()) {
			return it->second;
		}
		std::filesystem::create_directories(cache_dir);
		const std::string basename = model_tag + "_" + arch + "_" + precision + "_T" + std::to_string(tp) + "_H" +
		                             std::to_string(hp) + ".mxr";
		const std::string mxr = cache_dir + "/" + basename;
		// Offline/CI/shared precompiled artifact: if the bucket isn't cached
		// locally but a matching .mxr exists in the configured source, stage it
		// in (atomic copy) instead of the on-device compile. A bad staged file
		// is caught by the corrupt-recovery below and recompiled.
		if (!mxr_source.empty() && !std::filesystem::exists(mxr)) {
			const std::string src = mxr_source + "/" + basename;
			std::error_code ec;
			if (std::filesystem::exists(src, ec)) {
				const std::string tmp = mxr + ".staging";
				std::filesystem::remove(tmp, ec);
				std::filesystem::copy_file(src, tmp, std::filesystem::copy_options::overwrite_existing, ec);
				if (!ec) {
					std::filesystem::rename(tmp, mxr, ec);
				}
				if (ec) {
					std::filesystem::remove(tmp, ec);
				}
			}
		}
		migraphx::program prog;
		bool loaded = false;
		if (std::filesystem::exists(mxr)) {
			try {
				prog = migraphx::load(mxr.c_str());
				loaded = true;
			} catch (const std::exception &) {
				// A truncated/corrupt .mxr (e.g. an interrupted compile) — discard
				// and recompile rather than failing every predict.
				std::error_code ec;
				std::filesystem::remove(mxr, ec);
			}
		}
		if (!loaded) {
			migraphx::onnx_options opts;
			opts.set_input_parameter_shape("x", {1, static_cast<size_t>(tp), static_cast<size_t>(hp)});
			opts.set_input_parameter_shape("y", {1, static_cast<size_t>(tp)});
			opts.set_input_parameter_shape("cat_mask", {1, static_cast<size_t>(hp)});
			opts.set_input_parameter_shape("train_size", {1});
			opts.set_input_parameter_shape("d", {1});
			opts.set_external_data_path(weights_dir.c_str());
			prog = migraphx::parse_onnx(graph_path.c_str(), opts);
			// Reduce precision before compiling: bf16/fp16 halve VRAM/.mxr and,
			// where the arch has native reduced-precision GEMMs, run faster.
			if (precision == "bf16") {
				migraphx::quantize_bf16(prog);
			} else if (precision == "fp16") {
				migraphx::quantize_fp16(prog);
			}
			migraphx::compile_options co;
			co.set_offload_copy(true); // pass/return host buffers; migraphx moves them to/from VRAM
			prog.compile(migraphx::target("gpu"), co);
			// Save atomically (temp + rename) so an interrupted save never leaves
			// a corrupt .mxr behind.
			const std::string tmp = mxr + ".tmp";
			migraphx::save(prog, tmp.c_str());
			std::error_code ec;
			std::filesystem::rename(tmp, mxr, ec);
		}
		return programs.emplace(key, std::move(prog)).first->second;
	}
};

const char *PluginName(void) {
	return "migraphx";
}

void *PluginCreate(const TabFMPluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path || !params->weights_dir || !params->cache_dir || !params->arch) {
		SetError(err, err_len, "missing required create parameter (graph_path/weights_dir/cache_dir/arch)");
		return nullptr;
	}
	PreloadMigraphxGpuLibrary();
	try {
		// unique_ptr until the very end: the rejection branch and any exception
		// below must not leak the backend (this leaked before, on any throw).
		auto backend = std::unique_ptr<MigraphxPluginBackend>(new MigraphxPluginBackend());
		backend->graph_path = params->graph_path;
		backend->weights_dir = params->weights_dir;
		backend->cache_dir = params->cache_dir;
		backend->arch = params->arch;
		backend->precision = params->precision ? params->precision : "fp32";
		// 'tf32' is CUDA's mode and anything unknown must not silently run as
		// fp32 — a requested mode either happens or errors (the same
		// no-silent-substitution rule the device setting follows).
		if (backend->precision != "fp32" && backend->precision != "bf16" && backend->precision != "fp16") {
			SetError(err, err_len,
			         "precision '" + backend->precision + "' is not supported on the MIGraphX backend (fp32, bf16 "
			         "or fp16); 'tf32' is a CUDA-only mode");
			return nullptr;
		}
		backend->mxr_source = params->mxr_source ? params->mxr_source : "";
		backend->device_ordinal = params->device_ordinal;

		auto slash = backend->graph_path.find_last_of("/\\");
		auto dot = backend->graph_path.find_last_of('.');
		backend->model_tag = backend->graph_path.substr(
		    slash == std::string::npos ? 0 : slash + 1,
		    (dot == std::string::npos ? backend->graph_path.size() : dot) - (slash == std::string::npos ? 0 : slash + 1));
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("could not initialise the migraphx backend: ") + e.what());
		return nullptr;
	}
}

TabFMPluginStatus PluginRun(void *handle, const TabFMPluginRunInput *input, TabFMPluginRunOutput *output, char *err,
                            size_t err_len) {
	auto *backend = static_cast<MigraphxPluginBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return TABFM_PLUGIN_ERROR;
	}
	try {
		duckdb::anofox::ShapeBucket bucket = PadToShapeBucket(input->t, input->h);
		const int64_t tp = bucket.padded_t;
		const int64_t hp = bucket.padded_h;
		auto &prog = backend->GetProgram(tp, hp);

		// Pad the inputs to the bucket. Real data in [0:t, 0:h]; padded query
		// rows carry the -100 label sentinel; padded features are masked by `d`.
		std::vector<float> x(static_cast<size_t>(tp * hp), 0.0f);
		for (int64_t i = 0; i < input->t; i++) {
			for (int64_t j = 0; j < input->h; j++) {
				x[static_cast<size_t>(i * hp + j)] = input->x[static_cast<size_t>(i * input->h + j)];
			}
		}
		std::vector<float> y(static_cast<size_t>(tp), -100.0f);
		for (int64_t i = 0; i < input->t; i++) {
			y[static_cast<size_t>(i)] = input->y[static_cast<size_t>(i)];
		}
		std::vector<uint8_t> cat(static_cast<size_t>(hp), 0);
		for (int64_t j = 0; j < input->h; j++) {
			cat[static_cast<size_t>(j)] = input->cat_mask[static_cast<size_t>(j)];
		}
		int64_t train_size = input->train_size;
		int64_t d = input->d;

		migraphx::program_parameters mparams;
		auto pshapes = prog.get_parameter_shapes();
		for (auto *name : pshapes.names()) {
			migraphx::shape s = pshapes[name];
			const std::string n = name;
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
			mparams.add(name, migraphx::argument(s, ptr));
		}

		auto results = prog.eval(mparams);
		auto out_arg = results[0];
		auto os = out_arg.get_shape();
		auto lens = os.lengths(); // [1, tp, C]
		const int64_t C = lens.empty() ? 0 : static_cast<int64_t>(lens.back());
		const auto out_type = os.type();
		const char *raw = out_arg.data();

		// Read logit[i*C+c] as fp32 regardless of the compiled output dtype
		// (bf16/fp16 quantization may leave the output reduced-precision).
		auto read = [&](int64_t idx) -> float {
			if (out_type == migraphx_shape_bf16_type) {
				return Bf16ToFloat(reinterpret_cast<const uint16_t *>(raw)[idx]);
			}
			if (out_type == migraphx_shape_half_type) {
				return HalfToFloat(reinterpret_cast<const uint16_t *>(raw)[idx]);
			}
			return reinterpret_cast<const float *>(raw)[idx];
		};

		// Return only the real rows [0:t] in the ABI's [1, T, C] contract.
		const int64_t count = input->t * C;
		output->logits = static_cast<float *>(std::malloc(sizeof(float) * (size_t)count));
		output->shape = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * 3));
		if (!output->logits || !output->shape) {
			SetError(err, err_len, "allocation failed");
			return TABFM_PLUGIN_ERROR;
		}
		output->logits_len = count;
		output->shape_len = 3;
		output->shape[0] = 1;
		output->shape[1] = input->t;
		output->shape[2] = C;
		for (int64_t i = 0; i < input->t; i++) {
			for (int64_t c = 0; c < C; c++) {
				output->logits[i * C + c] = read(i * C + c);
			}
		}
		return TABFM_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("MIGraphX inference failed on ") + backend->arch + ": " + e.what());
		return TABFM_PLUGIN_ERROR;
	}
}

TabFMPluginStatus PluginPrecompile(void *handle, int64_t rows, int64_t features, char *err, size_t err_len) {
	auto *backend = static_cast<MigraphxPluginBackend *>(handle);
	if (!backend) {
		SetError(err, err_len, "null handle");
		return TABFM_PLUGIN_ERROR;
	}
	try {
		auto bucket = PadToShapeBucket(rows, features);
		backend->GetProgram(bucket.padded_t, bucket.padded_h); // compile + cache the .mxr
		return TABFM_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("MIGraphX compile failed for ") + backend->arch + " (bucket T=" +
		                            std::to_string(rows) + ",H=" + std::to_string(features) + "): " + e.what());
		return TABFM_PLUGIN_ERROR;
	}
}

void PluginFreeOutput(TabFMPluginRunOutput *output) {
	if (!output) {
		return;
	}
	std::free(output->logits);
	std::free(output->shape);
	output->logits = nullptr;
	output->shape = nullptr;
	output->logits_len = 0;
	output->shape_len = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<MigraphxPluginBackend *>(handle);
}

const TabFMPluginApi kApi = {
    TABFM_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" TABFM_PLUGIN_EXPORT const TabFMPluginApi *TabFMGetPluginApi(void) {
	return &kApi;
}
