/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_cuda_plugin.cpp — the CUDA backend as a loadable plugin.
 *
 * Phase 3 of docs/DYNAMIC_BACKENDS.md, and the same shape phase 1 gave ROCm:
 * a standalone shared library the extension dlopens when the user asks for a
 * CUDA device, not a compile-time flavor of the extension binary.
 *
 * Why a plugin rather than registering ORT's CUDA provider into the
 * extension's own ORT (the approach this replaces — see the CUDA section of
 * docs/DYNAMIC_BACKENDS.md for the full post-mortem):
 *
 *   - The release/community-extension build links ONNX Runtime STATICALLY.
 *     ORT's prebuilt provider libraries define classes (ConstantOfShape and
 *     many peers) that also exist in the core; when the core is static in the
 *     host executable, the executable's copies win symbol resolution and the
 *     provider binds to them. That is an ODR/interposition mismatch and it
 *     corrupts the heap (`free(): invalid pointer` inside Run()).
 *   - The provider libraries must also sit in the directory ORT resolves as
 *     Env::GetRuntimePath() so the core can call Provider_SetHost before their
 *     static initializers dereference it.
 *
 * Both constraints are satisfied by keeping GPU inference out of the extension
 * binary entirely: this .so links its OWN shared libonnxruntime from the ORT
 * GPU distribution, where core and providers ship together in one directory
 * and match by construction. CUDA is then reached through the ordinary
 * (classic) provider API — that distribution has the CUDA EP compiled in, so
 * no runtime provider registration is involved at all.
 *
 * Verified on real hardware (RTX A5000, driver 580.159.04, CUDA 12.8): this
 * configuration runs the fixture graph and agrees with CPU to 7.9e-07.
 *
 * Deliberately independent of duckdb and of the rest of this codebase: this
 * file and tabfm_plugin_abi.h are everything it needs, so the cpu flavor never
 * sees an ORT-GPU header and this .so can be built, shipped and versioned on
 * its own (tabfm_download_runtime('cuda')). Every std::exception at the API
 * boundary is caught and reported through the ABI's err buffer, never thrown
 * across it.
 *===----------------------------------------------------------------------===*/

#include "tabfm_plugin_abi.h"

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

void SetError(char *err, size_t err_len, const std::string &message) {
	if (!err || !err_len) {
		return;
	}
	std::strncpy(err, message.c_str(), err_len - 1);
	err[err_len - 1] = '\0';
}

struct CudaPluginBackend {
	// One env per plugin instance; ORT reference-counts its own globals, and
	// keeping it here ties its lifetime to the handle the caller destroys.
	Ort::Env env {ORT_LOGGING_LEVEL_WARNING, "anofox_tabfm_cuda"};
	Ort::Session session {nullptr};
	//! The graph's declared inputs, in graph order. Models differ: tabfm-v1
	//! takes (x, y, cat_mask, train_size, d); mitra omits cat_mask. Binding a
	//! name the graph does not declare is an ORT error, so Run feeds exactly
	//! this list.
	std::vector<std::string> input_names;
	std::mutex mutex; // Ort::Session::Run is thread-safe, but the engine may
	                  // share one handle across DuckDB threads and we also want
	                  // deterministic error reporting through the err buffer.
	int device_ordinal = 0;
};

const char *PluginName(void) {
	return "cuda";
}

void *PluginCreate(const TabFMPluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path || !params->weights_dir) {
		SetError(err, err_len, "missing required create parameter (graph_path/weights_dir)");
		return nullptr;
	}
	try {
		// unique_ptr until the very end: every early return and every exception
		// below (ORT option/session construction) must not leak the backend.
		auto backend = std::unique_ptr<CudaPluginBackend>(new CudaPluginBackend());
		backend->device_ordinal = params->device_ordinal;

		Ort::SessionOptions options;
		// The graph carries ONNX external-data references to the safetensors
		// file; the weights are not necessarily beside the staged graph, so
		// point ORT at the directory explicitly rather than relying on the
		// model's own location.
		options.AddConfigEntry("session.model_external_initializers_file_folder_path", params->weights_dir);

		// Precision contract (docs/GPU_HARDENING_PLAN.md P1/P2). ORT's CUDA EP
		// runs TF32 tensor-core rounding for fp32 matmuls BY DEFAULT on Ampere+,
		// so "fp32" must explicitly disable it to mean what it says — strict,
		// device-switch-does-not-change-answers fp32. "tf32" opts that rounding
		// back in. bf16/fp16 are MIGraphX quantize modes with no CUDA
		// counterpart here (that needs a real graph conversion), and a requested
		// mode either happens or errors — never a silent fp32 run.
		const std::string precision = params->precision ? params->precision : "fp32";
		if (precision != "fp32" && precision != "tf32") {
			SetError(err, err_len, "precision '" + precision +
			                           "' is not supported on the CUDA backend (fp32 or tf32); bf16/fp16 are "
			                           "MIGraphX modes on ROCm");
			return nullptr;
		}
		OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
		Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
		const std::string device_ordinal = std::to_string(params->device_ordinal);
		const char *option_keys[] = {"device_id", "use_tf32"};
		const char *option_values[] = {device_ordinal.c_str(), precision == "tf32" ? "1" : "0"};
		auto update_status = Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, option_keys, option_values, 2);
		if (update_status) {
			Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
			Ort::ThrowOnError(update_status);
		}
		try {
			options.AppendExecutionProvider_CUDA_V2(*cuda_options);
		} catch (...) {
			Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
			throw;
		}
		Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);

		backend->session = Ort::Session(backend->env, params->graph_path, options);
		Ort::AllocatorWithDefaultOptions alloc;
		for (size_t i = 0; i < backend->session.GetInputCount(); i++) {
			backend->input_names.emplace_back(backend->session.GetInputNameAllocated(i, alloc).get());
		}
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("could not initialise the CUDA backend on device ") +
		                           std::to_string(params->device_ordinal) + ": " + e.what());
		return nullptr;
	}
}

TabFMPluginStatus PluginRun(void *handle, const TabFMPluginRunInput *input, TabFMPluginRunOutput *output, char *err,
                            size_t err_len) {
	auto *backend = static_cast<CudaPluginBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return TABFM_PLUGIN_ERROR;
	}
	try {
		std::lock_guard<std::mutex> guard(backend->mutex);

		// No shape bucketing here, unlike MIGraphX: ORT handles the graph's
		// dynamic dimensions directly, so the real extents go straight in.
		auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		const std::vector<int64_t> x_shape = {1, input->t, input->h};
		const std::vector<int64_t> y_shape = {1, input->t};
		const std::vector<int64_t> mask_shape = {1, input->h};
		const std::vector<int64_t> scalar_shape = {1};
		int64_t train_size = input->train_size;
		int64_t d = input->d;

		// const_cast: ORT's CreateTensor borrows a mutable pointer but does not
		// write through it for inputs; the ABI's buffers stay owned by the caller.
		// Bind exactly the graph's declared inputs, in graph order — mitra's
		// contract omits cat_mask, tabfm-v1's includes it.
		std::vector<Ort::Value> inputs;
		std::vector<const char *> names;
		for (const auto &name : backend->input_names) {
			if (name == "x") {
				inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, const_cast<float *>(input->x),
				                                                 static_cast<size_t>(input->t * input->h),
				                                                 x_shape.data(), x_shape.size()));
			} else if (name == "y") {
				inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, const_cast<float *>(input->y),
				                                                 static_cast<size_t>(input->t), y_shape.data(),
				                                                 y_shape.size()));
			} else if (name == "cat_mask") {
				inputs.push_back(Ort::Value::CreateTensor<bool>(
				    mem_info, reinterpret_cast<bool *>(const_cast<uint8_t *>(input->cat_mask)),
				    static_cast<size_t>(input->h), mask_shape.data(), mask_shape.size()));
			} else if (name == "train_size") {
				inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, &train_size, 1, scalar_shape.data(),
				                                                   scalar_shape.size()));
			} else if (name == "d") {
				inputs.push_back(
				    Ort::Value::CreateTensor<int64_t>(mem_info, &d, 1, scalar_shape.data(), scalar_shape.size()));
			} else {
				SetError(err, err_len, "graph declares input '" + name + "' which this plugin does not provide");
				return TABFM_PLUGIN_ERROR;
			}
			names.push_back(name.c_str());
		}

		const char *output_names[] = {"logits"};
		auto outputs = backend->session.Run(Ort::RunOptions {nullptr}, names.data(), inputs.data(), inputs.size(),
		                                    output_names, 1);

		auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
		int64_t count = 1;
		for (auto dim : shape) {
			count *= dim;
		}
		output->logits = static_cast<float *>(std::malloc(sizeof(float) * static_cast<size_t>(count)));
		output->shape = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * shape.size()));
		if (!output->logits || !output->shape) {
			std::free(output->logits);
			std::free(output->shape);
			output->logits = nullptr;
			output->shape = nullptr;
			SetError(err, err_len, "allocation failed");
			return TABFM_PLUGIN_ERROR;
		}
		std::memcpy(output->logits, outputs[0].GetTensorData<float>(), sizeof(float) * static_cast<size_t>(count));
		std::memcpy(output->shape, shape.data(), sizeof(int64_t) * shape.size());
		output->logits_len = count;
		output->shape_len = static_cast<int64_t>(shape.size());
		return TABFM_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("CUDA inference failed on device ") +
		                           std::to_string(backend->device_ordinal) + ": " + e.what());
		return TABFM_PLUGIN_ERROR;
	}
}

// ORT compiles nothing ahead of time for CUDA — the session is built in
// create() and handles every shape — so warming a bucket is a no-op. Reported
// as OK rather than an error: the caller's contract is "this shape is ready
// afterwards", which is already true.
TabFMPluginStatus PluginPrecompile(void *handle, int64_t /*rows*/, int64_t /*features*/, char *err, size_t err_len) {
	if (!handle) {
		SetError(err, err_len, "null handle");
		return TABFM_PLUGIN_ERROR;
	}
	return TABFM_PLUGIN_OK;
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
	delete static_cast<CudaPluginBackend *>(handle);
}

const TabFMPluginApi kApi = {
    TABFM_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" TABFM_PLUGIN_EXPORT const TabFMPluginApi *TabFMGetPluginApi(void) {
	return &kApi;
}
