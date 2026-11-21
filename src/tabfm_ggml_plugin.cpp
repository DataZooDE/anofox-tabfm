/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_ggml_plugin.cpp — the ggml GPU backend as a loadable plugin (docs/GGML_PLAN.md, P8).
 *
 * The same engine the CPU build runs (src/tabfm_ggml_lower.cpp: ONNX graph -> ggml graph), pointed at a ggml GPU
 * device (CUDA or HIP, chosen when this library is built; see tools/ggml_plugin/CMakeLists.txt). It speaks
 * tabfm_plugin_abi.h, so the extension loads it exactly like the ORT-CUDA and MIGraphX plugins: nothing GPU-specific
 * is linked into the extension.
 *
 * What is different from the ORT plugins:
 *   - one engine for every device, so a device switch changes where the same graph runs, not which interpreter runs it;
 *   - the weights are read straight out of the memory-mapped safetensors file by the byte offsets the staged graph
 *     carries for its external initializers, then uploaded once into device memory;
 *   - ops the GPU backend cannot run (the custom kernels: Where, TopK, the float64 islands...) run on the CPU inside
 *     the same graph; ggml's scheduler moves the tensors across.
 *
 * Precision: a CUDA GPU multiplies fp32 matrices in TF32 by default on Ampere and later (10-bit mantissa; measured
 * 2e-3 relative error per matmul, up to 1.5 on the logits of a 12k-node graph). "fp32" (the default) turns that off
 * with NVIDIA_TF32_OVERRIDE=0 before the CUDA runtime starts, "tf32" leaves the hardware default. KNOWN LIMIT: that switch only
 * reaches cuBLAS; ggml's own small-batch f32 kernel (mul_mat_f, used when a matmul has <= 16 columns on Ampere+) issues TF32 tensor-core
 * instructions directly and has no switch, so "fp32" is strict for the large matmuls the Nori measurements cover but not for such
 * tiny ones. The process-wide setting is also latched by the first CUDA initialisation. fp16/bf16 are
 * refused: they would need a real graph conversion and a silent fp32 run would be a lie.
 *
 * Independent of duckdb. Every exception is caught at the API boundary and reported through the err buffer.
 *===----------------------------------------------------------------------===*/

#include "tabfm_ggml_engine.hpp"
#include "tabfm_ggml_lower.hpp"
#include "tabfm_onnx_reader.hpp"
#include "tabfm_plugin_abi.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using namespace duckdb::anofox; // NOLINT (the engine lives in the extension namespace, see tabfm_ggml_lower.hpp)

void SetError(char *err, size_t err_len, const std::string &message) {
	if (!err || !err_len) {
		return;
	}
	std::strncpy(err, message.c_str(), err_len - 1);
	err[err_len - 1] = '\0';
}

//! A read-only memory map of one file.
class MappedFile {
public:
	explicit MappedFile(const std::string &path) {
#if defined(_WIN32)
		file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (file_ == INVALID_HANDLE_VALUE) {
			throw std::runtime_error("cannot open '" + path + "'");
		}
		LARGE_INTEGER sz;
		GetFileSizeEx(file_, &sz);
		size_ = static_cast<size_t>(sz.QuadPart);
		mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
		if (!mapping_) {
			throw std::runtime_error("cannot map '" + path + "'");
		}
		data_ = static_cast<const uint8_t *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
#else
		const int fd = ::open(path.c_str(), O_RDONLY);
		if (fd < 0) {
			throw std::runtime_error("cannot open '" + path + "'");
		}
		struct stat st {};
		if (::fstat(fd, &st) != 0) {
			::close(fd);
			throw std::runtime_error("cannot stat '" + path + "'");
		}
		size_ = static_cast<size_t>(st.st_size);
		void *p = size_ ? ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0) : nullptr;
		::close(fd);
		data_ = p == MAP_FAILED ? nullptr : static_cast<const uint8_t *>(p);
#endif
		if (!data_ && size_) {
			throw std::runtime_error("cannot map '" + path + "'");
		}
	}
	~MappedFile() {
#if defined(_WIN32)
		if (data_) {
			UnmapViewOfFile(data_);
		}
		if (mapping_) {
			CloseHandle(mapping_);
		}
		if (file_ != INVALID_HANDLE_VALUE) {
			CloseHandle(file_);
		}
#else
		if (data_) {
			::munmap(const_cast<uint8_t *>(data_), size_);
		}
#endif
	}
	MappedFile(const MappedFile &) = delete;
	MappedFile &operator=(const MappedFile &) = delete;
	const uint8_t *data() const {
		return data_;
	}
	size_t size() const {
		return size_;
	}

private:
	const uint8_t *data_ = nullptr;
	size_t size_ = 0;
#if defined(_WIN32)
	HANDLE file_ = INVALID_HANDLE_VALUE;
	HANDLE mapping_ = nullptr;
#endif
};

struct GgmlPluginBackend {
	::anofox::onnxread::Graph graph;
	std::vector<std::string> input_names;
	std::string output_name;
	bool graph_has_train_size = false;
	std::vector<std::unique_ptr<MappedFile>> files; // one per external-data location (normally just model.safetensors)
	std::unordered_map<std::string, ExternalWeight> weights;
	std::unique_ptr<GgmlExecutor> executor;
	std::mutex mutex;
	int device_ordinal = 0;
};

size_t ElementCount(const std::vector<int64_t> &dims) {
	size_t n = 1;
	for (auto d : dims) {
		n *= static_cast<size_t>(d);
	}
	return n;
}

std::string JoinPath(const std::string &dir, const std::string &leaf) {
	if (dir.empty() || dir.back() == '/' || dir.back() == '\\') {
		return dir + leaf;
	}
	return dir + "/" + leaf;
}

void SetTf32(const std::string &precision) {
	if (precision != "fp32") {
		return; // "tf32": the hardware default
	}
	// Strict fp32 matmuls. Read by cuBLAS and by ggml's own tensor-core kernels' dispatch when the CUDA runtime starts, so it has to be
	// set before the first device is initialised; a value the user set explicitly wins.
#if defined(_WIN32)
	if (!std::getenv("NVIDIA_TF32_OVERRIDE")) {
		_putenv_s("NVIDIA_TF32_OVERRIDE", "0");
	}
#else
	setenv("NVIDIA_TF32_OVERRIDE", "0", 0);
#endif
}

const char *PluginName(void) {
	return "ggml";
}

void *PluginCreate(const TabFMPluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path || !params->weights_dir) {
		SetError(err, err_len, "missing required create parameter (graph_path/weights_dir)");
		return nullptr;
	}
	try {
		const std::string precision = params->precision && params->precision[0] ? params->precision : "fp32";
		if (precision != "fp32" && precision != "tf32") {
			SetError(err, err_len, "precision '" + precision + "' is not supported by the ggml GPU backend (fp32 or tf32); bf16/fp16 would need a graph conversion");
			return nullptr;
		}
		SetTf32(precision);

		auto backend = std::unique_ptr<GgmlPluginBackend>(new GgmlPluginBackend());
		backend->device_ordinal = params->device_ordinal;
		backend->graph = ::anofox::onnxread::ParseModelFile(params->graph_path);

		std::vector<std::string> ops;
		for (const auto &n : backend->graph.nodes) {
			ops.push_back(n.op_type);
		}
		const std::string refusal = GgmlRefusal(ops);
		if (!refusal.empty()) {
			SetError(err, err_len, "the ggml GPU backend cannot run this model's graph: " + refusal);
			return nullptr;
		}

		// External weights: the staged graph carries absolute byte offsets into the safetensors file.
		std::set<std::string> initializer_names;
		std::unordered_map<std::string, size_t> file_index;
		for (const auto &t : backend->graph.initializers) {
			initializer_names.insert(t.name);
			const bool external = t.location == ::anofox::onnxread::DataLocation::EXTERNAL;
			if (!external) {
				continue;
			}
			const auto loc = t.external.find("location");
			const auto off = t.external.find("offset");
			if (loc == t.external.end() || off == t.external.end()) {
				throw std::runtime_error("initializer '" + t.name + "' has no external location/offset");
			}
			auto found = file_index.find(loc->second);
			if (found == file_index.end()) {
				backend->files.push_back(std::unique_ptr<MappedFile>(new MappedFile(JoinPath(params->weights_dir, loc->second))));
				found = file_index.emplace(loc->second, backend->files.size() - 1).first;
			}
			const MappedFile &file = *backend->files[found->second];
			const size_t offset = static_cast<size_t>(std::stoull(off->second));
			const size_t count = ElementCount(t.dims);
			if (t.data_type != 1) {
				throw std::runtime_error("the weight '" + t.name + "' is stored as ONNX type " + std::to_string(t.data_type) +
				                         " (bf16/fp16); the ggml GPU backend reads float32 checkpoints only. Use the ONNX Runtime or MIGraphX plugin for this model");
			}
			if (offset + count * sizeof(float) > file.size()) {
				throw std::runtime_error("the weight '" + t.name + "' does not fit the checkpoint file '" + loc->second +
				                         "' (is the graph for a different checkpoint, or the file truncated?)");
			}
			backend->weights[t.name] = {reinterpret_cast<const float *>(file.data() + offset), count};
		}
		for (const auto &name : backend->graph.inputs) {
			if (!initializer_names.count(name)) {
				backend->input_names.push_back(name);
				backend->graph_has_train_size = backend->graph_has_train_size || name == "train_size";
			}
		}
		if (backend->graph.outputs.empty()) {
			throw std::runtime_error("the graph declares no outputs");
		}
		backend->output_name = backend->graph.outputs.front();
		for (const auto &o : backend->graph.outputs) {
			if (o == "logits") {
				backend->output_name = o;
			}
		}

		GgmlExecutorConfig config;
		// The GPU runs the matmuls; the threads serve the CPU kernels the GPU backend cannot (TopK, Where, the float64 islands).
		config.threads = static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
		config.device = "gpu:" + std::to_string(params->device_ordinal);
		auto *self = backend.get();
		config.weights = [self](const std::string &name, const std::vector<int64_t> &dims) -> ExternalWeight {
			const auto it = self->weights.find(name);
			if (it == self->weights.end() || it->second.count != ElementCount(dims)) {
				return ExternalWeight();
			}
			return it->second;
		};
		backend->executor.reset(new GgmlExecutor(backend->graph, std::move(config)));
		// Upload the weights now: a device that cannot hold the model fails here, at load, with the host's "could not be initialised" message
		// instead of inside the first query, and the first predict no longer pays the transfer.
		backend->executor->PreloadWeights();
		return backend.release();
	} catch (const std::exception &e) {
		SetError(err, err_len,
		         std::string("could not initialise the ggml GPU backend on device ") + std::to_string(params->device_ordinal) + ": " + e.what());
		return nullptr;
	} catch (...) {
		SetError(err, err_len, "could not initialise the ggml GPU backend: unknown error");
		return nullptr;
	}
}

TabFMPluginStatus PluginRun(void *handle, const TabFMPluginRunInput *input, TabFMPluginRunOutput *output, char *err, size_t err_len) {
	auto *backend = static_cast<GgmlPluginBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return TABFM_PLUGIN_ERROR;
	}
	try {
		std::lock_guard<std::mutex> guard(backend->mutex);
		// Same contract as the CPU engines: a graph without a train_size input derives the split from y's length, so it gets the prefix.
		const int64_t y_len = backend->graph_has_train_size ? input->t : input->train_size;
		HostTensors in;
		GgmlRunOptions options;
		options.threads = 1;
		for (const auto &name : backend->input_names) {
			if (name == "x") {
				HostTensor t;
				t.shape = {1, input->t, input->h};
				t.data.assign(input->x, input->x + input->t * input->h);
				in[name] = std::move(t);
			} else if (name == "y") {
				HostTensor t;
				t.shape = {1, y_len};
				t.data.assign(input->y, input->y + y_len);
				in[name] = std::move(t);
			} else if (name == "cat_mask") {
				HostTensor t;
				t.shape = {1, input->h};
				for (int64_t i = 0; i < input->h; i++) {
					t.data.push_back(input->cat_mask[i] ? 1.0f : 0.0f);
				}
				in[name] = std::move(t);
				options.bool_inputs.insert(name);
			} else if (name == "train_size") {
				options.int_inputs[name] = {{1}, {input->train_size}};
			} else if (name == "d") {
				options.int_inputs[name] = {{1}, {input->d}};
			} else {
				SetError(err, err_len, "graph declares input '" + name + "' which this plugin does not provide");
				return TABFM_PLUGIN_ERROR;
			}
		}
		options.outputs = {backend->output_name};
		auto outputs = backend->executor->Run(in, options);
		const auto found = outputs.find(backend->output_name);
		if (found == outputs.end()) {
			SetError(err, err_len, "the engine did not return '" + backend->output_name + "'");
			return TABFM_PLUGIN_ERROR;
		}
		const HostTensor &result = found->second;
		output->logits = static_cast<float *>(std::malloc(sizeof(float) * result.data.size()));
		output->shape = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * std::max<size_t>(1, result.shape.size())));
		if (!output->logits || !output->shape) {
			std::free(output->logits);
			std::free(output->shape);
			output->logits = nullptr;
			output->shape = nullptr;
			SetError(err, err_len, "allocation failed");
			return TABFM_PLUGIN_ERROR;
		}
		std::memcpy(output->logits, result.data.data(), sizeof(float) * result.data.size());
		std::memcpy(output->shape, result.shape.data(), sizeof(int64_t) * result.shape.size());
		output->logits_len = static_cast<int64_t>(result.data.size());
		output->shape_len = static_cast<int64_t>(result.shape.size());
		return TABFM_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("ggml GPU inference failed on device ") + std::to_string(backend->device_ordinal) + ": " + e.what());
		return TABFM_PLUGIN_ERROR;
	} catch (...) {
		SetError(err, err_len, "ggml GPU inference failed: unknown error");
		return TABFM_PLUGIN_ERROR;
	}
}

// ggml builds a graph per input-shape signature the first time it sees one and caches it; there is nothing to warm without data.
TabFMPluginStatus PluginPrecompile(void *handle, int64_t, int64_t, char *err, size_t err_len) {
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
	delete static_cast<GgmlPluginBackend *>(handle);
}

const char *const *PluginGraphInputs(void) {
	static const char *const inputs[] = {"x", "y", "cat_mask", "train_size", "d", nullptr};
	return inputs;
}

const TabFMPluginApi kApi = {
    TABFM_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy, PluginGraphInputs,
};

} // namespace

extern "C" TABFM_PLUGIN_EXPORT const TabFMPluginApi *TabFMGetPluginApi(void) {
	return &kApi;
}
