//===----------------------------------------------------------------------===//
// tabfm_ort_engine.cpp — ONNX Runtime engine (WS-C, HLD §4.4 / §9, S02)
//
// One process-wide Ort::Env; sessions are created from weight-free graph
// bytes with all initializers injected via AddExternalInitializers. S02:
// ORT COPIES injected tensors during session init — the caller frees the
// source arena right after CreateSession returns.
//===----------------------------------------------------------------------===//

#include "tabfm_ort_engine.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstring>

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// Dtype helpers & cast hooks
//===----------------------------------------------------------------------===//

idx_t TabFMDtypeSize(TabFMTensorDtype dtype) {
	switch (dtype) {
	case TabFMTensorDtype::F32:
		return 4;
	case TabFMTensorDtype::F16:
	case TabFMTensorDtype::BF16:
		return 2;
	case TabFMTensorDtype::I64:
		return 8;
	case TabFMTensorDtype::BOOL:
		return 1;
	default:
		throw InternalException("anofox_tabfm: unknown tensor dtype");
	}
}

namespace {

uint16_t F32ToF16Bits(float value) {
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	const uint32_t sign = (bits >> 16) & 0x8000u;
	const uint32_t exp = (bits >> 23) & 0xffu;
	uint32_t mant = bits & 0x7fffffu;
	if (exp == 0xffu) { // inf / nan
		uint32_t payload = mant ? (0x200u | (mant >> 13)) : 0u;
		return static_cast<uint16_t>(sign | 0x7c00u | payload);
	}
	int32_t e = static_cast<int32_t>(exp) - 127 + 15;
	if (e >= 31) { // overflow -> inf
		return static_cast<uint16_t>(sign | 0x7c00u);
	}
	if (e <= 0) { // subnormal or zero
		if (e < -10) {
			return static_cast<uint16_t>(sign);
		}
		mant |= 0x800000u;
		const uint32_t shift = static_cast<uint32_t>(14 - e);
		uint32_t half = mant >> shift;
		const uint32_t rem = mant & ((1u << shift) - 1u);
		const uint32_t halfway = 1u << (shift - 1u);
		if (rem > halfway || (rem == halfway && (half & 1u))) {
			half++;
		}
		return static_cast<uint16_t>(sign | half);
	}
	uint32_t half = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
	const uint32_t rem = mant & 0x1fffu;
	if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
		half++; // rounding carry propagates into the exponent correctly
	}
	return static_cast<uint16_t>(half);
}

uint16_t F32ToBF16Bits(float value) {
	uint32_t bits;
	std::memcpy(&bits, &value, sizeof(bits));
	if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x7fffffu)) {
		return static_cast<uint16_t>((bits >> 16) | 0x0040u); // quiet the NaN
	}
	// round-to-nearest-even on the truncated 16 bits
	const uint32_t lsb = (bits >> 16) & 1u;
	bits += 0x7fffu + lsb;
	return static_cast<uint16_t>(bits >> 16);
}

} // anonymous namespace

vector<uint16_t> CastF32ToF16(const float *src, idx_t count) {
	vector<uint16_t> result(count);
	for (idx_t i = 0; i < count; i++) {
		result[i] = F32ToF16Bits(src[i]);
	}
	return result;
}

vector<uint16_t> CastF32ToBF16(const float *src, idx_t count) {
	vector<uint16_t> result(count);
	for (idx_t i = 0; i < count; i++) {
		result[i] = F32ToBF16Bits(src[i]);
	}
	return result;
}

vector<float> CastBF16ToF32(const uint16_t *src, idx_t count) {
	vector<float> result(count);
	for (idx_t i = 0; i < count; i++) {
		const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
		std::memcpy(&result[i], &bits, sizeof(float));
	}
	return result;
}

//===----------------------------------------------------------------------===//
// Process-wide Ort::Env
//===----------------------------------------------------------------------===//

namespace {

Ort::Env &GetOrtEnv() {
	// Leaked intentionally (same pattern as the telemetry singleton): the env
	// owns ORT's logging sink and default thread pools; tearing it down during
	// dlclose / static destruction of the loadable extension is unsafe.
	static Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "anofox_tabfm");
	return *env;
}

// Guard the FIRST ORT API use. If the onnxruntime shared library resolved at
// runtime is older than the headers this was compiled against,
// OrtGetApiBase()->GetApi(ORT_API_VERSION) returns null and the very next ORT
// call dereferences a null API table (a bare SIGSEGV). This most commonly bites
// on Windows, where Windows ML ships an old C:\Windows\System32\onnxruntime.dll
// that the loader can find before our bundled copy. Turn that crash into an
// actionable error.
void EnsureUsableOrtApi() {
	const OrtApiBase *base = OrtGetApiBase();
	if (base && base->GetApi(ORT_API_VERSION)) {
		return;
	}
	const char *ver = (base && base->GetVersionString) ? base->GetVersionString() : "unknown";
	throw IOException(
	    "anofox_tabfm: the ONNX Runtime loaded at runtime is version %s, too old for this build (needs ORT API "
	    "version %d — ONNX Runtime >= 1.23). A stale onnxruntime.dll is being resolved before the bundled one "
	    "(on Windows this is usually C:\\Windows\\System32\\onnxruntime.dll from Windows ML). Put the extension's "
	    "own onnxruntime.dll first on the DLL search path (next to the host executable).",
	    ver, static_cast<int>(ORT_API_VERSION));
}

ONNXTensorElementDataType ToOnnxElementType(TabFMTensorDtype dtype) {
	switch (dtype) {
	case TabFMTensorDtype::F32:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	case TabFMTensorDtype::F16:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
	case TabFMTensorDtype::BF16:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
	case TabFMTensorDtype::I64:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
	case TabFMTensorDtype::BOOL:
		return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
	default:
		throw InternalException("anofox_tabfm: unknown tensor dtype");
	}
}

//! Best-effort extraction of a tensor name from an ORT error message.
//! ORT phrases vary; the common shapes are `tensor name: <name>` and
//! `... tensor '<name>' ...`.
string ExtractTensorName(const string &message) {
	auto pos = message.find("tensor name: ");
	if (pos != string::npos) {
		pos += 13;
		auto end = message.find_first_of(" ,)\n", pos);
		return message.substr(pos, end == string::npos ? string::npos : end - pos);
	}
	pos = message.find("tensor '");
	if (pos != string::npos) {
		pos += 8;
		auto end = message.find('\'', pos);
		if (end != string::npos) {
			return message.substr(pos, end - pos);
		}
	}
	return string();
}

[[noreturn]] void ThrowMappedCreateError(const Ort::Exception &error, const TabFMSessionConfig &config) {
	const string message = error.what();
	const int code = error.GetOrtErrorCode();
	const string task = config.model_tag.empty() ? string("classification") : config.model_tag;

	// Missing injection: the weight-free graph's external-data stubs cannot be
	// resolved because no initializers were injected. The signature depends on
	// how the session is created:
	//   * from a file (S02): ORT_NO_SUCHFILE + "cannot get file size";
	//   * from a byte array (how this extension loads the embedded graph):
	//     ORT_RUNTIME_EXCEPTION (code 6) + "model_path must not be empty"
	//     (initializer.cc cannot locate the external file with no model path).
	// Also match the export-time "External data path ... does not exist"
	// phrasing (S01). Match message shapes as well as codes to stay robust
	// across ORT versions and load paths.
	const bool missing_weights = code == ORT_NO_SUCHFILE || message.find("No such file") != string::npos ||
	                             message.find("cannot get file size") != string::npos ||
	                             message.find("filesystem error") != string::npos ||
	                             message.find("model_path must not be empty") != string::npos ||
	                             message.find("External data") != string::npos ||
	                             message.find("external data") != string::npos ||
	                             message.find("external file") != string::npos;
	if (missing_weights) {
		throw InvalidInputException(
		    "anofox_tabfm: model weights are not loaded — the model graph references external weights that were "
		    "not injected. Run: CALL tabfm_load('" +
		    task + "'); (downloading them first if needed: CALL tabfm_download('" + task +
		    "');). ORT said: " + message);
	}

	// Corrupted / mismatched checkpoint: ORT verifies name, dims and dtype of
	// every replacement against the stub TensorProto (S02 negative control b,
	// error code 1 = ORT_FAIL, "Replacement tensor's dimensions do not match").
	const bool corrupt = message.find("dimensions do not match") != string::npos ||
	                     message.find("Replacement tensor") != string::npos ||
	                     message.find("data type does not match") != string::npos ||
	                     message.find("ReplaceInitializedTensor") != string::npos;
	if (corrupt) {
		const string tensor_name = ExtractTensorName(message);
		string named = tensor_name.empty() ? string() : " (tensor '" + tensor_name + "')";
		throw IOException("anofox_tabfm: the checkpoint for '" + task +
		                  "' is corrupted or does not match the bundled model graph" + named +
		                  " — remove and re-download it: CALL tabfm_remove('" + task + "'); CALL tabfm_download('" +
		                  task + "');. ORT said: " + message);
	}

	throw InvalidInputException("anofox_tabfm: ONNX Runtime failed to create the session for '" + task +
	                            "' (ORT error code " + std::to_string(code) + "): " + message);
}

[[noreturn]] void ThrowFlavorMissingDeviceLocal(const string &requested) {
	throw InvalidInputException("anofox_tabfm: this build is the '" + string(TabFMFlavorName()) +
	                            "' flavor and does not carry '" + requested + "'; install the '" + requested +
	                            "' flavor from the anofox extension repository (SET custom_extension_repository = "
	                            "'https://ext.anofox.com/tabfm/" +
	                            requested + "') or SET anofox_tabfm_device='cpu'");
}

void AppendExecutionProviders(Ort::SessionOptions &options, const TabFMSessionConfig &config) {
	const auto &device = config.device_id;
	if (device == "cpu" || device.empty()) {
		return; // CPU EP is implicit
	}
	if (StringUtil::StartsWith(device, "cuda")) {
#ifdef TABFM_EP_CUDA
		OrtCUDAProviderOptions cuda_options {};
		cuda_options.device_id = config.device_ordinal;
		options.AppendExecutionProvider_CUDA(cuda_options);
		return;
#else
		ThrowFlavorMissingDeviceLocal("cuda");
#endif
	}
	if (StringUtil::StartsWith(device, "rocm") || StringUtil::StartsWith(device, "migraphx")) {
#ifdef TABFM_EP_MIGRAPHX
		OrtMIGraphXProviderOptions migraphx_options {};
		migraphx_options.device_id = config.device_ordinal;
		options.AppendExecutionProvider_MIGraphX(migraphx_options);
		return;
#else
		ThrowFlavorMissingDeviceLocal("rocm");
#endif
	}
	throw InvalidInputException("anofox_tabfm: unknown execution device '" + device +
	                            "' — expected 'cpu', 'cuda[:n]' or 'rocm[:n]'");
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// TabFMSession
//===----------------------------------------------------------------------===//

class TabFMSession {
public:
	TabFMSession(Ort::Session session_p, TabFMSessionConfig config_p, std::vector<std::string> injected_names_p,
	             std::vector<Ort::Value> injected_values_p)
	    : config(std::move(config_p)), injected_names(std::move(injected_names_p)),
	      injected_values(std::move(injected_values_p)), session(std::move(session_p)) {
		Ort::AllocatorWithDefaultOptions allocator;
		const auto input_count = session.GetInputCount();
		for (size_t i = 0; i < input_count; i++) {
			input_names.push_back(session.GetInputNameAllocated(i, allocator).get());
		}
		const auto output_count = session.GetOutputCount();
		for (size_t i = 0; i < output_count; i++) {
			output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
		}
	}

	// Member DECLARATION ORDER is the destruction contract here. ORT's
	// AddExternalInitializers keeps references to the injected OrtValues (and
	// their user buffers) for the whole session lifetime, so the Ort::Session
	// must be destroyed BEFORE `injected_values`/`injected_names`. Members are
	// destroyed in reverse declaration order, so `session` is declared LAST.
	// (At fixture scale ORT copies eagerly and this is moot; at real scale —
	// 913 tensors / 6.6 GB — dropping the values early is a use-after-free /
	// zeroed-weights hazard.) The constructor init-list mirrors this order.
	TabFMSessionConfig config;
	vector<string> input_names;
	vector<string> output_names;
	std::vector<std::string> injected_names;
	std::vector<Ort::Value> injected_values;
	Ort::Session session;
};

namespace {

// Shared SessionOptions setup + external-initializer injection. Returns the
// options; `names`/`values` are OUT params that MUST stay alive until the
// Ort::Session is constructed (AddExternalInitializers keeps references).
void PrepareSessionOptions(Ort::SessionOptions &options, const vector<TabFMTensorRef> &initializers,
                           const TabFMSessionConfig &config, std::vector<std::string> &names,
                           std::vector<Ort::Value> &values) {
	options.SetIntraOpNumThreads(static_cast<int>(MaxValue<int64_t>(1, config.intra_op_threads)));
	options.SetInterOpNumThreads(1); // inference runs inside a single DuckDB task (HLD §4.4)
	// Prepacking speeds up matmuls at ~+16% resident memory (anofox_tabfm_cpu_prepack).
	options.AddConfigEntry("session.disable_prepacking", config.prepack ? "0" : "1");
	AppendExecutionProviders(options, config);

	// Wrap the caller's buffers as non-owning Ort::Values for
	// AddExternalInitializers. ORT keeps references to these user buffers, so
	// the source arena must outlive the SESSION (LoadOrGetSession bundles it).
	names.reserve(initializers.size());
	values.reserve(initializers.size());
	static auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	for (auto &tensor : initializers) {
		if (!tensor.data) {
			throw InvalidInputException("anofox_tabfm: initializer '" + tensor.name + "' has no data buffer");
		}
		idx_t element_count = 1;
		for (auto dim : tensor.shape) {
			if (dim < 0) {
				throw InvalidInputException("anofox_tabfm: initializer '" + tensor.name +
				                            "' has a negative dimension — corrupted checkpoint?");
			}
			auto d = static_cast<idx_t>(dim);
			if (d != 0 && element_count > NumericLimits<idx_t>::Maximum() / d) {
				throw InvalidInputException("anofox_tabfm: initializer '" + tensor.name +
				                            "' shape product overflows — corrupted checkpoint?");
			}
			element_count *= d;
		}
		const idx_t dtype_size = TabFMDtypeSize(tensor.dtype);
		if (element_count != 0 && element_count > NumericLimits<idx_t>::Maximum() / dtype_size) {
			throw InvalidInputException("anofox_tabfm: initializer '" + tensor.name +
			                            "' byte size overflows — corrupted checkpoint?");
		}
		if (element_count * dtype_size != tensor.size_bytes) {
			throw InvalidInputException("anofox_tabfm: initializer '" + tensor.name +
			                            "' byte size does not match its shape — corrupted checkpoint?");
		}
		names.push_back(tensor.name);
		values.push_back(Ort::Value::CreateTensor(memory_info, const_cast<void *>(tensor.data), tensor.size_bytes,
		                                          tensor.shape.data(), tensor.shape.size(),
		                                          ToOnnxElementType(tensor.dtype)));
	}
	if (!names.empty()) {
		options.AddExternalInitializers(names, values);
	}
}

} // namespace

TabFMSessionHandle CreateSession(const void *graph_bytes, idx_t graph_size,
                                 const vector<TabFMTensorRef> &initializers, const TabFMSessionConfig &config) {
	if (!graph_bytes || graph_size == 0) {
		throw InvalidInputException("anofox_tabfm: cannot create a session from empty model graph bytes");
	}
	EnsureUsableOrtApi();
	Ort::SessionOptions options;
	std::vector<std::string> names;
	std::vector<Ort::Value> values;
	PrepareSessionOptions(options, initializers, config, names, values);
	try {
		Ort::Session session(GetOrtEnv(), graph_bytes, graph_size, options);
		return make_shared_ptr<TabFMSession>(std::move(session), config, std::move(names), std::move(values));
	} catch (const Ort::Exception &error) {
		ThrowMappedCreateError(error, config);
	}
}

TabFMSessionHandle CreateSessionFromPath(const string &graph_path, const vector<TabFMTensorRef> &initializers,
                                         const TabFMSessionConfig &config) {
	EnsureUsableOrtApi();
	Ort::SessionOptions options;
	std::vector<std::string> names;
	std::vector<Ort::Value> values;
	PrepareSessionOptions(options, initializers, config, names, values);
	try {
		// ORT paths are ORTCHAR_T*: wchar_t on Windows, char elsewhere. Cache
		// paths are ASCII, so a widening copy is sufficient on Windows.
#ifdef _WIN32
		std::wstring ort_path(graph_path.begin(), graph_path.end());
		Ort::Session session(GetOrtEnv(), ort_path.c_str(), options);
#else
		Ort::Session session(GetOrtEnv(), graph_path.c_str(), options);
#endif
		return make_shared_ptr<TabFMSession>(std::move(session), config, std::move(names), std::move(values));
	} catch (const Ort::Exception &error) {
		ThrowMappedCreateError(error, config);
	}
}

//===----------------------------------------------------------------------===//
// Run
//===----------------------------------------------------------------------===//

TabFMRunOutput Run(TabFMSession &session, const TabFMRunInput &input) {
	if (!input.x || !input.y || !input.cat_mask) {
		throw InternalException("anofox_tabfm: Run() called with null input buffers");
	}
	if (input.t <= 0 || input.h <= 0) {
		throw InvalidInputException("anofox_tabfm: Run() needs positive row count T and feature count H, got T=" +
		                            std::to_string(input.t) + " H=" + std::to_string(input.h));
	}

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

	// Scalars must outlive session.Run — keep them on this frame.
	int64_t train_size_value = input.train_size;
	int64_t d_value = input.d;

	const std::array<int64_t, 3> x_shape {1, input.t, input.h};
	const std::array<int64_t, 2> y_shape {1, input.t};
	const std::array<int64_t, 1> scalar_shape {1};
	const std::array<int64_t, 2> mask_shape {1, input.h};

	// Feed by the graph's own input names so exporter-side reordering cannot
	// silently misbind tensors.
	std::vector<const char *> feed_names;
	std::vector<Ort::Value> feed_values;
	for (auto &name : session.input_names) {
		feed_names.push_back(name.c_str());
		if (name == "x") {
			feed_values.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(input.x),
			                                                      static_cast<size_t>(input.t * input.h),
			                                                      x_shape.data(), x_shape.size()));
		} else if (name == "y") {
			feed_values.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(input.y),
			                                                      static_cast<size_t>(input.t), y_shape.data(),
			                                                      y_shape.size()));
		} else if (name == "train_size") {
			feed_values.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, &train_size_value, 1,
			                                                        scalar_shape.data(), scalar_shape.size()));
		} else if (name == "cat_mask") {
			feed_values.push_back(Ort::Value::CreateTensor<bool>(memory_info, const_cast<bool *>(input.cat_mask),
			                                                     static_cast<size_t>(input.h), mask_shape.data(),
			                                                     mask_shape.size()));
		} else if (name == "d") {
			feed_values.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, &d_value, 1, scalar_shape.data(),
			                                                        scalar_shape.size()));
		} else {
			throw InternalException("anofox_tabfm: model graph declares unexpected input '" + name +
			                        "' — the engine feeds x, y, train_size, cat_mask, d (HLD §4.4)");
		}
	}

	if (session.output_names.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: model graph declares no outputs; expected a 'logits' output (HLD §4.4). If you set "
		    "anofox_tabfm_model_manifest, point it at a compatible weight-free graph.");
	}
	// Prefer the output named "logits"; otherwise take the first output.
	idx_t output_index = 0;
	for (idx_t i = 0; i < session.output_names.size(); i++) {
		if (session.output_names[i] == "logits") {
			output_index = i;
			break;
		}
	}
	const char *output_name = session.output_names[output_index].c_str();

	try {
		auto outputs = session.session.Run(Ort::RunOptions {nullptr}, feed_names.data(), feed_values.data(),
		                                   feed_values.size(), &output_name, 1);
		auto &logits_value = outputs[0];
		auto info = logits_value.GetTensorTypeAndShapeInfo();
		if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
			throw InternalException("anofox_tabfm: model output '" + string(output_name) +
			                        "' is not float32 — unexpected graph");
		}
		TabFMRunOutput result;
		auto shape = info.GetShape();
		result.shape = vector<int64_t>(shape.begin(), shape.end());
		const auto element_count = info.GetElementCount();
		const float *data = logits_value.GetTensorData<float>();
		result.logits.assign(data, data + element_count);
		return result;
	} catch (const Ort::Exception &error) {
		throw InvalidInputException("anofox_tabfm: inference failed (ORT error code " +
		                            std::to_string(error.GetOrtErrorCode()) + "): " + string(error.what()));
	}
}

void ValidateTabFMOutput(const TabFMRunOutput &out, idx_t expected_t, idx_t min_classes, const char *task_name) {
	auto shape_str = [&]() {
		string s = "[";
		for (idx_t i = 0; i < out.shape.size(); i++) {
			s += (i ? ", " : "") + std::to_string(out.shape[i]);
		}
		return s + "]";
	};
	if (out.shape.size() != 3 || out.shape[0] != 1 ||
	    out.shape[1] != NumericCast<int64_t>(expected_t)) {
		throw InvalidInputException(
		    "anofox_tabfm: model produced logits of shape %s, but the engine contract is [1, %llu, C] for %s "
		    "(HLD §4.4). The configured graph does not match the engine — check anofox_tabfm_model_manifest.",
		    shape_str(), static_cast<unsigned long long>(expected_t), task_name);
	}
	const auto C = NumericCast<idx_t>(out.shape[2]);
	if (C < 1 || C < min_classes) {
		throw InvalidInputException(
		    "anofox_tabfm: model output has %llu class column(s) but %s needs at least %llu — the graph's "
		    "class head does not match the data. Check anofox_tabfm_model_manifest.",
		    static_cast<unsigned long long>(C), task_name, static_cast<unsigned long long>(min_classes));
	}
	// shape[0] == 1, so the element count is expected_t * C (both bounded by the
	// row/feature guardrails, no overflow).
	const idx_t expected = expected_t * C;
	if (out.logits.size() != expected) {
		throw InvalidInputException(
		    "anofox_tabfm: model returned %llu logits but the [1, %llu, %llu] output requires %llu — truncated or "
		    "malformed model output.",
		    static_cast<unsigned long long>(out.logits.size()), static_cast<unsigned long long>(expected_t),
		    static_cast<unsigned long long>(C), static_cast<unsigned long long>(expected));
	}
}

} // namespace anofox
} // namespace duckdb
