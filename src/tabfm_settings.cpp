#include "tabfm_registration.hpp"
#include "tabfm_cpu_budget.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace anofox {

namespace {

void ValidateDevice(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_device cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "auto" && value != "cpu" && value != "cuda" && value != "rocm" && value != "migraphx" &&
	    value != "coreml") {
		throw InvalidInputException("anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm', 'coreml' "
		                            "('migraphx' is accepted as an alias for 'rocm'), got '%s'",
		                            value);
	}
	parameter = Value(value == "migraphx" ? "rocm" : value);
}

void ValidateTraceLevel(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_trace_level cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "error" && value != "warn" && value != "info" && value != "debug" && value != "trace") {
		throw InvalidInputException(
		    "anofox_tabfm_trace_level must be one of 'error', 'warn', 'info', 'debug', 'trace', got '%s'", value);
	}
	parameter = Value(value);
}

void ValidateGpuPrecision(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_gpu_precision cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "fp32" && value != "tf32" && value != "bf16" && value != "fp16") {
		throw InvalidInputException("anofox_tabfm_gpu_precision must be 'fp32', 'tf32', 'bf16' or 'fp16', got '%s'",
		                            value);
	}
	parameter = Value(value);
}

void ValidatePositive(const char *name, ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("%s cannot be NULL", name);
	}
	auto value = BigIntValue::Get(parameter.DefaultCastAs(LogicalType::BIGINT));
	if (value <= 0) {
		throw InvalidInputException("%s must be positive, got %lld", name, value);
	}
}

void ValidateThreads(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_threads", context, scope, parameter);
}

void ValidateMaxRows(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_max_rows", context, scope, parameter);
}

void ValidateMaxFeatures(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_max_features", context, scope, parameter);
}

void ValidateMaxMemory(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_max_memory cannot be NULL");
	}
	auto value = StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR));
	if (value.empty()) {
		return; // '' = disabled
	}
	idx_t bytes;
	auto error = StringUtil::TryParseFormattedBytes(value, bytes);
	if (!error.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm_max_memory: %s (got '%s'); use a size like '16GB', or '' to disable", error, value);
	}
}

} // anonymous namespace

void RegisterTabfmSettings(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	config.AddExtensionOption("anofox_tabfm_accept_hf_license",
	                          "Accept the upstream model license (tabfm-non-commercial-v1.0: non-commercial use, "
	                          "no redistribution). Downloads of Google-licensed weights fail without this.",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(false));

	config.AddExtensionOption("anofox_tabfm_cache_dir",
	                          "Weight cache root directory (default ~/.cache/anofox-tabfm)", LogicalType::VARCHAR,
	                          Value("~/.cache/anofox-tabfm"));

	const auto default_threads = MaxValue<int64_t>(1, static_cast<int64_t>(UsableCoreCount()) / 2);
	config.AddExtensionOption("anofox_tabfm_threads", "ONNX Runtime intra-op thread count for CPU inference",
	                          LogicalType::BIGINT, Value::BIGINT(default_threads), ValidateThreads);

	config.AddExtensionOption("anofox_tabfm_max_rows", "Maximum rows per predict call or group",
	                          LogicalType::BIGINT, Value::BIGINT(10000), ValidateMaxRows);

	config.AddExtensionOption("anofox_tabfm_max_features", "Maximum feature columns per predict call",
	                          LogicalType::BIGINT, Value::BIGINT(500), ValidateMaxFeatures);

	config.AddExtensionOption(
	    "anofox_tabfm_max_memory",
	    "Refuse a predict call when this process's resident memory is already at or above this size (e.g. '16GB') "
	    "before the call starts, so the failure is a DuckDB exception instead of a cgroup OOM-kill. '' (default) "
	    "disables the check. Checked against resident memory at call time, not an estimate of the call's own "
	    "cost -- it does not bound how much a single large call can grow memory by itself.",
	    LogicalType::VARCHAR, Value(""), ValidateMaxMemory);

	config.AddExtensionOption("anofox_tabfm_default_model",
	                          "Default model id for tabfm_classify/regress/download/... when model := is not given. "
	                          "'' = resolve to the single-file manifest model, else the sole registered model.",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption("anofox_tabfm_trace_level", "Diagnostic verbosity: error|warn|info|debug|trace",
	                          LogicalType::VARCHAR, Value("warn"), ValidateTraceLevel);

	config.AddExtensionOption(
	    "anofox_tabfm_gpu_precision",
	    "GPU numeric mode: fp32|tf32|bf16|fp16. fp32 (default) is strict — a device switch does not change answers, "
	    "measured exact on both GPUs; on CUDA it disables TF32 tensor-core rounding. tf32 re-enables that rounding "
	    "(CUDA only; fp32 storage, faster matmuls). bf16/fp16 quantize the MIGraphX program on ROCm (~2x faster on "
	    "RDNA4, half the VRAM/.mxr, a few near-tie labels may flip) and are rejected on CUDA rather than silently "
	    "running fp32.",
	    LogicalType::VARCHAR, Value("fp32"), ValidateGpuPrecision);

	config.AddExtensionOption(
	    "anofox_tabfm_context_cache",
	    "Encode the labelled context once and reuse it across calls, for models that ship a split graph pair "
	    "(prepare/query). Off by default. It pays off when the same context is scored more than once -- chunked "
	    "scoring, repeated queries against a fixed training table -- and costs extra on a single call, which pays "
	    "for the context it will not reuse. Test-row predictions match the combined graph; the fitted values on "
	    "CONTEXT rows differ, because the query half has no label path and so no longer sees a context row's own "
	    "label. Inert for a model that ships no pair.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(false));

	config.AddExtensionOption(
	    "anofox_tabfm_cpu_prepack",
	    "Enable ONNX Runtime weight prepacking on the CPU EP: faster matmuls at ~+16% resident memory.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(true));

	config.AddExtensionOption("anofox_tabfm_device",
	                          "Execution device: auto|cpu|cuda|rocm|coreml ('migraphx' alias). Each flavor errors "
	                          "helpfully on devices it does not carry.",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);

	config.AddExtensionOption("anofox_tabfm_ep_path",
	                          "Directory holding the GPU backend plugins (libanofox_tabfm_cuda_plugin.so, "
	                          "libanofox_tabfm_migraphx_plugin.so) and the runtime libraries they load alongside "
	                          "themselves. CALL tabfm_download_runtime('cuda') populates it.",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption(
	    "anofox_tabfm_mxr_source",
	    "Directory holding precompiled MIGraphX .mxr programs (offline/CI/shared cache). Before compiling a "
	    "shape-bucket (~27 min on ROCm), a matching '<model>_<arch>_<precision>_T<t>_H<h>.mxr' here is staged into the "
	    "cache and reused; empty ('' default) always compiles on-device. Artifacts are arch- and ROCm-version-specific.",
	    LogicalType::VARCHAR, Value(""));
}

} // namespace anofox
} // namespace duckdb
