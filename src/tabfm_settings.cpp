#include "tabfm_registration.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

#include <thread>

namespace duckdb {
namespace anofox {

namespace {

void ValidateDevice(ClientContext &context, SetScope scope, Value &parameter) {
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_device cannot be NULL");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter));
	if (value != "auto" && value != "cpu" && value != "cuda" && value != "rocm" && value != "migraphx") {
		throw InvalidInputException("anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm' "
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
	if (value != "fp32" && value != "bf16" && value != "fp16") {
		throw InvalidInputException("anofox_tabfm_gpu_precision must be 'bf16', 'fp16' or 'fp32', got '%s'", value);
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

	const auto default_threads = MaxValue<int64_t>(1, static_cast<int64_t>(std::thread::hardware_concurrency()) / 2);
	config.AddExtensionOption("anofox_tabfm_threads", "ONNX Runtime intra-op thread count for CPU inference",
	                          LogicalType::BIGINT, Value::BIGINT(default_threads), ValidateThreads);

	config.AddExtensionOption("anofox_tabfm_max_rows", "Maximum rows per predict call or group",
	                          LogicalType::BIGINT, Value::BIGINT(10000), ValidateMaxRows);

	config.AddExtensionOption("anofox_tabfm_max_features", "Maximum feature columns per predict call",
	                          LogicalType::BIGINT, Value::BIGINT(500), ValidateMaxFeatures);

	config.AddExtensionOption("anofox_tabfm_model_manifest",
	                          "Path to an extra model manifest JSON (pluggable models, CI fixture)",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption("anofox_tabfm_trace_level", "Diagnostic verbosity: error|warn|info|debug|trace",
	                          LogicalType::VARCHAR, Value("warn"), ValidateTraceLevel);

	config.AddExtensionOption(
	    "anofox_tabfm_gpu_precision",
	    "MIGraphX compile precision on the ROCm GPU: bf16|fp16|fp32. bf16 (default) runs ~2x faster than fp32 on "
	    "RDNA4 and halves VRAM/.mxr, keeping fp32's exponent range; fp32 is the accuracy reference.",
	    LogicalType::VARCHAR, Value("bf16"), ValidateGpuPrecision);

	config.AddExtensionOption(
	    "anofox_tabfm_cpu_prepack",
	    "Enable ONNX Runtime weight prepacking on the CPU EP: faster matmuls at ~+16% resident memory.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(true));

	config.AddExtensionOption("anofox_tabfm_device",
	                          "Execution device: auto|cpu|cuda|rocm ('migraphx' alias). Each flavor errors "
	                          "helpfully on devices it does not carry.",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);

	config.AddExtensionOption("anofox_tabfm_ep_path",
	                          "Directory with ONNX Runtime provider / plugin-EP shared libraries",
	                          LogicalType::VARCHAR, Value(""));
}

} // namespace anofox
} // namespace duckdb
