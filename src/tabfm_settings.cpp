#include "tabfm_registration.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

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

//! Cores this process may actually run on.
//!
//! `std::thread::hardware_concurrency()` reports what the kernel can see, which inside a
//! container is the host. On a 64-core cpuset inside a 256-core host it returns 256, so a
//! default of `hardware_concurrency() / 2` becomes 128 intra-op threads per session -- and the
//! host runs several sessions concurrently, one per DuckDB task. Measured on such a pod: 132
//! threads in one duckdb process and a load average of 143 against 64 usable cores, for a query
//! configured with `SET threads = 4`.
//!
//! `sched_getaffinity` respects the cpuset and is the number that matters. Everything else falls
//! back to `hardware_concurrency()`, so behaviour is unchanged off Linux.
idx_t UsableCoreCount() {
#ifdef __linux__
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) == 0) {
		const auto affine = static_cast<idx_t>(CPU_COUNT(&set));
		if (affine > 0) {
			return affine;
		}
	}
#endif
	const auto visible = static_cast<idx_t>(std::thread::hardware_concurrency());
	return visible > 0 ? visible : 1;
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

	config.AddExtensionOption("anofox_tabfm_default_model",
	                          "Default model id for tabfm_classify/regress/download/... when model := is not given. "
	                          "'' = resolve to the single-file manifest model, else the sole registered model.",
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
	                          "Execution device: auto|cpu|cuda|rocm|coreml ('migraphx' alias). Each flavor errors "
	                          "helpfully on devices it does not carry.",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);

	config.AddExtensionOption("anofox_tabfm_ep_path",
	                          "Directory with ONNX Runtime provider / plugin-EP shared libraries",
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
