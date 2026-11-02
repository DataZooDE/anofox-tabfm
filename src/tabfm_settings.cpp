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
	    value != "coreml" && value != "mlx") {
		throw InvalidInputException("anofox_tabfm_device must be one of 'auto', 'cpu', 'cuda', 'rocm', 'coreml', "
		                            "'mlx' ('migraphx' is accepted as an alias for 'rocm'), got '%s'",
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

void ValidateMaxSessions(ClientContext &context, SetScope scope, Value &parameter) {
	ValidatePositive("anofox_tabfm_max_sessions", context, scope, parameter);
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

void ValidateContextCacheMemory(ClientContext &context, SetScope scope, Value &parameter) {
	if (scope == SetScope::SESSION) {
		// One cache per loaded model serves every connection, so one value bounds it; a per-connection
		// value would let whichever connection called last evict everyone else's contexts.
		throw InvalidInputException("anofox_tabfm_context_cache_memory is instance-wide: use SET or SET GLOBAL, "
		                            "not SET SESSION");
	}
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_context_cache_memory cannot be NULL");
	}
	auto value = StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR));
	idx_t bytes;
	auto trimmed = value;
	StringUtil::Trim(trimmed);
	auto error = trimmed == "0" ? string() : StringUtil::TryParseFormattedBytes(value, bytes);
	if (!error.empty()) {
		throw InvalidInputException("anofox_tabfm_context_cache_memory: %s (got '%s'); use a size like '1GB' or "
		                            "'512MB', or '0' to cache nothing",
		                            error, value);
	}
}

} // anonymous namespace

void RegisterTabfmSettings(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	config.AddExtensionOption("anofox_tabfm_accept_hf_license",
	                          "Accept the upstream licence of the gated models (tabfm-v1, the tabpfn-* models, causilo, "
	                          "limix-2m; several are non-commercial, see the license column of tabfm_list_models()). "
	                          "Downloads of those weights fail without this; the other models need no acknowledgement.",
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
	    "anofox_tabfm_max_sessions",
	    "Cap on cached model sessions across all models, devices and precisions (default 4); beyond it the "
	    "oldest-loaded session is evicted. Sessions are cached per (model, device, precision) so switching devices "
	    "does not rebuild multi-GB sessions, and this cap keeps that from accumulating unbounded memory.",
	    LogicalType::BIGINT, Value::BIGINT(4), ValidateMaxSessions);

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
	    "RDNA4, half the VRAM/.mxr; label flips vs fp32 are rare (~2%% measured) but NOT confined to near-ties — "
	    "measured flips include high-confidence rows, so validate bf16 on your own data) and are rejected on CUDA "
	    "rather than silently running fp32.",
	    LogicalType::VARCHAR, Value("fp32"), ValidateGpuPrecision);

	config.AddExtensionOption(
	    "anofox_tabfm_context_cache",
	    "Reuse the encoded labelled context across calls (and across the batches of one call, see opts "
	    "query_batch), for models that ship an exact prepare/query graph pair -- today tabicl-v2, on the CPU. "
	    "On by default: a cached context is keyed by the exact preprocessed support rows and labels, so a hit "
	    "can only ever answer for the same training data, and every output row, context rows included, "
	    "equals the combined graph's. The first call against a new context costs the same as without the "
	    "cache. false is the kill switch: every call runs the combined graph (a session SET applies to statements "
	    "bound after it; SET GLOBAL also stops prepared statements). Models without an exact pair, "
	    "and GPU sessions, always run the combined graph. Memory is bounded by "
	    "anofox_tabfm_context_cache_memory; tabfm_models() shows what is cached.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(true));

	config.AddExtensionOption(
	    "anofox_tabfm_context_cache_memory",
	    "Budget for the context cache (anofox_tabfm_context_cache), per loaded model and instance-wide (SET "
	    "applies to every connection, read when a statement runs), e.g. '256MB' (default) or "
	    "'0' to keep nothing. Least recently used contexts are evicted beyond it, and a context larger than the "
	    "whole budget serves its own call and is dropped. A tabicl-v2 context costs about 2 x 12 x S x 512 floats of attention keys and values plus about "
	    "0.2 MB per feature column: roughly 70 MB at 1000 support rows and 100 features. Cached contexts are "
	    "resident memory, which anofox_tabfm_max_memory counts.",
	    LogicalType::VARCHAR, Value("256MB"), ValidateContextCacheMemory, SetScope::GLOBAL);

	config.AddExtensionOption(
	    "anofox_tabfm_cpu_prepack",
	    "Enable ONNX Runtime weight prepacking on the CPU EP: faster matmuls at ~+16% resident memory.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(true));

	config.AddExtensionOption("anofox_tabfm_device",
	                          "Execution device: auto|cpu|cuda|rocm|coreml|mlx ('migraphx' alias). 'auto' resolves "
	                          "per model: it picks the best device that model can actually be served on (its plugin "
	                          "present and a graph available), else the CPU — so it never promises an accelerator a "
	                          "model cannot use. SELECT * FROM tabfm_backends() shows the choice and the reason. "
	                          "Naming a device explicitly is a hard request: it errors rather than falling back. "
	                          "coreml is flavor-gated and 'auto' does not reach for it.",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);

	config.AddExtensionOption("anofox_tabfm_ep_path",
	                          "Directory holding the GPU backend plugins (anofox_tabfm_{cuda,migraphx,mlx}_plugin, "
	                          "with this platform's library prefix and extension) and the runtime libraries they "
	                          "load alongside themselves. Defaults to the 'runtime' subdirectory of "
	                          "anofox_tabfm_cache_dir, which is where CALL tabfm_download_runtime(...) and CALL "
	                          "tabfm_accelerate() put them — set this only to point somewhere else.",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption(
	    "anofox_tabfm_mxr_source",
	    "Directory holding precompiled MIGraphX .mxr programs (offline/CI/shared cache). Before compiling a "
	    "shape-bucket (~27 min on ROCm), a matching '<graph-stem>_<content-hash8>_<arch>_<precision>_T<t>_H<h>.mxr' here is staged into the "
	    "cache and reused; empty ('' default) always compiles on-device. Artifacts are arch- and ROCm-version-specific.",
	    LogicalType::VARCHAR, Value(""));
}

} // namespace anofox
} // namespace duckdb
