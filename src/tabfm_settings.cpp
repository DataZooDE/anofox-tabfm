#include "tabfm_registration.hpp"
#include "tabfm_cpu_budget.hpp"
#include "tabfm_registry.hpp"

#include "duckdb/main/config.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace anofox {

namespace {

//! The gated models are read from the registry, not listed by hand: a hand-written list went stale
//! as soon as new gated models were added.
string AcceptHfLicenseDescription() {
	vector<string> gated;
	for (auto &spec : BuiltinModelSpecs()) {
		if (spec.license.gate_setting == "accept_hf_license") {
			gated.push_back(spec.id);
		}
	}
	return "Accept the upstream licence of the gated models (" + StringUtil::Join(gated, ", ") +
	       "; several are non-commercial, see the license and commercial columns of tabfm_list_models()). "
	       "Downloads of those weights fail without this; the other models need no acknowledgement.";
}

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

void ValidateContextCacheAdmit(ClientContext &context, SetScope scope, Value &parameter) {
	if (scope == SetScope::SESSION) {
		// Same reason as the budget: one cache per loaded model serves every connection.
		throw InvalidInputException("anofox_tabfm_context_cache_admit is instance-wide: use SET or SET GLOBAL, "
		                            "not SET SESSION");
	}
	if (parameter.IsNull()) {
		throw InvalidInputException("anofox_tabfm_context_cache_admit cannot be NULL; use 'second' or 'first'");
	}
	auto value = StringUtil::Lower(StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR)));
	StringUtil::Trim(value);
	if (value != "first" && value != "second") {
		throw InvalidInputException("anofox_tabfm_context_cache_admit must be 'second' (keep a context the second "
		                            "time it is seen) or 'first' (keep it at once), got '%s'",
		                            StringValue::Get(parameter.DefaultCastAs(LogicalType::VARCHAR)));
	}
	parameter = Value(value);
}

} // anonymous namespace

void RegisterTabfmSettings(ExtensionLoader &loader) {
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());

	config.AddExtensionOption("anofox_tabfm_accept_hf_license", AcceptHfLicenseDescription(), LogicalType::BOOLEAN,
	                          Value::BOOLEAN(false));

	config.AddExtensionOption("anofox_tabfm_cache_dir",
	                          "Weight cache root directory (default ~/.cache/anofox-tabfm)", LogicalType::VARCHAR,
	                          Value("~/.cache/anofox-tabfm"));

	const auto default_threads = MaxValue<int64_t>(1, static_cast<int64_t>(UsableCoreCount()) / 2);
	config.AddExtensionOption("anofox_tabfm_threads", "Thread count for CPU inference (the ggml engine's worker threads)",
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
	    "GPU numeric mode: fp32|tf32|bf16|fp16. fp32 (default) is strict: the ggml GPU plugin (cuda, rocm) computes "
	    "in float32 with tensor-core TF32 rounding disabled, so a device switch does not change answers beyond float "
	    "rounding. tf32 allows that rounding on CUDA (faster matmuls). bf16/fp16 apply to the mlx device only; on "
	    "cuda and rocm they are refused rather than silently run as fp32.",
	    LogicalType::VARCHAR, Value("fp32"), ValidateGpuPrecision);

	config.AddExtensionOption(
	    "anofox_tabfm_context_cache",
	    "Reuse the encoded labelled context across calls (and across the batches of one call, see opts "
	    "query_batch), for models that ship an exact prepare/query graph pair -- today tabicl-v2, on the CPU. "
	    "On by default: a cached context is keyed by the exact preprocessed support rows and labels, in "
	    "order, so a hit can only ever answer for the same training data, and every output row, context "
	    "rows included, equals the combined graph's. A context is kept the second time it is seen "
	    "(anofox_tabfm_context_cache_admit); until then a call costs the same as without the cache. "
	    "false is the kill switch: every call runs the combined graph (a session SET applies to statements "
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
	    "anofox_tabfm_context_cache_admit",
	    "When the context cache (anofox_tabfm_context_cache) keeps an encoded context: 'second' (default) "
	    "keeps it the second time its support rows are seen within the last 4096 contexts, so single-use "
	    "contexts (one per group of a GROUP BY, say) never take memory; the first call encodes and answers "
	    "the same, and the third is the first warm one. 'first' keeps every context at once (the cache's "
	    "behaviour before this setting): one warm call sooner for a context you know you will reuse, at "
	    "the cost of filling the budget with contexts used once. Instance-wide, like "
	    "anofox_tabfm_context_cache_memory (SET SESSION is refused); read when a statement runs.",
	    LogicalType::VARCHAR, Value("second"), ValidateContextCacheAdmit, SetScope::GLOBAL);

	config.AddExtensionOption(
	    "anofox_tabfm_cpu_prepack",
	    "Deprecated, no effect: it switched ONNX Runtime's weight prepacking, and no build contains ONNX Runtime any more. "
	    "The CPU engine (ggml) reads weights in place. Accepted so existing scripts keep working.",
	    LogicalType::BOOLEAN, Value::BOOLEAN(true));

	config.AddExtensionOption("anofox_tabfm_device",
	                          "Execution device: auto|cpu|cuda|rocm|mlx ('migraphx' alias for rocm). 'auto' resolves "
	                          "per model: it picks the best device that model can actually be served on (its plugin "
	                          "present and a graph available), else the CPU — so it never promises an accelerator a "
	                          "model cannot use. SELECT * FROM tabfm_backends() shows the choice and the reason. "
	                          "Naming a device explicitly is a hard request: it errors rather than falling back. "
	                          "cuda and rocm are served by the ggml GPU plugin, mlx by the MLX plugin. 'coreml' is "
	                          "accepted but deprecated: using it errors and points to 'mlx'.",
	                          LogicalType::VARCHAR, Value("auto"), ValidateDevice);

	config.AddExtensionOption("anofox_tabfm_ep_path",
	                          "Directory holding the GPU backend plugins (anofox_tabfm_{ggml-cuda,ggml-rocm,mlx}_plugin, "
	                          "with this platform's library prefix and extension). Defaults to the 'runtime' subdirectory of "
	                          "anofox_tabfm_cache_dir, which is where CALL tabfm_download_runtime(...) and CALL "
	                          "tabfm_accelerate() put them — set this only to point somewhere else.",
	                          LogicalType::VARCHAR, Value(""));

	config.AddExtensionOption(
	    "anofox_tabfm_mxr_source",
	    "Deprecated, no effect: it pointed at precompiled MIGraphX .mxr programs, and the ROCm lane (the ggml GPU plugin) "
	    "compiles nothing ahead of time. Accepted so existing scripts keep working.",
	    LogicalType::VARCHAR, Value(""));
}

} // namespace anofox
} // namespace duckdb
