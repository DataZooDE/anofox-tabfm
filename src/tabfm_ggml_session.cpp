#include "tabfm_ggml_session.hpp"

#include "tabfm_ggml_engine.hpp"
#include "tabfm_ggml_lower.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#if defined(_MSC_VER) && (defined(_M_X64))
#include <intrin.h>
#endif

namespace duckdb {
namespace anofox {

#ifdef TABFM_HAVE_GGML

namespace {

size_t ElementCount(const std::vector<int64_t> &dims) {
	size_t n = 1;
	for (auto d : dims) {
		n *= static_cast<size_t>(d);
	}
	return n;
}

bool IsExternal(const ::anofox::onnxread::Tensor &t) {
	return t.location == ::anofox::onnxread::DataLocation::EXTERNAL ||
	       (t.raw.empty() && t.floats.empty() && t.int64s.empty() && t.int32s.empty() && ElementCount(t.dims) > 0);
}

string Task(const TabFMSessionConfig &config) {
	return config.model_tag.empty() ? string("classification") : config.model_tag;
}

WeightLookup MakeLookup(const GgmlSessionState &state) {
	return [&state](const string &name, const std::vector<int64_t> &dims) -> ExternalWeight {
		const auto it = state.weights.find(name);
		if (it == state.weights.end() || it->second.dtype != TabFMTensorDtype::F32 ||
		    it->second.shape != vector<int64_t>(dims.begin(), dims.end())) {
			return ExternalWeight();
		}
		return {static_cast<const float *>(it->second.data), static_cast<size_t>(it->second.size_bytes / sizeof(float))};
	};
}

//! A failure inside the ggml engine, in the engine's voice: the SQL-visible message names the engine and the build option that returns to ORT.
[[noreturn]] void ThrowInference(const std::exception &error, const char *what) {
	throw InvalidInputException(string("anofox_tabfm: ") + what + " failed in the ggml CPU engine: " + error.what());
}

GgmlRunOptions BaseOptions(const GgmlSessionState &state) {
	GgmlRunOptions options;
	options.threads = static_cast<int>(std::max<int64_t>(1, state.config.intra_op_threads));
	options.weights = MakeLookup(state);
	return options;
}

HostTensor Copy(const float *data, std::vector<int64_t> shape) {
	HostTensor t;
	t.shape = std::move(shape);
	t.data.assign(data, data + ElementCount(t.shape));
	return t;
}

bool HasInput(const GgmlSessionState &state, const string &name) {
	return std::find(state.input_names.begin(), state.input_names.end(), name) != state.input_names.end();
}

string PickOutput(const GgmlSessionState &state) {
	for (auto &o : state.output_names) {
		if (o == "logits") {
			return o;
		}
	}
	return state.output_names.at(0);
}

TabFMRunOutput ToRunOutput(HostTensors &outputs, const string &name) {
	auto it = outputs.find(name);
	if (it == outputs.end()) {
		throw InternalException("anofox_tabfm: the ggml engine did not return output '" + name + "'");
	}
	TabFMRunOutput result;
	result.shape = vector<int64_t>(it->second.shape.begin(), it->second.shape.end());
	result.logits = std::move(it->second.data);
	return result;
}

} // namespace

//! The ggml CPU backend is compiled for AVX2 + FMA + F16C + BMI2 on x86 (no runtime dispatch in a static build): on an older CPU its first matmul would
//! kill the whole DuckDB process with SIGILL. Refuse instead, naming the way out. Empty when the CPU is fine (always on other architectures).
static string UnsupportedCpuReason() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
	int regs[4];
	__cpuid(regs, 0);
	if (regs[0] < 7) {
		return "its CPU predates the AVX2 instruction set";
	}
	__cpuidex(regs, 7, 0);
	const bool avx2 = (regs[1] & (1 << 5)) != 0, bmi2 = (regs[1] & (1 << 8)) != 0;
	__cpuid(regs, 1);
	const bool fma = (regs[2] & (1 << 12)) != 0, f16c = (regs[2] & (1 << 29)) != 0;
	if (!(avx2 && bmi2 && fma && f16c)) {
		return "its CPU lacks AVX2, FMA, F16C or BMI2";
	}
#else
	__builtin_cpu_init();
	if (!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("bmi2"))) {
		return "its CPU lacks AVX2, FMA, F16C or BMI2";
	}
#endif
#endif
	return string();
}

std::unique_ptr<GgmlSessionState> CreateGgmlSessionState(::anofox::onnxread::Graph graph, const vector<TabFMTensorRef> &initializers,
                                                         const TabFMSessionConfig &config) {
	const string cpu_reason = UnsupportedCpuReason();
	if (!cpu_reason.empty()) {
		throw InvalidInputException("anofox_tabfm: the ggml CPU engine cannot run on this machine: " + cpu_reason +
		                            ". Rebuild the extension with TABFM_CPU_ENGINE=ort (make release TABFM_CPU_ENGINE=ort) to run on ONNX Runtime instead.");
	}
	auto state = std::make_unique<GgmlSessionState>();
	state->config = config;
	const string task = Task(config);

	// The engine either runs every op of the graph or says which it cannot (and how to run the model through ONNX Runtime instead).
	std::vector<std::string> ops;
	for (const auto &n : graph.nodes) {
		ops.push_back(n.op_type);
	}
	const string refusal = GgmlRefusal(ops);
	if (!refusal.empty()) {
		throw InvalidInputException("anofox_tabfm: the ggml CPU engine cannot run the graph of '" + task + "': " + refusal);
	}

	for (const auto &t : initializers) {
		if (!t.data) {
			throw InvalidInputException("anofox_tabfm: initializer '" + t.name + "' has no data buffer");
		}
		state->weights[t.name] = t;
	}
	std::set<string> initializer_names;
	for (const auto &t : graph.initializers) {
		initializer_names.insert(t.name);
		if (!IsExternal(t)) {
			continue;
		}
		const auto it = state->weights.find(t.name);
		if (it == state->weights.end()) {
			throw InvalidInputException("anofox_tabfm: model weights are not loaded — the model graph references external weights that were "
			                            "not injected (first missing: '" + t.name + "'). Run: CALL tabfm_load('" + task +
			                            "'); (downloading them first if needed: CALL tabfm_download('" + task + "');).");
		}
		const auto &w = it->second;
		if (w.dtype != TabFMTensorDtype::F32 || w.shape != vector<int64_t>(t.dims.begin(), t.dims.end()) ||
		    w.size_bytes != ElementCount(w.shape) * sizeof(float)) {
			throw IOException("anofox_tabfm: the checkpoint for '" + task + "' is corrupted or does not match the bundled model graph (tensor '" +
			                  t.name + "') — remove and re-download it: CALL tabfm_remove('" + task + "'); CALL tabfm_download('" + task + "');");
		}
	}
	for (const auto &name : graph.inputs) {
		if (!initializer_names.count(name)) {
			state->input_names.push_back(name);
		}
	}
	state->output_names.assign(graph.outputs.begin(), graph.outputs.end());
	state->graph = std::move(graph);
	GgmlExecutorConfig executor_config;
	executor_config.threads = static_cast<int>(std::max<int64_t>(1, config.intra_op_threads));
	executor_config.weights = MakeLookup(*state);
	state->executor = std::make_unique<GgmlExecutor>(state->graph, std::move(executor_config));
	return state;
}

TabFMRunOutput GgmlRun(const GgmlSessionState &state, const TabFMRunInput &input) {
	// Same contract as the ORT path: a graph without a train_size input derives the split from the length of y, so it gets the training prefix.
	const bool graph_has_train_size = HasInput(state, "train_size");
	const int64_t y_len = graph_has_train_size ? input.t : input.train_size;
	HostTensors in;
	GgmlRunOptions options = BaseOptions(state);
	for (const auto &name : state.input_names) {
		if (name == "x") {
			in["x"] = Copy(input.x, {1, input.t, input.h});
		} else if (name == "y") {
			in["y"] = Copy(input.y, {1, y_len});
		} else if (name == "train_size") {
			options.int_inputs[name] = {{1}, {input.train_size}};
		} else if (name == "d") {
			options.int_inputs[name] = {{1}, {input.d}};
		} else if (name == "cat_mask") {
			HostTensor mask;
			mask.shape = {1, input.h};
			for (int64_t i = 0; i < input.h; i++) {
				mask.data.push_back(input.cat_mask[i] ? 1.0f : 0.0f);
			}
			in[name] = std::move(mask);
			options.bool_inputs.insert(name);
		} else {
			throw InternalException("anofox_tabfm: model graph declares unexpected input '" + name +
			                        "' — the engine feeds x, y, train_size, cat_mask, d (HLD §4.4)");
		}
	}
	if (state.output_names.empty()) {
		throw InvalidInputException("anofox_tabfm: model graph declares no outputs; expected a 'logits' output (HLD §4.4).");
	}
	const string output = PickOutput(state);
	options.outputs = {output};
	try {
		auto outputs = state.executor->Run(in, options);
		return ToRunOutput(outputs, output);
	} catch (const std::exception &error) {
		ThrowInference(error, "inference");
	}
}

TabFMPreparedContext GgmlRunPrepare(const GgmlSessionState &state, const float *x, const float *y, int64_t s, int64_t h,
                                    const vector<string> &wanted) {
	HostTensors in;
	GgmlRunOptions options = BaseOptions(state);
	for (const auto &name : state.input_names) {
		if (name == "x") {
			in["x"] = Copy(x, {1, s, h});
		} else if (name == "y") {
			in["y"] = Copy(y, {1, s});
		} else {
			throw InvalidInputException("anofox_tabfm: the prepare graph declares input '" + name +
			                            "'; the split contract is prepare(x, y) -> the context tensors the query graph takes. This is not the "
			                            "prepare half of a split pair.");
		}
	}
	options.outputs = wanted;
	try {
		auto outputs = state.executor->Run(in, options);
		TabFMPreparedContext context;
		context.context_rows = s;
		for (const auto &name : wanted) {
			auto it = outputs.find(name);
			if (it == outputs.end()) {
				throw InvalidInputException("anofox_tabfm: the prepare graph does not produce '" + name + "', which the query graph needs");
			}
			context.names.push_back(name);
			context.shapes.emplace_back(it->second.shape.begin(), it->second.shape.end());
			context.bytes += it->second.data.size() * sizeof(float);
			context.data.push_back(std::move(it->second.data));
		}
		return context;
	} catch (const InvalidInputException &) {
		throw;
	} catch (const std::exception &error) {
		ThrowInference(error, "encoding the labelled context");
	}
}

TabFMRunOutput GgmlRunQuery(const GgmlSessionState &state, const float *x, int64_t q, int64_t h, const float *y, int64_t s,
                            const TabFMPreparedContext &context) {
	HostTensors in;
	GgmlRunOptions options = BaseOptions(state);
	for (const auto &name : state.input_names) {
		if (name == "x") {
			in["x"] = Copy(x, {1, q, h});
		} else if (name == "y") {
			in["y"] = Copy(y, {1, s});
		} else {
			idx_t found = context.names.size();
			for (idx_t i = 0; i < context.names.size(); i++) {
				if (context.names[i] == name) {
					found = i;
					break;
				}
			}
			if (found == context.names.size()) {
				throw InternalException("anofox_tabfm: the query graph takes a context tensor '" + name + "' that the prepared context does not carry");
			}
			HostTensor t;
			t.shape = context.shapes[found];
			t.data = context.data[found]; // the cached context stays intact; the runner owns its copy for this call
			in[name] = std::move(t);
		}
	}
	if (state.output_names.empty()) {
		throw InvalidInputException("anofox_tabfm: the query graph declares no outputs; expected 'logits'.");
	}
	const string output = PickOutput(state);
	options.outputs = {output};
	try {
		auto outputs = state.executor->Run(in, options);
		return ToRunOutput(outputs, output);
	} catch (const std::exception &error) {
		ThrowInference(error, "inference against the cached context");
	}
}

#endif // TABFM_HAVE_GGML

} // namespace anofox
} // namespace duckdb
