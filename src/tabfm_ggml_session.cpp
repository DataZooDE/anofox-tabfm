#include "tabfm_ggml_session.hpp"

#include "tabfm_ggml.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace duckdb {
namespace anofox {

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

//! A failure inside the ggml engine, in the engine's voice: the SQL-visible message names the engine.
[[noreturn]] void ThrowInference(const std::exception &error, const char *what) {
	throw InvalidInputException(string("anofox_tabfm: ") + what + " failed in the ggml CPU engine: " + error.what());
}

GgmlRunOptions BaseOptions(const GgmlSessionState &state) {
	GgmlRunOptions options;
	options.threads = static_cast<int>(std::max<int64_t>(1, state.config.threads));
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

std::unique_ptr<GgmlSessionState> CreateGgmlSessionState(::anofox::onnxread::Graph graph, const vector<TabFMTensorRef> &initializers,
                                                         const TabFMSessionConfig &config) {
	const string cpu_reason = UnsupportedCpuReason();
	if (!cpu_reason.empty()) {
		throw InvalidInputException("anofox_tabfm: the ggml CPU engine cannot run on this machine: " + cpu_reason +
		                            ". Run on a machine with those CPU instructions, or SET anofox_tabfm_device to a GPU (with its backend plugin in anofox_tabfm_ep_path).");
	}
	auto state = std::make_unique<GgmlSessionState>();
	state->config = config;
	const string task = Task(config);

	// The engine either runs every op of the graph or says which it cannot.
	std::vector<std::string> ops;
	for (const auto &n : graph.nodes) {
		ops.push_back(n.op_type);
	}
	const string refusal = GgmlRefusal(ops, "Please report these op names; a GPU backend plugin (SET anofox_tabfm_ep_path, SET anofox_tabfm_device) may still run the graph.");
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
	executor_config.threads = static_cast<int>(std::max<int64_t>(1, config.threads));
	executor_config.weights = MakeLookup(*state);
	state->executor = std::make_unique<GgmlExecutor>(state->graph, std::move(executor_config));
	return state;
}

TabFMRunOutput GgmlRun(const GgmlSessionState &state, const TabFMRunInput &input) {
	// The graph contract: a graph without a train_size input derives the split from the length of y, so it gets the training prefix.
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


} // namespace anofox
} // namespace duckdb
