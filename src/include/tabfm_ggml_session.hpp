#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ggml_session.hpp — the ggml CPU engine behind TabFMSession (docs/GGML_PLAN.md, P5).
//
// The engine's backends (OrtBackend, SplitOrtBackend with its context cache) only ever talk to a TabFMSession through CreateSession / Run /
// RunPrepare / RunQuery / SplitContextInputs / SessionHasOutput. A TabFMSession that holds a GgmlSessionState answers those with
// GgmlRunGraph instead of an Ort::Session, so every backend, cache, memory check and SQL surface above it is shared by both engines.
//
// Lifetime: like ORT's AddExternalInitializers, the session keeps REFERENCES to the weight buffers behind TabFMTensorRef (the safetensors mapping
// or the bf16 upcast arena); the owner keeps them alive for the life of the session (OrtBackend's member order already guarantees it).
//===----------------------------------------------------------------------===//

#include "tabfm_ggml_lower.hpp"
#include "tabfm_ort_engine.hpp"
#include "tabfm_onnx_reader.hpp"

#include <memory>
#include <unordered_map>

namespace duckdb {
namespace anofox {

struct GgmlSessionState {
	::anofox::onnxread::Graph graph;
	//! Runtime inputs and outputs of the graph (initializers are not inputs), in graph order.
	vector<string> input_names;
	vector<string> output_names;
	//! The caller's weight buffers by ONNX initializer name (not owned).
	std::unordered_map<string, TabFMTensorRef> weights;
	TabFMSessionConfig config;
	//! Compiles each input-shape signature once and keeps the thread pool; reads `graph` and the weight buffers above. Declared last: it is
	//! destroyed first, before anything it points into.
	std::unique_ptr<GgmlExecutor> executor;
};

//! Validate the graph against what the engine implements and the injected weights against what the graph needs. Throws the same
//! actionable errors as ORT session creation would: missing weights -> the tabfm_load hint, a mismatching tensor -> corrupted checkpoint,
//! an op the engine lacks -> the op names and the way back to ORT.
std::unique_ptr<GgmlSessionState> CreateGgmlSessionState(::anofox::onnxread::Graph graph, const vector<TabFMTensorRef> &initializers,
                                                         const TabFMSessionConfig &config);

TabFMRunOutput GgmlRun(const GgmlSessionState &state, const TabFMRunInput &input);
TabFMPreparedContext GgmlRunPrepare(const GgmlSessionState &state, const float *x, const float *y, int64_t s, int64_t h,
                                    const vector<string> &wanted);
TabFMRunOutput GgmlRunQuery(const GgmlSessionState &state, const float *x, int64_t q, int64_t h, const float *y, int64_t s,
                            const TabFMPreparedContext &context);


} // namespace anofox
} // namespace duckdb
