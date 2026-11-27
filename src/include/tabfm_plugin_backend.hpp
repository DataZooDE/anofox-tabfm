#pragma once

//===----------------------------------------------------------------------===//
// tabfm_plugin_backend — a GPU backend served by a plugin from anofox-ggml-adapter (plugin ABI v3).
//
// Every GPU backend is a library the extension dlopens on demand (the ggml plugin for CUDA and ROCm, MLX for Apple); a GPU is never a
// build. The plugins are model-agnostic, so the TabFM contract lives here: which weights the graph reads (mapped from the checkpoint and
// handed over by name), which inputs a TabFM graph takes and how they are fed (the y prefix for a graph without train_size), and which
// output is the logits. The adapter's loader reports every failure by kind; this file words it with tabfm's remedies.
//===----------------------------------------------------------------------===//

#include "tabfm_session.hpp" // TabFMBackend

namespace duckdb {
namespace anofox {

struct TabFMPluginRequest {
	//! The external-data graph (its initializers point into a file in `weights_dir`); empty for mitra's MLX hand-port, which reads
	//! every tensor of `weights_path` instead.
	string graph_path;
	string weights_dir;
	string weights_path; //!< the model's safetensors (used when there is no graph)
	string model;        //!< the plugin's model name: "<model>" or, for MLX, "<model>-<task>" (selects mitra's fast path)
	string arch;
	string precision;
	string cache_dir;
	int device_ordinal = 0;
	string out_of_memory_remedy;
};

//! Load the plugin at `library_path` and open a session for `request`. Throws a DuckDB exception naming the fix.
unique_ptr<TabFMBackend> LoadPluginBackend(const string &library_path, const TabFMPluginRequest &request);

//! True when the library loads and speaks the plugin ABI this build was made with; otherwise `*error` says why.
bool PluginLoadable(const string &library_path, string *error = nullptr);

//! The inputs a TabFM graph may declare (each fed by LoadPluginBackend's backend).
const vector<string> &TabFMPluginInputs();

} // namespace anofox
} // namespace duckdb
