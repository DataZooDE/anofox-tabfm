//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_model_spec.hpp — the multi-model registry's unit of description
// (multi-model design, FR-5.1 / M4). A `ModelSpec` is one *model* — possibly
// several task-specialized checkpoints — parsed from a manifest.
//
// Supersedes the per-(model,task) `ModelManifest` for registry/selection while
// reusing its `ManifestFile`/`EngineProfile`/`TabFMTask`. `ParseModelSpec`
// accepts BOTH the v1 manifest (flat `task`/`files`/`graph`/`license`-string)
// and the v2 manifest (`schema_version:2`, `weights`/`graph` keyed by task,
// `license` object, `capabilities`, `tensor_contract`, `size_regime`). v1 →
// a single-task spec; v2 → a multi-task spec. All existing v1 manifests parse
// unchanged.
//===----------------------------------------------------------------------===//

#pragma once

#include "tabfm_manifest.hpp" // ManifestFile, EngineProfile, TabFMTask

#include "duckdb/common/common.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

#include <utility>

namespace duckdb {
namespace anofox {

//! License block. v2 supplies it as an object; a v1 license *string* maps to
//! `{id=<string>, commercial=false, redistributable=false, gate_setting=""}`.
struct ModelLicense {
	string id;
	//! May the weights be used commercially? Drives tabfm_list_models() and the
	//! gate. v1 defaults to false (conservative — the only v1 model is gated).
	bool commercial = false;
	bool redistributable = false;
	//! e.g. "Built with PriorLabs-TabPFN"; empty = none.
	string attribution;
	//! When non-empty (e.g. "accept_hf_license") the license gate fires for this
	//! model. Empty = no gate (commercial-clean).
	string gate_setting;
};

//! The downloadable + graph artifacts for one task within a model.
struct ModelTaskArtifacts {
	string repo;
	string revision = "main";
	vector<ManifestFile> files;
	//! Bundled graph id or an on-disk .onnx path (resolved relative to manifest).
	string graph;
	//! Optional GPU-format graphs (docs/GPU_HARDENING_PLAN.md P3). Two fields,
	//! not one, and deliberately no cross-fallback: spike U1 showed MIGraphX
	//! cannot run a plain external-data ONNX (the fixture graph fails at eval),
	//! so pointing ROCm at an ext_graph would trade a clear "no GPU graph for
	//! this model" for a runtime MIGraphX error. A model whose single export
	//! happens to satisfy both simply names the same file twice.
	//!
	//! ext_graph: external-data ONNX for the ORT-based CUDA plugin; its data
	//! file(s) sit beside it, and its offsets match the model's own weights by
	//! construction, so no header check applies.
	string ext_graph;
	//! migraphx_graph: the MIGraphX-compatible variant (Shape-rewritten,
	//! external-data) for the ROCm plugin.
	string migraphx_graph;
	//! Exactly one of the two tensor-map forms is populated (or neither).
	string tensor_map_path;
	unordered_map<string, string> tensor_map;
	//! Model-wide in v2 (mirrored into each task); per-task in v1.
	string preprocessing_profile;
};

//! One declared graph tensor (name/shape/dtype). `shape` holds symbolic dims as
//! strings ("1","T","H") so the generic engine can bind by name (P4).
struct TensorContractEntry {
	string logical;         // "x"/"y"/"train_size"/"cat_mask"/"d"/"logits"
	string name;            // the ONNX graph tensor name
	vector<string> shape;   // symbolic or numeric dim strings
	string dtype;           // "f32"|"i64"|"bool"|...
};

//! The manifest's declared input/output tensor contract. Empty ⇒ the engine
//! falls back to the built-in TabFM contract (x/y/train_size/cat_mask/d).
struct TensorContract {
	vector<TensorContractEntry> inputs;
	vector<TensorContractEntry> outputs;
	bool empty() const {
		return inputs.empty() && outputs.empty();
	}
};

//! Per-model guardrails; -1 = unset (fall back to session settings).
struct SizeRegime {
	int64_t max_rows = -1;
	int64_t max_features = -1;
	int64_t max_classes = -1;
};

//! A full model description — the registry's keyed unit.
struct ModelSpec {
	int schema_version = 1;
	//! Addressable via `model := '<id>'`.
	string id;
	string display_name;
	//! "icl-transformer" | "retrieval-icl" | ... ; selects the backend later.
	string family;
	ModelLicense license;
	//! One entry (v1) or several (v2), keyed by task.
	map<TabFMTask, ModelTaskArtifacts> tasks;
	//! "classify" | "regress" | "impute" | ... ; v2 explicit, v1 inferred.
	vector<string> capabilities;
	SizeRegime size_regime;
	//! Empty ⇒ built-in TabFM contract (P4).
	TensorContract tensor_contract;
	//! device → dtype; "cpu"→"f32" always present. From `engine_profiles` (v1)
	//! or `compute` (v2).
	map<string, EngineProfile> engine_profiles;
	//! Model-wide preprocessing profile (also mirrored into each task's entry).
	string preprocessing_profile;
	//! Directory the manifest was loaded from (relative graph/tensor-map paths
	//! resolve against it). Empty for built-ins / inline manifests.
	string source_dir;

	bool HasTask(TabFMTask task) const {
		return tasks.find(task) != tasks.end();
	}
	bool HasCapability(const string &capability) const;
	//! The capability string the task maps to ("classification"→"classify").
	static string TaskCapability(TabFMTask task);
};

//! Which graph a GPU backend should run, if any (docs/GPU_HARDENING_PLAN.md
//! P3). Pure so CI can pin the gate without hardware — the same treatment
//! CanReuseSession got after the device-switch bug, and for the same reason:
//! this exact gate is what silently locked every model but tabfm-v1 out of
//! the GPUs.
//!
//! A model-provided graph wins unconditionally: its external-data offsets
//! match its own weights by construction, so the bundled-graph header check
//! does not apply to it. The bundled graph (tabfm-v1's) is only usable when
//! it exists AND the downloaded weights match the header its offsets were
//! baked against.
enum class GpuGraphSource : uint8_t { NONE, MODEL_PROVIDED, BUNDLED };

inline GpuGraphSource SelectGpuGraph(bool model_provides_graph, bool bundled_graph_exists,
                                     bool bundled_header_matches) {
	if (model_provides_graph) {
		return GpuGraphSource::MODEL_PROVIDED;
	}
	if (bundled_graph_exists && bundled_header_matches) {
		return GpuGraphSource::BUNDLED;
	}
	return GpuGraphSource::NONE;
}

//! True when the raw anofox_tabfm_device setting explicitly names this GPU
//! backend ('migraphx' is the documented alias for 'rocm'). 'auto' is never
//! explicit: on a missing GPU graph an explicit request must hard-error while
//! 'auto' quietly falls back to CPU — that asymmetry is the contract.
inline bool IsExplicitGpuRequest(const string &device_setting, const string &backend) {
	if (device_setting == backend) {
		return true;
	}
	return backend == "rocm" && device_setting == "migraphx";
}

//! The error for "you asked for this GPU but the model ships no graph it can
//! run". Found on hardware (examples on a pod): the previous path surfaced
//! "SET anofox_tabfm_ep_path" — the one thing already configured. graph_kind
//! is "ext_graph" (CUDA) or "migraphx_graph" (ROCm).
inline string NoGpuGraphMessage(const string &device, const string &model, const string &task_name,
                                const string &graph_kind) {
	return "anofox_tabfm: device '" + device + "' was requested, but model '" + model +
	       "' has no " + device + "-servable graph for task '" + task_name +
	       "': it declares no " + graph_kind +
	       " and the bundled GPU graph does not match its weights. Register the model with " + task_name + "_" +
	       graph_kind + " := '<graph.onnx>' (CALL tabfm_register_model), or SET anofox_tabfm_device='cpu'.";
}

//! Embedded-resource id of a bundled GPU graph (kind = "ext" | "migraphx").
//! tabfm-v1 keeps its unqualified pre-multi-model ids; every other model is
//! model-qualified so the same task can bundle one graph per model.
inline string BundledGpuGraphId(const string &model, const string &kind, const string &task_name) {
	if (model == "tabfm-v1") {
		return "graph_" + kind + "_" + task_name;
	}
	// Registry ids and resource-file stems differ for the catalog models; this
	// function owns the mapping so the engine never string-mangles.
	string stem = model;
	if (model == "tabpfn-v2") {
		stem = "tabpfn";
	} else if (model == "tabpfn-v2-5") {
		stem = "tabpfn25";
	} else if (model == "tabpfn-v3") {
		stem = "tabpfn3";
	} else if (model == "tabicl-v2") {
		stem = "tabicl";
	} else if (model == "orion-bix") {
		stem = "orion_bix";
	}
	return "graph_" + kind + "_" + stem + "_" + task_name;
}

//! SHA-256 (hex) of the safetensors JSON header each bundled external-data /
//! migraphx graph pair was generated against (tools/make_external_graph.py and
//! tools/make_migraphx_graph.py print it). A byte-identical header guarantees
//! the graphs' baked offsets index the downloaded weights correctly; empty
//! means "this model bundles no GPU graphs" and the caller falls back.
inline string ExpectedWeightsHeaderShaFor(const string &model, const string &task_name) {
	if (model == "tabfm-v1") {
		if (task_name == "classification") {
			return "534d6d38b49b323bb38682858f232573c254689df03d3d9f17e7504716a31d96";
		}
		if (task_name == "regression") {
			return "35c346e4e29f61b493a9e601e66bf0ae241d0fb76623a3336c61408cfc3e88d0";
		}
	}
	if (model == "mitra") {
		if (task_name == "classification") {
			return "cb1a261bb3d9ca505e0db66e21df85bec6777f7e104247d4a71a8eb8d8b3b96a";
		}
		if (task_name == "regression") {
			return "44f9293fa2d81ccf56dffab09600ec4f775c5c30bbc2af551cf4f6b2cf889f01";
		}
	}
	// The single_eval_pos family (x, y inputs only): ext graphs bundle for the
	// CUDA + CPU-low-memory paths; no migraphx variants — MIGraphX compiles
	// per fixed shape, and these models read the train/test split from y's
	// length, so bucketed compiles would need one per distinct train_size.
	if (model == "tabpfn-v2") {
		if (task_name == "classification") {
			return "e97043c10b4572d6011cb1e389db2c7d57425213c761288f935188e25e953362";
		}
		if (task_name == "regression") {
			return "ca4435a405cbd17afc8cf08346954705d7cc01fc6d08bdcf9c06363faed867a9";
		}
	}
	if (model == "tabicl-v2") {
		if (task_name == "classification") {
			return "085731de6a7b33e6fcbda7e1b3cba725d798d30f93ec3cdc91c3ac2c2c762d3f";
		}
		if (task_name == "regression") {
			return "d792dd9433bdf78773eddcd4bda3e0e49550aec0a2df1a0bfa36afebf320e8ae";
		}
	}
	if (model == "orion-bix" && task_name == "classification") {
		return "c2b7ff39add2b0c1c2d3ddabbaf413e8c15f433b620f3091f0562f376255d166";
	}
	if (model == "tabpfn-v2-5") {
		if (task_name == "classification") {
			return "b230477af81d4ac5bff856b2f9dcc281d5b9a04d659a5dee335553f0f49897ea";
		}
		if (task_name == "regression") {
			return "8865ee281d0172e31e1a03d1d43057ac8e69b88a27b2ce0a93ee77b865f45737";
		}
	}
	if (model == "tabpfn-v3") {
		if (task_name == "classification") {
			return "c0f3a23322d1ec039356b618565e5e1c62378613e3081167881220968933f04b";
		}
		if (task_name == "regression") {
			return "92cf58d84bf968d6e5de90f7ccc29c0d1e79e8f0e9a04c7e0ab66274185d061c";
		}
	}
	return "";
}

//! Parse + strictly validate a v1 or v2 manifest into a ModelSpec.
ModelSpec ParseModelSpec(const string &json, const string &manifest_path = "(inline manifest)");

//! Read `path` and parse it (IOException if unreadable).
ModelSpec LoadModelSpecFile(const string &path);

} // namespace anofox
} // namespace duckdb
