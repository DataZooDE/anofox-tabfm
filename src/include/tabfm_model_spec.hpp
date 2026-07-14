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

//! Parse + strictly validate a v1 or v2 manifest into a ModelSpec.
ModelSpec ParseModelSpec(const string &json, const string &manifest_path = "(inline manifest)");

//! Read `path` and parse it (IOException if unreadable).
ModelSpec LoadModelSpecFile(const string &path);

} // namespace anofox
} // namespace duckdb
