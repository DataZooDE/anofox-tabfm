// tabfm_registry.cpp — the model registry (P2). Single source of truth for the
// set of models: the built-ins compiled into the binary + user manifests from
// anofox_tabfm_model_manifest (file or dir). Schema: tabfm_registry.hpp.

#include "tabfm_registry.hpp"
#include "tabfm_manifest.hpp" // BuiltinTabFMManifestJson

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <set>

namespace duckdb {
namespace anofox {

vector<ModelSpec> BuiltinModelSpecs() {
	// Merge the two per-task built-in TabFM v1 manifests into one multi-task
	// model spec (id "tabfm-v1"). Each task keeps its own graph/tensor-map/repo.
	auto spec = ParseModelSpec(BuiltinTabFMManifestJson(TabFMTask::CLASSIFICATION),
	                           "(built-in tabfm-v1 classification)");
	auto reg_spec =
	    ParseModelSpec(BuiltinTabFMManifestJson(TabFMTask::REGRESSION), "(built-in tabfm-v1 regression)");
	spec.tasks.emplace(TabFMTask::REGRESSION, reg_spec.tasks.at(TabFMTask::REGRESSION));
	spec.capabilities = {"classify", "regress"};
	spec.display_name = "Google TabFM v1";
	spec.family = "icl-transformer";
	// Non-commercial weights → the license gate fires (unchanged behavior).
	spec.license.commercial = false;
	spec.license.redistributable = false;
	spec.license.gate_setting = "accept_hf_license";
	return {std::move(spec)};
}

string ModelRegistry::RegisteredIds() const {
	string out; // models_ is a std::map → keys already sorted
	for (auto &kv : models_) {
		if (!out.empty()) {
			out += ", ";
		}
		out += kv.first;
	}
	return out;
}

const ModelSpec &ModelRegistry::Get(const string &id) const {
	auto it = models_.find(id);
	if (it == models_.end()) {
		throw InvalidInputException("tabfm: unknown model '%s'. Registered: %s. See tabfm_list_models().", id,
		                            RegisteredIds());
	}
	return it->second;
}

const ModelSpec &ModelRegistry::Resolve(const string &requested, const string &default_setting) const {
	if (!requested.empty()) {
		return Get(requested);
	}
	if (!default_setting.empty()) {
		return Get(default_setting);
	}
	if (!implicit_default_.empty()) {
		return Get(implicit_default_);
	}
	if (models_.size() == 1) {
		return models_.begin()->second;
	}
	throw InvalidInputException(
	    "tabfm: %llu models are registered (%s) and none is selected. Choose one with "
	    "model := '<id>' or SET anofox_tabfm_default_model = '<id>'. See tabfm_list_models().",
	    static_cast<unsigned long long>(models_.size()), RegisteredIds());
}

ModelRegistry ModelRegistry::Build(const string &manifest_source) {
	ModelRegistry registry;
	for (auto &spec : BuiltinModelSpecs()) {
		registry.models_[spec.id] = std::move(spec);
	}
	if (manifest_source.empty()) {
		return registry;
	}
	auto fs = FileSystem::CreateLocal();
	if (fs->DirectoryExists(manifest_source)) {
		// A directory of manifests: register every *.json (merged, no implicit
		// default). Sort the file list so loading is deterministic, and reject two
		// user manifests that declare the same id (a user file MAY shadow a
		// built-in, but two user files with one id is nondeterministic → error).
		vector<string> names;
		fs->ListFiles(manifest_source, [&](const string &name, bool is_dir) {
			if (!is_dir && StringUtil::EndsWith(StringUtil::Lower(name), ".json")) {
				names.push_back(name);
			}
		});
		std::sort(names.begin(), names.end());
		std::set<string> user_ids;
		for (auto &name : names) {
			auto spec = LoadModelSpecFile(fs->JoinPath(manifest_source, name));
			if (!user_ids.insert(spec.id).second) {
				throw InvalidInputException(
				    "tabfm: two manifests in directory '%s' both declare model id '%s' — ids must be unique",
				    manifest_source, spec.id);
			}
			registry.models_[spec.id] = std::move(spec);
		}
	} else if (fs->FileExists(manifest_source)) {
		// A single file selects that model as the active default (back-compat).
		auto spec = LoadModelSpecFile(manifest_source);
		registry.implicit_default_ = spec.id;
		registry.models_[spec.id] = std::move(spec);
	} else {
		throw IOException(
		    "tabfm: anofox_tabfm_model_manifest '%s' is neither a readable file nor a directory", manifest_source);
	}
	return registry;
}

} // namespace anofox
} // namespace duckdb
