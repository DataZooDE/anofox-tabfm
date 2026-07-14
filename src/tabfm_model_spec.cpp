// tabfm_model_spec.cpp — parse a v1 OR v2 manifest into a ModelSpec, the
// multi-model registry's unit (P1). Schema contract: tabfm_model_spec.hpp.
//
// v1 (flat task/files/graph/license-string) → a single-task spec.
// v2 (schema_version:2, weights/graph keyed by task, license object,
//     capabilities, tensor_contract, size_regime) → a multi-task spec.
// Existing v1 manifests parse unchanged; the older ModelManifest path is left
// untouched (retired in P2 when the registry adopts ModelSpec).

#include "tabfm_model_spec.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include "yyjson.hpp"

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT

bool ModelSpec::HasCapability(const string &capability) const {
	for (auto &c : capabilities) {
		if (c == capability) {
			return true;
		}
	}
	return false;
}

string ModelSpec::TaskCapability(TabFMTask task) {
	return task == TabFMTask::CLASSIFICATION ? "classify" : "regress";
}

namespace {

struct YyjsonDoc {
	explicit YyjsonDoc(yyjson_doc *doc_p) : doc(doc_p) {
	}
	~YyjsonDoc() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	YyjsonDoc(const YyjsonDoc &) = delete;
	YyjsonDoc &operator=(const YyjsonDoc &) = delete;
	yyjson_doc *doc;
};

[[noreturn]] void Fail(const string &path, const string &msg) {
	throw InvalidInputException("Model manifest %s: %s", path, msg);
}

string ReqStr(yyjson_val *obj, const char *field, const string &path) {
	auto val = yyjson_obj_get(obj, field);
	if (!val || !yyjson_is_str(val)) {
		Fail(path, "missing required string field \"" + string(field) + "\"");
	}
	string s(yyjson_get_str(val), yyjson_get_len(val));
	if (s.empty()) {
		Fail(path, "field \"" + string(field) + "\" must be a non-empty string");
	}
	return s;
}

string OptStr(yyjson_val *obj, const char *field, const string &def, const string &path) {
	auto val = yyjson_obj_get(obj, field);
	if (!val || yyjson_is_null(val)) {
		return def;
	}
	if (!yyjson_is_str(val)) {
		Fail(path, "field \"" + string(field) + "\" must be a string");
	}
	return string(yyjson_get_str(val), yyjson_get_len(val));
}

bool OptBool(yyjson_val *obj, const char *field, bool def) {
	auto val = yyjson_obj_get(obj, field);
	return (val && yyjson_is_bool(val)) ? yyjson_get_bool(val) : def;
}

int64_t OptInt(yyjson_val *obj, const char *field, int64_t def) {
	auto val = yyjson_obj_get(obj, field);
	return (val && yyjson_is_int(val)) ? yyjson_get_sint(val) : def;
}

TabFMTask ParseTaskName(const string &task, const string &path) {
	if (task == "classification") {
		return TabFMTask::CLASSIFICATION;
	}
	if (task == "regression") {
		return TabFMTask::REGRESSION;
	}
	Fail(path, "task must be 'classification' or 'regression', got '" + task + "'");
}

// Parse a "files" array. `bytes` is optional (0 = unknown) — v2 checkpoints may
// omit it; v1's own strict parser (ModelManifest) is unchanged.
vector<ManifestFile> ParseFileArray(yyjson_val *files_val, const string &path) {
	if (!files_val || !yyjson_is_arr(files_val) || yyjson_arr_size(files_val) == 0) {
		Fail(path, "\"files\" must be a non-empty array");
	}
	vector<ManifestFile> files;
	yyjson_arr_iter iter;
	yyjson_arr_iter_init(files_val, &iter);
	yyjson_val *entry;
	while ((entry = yyjson_arr_iter_next(&iter))) {
		if (!yyjson_is_obj(entry)) {
			Fail(path, "every \"files\" entry must be an object");
		}
		ManifestFile file;
		file.path = ReqStr(entry, "path", path);
		file.url = OptStr(entry, "url", "", path);
		auto bytes = OptInt(entry, "bytes", 0);
		if (bytes < 0) {
			Fail(path, "file \"" + file.path + "\": \"bytes\" must be non-negative");
		}
		file.bytes = UnsafeNumericCast<idx_t>(bytes);
		files.push_back(std::move(file));
	}
	return files;
}

// tensor_map: a path string, or an inline {onnx -> st} object. Fills the two
// out-params on `art` (identity when absent).
void ParseTensorMapInto(yyjson_val *map_val, ModelTaskArtifacts &art, const string &path) {
	if (!map_val || yyjson_is_null(map_val)) {
		return;
	}
	if (yyjson_is_str(map_val)) {
		art.tensor_map_path = string(yyjson_get_str(map_val), yyjson_get_len(map_val));
		return;
	}
	if (yyjson_is_obj(map_val)) {
		yyjson_obj_iter iter;
		yyjson_obj_iter_init(map_val, &iter);
		yyjson_val *key;
		while ((key = yyjson_obj_iter_next(&iter))) {
			auto val = yyjson_obj_iter_get_val(key);
			if (yyjson_is_str(val)) {
				art.tensor_map.emplace(string(yyjson_get_str(key), yyjson_get_len(key)),
				                       string(yyjson_get_str(val), yyjson_get_len(val)));
			}
		}
		return;
	}
	Fail(path, "\"tensor_map\" must be a path string or an {onnx -> safetensors} object");
}

// A v2 graph.tensor_map is either SHARED across tasks (a path string, or an
// inline {onnx -> st} object — Mitra) or keyed PER TASK ({classification: <map>,
// regression: <map>}) when the tasks' graphs have different initializers (TabPFN
// clf 29 vs reg 130). Disambiguate on the first key being a task name.
bool IsTaskKeyedTensorMap(yyjson_val *map_val) {
	if (!map_val || !yyjson_is_obj(map_val)) {
		return false;
	}
	yyjson_obj_iter it;
	yyjson_obj_iter_init(map_val, &it);
	yyjson_val *k = yyjson_obj_iter_next(&it);
	if (!k) {
		return false;
	}
	string key(yyjson_get_str(k), yyjson_get_len(k));
	return key == "classification" || key == "regression";
}

void ParseEngineProfilesInto(yyjson_val *profiles_val, map<string, EngineProfile> &out, const string &path) {
	if (profiles_val && yyjson_is_obj(profiles_val)) {
		yyjson_obj_iter iter;
		yyjson_obj_iter_init(profiles_val, &iter);
		yyjson_val *key;
		while ((key = yyjson_obj_iter_next(&iter))) {
			auto val = yyjson_obj_iter_get_val(key);
			if (!yyjson_is_obj(val)) {
				continue;
			}
			auto dtype = OptStr(val, "dtype", "", path);
			if (dtype != "f32" && dtype != "bf16" && dtype != "fp16") {
				Fail(path, "engine profile dtype must be 'f32'|'bf16'|'fp16', got '" + dtype + "'");
			}
			EngineProfile p;
			p.dtype = std::move(dtype);
			out.emplace(string(yyjson_get_str(key), yyjson_get_len(key)), std::move(p));
		}
	}
	if (out.find("cpu") == out.end()) {
		EngineProfile cpu;
		cpu.dtype = "f32"; // HLD D5: CPU stays fp32
		out.emplace("cpu", std::move(cpu));
	}
}

void ParseContractSide(yyjson_val *side, vector<TensorContractEntry> &out, const string &path) {
	if (!side || !yyjson_is_obj(side)) {
		return;
	}
	yyjson_obj_iter iter;
	yyjson_obj_iter_init(side, &iter);
	yyjson_val *key;
	while ((key = yyjson_obj_iter_next(&iter))) {
		auto val = yyjson_obj_iter_get_val(key);
		if (!yyjson_is_obj(val)) {
			continue;
		}
		TensorContractEntry e;
		e.logical = string(yyjson_get_str(key), yyjson_get_len(key));
		e.name = OptStr(val, "name", e.logical, path);
		e.dtype = OptStr(val, "dtype", "f32", path);
		auto shape = yyjson_obj_get(val, "shape");
		if (shape && yyjson_is_arr(shape)) {
			yyjson_arr_iter sit;
			yyjson_arr_iter_init(shape, &sit);
			yyjson_val *dim;
			while ((dim = yyjson_arr_iter_next(&sit))) {
				if (yyjson_is_str(dim)) {
					e.shape.emplace_back(yyjson_get_str(dim), yyjson_get_len(dim));
				} else if (yyjson_is_int(dim)) {
					e.shape.emplace_back(std::to_string(yyjson_get_sint(dim)));
				}
			}
		}
		out.push_back(std::move(e));
	}
}

//===----------------------------------------------------------------------===//
// v1
//===----------------------------------------------------------------------===//

// True when a v1 `license` id needs no acceptance gate: "none"/absent, or a
// well-known permissive license. Case-insensitive on the SPDX-style id.
bool LicenseIsUngated(const string &license_id) {
	if (license_id.empty() || license_id == "none") {
		return true;
	}
	auto id = StringUtil::Lower(license_id);
	static const char *kPermissive[] = {"apache-2.0", "mit",       "bsd-2-clause", "bsd-3-clause", "isc",
	                                    "cc0-1.0",    "cc-by-4.0", "mpl-2.0",      "unlicense",    "zlib"};
	for (auto *p : kPermissive) {
		if (id == p) {
			return true;
		}
	}
	return false;
}

void ParseV1(yyjson_val *root, ModelSpec &spec, const string &path) {
	if (yyjson_obj_get(root, "model")) {
		spec.id = ReqStr(root, "model", path);
	} else if (yyjson_obj_get(root, "model_id")) {
		spec.id = ReqStr(root, "model_id", path);
	} else {
		Fail(path, "missing required field \"model\" (or its alias \"model_id\")");
	}
	spec.family = OptStr(root, "family", "", path);
	// license optional (download-only manifests may omit it). v1 heuristic:
	// "none"/absent is ungated; a well-known PERMISSIVE license (apache-2.0, mit,
	// bsd, …) is commercial/ungated; any other *named* license stays gated behind
	// accept_hf_license (conservative — this preserves the built-in HF weight
	// gate and is safe for unknown restrictive licenses). v2 models declare the
	// license explicitly (object) and never take this path.
	spec.license.id = OptStr(root, "license", "none", path);
	spec.license.commercial = LicenseIsUngated(spec.license.id);
	spec.license.redistributable = spec.license.commercial;
	spec.license.gate_setting = spec.license.commercial ? "" : "accept_hf_license";

	auto task = ParseTaskName(ReqStr(root, "task", path), path);
	ModelTaskArtifacts art;
	art.repo = OptStr(root, "repo", "", path);
	art.revision = OptStr(root, "revision", "main", path);
	art.files = ParseFileArray(yyjson_obj_get(root, "files"), path);
	// graph + preprocessing_profile are optional here (a download/weights-only
	// manifest need not carry them); the engine validates the graph when it needs
	// one (ResolveGraphPath errors clearly if it is missing).
	art.graph = OptStr(root, "graph", "", path);
	ParseTensorMapInto(yyjson_obj_get(root, "tensor_map"), art, path);
	art.preprocessing_profile = OptStr(root, "preprocessing_profile", "", path);
	spec.preprocessing_profile = art.preprocessing_profile;
	spec.tasks.emplace(task, std::move(art));

	spec.capabilities.push_back(ModelSpec::TaskCapability(task));
	ParseEngineProfilesInto(yyjson_obj_get(root, "engine_profiles"), spec.engine_profiles, path);
}

//===----------------------------------------------------------------------===//
// v2
//===----------------------------------------------------------------------===//

void ParseV2(yyjson_val *root, ModelSpec &spec, const string &path) {
	spec.id = ReqStr(root, "id", path);
	spec.display_name = OptStr(root, "display_name", "", path);
	spec.family = OptStr(root, "family", "", path);

	// license object
	auto lic = yyjson_obj_get(root, "license");
	if (!lic || !yyjson_is_obj(lic)) {
		Fail(path, "v2 \"license\" must be an object with an \"id\"");
	}
	spec.license.id = ReqStr(lic, "id", path);
	spec.license.commercial = OptBool(lic, "commercial", false);
	spec.license.redistributable = OptBool(lic, "redistributable", false);
	spec.license.attribution = OptStr(lic, "attribution", "", path);
	spec.license.gate_setting = OptStr(lic, "gate_setting", "", path);

	spec.preprocessing_profile = ReqStr(root, "preprocessing_profile", path);

	// weights: object keyed by task name (at least one)
	auto weights = yyjson_obj_get(root, "weights");
	if (!weights || !yyjson_is_obj(weights) || yyjson_obj_size(weights) == 0) {
		Fail(path, "v2 \"weights\" must be a non-empty object keyed by task");
	}
	auto graph = yyjson_obj_get(root, "graph");
	// The tensor map is either shared across tasks (path/inline object) or keyed
	// per task (graphs with different initializers, e.g. TabPFN clf vs reg).
	yyjson_val *tmap_val = (graph && yyjson_is_obj(graph)) ? yyjson_obj_get(graph, "tensor_map") : nullptr;
	const bool task_keyed_map = IsTaskKeyedTensorMap(tmap_val);
	ModelTaskArtifacts shared_map;
	if (tmap_val && !task_keyed_map) {
		ParseTensorMapInto(tmap_val, shared_map, path);
	}
	yyjson_obj_iter wit;
	yyjson_obj_iter_init(weights, &wit);
	yyjson_val *wkey;
	while ((wkey = yyjson_obj_iter_next(&wit))) {
		string task_name(yyjson_get_str(wkey), yyjson_get_len(wkey));
		auto task = ParseTaskName(task_name, path);
		auto wval = yyjson_obj_iter_get_val(wkey);
		if (!yyjson_is_obj(wval)) {
			Fail(path, "weights[\"" + task_name + "\"] must be an object");
		}
		ModelTaskArtifacts art;
		art.repo = OptStr(wval, "repo", "", path);
		art.revision = OptStr(wval, "revision", "main", path);
		art.files = ParseFileArray(yyjson_obj_get(wval, "files"), path);
		art.preprocessing_profile = spec.preprocessing_profile; // model-wide
		if (graph && yyjson_is_obj(graph)) {
			art.graph = OptStr(graph, task_name.c_str(), "", path);
		}
		if (art.graph.empty()) {
			Fail(path, "v2 \"graph\" has no entry for task '" + task_name + "'");
		}
		if (task_keyed_map) {
			ParseTensorMapInto(yyjson_obj_get(tmap_val, task_name.c_str()), art, path);
		} else {
			art.tensor_map_path = shared_map.tensor_map_path;
			art.tensor_map = shared_map.tensor_map;
		}
		spec.tasks.emplace(task, std::move(art));
	}

	// capabilities (explicit array)
	auto caps = yyjson_obj_get(root, "capabilities");
	if (caps && yyjson_is_arr(caps)) {
		yyjson_arr_iter cit;
		yyjson_arr_iter_init(caps, &cit);
		yyjson_val *c;
		while ((c = yyjson_arr_iter_next(&cit))) {
			if (yyjson_is_str(c)) {
				spec.capabilities.emplace_back(yyjson_get_str(c), yyjson_get_len(c));
			}
		}
	}
	if (spec.capabilities.empty()) {
		for (auto &kv : spec.tasks) {
			spec.capabilities.push_back(ModelSpec::TaskCapability(kv.first));
		}
	}

	// size_regime
	if (auto sr = yyjson_obj_get(root, "size_regime")) {
		spec.size_regime.max_rows = OptInt(sr, "max_rows", -1);
		spec.size_regime.max_features = OptInt(sr, "max_features", -1);
		spec.size_regime.max_classes = OptInt(sr, "max_classes", -1);
	}

	// tensor_contract
	if (auto tc = yyjson_obj_get(root, "tensor_contract")) {
		ParseContractSide(yyjson_obj_get(tc, "inputs"), spec.tensor_contract.inputs, path);
		ParseContractSide(yyjson_obj_get(tc, "outputs"), spec.tensor_contract.outputs, path);
	}

	// compute → engine profiles (cpu dtype); engine_profiles also accepted
	auto compute = yyjson_obj_get(root, "compute");
	if (compute && yyjson_is_obj(compute)) {
		auto cpu = OptStr(compute, "cpu", "f32", path);
		if (cpu == "f32" || cpu == "bf16" || cpu == "fp16") {
			EngineProfile p;
			p.dtype = cpu;
			spec.engine_profiles.emplace("cpu", std::move(p));
		}
	}
	ParseEngineProfilesInto(yyjson_obj_get(root, "engine_profiles"), spec.engine_profiles, path);
}

} // anonymous namespace

ModelSpec ParseModelSpec(const string &json, const string &manifest_path) {
	yyjson_read_err error;
	YyjsonDoc guard(yyjson_read_opts(const_cast<char *>(json.c_str()), json.size(), YYJSON_READ_NOFLAG, nullptr,
	                                 &error));
	if (!guard.doc) {
		throw InvalidInputException("Model manifest %s: malformed JSON: %s (at byte %llu)", manifest_path, error.msg,
		                            static_cast<unsigned long long>(error.pos));
	}
	auto root = yyjson_doc_get_root(guard.doc);
	if (!yyjson_is_obj(root)) {
		Fail(manifest_path, "the top-level JSON value must be an object");
	}

	ModelSpec spec;
	spec.schema_version = static_cast<int>(OptInt(root, "schema_version", 1));
	if (spec.schema_version >= 2) {
		ParseV2(root, spec, manifest_path);
	} else {
		ParseV1(root, spec, manifest_path);
	}

	if (spec.tasks.empty()) {
		Fail(manifest_path, "model '" + spec.id + "' declares no task checkpoints");
	}
	return spec;
}

ModelSpec LoadModelSpecFile(const string &path) {
	auto fs = FileSystem::CreateLocal();
	if (!fs->FileExists(path)) {
		throw IOException("Model manifest file \"%s\" does not exist or is not readable", path);
	}
	auto handle = fs->OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = UnsafeNumericCast<idx_t>(handle->GetFileSize());
	string json(size, '\0');
	handle->Read(const_cast<char *>(json.data()), size);
	auto spec = ParseModelSpec(json, path);
	// Relative graph/tensor-map paths in this manifest resolve against its dir.
	auto slash = path.find_last_of("/\\");
	spec.source_dir = (slash == string::npos) ? string(".") : path.substr(0, slash);
	return spec;
}

} // namespace anofox
} // namespace duckdb
