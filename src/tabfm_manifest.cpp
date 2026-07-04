// tabfm_manifest.cpp — model manifest loader + built-in TabFM v1 manifests
// (WS-B, HLD §2 D7). Schema contract documented in tabfm_manifest.hpp.
//
// JSON parsing uses DuckDB's bundled yyjson (namespace duckdb_yyjson).

#include "tabfm_manifest.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/numeric_utils.hpp"

#include "yyjson.hpp"

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT

const char *TabFMTaskName(TabFMTask task) {
	switch (task) {
	case TabFMTask::CLASSIFICATION:
		return "classification";
	case TabFMTask::REGRESSION:
		return "regression";
	default:
		throw InternalException("unknown TabFMTask");
	}
}

namespace {

//! RAII owner for a yyjson document (see S02 friction log #3: parsed JSON
//! must be bound to a named owner for as long as values are dereferenced).
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

string GetRequiredString(yyjson_val *obj, const char *field, const string &manifest_path) {
	auto val = yyjson_obj_get(obj, field);
	if (!val) {
		throw InvalidInputException("Model manifest %s: missing required field \"%s\"", manifest_path, field);
	}
	if (!yyjson_is_str(val)) {
		throw InvalidInputException("Model manifest %s: field \"%s\" must be a string", manifest_path, field);
	}
	string result(yyjson_get_str(val), yyjson_get_len(val));
	if (result.empty()) {
		throw InvalidInputException("Model manifest %s: field \"%s\" must be a non-empty string", manifest_path,
		                            field);
	}
	return result;
}

string GetOptionalString(yyjson_val *obj, const char *field, const string &default_value,
                         const string &manifest_path) {
	auto val = yyjson_obj_get(obj, field);
	if (!val) {
		return default_value;
	}
	if (!yyjson_is_str(val)) {
		throw InvalidInputException("Model manifest %s: field \"%s\" must be a string", manifest_path, field);
	}
	return string(yyjson_get_str(val), yyjson_get_len(val));
}

TabFMTask ParseTask(const string &task, const string &manifest_path) {
	if (task == "classification") {
		return TabFMTask::CLASSIFICATION;
	}
	if (task == "regression") {
		return TabFMTask::REGRESSION;
	}
	throw InvalidInputException("Model manifest %s: \"task\" must be one of 'classification', 'regression', "
	                            "got '%s'",
	                            manifest_path, task);
}

vector<ManifestFile> ParseFiles(yyjson_val *root, const string &manifest_path) {
	auto files_val = yyjson_obj_get(root, "files");
	if (!files_val) {
		throw InvalidInputException("Model manifest %s: missing required field \"files\"", manifest_path);
	}
	if (!yyjson_is_arr(files_val) || yyjson_arr_size(files_val) == 0) {
		throw InvalidInputException("Model manifest %s: \"files\" must be a non-empty array", manifest_path);
	}
	vector<ManifestFile> files;
	yyjson_arr_iter iter;
	yyjson_arr_iter_init(files_val, &iter);
	yyjson_val *entry;
	while ((entry = yyjson_arr_iter_next(&iter))) {
		if (!yyjson_is_obj(entry)) {
			throw InvalidInputException("Model manifest %s: every \"files\" entry must be an object", manifest_path);
		}
		ManifestFile file;
		file.path = GetRequiredString(entry, "path", manifest_path);
		file.url = GetOptionalString(entry, "url", "", manifest_path);
		auto bytes_val = yyjson_obj_get(entry, "bytes");
		if (!bytes_val) {
			throw InvalidInputException("Model manifest %s: file \"%s\" is missing the required field \"bytes\"",
			                            manifest_path, file.path);
		}
		if (!yyjson_is_int(bytes_val) || (yyjson_is_sint(bytes_val) && yyjson_get_sint(bytes_val) < 0)) {
			throw InvalidInputException("Model manifest %s: file \"%s\": \"bytes\" must be a non-negative integer",
			                            manifest_path, file.path);
		}
		file.bytes = UnsafeNumericCast<idx_t>(yyjson_get_uint(bytes_val));
		files.push_back(std::move(file));
	}
	return files;
}

void ParseTensorMap(yyjson_val *root, ModelManifest &manifest, const string &manifest_path) {
	auto map_val = yyjson_obj_get(root, "tensor_map");
	if (!map_val) {
		return; // optional: absent = identity mapping
	}
	if (yyjson_is_str(map_val)) {
		manifest.tensor_map_path = string(yyjson_get_str(map_val), yyjson_get_len(map_val));
		return;
	}
	if (yyjson_is_obj(map_val)) {
		yyjson_obj_iter iter;
		yyjson_obj_iter_init(map_val, &iter);
		yyjson_val *key;
		while ((key = yyjson_obj_iter_next(&iter))) {
			auto val = yyjson_obj_iter_get_val(key);
			if (!yyjson_is_str(val)) {
				throw InvalidInputException("Model manifest %s: \"tensor_map\" values must be safetensors key "
				                            "strings",
				                            manifest_path);
			}
			manifest.tensor_map.emplace(string(yyjson_get_str(key), yyjson_get_len(key)),
			                            string(yyjson_get_str(val), yyjson_get_len(val)));
		}
		return;
	}
	throw InvalidInputException("Model manifest %s: \"tensor_map\" must be a path string or an object mapping "
	                            "ONNX initializer names to safetensors keys",
	                            manifest_path);
}

void ParseEngineProfiles(yyjson_val *root, ModelManifest &manifest, const string &manifest_path) {
	auto profiles_val = yyjson_obj_get(root, "engine_profiles");
	if (profiles_val) {
		if (!yyjson_is_obj(profiles_val)) {
			throw InvalidInputException("Model manifest %s: \"engine_profiles\" must be an object keyed by device",
			                            manifest_path);
		}
		yyjson_obj_iter iter;
		yyjson_obj_iter_init(profiles_val, &iter);
		yyjson_val *key;
		while ((key = yyjson_obj_iter_next(&iter))) {
			auto val = yyjson_obj_iter_get_val(key);
			string device(yyjson_get_str(key), yyjson_get_len(key));
			if (!yyjson_is_obj(val)) {
				throw InvalidInputException("Model manifest %s: engine profile \"%s\" must be an object",
				                            manifest_path, device);
			}
			auto dtype_val = yyjson_obj_get(val, "dtype");
			if (!dtype_val || !yyjson_is_str(dtype_val)) {
				throw InvalidInputException("Model manifest %s: engine profile \"%s\" is missing the required "
				                            "field \"dtype\"",
				                            manifest_path, device);
			}
			string dtype(yyjson_get_str(dtype_val), yyjson_get_len(dtype_val));
			if (dtype != "f32" && dtype != "bf16" && dtype != "fp16") {
				throw InvalidInputException("Model manifest %s: engine profile \"%s\": \"dtype\" must be one of "
				                            "'f32', 'bf16', 'fp16', got '%s'",
				                            manifest_path, device, dtype);
			}
			EngineProfile profile;
			profile.dtype = std::move(dtype);
			manifest.engine_profiles.emplace(std::move(device), std::move(profile));
		}
	}
	// the cpu/f32 profile is always present (HLD D5: CPU stays fp32)
	if (manifest.engine_profiles.find("cpu") == manifest.engine_profiles.end()) {
		EngineProfile cpu;
		cpu.dtype = "f32";
		manifest.engine_profiles.emplace("cpu", std::move(cpu));
	}
}

} // anonymous namespace

ModelManifest ParseModelManifest(const string &json, const string &manifest_path) {
	yyjson_read_err error;
	YyjsonDoc guard(yyjson_read_opts(const_cast<char *>(json.c_str()), json.size(), YYJSON_READ_NOFLAG, nullptr,
	                                 &error));
	if (!guard.doc) {
		throw InvalidInputException("Model manifest %s: malformed JSON: %s (at byte %llu)", manifest_path, error.msg,
		                            static_cast<unsigned long long>(error.pos));
	}
	auto root = yyjson_doc_get_root(guard.doc);
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("Model manifest %s: the top-level JSON value must be an object", manifest_path);
	}

	ModelManifest manifest;
	// "model" is the canonical field; "model_id" is accepted as an alias
	// (the WS-A CI fixture uses "model_id").
	if (yyjson_obj_get(root, "model")) {
		manifest.model = GetRequiredString(root, "model", manifest_path);
	} else if (yyjson_obj_get(root, "model_id")) {
		manifest.model = GetRequiredString(root, "model_id", manifest_path);
	} else {
		throw InvalidInputException("Model manifest %s: missing required field \"model\" (or its alias \"model_id\")",
		                            manifest_path);
	}
	manifest.task = ParseTask(GetRequiredString(root, "task", manifest_path), manifest_path);
	manifest.repo = GetOptionalString(root, "repo", "", manifest_path);
	manifest.revision = GetOptionalString(root, "revision", "main", manifest_path);
	manifest.files = ParseFiles(root, manifest_path);
	manifest.graph = GetRequiredString(root, "graph", manifest_path);
	ParseTensorMap(root, manifest, manifest_path);
	manifest.preprocessing_profile = GetRequiredString(root, "preprocessing_profile", manifest_path);
	manifest.license = GetRequiredString(root, "license", manifest_path);
	ParseEngineProfiles(root, manifest, manifest_path);

	// every file must be resolvable to a URL (FR-1.x download path)
	if (manifest.repo.empty()) {
		for (auto &file : manifest.files) {
			if (file.url.empty()) {
				throw InvalidInputException("Model manifest %s: file \"%s\" has no \"url\" and the manifest has "
				                            "no \"repo\" to derive one from",
				                            manifest_path, file.path);
			}
		}
	}
	return manifest;
}

ModelManifest LoadModelManifestFile(const string &path) {
	auto fs = FileSystem::CreateLocal();
	if (!fs->FileExists(path)) {
		throw IOException("Model manifest file \"%s\" does not exist or is not readable", path);
	}
	auto handle = fs->OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = UnsafeNumericCast<idx_t>(handle->GetFileSize());
	string json(size, '\0');
	handle->Read(const_cast<char *>(json.data()), size);
	return ParseModelManifest(json, path);
}

string ResolveManifestFileUrl(const ModelManifest &manifest, const ManifestFile &file) {
	if (!file.url.empty()) {
		return file.url;
	}
	return "https://huggingface.co/" + manifest.repo + "/resolve/" + manifest.revision + "/" + file.path;
}

//===----------------------------------------------------------------------===//
// Built-in TabFM v1 manifests (HLD D7; sizes verified live in S03)
//===----------------------------------------------------------------------===//

namespace {

// classification/model.safetensors size verified 2026-07-04 (S03 RESULTS:
// x-linked-size 6557888408). The regression checkpoint size was not captured
// by the spikes; bytes=0 means "unknown, size check skipped" per the schema.
const char *const BUILTIN_TABFM_V1_CLASSIFICATION = R"json({
	"model": "tabfm-v1",
	"task": "classification",
	"repo": "google/tabfm-1.0.0-pytorch",
	"revision": "main",
	"files": [
		{"path": "classification/model.safetensors", "bytes": 6557888408}
	],
	"graph": "graph_classification",
	"tensor_map": "tensor_map_classification.json",
	"preprocessing_profile": "tabfm-v1",
	"license": "tabfm-non-commercial-v1.0",
	"engine_profiles": {
		"cpu": {"dtype": "f32"},
		"cuda": {"dtype": "bf16"},
		"rocm": {"dtype": "fp16"}
	}
})json";

const char *const BUILTIN_TABFM_V1_REGRESSION = R"json({
	"model": "tabfm-v1",
	"task": "regression",
	"repo": "google/tabfm-1.0.0-pytorch",
	"revision": "main",
	"files": [
		{"path": "regression/model.safetensors", "bytes": 0}
	],
	"graph": "graph_regression",
	"tensor_map": "tensor_map_regression.json",
	"preprocessing_profile": "tabfm-v1",
	"license": "tabfm-non-commercial-v1.0",
	"engine_profiles": {
		"cpu": {"dtype": "f32"},
		"cuda": {"dtype": "bf16"},
		"rocm": {"dtype": "fp16"}
	}
})json";

} // anonymous namespace

const char *BuiltinTabFMManifestJson(TabFMTask task) {
	return task == TabFMTask::CLASSIFICATION ? BUILTIN_TABFM_V1_CLASSIFICATION : BUILTIN_TABFM_V1_REGRESSION;
}

ModelManifest BuiltinTabFMManifest(TabFMTask task) {
	return ParseModelManifest(BuiltinTabFMManifestJson(task),
	                          string("(built-in tabfm-v1 ") + TabFMTaskName(task) + " manifest)");
}

} // namespace anofox
} // namespace duckdb
