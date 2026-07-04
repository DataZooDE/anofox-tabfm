//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_manifest.hpp — model manifest (HLD §2 D7, drives FR-5.1 pluggable
// models). A manifest is a JSON document describing one (model, task) pair:
//
// {
//   "model": "tabfm-v1",                     // required: manifest/model id
//                                            // ("model_id" accepted as alias)
//   "task": "classification",                // required: "classification" |
//                                            //           "regression"
//   "repo": "google/tabfm-1.0.0-pytorch",    // HF repo; required unless every
//                                            // file carries an explicit "url"
//   "revision": "main",                      // optional, default "main"
//   "files": [                               // required, non-empty
//     {"path": "classification/model.safetensors", // required
//      "url": "https://...",                 // optional; when absent derived:
//                                            // https://huggingface.co/<repo>/
//                                            //   resolve/<revision>/<path>
//      "bytes": 6557888408}                  // required, >= 0 (0 = unknown,
//                                            // size check skipped)
//   ],
//   "graph": "graph_classification",         // required: bundled graph id or
//                                            // path to a weight-free .onnx
//   "tensor_map": "tensor_map.json" | {...}, // optional: path to, or inline,
//                                            // {onnx initializer -> st key};
//                                            // absent = identity mapping
//   "preprocessing_profile": "tabfm-v1",     // required (HLD §4.5)
//   "license": "tabfm-non-commercial-v1.0",  // required license id
//   "engine_profiles": {                     // optional; "cpu": {"dtype":
//     "cpu":  {"dtype": "f32"},              //   "f32"} is always present
//     "cuda": {"dtype": "bf16"},             //   after parsing (defaulted in)
//     "rocm": {"dtype": "fp16"}              // dtype must be f32|bf16|fp16
//   }
// }
//
// Unknown fields are ignored (forward compatibility: WS-A fixtures may carry
// e.g. per-file "sha256"). All validation failures are InvalidInputException
// naming the offending field and the manifest path.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace anofox {

enum class TabFMTask : uint8_t { CLASSIFICATION = 0, REGRESSION = 1 };

//! "classification" / "regression".
const char *TabFMTaskName(TabFMTask task);

struct ManifestFile {
	//! Repo-relative path; also the cache-relative path.
	string path;
	//! Explicit download URL; empty means "derive from repo/revision/path".
	string url;
	//! Expected size in bytes; 0 = unknown (size check skipped downstream).
	idx_t bytes = 0;
};

struct EngineProfile {
	//! Compute dtype for this execution provider: "f32" | "bf16" | "fp16".
	string dtype;
};

struct ModelManifest {
	string model;
	TabFMTask task = TabFMTask::CLASSIFICATION;
	string repo;
	string revision = "main";
	vector<ManifestFile> files;
	string graph;
	//! Exactly one of the two tensor-map forms is populated (or neither).
	string tensor_map_path;
	unordered_map<string, string> tensor_map;
	string preprocessing_profile;
	string license;
	//! Keyed by device ("cpu"/"cuda"/"rocm"/...); "cpu" -> f32 guaranteed.
	map<string, EngineProfile> engine_profiles;
};

//! Parse and strictly validate a manifest from a JSON string. `manifest_path`
//! is used in every error message; pass the file path when loading from disk.
ModelManifest ParseModelManifest(const string &json, const string &manifest_path = "(inline manifest)");

//! Read `path` from the local filesystem and parse it. IOException when the
//! file cannot be read; InvalidInputException on validation failures.
ModelManifest LoadModelManifestFile(const string &path);

//! Download URL for a manifest file: the explicit "url" when present, else
//! https://huggingface.co/<repo>/resolve/<revision>/<path> (HLD D3).
string ResolveManifestFileUrl(const ModelManifest &manifest, const ManifestFile &file);

//! Raw JSON of the built-in TabFM v1 manifest for `task` (HLD D7); exposed so
//! integration can register/display the canonical text.
const char *BuiltinTabFMManifestJson(TabFMTask task);

//! The built-in TabFM v1 manifest, parsed through the same validation path.
ModelManifest BuiltinTabFMManifest(TabFMTask task);

} // namespace anofox
} // namespace duckdb
