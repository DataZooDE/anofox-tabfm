//===----------------------------------------------------------------------===//
// tabfm_engine.cpp — the real TabFM predict engine (Phase-2 integration).
//
// Wires the independent modules into one forward pass behind the PredictEngine
// seam (tabfm_predict.hpp):
//
//   rows ──► ColumnDataCollection ──► PreprocessBatch (WS-F)
//            │                         x[T,H] f32, y[T], cat_mask[H], d,
//            │                         train_size, label_decoder, target stats
//            ▼
//   manifest (WS-B) ─► graph.onnx + model.safetensors + tensor_map
//            │           (SafetensorsView → F32Arena, injected by name)
//            ▼
//   CreateSession (WS-C, cached in TabFMState) ─► Run ─► logits[1,T,C]
//            ▼
//   decode: classification argmax + softmax(temperature) → label/proba,
//           regression inverse-transform → yhat; scattered back to input order.
//
// v1 runs a single estimator (n_estimators>1 is rejected at bind); the
// ensemble layer (WS-F) is added on top later.
//===----------------------------------------------------------------------===//

#include "tabfm_predict.hpp"
#include "tabfm_preprocess.hpp"
#include "tabfm_manifest.hpp"
#include "tabfm_registry.hpp"
#include "tabfm_safetensors.hpp"
#include "tabfm_ckpt.hpp"
#include "tabfm_ort_engine.hpp"
#include "tabfm_bundled_resources.hpp"
#include "tabfm_plugin_backend.hpp"
#include "tabfm_plugin_artifacts.hpp"
#include "tabfm_mxr_cache_key.hpp"
#include "tabfm_shape_bucket.hpp"
#include "tabfm_mlx_ops.hpp"
#include "tabfm_onnx_reader.hpp"
#include "tabfm_state.hpp"
#include "tabfm_context_cache.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb_compat.hpp"

#include "yyjson.hpp"

#include <algorithm>
#include <openssl/evp.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <atomic>
#include <algorithm>
#include <fstream>
#include <list>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace duckdb {
namespace anofox {

//! Bumped when a plugin is fetched or verified, so both memos below answer
//! afresh without any entry being erased underneath a live reference.
std::atomic<uint64_t> g_plugin_probe_generation {0};

void BumpPluginProbeGeneration() {
	g_plugin_probe_generation.fetch_add(1);
}

namespace {

//===--------------------------------------------------------------------===//
// File + manifest resolution
//===--------------------------------------------------------------------===//

string DirName(const string &path) {
	auto slash = path.find_last_of("/\\");
	return slash == string::npos ? string(".") : path.substr(0, slash);
}

string BaseName(const string &path) {
	auto slash = path.find_last_of("/\\");
	return slash == string::npos ? path : path.substr(slash + 1);
}

string JoinPath(FileSystem &fs, const string &dir, const string &name) {
	if (name.empty() || name[0] == '/' || (name.size() > 1 && name[1] == ':')) {
		return name; // already absolute
	}
	return fs.JoinPath(dir, name);
}

string ReadWholeFile(FileSystem &fs, const string &path) {
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = NumericCast<idx_t>(fs.GetFileSize(*handle));
	string buffer(size, '\0');
	// Read in bounded chunks with explicit offsets: a single Read() of a
	// multi-GB file short-reads on Linux (read(2) caps at ~2 GB) and leaves the
	// tail zero-initialized — which silently corrupts large weight files.
	constexpr idx_t kChunk = 1ULL << 30; // 1 GiB
	idx_t done = 0;
	while (done < size) {
		idx_t want = MinValue<idx_t>(kChunk, size - done);
		fs.Read(*handle, (void *)(buffer.data() + done), NumericCast<int64_t>(want), NumericCast<idx_t>(done));
		done += want;
	}
	return buffer;
}

//! The resolved on-disk artifacts for one (model, task).
struct ResolvedModel {
	ModelManifest manifest;
	string manifest_dir;
	// Exactly one graph source is set: a bundled graph compiled into the binary
	// (built-in manifest ids like "graph_classification"), or an on-disk path
	// (custom/scenario/fixture manifests that point at an .onnx file).
	string graph_path;
	BundledResource graph_bundle;
	// Model-provided GPU graphs, resolved to on-disk paths ("" = not declared).
	// Their external-data files must sit beside them; the dispatch hands the
	// containing directory to the plugin as weights_dir.
	string ext_graph_path;
	string migraphx_graph_path;
	string weights_path;
	unordered_map<string, string> tensor_map; // onnx initializer name -> st key
	//! Checkpoint keys this task's graph does not read (the tensor map's optional "unused_by_graph"
	//! list). Skipped at injection: ORT rejects an injected initializer its graph lacks. Empty for
	//! every model whose graphs use the whole file.
	unordered_set<string> unused_weights;
	//! The map declared `"complete": true`: it names EVERY initializer its graph declares, so a
	//! weights-file tensor it does not name is not this graph's and must not be injected (see
	//! TensorMapDeclaresComplete). False keeps the "m." fallback for unmapped tensors.
	bool tensor_map_complete = false;
	//! The map's "ckpt_aliases": graph-side keys a raw checkpoint stores under another name.
	unordered_map<string, string> ckpt_aliases;
	string cache_key;
	// Manifest-declared tensor contract (P4): graph input/output names to verify
	// against the actual graph at load. Empty = no declared contract.
	vector<string> contract_inputs;
	vector<string> contract_outputs;
	// Set by the spec phase, consumed by the artifact phase: built-ins carry a
	// bundled graph + cache weights, a user manifest resolves next to itself.
	bool is_builtin = false;
	string source_dir;
	//! size_regime.max_classes from the registry; -1 when the spec omits it.
	int64_t max_classes = -1;
	//! The registry says this (model, task) needs its context rows evaluated a second time as queries to get
	//! usable fitted values (ModelSpec::fitted_second_pass).
	bool fitted_second_pass = false;
};

// {onnx initializer name -> safetensors key} from a tensor-map document: the
// "initializers" object, or a bare {name: key} map.
unordered_map<string, string> ParseTensorMapJson(const string &json, const string &source) {
	unordered_map<string, string> result;
	using namespace duckdb_yyjson; // NOLINT
	auto doc = yyjson_read(json.c_str(), json.size(), 0);
	if (!doc) {
		throw InvalidInputException("tabfm: cannot parse tensor map '%s'", source);
	}
	auto root = yyjson_doc_get_root(doc);
	auto inits = yyjson_obj_get(root, "initializers");
	auto obj = (inits && yyjson_is_obj(inits)) ? inits : root; // accept bare map too
	if (obj && yyjson_is_obj(obj)) {
		size_t idx, max;
		yyjson_val *key, *val;
		yyjson_obj_foreach(obj, idx, max, key, val) {
			if (yyjson_is_str(val)) {
				result[yyjson_get_str(key)] = yyjson_get_str(val);
			}
		}
	}
	yyjson_doc_free(doc);
	return result;
}

// The optional "unused_by_graph" array of a tensor-map document: checkpoint keys present in the weights
// file that this task's graph does not use. Needed when ONE file serves several tasks whose graphs read
// different subsets of it (tabdpt-v1-3: cls_y_encoders.* for classification, reg_y_encoders.* for
// regression). The whole file is otherwise injected, and ORT refuses an initializer the graph does not
// declare -- so the list is explicit and opt-in; a map without it behaves exactly as before.
unordered_set<string> ParseUnusedByGraph(const string &json) {
	unordered_set<string> result;
	using namespace duckdb_yyjson; // NOLINT
	auto doc = yyjson_read(json.c_str(), json.size(), 0);
	if (!doc) {
		return result; // ParseTensorMapJson already reported the unparseable document
	}
	auto arr = yyjson_obj_get(yyjson_doc_get_root(doc), "unused_by_graph");
	if (arr && yyjson_is_arr(arr)) {
		size_t idx, max;
		yyjson_val *val;
		yyjson_arr_foreach(arr, idx, max, val) {
			if (yyjson_is_str(val)) {
				result.insert(yyjson_get_str(val));
			}
		}
	}
	yyjson_doc_free(doc);
	return result;
}

// Whether a tensor-map document carries `"complete": true`.
//
// One weights file can hold MORE tensors than one task's graph declares: TabPFN-3.5
// publishes a single checkpoint per variant with BOTH heads, and each task's graph
// uses only its own. Every tensor in the file is injected, unmapped ones under the
// "m." wrapper fallback, and ORT rejects an injected name the graph lacks ("Failed
// to find existing initializer"). A map cannot be assumed complete in general --
// TabICL's fixture map omits 23 initializers and relies on that fallback -- so the
// map has to say so. Only then are unmapped file tensors skipped.
bool TensorMapDeclaresComplete(const string &json) {
	using namespace duckdb_yyjson; // NOLINT
	auto doc = yyjson_read(json.c_str(), json.size(), 0);
	if (!doc) {
		return false;
	}
	auto flag = yyjson_obj_get(yyjson_doc_get_root(doc), "complete");
	const bool complete = flag && yyjson_is_bool(flag) && yyjson_get_bool(flag);
	yyjson_doc_free(doc);
	return complete;
}

// Load {onnx -> safetensors} from the manifest: inline map, or the
// "initializers" object of the tensor-map JSON file, else identity.
// `unused_weights` (optional) receives the document's "unused_by_graph" list.
unordered_map<string, string> LoadTensorMap(FileSystem &fs, const ModelManifest &manifest, const string &dir,
                                            bool use_bundle, unordered_set<string> *unused_weights = nullptr,
                                            bool *complete = nullptr,
                                            unordered_map<string, string> *ckpt_aliases = nullptr) {
	if (!manifest.tensor_map.empty()) {
		return manifest.tensor_map;
	}
	unordered_map<string, string> result;
	if (manifest.tensor_map_path.empty()) {
		return result; // identity mapping (handled at injection)
	}
	// Built-in models read their bundled tensor map (embedded in the binary);
	// user/fixture manifests ALWAYS read from their own directory — otherwise a
	// fixture that happens to share a filename with a bundled resource would pick
	// up the bundled (real-model) map instead of its own.
	string json;
	string source = manifest.tensor_map_path;
	BundledResource bundled = use_bundle ? GetBundledResource(manifest.tensor_map_path) : BundledResource {};
	if (bundled.data) {
		json.assign(bundled.data, bundled.size);
	} else {
		source = JoinPath(fs, dir, manifest.tensor_map_path);
		json = ReadWholeFile(fs, source);
	}
	if (unused_weights) {
		*unused_weights = ParseUnusedByGraph(json);
	}
	if (complete) {
		*complete = TensorMapDeclaresComplete(json);
	}
	if (ckpt_aliases) {
		*ckpt_aliases = ParseCkptAliases(json.data(), json.size());
	}
	return ParseTensorMapJson(json, source);
}

string ResolveGraphPath(FileSystem &fs, const ModelManifest &manifest, const string &dir) {
	const auto &graph = manifest.graph;
	// A path (has an extension / separator) is resolved relative to the manifest.
	string candidate = JoinPath(fs, dir, StringUtil::EndsWith(graph, ".onnx") ? graph : graph + ".onnx");
	if (fs.FileExists(candidate)) {
		return candidate;
	}
	if (fs.FileExists(JoinPath(fs, dir, graph))) {
		return JoinPath(fs, dir, graph);
	}
	throw InvalidInputException(
	    "tabfm: model graph '%s' for task '%s' was not found next to the manifest (%s). The weight-free graph must "
	    "ship with the model artifacts.",
	    graph, TabFMTaskName(manifest.task), dir);
}

// Locate the safetensors weights: the WS-D cache layout, else next to the
// manifest (air-gapped / fixture). Errors with the SQL-API §5 download hint.
string ResolveWeightsPath(FileSystem &fs, const ModelManifest &manifest, const string &dir, const string &cache_dir,
                          const string &task_name) {
	if (manifest.files.empty()) {
		throw InvalidInputException("tabfm: manifest for task '%s' lists no weight files", task_name);
	}
	const auto &file = manifest.files.front();

	// A manifest declares the DOWNLOADABLE artifact, which for several models is
	// a raw PyTorch `.ckpt`. Some of those checkpoints cannot be injected as-is:
	// their state_dict keys are the upstream module names, while the committed
	// tensor map is keyed to the names the model exposes only after being loaded
	// through its Python package (TabPFN, for instance, maps `m.blocks.*` ->
	// `blocks.*`, but the raw ckpt carries `transformer_encoder.layers.*`). For
	// those, `tools/export_*/convert_weights.py` writes a `model.safetensors`
	// NEXT TO the checkpoint, in the same cache slug.
	//
	// So prefer a sibling `model.safetensors` over the declared file. Nothing
	// else pointed at the converter's output, which is why `model :=
	// 'tabpfn-v2'` failed with "Failed to find existing initializer" even
	// though the correctly converted weights were sitting in the cache.
	// Restricted to `.ckpt` on purpose. A manifest that already names a
	// safetensors is authoritative — several fixtures keep both tasks' weights in
	// ONE directory under different names (`model.safetensors` +
	// `model_regression.safetensors`), so rewriting the basename unconditionally
	// makes the regression task silently load the classification weights.
	auto converted_sibling = [&](const string &path) {
		if (!StringUtil::EndsWith(StringUtil::Lower(path), ".ckpt")) {
			return string();
		}
		auto parts = StringUtil::Split(path, "/");
		if (parts.empty()) {
			return string();
		}
		parts.back() = "model.safetensors";
		return StringUtil::Join(parts, "/");
	};

	vector<string> candidates;
	auto add_candidates = [&](const string &relative) {
		if (relative.empty()) {
			return;
		}
		if (!cache_dir.empty()) {
			// Match the download-side cache slug (WeightsManifest::CacheSlug): a
			// repo-less model (e.g. a user manifest with per-file urls) caches
			// under its model id, not a repo path. Resolve must look there too,
			// else a downloaded repo-less model reads as "not downloaded".
			auto slug_base = manifest.repo.empty() ? manifest.model : StringUtil::Replace(manifest.repo, "/", "__");
			candidates.push_back(fs.JoinPath(fs.JoinPath(cache_dir, slug_base + "@" + manifest.revision), relative));
		}
		candidates.push_back(JoinPath(fs, dir, relative));
		// fixture layout: a bare model.safetensors beside the manifest
		candidates.push_back(JoinPath(fs, dir, StringUtil::Split(relative, "/").back()));
	};
	add_candidates(converted_sibling(file.path));
	add_candidates(file.path);
	for (auto &candidate : candidates) {
		if (fs.FileExists(candidate)) {
			return candidate;
		}
	}
	// Name the MODEL and the task separately — this used to interpolate the task
	// into the "model '%s'" slot, so a missing tabpfn-v2-5 reported itself as
	// model 'classification'.
	throw InvalidInputException(
	    "tabfm: model '%s' has no downloaded weights for task '%s'. Run: CALL tabfm_download('%s', model := '%s');",
	    manifest.model, task_name, task_name, manifest.model);
}

// Bridge a resolved ModelSpec's per-task artifacts into the per-(model,task)
// ModelManifest the downstream weights/graph/tensor-map resolution consumes.
ModelManifest SpecTaskToManifest(const ModelSpec &spec, TabFMTask task) {
	const auto &art = spec.tasks.at(task);
	ModelManifest m;
	m.model = spec.id;
	m.task = task;
	m.repo = art.repo;
	m.revision = art.revision;
	m.files = art.files;
	m.graph = art.graph;
	m.ext_graph = art.ext_graph;
	m.migraphx_graph = art.migraphx_graph;
	m.tensor_map_path = art.tensor_map_path;
	m.tensor_map = art.tensor_map;
	m.preprocessing_profile = art.preprocessing_profile;
	m.license = spec.license.id;
	m.engine_profiles = spec.engine_profiles;
	return m;
}

// Phase 1 — registry lookup only: which model, and does it declare the task?
// Pure in-memory, so callers can learn the preprocessing profile (and validate
// their data against it) before phase 2 touches the filesystem.
ResolvedModel ResolveModelSpec(const PredictContext &ctx, TabFMTask task, const string &requested_model) {
	ResolvedModel resolved;
	const auto task_name = TabFMTaskName(task);
	// The registry: built-ins + SQL-registered models (CALL tabfm_register_model).
	// file or dir). Selection precedence: model := → default_model → single-file
	// manifest → sole model.
	auto registry = ModelRegistry::Build(ctx.db ? TabFMState::Get(*ctx.db)->RegisteredSpecs() : vector<ModelSpec>());
	const ModelSpec &spec = registry.Resolve(requested_model, ctx.default_model);
	if (!spec.HasTask(task)) {
		const auto capable = registry.IdsWithTask(task);
		throw InvalidInputException(
		    "tabfm: model '%s' does not support task '%s'. Choose a model that declares it, or use the matching "
		    "tabfm_%s function.%s",
		    spec.id, task_name, task == TabFMTask::CLASSIFICATION ? "classify" : "regress",
		    capable.empty() ? string() : " Models that support it: " + capable + " (model := '<id>').");
	}
	resolved.manifest = SpecTaskToManifest(spec, task);
	for (auto &e : spec.tensor_contract.inputs) {
		resolved.contract_inputs.push_back(e.name);
	}
	for (auto &e : spec.tensor_contract.outputs) {
		resolved.contract_outputs.push_back(e.name);
	}
	// Built-ins carry a bundled graph + cache weights → cache_dir; a user manifest's
	// relative paths resolve against its own directory. Only built-ins consult the
	// bundle: a fixture that shares a filename with a bundled resource must load
	// its OWN local file, not the embedded real-model one.
	resolved.is_builtin = spec.source_dir.empty();
	resolved.source_dir = spec.source_dir;
	resolved.max_classes = spec.size_regime.max_classes;
	resolved.fitted_second_pass = spec.FittedSecondPass(task);
	return resolved;
}

// Phase 2 — the filesystem work: weights, graph and tensor map. Split from the
// spec phase so a data error (e.g. no usable feature columns) is reported
// before "model not downloaded" for a query that could never run anyway.
void ResolveModelArtifacts(FileSystem &fs, const PredictContext &ctx, TabFMTask task, ResolvedModel &resolved) {
	const auto task_name = TabFMTaskName(task);
	const bool is_builtin = resolved.is_builtin;
	resolved.manifest_dir = is_builtin ? ctx.cache_dir : resolved.source_dir;
	// weights first: "not downloaded" is the common, actionable error (§5).
	resolved.weights_path =
	    ResolveWeightsPath(fs, resolved.manifest, resolved.manifest_dir, ctx.cache_dir, task_name);
	// Graph: for built-ins a bundled id ("graph_classification") resolves to
	// embedded bytes; user manifests always resolve a path next to the manifest.
	BundledResource bundled = is_builtin ? GetBundledResource(resolved.manifest.graph) : BundledResource {};
	if (bundled.data) {
		resolved.graph_bundle = bundled;
	} else {
		resolved.graph_path = ResolveGraphPath(fs, resolved.manifest, resolved.manifest_dir);
	}
	// Model-provided GPU graphs resolve like the plain graph — against the
	// manifest dir — but must exist NOW if declared: a bad path failing at
	// declaration time beats a GPU dispatch that silently falls back later.
	auto resolve_gpu_graph = [&](const string &declared, const char *what) -> string {
		if (declared.empty()) {
			return "";
		}
		auto candidate = JoinPath(fs, resolved.manifest_dir, declared);
		if (!fs.FileExists(candidate)) {
			throw InvalidInputException(
			    "tabfm: model '%s' declares %s '%s' for task '%s' but the file does not exist at '%s'. Its "
			    "external-data files must sit beside it.",
			    resolved.manifest.model, what, declared, task_name, candidate);
		}
		return candidate;
	};
	resolved.ext_graph_path = resolve_gpu_graph(resolved.manifest.ext_graph, "ext_graph");
	resolved.migraphx_graph_path = resolve_gpu_graph(resolved.manifest.migraphx_graph, "migraphx_graph");
	resolved.tensor_map =
	    LoadTensorMap(fs, resolved.manifest, resolved.manifest_dir, is_builtin, &resolved.unused_weights,
	                  &resolved.tensor_map_complete, &resolved.ckpt_aliases);
	resolved.cache_key = TabFMModelCacheKey(resolved.manifest.model, task_name, resolved.manifest.revision);
}

ResolvedModel ResolveModel(FileSystem &fs, const PredictContext &ctx, TabFMTask task, const string &requested_model) {
	auto resolved = ResolveModelSpec(ctx, task, requested_model);
	ResolveModelArtifacts(fs, ctx, task, resolved);
	return resolved;
}

//===--------------------------------------------------------------------===//
// Session loading (cached in TabFMState)
//===--------------------------------------------------------------------===//

TabFMTensorDtype ToEngineDtype(SafetensorsDtype dtype) {
	return dtype == SafetensorsDtype::I64 ? TabFMTensorDtype::I64 : TabFMTensorDtype::F32;
}

// A read-only memory-mapping of the safetensors cache file. Preferred over a
// full read: the multi-GB weights are never copied into an anonymous heap
// buffer — ORT reads the injected initializers straight from the mapped, file-
// backed pages (which the OS can reclaim/share), and the up-front 6.6 GB read
// (and its short-read hazard) disappears. Local files only; POSIX only for now.
struct MappedFile {
	const char *data = nullptr;
	idx_t size = 0;
#ifndef _WIN32
	void *base = nullptr;
	idx_t map_len = 0;
	int fd = -1;
#endif

	MappedFile() = default;
	MappedFile(const MappedFile &) = delete;
	MappedFile &operator=(const MappedFile &) = delete;
	MappedFile(MappedFile &&o) noexcept {
		*this = std::move(o);
	}
	MappedFile &operator=(MappedFile &&o) noexcept {
		if (this != &o) {
			Reset();
			data = o.data;
			size = o.size;
#ifndef _WIN32
			base = o.base;
			map_len = o.map_len;
			fd = o.fd;
			o.base = nullptr;
			o.map_len = 0;
			o.fd = -1;
#endif
			o.data = nullptr;
			o.size = 0;
		}
		return *this;
	}
	~MappedFile() {
		Reset();
	}
	bool valid() const {
		return data != nullptr;
	}
	void Reset() {
#ifndef _WIN32
		if (base) {
			munmap(base, map_len);
			base = nullptr;
		}
		if (fd >= 0) {
			close(fd);
			fd = -1;
		}
#endif
		data = nullptr;
		size = 0;
	}
};

// mmap `path` read-only; returns an invalid MappedFile if it cannot be mapped
// (Windows, a non-local/virtual path, or any OS error) so the caller falls back
// to a full read. TABFM_DISABLE_MMAP forces that fallback (benchmarking/debug).
MappedFile TryMapFile(const string &path) {
	MappedFile m;
#ifndef _WIN32
	if (std::getenv("TABFM_DISABLE_MMAP")) {
		return m;
	}
	int fd = open(path.c_str(), O_RDONLY);
	if (fd < 0) {
		return m;
	}
	struct stat st;
	if (fstat(fd, &st) != 0 || st.st_size <= 0) {
		close(fd);
		return m;
	}
	void *p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
	if (p == MAP_FAILED) {
		close(fd);
		return m;
	}
	m.base = p;
	m.map_len = NumericCast<idx_t>(st.st_size);
	m.fd = fd;
	m.data = reinterpret_cast<const char *>(p);
	m.size = m.map_len;
#endif
	return m;
}

// ORT-backed inference backend (CPU EP, or CUDA EP on the cuda flavor): the ORT
// session PLUS the source buffers it injected. AddExternalInitializers keeps
// references to the user buffers (and reads them lazily during inference), so
// the safetensors bytes and the bf16-upcast arena must outlive the session. The
// bytes come from either the mmap (preferred) or a full read (`weights`), or
// neither (external-data path: ORT reads weights from disk). Declaration order
// is the destruction contract: `session` is LAST so it is destroyed FIRST,
// before the buffers it references.
struct OrtBackend : public TabFMBackend {
	string weights;             // fallback: full read (f32 tensors point in here)
	MappedFile mapping;         // preferred: mmap of the cache file
	F32Arena arena;             // owns bf16->f32 upcast copies (empty for f32 models)
	TabFMSessionHandle session; // last: destroyed first, before the buffers

	TabFMRunOutput Run(const TabFMRunInput &input) override {
		return ::duckdb::anofox::Run(*session, input);
	}
};

// The split pair, found next to the combined graph by the names the exporter
// writes (tools/export_tabicl --split-context):
//
//   graph_<rest>.onnx  ->  graph_prepare_<rest>.onnx   graph_query_<rest>.onnx
//
// and each half's tensor map named from its own graph, by the rule the combined
// graph and map already follow (graph_ -> tensor_map_, .onnx -> .json).
//
// Discovered rather than declared in the manifest: the pair is produced by a
// single exporter flag, a deployment has both files or neither, and no existing
// manifest has to change to gain the fast path — or to keep working without it.
// Same shape as the bundled external-data lookup above, which also engages only
// when every artifact it needs is actually present.
// Either form of a half: bytes compiled into the binary, or a file next to the
// combined graph. Which one a model uses is not a property of the split — it is
// how that model already carries its combined graph, and the pair follows it.
struct SplitHalf {
	string graph_path;
	string map_path;
	BundledResource graph_bundle;
	BundledResource map_bundle;

	bool bundled() const {
		return graph_bundle.data != nullptr;
	}
	bool Available() const {
		return bundled() ? map_bundle.data != nullptr : !graph_path.empty();
	}
};

struct SplitGraphs {
	SplitHalf prepare;
	SplitHalf query;
	//! The prepare graph emits `context_logits`: every output row of the pair equals the combined
	//! graph's, so the cache may serve it (UseContextCache).
	bool exact = false;

	bool Available() const {
		return prepare.Available() && query.Available();
	}
};

// Does this pair follow the exact context-cache contract (ContextCachePairProblem)? Read from the graphs
// themselves, without building an ORT session: the answer decides whether a session is built at all, and
// a pair that does not qualify is simply not used -- the combined graph serves, rather than every call
// failing. Memoized because the lookup runs on every call with the cache on. A bundled graph is immutable
// for the life of the process; a file is keyed by path, size AND mtime, so a re-export in place that keeps
// the size is re-read.
string GraphIdentity(const char *bundle, const string &path) {
	if (bundle) {
		return "b:" + std::to_string(reinterpret_cast<uintptr_t>(bundle));
	}
	struct stat st;
	if (stat(path.c_str(), &st) != 0) {
		return "f:" + path + ":missing";
	}
	return "f:" + path + ":" + std::to_string(st.st_size) + ":" + std::to_string(static_cast<int64_t>(st.st_mtime));
}

::anofox::onnxread::Graph ParseGraph(const char *bundle, idx_t size, const string &path, const char *what) {
	return bundle ? ::anofox::onnxread::ParseModelBytes(reinterpret_cast<const uint8_t *>(bundle), size, what)
	              : ::anofox::onnxread::ParseModelFile(path);
}

bool SplitPairIsExact(const SplitGraphs &split, const ResolvedModel &resolved) {
	static mutex memo_lock;
	static map<string, bool> memo;
	const auto key = GraphIdentity(split.prepare.graph_bundle.data, split.prepare.graph_path) + "|" +
	                 GraphIdentity(split.query.graph_bundle.data, split.query.graph_path) + "|" +
	                 GraphIdentity(resolved.graph_bundle.data, resolved.graph_path);
	{
		lock_guard<mutex> guard(memo_lock);
		auto entry = memo.find(key);
		if (entry != memo.end()) {
			return entry->second;
		}
	}
	bool exact = false;
	try {
		auto prepare = ParseGraph(split.prepare.graph_bundle.data, split.prepare.graph_bundle.size,
		                          split.prepare.graph_path, "bundled prepare graph");
		auto query = ParseGraph(split.query.graph_bundle.data, split.query.graph_bundle.size, split.query.graph_path,
		                        "bundled query graph");
		auto combined = ParseGraph(resolved.graph_bundle.data, resolved.graph_bundle.size, resolved.graph_path,
		                           "bundled combined graph");
		auto as_vec = [](const std::vector<std::string> &v) { return vector<string>(v.begin(), v.end()); };
		exact = ContextCachePairProblem(as_vec(prepare.inputs), as_vec(prepare.outputs), as_vec(query.inputs),
		                                as_vec(query.outputs), as_vec(combined.inputs))
		            .empty();
	} catch (const std::exception &) {
		// Unreadable: not served from the cache. The combined graph still answers.
	}
	lock_guard<mutex> guard(memo_lock);
	return memo.emplace(key, exact).first->second;
}

SplitGraphs FindSplitGraphs(FileSystem &fs, const ResolvedModel &resolved) {
	SplitGraphs split;
	SplitHalf prepare, query;

	if (resolved.graph_bundle.data) {
		// Built-in: the combined graph is compiled in under an id like
		// "graph_tabicl_classification", and a re-export that adds the pair to
		// resources/ registers it under the ids the same rule produces. This is the
		// path that matters for the shipped models — they carry no graph on disk,
		// so a file-only lookup could never find their pair.
		const auto &id = resolved.manifest.graph;
		if (!StringUtil::StartsWith(id, "graph_")) {
			return split;
		}
		const auto rest = id.substr(6);
		auto half = [&](const char *which) {
			SplitHalf h;
			h.graph_bundle = GetBundledResource("graph_" + string(which) + "_" + rest);
			h.map_bundle = GetBundledResource("tensor_map_" + string(which) + "_" + rest + ".json");
			return h;
		};
		prepare = half("prepare");
		query = half("query");
	} else if (!resolved.graph_path.empty()) {
		const auto base = BaseName(resolved.graph_path);
		if (!StringUtil::StartsWith(base, "graph_") || !StringUtil::EndsWith(base, ".onnx")) {
			return split;
		}
		const auto dir = DirName(resolved.graph_path);
		const auto rest = base.substr(6, base.size() - 6 - 5); // between "graph_" and ".onnx"
		auto half = [&](const char *which) {
			SplitHalf h;
			auto graph = JoinPath(fs, dir, "graph_" + string(which) + "_" + rest + ".onnx");
			auto map = JoinPath(fs, dir, "tensor_map_" + string(which) + "_" + rest + ".json");
			if (fs.FileExists(graph) && fs.FileExists(map)) {
				h.graph_path = graph;
				h.map_path = map;
			}
			return h;
		};
		prepare = half("prepare");
		query = half("query");
	} else {
		return split;
	}

	// All four artifacts or none: a half-present pair is a broken export, and
	// engaging on it would fail deep inside session creation instead of falling
	// back cleanly to the combined graph.
	if (!prepare.Available() || !query.Available()) {
		return split;
	}
	split.prepare = std::move(prepare);
	split.query = std::move(query);
	split.exact = SplitPairIsExact(split, resolved);
	return split;
}

// "1, 20, 3" — a graph's actual output shape, for an error that has to name it.
string ShapeString(const vector<int64_t> &shape) {
	string out;
	for (idx_t i = 0; i < shape.size(); i++) {
		out += (i ? ", " : "") + std::to_string(shape[i]);
	}
	return out;
}

// A model exported as a support/query PAIR (tools/export_tabicl --split-context, kv.py): the labelled
// context is encoded once by the prepare graph and reused by every query batch that arrives with the same
// context. That encoding is what the combined graph re-runs on every call -- 83-98% of a tabicl-v2 call's
// CPU time at a 375- or 1000-row context (DataZooDE/anofox-tabfm#37, docs/CONTEXT_CACHE_PLAN.md).
//
// Only EXACT pairs are served (UseContextCache): the prepare graph emits `context_logits`, the context
// rows' outputs as the combined graph computes them, so no output row depends on whether the cache ran.
//
// Declaration order is the destruction contract, exactly as OrtBackend: the sessions are LAST so they are
// destroyed FIRST, before the weight buffers ORT reads lazily during inference.
struct SplitOrtBackend : public TabFMBackend {
	string weights;
	MappedFile mapping;
	F32Arena arena;
	//! The query graph's non-x inputs -- what prepare has to supply.
	vector<string> context_inputs;
	//! Counters shared with the LoadedModel, so tabfm_models() can report them.
	shared_ptr<ContextCacheStats> stats = make_shared_ptr<ContextCacheStats>();

	// The cached contexts, most recently used first, guarded by `cache_lock` (taken for every access).
	// The caller's per-device lock (HLD §6, keyed on the RESOLVED device since #42) also serializes calls
	// today, so the lock is uncontended; it is here so that the list stays safe if that lock is ever
	// re-keyed again -- before #42 a connection on 'auto' and one on 'cpu' took different mutexes for the
	// same backend.
	//
	// The key is the support rows and labels themselves, in order. The 64-bit hash only FINDS a candidate; a
	// hit is accepted after a full compare of the stored rows. A hash match alone would not fail on a
	// collision, it would answer a query against someone else's training data (KV_CACHE_DESIGN.md). The
	// compare costs O(S*H) floats, nothing next to the forward pass it saves.
	//
	// Row order is part of the key on purpose. TabICL has no positional term, but its float reductions are
	// not order-invariant: on the real weights the combined graph's answers for a permuted support table
	// move by up to 1.6e-4 in log-probability and 5e-5 x std(y), well above the pair's exactness bound
	// (docs/CONTEXT_CACHE_PLAN.md, "Phase 1b"). Serving another order from this entry would answer for the
	// order it was encoded in, not the caller's.
	struct Entry {
		hash_t hash = 0;
		int64_t s = 0;
		int64_t h = 0;
		vector<float> x;
		vector<float> y;
		TabFMPreparedContext state;
		//! state + context rows' outputs + the key: what this entry keeps resident.
		idx_t bytes = 0;
	};
	std::list<Entry> entries;
	//! Recently seen keys: a context is kept the second time its key is seen (anofox_tabfm_context_cache_admit).
	ContextSightings sightings;
	mutex cache_lock;

	TabFMSessionHandle prepare;
	TabFMSessionHandle query;

	static hash_t KeyHash(const float *x, const float *y, int64_t s, int64_t h) {
		auto hx = Hash(reinterpret_cast<const char *>(x), static_cast<size_t>(s * h) * sizeof(float));
		auto hy = Hash(reinterpret_cast<const char *>(y), static_cast<size_t>(s) * sizeof(float));
		return CombineHash(CombineHash(hx, hy), CombineHash(Hash(s), Hash(h)));
	}

	TabFMRunOutput Run(const TabFMRunInput &input) override {
		return RunBatched(input, 0);
	}

	// One lookup (or one encode) per call, whatever query_batch is: the support rows are hashed, compared
	// and copied once, the context rows' outputs are emitted once, and the query rows go through the query
	// graph in batches straight from the caller's buffer -- also on a first sighting, whose context is not
	// kept but still serves every batch of its own call.
	TabFMRunOutput RunBatched(const TabFMRunInput &input, int64_t query_batch) override {
		// The cache's own lock. Every caller also holds the per-device lock across the whole call today,
		// so this is uncontended; it makes the list's safety a property of this type rather than of a
		// caller's lock keying, which once went wrong (#42).
		lock_guard<mutex> guard(cache_lock);
		const int64_t s = input.train_size;
		const int64_t q = input.t - s;
		if (s <= 0 || q < 0) {
			throw InternalException("anofox_tabfm: split model called with train_size=" + std::to_string(s) +
			                        " of T=" + std::to_string(input.t));
		}
		const idx_t budget = input.context_cache_bytes;
		// A budget lowered since the last call applies now, before anything else is kept.
		Trim(budget);

		const auto hash = KeyHash(input.x, input.y, s, input.h);
		// Every lookup is a sighting, hits included: a context that is in use stays "seen", so if it is
		// evicted and comes back it is kept again at once.
		const bool seen_before = sightings.Sight(hash);
		auto hit = Find(hash, input, s);
		// Kept here for a context that is not cached (a first sighting, or larger than the budget): it lives
		// for this whole call, so every batch of the call uses it and it is encoded once.
		Entry uncached;
		const Entry *context;
		if (hit != entries.end()) {
			// Most recently used goes to the front; eviction takes from the back.
			if (hit != entries.begin()) {
				entries.splice(entries.begin(), entries, hit);
			}
			context = &entries.front();
			stats->hits++;
		} else {
			// Built in full before it is published, so a prepare that throws leaves the cache as it was
			// rather than holding a half-built context that a later call could match.
			Entry fresh;
			fresh.hash = hash;
			fresh.s = s;
			fresh.h = input.h;
			auto wanted = context_inputs;
			wanted.push_back("context_logits");
			fresh.state = RunPrepare(*prepare, input.x, input.y, s, input.h, wanted);
			stats->encodes++;
			TakeContextLogits(fresh.state, s);
			const auto x_count = static_cast<size_t>(s) * static_cast<size_t>(input.h);
			fresh.x.assign(input.x, input.x + x_count);
			fresh.y.assign(input.y, input.y + static_cast<size_t>(s));
			fresh.bytes = fresh.state.bytes + (fresh.x.size() + fresh.y.size()) * sizeof(float);
			stats->misses++;
			if (!seen_before) {
				stats->first_seen++;
			}
			const bool admit = seen_before || input.context_cache_admit_first;
			if (admit && fresh.bytes <= budget) {
				Trim(budget - fresh.bytes);
				entries.push_front(std::move(fresh));
				stats->bytes += entries.front().bytes;
				stats->entries++;
				context = &entries.front();
			} else {
				uncached = std::move(fresh);
				context = &uncached;
			}
		}

		const int64_t c = context->state.context_classes;
		TabFMRunOutput out;
		out.shape = {1, input.t, c};
		out.logits.reserve(static_cast<size_t>(input.t) * static_cast<size_t>(c));
		out.logits.insert(out.logits.end(), context->state.context_logits.begin(),
		                  context->state.context_logits.end());
		const int64_t batch = query_batch > 0 ? query_batch : q;
		for (int64_t offset = 0; offset < q; offset += batch) {
			const int64_t n = MinValue<int64_t>(batch, q - offset);
			auto scored = RunQuery(*query, input.x + (s + offset) * input.h, n, input.h, input.y, s, context->state);
			if (scored.shape.size() != 3 || scored.shape[1] != n || scored.shape.back() != c) {
				throw InvalidInputException(
				    "anofox_tabfm: the query graph returned [%s] for %lld query rows; expected [1, %lld, %lld].",
				    ShapeString(scored.shape), static_cast<long long>(n), static_cast<long long>(n),
				    static_cast<long long>(c));
			}
			out.logits.insert(out.logits.end(), scored.logits.begin(), scored.logits.end());
		}
		return out;
	}

	void TrimContextCache(idx_t budget) override {
		lock_guard<mutex> guard(cache_lock);
		Trim(budget);
	}

private:
	std::list<Entry>::iterator Find(hash_t hash, const TabFMRunInput &input, int64_t s) {
		const auto x_count = static_cast<size_t>(s) * static_cast<size_t>(input.h);
		for (auto it = entries.begin(); it != entries.end(); ++it) {
			if (it->hash != hash || it->s != s || it->h != input.h || it->x.size() != x_count ||
			    it->y.size() != static_cast<size_t>(s)) {
				continue;
			}
			if (std::memcmp(it->x.data(), input.x, x_count * sizeof(float)) == 0 &&
			    std::memcmp(it->y.data(), input.y, static_cast<size_t>(s) * sizeof(float)) == 0) {
				return it;
			}
		}
		return entries.end();
	}

	//! Evict least recently used entries until the rest fit in `limit` bytes.
	void Trim(idx_t limit) {
		while (!entries.empty() && stats->bytes.load() > limit) {
			stats->bytes -= entries.back().bytes;
			stats->entries--;
			entries.pop_back();
		}
	}

	// Move the prepare graph's `context_logits` out of the named state into the fitted values. It is not a
	// query input, so keeping it among the named tensors would only feed it nowhere. It stays counted in
	// `bytes`: it is still resident.
	static void TakeContextLogits(TabFMPreparedContext &prepared, int64_t s) {
		for (idx_t i = 0; i < prepared.names.size(); i++) {
			if (prepared.names[i] != "context_logits") {
				continue;
			}
			const auto &shape = prepared.shapes[i];
			if (shape.size() != 3 || shape[0] != 1 || shape[1] != s) {
				throw InvalidInputException("anofox_tabfm: the prepare graph's context_logits is shaped [%s] for %lld "
				                            "context rows; expected [1, %lld, C].",
				                            ShapeString(shape), static_cast<long long>(s), static_cast<long long>(s));
			}
			prepared.context_classes = shape[2];
			prepared.context_logits = std::move(prepared.data[i]);
			prepared.names.erase(prepared.names.begin() + NumericCast<int64_t>(i));
			prepared.shapes.erase(prepared.shapes.begin() + NumericCast<int64_t>(i));
			prepared.data.erase(prepared.data.begin() + NumericCast<int64_t>(i));
			return;
		}
		throw InternalException("anofox_tabfm: the prepare graph declared context_logits but did not return it");
	}
};

// Register a freshly built backend in the DB-instance state and return the
// LoadedModel snapshot. The void handle aliases the TabFMBackend base pointer so
// the predict loop can recover it and dispatch Run() virtually.
shared_ptr<LoadedModel> RegisterBackend(TabFMState &state, const string &cache_key,
                                        shared_ptr<TabFMBackend> backend, const string &device_id, idx_t bytes,
                                        bool split_context = false, const string &precision = "",
                                        idx_t max_sessions = 0,
                                        shared_ptr<ContextCacheStats> context_stats = nullptr) {
	auto model = make_shared_ptr<LoadedModel>();
	model->model_key = cache_key;
	model->session = shared_ptr<void>(std::move(backend));
	model->device_id = device_id;
	model->precision = precision;
	model->dtype = "f32";
	model->bytes = bytes;
	model->split_context = split_context;
	model->context_stats = std::move(context_stats);
	state.Register(cache_key, model, max_sessions);
	return model;
}

// Sha256Hex is defined past this anonymous namespace: it is declared in
// tabfm_predict.hpp so the plugin download can verify its published digest.

// Read the safetensors JSON header bytes ([8, 8+header_len)). Returns false on
// any I/O or sanity failure (caller falls back to injection).
bool ReadWeightsHeaderBytes(FileSystem &fs, const string &path, string &header) {
	try {
		auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
		uint64_t header_len = 0;
		fs.Read(*handle, &header_len, 8, 0);
		if (header_len == 0 || header_len > (1ULL << 26)) { // >64 MiB header is nonsense
			return false;
		}
		header.resize(header_len);
		fs.Read(*handle, (void *)header.data(), NumericCast<int64_t>(header_len), 8);
		return true;
	} catch (...) {
		return false;
	}
}

// True iff the downloaded safetensors' JSON header is byte-identical to the
// checkpoint the bundled external-data / migraphx graphs were baked against — so
// the graphs' baked external offsets are correct. Also requires the weights to be
// a local file named model.safetensors (the graphs' external-data location).
// The (model, task)-keyed hash table lives in tabfm_model_spec.hpp
// (ExpectedWeightsHeaderShaFor) beside the bundled-id naming, since GPU graph
// selection is a model-spec concern shared by three backends.
// WeightsHeaderMatches is defined just past this anonymous namespace: it is
// declared in tabfm_predict.hpp so tabfm_backends() asks dispatch's exact
// question rather than a lookalike, and an anonymous-namespace copy here would
// make every call in this file ambiguous between the two.

// Stage a bundled graph next to the weights (idempotent by size) so external-data
// "model.safetensors" resolves. Returns false if it cannot be written.
bool StageBundledGraph(FileSystem &fs, const BundledResource &graph, const string &graph_path) {
	// Freshness by CONTENT, not by size.
	//
	// This compared byte sizes alone, which is a stale-graph hazard with a very
	// quiet failure: a re-export that happens to preserve the size leaves the
	// old graph on every existing user's disk forever. The .mxr program cache is
	// keyed on the graph's content hash, so the stale graph would then compile
	// to a consistently wrong program and keep serving it -- the exact shape of
	// the bug that made content hashing the .mxr key in the first place.
	//
	// The tabdpt re-export in this branch changed sizes (660509 -> 659729
	// bytes), so it escaped by luck rather than by design. One FNV-1a pass over
	// a few MB, once per session load, buys that back.
	try {
		if (fs.FileExists(graph_path)) {
			auto h = fs.OpenFile(graph_path, FileFlags::FILE_FLAGS_READ);
			const auto on_disk = NumericCast<idx_t>(fs.GetFileSize(*h));
			if (on_disk == graph.size) {
				string existing;
				existing.resize(on_disk);
				idx_t read = 0;
				while (read < on_disk) {
					auto got = h->Read(const_cast<char *>(existing.data()) + read, on_disk - read);
					if (got <= 0) {
						break;
					}
					read += NumericCast<idx_t>(got);
				}
				if (read == on_disk &&
				    anofox_tabfm_mxr::Fnv1a64(existing) ==
				        anofox_tabfm_mxr::Fnv1a64(string(graph.data, NumericCast<size_t>(graph.size)))) {
					return true;
				}
			}
		}
	} catch (...) { // NOLINT: any probe failure just means "rewrite"
	}
	std::ofstream out(graph_path, std::ios::binary | std::ios::trunc);
	out.write(graph.data, NumericCast<std::streamsize>(graph.size));
	out.close();
	return static_cast<bool>(out);
}

// Low-memory load path: the graph references the weights as ONNX external-data on
// disk, so ORT reads them itself (no in-memory injection, no copy). Peak RSS
// drops ~2.6x on the real model. Only engages when a bundled external-data graph
// exists for the task AND the downloaded safetensors header matches exactly
// (else the baked offsets could be wrong -> fall back to injection). Returns
// nullptr to signal "fall back".
// Defined further down, where the per-model 'auto' logic lives; declared here
// because every dispatch branch below must agree with it. Re-resolving with
// ResolveDevice(ctx.device, ...) instead would answer a DIFFERENT question --
// the session-wide one -- so under 'auto' the session would be built for one
// device while dispatch looked for another, and no GPU branch would ever fire.
const string &ResolvedDeviceCached(FileSystem &fs, const ResolvedModel &resolved, const PredictContext &ctx);

//! The discovered device row this (model, task) resolves to. One lookup so no
//! branch can disagree with the session cache about which device it is on.
TabFMDeviceInfo DeviceInfoFor(FileSystem &fs, const ResolvedModel &resolved, const PredictContext &ctx,
                              const vector<TabFMDeviceInfo> &devices) {
	const string &id = ResolvedDeviceCached(fs, resolved, ctx);
	for (auto &device : devices) {
		if (device.device_id == id) {
			return device;
		}
	}
	// The id came from this same list once, so a miss means the topology
	// changed underneath us -- a card removed, a driver reloaded, an NVML probe
	// that failed this time. What to do about it depends entirely on whether
	// the user asked for that device by name.
	if (VanishedDeviceMustThrow(id, ctx.device)) {
		throw InvalidInputException(
		    "anofox_tabfm: device '" + ctx.device + "' resolved to '" + id +
		    "' earlier in this process, but it is no longer among the discovered devices — the card or its driver "
		    "may have gone away. Check SELECT * FROM tabfm_devices(); and restart the process to re-probe, or SET "
		    "anofox_tabfm_device='cpu'.");
	}
	// Under 'auto' the user asked for "whatever works", and the CPU does.
	for (auto &device : devices) {
		if (device.device_id == "cpu") {
			return device;
		}
	}
	TabFMDeviceInfo cpu;
	cpu.device_id = "cpu";
	cpu.ep = "CPUExecutionProvider";
	cpu.usable = true;
	return cpu;
}

shared_ptr<LoadedModel> TryExternalDataSession(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                               const PredictContext &ctx) {
	if (std::getenv("TABFM_DISABLE_EXTERNAL_DATA")) {
		return nullptr;
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	auto graph = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, "ext", task_name));
	if (!graph.data || !WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.model, resolved.manifest.task)) {
		return nullptr;
	}
	// Place the external-data graph next to the weights (idempotent) so ORT can
	// resolve the relative "model.safetensors" reference.
	const auto dir = DirName(resolved.weights_path);
	const auto graph_path = fs.JoinPath(dir, BundledGpuGraphId(resolved.manifest.model, "ext", task_name) + ".onnx");
	if (!StageBundledGraph(fs, graph, graph_path)) {
		return nullptr;
	}

	TabFMSessionConfig config;
	config.intra_op_threads = MaxValue<int64_t>(1, ctx.threads);
	config.prepack = ctx.cpu_prepack;
	auto devices = DiscoverDevices();
	auto device = DeviceInfoFor(fs, resolved, ctx, devices);
	config.device_id = device.device_id;
	config.device_ordinal = device.device_ordinal;
	config.model_tag = task_name;
	config.ep_path = ctx.ep_path;
	// No injected initializers: ORT loads them from the safetensors via the
	// graph's external-data references.
	auto session = CreateSessionFromPath(graph_path, {}, config);
	auto backend = make_shared_ptr<OrtBackend>();
	backend->session = std::move(session);
	return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, 0, false,
	                       SessionPrecisionFor(config.device_id, ctx.gpu_precision),
	                       NumericCast<idx_t>(ctx.max_sessions));
}

// AMD ROCm GPU backend. ORT's MIGraphX EP cannot run this model (re-inlines
// weights -> 2 GB proto), so ROCm bypasses ORT entirely: a standalone plugin
// (docs/DYNAMIC_BACKENDS.md phase 1, src/tabfm_migraphx_plugin.cpp) parses the
// migraphx-ready graph (external-data + Shape-rewrite) directly and compiles
// per shape-bucket (cached to .mxr). Engages only when the resolved device is
// a rocm GPU and a bundled migraphx graph + matching weights exist; nullptr
//! Servability for a resolved model on a backend: the pure predicate in
//! tabfm_model_spec.hpp plus the one fact only the filesystem can answer —
//! whether the directory a BUNDLED graph must be staged into is even there.
//!
//! Deliberately existence, not writability. Answering "can I write here?"
//! honestly means attempting a write, and this runs on the capability path
//! that tabfm_backends() and device resolution call for every (model, device)
//! pair — creating and deleting probe files across the catalog to answer a
//! question staging itself answers a moment later. So a read-only directory
//! is still caught at staging, where it throws for an explicit request; what
//! this rules out cheaply is the case where the directory is absent
//! altogether. The distinct reason string exists because reporting either as
//! "no graph" would send the user to register one, which would not help.
GpuServabilityResult GpuServabilityFor(FileSystem &fs, const ResolvedModel &resolved, const string &backend,
                                       const string &task_name, bool model_provides_graph, bool bundled_graph_exists,
                                       bool bundled_header_matches, const string &precision,
                                       vector<string> unsupported_ops = {}, string known_wrong = string()) {
	GpuServabilityInputs in;
	in.backend = backend;
	in.model = resolved.manifest.model;
	in.task_name = task_name;
	in.model_provides_graph = model_provides_graph;
	in.bundled_graph_exists = bundled_graph_exists;
	in.bundled_header_matches = bundled_header_matches;
	in.precision = precision;
	in.unsupported_ops = std::move(unsupported_ops);
	in.known_wrong = std::move(known_wrong);
	// Only a bundled graph needs staging; a model-provided one runs where it
	// already is, so do not fail it on someone else's directory.
	in.weights_dir_stageable = true;
	if (!model_provides_graph && bundled_graph_exists && bundled_header_matches) {
		const auto weights_dir = DirName(resolved.weights_path);
		in.weights_dir_stageable = weights_dir.empty() || fs.DirectoryExists(weights_dir);
	}
	return EvaluateGpuServability(in);
}

//! A failure AFTER a device was committed to, phrased for whoever chose it.
//!
//! These paths are reached when the cheap capability check passed and
//! something later did not -- most often a bundled graph that could not be
//! staged beside read-only weights. They stay hard errors under 'auto' as well
//! as under an explicit request, because degrading here is the silent fallback
//! the whole design forbids. What has to differ is the TEXT: telling someone
//! who never typed a device name that "device 'mlx' was requested" is
//! confusing, and pointing them at a graph registration will not fix a
//! directory they cannot write.
string GpuPostCommitFailure(const string &device, const string &device_setting, const string &model,
                            const string &what_failed, const string &fix) {
	const bool explicit_request = IsExplicitGpuRequest(device_setting, BackendOfDeviceId(device));
	const string who = explicit_request
	                       ? "device '" + device + "' was requested"
	                       : "'auto' selected device '" + device + "' for model '" + model +
	                             "', because that is the best device this model can be served on";
	return "anofox_tabfm: " + who + ", but " + what_failed + ". " + fix +
	       " Or SET anofox_tabfm_device='cpu' to run this model on the CPU.";
}

//! Wrap a servability reason as the user-facing refusal for an explicitly
//! requested device. Keeps the "here is the device, here is why, here is the
//! way out" shape every §5 error has.
string GpuRefusalMessage(const string &device, const string &reason) {
	return "anofox_tabfm: device '" + device + "' was requested, but " + reason +
	       ". SET anofox_tabfm_device='cpu' to run this model on the CPU, or SELECT * FROM tabfm_backends() to see "
	       "which devices can serve it.";
}

// => fall back to the CPU/ORT path (no migraphx graph shipped for this task).
//
// Past the point where a migraphx graph is confirmed to exist for a resolved
// rocm device, this must succeed or throw: silently falling back to CPU here
// would be exactly the "requested device quietly becomes CPU" failure mode
// the equivalence suite's tier 4 exists to catch (DYNAMIC_BACKENDS.md).
//! Refuses a call whose padded bucket is over the model's ROCm ceiling (RocmBucketRefusal), before the
//! plugin starts a MIGraphX compile that would abort the process on a failed VRAM allocation. Wraps the
//! plugin backend, so every route into a compile goes through it: a predict and tabfm_gpu_precompile.
class RocmCeilingBackend : public TabFMBackend {
public:
	RocmCeilingBackend(shared_ptr<TabFMBackend> inner, string model, int64_t vram_total)
	    : inner(std::move(inner)), model(std::move(model)), vram_total(vram_total) {
	}
	TabFMRunOutput Run(const TabFMRunInput &input) override {
		Check(input.t, input.h);
		return inner->Run(input);
	}
	void Precompile(int64_t rows, int64_t features) override {
		Check(rows, features);
		inner->Precompile(rows, features);
	}

private:
	void Check(int64_t rows, int64_t features) const {
		auto bucket = PadToShapeBucket(rows, features);
		auto reason = RocmBucketRefusal(model, bucket.padded_t, bucket.padded_h, vram_total);
		if (!reason.empty()) {
			throw InvalidInputException("anofox_tabfm: " + reason + ".");
		}
	}
	shared_ptr<TabFMBackend> inner;
	string model;
	int64_t vram_total;
};

shared_ptr<LoadedModel> TryMIGraphXBackend(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                           const PredictContext &ctx) {
	auto devices = DiscoverDevices();
	auto device = DeviceInfoFor(fs, resolved, ctx, devices);
	if (!StringUtil::StartsWith(device.device_id, "rocm")) {
		return nullptr; // not the GPU path (cpu / cuda handled by the ORT backend)
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	// Graph source (docs/GPU_HARDENING_PLAN.md P3): a model-provided
	// migraphx_graph wins; the bundled tabfm-v1 graph needs its header match.
	auto graph = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, "migraphx", task_name));
	const bool bundled_matches =
	    graph.data && WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.model, resolved.manifest.task);
	// One predicate, shared with tabfm_backends() and with device resolution,
	// so a refusal here and the matrix a user queries cannot disagree.
	auto servability = GpuServabilityFor(fs, resolved, "rocm", task_name, !resolved.migraphx_graph_path.empty(),
	                                     graph.data != nullptr, bundled_matches, ctx.gpu_precision);
	if (!servability.supported) {
		// Explicitly-requested ROCm + no runnable graph is an error here, with
		// the real cause; declining silently used to surface a downstream
		// message blaming ep_path (found running the examples on GPU hardware).
		if (IsExplicitGpuRequest(ctx.device, "rocm")) {
			throw InvalidInputException(GpuRefusalMessage("rocm", servability.reason));
		}
		return nullptr;
	}
	string graph_path;
	string weights_dir;
	if (servability.source == GpuGraphSource::MODEL_PROVIDED) {
		graph_path = resolved.migraphx_graph_path;
		weights_dir = DirName(graph_path); // external data sits beside the graph
	} else {
		weights_dir = DirName(resolved.weights_path);
		graph_path = fs.JoinPath(weights_dir, BundledGpuGraphId(resolved.manifest.model, "migraphx", task_name) + ".onnx");
		if (!StageBundledGraph(fs, graph, graph_path)) {
			// Servability said the directory was writable; if staging still
			// failed the cause is not one a capability check can predict.
			// Loud under 'auto' too: the capability check said this model was
			// servable here, so falling back now would be the silent
			// degradation the design forbids.
			throw IOException(GpuPostCommitFailure(
			    device.device_id, ctx.device, resolved.manifest.model,
			    "the bundled GPU graph could not be staged beside the weights at '" + graph_path +
			        "' (read-only weights directory?)",
			    "Make that directory writable."));
		}
	}
	const auto mxr_dir = fs.JoinPath(ctx.cache_dir, "migraphx");

	if (ctx.ep_path.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: device 'rocm' was resolved but no backend plugin directory is configured. SET "
		    "anofox_tabfm_ep_path to the directory holding libanofox_tabfm_migraphx_plugin.so.");
	}
	const auto plugin_path = fs.JoinPath(ctx.ep_path, PluginFileName("rocm"));

	TabFMPluginCreateParams params {};
	params.graph_path = graph_path.c_str();
	params.weights_dir = weights_dir.c_str();
	params.cache_dir = mxr_dir.c_str();
	params.arch = device.arch.c_str();
	params.precision = ctx.gpu_precision.c_str();
	params.mxr_source = ctx.mxr_source.c_str();
	params.device_ordinal = device.device_ordinal;
	shared_ptr<TabFMBackend> backend = LoadPluginBackend(plugin_path, params);
	if (!RocmBucketCeilingNote(resolved.manifest.model).empty()) {
		backend = make_shared_ptr<RocmCeilingBackend>(std::move(backend), resolved.manifest.model, device.vram_total);
	}
	return RegisterBackend(state, resolved.cache_key, std::move(backend), device.device_id, 0, false,
	                       ctx.gpu_precision, NumericCast<idx_t>(ctx.max_sessions));
}

// NVIDIA CUDA GPU backend. Structurally identical to the ROCm path above, and
// for a related reason: GPU inference cannot run inside this binary's ORT.
// The release build links ORT statically, and a static core cannot load ORT's
// prebuilt provider libraries — their classes collide with the core's own and
// the executable's copies win symbol resolution, corrupting the heap mid-Run().
// So CUDA also runs through a standalone plugin (src/tabfm_cuda_plugin.cpp)
// that links its own shared ORT-GPU distribution, where core and providers
// match by construction. See the CUDA section of docs/DYNAMIC_BACKENDS.md.
//
// Reuses the same external-data graph the CPU low-memory path stages, since
// ORT resolves the weights itself. Engages only when the resolved device is a
// cuda GPU and that graph + matching weights exist; nullptr => fall back.
//
// As with ROCm, past the point where the graph is confirmed to exist for a
// resolved cuda device this must succeed or throw — quietly running on CPU
// after the user asked for a GPU is the tier-4 failure the equivalence suite
// exists to catch.
shared_ptr<LoadedModel> TryCudaBackend(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                       const PredictContext &ctx) {
	auto devices = DiscoverDevices();
	auto device = DeviceInfoFor(fs, resolved, ctx, devices);
	if (!StringUtil::StartsWith(device.device_id, "cuda")) {
		return nullptr; // not the CUDA path (cpu handled by the ORT backend, rocm above)
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	// Graph source (docs/GPU_HARDENING_PLAN.md P3): a model-provided ext_graph
	// wins; the bundled tabfm-v1 graph needs its header match.
	auto graph = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, "ext", task_name));
	const bool bundled_matches =
	    graph.data && WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.model, resolved.manifest.task);
	auto servability = GpuServabilityFor(fs, resolved, "cuda", task_name, !resolved.ext_graph_path.empty(),
	                                     graph.data != nullptr, bundled_matches, ctx.gpu_precision, {},
	                                     CudaKnownWrong(resolved.manifest.model, task_name));
	if (!servability.supported) {
		// Same contract as the ROCm branch above: an explicit 'cuda' with no
		// runnable graph names the model and the fix, never ep_path.
		if (IsExplicitGpuRequest(ctx.device, "cuda")) {
			throw InvalidInputException(GpuRefusalMessage("cuda", servability.reason));
		}
		return nullptr;
	}
	string graph_path;
	string weights_dir;
	if (servability.source == GpuGraphSource::MODEL_PROVIDED) {
		graph_path = resolved.ext_graph_path;
		weights_dir = DirName(graph_path); // external data sits beside the graph
	} else {
		weights_dir = DirName(resolved.weights_path);
		graph_path = fs.JoinPath(weights_dir, BundledGpuGraphId(resolved.manifest.model, "ext", task_name) + ".onnx");
		if (!StageBundledGraph(fs, graph, graph_path)) {
			// Loud under 'auto' too: the capability check said this model was
			// servable here, so falling back now would be the silent
			// degradation the design forbids.
			throw IOException(GpuPostCommitFailure(
			    device.device_id, ctx.device, resolved.manifest.model,
			    "the bundled GPU graph could not be staged beside the weights at '" + graph_path +
			        "' (read-only weights directory?)",
			    "Make that directory writable."));
		}
	}

	if (ctx.ep_path.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: device 'cuda' was resolved but no backend plugin directory is configured. SET "
		    "anofox_tabfm_ep_path to the directory holding libanofox_tabfm_cuda_plugin.so (CALL "
		    "tabfm_download_runtime('cuda') to fetch it).");
	}
	const auto plugin_path = fs.JoinPath(ctx.ep_path, PluginFileName("cuda"));

	TabFMPluginCreateParams params {};
	params.graph_path = graph_path.c_str();
	params.weights_dir = weights_dir.c_str();
	params.cache_dir = ctx.cache_dir.c_str();
	params.arch = device.arch.c_str();
	params.precision = ctx.gpu_precision.c_str();
	params.mxr_source = "";
	params.device_ordinal = device.device_ordinal;
	shared_ptr<TabFMBackend> backend = LoadPluginBackend(plugin_path, params);
	return RegisterBackend(state, resolved.cache_key, std::move(backend), device.device_id, 0, false,
	                       ctx.gpu_precision, NumericCast<idx_t>(ctx.max_sessions));
}

// Apple MLX backend (docs/MLX_PLAN.md; spike verdicts in
// docs/MLX_SPIKE_RESULTS.md). A plugin like the two above, but the resemblance
// stops at the loader: MLX has no ONNX importer and none exists, so the plugin
// runs a forward TRANSCRIBED from the model's reference implementation rather
// than any graph we ship. Two consequences shape this function:
//
//   * there is no graph to select. SelectGpuGraph / StageBundledGraph /
//     WeightsHeaderMatches have nothing to do here — the plugin mmaps the same
//     cached safetensors every backend reads, addressed by tensor name.
//   * it serves the models somebody has hand-ported, not every model with a
//     graph. Asking for one it does not implement has to be an error naming
//     what IS implemented, because the alternative — falling through to the
//     CPU path — is the tier-4 "requested device quietly becomes CPU" failure.
//
// Since 'auto' never resolves to mlx (tabfm_devices.cpp keeps the plugin lanes
// explicit), reaching here at all means the user asked for it by name, so every
// failure below throws rather than returning nullptr.
shared_ptr<LoadedModel> TryMlxBackend(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                      const PredictContext &ctx) {
	auto devices = DiscoverDevices();
	auto device = DeviceInfoFor(fs, resolved, ctx, devices);
	if (!StringUtil::StartsWith(device.device_id, "mlx")) {
		return nullptr; // not the MLX path
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);

	// The MLX backend executes the model's ONNX graph, so it needs the same
	// external-data graph the CUDA plugin runs -- selected the same way, since
	// the rule ("a model-provided graph wins; the bundled one needs its header
	// to match the downloaded weights") is a property of the graphs, not of the
	// accelerator. mitra is the one exception: it also has a hand-ported
	// forward, which the plugin uses when no graph is supplied.
	auto graph = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, "ext", task_name));
	const bool bundled_matches =
	    graph.data && WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.model, resolved.manifest.task);
	string graph_path;
	string weights_dir = DirName(resolved.weights_path);
	switch (SelectGpuGraph(!resolved.ext_graph_path.empty(), graph.data != nullptr, bundled_matches)) {
	case GpuGraphSource::MODEL_PROVIDED:
		graph_path = resolved.ext_graph_path;
		weights_dir = DirName(graph_path); // external data sits beside the graph
		break;
	case GpuGraphSource::BUNDLED:
		graph_path = fs.JoinPath(weights_dir, BundledGpuGraphId(resolved.manifest.model, "ext", task_name) + ".onnx");
		if (!StageBundledGraph(fs, graph, graph_path)) {
			graph_path.clear();
		}
		break;
	case GpuGraphSource::NONE:
		graph_path.clear();
		break;
	}
	// No graph and no hand-port means nothing to run, so this must say so
	// rather than fall through to CPU -- the tier-4 contract. It used to be
	// written as though only an explicit 'mlx' could reach it, which was true
	// while 'auto' never resolved to mlx and is not any more: servability is
	// checked before auto commits, but staging can still fail afterwards on a
	// weights directory that is not writable.
	// The plugin would refuse such a graph too, but only after being loaded; say it here, with the way out.
	if (!graph_path.empty()) {
		auto missing = MlxUnsupportedOps(nullptr, 0, graph_path);
		auto known_wrong = MlxKnownWrong(resolved.manifest.model, task_name);
		if (!missing.empty() || !known_wrong.empty()) {
			GpuServabilityInputs in;
			in.backend = "mlx";
			in.model = resolved.manifest.model;
			in.task_name = task_name;
			in.precision = ctx.gpu_precision;
			in.model_provides_graph = true; // a graph exists; the ops are what block it
			in.unsupported_ops = std::move(missing);
			in.known_wrong = std::move(known_wrong);
			throw InvalidInputException(GpuRefusalMessage(device.device_id, EvaluateGpuServability(in).reason));
		}
	}
	if (graph_path.empty() && resolved.manifest.model != "mitra") {
		throw InvalidInputException(GpuPostCommitFailure(
		    device.device_id, ctx.device, resolved.manifest.model,
		    "no runnable graph for task '" + task_name +
		        "' could be prepared — the bundled graph could not be staged beside the weights, or none matches "
		        "them",
		    "Make the weights directory writable, or register the model with " + task_name +
		        "_ext_graph := '<graph.onnx>'."));
	}
	if (ctx.ep_path.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: device 'mlx' was resolved but no backend plugin directory is configured. SET "
		    "anofox_tabfm_ep_path to the directory holding libanofox_tabfm_mlx_plugin.dylib (CALL "
		    "tabfm_download_runtime('mlx') to fetch it).");
	}
	const auto plugin_path = fs.JoinPath(ctx.ep_path, PluginFileName("mlx"));

	// `arch` is backend-defined (see tabfm_plugin_abi.h): the GPU plugins read a
	// compute architecture there, but MLX's hardware needs no such selector. It
	// carries "<model>-<task>" so the plugin can pick mitra's hand-ported fast
	// path, and so a task/weights mismatch is caught rather than computed.
	const string arch = resolved.manifest.model + "-" + task_name;

	TabFMPluginCreateParams params {};
	params.graph_path = graph_path.c_str(); // no graph: the forward is transcribed, not loaded
	params.weights_dir = weights_dir.c_str();
	params.cache_dir = ctx.cache_dir.c_str();
	params.arch = arch.c_str();
	params.precision = ctx.gpu_precision.c_str();
	params.mxr_source = "";
	params.device_ordinal = device.device_ordinal;
	shared_ptr<TabFMBackend> backend = LoadPluginBackend(plugin_path, params);

	// Unlike the CUDA and ROCm registrations above, which report 0 bytes because
	// their weights live in VRAM the host never counts, Apple Silicon's memory is
	// UNIFIED: the plugin's tensors are ordinary resident pages of this process,
	// competing with DuckDB's buffer manager for the same pool. Reporting 0 would
	// make tabfm_models() claim a multi-hundred-MB session costs nothing, on the
	// one backend where that number is real. (The anofox_tabfm_max_memory gates
	// read process RSS rather than this field, so they already see MLX correctly
	// -- arguably better than they see a discrete GPU.)
	idx_t weight_bytes = 0;
	try {
		auto handle = fs.OpenFile(resolved.weights_path, FileFlags::FILE_FLAGS_READ);
		weight_bytes = NumericCast<idx_t>(fs.GetFileSize(*handle));
	} catch (const std::exception &) {
		weight_bytes = 0; // size is a diagnostic, never a reason to fail a load
	}
	return RegisterBackend(state, resolved.cache_key, std::move(backend), device.device_id, weight_bytes, false,
	                       SessionPrecisionFor(device.device_id, ctx.gpu_precision),
	                       NumericCast<idx_t>(ctx.max_sessions));
}

// The device a setting resolves to, memoized per distinct setting string.
//
// `DeviceMutex` must be keyed on this rather than on the setting, or two
// connections whose settings differ textually while resolving to the same
// physical device -- 'auto' and 'cpu' being the everyday pair -- take DIFFERENT
// mutexes while sharing one backend under `resolved.cache_key`, and stop
// excluding each other. That was harmless when the only shared mutable state was
// the forward-cost number; SplitOrtBackend added a labelled-context cache that is
// cleared and rebuilt in place on a miss, so two racing connections with
// different context tables could observe a half-rebuilt one -- UB on the vectors,
// not merely a wrong answer.
//
// Memoized because the fix has to work on the hot path: `DiscoverDevices()`
// dlopens NVML and calls nvmlInit on every invocation, which is fine once per
// session load and not fine once per forward pass.
//
// Process-wide rather than per-database-instance on purpose: the device topology
// belongs to the machine, not to a DuckDB instance, and two instances in one
// process should agree about it. It is deliberately NOT used by `tabfm_devices()`,
// so that diagnostic keeps probing live hardware.
//! Is the plugin for `backend` present at `ep_path` and of an ABI we speak?
//! Memoized per (ep_path, backend): the answer is a property of the
//! filesystem, and the probe dlopens a library that is deliberately never
//! unloaded, so asking once per predict would map it repeatedly.
//!
//! A consequence worth stating plainly: fetching a plugin mid-session does not
//! change what 'auto' does until the next connection. That is the deliberate
//! trade -- the alternative is invalidating a process-wide memo that
//! DeviceMutex also keys on, which is the area issue #42 was about.
//! tabfm_accelerate() says so in its output.

//! Resolve the device for THIS model, not merely for this session.
//!
//! An explicit request is unchanged: ResolveDevice decides, and every
//! downstream refusal stays a hard error via IsExplicitGpuRequest.
//!
//! 'auto' is where this differs, and the difference is the point. The old
//! 'auto' could only ever pick a lane the compiled FLAVOR carried, so on the
//! released cpu-flavor builds it was always cpu -- a user with a card and a
//! plugin installed still got the CPU, on every platform. Making it pick a GPU
//! session-wide would have been worse than useless: the codebase's
//! anti-silent-fallback contract is enforced by IsExplicitGpuRequest, a string
//! compare on the raw setting, whose asymmetry is deliberate -- an explicit
//! request hard-errors where 'auto' declines quietly. So the moment 'auto'
//! resolved to rocm, the nine catalog models with no MIGraphX graph would have
//! been served on the CPU with nothing said.
//!
//! Resolving per (model, task) dissolves that rather than negotiating with it.
//! 'auto' only ever names a device this model can actually be served on, so
//! there is no fallback event left to report: tabfm-v1 goes to rocm and tabdpt
//! goes to cpu in the same session, each because that is the best device that
//! model supports. IsExplicitGpuRequest is untouched.
string ResolveDeviceForModel(FileSystem &fs, const ResolvedModel &resolved, const PredictContext &ctx) {
	auto devices = DiscoverDevices();
	if (StringUtil::Lower(ctx.device) != "auto") {
		return ResolveDevice(ctx.device, devices).device_id;
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	for (auto &device : devices) {
		if (device.device_id == "cpu" || !device.usable) {
			continue;
		}
		const string backend = BackendOfDeviceId(device.device_id);
		// coreml is an in-process ORT provider, not a plugin, and is unverified
		// (docs/DYNAMIC_BACKENDS.md). 'auto' does not reach for it.
		if (backend != "cuda" && backend != "rocm" && backend != "mlx") {
			continue;
		}
		// Is the lane even here? Cheap, local, and never create(): a plugin
		// that is absent simply means this lane is not a candidate, which
		// cannot turn a working CPU install into a failing one.
		if (!PluginAvailableCached(ctx.ep_path, backend)) {
			continue;
		}
		const bool is_rocm = backend == "rocm";
		const string kind = is_rocm ? "migraphx" : "ext";
		const string &model_graph = is_rocm ? resolved.migraphx_graph_path : resolved.ext_graph_path;
		auto bundled = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, kind, task_name));
		const bool header_matches = bundled.data && WeightsHeaderMatches(fs, resolved.weights_path,
		                                                                 resolved.manifest.model,
		                                                                 resolved.manifest.task);
		// MLX interprets the graph op by op; one it cannot execute must not win the lane here and then
		// refuse at load, when the CPU could have served it.
		vector<string> unsupported_ops;
		if (backend == "mlx") {
			unsupported_ops = !model_graph.empty() ? MlxUnsupportedOps(nullptr, 0, model_graph)
			                  : header_matches     ? MlxUnsupportedOps(bundled.data, bundled.size, string())
			                                       : vector<string>();
		}
		auto verdict = GpuServabilityFor(fs, resolved, backend, task_name, !model_graph.empty(),
		                                 bundled.data != nullptr, header_matches, ctx.gpu_precision,
		                                 std::move(unsupported_ops),
		                                 GpuKnownWrong(backend, resolved.manifest.model, task_name));
		if (verdict.supported) {
			return device.device_id;
		}
	}
	return "cpu";
}

// Memoized because the fix has to work on the hot path: `DiscoverDevices()`
// dlopens NVML and calls nvmlInit on every invocation, which is fine once per
// session load and not fine once per forward pass.
//
// The key carries everything the answer depends on. It used to be the setting
// alone, which was correct only while resolution was a pure function of the
// machine. It is now also a function of the MODEL (a per-model 'auto'), the
// PRECISION (a mode the backend cannot run makes it unservable there) and
// EP_PATH (whether the lane's plugin is present) -- and ep_path is a
// per-connection setting, so leaving it out would let two databases in one
// process inherit whichever raced first.
const string &ResolvedDeviceCached(FileSystem &fs, const ResolvedModel &resolved, const PredictContext &ctx) {
	static mutex memo_lock;
	static map<string, string> memo;
	// The generation leads, so a freshly installed plugin invalidates every
	// prior answer at once (see BumpPluginProbeGeneration).
	const string key = std::to_string(g_plugin_probe_generation.load()) + "\x1f" + ctx.device + "\x1f" +
	                   resolved.manifest.model + "\x1f" + TabFMTaskName(resolved.manifest.task) + "\x1f" +
	                   ctx.gpu_precision + "\x1f" + ctx.ep_path + "\x1f" + ctx.cache_dir + "\x1f" +
	                   resolved.ext_graph_path + "\x1f" + resolved.migraphx_graph_path + "\x1f" +
	                   resolved.weights_path + "\x1f" + (ctx.context_cache ? "1" : "0");
	{
		lock_guard<mutex> guard(memo_lock);
		auto entry = memo.find(key);
		if (entry != memo.end()) {
			return entry->second;
		}
	}
	// Discovery outside the memo lock: it reaches into the driver, and holding a
	// lock across that would serialize unrelated first-time resolutions behind it.
	// Two connections racing here compute the same answer, so the re-lookup on the
	// way back in is a correctness no-op rather than a guard.
	auto device_id = ResolveDeviceForModel(fs, resolved, ctx);
	lock_guard<mutex> guard(memo_lock);
	return memo.emplace(key, std::move(device_id)).first->second;
}

// Build (or reuse) the ORT session for `resolved` and return a snapshot of the
// LoadedModel. Order: MIGraphX plugin (ROCm GPU) -> CUDA plugin (NVIDIA GPU)
// -> low-memory external-data (CPU, ORT reads weights from disk) -> read/mmap
// + in-memory injection. Both GPU backends run in dlopen'd plugins with their
// own runtimes rather than through this binary's ORT (DYNAMIC_BACKENDS.md).
shared_ptr<LoadedModel> LoadOrGetSession(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                         const PredictContext &ctx) {
	// The device the caller is asking for right now. A cached session belongs to
	// the device it was BUILT for, so reusing it whenever the key matches
	// silently ignores anofox_tabfm_device: a connection that ran anything on
	// the cpu and then SET anofox_tabfm_device='cuda' kept getting the cpu
	// session and never touched the GPU. That is precisely the "requested device
	// quietly becomes CPU" outcome the tier-4 contract in
	// docs/DYNAMIC_BACKENDS.md exists to rule out, and it is invisible without
	// looking at tabfm_models().device.
	const string &wanted_device = ResolvedDeviceCached(fs, resolved, ctx);
	// gpu_precision only shapes GPU sessions; for CPU it is "" so flipping the
	// setting does not rebuild a session it never influenced.
	const string wanted_precision = SessionPrecisionFor(wanted_device, ctx.gpu_precision);

	// Does this call take the context cache? An exact pair, the setting on (the default), and the CPU
	// (UseContextCache). Decided BEFORE the session lookup, because a session already built the other way
	// has to be rebuilt rather than reused (see below). A GPU session never looks for the pair at all, so
	// it is built and served exactly as it was before the cache existed.
	SplitGraphs split;
	if (ctx.context_cache && wanted_device == "cpu") {
		split = FindSplitGraphs(fs, resolved);
	}
	const bool want_split = split.Available() && UseContextCache(ctx.context_cache, split.exact, wanted_device);

	// Sessions cache per (model, device, precision) since P5 of
	// docs/GPU_HARDENING_PLAN.md: S6 measured 18–27 s of pure rebuild per
	// device alternation under the old one-slot-per-key design. The lookup is
	// exact, so a device or precision switch is a cache MISS that builds a new
	// entry beside the old one — not a replacement — and tabfm_unload(key)
	// still frees every entry by one name (the two-level map guarantees it).
	//
	// The context-cache (split) session has its own slot beside the combined one, so the two coexist:
	// generation and imputation (which do not use the cache) and a connection with the kill switch on
	// get the combined session without evicting the split one and every context cached on it. They
	// shared one slot until the default flipped on, a leftover of the opt-in, single-context design of
	// #40 rather than a correctness requirement: the entries are independent, and tabfm_unload(key) frees
	// both. anofox_tabfm_max_sessions still bounds the total.
	if (auto snapshot = state.Snapshot(resolved.cache_key, wanted_device, wanted_precision, want_split)) {
		if (CanReuseSession(*snapshot, want_split, wanted_device, wanted_precision)) {
			return snapshot;
		}
	}

	// Apple MLX first: a GPU device never takes the cache path (want_split is false for it), so this is
	// unchanged by the cache.
	if (auto gpu = TryMlxBackend(fs, state, resolved, ctx)) {
		return gpu;
	}

	// The alternative backends below are driven by graphs compiled into the binary, and none of them has a
	// split form. want_split is only ever true on the CPU, so they are reached exactly as before; the
	// guard keeps a CPU split session from being shadowed by the low-memory external-data path.
	if (!want_split) {
		// ROCm GPU: direct MIGraphX backend (bypasses ORT's unusable MIGraphX EP).
		if (auto gpu = TryMIGraphXBackend(fs, state, resolved, ctx)) {
			return gpu;
		}
		// NVIDIA GPU: CUDA backend plugin (its own shared ORT-GPU runtime).
		if (auto gpu = TryCudaBackend(fs, state, resolved, ctx)) {
			return gpu;
		}
		// Low-memory path (external-data graph; ORT reads weights from disk).
		if (auto external = TryExternalDataSession(fs, state, resolved, ctx)) {
			return external;
		}
	}

	// Prefer an mmap of the (local) cache file: ORT reads the injected
	// initializers straight from file-backed pages, so the multi-GB weights are
	// never copied into an anonymous heap buffer. Fall back to a full read for a
	// non-local/virtual path or any OS that cannot map it (a short read there
	// silently zeroes the tail of a multi-GB file — see ReadWholeFile).
	auto mapping = TryMapFile(resolved.weights_path);
	string weights;
	const_data_ptr_t bytes;
	idx_t nbytes;
	if (mapping.valid()) {
		bytes = const_data_ptr_cast(mapping.data);
		nbytes = mapping.size;
	} else {
		weights = ReadWholeFile(fs, resolved.weights_path);
		bytes = const_data_ptr_cast(weights.data());
		nbytes = weights.size();
	}
	// The tensor map is onnx->st; invert it, and fall back to the "m." wrapper
	// prefix for any tensor the map omits.
	unordered_map<string, string> st_to_onnx;
	for (auto &kv : resolved.tensor_map) {
		st_to_onnx[kv.second] = kv.first;
	}
	F32Arena arena; // owns bf16->f32 upcasts (safetensors path); empty for ckpt
	// The checkpoint's state_dict, held to the same lifetime as `arena`: a strided view's elements are
	// gathered into a copy the state_dict owns, and the injected initializers point into it.
	unordered_map<string, CkptTensor> sd;
	vector<TabFMTensorRef> initializers;
	// The checkpoint key each entry of `initializers` came from, same order. The
	// split halves are injected from their OWN tensor maps (each names only the
	// initializers its graph declares), and renaming needs the source key.
	vector<string> initializer_keys;
	if (IsTorchCkpt(bytes, nbytes)) {
		// Native PyTorch checkpoint (.ckpt): recover the state_dict and inject each
		// tensor by graph name. The bytes alias the mmap/read buffer (kept alive by
		// the backend below), so no arena copy is needed for f32/i64.
		sd = ReadTorchCkpt(bytes, nbytes);
		ApplyCkptAliases(resolved.ckpt_aliases, sd);
		// Does this checkpoint's namespace actually match the committed tensor
		// map? For some models it does not: the map is keyed to the names the
		// weights carry only after a `convert_weights.py` pass, and injecting the
		// raw ckpt would fail later with an opaque ORT "initializer not found".
		// Say so here, where we can name the fix.
		//
		// Count from the MAP's side, not the checkpoint's: st_to_onnx's keys are
		// exactly the tensors the graph needs. Even one missing means injection
		// cannot succeed, so a "did anything match?" test is too weak — TabPFN-2.5
		// shares enough names with its raw checkpoint to pass that and still die
		// on an opaque ORT shape error further down.
		if (!st_to_onnx.empty()) {
			idx_t missing = 0;
			string first_missing;
			for (auto &kv : st_to_onnx) {
				if (sd.find(kv.first) == sd.end()) {
					if (missing == 0) {
						first_missing = kv.first;
					}
					missing++;
				}
			}
			if (missing > 0) {
				throw InvalidInputException(
				    "tabfm: the checkpoint at '%s' is missing %llu of the %llu tensors model '%s' needs (e.g. '%s') "
				    "— its tensor names are not the ones the graph was exported against. This checkpoint needs the "
				    "one-time conversion step: run tools/export_*/convert_weights.py for this model, which writes a "
				    "model.safetensors next to it that the engine then prefers automatically. See "
				    "docs/REAL_MODELS.md.",
				    resolved.weights_path, static_cast<unsigned long long>(missing),
				    static_cast<unsigned long long>(st_to_onnx.size()), resolved.manifest.model, first_missing);
			}
		}
		initializers.reserve(sd.size());
		for (auto &entry : sd) {
			if (resolved.unused_weights.count(entry.first)) {
				continue; // in the file, not in this task's graph (tensor map "unused_by_graph")
			}
			const CkptTensor &t = entry.second;
			auto it = st_to_onnx.find(entry.first);
			if (it == st_to_onnx.end() && resolved.tensor_map_complete) {
				continue; // see TensorMapDeclaresComplete
			}
			TabFMTensorRef ref;
			ref.name = it != st_to_onnx.end() ? it->second : ("m." + entry.first);
			if (t.dtype == "f32") {
				ref.dtype = TabFMTensorDtype::F32;
			} else if (t.dtype == "i64") {
				ref.dtype = TabFMTensorDtype::I64;
			} else if (t.dtype == "bool") {
				ref.dtype = TabFMTensorDtype::BOOL;
			} else {
				throw InvalidInputException(
				    "tabfm: checkpoint tensor '%s' has dtype '%s'; the engine injects f32/i64/bool. Re-export the "
				    "weights as float32, or file an issue to add the dtype.",
				    entry.first, t.dtype);
			}
			ref.shape = t.shape;
			ref.data = t.data;
			ref.size_bytes = t.nbytes;
			initializers.push_back(std::move(ref));
			initializer_keys.push_back(entry.first);
		}
	} else {
		// safetensors: build one injected initializer per tensor, named by the
		// graph (ONNX) initializer name.
		auto view = ParseSafetensors(bytes, nbytes, resolved.weights_path);
		arena = MaterializeF32Arena(view);
		initializers.reserve(view.tensors.size());
		for (auto &entry : view.tensors) {
			const string &st_key = entry.first;
			if (resolved.unused_weights.count(st_key)) {
				continue; // in the file, not in this task's graph (tensor map "unused_by_graph")
			}
			auto it = st_to_onnx.find(st_key);
			if (it == st_to_onnx.end() && resolved.tensor_map_complete) {
				continue; // another task's tensor (the map names all of this graph's)
			}
			auto &m = arena.Get(st_key);
			TabFMTensorRef ref;
			ref.name = it != st_to_onnx.end() ? it->second : ("m." + st_key);
			ref.dtype = ToEngineDtype(m.dtype);
			ref.shape = m.shape;
			ref.data = m.data;
			ref.size_bytes = m.nbytes;
			initializers.push_back(std::move(ref));
			initializer_keys.push_back(st_key);
		}
	}

	TabFMSessionConfig config;
	config.intra_op_threads = MaxValue<int64_t>(1, ctx.threads);
	config.prepack = ctx.cpu_prepack;
	auto devices = DiscoverDevices();
	auto device = DeviceInfoFor(fs, resolved, ctx, devices);
	config.device_id = device.device_id;
	config.device_ordinal = device.device_ordinal;
	config.model_tag = TabFMTaskName(resolved.manifest.task);
	config.contract_inputs = resolved.contract_inputs;
	config.contract_outputs = resolved.contract_outputs;
	config.ep_path = ctx.ep_path;

	const idx_t weight_bytes = nbytes;

	if (want_split) {
		// One half's injection set: the initializers ITS graph declares, named as
		// its own tensor map names them. Not a filter on the combined set — the
		// prepare graph is a proper subset of the checkpoint (no predictor head),
		// and ORT rejects an injected initializer the graph does not have
		// ("Failed to find existing initializer"), so each half is built from its
		// own map or not at all.
		unordered_map<string, idx_t> by_key;
		for (idx_t i = 0; i < initializer_keys.size(); i++) {
			by_key[initializer_keys[i]] = i;
		}
		// Driven by the MAP, not by the checkpoint: the map's keys are exactly the
		// initializers this graph declares, and two of them may name the same
		// checkpoint tensor (tied weights), which a checkpoint-side walk would
		// inject only once and leave the graph short.
		auto for_half = [&](const SplitHalf &side, const char *half) {
			const string source = side.bundled() ? ("bundled tensor_map_" + string(half)) : side.map_path;
			auto json = side.bundled() ? string(side.map_bundle.data, side.map_bundle.size)
			                           : ReadWholeFile(fs, side.map_path);
			auto onnx_to_st = ParseTensorMapJson(json, source);
			if (onnx_to_st.empty()) {
				throw InvalidInputException(
				    "tabfm: the %s half's tensor map '%s' names no initializers, so its weights cannot be injected.",
				    half, source);
			}
			vector<TabFMTensorRef> subset;
			subset.reserve(onnx_to_st.size());
			for (auto &kv : onnx_to_st) {
				auto it = by_key.find(kv.second);
				if (it == by_key.end()) {
					throw InvalidInputException(
					    "tabfm: the %s half's graph needs '%s' (checkpoint tensor '%s'), which '%s' does not contain — "
					    "the split graphs and the weights come from different exports.",
					    half, kv.first, kv.second, resolved.weights_path);
				}
				auto ref = initializers[it->second];
				ref.name = kv.first;
				subset.push_back(std::move(ref));
			}
			return subset;
		};

		// The manifest's tensor_contract describes the COMBINED graph (x/y/logits).
		// The query half legitimately takes more inputs than that, so validating it
		// against the combined contract would reject a correct pair.
		config.contract_inputs.clear();
		config.contract_outputs.clear();

		auto session_for = [&](const SplitHalf &side, const char *half) {
			auto inits = for_half(side, half);
			return side.bundled() ? CreateSession(side.graph_bundle.data, side.graph_bundle.size, inits, config)
			                      : CreateSessionFromPath(side.graph_path, inits, config);
		};

		auto backend = make_shared_ptr<SplitOrtBackend>();
		const string task_name = TabFMTaskName(resolved.manifest.task);
		config.model_tag = task_name + " (prepare)";
		backend->prepare = session_for(split.prepare, "prepare");
		config.model_tag = task_name + " (query)";
		backend->query = session_for(split.query, "query");
		backend->context_inputs = SplitContextInputs(*backend->query);
		if (backend->context_inputs.empty() || !SessionHasOutput(*backend->prepare, "context_logits")) {
			throw InvalidInputException(
			    "tabfm: the prepare/query pair of '%s' does not follow the context-cache contract (prepare must "
			    "emit context_logits, query must take the prepared state). Re-export the pair with "
			    "tools/export_tabicl --split-context, or SET anofox_tabfm_context_cache = false.",
			    resolved.manifest.model);
		}
		backend->weights = std::move(weights);
		backend->mapping = std::move(mapping);
		backend->arena = std::move(arena);
		auto stats = backend->stats;
		return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, weight_bytes, true,
		                       SessionPrecisionFor(config.device_id, ctx.gpu_precision),
		                       NumericCast<idx_t>(ctx.max_sessions), std::move(stats));
	}

	auto session = resolved.graph_bundle.data
	                   ? CreateSession(resolved.graph_bundle.data, resolved.graph_bundle.size, initializers, config)
	                   : CreateSessionFromPath(resolved.graph_path, initializers, config);
	// ORT keeps references to the injected user buffers (the mmap or read buffer +
	// arena) and to the OrtValues (inside the session) — all must OUTLIVE the
	// session, so keep the buffers in the holder alongside it.
	auto backend = make_shared_ptr<OrtBackend>();
	backend->weights = std::move(weights);
	backend->mapping = std::move(mapping);
	backend->arena = std::move(arena);
	backend->session = std::move(session);
	return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, weight_bytes, false,
	                       SessionPrecisionFor(config.device_id, ctx.gpu_precision),
	                       NumericCast<idx_t>(ctx.max_sessions));
}

//===--------------------------------------------------------------------===//
// rows (vector<vector<Value>>) -> ColumnDataCollection for the preprocessor
//===--------------------------------------------------------------------===//

ColumnDataCollection BuildCollection(const vector<vector<Value>> &rows, const LogicalType &row_type,
                                     vector<PreprocessColumnSpec> &out_columns, idx_t target_idx) {
	auto &fields = StructType::GetChildTypes(row_type);
	vector<LogicalType> types;
	out_columns.clear();
	for (idx_t c = 0; c < fields.size(); c++) {
		types.push_back(fields[c].second);
		PreprocessColumnSpec spec;
		spec.name = duckdb_compat::Str(fields[c].first);
		spec.type = fields[c].second;
		spec.is_target = (c == target_idx);
		spec.is_feature = (c != target_idx);
		out_columns.push_back(std::move(spec));
	}

	ColumnDataCollection collection(Allocator::DefaultAllocator(), types);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	for (auto &row : rows) {
		if (chunk.size() == STANDARD_VECTOR_SIZE) {
			collection.Append(chunk);
			chunk.Reset();
		}
		auto idx = chunk.size();
		for (idx_t c = 0; c < types.size(); c++) {
			chunk.SetValue(c, idx, row[c]);
		}
		duckdb_compat::SetChildCardinality(chunk, idx + 1);
	}
	if (chunk.size() > 0) {
		collection.Append(chunk);
	}
	return collection;
}

//===--------------------------------------------------------------------===//
// Decode
//===--------------------------------------------------------------------===//

void SoftmaxInPlace(vector<double> &v, double temperature) {
	double t = temperature > 0 ? temperature : 1.0;
	double m = v.empty() ? 0.0 : v[0];
	for (auto x : v) {
		m = MaxValue(m, x);
	}
	double sum = 0;
	for (auto &x : v) {
		x = std::exp((x - m) / t);
		sum += x;
	}
	if (sum > 0) {
		for (auto &x : v) {
			x /= sum;
		}
	}
}

//===--------------------------------------------------------------------===//
// The engine
//===--------------------------------------------------------------------===//

// The context cache's instance-level settings, read at execution. The budget is GLOBAL-scoped (one cache per
// loaded model serves every connection, so one value bounds it), and the global kill switch applies to
// statements prepared before it was set. Lookups that fail keep the defaults.
void ApplyInstanceContextCacheSettings(DatabaseInstance &db, PredictContext &ctx) {
	Value value;
	if (duckdb_compat::TryGetContextSetting(db, "anofox_tabfm_context_cache", value) && !value.IsNull()) {
		ctx.context_cache = ctx.context_cache && BooleanValue::Get(value);
	}
	if (duckdb_compat::TryGetContextSetting(db, "anofox_tabfm_context_cache_memory", value) && !value.IsNull()) {
		auto raw = value.ToString();
		StringUtil::Trim(raw);
		idx_t bytes = 0;
		if (raw == "0") {
			ctx.context_cache_bytes = 0;
		} else if (StringUtil::TryParseFormattedBytes(raw, bytes).empty()) {
			ctx.context_cache_bytes = bytes;
		}
	}
	if (duckdb_compat::TryGetContextSetting(db, "anofox_tabfm_context_cache_admit", value) && !value.IsNull()) {
		auto raw = StringUtil::Lower(value.ToString());
		StringUtil::Trim(raw);
		ctx.context_cache_admit_first = raw == "first";
	}
}

class TabFMRealEngine : public PredictEngine {
public:
	TabFMPredictResult Predict(const PredictInput &in) override {
		if (in.opts.n_estimators > 1) {
			throw NotImplementedException(
			    "tabfm: n_estimators > 1 (ensemble) is a later milestone (M3); use n_estimators = 1");
		}
		if (!in.ctx.db) {
			throw InternalException("tabfm: predict engine invoked without a database handle");
		}
		auto fs = FileSystem::CreateLocal();
		const auto task =
		    in.opts.task == TabFMTask::CLASSIFICATION ? TabFMTask::CLASSIFICATION : TabFMTask::REGRESSION;

		// Pick the model up front (registry only, no filesystem yet): its
		// preprocessing profile decides whether the engine standardizes features.
		// TabFM/Mitra want the default z-score (Mitra is rank-invariant to it);
		// TabPFN/TabICL normalize INSIDE the graph and must get raw features —
		// they declare a "*_raw" profile, and z-scoring here would double-normalize
		// them. The model's ARTIFACTS are resolved after preprocessing, so a
		// relation that can never be scored says so before "not downloaded".
		auto resolved = ResolveModelSpec(in.ctx, task, in.opts.model);
		const bool standardize = !StringUtil::EndsWith(resolved.manifest.preprocessing_profile, "_raw");

		// 1. preprocess
		vector<PreprocessColumnSpec> columns;
		auto collection = BuildCollection(in.rows, in.row_type, columns, in.target_idx);
		auto pp_task = task == TabFMTask::CLASSIFICATION ? PreprocessTask::CLASSIFICATION : PreprocessTask::REGRESSION;
		auto batch = PreprocessBatch(collection, columns, pp_task, standardize);

		// Nothing survived the unique-feature filter (no feature columns at all,
		// or every one is constant on the training rows). The tensors would be
		// empty and the engine would trip its null-buffer assertion, so say what
		// actually went wrong instead (issue #17).
		if (batch.H == 0) {
			throw InvalidInputException(
			    "tabfm: no usable feature columns — the relation has none besides the target '%s', or every feature "
			    "column is constant across the training rows. Add at least one feature column that varies, or widen "
			    "the features := [...] list.",
			    in.target_name);
		}

		// The data is scorable — now resolve weights/graph/tensor map from disk.
		ResolveModelArtifacts(*fs, in.ctx, task, resolved);

		// The class ceiling is a property of the SELECTED model — its head is
		// exactly this wide — so read it from that model's registry entry rather
		// than asserting a number that happens to be right for the built-ins.
		// Saying "TabFM v1" while running tabicl-v2, or blaming the graph, sends
		// people auditing a registration that is not wrong (#26). A spec that
		// declares nothing keeps the historical ceiling; ValidateTabFMOutput
		// remains the backstop for a graph that contradicts its own spec.
		const idx_t class_ceiling =
		    resolved.max_classes > 0 ? NumericCast<idx_t>(resolved.max_classes) : 10;
		if (task == TabFMTask::CLASSIFICATION && batch.label_decoder.size() > class_ceiling) {
			throw InvalidInputException(
			    "anofox_tabfm: '%s' supports at most %llu classes and '%s' has %llu. This is a property of the "
			    "model, not of your data or your graph. Reduce the class count, or decompose the problem "
			    "(one-vs-rest, hierarchical grouping) so each call stays within the limit; SELECT model, "
			    "max_classes FROM tabfm_list_models(); shows the ceiling for every registered model.",
			    resolved.manifest.model, static_cast<unsigned long long>(class_ceiling), in.target_name,
			    static_cast<unsigned long long>(batch.label_decoder.size()));
		}

		// 2. materialize the input tensors (float32) — CPU-only work, done OUTSIDE
		// the per-device lock so it can overlap another group's inference on the
		// same device.
		vector<float> x(batch.x.size());
		for (idx_t i = 0; i < batch.x.size(); i++) {
			x[i] = static_cast<float>(batch.x[i]);
		}
		vector<float> y(batch.y.size());
		for (idx_t i = 0; i < batch.y.size(); i++) {
			y[i] = static_cast<float>(batch.y[i]);
		}
		// std::vector<bool> is bit-packed (no .data()); materialize a real bool array.
		auto cat_mask = make_unsafe_uniq_array<bool>(batch.cat_mask.size());
		for (idx_t i = 0; i < batch.cat_mask.size(); i++) {
			cat_mask[i] = batch.cat_mask[i];
		}
		TabFMRunInput run_input;
		run_input.x = x.data();
		run_input.y = y.data();
		run_input.cat_mask = cat_mask.get();
		run_input.t = NumericCast<int64_t>(batch.T);
		run_input.h = NumericCast<int64_t>(batch.H);
		run_input.train_size = NumericCast<int64_t>(batch.train_size);
		run_input.d = NumericCast<int64_t>(batch.d);

		// The context cache's settings as they are NOW, not as they were bound: the budget is instance-wide
		// (one cache per loaded model serves every connection), and SET GLOBAL anofox_tabfm_context_cache =
		// false must stop a statement prepared before it. A session-level SET of the kill switch applies to
		// statements bound after it.
		auto ctx = in.ctx;
		ApplyInstanceContextCacheSettings(*ctx.db, ctx);

		// Fitted values for the labelled rows. A graph that adds a context row's label to that row's own
		// representation and decodes every row with a head trained only on rows without one returns, for the
		// context rows, outputs the head never saw: measured on the real weights, tabfm-v1 regression corr -0.016
		// and tabicl-v2 regression -0.364 against the labels (eager PyTorch agrees with this engine to four
		// digits), where a second pass gives +0.93 and +0.86; kumo-tabular and tabfm-v1-1 likewise (the
		// registry lists the measurements, test_tabfm_registry.cpp). So for a model the registry marks
		// fitted_second_pass the context rows are appended AFTER the real rows, as queries, and their fitted
		// values are read from there. Appending (not prepending) keeps the split at train_size, and a query row's
		// answer depends only on the context, so every real query row is bit-identical (measured: max diff 0).
		// Only the caller that returns those rows asks for it (opts.context_values). A split-context backend gets
		// the same appended rows: its prepare graph's context_logits are the combined graph's context outputs, the
		// same unusable values, while the appended rows go through its query graph against the cached context.
		const idx_t fit_rows = batch.train_size;
		const bool want_second_pass = resolved.fitted_second_pass && in.opts.context_values && fit_rows > 0;
		if (want_second_pass) {
			// x and y are this call's own copies: append the context rows to them as unlabelled queries.
			const idx_t context_floats = fit_rows * batch.H;
			const idx_t real_floats = x.size();
			x.resize(real_floats + context_floats);
			std::copy(x.begin(), x.begin() + NumericCast<int64_t>(context_floats),
			          x.begin() + NumericCast<int64_t>(real_floats));
			y.resize(batch.T + fit_rows, static_cast<float>(kTargetPadSentinel));
			run_input.x = x.data();
			run_input.y = y.data();
			run_input.t = NumericCast<int64_t>(batch.T + fit_rows);
		}

		// 3. load + forward. Only the session load and the forward pass are
		// serialized per device (the expensive, non-reentrant parts); the model was
		// already resolved above.
		auto state = TabFMState::Get(*ctx.db);
		TabFMRunOutput out;
		{
			lock_guard<mutex> device_guard(state->DeviceMutex(ResolvedDeviceCached(*fs, resolved, ctx)));
			auto model = LoadOrGetSession(*fs, *state, resolved, ctx);
			auto *backend = reinterpret_cast<TabFMBackend *>(model->session.get());
			run_input.context_cache_bytes = ctx.context_cache_bytes;
			run_input.context_cache_admit_first = ctx.context_cache_admit_first;
			// A lowered budget frees cached contexts before the memory check below, so following that
			// check's advice (shrink the cache) can actually get a refused call through.
			backend->TrimContextCache(ctx.context_cache_bytes);

			const int64_t s = run_input.train_size;
			const int64_t q = run_input.t - s;
			const int64_t query_batch = NumericCast<int64_t>(in.opts.query_batch);
			// The largest single pass: what anofox_tabfm_max_memory's measured cost is keyed on.
			const int64_t pass_rows = s + (query_batch > 0 ? MinValue<int64_t>(query_batch, q) : q);

			// anofox_tabfm_max_memory, second half. The bind-time check refuses when
			// memory ALREADY held is over the ceiling; it cannot see a call that is
			// small at entry and enormous at exit, which is the one the OOM killer
			// takes. Nothing here estimates that from the model -- the manifest
			// carries no hidden size or layer count, and a guessed constant would
			// refuse work that would have succeeded. Instead the cost of this exact
			// shape is measured the first time it runs and remembered, which is
			// enough because the calls that hit this repeat: one per group, one per
			// chunk, same T and H every time.
			const auto ceiling = ctx.max_memory_bytes;
			const auto model_key = model->model_key;
			// The RESOLVED device, not ctx.device: that is the setting, and 'auto'
			// would key the same physical device differently from an explicit 'cpu'.
			const auto &cost_device = model->device_id;
			const auto stats = model->context_stats;
			idx_t before = 0;
			idx_t cached_before = 0;
			if (ceiling > 0) {
				before = CurrentProcessResidentBytes();
				cached_before = stats ? stats->bytes.load() : 0;
				auto expected = state->ForwardCost(model_key, cost_device, pass_rows, run_input.h);
				if (before > 0 && expected > 0 && before + expected >= ceiling) {
					throw InvalidInputException(
					    "anofox_tabfm: a forward pass of this shape (%lld rows x %lld features, model "
					    "'%s') was measured to add %llu bytes, and resident memory is already %llu -- "
					    "together that is at or above anofox_tabfm_max_memory (%llu bytes), so this call "
					    "is refused rather than risking an OOM kill. Send fewer rows per call or score "
					    "them in batches with opts := MAP{'query_batch': 'N'}, raise the ceiling with SET "
					    "anofox_tabfm_max_memory = '<size>', shrink the context cache with SET "
					    "anofox_tabfm_context_cache_memory = '<size>', unload unused models with "
					    "tabfm_unload(...), or disable the check with SET anofox_tabfm_max_memory = ''",
					    static_cast<long long>(pass_rows), static_cast<long long>(run_input.h), model_key.c_str(),
					    static_cast<unsigned long long>(expected), static_cast<unsigned long long>(before),
					    static_cast<unsigned long long>(ceiling));
				}
			}

			// opts.query_batch: the query rows go through at most N per pass. With the context cache the
			// context is looked up or encoded once for the whole call; otherwise each pass re-runs it,
			// which still bounds the peak (TabFMBackend::RunBatched).
			out = backend->RunBatched(run_input, query_batch);

			// Recorded only on the way out of a successful call, so a shape that has
			// never completed never produces an estimate. The first call of a shape
			// is therefore unguarded -- stated plainly because it is the limit of
			// what measuring rather than modelling can do. Bytes this call ADDED to
			// the context cache are not the forward pass's cost: they are retained
			// on purpose, bounded by their own budget, and counted in `before` next
			// time.
			if (ceiling > 0 && before > 0) {
				auto after = CurrentProcessResidentBytes();
				const idx_t cached_after = stats ? stats->bytes.load() : 0;
				const idx_t retained = cached_after > cached_before ? cached_after - cached_before : 0;
				if (after > before + retained) {
					state->RecordForwardCost(model_key, cost_device, pass_rows, run_input.h,
					                         after - before - retained);
				}
			}
		}

		// 4. decode logits[1,T,C] -> per-source-row predictions. After a second pass the output is [1,T+S,C]: rows
		// [0,S) are the context rows' un-trained values, [S,T) the real queries, [T,T+S) the fitted values. Fold them
		// back into the [1,T,C] shape everything below expects, fitted values taking the place of rows [0,S).
		if (want_second_pass && out.shape.size() == 3 && out.shape[1] == NumericCast<int64_t>(batch.T + fit_rows)) {
			const idx_t c = NumericCast<idx_t>(out.shape.back());
			for (idx_t t = 0; t < fit_rows; t++) {
				std::copy(out.logits.begin() + NumericCast<idx_t>((batch.T + t) * c),
				          out.logits.begin() + NumericCast<idx_t>((batch.T + t + 1) * c),
				          out.logits.begin() + NumericCast<idx_t>(t * c));
			}
			out.logits.resize(NumericCast<idx_t>(batch.T * c));
			out.shape[1] = NumericCast<int64_t>(batch.T);
		}
		return Decode(in, batch, out, task);
	}

private:
	static TabFMPredictResult Decode(const PredictInput &in, const PreprocessedBatch &batch,
	                                 const TabFMRunOutput &out, TabFMTask task) {
		const idx_t T = batch.T;
		const idx_t n_rows = in.rows.size();
		const idx_t n_classes = batch.label_decoder.size();
		// Fail loudly on any graph whose output does not match the contract
		// [1, T, C] rather than indexing out of bounds or decoding with the wrong
		// stride / zero-filled classes (classification needs C >= #labels,
		// regression needs C >= 1).
		ValidateTabFMOutput(out, T, task == TabFMTask::CLASSIFICATION ? n_classes : 1, TabFMTaskName(task));
		const idx_t C = NumericCast<idx_t>(out.shape.back());

		TabFMPredictResult result;
		result.yhat.resize(n_rows);
		result.yhat_score.resize(n_rows);
		if (in.opts.detail && task == TabFMTask::CLASSIFICATION) {
			result.proba.resize(n_rows);
		}

		for (idx_t t = 0; t < T; t++) {
			const idx_t src = batch.row_source_index[t];
			if (task == TabFMTask::CLASSIFICATION) {
				vector<double> logits(n_classes);
				for (idx_t c = 0; c < n_classes && c < C; c++) {
					logits[c] = out.logits[t * C + c];
				}
				SoftmaxInPlace(logits, in.opts.softmax_temperature);
				idx_t best = 0;
				for (idx_t c = 1; c < n_classes; c++) {
					if (logits[c] > logits[best]) {
						best = c;
					}
				}
				result.yhat[src] = batch.label_decoder[best];
				result.yhat_score[src] = Value::DOUBLE(logits.empty() ? 0.0 : logits[best]);
				if (!result.proba.empty()) {
					vector<Value> keys, vals;
					for (idx_t c = 0; c < n_classes; c++) {
						keys.emplace_back(batch.label_decoder[c].ToString());
						vals.emplace_back(Value::DOUBLE(logits[c]));
					}
					result.proba[src] =
					    Value::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE, std::move(keys), std::move(vals));
				}
			} else {
				double raw = C > 0 ? out.logits[t * C] : 0.0;
				double yhat = raw * batch.target_scale + batch.target_mean;
				result.yhat[src] = Value::DOUBLE(yhat);
				result.yhat_score[src] = Value(LogicalType::DOUBLE); // NULL
			}
		}
		return result;
	}
};

} // anonymous namespace

bool ContextSightings::Sight(hash_t key) {
	auto found = index.find(key);
	if (found != index.end()) {
		order.splice(order.begin(), order, found->second);
		return true;
	}
	if (capacity == 0) {
		return false;
	}
	order.push_front(key);
	index[key] = order.begin();
	while (order.size() > capacity) {
		index.erase(order.back());
		order.pop_back();
	}
	return false;
}

string ContextCachePairProblem(const vector<string> &prepare_inputs, const vector<string> &prepare_outputs,
                               const vector<string> &query_inputs, const vector<string> &query_outputs,
                               const vector<string> &combined_inputs) {
	auto has = [](const vector<string> &names, const string &name) {
		return std::find(names.begin(), names.end(), name) != names.end();
	};
	for (auto &name : combined_inputs) {
		if (name != "x" && name != "y") {
			// cat_mask, d, train_size, ...: inputs the prepare graph does not take and the cache key does
			// not cover, so two calls differing only there would share a context.
			return "the combined graph takes '" + name + "', which the cache key does not cover";
		}
	}
	if (prepare_inputs.size() != 2 || !has(prepare_inputs, "x") || !has(prepare_inputs, "y")) {
		return "the prepare graph must take exactly x and y";
	}
	if (!has(prepare_outputs, "context_logits")) {
		return "the prepare graph does not emit context_logits, so the context rows' outputs would differ";
	}
	if (!has(query_inputs, "x")) {
		return "the query graph does not take x";
	}
	if (!has(query_outputs, "logits")) {
		return "the query graph does not emit logits";
	}
	bool takes_state = false;
	for (auto &name : query_inputs) {
		if (name == "x" || name == "y") {
			continue;
		}
		if (name == "context_logits" || !has(prepare_outputs, name)) {
			return "the query graph takes '" + name + "', which the prepare graph does not emit";
		}
		takes_state = true;
	}
	if (!takes_state) {
		return "the query graph takes no prepared state";
	}
	return string();
}

string Sha256Hex(const_data_ptr_t data, idx_t len) {
	unsigned char digest[EVP_MAX_MD_SIZE];
	unsigned int n = 0;
	EVP_Digest(data, len, digest, &n, EVP_sha256(), nullptr);
	static const char *hex = "0123456789abcdef";
	string out;
	out.reserve(static_cast<size_t>(n) * 2);
	for (unsigned int i = 0; i < n; i++) {
		out.push_back(hex[digest[i] >> 4]);
		out.push_back(hex[digest[i] & 0xf]);
	}
	return out;
}

// Defined with external linkage (declared in tabfm_predict.hpp) so
// tabfm_backends() consults the SAME memo dispatch does -- a capability
// relation that disagrees with dispatch about whether a lane exists is worse
// than none, because every refusal message points users at it.
bool PluginAvailableCached(const string &ep_path, const string &backend) {
	static mutex memo_lock;
	static map<string, bool> memo;
	const string key =
	    std::to_string(g_plugin_probe_generation.load()) + "\x1f" + ep_path + "\x1f" + backend;
	{
		lock_guard<mutex> guard(memo_lock);
		auto entry = memo.find(key);
		if (entry != memo.end()) {
			return entry->second;
		}
	}
	bool available = false;
	if (!ep_path.empty()) {
		auto fs = FileSystem::CreateLocal();
		available = PluginLoadable(fs->JoinPath(ep_path, PluginFileName(backend)));
	}
	lock_guard<mutex> guard(memo_lock);
	return memo.emplace(key, available).first->second;
}

vector<string> MlxUnsupportedOps(const char *bundled_graph, idx_t bundled_size, const string &graph_path) {
	static mutex memo_lock;
	static map<string, vector<string>> memo;
	// A bundled graph is immutable for the life of the process; a file is keyed by path and size.
	string key;
	if (bundled_graph) {
		key = "b:" + std::to_string(reinterpret_cast<uintptr_t>(bundled_graph));
	} else {
		struct stat st;
		key = "f:" + graph_path + ":" + std::to_string(stat(graph_path.c_str(), &st) == 0 ? st.st_size : -1);
	}
	{
		lock_guard<mutex> guard(memo_lock);
		auto entry = memo.find(key);
		if (entry != memo.end()) {
			return entry->second;
		}
	}
	vector<string> missing;
	try {
		auto graph = bundled_graph ? ::anofox::onnxread::ParseModelBytes(
		                                 reinterpret_cast<const uint8_t *>(bundled_graph), bundled_size, "bundled graph")
		                           : ::anofox::onnxread::ParseModelFile(graph_path);
		std::vector<std::string> ops;
		ops.reserve(graph.nodes.size());
		for (auto &node : graph.nodes) {
			ops.push_back(node.op_type);
		}
		for (auto &op : MlxMissingOps(ops)) {
			missing.push_back(op);
		}
	} catch (const std::exception &) {
		// Unreadable here means unreadable for the plugin too, which reports it with more context.
	}
	lock_guard<mutex> guard(memo_lock);
	return memo.emplace(key, std::move(missing)).first->second;
}

bool WeightsHeaderMatches(FileSystem &fs, const string &weights_path, const string &model, TabFMTask task) {
	const string expected = ExpectedWeightsHeaderShaFor(model, TabFMTaskName(task));
	if (expected.empty()) {
		return false;
	}
	if (StringUtil::Split(weights_path, "/").back() != "model.safetensors") {
		return false;
	}
	string header;
	if (!ReadWeightsHeaderBytes(fs, weights_path, header)) {
		return false;
	}
	return Sha256Hex(const_data_ptr_cast(header.data()), header.size()) == expected;
}

PredictEngine &GetPredictEngine() {
	static TabFMRealEngine engine;
	return engine;
}

namespace {

//! The newest modification time among the .mxr programs for one padded bucket
//! in the MIGraphX cache directory (0 when there are none). The plugin names
//! each program <stem>_<arch>_<precision>_T<t>_H<h>.mxr, so a bucket suffix
//! match finds every model's program for it; the caller compares the value
//! before and after a precompile, which isolates the one it wrote.
int64_t NewestBucketMxr(FileSystem &fs, const string &mxr_dir, int64_t padded_t, int64_t padded_h) {
	const string suffix = "_T" + std::to_string(padded_t) + "_H" + std::to_string(padded_h) + ".mxr";
	int64_t newest = 0;
	if (!fs.DirectoryExists(mxr_dir)) {
		return newest;
	}
	fs.ListFiles(mxr_dir, [&](const string &name, bool is_dir) {
		if (is_dir || !StringUtil::EndsWith(name, suffix)) {
			return;
		}
		try {
			auto handle = fs.OpenFile(fs.JoinPath(mxr_dir, name), FileFlags::FILE_FLAGS_READ);
			newest = MaxValue<int64_t>(newest, fs.GetLastModifiedTime(*handle).value);
		} catch (std::exception &) {
			// A program being written or removed concurrently: not ours to report.
		}
	});
	return newest;
}

} // namespace

TabFMPrecompileResult TabFMGpuPrecompile(const PredictContext &bound_ctx, TabFMTask task, int64_t rows,
                                          int64_t features) {
	auto ctx = bound_ctx;
	if (ctx.db) {
		ApplyInstanceContextCacheSettings(*ctx.db, ctx);
	}
	if (!ctx.db) {
		throw InternalException("tabfm: precompile invoked without a database handle");
	}
	const auto started = std::chrono::steady_clock::now();
	auto fs = FileSystem::CreateLocal();
	auto resolved = ResolveModel(*fs, ctx, task, string());
	auto state = TabFMState::Get(*ctx.db);
	TabFMPrecompileResult result;
	result.model = resolved.manifest.model;
	// Loads/caches the backend (registered in state) and warms the shape-bucket:
	// on ROCm this is the expensive MIGraphX compile + .mxr cache; on CPU/CUDA/MLX
	// the backend has no per-shape compile and the call just leaves the session warm.
	const string &device = ResolvedDeviceCached(*fs, resolved, ctx);
	result.device = device;
	lock_guard<mutex> device_guard(state->DeviceMutex(device));
	auto model = LoadOrGetSession(*fs, *state, resolved, ctx);
	auto *backend = reinterpret_cast<TabFMBackend *>(model->session.get());
	// Only the MIGraphX backend pads to a bucket and compiles per bucket. Same
	// table as the plugin (tabfm_shape_bucket.hpp), so the reported bucket is
	// the one the .mxr is keyed on.
	result.bucketed = StringUtil::StartsWith(model->device_id, "rocm");
	int64_t before = 0;
	string mxr_dir;
	if (result.bucketed) {
		auto bucket = PadToShapeBucket(rows, features);
		result.bucket_rows = bucket.padded_t;
		result.bucket_features = bucket.padded_h;
		// A bucket over the model's ceiling is reported, not compiled: the compile would abort the process.
		int64_t vram_total = -1;
		for (auto &d : DiscoverDevices()) {
			if (d.device_id == model->device_id) {
				vram_total = d.vram_total;
			}
		}
		result.reason = RocmBucketRefusal(result.model, bucket.padded_t, bucket.padded_h, vram_total);
		if (!result.reason.empty()) {
			result.status = "refused";
			result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
			return result;
		}
		mxr_dir = fs->JoinPath(ctx.cache_dir, "migraphx");
		before = NewestBucketMxr(*fs, mxr_dir, bucket.padded_t, bucket.padded_h);
	}
	backend->Precompile(rows, features);
	if (result.bucketed) {
		result.status =
		    NewestBucketMxr(*fs, mxr_dir, result.bucket_rows, result.bucket_features) > before ? "compiled" : "cached";
	} else {
		result.status = "not needed";
	}
	result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	return result;
}

} // namespace anofox
} // namespace duckdb
