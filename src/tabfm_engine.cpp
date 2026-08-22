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
#include "tabfm_state.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/database.hpp"

#include "yyjson.hpp"

#include <openssl/evp.h>

#include <cmath>
#include <cstring>
#include <fstream>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace duckdb {
namespace anofox {

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

// Load {onnx -> safetensors} from the manifest: inline map, or the
// "initializers" object of the tensor-map JSON file, else identity.
unordered_map<string, string> LoadTensorMap(FileSystem &fs, const ModelManifest &manifest, const string &dir,
                                            bool use_bundle) {
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
		throw InvalidInputException(
		    "tabfm: model '%s' does not support task '%s'. Choose a model that declares it, or use the matching "
		    "tabfm_%s function.",
		    spec.id, task_name, task == TabFMTask::CLASSIFICATION ? "classify" : "regress");
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
	resolved.tensor_map = LoadTensorMap(fs, resolved.manifest, resolved.manifest_dir, is_builtin);
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

	bool Available() const {
		return prepare.Available() && query.Available();
	}
};

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

// A model exported as a support/query PAIR (tools/export_tabicl --split-context):
// the labelled context is encoded once by the prepare graph and reused by every
// query batch that arrives with the same context. That encoding is what the
// combined graph re-runs on every call — 71-80% of a tabicl-v2 forward pass at a
// 64-375 row context (DataZooDE/anofox-tabfm#37).
//
// Declaration order is the destruction contract, exactly as OrtBackend: the
// sessions are LAST so they are destroyed FIRST, before the weight buffers ORT
// reads lazily during inference.
struct SplitOrtBackend : public TabFMBackend {
	string weights;
	MappedFile mapping;
	F32Arena arena;
	//! The query graph's non-(x,y) inputs — what prepare has to supply.
	vector<string> context_inputs;

	// The one cached context. Guarded by the caller's per-device lock (HLD §6),
	// which is held across the whole forward pass, so no locking is needed here.
	//
	// That is only true because the caller keys that lock on the RESOLVED device
	// (`ResolvedDeviceCached`). It keyed on the raw `anofox_tabfm_device` setting
	// until #42: a connection on 'auto' and one on 'cpu' resolve to the same
	// backend but took different mutexes, so they did not exclude each other and
	// could tear these vectors while one rebuilt them. Anything that re-keys that
	// lock on the setting again reopens the hole, and nothing here would notice.
	//
	// The support rows are kept and compared VERBATIM rather than hashed: a hash
	// collision here would not fail, it would answer a query against someone
	// else's training data. A memcmp of a few hundred kB costs nothing next to
	// the forward pass it guards.
	vector<float> context_x;
	vector<float> context_y;
	int64_t context_h = 0;
	TabFMPreparedContext context;

	TabFMSessionHandle prepare;
	TabFMSessionHandle query;

	bool ContextMatches(const TabFMRunInput &input, int64_t s) const {
		if (context.context_rows != s || context_h != input.h) {
			return false;
		}
		const auto x_count = static_cast<size_t>(s) * static_cast<size_t>(input.h);
		if (context_x.size() != x_count || context_y.size() != static_cast<size_t>(s)) {
			return false;
		}
		return std::memcmp(context_x.data(), input.x, x_count * sizeof(float)) == 0 &&
		       std::memcmp(context_y.data(), input.y, static_cast<size_t>(s) * sizeof(float)) == 0;
	}

	TabFMRunOutput Run(const TabFMRunInput &input) override {
		const int64_t s = input.train_size;
		const int64_t q = input.t - s;
		if (s <= 0 || q < 0) {
			throw InternalException("anofox_tabfm: split model called with train_size=" + std::to_string(s) +
			                        " of T=" + std::to_string(input.t));
		}

		if (!ContextMatches(input, s)) {
			// Invalidate BEFORE building, and publish only once everything below has
			// succeeded. Both passes can throw, and a half-built context that still
			// looked valid would be reused by the next call — silently, against the
			// wrong training data. Dropping the old one first also keeps the peak at
			// one context rather than two, which at real scale is hundreds of MB.
			context = TabFMPreparedContext();
			context_x.clear();
			context_y.clear();
			context_h = 0;

			auto prepared = RunPrepare(*prepare, input.x, input.y, s, input.h, context_inputs);

			// The context rows' own fitted values. Every row the engine returns needs
			// one, including the labelled ones, and the combined graph produces them
			// in the same pass. Here they are a query pass over the support rows —
			// which is exact for a cache, because a query row's answer depends only
			// on the context, never on which other rows shared its batch.
			auto fitted = RunQuery(*query, input.x, s, input.h, input.y, s, prepared);
			if (fitted.shape.size() != 3 || fitted.shape[1] != s) {
				throw InvalidInputException(
				    "anofox_tabfm: the query graph returned %llu values shaped [%s] for %lld context rows; expected "
				    "[1, %lld, C]. The prepare/query pair does not match this model.",
				    static_cast<unsigned long long>(fitted.logits.size()), ShapeString(fitted.shape),
				    static_cast<long long>(s), static_cast<long long>(s));
			}
			prepared.context_classes = fitted.shape.back();
			prepared.context_logits = std::move(fitted.logits);
			prepared.bytes += prepared.context_logits.size() * sizeof(float);

			const auto x_count = static_cast<size_t>(s) * static_cast<size_t>(input.h);
			context = std::move(prepared);
			context_x.assign(input.x, input.x + x_count);
			context_y.assign(input.y, input.y + static_cast<size_t>(s));
			context_h = input.h;
		}

		const int64_t c = context.context_classes;
		TabFMRunOutput out;
		out.shape = {1, input.t, c};
		out.logits.reserve(static_cast<size_t>(input.t) * static_cast<size_t>(c));
		out.logits.insert(out.logits.end(), context.context_logits.begin(), context.context_logits.end());
		if (q > 0) {
			auto scored = RunQuery(*query, input.x + s * input.h, q, input.h, input.y, s, context);
			if (scored.shape.size() != 3 || scored.shape[1] != q || scored.shape.back() != c) {
				throw InvalidInputException(
				    "anofox_tabfm: the query graph returned [%s] for %lld query rows; expected [1, %lld, %lld].",
				    ShapeString(scored.shape), static_cast<long long>(q), static_cast<long long>(q),
				    static_cast<long long>(c));
			}
			out.logits.insert(out.logits.end(), scored.logits.begin(), scored.logits.end());
		}
		return out;
	}
};

// Register a freshly built backend in the DB-instance state and return the
// LoadedModel snapshot. The void handle aliases the TabFMBackend base pointer so
// the predict loop can recover it and dispatch Run() virtually.
shared_ptr<LoadedModel> RegisterBackend(TabFMState &state, const string &cache_key,
                                        shared_ptr<TabFMBackend> backend, const string &device_id, idx_t bytes,
                                        bool split_context = false, const string &precision = "",
                                        idx_t max_sessions = 0) {
	auto model = make_shared_ptr<LoadedModel>();
	model->model_key = cache_key;
	model->session = shared_ptr<void>(std::move(backend));
	model->device_id = device_id;
	model->precision = precision;
	model->dtype = "f32";
	model->bytes = bytes;
	model->split_context = split_context;
	state.Register(cache_key, model, max_sessions);
	return model;
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

// Stage a bundled graph next to the weights (idempotent by size) so external-data
// "model.safetensors" resolves. Returns false if it cannot be written.
bool StageBundledGraph(FileSystem &fs, const BundledResource &graph, const string &graph_path) {
	try {
		if (fs.FileExists(graph_path)) {
			auto h = fs.OpenFile(graph_path, FileFlags::FILE_FLAGS_READ);
			if (NumericCast<idx_t>(fs.GetFileSize(*h)) == graph.size) {
				return true;
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
	auto device = ResolveDevice(ctx.device, devices);
	config.device_id = device.device_id;
	config.device_ordinal = device.device_ordinal;
	config.model_tag = task_name;
	config.ep_path = ctx.ep_path;
	// No injected initializers: ORT loads them from the safetensors via the
	// graph's external-data references.
	auto session = CreateSessionFromPath(graph_path, {}, config);
	auto backend = make_shared_ptr<OrtBackend>();
	backend->session = std::move(session);
	return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, 0, false, "",
	                       NumericCast<idx_t>(ctx.max_sessions));
}

// AMD ROCm GPU backend. ORT's MIGraphX EP cannot run this model (re-inlines
// weights -> 2 GB proto), so ROCm bypasses ORT entirely: a standalone plugin
// (docs/DYNAMIC_BACKENDS.md phase 1, src/tabfm_migraphx_plugin.cpp) parses the
// migraphx-ready graph (external-data + Shape-rewrite) directly and compiles
// per shape-bucket (cached to .mxr). Engages only when the resolved device is
// a rocm GPU and a bundled migraphx graph + matching weights exist; nullptr
// => fall back to the CPU/ORT path (no migraphx graph shipped for this task).
//
// Past the point where a migraphx graph is confirmed to exist for a resolved
// rocm device, this must succeed or throw: silently falling back to CPU here
// would be exactly the "requested device quietly becomes CPU" failure mode
// the equivalence suite's tier 4 exists to catch (DYNAMIC_BACKENDS.md).
shared_ptr<LoadedModel> TryMIGraphXBackend(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                           const PredictContext &ctx) {
	auto devices = DiscoverDevices();
	auto device = ResolveDevice(ctx.device, devices);
	if (!StringUtil::StartsWith(device.device_id, "rocm")) {
		return nullptr; // not the GPU path (cpu / cuda handled by the ORT backend)
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	// Graph source (docs/GPU_HARDENING_PLAN.md P3): a model-provided
	// migraphx_graph wins; the bundled tabfm-v1 graph needs its header match.
	auto graph = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, "migraphx", task_name));
	const bool bundled_matches =
	    graph.data && WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.model, resolved.manifest.task);
	string graph_path;
	string weights_dir;
	switch (SelectGpuGraph(!resolved.migraphx_graph_path.empty(), graph.data != nullptr, bundled_matches)) {
	case GpuGraphSource::MODEL_PROVIDED:
		graph_path = resolved.migraphx_graph_path;
		weights_dir = DirName(graph_path); // external data sits beside the graph
		break;
	case GpuGraphSource::BUNDLED: {
		weights_dir = DirName(resolved.weights_path);
		graph_path = fs.JoinPath(weights_dir, BundledGpuGraphId(resolved.manifest.model, "migraphx", task_name) + ".onnx");
		if (!StageBundledGraph(fs, graph, graph_path)) {
			return nullptr;
		}
		break;
	}
	case GpuGraphSource::NONE:
		// Explicitly-requested ROCm + no runnable graph is an error here, with
		// the real cause; declining silently used to surface a downstream
		// message blaming ep_path (found running the examples on GPU hardware).
		if (IsExplicitGpuRequest(ctx.device, "rocm")) {
			throw InvalidInputException(
			    NoGpuGraphMessage("rocm", resolved.manifest.model, task_name, "migraphx_graph"));
		}
		return nullptr;
	}
	const auto mxr_dir = fs.JoinPath(ctx.cache_dir, "migraphx");

	if (ctx.ep_path.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: device 'rocm' was resolved but no backend plugin directory is configured. SET "
		    "anofox_tabfm_ep_path to the directory holding libanofox_tabfm_migraphx_plugin.so.");
	}
	const auto plugin_path = fs.JoinPath(ctx.ep_path, "libanofox_tabfm_migraphx_plugin.so");

	TabFMPluginCreateParams params {};
	params.graph_path = graph_path.c_str();
	params.weights_dir = weights_dir.c_str();
	params.cache_dir = mxr_dir.c_str();
	params.arch = device.arch.c_str();
	params.precision = ctx.gpu_precision.c_str();
	params.mxr_source = ctx.mxr_source.c_str();
	params.device_ordinal = device.device_ordinal;
	shared_ptr<TabFMBackend> backend = LoadPluginBackend(plugin_path, params);
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
	auto device = ResolveDevice(ctx.device, devices);
	if (!StringUtil::StartsWith(device.device_id, "cuda")) {
		return nullptr; // not the CUDA path (cpu handled by the ORT backend, rocm above)
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	// Graph source (docs/GPU_HARDENING_PLAN.md P3): a model-provided ext_graph
	// wins; the bundled tabfm-v1 graph needs its header match.
	auto graph = GetBundledResource(BundledGpuGraphId(resolved.manifest.model, "ext", task_name));
	const bool bundled_matches =
	    graph.data && WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.model, resolved.manifest.task);
	string graph_path;
	string weights_dir;
	switch (SelectGpuGraph(!resolved.ext_graph_path.empty(), graph.data != nullptr, bundled_matches)) {
	case GpuGraphSource::MODEL_PROVIDED:
		graph_path = resolved.ext_graph_path;
		weights_dir = DirName(graph_path); // external data sits beside the graph
		break;
	case GpuGraphSource::BUNDLED: {
		weights_dir = DirName(resolved.weights_path);
		graph_path = fs.JoinPath(weights_dir, BundledGpuGraphId(resolved.manifest.model, "ext", task_name) + ".onnx");
		if (!StageBundledGraph(fs, graph, graph_path)) {
			return nullptr;
		}
		break;
	}
	case GpuGraphSource::NONE:
		// Same contract as the ROCm branch above: an explicit 'cuda' with no
		// runnable graph names the model and the fix, never ep_path.
		if (IsExplicitGpuRequest(ctx.device, "cuda")) {
			throw InvalidInputException(NoGpuGraphMessage("cuda", resolved.manifest.model, task_name, "ext_graph"));
		}
		return nullptr;
	}

	if (ctx.ep_path.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: device 'cuda' was resolved but no backend plugin directory is configured. SET "
		    "anofox_tabfm_ep_path to the directory holding libanofox_tabfm_cuda_plugin.so (CALL "
		    "tabfm_download_runtime('cuda') to fetch it).");
	}
	const auto plugin_path = fs.JoinPath(ctx.ep_path, "libanofox_tabfm_cuda_plugin.so");

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
const string &ResolvedDeviceCached(const string &setting) {
	static mutex memo_lock;
	static map<string, string> memo;
	{
		lock_guard<mutex> guard(memo_lock);
		auto entry = memo.find(setting);
		if (entry != memo.end()) {
			return entry->second;
		}
	}
	// Discovery outside the memo lock: it reaches into the driver, and holding a
	// lock across that would serialize unrelated first-time resolutions behind it.
	// Two connections racing here compute the same answer, so the re-lookup on the
	// way back in is a correctness no-op rather than a guard.
	auto devices = DiscoverDevices();
	auto resolved = ResolveDevice(setting, devices);
	lock_guard<mutex> guard(memo_lock);
	return memo.emplace(setting, resolved.device_id).first->second;
}

// Build (or reuse) the ORT session for `resolved` and return a snapshot of the
// LoadedModel. Order: MIGraphX plugin (ROCm GPU) -> CUDA plugin (NVIDIA GPU)
// -> low-memory external-data (CPU, ORT reads weights from disk) -> read/mmap
// + in-memory injection. Both GPU backends run in dlopen'd plugins with their
// own runtimes rather than through this binary's ORT (DYNAMIC_BACKENDS.md).
shared_ptr<LoadedModel> LoadOrGetSession(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                         const PredictContext &ctx) {
	// Does this model ship a split (prepare/query) pair, and did the user ask for
	// it? Decided BEFORE the cache lookup, because a session already built the
	// other way has to be rebuilt rather than reused (see below).
	SplitGraphs split;
	if (ctx.context_cache) {
		split = FindSplitGraphs(fs, resolved);
	}
	const bool want_split = split.Available();

	// The device the caller is asking for right now. A cached session belongs to
	// the device it was BUILT for, so reusing it whenever the key matches
	// silently ignores anofox_tabfm_device: a connection that ran anything on
	// the cpu and then SET anofox_tabfm_device='cuda' kept getting the cpu
	// session and never touched the GPU. That is precisely the "requested device
	// quietly becomes CPU" outcome the tier-4 contract in
	// docs/DYNAMIC_BACKENDS.md exists to rule out, and it is invisible without
	// looking at tabfm_models().device.
	const string &wanted_device = ResolvedDeviceCached(ctx.device);
	// gpu_precision only shapes GPU sessions; for CPU it is "" so flipping the
	// setting does not rebuild a session it never influenced.
	const bool wanted_is_gpu =
	    StringUtil::StartsWith(wanted_device, "rocm") || StringUtil::StartsWith(wanted_device, "cuda");
	const string wanted_precision = wanted_is_gpu ? ctx.gpu_precision : "";

	// Sessions cache per (model, device, precision) since P5 of
	// docs/GPU_HARDENING_PLAN.md: S6 measured 18–27 s of pure rebuild per
	// device alternation under the old one-slot-per-key design. The lookup is
	// exact, so a device or precision switch is a cache MISS that builds a new
	// entry beside the old one — not a replacement — and tabfm_unload(key)
	// still frees every entry by one name (the two-level map guarantees it).
	if (auto snapshot = state.Snapshot(resolved.cache_key, wanted_device, wanted_precision)) {
		if (CanReuseSession(*snapshot, want_split, wanted_device, wanted_precision)) {
			return snapshot;
		}
		// anofox_tabfm_context_cache changed since this configuration was
		// loaded. Fall through and rebuild the SAME (device, precision) entry —
		// split and combined sessions of one configuration must not coexist,
		// could not reach.
	}

	// The alternative backends below are driven by graphs compiled into the
	// binary, and none of them has a split form, so a model that ships a pair
	// takes the injection path below instead. That includes the CUDA plugin:
	// like the MIGraphX one it runs a bundled graph_ext_/graph_migraphx_ graph,
	// so letting a split-pair model reach it would silently drop the labelled
	// context cache (#40) rather than use it.
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
	vector<TabFMTensorRef> initializers;
	// The checkpoint key each entry of `initializers` came from, same order. The
	// split halves are injected from their OWN tensor maps (each names only the
	// initializers its graph declares), and renaming needs the source key.
	vector<string> initializer_keys;
	if (IsTorchCkpt(bytes, nbytes)) {
		// Native PyTorch checkpoint (.ckpt): recover the state_dict and inject each
		// tensor by graph name. The bytes alias the mmap/read buffer (kept alive by
		// the backend below), so no arena copy is needed for f32/i64.
		auto sd = ReadTorchCkpt(bytes, nbytes);
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
			const CkptTensor &t = entry.second;
			TabFMTensorRef ref;
			auto it = st_to_onnx.find(entry.first);
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
			auto &m = arena.Get(st_key);
			TabFMTensorRef ref;
			auto it = st_to_onnx.find(st_key);
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
	auto device = ResolveDevice(ctx.device, devices);
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
		if (backend->context_inputs.empty()) {
			throw InvalidInputException(
			    "tabfm: the query half of '%s' takes only x and y, so it is not a query graph. Re-export the pair with "
			    "tools/export_tabicl --split-context, or unset anofox_tabfm_context_cache.",
			    resolved.manifest.model);
		}
		backend->weights = std::move(weights);
		backend->mapping = std::move(mapping);
		backend->arena = std::move(arena);
		return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, weight_bytes, true, "",
		                       NumericCast<idx_t>(ctx.max_sessions));
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
	return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, weight_bytes, false, "",
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
		spec.name = fields[c].first;
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
		chunk.SetCardinality(idx + 1);
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

		// 3. load + forward. Only the session load and the forward pass are
		// serialized per device (the expensive, non-reentrant parts); the model was
		// already resolved above.
		auto state = TabFMState::Get(*in.ctx.db);
		TabFMRunOutput out;
		{
			lock_guard<mutex> device_guard(state->DeviceMutex(ResolvedDeviceCached(in.ctx.device)));
			auto model = LoadOrGetSession(*fs, *state, resolved, in.ctx);
			auto *backend = reinterpret_cast<TabFMBackend *>(model->session.get());

			// anofox_tabfm_max_memory, second half. The bind-time check refuses when
			// memory ALREADY held is over the ceiling; it cannot see a call that is
			// small at entry and enormous at exit, which is the one the OOM killer
			// takes. Nothing here estimates that from the model -- the manifest
			// carries no hidden size or layer count, and a guessed constant would
			// refuse work that would have succeeded. Instead the cost of this exact
			// shape is measured the first time it runs and remembered, which is
			// enough because the calls that hit this repeat: one per group, one per
			// chunk, same T and H every time.
			const auto ceiling = in.ctx.max_memory_bytes;
			const auto model_key = model->model_key;
			// The RESOLVED device, not in.ctx.device: that is the setting, and 'auto'
			// would key the same physical device differently from an explicit 'cpu'.
			const auto &cost_device = model->device_id;
			idx_t before = 0;
			if (ceiling > 0) {
				before = CurrentProcessResidentBytes();
				auto expected = state->ForwardCost(model_key, cost_device, run_input.t, run_input.h);
				if (before > 0 && expected > 0 && before + expected >= ceiling) {
					throw InvalidInputException(
					    "anofox_tabfm: a forward pass of this shape (%lld rows x %lld features, model "
					    "'%s') was measured to add %llu bytes, and resident memory is already %llu -- "
					    "together that is at or above anofox_tabfm_max_memory (%llu bytes), so this call "
					    "is refused rather than risking an OOM kill. Send fewer rows per call, raise the "
					    "ceiling with SET anofox_tabfm_max_memory = '<size>', unload unused models with "
					    "tabfm_unload(...), or disable the check with SET anofox_tabfm_max_memory = ''",
					    static_cast<long long>(run_input.t), static_cast<long long>(run_input.h),
					    model_key.c_str(), static_cast<unsigned long long>(expected),
					    static_cast<unsigned long long>(before), static_cast<unsigned long long>(ceiling));
				}
			}

			out = backend->Run(run_input);

			// Recorded only on the way out of a successful call, so a shape that has
			// never completed never produces an estimate. The first call of a shape
			// is therefore unguarded -- stated plainly because it is the limit of
			// what measuring rather than modelling can do.
			if (ceiling > 0 && before > 0) {
				auto after = CurrentProcessResidentBytes();
				if (after > before) {
					state->RecordForwardCost(model_key, cost_device, run_input.t, run_input.h,
					                         after - before);
				}
			}
		}

		// 4. decode logits[1,T,C] -> per-source-row predictions
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

PredictEngine &GetPredictEngine() {
	static TabFMRealEngine engine;
	return engine;
}

void TabFMGpuPrecompile(const PredictContext &ctx, TabFMTask task, int64_t rows, int64_t features) {
	if (!ctx.db) {
		throw InternalException("tabfm: precompile invoked without a database handle");
	}
	auto fs = FileSystem::CreateLocal();
	auto resolved = ResolveModel(*fs, ctx, task, string());
	auto state = TabFMState::Get(*ctx.db);
	// Loads/caches the backend (registered in state) and warms the shape-bucket:
	// on ROCm this is the expensive MIGraphX compile + .mxr cache; on CPU/CUDA the
	// no-op default just leaves the freshly-built ORT session warm.
	lock_guard<mutex> device_guard(state->DeviceMutex(ResolvedDeviceCached(ctx.device)));
	auto model = LoadOrGetSession(*fs, *state, resolved, ctx);
	auto *backend = reinterpret_cast<TabFMBackend *>(model->session.get());
	backend->Precompile(rows, features);
}

} // namespace anofox
} // namespace duckdb
