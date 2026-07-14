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
#include "tabfm_ort_engine.hpp"
#include "tabfm_bundled_resources.hpp"
#include "tabfm_migraphx.hpp"
#include "tabfm_state.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/database.hpp"

#include "yyjson.hpp"

#include <openssl/evp.h>

#include <cmath>
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
	string weights_path;
	unordered_map<string, string> tensor_map; // onnx initializer name -> st key
	string cache_key;
	// Manifest-declared tensor contract (P4): graph input/output names to verify
	// against the actual graph at load. Empty = no declared contract.
	vector<string> contract_inputs;
	vector<string> contract_outputs;
};

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
	vector<string> candidates;
	if (!cache_dir.empty()) {
		// Match the download-side cache slug (WeightsManifest::CacheSlug): a
		// repo-less model (e.g. a user manifest with per-file urls) caches under
		// its model id, not a repo path. Resolve must look there too, else a
		// downloaded repo-less model reads as "not downloaded".
		auto slug_base = manifest.repo.empty() ? manifest.model : StringUtil::Replace(manifest.repo, "/", "__");
		candidates.push_back(fs.JoinPath(fs.JoinPath(cache_dir, slug_base + "@" + manifest.revision), file.path));
	}
	candidates.push_back(JoinPath(fs, dir, file.path));
	// fixture layout: a bare model.safetensors beside the manifest
	candidates.push_back(JoinPath(fs, dir, StringUtil::Split(file.path, "/").back()));
	for (auto &candidate : candidates) {
		if (fs.FileExists(candidate)) {
			return candidate;
		}
	}
	throw InvalidInputException("tabfm: model '%s' is not downloaded. Run: CALL tabfm_download('%s');",
	                            task_name, task_name);
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
	m.tensor_map_path = art.tensor_map_path;
	m.tensor_map = art.tensor_map;
	m.preprocessing_profile = art.preprocessing_profile;
	m.license = spec.license.id;
	m.engine_profiles = spec.engine_profiles;
	return m;
}

ResolvedModel ResolveModel(FileSystem &fs, const PredictContext &ctx, TabFMTask task, const string &requested_model) {
	ResolvedModel resolved;
	const auto task_name = TabFMTaskName(task);
	// The registry: built-ins + user manifests (anofox_tabfm_model_manifest,
	// file or dir). Selection precedence: model := → default_model → single-file
	// manifest → sole model.
	auto registry = ModelRegistry::Build(ctx.model_manifest_path,
	                                    ctx.db ? TabFMState::Get(*ctx.db)->RegisteredSpecs() : vector<ModelSpec>());
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
	const bool is_builtin = spec.source_dir.empty();
	resolved.manifest_dir = is_builtin ? ctx.cache_dir : spec.source_dir;
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
	resolved.tensor_map = LoadTensorMap(fs, resolved.manifest, resolved.manifest_dir, is_builtin);
	resolved.cache_key = TabFMModelCacheKey(resolved.manifest.model, task_name, resolved.manifest.revision);
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

// Register a freshly built backend in the DB-instance state and return the
// LoadedModel snapshot. The void handle aliases the TabFMBackend base pointer so
// the predict loop can recover it and dispatch Run() virtually.
shared_ptr<LoadedModel> RegisterBackend(TabFMState &state, const string &cache_key,
                                        shared_ptr<TabFMBackend> backend, const string &device_id, idx_t bytes) {
	auto model = make_shared_ptr<LoadedModel>();
	model->model_key = cache_key;
	model->session = shared_ptr<void>(std::move(backend));
	model->device_id = device_id;
	model->dtype = "f32";
	model->bytes = bytes;
	state.Register(cache_key, model);
	return model;
}

// SHA-256 (hex) of the safetensors JSON header the bundled external-data graphs
// were generated against (tools/make_external_graph.py prints these). A
// byte-identical header guarantees the graph's baked offsets are correct; any
// other layout falls back to the injection path. Empty => no external-data graph.
const char *ExpectedWeightsHeaderSha(TabFMTask task) {
	switch (task) {
	case TabFMTask::CLASSIFICATION:
		return "534d6d38b49b323bb38682858f232573c254689df03d3d9f17e7504716a31d96";
	case TabFMTask::REGRESSION:
		return "35c346e4e29f61b493a9e601e66bf0ae241d0fb76623a3336c61408cfc3e88d0";
	default:
		return "";
	}
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
bool WeightsHeaderMatches(FileSystem &fs, const string &weights_path, TabFMTask task) {
	const char *expected = ExpectedWeightsHeaderSha(task);
	if (!expected || !*expected) {
		return false;
	}
	if (StringUtil::Split(weights_path, "/").back() != "model.safetensors") {
		return false;
	}
	string header;
	if (!ReadWeightsHeaderBytes(fs, weights_path, header)) {
		return false;
	}
	return Sha256Hex(const_data_ptr_cast(header.data()), header.size()) == string(expected);
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
	auto graph = GetBundledResource("graph_ext_" + task_name);
	if (!graph.data || !WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.task)) {
		return nullptr;
	}
	// Place the external-data graph next to the weights (idempotent) so ORT can
	// resolve the relative "model.safetensors" reference.
	const auto dir = DirName(resolved.weights_path);
	const auto graph_path = fs.JoinPath(dir, "graph_ext_" + task_name + ".onnx");
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
	// No injected initializers: ORT loads them from the safetensors via the
	// graph's external-data references.
	auto session = CreateSessionFromPath(graph_path, {}, config);
	auto backend = make_shared_ptr<OrtBackend>();
	backend->session = std::move(session);
	return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, 0);
}

// Direct AMD MIGraphX GPU backend. ORT's MIGraphX EP cannot run this model
// (re-inlines weights -> 2 GB proto), so ROCm bypasses ORT: parse the
// migraphx-ready graph (external-data + Shape-rewrite), compile per shape-bucket
// (cached to .mxr), run. Engages only when the resolved device is a rocm GPU and
// a bundled migraphx graph + matching weights exist; nullptr => fall back to the
// CPU/ORT path.
shared_ptr<LoadedModel> TryMIGraphXBackend(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                           const PredictContext &ctx) {
	auto devices = DiscoverDevices();
	auto device = ResolveDevice(ctx.device, devices);
	if (!StringUtil::StartsWith(device.device_id, "rocm")) {
		return nullptr; // not the GPU path (cpu / cuda handled by the ORT backend)
	}
	const string task_name = TabFMTaskName(resolved.manifest.task);
	auto graph = GetBundledResource("graph_migraphx_" + task_name);
	if (!graph.data || !WeightsHeaderMatches(fs, resolved.weights_path, resolved.manifest.task)) {
		return nullptr;
	}
	const auto dir = DirName(resolved.weights_path);
	const auto graph_path = fs.JoinPath(dir, "graph_migraphx_" + task_name + ".onnx");
	if (!StageBundledGraph(fs, graph, graph_path)) {
		return nullptr;
	}
	const auto mxr_dir = fs.JoinPath(ctx.cache_dir, "migraphx");
	shared_ptr<TabFMBackend> backend = MakeMIGraphXBackend(graph_path, dir, mxr_dir, device.arch,
	                                                       device.device_ordinal, ctx.gpu_precision, ctx.mxr_source);
	return RegisterBackend(state, resolved.cache_key, std::move(backend), device.device_id, 0);
}

// Build (or reuse) the ORT session for `resolved` and return a snapshot of the
// LoadedModel. Order: direct MIGraphX (ROCm GPU) -> low-memory external-data
// (CPU/CUDA, ORT reads weights from disk) -> read/mmap + in-memory injection.
shared_ptr<LoadedModel> LoadOrGetSession(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                         const PredictContext &ctx) {
	if (auto snapshot = state.Snapshot(resolved.cache_key)) {
		return snapshot;
	}

	// ROCm GPU: direct MIGraphX backend (bypasses ORT's unusable MIGraphX EP).
	if (auto gpu = TryMIGraphXBackend(fs, state, resolved, ctx)) {
		return gpu;
	}
	// Low-memory path (external-data graph; ORT reads weights from disk).
	if (auto external = TryExternalDataSession(fs, state, resolved, ctx)) {
		return external;
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
	auto view = ParseSafetensors(bytes, nbytes, resolved.weights_path);
	auto arena = MaterializeF32Arena(view);

	// Build one injected initializer per safetensors tensor, named by the graph
	// (ONNX) initializer name. The tensor map is onnx->st; invert it, and fall
	// back to the "m." wrapper prefix for any tensor the map omits.
	unordered_map<string, string> st_to_onnx;
	for (auto &kv : resolved.tensor_map) {
		st_to_onnx[kv.second] = kv.first;
	}
	vector<TabFMTensorRef> initializers;
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

	const idx_t weight_bytes = nbytes;
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
	return RegisterBackend(state, resolved.cache_key, std::move(backend), config.device_id, weight_bytes);
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

		// Resolve the model up front (no engine access yet): its preprocessing
		// profile decides whether the engine standardizes features. TabFM/Mitra
		// want the default z-score (Mitra is rank-invariant to it); TabPFN/TabICL
		// normalize INSIDE the graph and must get raw features — they declare a
		// "*_raw" profile, and z-scoring here would double-normalize them.
		auto resolved = ResolveModel(*fs, in.ctx, task, in.opts.model);
		const bool standardize = !StringUtil::EndsWith(resolved.manifest.preprocessing_profile, "_raw");

		// 1. preprocess
		vector<PreprocessColumnSpec> columns;
		auto collection = BuildCollection(in.rows, in.row_type, columns, in.target_idx);
		auto pp_task = task == TabFMTask::CLASSIFICATION ? PreprocessTask::CLASSIFICATION : PreprocessTask::REGRESSION;
		auto batch = PreprocessBatch(collection, columns, pp_task, standardize);

		if (task == TabFMTask::CLASSIFICATION && batch.label_decoder.size() > 10) {
			throw InvalidInputException(
			    "target '%s' has %llu distinct labels; TabFM v1 supports at most 10. Consider grouping rare labels.",
			    in.target_name, static_cast<unsigned long long>(batch.label_decoder.size()));
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
			lock_guard<mutex> device_guard(state->DeviceMutex(in.ctx.device));
			auto model = LoadOrGetSession(*fs, *state, resolved, in.ctx);
			auto *backend = reinterpret_cast<TabFMBackend *>(model->session.get());
			out = backend->Run(run_input);
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
	lock_guard<mutex> device_guard(state->DeviceMutex(ctx.device));
	auto model = LoadOrGetSession(*fs, *state, resolved, ctx);
	auto *backend = reinterpret_cast<TabFMBackend *>(model->session.get());
	backend->Precompile(rows, features);
}

} // namespace anofox
} // namespace duckdb
