#include "tabfm_registration.hpp"
#include "tabfm_soname_patch.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_predict.hpp"
#include "tabfm_registry.hpp"
#include "tabfm_state.hpp"
#include "tabfm_weights.hpp"
#include "telemetry.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_open_flags.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"

#include "duckdb/common/numeric_utils.hpp"

#include "yyjson.hpp"
#include "miniz.hpp"

#include <algorithm>
#include <cstdlib>
#include "anofox_tabfm_banner.hpp"

namespace duckdb {
namespace anofox {

// Weights lifecycle: tabfm_download / tabfm_models / tabfm_load / tabfm_unload /
// tabfm_remove (FR-1, FR-2, FR-4). Owned by workstream WS-D.
//
// Design (HLD §4.2, spike S03):
// - Every remote byte flows through FileSystem::GetFileSystem(context), so
//   httpfs supplies TLS/proxy/secrets and DuckDB core supplies the
//   "INSTALL httpfs; LOAD httpfs;" MissingExtensionException for https:// URLs.
// - Preflight via OpenFile + GetFileSize is HEAD-cheap (S03 §4); the body is
//   streamed in 8 MiB chunks to <target>.part, fsynced, then atomically
//   renamed (FR-1.6). Byte counts are verified against the manifest and the
//   remote size (FR-1.7). Restart-not-resume: .part is truncated on reopen.
// - Download source handles are opened with CachingMode::NO_CACHING and the
//   plain (non-Caching) FileSystem, so DuckDB's external file cache never
//   buffers multi-GB weight bodies (S03 §2c caveat).

// A gated repository answers an anonymous GET with 401/403, and httpfs reports
// only the status line. Rewrite that into the CREATE SECRET (and, for
// HuggingFace, the license page) that actually unblocks the download.
// Declared in tabfm_weights.hpp; unit-tested in test/cpp/test_tabfm_weights.cpp.
string HttpAuthRemediation(const string &error, const string &url) {
	// Match the status as a standalone number: "wrote 4031 of ..." is not a 403.
	auto find_status = [&error](const char *code) -> bool {
		for (size_t pos = error.find(code); pos != string::npos; pos = error.find(code, pos + 1)) {
			const bool digit_before = pos > 0 && StringUtil::CharacterIsDigit(error[pos - 1]);
			const size_t after = pos + 3;
			const bool digit_after = after < error.size() && StringUtil::CharacterIsDigit(error[after]);
			if (!digit_before && !digit_after) {
				return true;
			}
		}
		return false;
	};
	const bool unauthorized = find_status("401");
	if (!unauthorized && !find_status("403")) {
		return ""; // 404 / connection / missing-httpfs keep their own remediation
	}
	const char *status = unauthorized ? "401 Unauthorized" : "403 Forbidden";

	// Split "<scheme>://<host>/<path>" — the query string never enters a message.
	auto scheme_end = url.find("://");
	auto host_start = scheme_end == string::npos ? 0 : scheme_end + 3;
	auto host_end = url.find('/', host_start);
	const string host = url.substr(host_start, host_end == string::npos ? string::npos : host_end - host_start);
	const string scheme = scheme_end == string::npos ? "https" : url.substr(0, scheme_end);

	if (StringUtil::EndsWith(host, "huggingface.co") && host_end != string::npos) {
		// "<owner>/<repo>/resolve/<rev>/<file>" → the gated repo is the first two.
		auto parts = StringUtil::Split(url.substr(host_end + 1), '/');
		if (parts.size() >= 2) {
			const string repo = parts[0] + "/" + parts[1];
			return StringUtil::Format(
			    "tabfm_download: access denied (HTTP %s) for the gated repository %s://%s/%s. "
			    "Fix both: (1) accept the model license at %s://%s/%s while signed in, then "
			    "(2) supply your token — CREATE SECRET hf (TYPE http, BEARER_TOKEN '<hf_token>', SCOPE '%s://%s');",
			    status, scheme, host, repo, scheme, host, repo, scheme, host);
		}
	}
	return StringUtil::Format("tabfm_download: access denied (HTTP %s) by host %s. Supply credentials with "
	                          "CREATE SECRET dl (TYPE http, BEARER_TOKEN '<token>', SCOPE '%s://%s');",
	                          status, host, scheme, host);
}

namespace {

constexpr idx_t kChunkBytes = 8ULL * 1024 * 1024;

//===--------------------------------------------------------------------===//
// Manifest resolution
// integration: swap to anofox::ModelManifest (WS-B, src/include/tabfm_manifest.hpp)
//===--------------------------------------------------------------------===//

struct WeightsFileEntry {
	string path;       // path inside the model dir, e.g. "classification/model.safetensors"
	string url;        // download source; empty = air-gapped manifest (pre-seeded cache only)
	int64_t bytes = -1; // expected size; -1 = unknown (use remote preflight size)
};

struct WeightsManifest {
	string model;    // manifest id, e.g. "tabfm-v1"
	string task;     // "classification" / "regression" / custom
	string repo;     // e.g. "google/tabfm-1.0.0-pytorch"
	string revision; // default "main"
	string license;  // license id; "" or "none" = ungated
	vector<WeightsFileEntry> files;
	bool builtin = false;
	//! Registered/user models resolve their files here instead of the cache
	//! ("" for built-ins). Without it tabfm_models() cannot see them at all.
	string source_dir;

	bool IsGated() const {
		return !license.empty() && license != "none";
	}
	// Cache layout (HLD §5): <cache_dir>/<repo with '/'→'__'>@<revision>/<file path>
	string CacheSlug(const string &effective_revision) const {
		auto base = repo.empty() ? model : StringUtil::Replace(repo, "/", "__");
		return base + "@" + effective_revision;
	}
};

string ExpandHomeDirectory(const string &path) {
	if (path.empty() || path[0] != '~') {
		return path;
	}
	const char *home = std::getenv("HOME");
	string home_dir = home ? string(home) : FileSystem::CreateLocal()->GetHomeDirectory();
	return home_dir + path.substr(1);
}

string GetCacheDir(ClientContext &context) {
	Value value;
	string dir = "~/.cache/anofox-tabfm";
	if (context.TryGetCurrentSetting("anofox_tabfm_cache_dir", value) && !value.IsNull()) {
		dir = value.ToString();
	}
	return ExpandHomeDirectory(dir);
}

bool LicenseAccepted(ClientContext &context) {
	Value value;
	if (!context.TryGetCurrentSetting("anofox_tabfm_accept_hf_license", value) || value.IsNull()) {
		return false;
	}
	return BooleanValue::Get(value.DefaultCastAs(LogicalType::BOOLEAN));
}

// Build the weights-view of one (model, task) from a registry ModelSpec. The
// single manifest parser is now ParseModelSpec (tabfm_model_spec.cpp); this
// module no longer parses manifest JSON — it shares the registry with predict.
WeightsManifest WeightsFromSpec(const ModelSpec &spec, TabFMTask task) {
	WeightsManifest result;
	result.model = spec.id;
	result.task = TabFMTaskName(task);
	const auto &art = spec.tasks.at(task);
	result.repo = art.repo;
	result.revision = art.revision;
	// Gated iff the license declares a gate_setting (e.g. accept_hf_license).
	result.license = spec.license.gate_setting.empty() ? "none" : spec.license.id;
	result.builtin = spec.source_dir.empty(); // built-ins carry no source dir
	result.source_dir = spec.source_dir;
	for (auto &f : art.files) {
		WeightsFileEntry entry;
		entry.path = f.path;
		entry.url = f.url;
		entry.bytes = f.bytes == 0 ? -1 : NumericCast<int64_t>(f.bytes);
		result.files.push_back(std::move(entry));
	}
	return result;
}

string StringSetting(ClientContext &context, const char *name) {
	Value value;
	if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) {
		return value.ToString();
	}
	return "";
}

// The lifecycle functions (download/load/unload/remove/gpu_precompile) accept an
// optional `model :=` to target a specific registered model; empty resolves the
// default (anofox_tabfm_default_model or the sole registered model).
string ModelArg(TableFunctionBindInput &input) {
	auto it = input.named_parameters.find("model");
	return (it == input.named_parameters.end() || it->second.IsNull()) ? string() : it->second.ToString();
}

// Every (model, task) the registry knows this session — built-ins + user
// plus SQL-registered models. The single source of
// truth, shared with the predict path.
vector<WeightsManifest> ResolveManifests(ClientContext &context) {
	auto registry = ModelRegistry::Build(TabFMState::Get(context)->RegisteredSpecs());
	vector<WeightsManifest> result;
	for (auto &kv : registry.Models()) {
		for (auto &task_kv : kv.second.tasks) {
			result.push_back(WeightsFromSpec(kv.second, task_kv.first));
		}
	}
	return result;
}

TabFMTask ParseTaskArg(const string &task, const char *func_name) {
	if (task == "classification") {
		return TabFMTask::CLASSIFICATION;
	}
	if (task == "regression") {
		return TabFMTask::REGRESSION;
	}
	throw InvalidInputException("%s: unknown task '%s'. Valid tasks: classification, regression", func_name, task);
}

// Resolve one (model, task) via the registry: the resolved model (model_id, may
// be "" → default_model / single-file manifest / sole model) for `task`.
WeightsManifest FindManifest(ClientContext &context, const char *func_name, const string &model_id,
                             const string &task) {
	auto tfm_task = ParseTaskArg(task, func_name);
	auto registry = ModelRegistry::Build(TabFMState::Get(context)->RegisteredSpecs());
	const ModelSpec &spec = registry.Resolve(model_id, StringSetting(context, "anofox_tabfm_default_model"));
	if (!spec.HasTask(tfm_task)) {
		throw InvalidInputException(
		    "%s: model '%s' does not support task '%s'. Select a model that does (model := '<id>', or "
		    "CALL tabfm_register_model(...)); see tabfm_list_models().",
		    func_name, spec.id, task);
	}
	return WeightsFromSpec(spec, tfm_task);
}

string RequireTaskArgument(const TableFunctionBindInput &input, const char *func_name) {
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("%s: task cannot be NULL", func_name);
	}
	return StringValue::Get(input.inputs[0]);
}

// URL shown to the user: host+path only — pre-signed CDN URLs carry auth in
// the query string, which must never leak into results or error messages.
string SanitizeUrl(const string &url) {
	auto pos = url.find('?');
	return pos == string::npos ? url : url.substr(0, pos);
}

string FileUrl(const WeightsManifest &manifest, const WeightsFileEntry &file, const string &revision) {
	if (!file.url.empty()) {
		return file.url;
	}
	// Built-in models derive the HF URL from repo/revision/path. User manifests
	// are air-gapped unless a file carries an explicit url (FR-1.5); a future
	// downloadable provider (e.g. Mitra) will carry per-file urls / url_template.
	if (manifest.builtin) {
		return "https://huggingface.co/" + manifest.repo + "/resolve/" + revision + "/" + file.path;
	}
	return "";
}

string ParentDirectory(const string &path) {
	auto pos = path.find_last_of('/');
	return pos == string::npos ? string(".") : path.substr(0, pos);
}

bool DirectoryIsEmpty(FileSystem &fs, const string &dir) {
	bool empty = true;
	fs.ListFiles(dir, [&](const string &, bool) { empty = false; });
	return empty;
}

void RemoveDirectoryIfEmpty(FileSystem &fs, const string &dir) {
	if (fs.DirectoryExists(dir) && DirectoryIsEmpty(fs, dir)) {
		fs.RemoveDirectory(dir);
	}
}

//===--------------------------------------------------------------------===//
// License gate (FR-2.1)
//===--------------------------------------------------------------------===//

void RequireLicenseAccepted(ClientContext &context, const WeightsManifest &manifest) {
	if (!manifest.IsGated() || LicenseAccepted(context)) {
		return;
	}
	auto what = manifest.repo.empty() ? manifest.model : manifest.repo;
	throw InvalidConfigurationException(
	    "tabfm_download: weights in '%s' are licensed '%s' (non-commercial, no redistribution). "
	    "Run: SET anofox_tabfm_accept_hf_license = true;",
	    what, manifest.license);
}

// Acceptance metadata, written once on the first gated download (FR-2.1).
void RecordLicenseAcceptance(ClientContext &context, const string &cache_dir, const WeightsManifest &manifest,
                             const string &revision) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto lock_path = cache_dir + "/manifest-lock.json";
	if (fs.FileExists(lock_path)) {
		return;
	}
	fs.CreateDirectoriesRecursive(cache_dir);
	auto timestamp = Timestamp::ToString(Timestamp::GetCurrentTimestamp());
	auto content = StringUtil::Format("{\n\t\"license\": \"%s\",\n\t\"timestamp\": \"%s\",\n\t\"revision\": \"%s\"\n}\n",
	                                  manifest.license, timestamp, revision);
	auto handle =
	    fs.OpenFile(lock_path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
	handle->Write((void *)content.data(), content.size());
	handle->Sync();
}

//===--------------------------------------------------------------------===//
// tabfm_download(task [, revision])
//===--------------------------------------------------------------------===//

struct DownloadItem {
	string cache_path;
	string url;         // "" = no source (air-gapped manifest)
	int64_t bytes = -1; // expected bytes from the manifest, -1 unknown
};

struct DownloadBindData : public TableFunctionData {
	WeightsManifest manifest;
	string revision;
	string cache_dir;
	vector<DownloadItem> items;
};

struct DownloadGlobalState : public GlobalTableFunctionState {
	idx_t next = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<FunctionData> DownloadBind(ClientContext &context, TableFunctionBindInput &input,
                                      vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_download");

	auto task = RequireTaskArgument(input, "tabfm_download");
	auto result = make_uniq<DownloadBindData>();
	result->manifest = FindManifest(context, "tabfm_download", ModelArg(input), task);
	result->revision = result->manifest.revision;
	if (input.inputs.size() > 1) {
		if (input.inputs[1].IsNull()) {
			throw InvalidInputException("tabfm_download: revision cannot be NULL");
		}
		result->revision = StringValue::Get(input.inputs[1]);
	}

	// License gate BEFORE any network or filesystem I/O (FR-2.1).
	RequireLicenseAccepted(context, result->manifest);

	result->cache_dir = GetCacheDir(context);
	auto base_dir = result->cache_dir + "/" + result->manifest.CacheSlug(result->revision);
	for (auto &file : result->manifest.files) {
		DownloadItem item;
		item.cache_path = base_dir + "/" + file.path;
		item.url = FileUrl(result->manifest, file, result->revision);
		item.bytes = file.bytes;
		result->items.push_back(std::move(item));
	}

	names = {"file", "url", "bytes", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR};
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> DownloadInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<DownloadBindData>();
	if (bind.manifest.IsGated()) {
		// The gate passed at bind time; persist the acceptance (FR-2.1).
		RecordLicenseAcceptance(context, bind.cache_dir, bind.manifest, bind.revision);
	}
	return make_uniq<DownloadGlobalState>();
}

// Streams one source file to <cache_path>.part, fsyncs, verifies the byte
// count, then atomically renames (FR-1.6/1.7). Returns the byte count.
idx_t FetchFile(ClientContext &context, const DownloadItem &item) {
	auto &fs = FileSystem::GetFileSystem(context);

	// Preflight + open. For https:// URLs without httpfs loaded, DuckDB core
	// throws a MissingExtensionException here whose text already contains the
	// "INSTALL httpfs; LOAD httpfs;" remediation — let it propagate (S03 §5).
	auto read_flags = FileOpenFlags(FileFlags::FILE_FLAGS_READ);
	read_flags.SetCachingMode(CachingMode::NO_CACHING); // S03 §2c: never cache weight bodies
	unique_ptr<FileHandle> src;
	int64_t remote_size;
	try {
		src = fs.OpenFile(item.url, read_flags);
		remote_size = fs.GetFileSize(*src); // HEAD-cheap even on multi-GB files (S03 §4)
	} catch (std::exception &e) {
		// A gated repo fails here with a bare 401/403; everything else (404,
		// connection errors, the missing-httpfs MissingExtensionException) keeps
		// its own remediation and propagates untouched.
		auto hint = HttpAuthRemediation(e.what(), item.url);
		if (hint.empty()) {
			throw;
		}
		throw IOException(hint);
	}
	if (item.bytes >= 0 && remote_size >= 0 && remote_size != item.bytes) {
		throw IOException("tabfm_download: size mismatch for %s (manifest expects %lld bytes, source reports %lld) "
		                  "— download aborted",
		                  SanitizeUrl(item.url), item.bytes, remote_size);
	}

	fs.CreateDirectoriesRecursive(ParentDirectory(item.cache_path));
	auto part = item.cache_path + ".part";
	// FILE_CREATE_NEW truncates a stale .part from a crashed run: restart, not resume.
	auto dst = fs.OpenFile(part, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);

	auto buffer = make_unsafe_uniq_array<char>(kChunkBytes);
	idx_t total = 0;
	while (true) {
		auto n = src->Read(buffer.get(), kChunkBytes);
		if (n <= 0) {
			break;
		}
		dst->Write(buffer.get(), n);
		total += NumericCast<idx_t>(n);
	}
	dst->Sync();
	dst.reset();

	auto expected = item.bytes >= 0 ? item.bytes : remote_size;
	if (expected >= 0 && NumericCast<int64_t>(total) != expected) {
		fs.TryRemoveFile(part);
		throw IOException("tabfm_download: size mismatch for %s (wrote %llu of %lld bytes) — download aborted, "
		                  "partial file removed",
		                  SanitizeUrl(item.url), total, expected);
	}
	fs.MoveFile(part, item.cache_path); // atomic publish (FR-1.6)
	return total;
}

void DownloadExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<DownloadBindData>();
	auto &state = data.global_state->Cast<DownloadGlobalState>();
	auto &fs = FileSystem::GetFileSystem(context);

	idx_t out = 0;
	while (state.next < bind.items.size() && out < STANDARD_VECTOR_SIZE) {
		auto &item = bind.items[state.next++];
		string status;
		int64_t bytes = 0;
		bool cached = false;
		if (fs.FileExists(item.cache_path)) {
			auto handle = fs.OpenFile(item.cache_path, FileFlags::FILE_FLAGS_READ);
			bytes = fs.GetFileSize(*handle);
			// only trust the cached copy if it matches the manifest size (FR-1.7)
			cached = item.bytes < 0 || bytes == item.bytes;
		}
		if (cached) {
			status = "cached";
		} else {
			if (item.url.empty()) {
				throw IOException("tabfm_download: '%s' is not cached and the manifest provides no download url "
				                  "(air-gapped manifest). Pre-seed the cache dir (FR-1.5) or add a url.",
				                  item.cache_path);
			}
			bytes = NumericCast<int64_t>(FetchFile(context, item));
			status = "downloaded";
		}
		output.SetValue(0, out, Value(item.cache_path));
		output.SetValue(1, out, Value(SanitizeUrl(item.url)));
		output.SetValue(2, out, Value::BIGINT(bytes));
		output.SetValue(3, out, Value(status));
		out++;
	}
	output.SetCardinality(out);
}

//===--------------------------------------------------------------------===//
// tabfm_download_runtime(backend) — docs/DYNAMIC_BACKENDS.md phase 3
//
// Fetches an ONNX Runtime execution-provider plugin library so a device can
// be driven without a compile-time flavor build (SET anofox_tabfm_ep_path
// names the directory this populates; tabfm_ort_engine.cpp's
// RegisterCudaProvider reads from it at predict time).
//
// The provider .so is not published standalone anywhere; it ships inside an
// onnxruntime-gpu wheel (a ZIP) alongside the small providers_shared.so it
// links against. A GitHub release tarball also carries it, but as a .tar.gz —
// this vendors miniz (already linked into duckdb core) for ZIP reading and
// would otherwise need a TAR reader too, so the wheel is the lighter path.
// Extraction is local-disk random-access (mz_zip_reader_init_file + locate +
// extract_to_file): only the two needed entries are decompressed, not the
// whole wheel.
//
// NOT the plain PyPI `onnxruntime-gpu` package: as of 1.28+ that targets
// CUDA 13 (needs libcublasLt.so.13 and friends) with no CUDA-12 variant on
// PyPI proper — discovered by actually running this on a RunPod CUDA-12.4
// image, where the plain-PyPI wheel's provider failed to dlopen. Microsoft
// publishes a CUDA-12-targeted build on a separate Azure Artifacts feed
// (linked from https://onnxruntime.ai/docs/install/, "onnxruntime-cuda-12"),
// which is what CUDA_12_WHEEL_URL below points at — matches the CUDA major
// this codebase otherwise defaults to (cmake/ort.cmake's TABFM_ORT_CUDA_MAJOR).
// That feed's download links are pip-index redirects to a time-limited SAS
// URL, not the final blob itself; FetchFile follows the redirect like any
// other HTTP client.
//===--------------------------------------------------------------------===//

namespace {

//! One backend -> the wheel that carries its provider .so(s),
//! and which entries inside it to extract. Deliberately narrow: only what has
//! been verified to exist in the wheel (see docs/DYNAMIC_BACKENDS.md phase 3
//! measurements) is offered — an unlisted backend errors naming what is.
//! One file to lift out of the wheel. `dest_name` exists because the ORT core
//! must land under the exact name the plugin's DT_NEEDED asks for.
//! `patch_soname` renames the core in flight (upstream SONAME →
//! libanofoxort_gpu.so, equal length, docs/GPU_HARDENING_PLAN.md S2): a core
//! keeping upstream's SONAME gets shadowed by any host-loaded ORT, so the
//! shipped runtime carries a private name and the plugin links against it.
struct RuntimeEntry {
	string zip_path;
	string dest_name;
	bool patch_soname = false;
};

struct RuntimeArtifact {
	string wheel_url;
	int64_t wheel_bytes = 0;
	vector<RuntimeEntry> zip_entries;
	//! The backend plugin itself, fetched from this repo's GitHub release
	//! assets (built by .github/workflows/gpu_plugins.yml — neither plugin
	//! needs a GPU to build). Empty url = no plugin-carrying release is pinned
	//! in this build yet.
	string plugin_name;
	string plugin_url;
};

//! The release tag whose assets carry the built plugins. "" until the first
//! plugin-carrying release is cut; bumped per release thereafter. Kept empty
//! rather than guessed so download errors say "not published yet" instead of
//! 404-ing at a URL that never existed.
constexpr const char *TABFM_PLUGIN_RELEASE_TAG = "";

string PluginReleaseUrl(const string &asset) {
	if (string(TABFM_PLUGIN_RELEASE_TAG).empty()) {
		return "";
	}
	return string("https://github.com/DataZooDE/anofox-tabfm/releases/download/") + TABFM_PLUGIN_RELEASE_TAG + "/" +
	       asset;
}

//! Currently the only published, verified source: onnxruntime-gpu 1.29.0's
//! manylinux CUDA-12 wheel from Microsoft's onnxruntime-cuda-12 Azure
//! Artifacts feed (cp312 tag — the C++ .so payload is identical across cp3xx
//! tags; the tag only affects the Python bindings we discard). CUDA 13 /
//! other ORT versions are not offered yet (no reason they couldn't be, just
//! not measured — extend this map when they are).
bool ResolveRuntimeArtifact(const string &backend, RuntimeArtifact &out, string &error) {
	if (backend == "cuda") {
		out.wheel_url = "https://aiinfra.pkgs.visualstudio.com/2692857e-05ef-43b4-ba9c-ccf1c22c437c/"
		                "_packaging/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/download/onnxruntime-gpu/1.29/"
		                "onnxruntime_gpu-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl";
		out.wheel_bytes = 475434900;
		// The ORT *core* comes along too, not just the providers: the CUDA
		// backend is a standalone plugin with its own shared runtime, because a
		// statically-linked ORT cannot host these provider libraries at all (see
		// src/tabfm_cuda_plugin.cpp). Core and providers must come from one
		// distribution — pairing a CPU-flavor core with a GPU provider crashes.
		out.zip_entries = {{"onnxruntime/capi/libonnxruntime.so.1.29.0", ORT_RENAMED_SONAME, /*patch_soname=*/true},
		                   {"onnxruntime/capi/libonnxruntime_providers_cuda.so", "libonnxruntime_providers_cuda.so"},
		                   {"onnxruntime/capi/libonnxruntime_providers_shared.so",
		                    "libonnxruntime_providers_shared.so"}};
		out.plugin_name = "libanofox_tabfm_cuda_plugin.so";
		out.plugin_url = PluginReleaseUrl(out.plugin_name);
		return true;
	}
	if (backend == "rocm") {
		out.plugin_name = "libanofox_tabfm_migraphx_plugin.so";
		out.plugin_url = PluginReleaseUrl(out.plugin_name);
		if (out.plugin_url.empty()) {
			error = "tabfm_download_runtime: no plugin-carrying release is pinned in this build yet. The MIGraphX "
			        "plugin is built by CI (workflow 'GPU backend plugins', artifact "
			        "anofox-tabfm-migraphx-plugin) and by the anofox_tabfm_migraphx_plugin CMake target — place "
			        "libanofox_tabfm_migraphx_plugin.so in the anofox_tabfm_ep_path directory.";
			return false;
		}
		return true;
	}
	error = "tabfm_download_runtime: unknown backend '" + backend + "' — expected 'cuda' or 'rocm'.";
	return false;
}

struct DownloadRuntimeBindData : public TableFunctionData {
	string backend;
	string ep_path;
	string cache_dir;
	RuntimeArtifact artifact;
};

struct DownloadRuntimeGlobalState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<GlobalTableFunctionState> DownloadRuntimeInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<DownloadRuntimeGlobalState>();
}

unique_ptr<FunctionData> DownloadRuntimeBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_download_runtime");

	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("tabfm_download_runtime: backend cannot be NULL — expected 'cuda' or 'rocm'");
	}
	auto result = make_uniq<DownloadRuntimeBindData>();
	result->backend = StringUtil::Lower(StringValue::Get(input.inputs[0]));

	string error;
	if (!ResolveRuntimeArtifact(result->backend, result->artifact, error)) {
		throw InvalidInputException(error);
	}

	result->cache_dir = GetCacheDir(context);
	result->ep_path = result->cache_dir + "/runtime";
	Value setting;
	if (context.TryGetCurrentSetting("anofox_tabfm_ep_path", setting) && !setting.IsNull() &&
	    !setting.ToString().empty()) {
		result->ep_path = ExpandHomeDirectory(setting.ToString());
	}

	names = {"file", "bytes", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::VARCHAR};
	return std::move(result);
}

void DownloadRuntimeExecute(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<DownloadRuntimeBindData>();
	auto &state = data.global_state->Cast<DownloadRuntimeGlobalState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;

	auto &fs = FileSystem::GetFileSystem(context);
	fs.CreateDirectoriesRecursive(bind.ep_path);

	// Which artifacts are already present — fetching is per-file idempotent, so
	// a partial prior run only re-fetches what it must. The plugin (when a
	// release carries one) is one more target alongside the wheel entries.
	vector<string> targets;
	bool all_present = true;
	for (auto &entry : bind.artifact.zip_entries) {
		string target = bind.ep_path + "/" + entry.dest_name;
		targets.push_back(target);
		if (!fs.FileExists(target)) {
			all_present = false;
		}
	}
	const string plugin_target =
	    bind.artifact.plugin_url.empty() ? "" : bind.ep_path + "/" + bind.artifact.plugin_name;
	if (!plugin_target.empty()) {
		targets.push_back(plugin_target);
		if (!fs.FileExists(plugin_target)) {
			all_present = false;
		}
	}

	idx_t out = 0;
	if (all_present) {
		for (size_t i = 0; i < targets.size(); i++) {
			auto handle = fs.OpenFile(targets[i], FileFlags::FILE_FLAGS_READ);
			output.SetValue(0, out, Value(targets[i]));
			output.SetValue(1, out, Value::BIGINT(NumericCast<int64_t>(fs.GetFileSize(*handle))));
			output.SetValue(2, out, Value("cached"));
			out++;
		}
		output.SetCardinality(out);
		return;
	}

	// Whole-wheel download (FetchFile — same atomic .part-then-rename fetch
	// tabfm_download uses), then in-memory ZIP extraction of just the needed
	// entries: the vendored miniz is built with MINIZ_NO_STDIO (no file-based
	// zip reader), so the archive is read into a heap buffer and extracted via
	// mz_zip_reader_init_mem / extract_to_heap rather than the file-path API.
	// The wheel itself is not kept — it is scratch, not a cache entry
	// (re-downloaded on the next miss, same as a truncated .part would be).
	// Skipped entirely for a backend with no wheel (rocm ships only a plugin).
	if (!bind.artifact.wheel_url.empty()) {
		const string wheel_path = bind.ep_path + "/.onnxruntime_gpu-1.29.0-cp312.whl";
		DownloadItem wheel_item;
		wheel_item.cache_path = wheel_path;
		wheel_item.url = bind.artifact.wheel_url;
		wheel_item.bytes = bind.artifact.wheel_bytes;
		FetchFile(context, wheel_item);

		auto wheel_handle = fs.OpenFile(wheel_path, FileFlags::FILE_FLAGS_READ);
		const idx_t wheel_size = fs.GetFileSize(*wheel_handle);
		auto wheel_bytes = make_unsafe_uniq_array<char>(wheel_size);
		wheel_handle->Read(wheel_bytes.get(), wheel_size);
		wheel_handle.reset();

		duckdb_miniz::mz_zip_archive zip {};
		if (!duckdb_miniz::mz_zip_reader_init_mem(&zip, wheel_bytes.get(), wheel_size, 0)) {
			fs.TryRemoveFile(wheel_path);
			throw IOException("tabfm_download_runtime: '%s' downloaded but is not a readable zip archive — the source "
			                  "may have changed shape; please report this.",
			                  SanitizeUrl(bind.artifact.wheel_url));
		}
		for (size_t i = 0; i < bind.artifact.zip_entries.size(); i++) {
			const string &entry = bind.artifact.zip_entries[i].zip_path;
			int file_index = duckdb_miniz::mz_zip_reader_locate_file(&zip, entry.c_str(), nullptr, 0);
			if (file_index < 0) {
				duckdb_miniz::mz_zip_reader_end(&zip);
				fs.TryRemoveFile(wheel_path);
				throw IOException("tabfm_download_runtime: expected entry '%s' not found in the downloaded wheel — the "
				                  "source may have changed shape; please report this.",
				                  entry);
			}
			size_t extracted_size = 0;
			void *extracted =
			    duckdb_miniz::mz_zip_reader_extract_to_heap(&zip, static_cast<duckdb_miniz::mz_uint>(file_index),
			                                                &extracted_size, 0);
			if (!extracted) {
				duckdb_miniz::mz_zip_reader_end(&zip);
				fs.TryRemoveFile(wheel_path);
				throw IOException("tabfm_download_runtime: failed to extract '%s' from the downloaded wheel.", entry);
			}
			if (bind.artifact.zip_entries[i].patch_soname) {
				// Exactly one occurrence or refuse: zero means the wheel changed
				// shape, several means the only-the-.dynstr-entry assumption broke.
				// Either way a half-renamed core must fail HERE, at download time,
				// not later as a plugin that quietly binds to the host's ORT.
				auto replaced = PatchOrtSonameInPlace(static_cast<char *>(extracted), extracted_size);
				if (replaced != 1) {
					duckdb_miniz::mz_free(extracted);
					duckdb_miniz::mz_zip_reader_end(&zip);
					fs.TryRemoveFile(wheel_path);
					throw IOException("tabfm_download_runtime: expected exactly one SONAME string in '%s' but found %llu "
					                  "— the wheel changed shape; please report this.",
					                  entry, static_cast<unsigned long long>(replaced));
				}
			}
			const string tmp = targets[i] + ".part";
			auto dst = fs.OpenFile(tmp, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
			dst->Write(extracted, NumericCast<int64_t>(extracted_size));
			dst->Sync();
			dst.reset();
			duckdb_miniz::mz_free(extracted);
			fs.MoveFile(tmp, targets[i]); // atomic publish, same discipline as FetchFile
		}
		duckdb_miniz::mz_zip_reader_end(&zip);
		wheel_bytes.reset();
		fs.TryRemoveFile(wheel_path);
	}

	// The backend plugin, from this repo's release assets (built by the
	// gpu_plugins workflow): same atomic FetchFile, idempotent via the
	// existence check above, reported as one more output row below.
	if (!plugin_target.empty() && !fs.FileExists(plugin_target)) {
		DownloadItem plugin_item;
		plugin_item.cache_path = plugin_target;
		plugin_item.url = bind.artifact.plugin_url;
		plugin_item.bytes = -1; // release assets carry sha256 sidecars, not pinned sizes
		FetchFile(context, plugin_item);
	}

	for (auto &target : targets) {
		auto handle = fs.OpenFile(target, FileFlags::FILE_FLAGS_READ);
		output.SetValue(0, out, Value(target));
		output.SetValue(1, out, Value::BIGINT(NumericCast<int64_t>(fs.GetFileSize(*handle))));
		output.SetValue(2, out, Value("downloaded"));
		out++;
	}
	output.SetCardinality(out);
}

} // anonymous namespace

//===--------------------------------------------------------------------===//
// tabfm_models()
//===--------------------------------------------------------------------===//

struct ModelsRow {
	string model;
	string task;
	string revision;
	string path;
	int64_t bytes = 0;
	bool loaded = false;
	string license;
	//! Which backend actually serves this model right now: "cpu", "cuda:0",
	//! "rocm:0"... Empty while unloaded. Without this there is no way to tell a
	//! GPU run from a CPU one short of deliberately breaking the plugin path and
	//! checking that it complains -- which is how the ROCm run in
	//! tools/gpu_test/usecase_equivalence.sql had to be validated.
	string device;
};

struct ModelsBindData : public TableFunctionData {};

struct ModelsGlobalState : public GlobalTableFunctionState {
	vector<ModelsRow> rows;
	idx_t next = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<FunctionData> ModelsBind(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_models");
	names = {"model", "task", "revision", "path", "bytes", "loaded", "license", "device"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::BIGINT,  LogicalType::BOOLEAN, LogicalType::VARCHAR, LogicalType::VARCHAR};
	return make_uniq<ModelsBindData>();
}

unique_ptr<GlobalTableFunctionState> ModelsInit(ClientContext &context, TableFunctionInitInput &) {
	auto state = make_uniq<ModelsGlobalState>();
	auto tabfm_state = TabFMState::Get(context); // the live loaded-model map
	auto &fs = FileSystem::GetFileSystem(context);
	auto cache_dir = GetCacheDir(context);
	// A missing cache dir means no *downloaded* revisions — registered models
	// keep their weights in source_dir and must still be listed (a fresh
	// machine, or CI, has no cache dir at all; found when CI showed zero rows
	// for a just-registered model that listed fine on a warmed-up laptop).
	const bool cache_dir_exists = fs.DirectoryExists(cache_dir);
	for (auto &manifest : ResolveManifests(context)) {
		// Emit one row when every declared file exists under base_dir. 'loaded'
		// and 'device' reflect the real DB-instance TabFMState: a model is
		// loaded once a predict (or lifecycle load) has warmed its session; the
		// key format is shared with the engine via TabFMModelCacheKey.
		auto emit_row = [&](const string &base_dir, const string &revision) {
			ModelsRow row;
			row.model = manifest.model;
			row.task = manifest.task;
			row.revision = revision;
			row.license = manifest.license;
			bool complete = true;
			for (idx_t i = 0; i < manifest.files.size(); i++) {
				// ListableArtifactPath: a .ckpt whose converted sibling exists is
				// servable, so it is listable (hiding it made exactly the
				// converted-and-working models invisible).
				auto path = ListableArtifactPath(base_dir + "/" + manifest.files[i].path,
				                                 [&](const string &p) { return fs.FileExists(p); });
				if (path.empty()) {
					complete = false;
					break;
				}
				auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
				row.bytes += fs.GetFileSize(*handle);
				if (i == 0) {
					row.path = path; // primary artifact (the model file) per SQL-API §3
				}
			}
			if (!complete) {
				return;
			}
			// Sessions cache per (device, precision) since GPU_HARDENING_PLAN P5,
			// so a model can be loaded on several backends at once: one row per
			// loaded configuration, or a single unloaded row when none is.
			auto snapshots = tabfm_state->SnapshotsFor(TabFMModelCacheKey(manifest.model, manifest.task, revision));
			if (snapshots.empty()) {
				state->rows.push_back(std::move(row));
				return;
			}
			for (auto &snapshot : snapshots) {
				ModelsRow loaded_row = row;
				loaded_row.loaded = true;
				loaded_row.device = snapshot->device_id;
				state->rows.push_back(std::move(loaded_row));
			}
		};
		// one cache entry per downloaded revision: <slug-base>@<revision>
		auto slug_base = manifest.CacheSlug("");
		vector<string> revisions;
		if (cache_dir_exists) {
			fs.ListFiles(cache_dir, [&](const string &name, bool is_dir) {
				if (is_dir && StringUtil::StartsWith(name, slug_base)) {
					revisions.push_back(name.substr(slug_base.size()));
				}
			});
		}
		std::sort(revisions.begin(), revisions.end());
		for (auto &revision : revisions) {
			emit_row(cache_dir + "/" + manifest.CacheSlug(revision), revision);
		}
		// Registered/user models keep their weights next to the model
		// (source_dir), not in the download cache — the cache walk above cannot
		// see them, which left tabfm_models() (and its device column, the one
		// way to verify a GPU actually served a query) blind for exactly the
		// models tabfm_register_model adds. Found when the first registered
		// model ran on a GPU and could not prove it.
		if (!manifest.builtin && !manifest.source_dir.empty() && revisions.empty()) {
			emit_row(manifest.source_dir, manifest.revision);
		}
	}
	return std::move(state);
}

void ModelsExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ModelsGlobalState>();
	idx_t out = 0;
	while (state.next < state.rows.size() && out < STANDARD_VECTOR_SIZE) {
		auto &row = state.rows[state.next++];
		output.SetValue(0, out, Value(row.model));
		output.SetValue(1, out, Value(row.task));
		output.SetValue(2, out, Value(row.revision));
		output.SetValue(3, out, Value(row.path));
		output.SetValue(4, out, Value::BIGINT(row.bytes));
		output.SetValue(5, out, Value::BOOLEAN(row.loaded));
		output.SetValue(6, out, Value(row.license));
		output.SetValue(7, out, Value(row.device));
		out++;
	}
	output.SetCardinality(out);
}

//===--------------------------------------------------------------------===//
// tabfm_list_models() — the REGISTRY view: every known model, downloaded or not
//===--------------------------------------------------------------------===//

struct ListModelsRow {
	string model, family, capabilities, license;
	bool commercial = false;
	int64_t max_rows = -1, max_features = -1, max_classes = -1;
	bool downloaded = false;
};

struct ListModelsGlobalState : public GlobalTableFunctionState {
	vector<ListModelsRow> rows;
	idx_t next = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

// A task's weights are "complete" in the cache iff every file exists AND (when
// the manifest declares a byte count) its size matches. This mirrors the
// download path, which only trusts size-matching files — a truncated or
// zero-byte artifact must not report as downloaded (list_models parity).
bool TaskWeightsComplete(FileSystem &fs, const string &base_dir, const vector<WeightsFileEntry> &files) {
	if (files.empty()) {
		return false;
	}
	for (auto &f : files) {
		auto path = base_dir + "/" + f.path;
		if (!fs.FileExists(path)) {
			return false;
		}
		if (f.bytes >= 0) {
			auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
			if (fs.GetFileSize(*handle) != f.bytes) {
				return false;
			}
		}
	}
	return true;
}

unique_ptr<FunctionData> ListModelsBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                        vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_list_models");
	names = {"model",    "family",       "capabilities", "license",   "commercial",
	         "max_rows", "max_features", "max_classes",  "downloaded"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::BOOLEAN, LogicalType::BIGINT,
	                LogicalType::BIGINT,  LogicalType::BIGINT,  LogicalType::BOOLEAN};
	return make_uniq<ModelsBindData>();
}

unique_ptr<GlobalTableFunctionState> ListModelsInit(ClientContext &context, TableFunctionInitInput &) {
	auto state = make_uniq<ListModelsGlobalState>();
	auto registry = ModelRegistry::Build(TabFMState::Get(context)->RegisteredSpecs());
	auto &fs = FileSystem::GetFileSystem(context);
	auto cache_dir = GetCacheDir(context);
	for (auto &kv : registry.Models()) {
		const auto &spec = kv.second;
		ListModelsRow row;
		row.model = spec.id;
		row.family = spec.family;
		row.capabilities = StringUtil::Join(spec.capabilities, ", ");
		row.license = spec.license.id;
		row.commercial = spec.license.commercial;
		row.max_rows = spec.size_regime.max_rows;
		row.max_features = spec.size_regime.max_features;
		row.max_classes = spec.size_regime.max_classes;
		// downloaded = any task's weights are complete (present AND size-matched)
		for (auto &task_kv : spec.tasks) {
			auto wm = WeightsFromSpec(spec, task_kv.first);
			auto base = cache_dir + "/" + wm.CacheSlug(wm.revision);
			if (TaskWeightsComplete(fs, base, wm.files)) {
				row.downloaded = true;
				break;
			}
		}
		state->rows.push_back(std::move(row));
	}
	return std::move(state);
}

void ListModelsExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<ListModelsGlobalState>();
	auto nbig = [](int64_t v) { return v < 0 ? Value(LogicalType::BIGINT) : Value::BIGINT(v); };
	idx_t out = 0;
	while (state.next < state.rows.size() && out < STANDARD_VECTOR_SIZE) {
		auto &r = state.rows[state.next++];
		output.SetValue(0, out, Value(r.model));
		output.SetValue(1, out, Value(r.family));
		output.SetValue(2, out, Value(r.capabilities));
		output.SetValue(3, out, Value(r.license));
		output.SetValue(4, out, Value::BOOLEAN(r.commercial));
		output.SetValue(5, out, nbig(r.max_rows));
		output.SetValue(6, out, nbig(r.max_features));
		output.SetValue(7, out, nbig(r.max_classes));
		output.SetValue(8, out, Value::BOOLEAN(r.downloaded));
		out++;
	}
	output.SetCardinality(out);
}

//===--------------------------------------------------------------------===//
// tabfm_load(task) / tabfm_unload([task]) — status-row surface
//===--------------------------------------------------------------------===//

struct LifecycleBindData : public TableFunctionData {
	string task;   // "all" for bare tabfm_unload()
	string status; // "loaded" / "unloaded"
};

struct LifecycleGlobalState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};

void SetLifecycleColumns(vector<LogicalType> &return_types, vector<string> &names) {
	names = {"task", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
}

unique_ptr<FunctionData> LoadBind(ClientContext &context, TableFunctionBindInput &input,
                                  vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_load");
	auto task = RequireTaskArgument(input, "tabfm_load");
	auto manifest = FindManifest(context, "tabfm_load", ModelArg(input), task);

	// cache presence check — the §5 'not downloaded' contract
	auto &fs = FileSystem::GetFileSystem(context);
	auto base_dir = GetCacheDir(context) + "/" + manifest.CacheSlug(manifest.revision);
	for (auto &file : manifest.files) {
		if (!fs.FileExists(base_dir + "/" + file.path)) {
			throw InvalidInputException("tabfm_load: model '%s' is not downloaded. Run: CALL tabfm_download('%s');",
			                            task, task);
		}
	}

	auto result = make_uniq<LifecycleBindData>();
	result->task = task;
	// tabfm_load verifies the weights are present and ready (the §5 contract).
	// The heavy ORT session is warmed lazily on the first predict and shows up in
	// tabfm_models().loaded / is released by tabfm_unload from that point on.
	result->status = "loaded";
	SetLifecycleColumns(return_types, names);
	return std::move(result);
}

unique_ptr<FunctionData> UnloadBind(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_unload");
	auto result = make_uniq<LifecycleBindData>();
	auto tabfm_state = TabFMState::Get(context);
	if (input.inputs.empty()) {
		result->task = "all";
		tabfm_state->UnloadAll(); // free every warmed session for this DB instance
	} else {
		result->task = RequireTaskArgument(input, "tabfm_unload");
		auto manifest = FindManifest(context, "tabfm_unload", ModelArg(input), result->task); // validates the task name
		// free the warmed session, if any (predict / tabfm_load registered it).
		tabfm_state->Unload(TabFMModelCacheKey(manifest.model, result->task, manifest.revision));
	}
	result->status = "unloaded";
	SetLifecycleColumns(return_types, names);
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> LifecycleInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<LifecycleGlobalState>();
}

void LifecycleExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->Cast<LifecycleBindData>();
	auto &state = data.global_state->Cast<LifecycleGlobalState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;
	output.SetValue(0, 0, Value(bind.task));
	output.SetValue(1, 0, Value(bind.status));
	output.SetCardinality(1);
}

//===--------------------------------------------------------------------===//
// tabfm_remove(task [, revision])
//===--------------------------------------------------------------------===//

struct RemoveBindData : public TableFunctionData {
	string task;
	string revision;
	string base_dir;
	vector<string> files; // absolute cache paths from the manifest
};

struct RemoveGlobalState : public GlobalTableFunctionState {
	vector<string> removed;
	idx_t next = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

unique_ptr<FunctionData> RemoveBind(ClientContext &context, TableFunctionBindInput &input,
                                    vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_remove");
	auto task = RequireTaskArgument(input, "tabfm_remove");
	auto manifest = FindManifest(context, "tabfm_remove", ModelArg(input), task);

	auto result = make_uniq<RemoveBindData>();
	result->task = task;
	result->revision = manifest.revision;
	if (input.inputs.size() > 1) {
		if (input.inputs[1].IsNull()) {
			throw InvalidInputException("tabfm_remove: revision cannot be NULL");
		}
		result->revision = StringValue::Get(input.inputs[1]);
	}
	result->base_dir = GetCacheDir(context) + "/" + manifest.CacheSlug(result->revision);
	for (auto &file : manifest.files) {
		result->files.push_back(result->base_dir + "/" + file.path);
	}

	names = {"file", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> RemoveInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<RemoveBindData>();
	auto state = make_uniq<RemoveGlobalState>();
	auto &fs = FileSystem::GetFileSystem(context);

	vector<string> parents;
	for (auto &path : bind.files) {
		fs.TryRemoveFile(path + ".part"); // crashed-download remnants go too
		if (fs.FileExists(path)) {
			fs.RemoveFile(path);
			state->removed.push_back(path);
		}
		auto parent = ParentDirectory(path);
		if (std::find(parents.begin(), parents.end(), parent) == parents.end()) {
			parents.push_back(parent);
		}
	}
	if (state->removed.empty()) {
		throw InvalidInputException("tabfm_remove: no cached weights for task '%s' (revision '%s') under '%s'. "
		                            "Nothing to delete.",
		                            bind.task, bind.revision, bind.base_dir);
	}
	// prune now-empty directories, innermost first, then the model dir itself
	for (auto &parent : parents) {
		auto dir = parent;
		while (dir.size() >= bind.base_dir.size() && StringUtil::StartsWith(dir, bind.base_dir)) {
			RemoveDirectoryIfEmpty(fs, dir);
			if (dir == bind.base_dir) {
				break;
			}
			dir = ParentDirectory(dir);
		}
	}
	return std::move(state);
}

void RemoveExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<RemoveGlobalState>();
	idx_t out = 0;
	while (state.next < state.removed.size() && out < STANDARD_VECTOR_SIZE) {
		output.SetValue(0, out, Value(state.removed[state.next++]));
		output.SetValue(1, out, Value("removed"));
		out++;
	}
	output.SetCardinality(out);
}

//===--------------------------------------------------------------------===//
// tabfm_gpu_precompile(task [, rows, features]) — warm the shape-bucket OFF the
// query path. On ROCm this runs (and caches to .mxr) the MIGraphX compile that
// otherwise stalls the first predict for minutes; on CPU/CUDA it just warms the
// ORT session. rows/features default to the smallest bucket.
//===--------------------------------------------------------------------===//

struct PrecompileBindData : public TableFunctionData {
	string task;
	TabFMTask task_enum = TabFMTask::CLASSIFICATION;
	int64_t rows = 1;
	int64_t features = 1;
	PredictContext ctx;
};

unique_ptr<FunctionData> PrecompileBind(ClientContext &context, TableFunctionBindInput &input,
                                        vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_gpu_precompile");
	auto task = RequireTaskArgument(input, "tabfm_gpu_precompile");
	FindManifest(context, "tabfm_gpu_precompile", ModelArg(input), task); // validates the task name
	auto result = make_uniq<PrecompileBindData>();
	result->task = task;
	result->task_enum = task == "regression" ? TabFMTask::REGRESSION : TabFMTask::CLASSIFICATION;
	if (input.inputs.size() >= 3) {
		result->rows = input.inputs[1].GetValue<int64_t>();
		result->features = input.inputs[2].GetValue<int64_t>();
	}
	if (result->rows < 1 || result->features < 1) {
		throw InvalidInputException("tabfm_gpu_precompile: rows and features must be >= 1");
	}
	auto &ctx = result->ctx;
	ctx.db = &DatabaseInstance::GetDatabase(context);
	ctx.cache_dir = GetCacheDir(context);
	Value v;
	if (context.TryGetCurrentSetting("anofox_tabfm_threads", v) && !v.IsNull()) {
		ctx.threads = v.GetValue<int64_t>();
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_device", v) && !v.IsNull()) {
		ctx.device = v.ToString();
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_gpu_precision", v) && !v.IsNull()) {
		ctx.gpu_precision = StringUtil::Lower(v.ToString());
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_mxr_source", v) && !v.IsNull()) {
		ctx.mxr_source = v.ToString();
	}
	names = {"task", "rows", "features", "device", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::VARCHAR,
	                LogicalType::VARCHAR};
	return std::move(result);
}

void PrecompileExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<LifecycleGlobalState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	state.done = true;
	auto &bind = data.bind_data->Cast<PrecompileBindData>();
	TabFMGpuPrecompile(bind.ctx, bind.task_enum, bind.rows, bind.features); // the (slow) compile happens here
	output.SetValue(0, 0, Value(bind.task));
	output.SetValue(1, 0, Value::BIGINT(bind.rows));
	output.SetValue(2, 0, Value::BIGINT(bind.features));
	output.SetValue(3, 0, Value(bind.ctx.device));
	output.SetValue(4, 0, Value("compiled"));
	output.SetCardinality(1);
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
// tabfm_register_model / tabfm_unregister_model — pure-SQL model registration
// (no JSON manifest file). The spec is held in TabFMState for the db-instance
// lifetime and merged into the registry alongside the built-ins.
//===--------------------------------------------------------------------===//

string NamedStr(TableFunctionBindInput &input, const char *key, const string &def = "") {
	auto it = input.named_parameters.find(key);
	return (it == input.named_parameters.end() || it->second.IsNull()) ? def : it->second.ToString();
}
int64_t NamedInt(TableFunctionBindInput &input, const char *key, int64_t def) {
	auto it = input.named_parameters.find(key);
	return (it == input.named_parameters.end() || it->second.IsNull()) ? def : it->second.GetValue<int64_t>();
}

struct RegisterModelBindData : public TableFunctionData {
	string model_id;
	string capabilities;
};

unique_ptr<FunctionData> RegisterModelBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_register_model");
	auto id = NamedStr(input, "id");
	if (id.empty()) {
		throw InvalidInputException("tabfm_register_model: 'id' is required, e.g. "
		                            "CALL tabfm_register_model(id := 'my_model', classification_graph := '<path|url>');");
	}
	auto clf_graph = NamedStr(input, "classification_graph");
	auto reg_graph = NamedStr(input, "regression_graph");
	// Model-provided GPU graphs (docs/GPU_HARDENING_PLAN.md P3): ext for the
	// CUDA plugin, migraphx for the ROCm plugin; no cross-fallback (see
	// ModelTaskArtifacts). External-data files must sit beside each graph.
	auto clf_ext = NamedStr(input, "classification_ext_graph");
	auto reg_ext = NamedStr(input, "regression_ext_graph");
	auto clf_mgx = NamedStr(input, "classification_migraphx_graph");
	auto reg_mgx = NamedStr(input, "regression_migraphx_graph");

	ModelSpec spec;
	spec.schema_version = 2;
	spec.id = id;
	spec.display_name = NamedStr(input, "display_name", id);
	spec.family = NamedStr(input, "family", "icl-transformer");
	spec.preprocessing_profile = NamedStr(input, "preprocessing_profile", "tabfm_v1_minimal");
	spec.license.id = NamedStr(input, "license", "none");
	spec.license.gate_setting = NamedStr(input, "gate_setting", "");
	auto comm = input.named_parameters.find("commercial");
	spec.license.commercial = (comm != input.named_parameters.end() && !comm->second.IsNull())
	                              ? BooleanValue::Get(comm->second.DefaultCastAs(LogicalType::BOOLEAN))
	                              : spec.license.gate_setting.empty(); // gated ⇒ treated as restricted
	spec.license.redistributable = spec.license.commercial;
	spec.size_regime.max_rows = NamedInt(input, "max_rows", -1);
	spec.size_regime.max_features = NamedInt(input, "max_features", -1);
	spec.size_regime.max_classes = NamedInt(input, "max_classes", -1);
	// A non-empty source_dir marks this as a disk-resolved user model (not a
	// built-in): graph/weights/tensor_map paths resolve against it (default: CWD).
	spec.source_dir = NamedStr(input, "base_dir", ".");

	auto shared_tmap = NamedStr(input, "tensor_map");
	auto weights_repo = NamedStr(input, "weights_repo");
	auto weights_revision = NamedStr(input, "weights_revision", "main");
	auto named_list = [&](const char *key) -> vector<Value> {
		auto it = input.named_parameters.find(key);
		if (it == input.named_parameters.end() || it->second.IsNull()) {
			return {};
		}
		return ListValue::GetChildren(it->second);
	};
	// Per-task weight files: a `<task>_files` LIST (with parallel _file_urls /
	// _file_bytes) for multi-file models, else the single `<task>_weights` scalar.
	auto build_files = [&](const char *files_k, const char *urls_k, const char *bytes_k, const string &one_path,
	                       const string &one_url, int64_t one_bytes) -> vector<ManifestFile> {
		vector<ManifestFile> files;
		auto paths = named_list(files_k);
		if (!paths.empty()) {
			auto urls = named_list(urls_k);
			auto byts = named_list(bytes_k);
			for (idx_t i = 0; i < paths.size(); i++) {
				ManifestFile f;
				f.path = paths[i].ToString();
				f.url = (i < urls.size() && !urls[i].IsNull()) ? urls[i].ToString() : "";
				f.bytes = (i < byts.size() && !byts[i].IsNull()) ? NumericCast<idx_t>(byts[i].GetValue<int64_t>()) : 0;
				files.push_back(std::move(f));
			}
		} else if (!one_path.empty()) {
			ManifestFile f;
			f.path = one_path;
			f.url = one_url;
			f.bytes = one_bytes < 0 ? 0 : NumericCast<idx_t>(one_bytes);
			files.push_back(std::move(f));
		}
		return files;
	};
	// Register a task if it has a graph OR weights: a graph-less task is
	// download-only (predict errors clearly, but download/list work).
	auto add_task = [&](TabFMTask task, const string &graph, vector<ManifestFile> files, const string &tmap,
	                    const string &ext_graph, const string &migraphx_graph) {
		if (graph.empty() && files.empty()) {
			return;
		}
		if (files.empty()) {
			ManifestFile f;
			f.path = "model.safetensors";
			files.push_back(std::move(f));
		}
		ModelTaskArtifacts art;
		art.graph = graph;
		art.ext_graph = ext_graph;
		art.migraphx_graph = migraphx_graph;
		art.preprocessing_profile = spec.preprocessing_profile;
		art.repo = weights_repo;
		art.revision = weights_revision;
		if (!tmap.empty()) {
			art.tensor_map_path = tmap;
		}
		art.files = std::move(files);
		spec.tasks.emplace(task, std::move(art));
		spec.capabilities.push_back(ModelSpec::TaskCapability(task));
	};
	add_task(TabFMTask::CLASSIFICATION, clf_graph,
	         build_files("classification_files", "classification_file_urls", "classification_file_bytes",
	                     NamedStr(input, "classification_weights"), NamedStr(input, "classification_weights_url"),
	                     NamedInt(input, "classification_weights_bytes", -1)),
	         NamedStr(input, "classification_tensor_map", shared_tmap), clf_ext, clf_mgx);
	add_task(TabFMTask::REGRESSION, reg_graph,
	         build_files("regression_files", "regression_file_urls", "regression_file_bytes",
	                     NamedStr(input, "regression_weights"), NamedStr(input, "regression_weights_url"),
	                     NamedInt(input, "regression_weights_bytes", -1)),
	         NamedStr(input, "regression_tensor_map", shared_tmap), reg_ext, reg_mgx);
	if (spec.tasks.empty()) {
		throw InvalidInputException("tabfm_register_model: provide a graph and/or weights for at least one task "
		                            "(classification_graph / regression_graph / classification_weights / ...).");
	}

	// Optional tensor contract: comma-separated graph I/O names to verify at load.
	auto contract = [](const string &csv, vector<TensorContractEntry> &out) {
		for (auto &name : StringUtil::Split(csv, ',')) {
			string n = name;
			while (!n.empty() && std::isspace((unsigned char)n.front())) {
				n.erase(n.begin());
			}
			while (!n.empty() && std::isspace((unsigned char)n.back())) {
				n.pop_back();
			}
			if (!n.empty()) {
				TensorContractEntry e;
				e.logical = n;
				e.name = n;
				out.push_back(std::move(e));
			}
		}
	};
	contract(NamedStr(input, "contract_inputs"), spec.tensor_contract.inputs);
	contract(NamedStr(input, "contract_outputs"), spec.tensor_contract.outputs);

	TabFMState::Get(context)->RegisterModelSpec(spec);

	auto result = make_uniq<RegisterModelBindData>();
	result->model_id = id;
	result->capabilities = StringUtil::Join(spec.capabilities, ", ");
	names = {"model", "capabilities", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	return std::move(result);
}

struct OneRowState : public GlobalTableFunctionState {
	bool done = false;
	idx_t MaxThreads() const override {
		return 1;
	}
};
unique_ptr<GlobalTableFunctionState> OneRowInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<OneRowState>();
}
void RegisterModelExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<OneRowState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	auto &bind = data.bind_data->Cast<RegisterModelBindData>();
	output.SetCardinality(1);
	output.SetValue(0, 0, Value(bind.model_id));
	output.SetValue(1, 0, Value(bind.capabilities));
	output.SetValue(2, 0, Value("registered"));
	state.done = true;
}

struct UnregisterBindData : public TableFunctionData {
	string model_id;
	bool existed = false;
};
unique_ptr<FunctionData> UnregisterModelBind(ClientContext &context, TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types, vector<string> &names) {
	PostHogTelemetry::Instance().RecordFunctionCall("tabfm_unregister_model");
	if (input.inputs.empty() || input.inputs[0].IsNull()) {
		throw InvalidInputException("tabfm_unregister_model: model id cannot be NULL");
	}
	auto result = make_uniq<UnregisterBindData>();
	result->model_id = StringValue::Get(input.inputs[0]);
	result->existed = TabFMState::Get(context)->UnregisterModelSpec(result->model_id);
	names = {"model", "status"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	return std::move(result);
}
void UnregisterModelExecute(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<OneRowState>();
	if (state.done) {
		output.SetCardinality(0);
		return;
	}
	auto &bind = data.bind_data->Cast<UnregisterBindData>();
	output.SetCardinality(1);
	output.SetValue(0, 0, Value(bind.model_id));
	output.SetValue(1, 0, Value(bind.existed ? "unregistered" : "not registered"));
	state.done = true;
}

void RegisterSet(ExtensionLoader &loader, const string &full_name, const string &alias,
                 const vector<vector<LogicalType>> &overloads, table_function_bind_t bind,
                 table_function_init_global_t init, table_function_t execute, const char *description = nullptr,
                 const char *example = nullptr, bool with_model = false) {
	TableFunctionSet set(full_name);
	for (auto &arguments : overloads) {
		TableFunction func(full_name, arguments, execute, bind, init);
		if (with_model) {
			// Optional `model :=` to target a specific model (else the default).
			func.named_parameters["model"] = LogicalType::VARCHAR;
		}
		set.AddFunction(std::move(func));
	}
	// Surface the function in duckdb_functions() with a description + example.
	vector<FunctionDescription> descriptions;
	if (description) {
		FunctionDescription fd;
		fd.description = description;
		if (example) {
			fd.examples = {example};
		}
		descriptions.push_back(std::move(fd));
	}
	RegisterTableFunctionSetWithAlias(loader, std::move(set), alias, std::move(descriptions));
}

} // anonymous namespace

void RegisterWeightsFunctions(ExtensionLoader &loader) {
	// CALL tabfm_download(task [, revision]) — one result row per file.
	RegisterSet(loader, "anofox_tabfm_download", "tabfm_download",
	            {{LogicalType::VARCHAR}, {LogicalType::VARCHAR, LogicalType::VARCHAR}}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, DownloadBind), DownloadInit,
	            DATAZOO_GUARD(ANOFOX_TABFM_BANNER, DownloadExecute),
	            "Download the TabFM model weights for a task ('classification' or 'regression') from Hugging Face "
	            "into the local cache. Requires SET anofox_tabfm_accept_hf_license = true. Returns one row per file "
	            "(file, url, bytes, status).",
	            "CALL tabfm_download('classification');", true);
	// CALL tabfm_download_runtime(backend) — 'cuda' today; 'rocm' errors naming
	// the (not yet published) build-your-own path.
	RegisterSet(loader, "anofox_tabfm_download_runtime", "tabfm_download_runtime", {{LogicalType::VARCHAR}},
	            DATAZOO_GUARD(ANOFOX_TABFM_BANNER, DownloadRuntimeBind), DownloadRuntimeInit,
	            DATAZOO_GUARD(ANOFOX_TABFM_BANNER, DownloadRuntimeExecute),
	            "Download an ONNX Runtime execution-provider plugin library ('cuda') into the directory named by SET "
	            "anofox_tabfm_ep_path (default: the cache dir's 'runtime' subdirectory), so that device can be "
	            "driven without a matching compile-time flavor build. Returns one row per extracted file (file, "
	            "bytes, status).",
	            "CALL tabfm_download_runtime('cuda');");
	// SELECT * FROM tabfm_models();
	RegisterSet(loader, "anofox_tabfm_models", "tabfm_models", {{}}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, ModelsBind), ModelsInit, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, ModelsExecute),
	            "List the TabFM models known to the local cache (model, task, revision, path, bytes, loaded, "
	            "license).",
	            "SELECT * FROM tabfm_models();");
	// CALL tabfm_load(task);
	RegisterSet(loader, "anofox_tabfm_load", "tabfm_load", {{LogicalType::VARCHAR}}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, LoadBind), LifecycleInit,
	            LifecycleExecute,
	            "Eagerly load a downloaded TabFM model for a task into memory so the first predict is warm "
	            "(otherwise the model loads lazily on first use).",
	            "CALL tabfm_load('classification');", true);
	// CALL tabfm_unload([task]);
	RegisterSet(loader, "anofox_tabfm_unload", "tabfm_unload", {{}, {LogicalType::VARCHAR}}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, UnloadBind),
	            LifecycleInit, LifecycleExecute,
	            "Unload a loaded TabFM model from memory (all models if no task is given), freeing its RAM/VRAM.",
	            "CALL tabfm_unload('classification');", true);
	// CALL tabfm_remove(task [, revision]);
	RegisterSet(loader, "anofox_tabfm_remove", "tabfm_remove",
	            {{LogicalType::VARCHAR}, {LogicalType::VARCHAR, LogicalType::VARCHAR}}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, RemoveBind), RemoveInit,
	            DATAZOO_GUARD(ANOFOX_TABFM_BANNER, RemoveExecute),
	            "Delete a downloaded TabFM model's weights from the local cache (by task, optionally a specific "
	            "revision).",
	            "CALL tabfm_remove('classification');", true);
	// SELECT * FROM tabfm_list_models();  — the registry (all known models).
	RegisterSet(loader, "anofox_tabfm_list_models", "tabfm_list_models", {{}}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, ListModelsBind), ListModelsInit,
	            DATAZOO_GUARD(ANOFOX_TABFM_BANNER, ListModelsExecute),
	            "List every model in the registry (built-ins + user manifests), downloaded or not: model, family, "
	            "capabilities, license, commercial, size regime (max_rows/features/classes), downloaded.",
	            "SELECT * FROM tabfm_list_models();");
	// CALL tabfm_gpu_precompile(task [, rows, features]);
	RegisterSet(loader, "anofox_tabfm_gpu_precompile", "tabfm_gpu_precompile",
	            {{LogicalType::VARCHAR}, {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT}},
	            PrecompileBind, LifecycleInit, PrecompileExecute,
	            "Warm the GPU path for a task by compiling the model for a shape bucket ahead of the first predict "
	            "(on ROCm this builds and caches the .mxr program; a no-op cost on CPU/CUDA). Returns task, rows, "
	            "features, device, status.",
	            "CALL tabfm_gpu_precompile('classification', 1000, 50);", true);

	// CALL tabfm_register_model(id := '...', classification_graph := '...', ...);
	{
		TableFunction f("anofox_tabfm_register_model", {}, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, RegisterModelExecute), DATAZOO_GUARD(ANOFOX_TABFM_BANNER, RegisterModelBind), OneRowInit);
		for (auto *k : {"id", "display_name", "family", "classification_graph", "regression_graph",
		                "classification_ext_graph", "regression_ext_graph", "classification_migraphx_graph",
		                "regression_migraphx_graph",
		                "classification_weights", "regression_weights", "classification_weights_url",
		                "regression_weights_url", "weights_revision", "tensor_map", "classification_tensor_map",
		                "regression_tensor_map", "weights_repo", "license", "gate_setting", "preprocessing_profile",
		                "base_dir", "contract_inputs", "contract_outputs"}) {
			f.named_parameters[k] = LogicalType::VARCHAR;
		}
		f.named_parameters["commercial"] = LogicalType::BOOLEAN;
		for (auto *k : {"max_rows", "max_features", "max_classes", "classification_weights_bytes",
		                "regression_weights_bytes"}) {
			f.named_parameters[k] = LogicalType::BIGINT;
		}
		// Multi-file-per-task (e.g. weights + config): parallel LISTs.
		for (auto *k : {"classification_files", "regression_files", "classification_file_urls",
		                "regression_file_urls"}) {
			f.named_parameters[k] = LogicalType::LIST(LogicalType::VARCHAR);
		}
		for (auto *k : {"classification_file_bytes", "regression_file_bytes"}) {
			f.named_parameters[k] = LogicalType::LIST(LogicalType::BIGINT);
		}
		TableFunctionSet set("anofox_tabfm_register_model");
		set.AddFunction(std::move(f));
		vector<FunctionDescription> d;
		FunctionDescription fd;
		fd.description =
		    "Register a model in SQL (no manifest file). Named args: id, classification_graph / regression_graph "
		    "(path or url to the weight-free ONNX graph), classification_weights / regression_weights, tensor_map "
		    "(or classification_tensor_map / regression_tensor_map), weights_repo, license, commercial, "
		    "gate_setting, preprocessing_profile, max_rows / max_features / max_classes. Then use model := '<id>'.";
		fd.examples = {"CALL tabfm_register_model(id := 'my', classification_graph := '/p/g.onnx', "
		               "classification_weights := '/p/w.safetensors', tensor_map := '/p/map.json', "
		               "license := 'apache-2.0');"};
		d.push_back(std::move(fd));
		RegisterTableFunctionSetWithAlias(loader, std::move(set), "tabfm_register_model", std::move(d));
	}
	// CALL tabfm_unregister_model('id');
	RegisterSet(loader, "anofox_tabfm_unregister_model", "tabfm_unregister_model", {{LogicalType::VARCHAR}},
	            DATAZOO_GUARD(ANOFOX_TABFM_BANNER, UnregisterModelBind), OneRowInit, DATAZOO_GUARD(ANOFOX_TABFM_BANNER, UnregisterModelExecute),
	            "Remove a model registered with tabfm_register_model. Returns model, status.",
	            "CALL tabfm_unregister_model('my_model');");
}

} // namespace anofox
} // namespace duckdb
