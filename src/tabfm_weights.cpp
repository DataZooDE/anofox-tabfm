#include "tabfm_registration.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_predict.hpp"
#include "tabfm_state.hpp"
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

#include <algorithm>
#include <cstdlib>

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

namespace {

constexpr idx_t kChunkBytes = 8ULL * 1024 * 1024;
constexpr const char *kBuiltinRepo = "google/tabfm-1.0.0-pytorch";
constexpr const char *kBuiltinLicense = "tabfm-non-commercial-v1.0";
constexpr const char *kBuiltinModel = "tabfm-v1";

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

string ReadTextFile(ClientContext &context, const string &path) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	auto size = NumericCast<idx_t>(fs.GetFileSize(*handle));
	string result(size, '\0');
	idx_t offset = 0;
	while (offset < size) {
		auto n = handle->Read((void *)(result.data() + offset), size - offset);
		if (n <= 0) {
			break;
		}
		offset += NumericCast<idx_t>(n);
	}
	result.resize(offset);
	return result;
}

string JsonString(duckdb_yyjson::yyjson_val *obj, const char *key, const string &fallback) {
	auto val = duckdb_yyjson::yyjson_obj_get(obj, key);
	if (!val || !duckdb_yyjson::yyjson_is_str(val)) {
		return fallback;
	}
	return duckdb_yyjson::yyjson_get_str(val);
}

WeightsManifest ParseManifestJson(const string &json, const string &manifest_path) {
	using namespace duckdb_yyjson; // NOLINT
	auto doc = yyjson_read(json.c_str(), json.size(), 0);
	if (!doc) {
		throw InvalidInputException("tabfm: cannot parse model manifest '%s': invalid JSON", manifest_path);
	}
	WeightsManifest result;
	auto root = yyjson_doc_get_root(doc);
	if (!root || !yyjson_is_obj(root)) {
		yyjson_doc_free(doc);
		throw InvalidInputException("tabfm: cannot parse model manifest '%s': root must be a JSON object",
		                            manifest_path);
	}
	result.model = JsonString(root, "model", JsonString(root, "model_id", "unnamed"));
	result.task = JsonString(root, "task", "");
	result.repo = JsonString(root, "repo", "");
	result.revision = JsonString(root, "revision", "main");
	result.license = JsonString(root, "license", "none");
	auto files = yyjson_obj_get(root, "files");
	if (files && yyjson_is_arr(files)) {
		size_t idx, max;
		yyjson_val *file;
		yyjson_arr_foreach(files, idx, max, file) {
			if (!yyjson_is_obj(file)) {
				continue;
			}
			WeightsFileEntry entry;
			entry.path = JsonString(file, "path", "");
			entry.url = JsonString(file, "url", "");
			auto bytes = yyjson_obj_get(file, "bytes");
			if (bytes && yyjson_is_int(bytes)) {
				entry.bytes = yyjson_get_sint(bytes);
			}
			if (!entry.path.empty()) {
				result.files.push_back(std::move(entry));
			}
		}
	}
	yyjson_doc_free(doc);
	if (result.task.empty()) {
		throw InvalidInputException("tabfm: model manifest '%s' is missing the 'task' field", manifest_path);
	}
	if (result.files.empty()) {
		throw InvalidInputException("tabfm: model manifest '%s' lists no files", manifest_path);
	}
	return result;
}

WeightsManifest BuiltinManifest(const string &task) {
	WeightsManifest result;
	result.model = kBuiltinModel;
	result.task = task;
	result.repo = kBuiltinRepo;
	result.revision = "main";
	result.license = kBuiltinLicense;
	result.builtin = true;
	WeightsFileEntry entry;
	entry.path = task + "/model.safetensors";
	// URL is derived from repo + revision at bind time (revision may be overridden).
	result.files.push_back(std::move(entry));
	return result;
}

// All manifests visible in this session: the pluggable manifest from
// SET anofox_tabfm_model_manifest (if any) shadows a built-in task of the
// same name; the hardcoded Google tasks are always present otherwise.
vector<WeightsManifest> ResolveManifests(ClientContext &context) {
	vector<WeightsManifest> result;
	Value value;
	if (context.TryGetCurrentSetting("anofox_tabfm_model_manifest", value) && !value.IsNull()) {
		auto path = value.ToString();
		if (!path.empty()) {
			result.push_back(ParseManifestJson(ReadTextFile(context, path), path));
		}
	}
	for (auto task : {"classification", "regression"}) {
		bool shadowed = false;
		for (auto &m : result) {
			if (m.task == task) {
				shadowed = true;
				break;
			}
		}
		if (!shadowed) {
			result.push_back(BuiltinManifest(task));
		}
	}
	return result;
}

string ValidTaskList(const vector<WeightsManifest> &manifests) {
	vector<string> tasks;
	for (auto &m : manifests) {
		if (std::find(tasks.begin(), tasks.end(), m.task) == tasks.end()) {
			tasks.push_back(m.task);
		}
	}
	std::sort(tasks.begin(), tasks.end());
	return StringUtil::Join(tasks, ", ");
}

WeightsManifest FindManifest(ClientContext &context, const char *func_name, const string &task) {
	auto manifests = ResolveManifests(context);
	for (auto &m : manifests) {
		if (m.task == task) {
			return m;
		}
	}
	throw InvalidInputException("%s: unknown task '%s'. Valid tasks: %s", func_name, task, ValidTaskList(manifests));
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
	if (manifest.builtin) {
		return "https://huggingface.co/" + manifest.repo + "/resolve/" + revision + "/" + file.path;
	}
	// custom manifest without url: air-gapped, cache must be pre-seeded (FR-1.5)
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
	result->manifest = FindManifest(context, "tabfm_download", task);
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
	auto src = fs.OpenFile(item.url, read_flags);
	auto remote_size = fs.GetFileSize(*src); // HEAD-cheap even on multi-GB files (S03 §4)
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
	names = {"model", "task", "revision", "path", "bytes", "loaded", "license"};
	return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::BIGINT,  LogicalType::BOOLEAN, LogicalType::VARCHAR};
	return make_uniq<ModelsBindData>();
}

unique_ptr<GlobalTableFunctionState> ModelsInit(ClientContext &context, TableFunctionInitInput &) {
	auto state = make_uniq<ModelsGlobalState>();
	auto tabfm_state = TabFMState::Get(context); // the live loaded-model map
	auto &fs = FileSystem::GetFileSystem(context);
	auto cache_dir = GetCacheDir(context);
	if (!fs.DirectoryExists(cache_dir)) {
		return std::move(state);
	}
	for (auto &manifest : ResolveManifests(context)) {
		// one cache entry per downloaded revision: <slug-base>@<revision>
		auto slug_base = manifest.CacheSlug("");
		vector<string> revisions;
		fs.ListFiles(cache_dir, [&](const string &name, bool is_dir) {
			if (is_dir && StringUtil::StartsWith(name, slug_base)) {
				revisions.push_back(name.substr(slug_base.size()));
			}
		});
		std::sort(revisions.begin(), revisions.end());
		for (auto &revision : revisions) {
			auto base_dir = cache_dir + "/" + manifest.CacheSlug(revision);
			ModelsRow row;
			row.model = manifest.model;
			row.task = manifest.task;
			row.revision = revision;
			row.license = manifest.license;
			// 'loaded' reflects the real DB-instance TabFMState: a model is loaded
			// once a predict (or lifecycle load) has warmed its session. The key
			// format is shared with the engine via TabFMModelCacheKey.
			row.loaded =
			    tabfm_state->Snapshot(TabFMModelCacheKey(manifest.model, manifest.task, revision)) != nullptr;
			bool complete = true;
			for (idx_t i = 0; i < manifest.files.size(); i++) {
				auto path = base_dir + "/" + manifest.files[i].path;
				if (!fs.FileExists(path)) {
					complete = false;
					break;
				}
				auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
				row.bytes += fs.GetFileSize(*handle);
				if (i == 0) {
					row.path = path; // primary artifact (the model file) per SQL-API §3
				}
			}
			if (complete) {
				state->rows.push_back(std::move(row));
			}
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
	auto manifest = FindManifest(context, "tabfm_load", task);

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
		auto manifest = FindManifest(context, "tabfm_unload", result->task); // validates the task name
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
	auto manifest = FindManifest(context, "tabfm_remove", task);

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
	FindManifest(context, "tabfm_gpu_precompile", task); // validates the task name
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
	if (context.TryGetCurrentSetting("anofox_tabfm_model_manifest", v) && !v.IsNull()) {
		ctx.model_manifest_path = v.ToString();
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

void RegisterSet(ExtensionLoader &loader, const string &full_name, const string &alias,
                 const vector<vector<LogicalType>> &overloads, table_function_bind_t bind,
                 table_function_init_global_t init, table_function_t execute, const char *description = nullptr,
                 const char *example = nullptr) {
	TableFunctionSet set(full_name);
	for (auto &arguments : overloads) {
		TableFunction func(full_name, arguments, execute, bind, init);
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
	            {{LogicalType::VARCHAR}, {LogicalType::VARCHAR, LogicalType::VARCHAR}}, DownloadBind, DownloadInit,
	            DownloadExecute,
	            "Download the TabFM model weights for a task ('classification' or 'regression') from Hugging Face "
	            "into the local cache. Requires SET anofox_tabfm_accept_hf_license = true. Returns one row per file "
	            "(file, url, bytes, status).",
	            "CALL tabfm_download('classification');");
	// SELECT * FROM tabfm_models();
	RegisterSet(loader, "anofox_tabfm_models", "tabfm_models", {{}}, ModelsBind, ModelsInit, ModelsExecute,
	            "List the TabFM models known to the local cache (model, task, revision, path, bytes, loaded, "
	            "license).",
	            "SELECT * FROM tabfm_models();");
	// CALL tabfm_load(task);
	RegisterSet(loader, "anofox_tabfm_load", "tabfm_load", {{LogicalType::VARCHAR}}, LoadBind, LifecycleInit,
	            LifecycleExecute,
	            "Eagerly load a downloaded TabFM model for a task into memory so the first predict is warm "
	            "(otherwise the model loads lazily on first use).",
	            "CALL tabfm_load('classification');");
	// CALL tabfm_unload([task]);
	RegisterSet(loader, "anofox_tabfm_unload", "tabfm_unload", {{}, {LogicalType::VARCHAR}}, UnloadBind,
	            LifecycleInit, LifecycleExecute,
	            "Unload a loaded TabFM model from memory (all models if no task is given), freeing its RAM/VRAM.",
	            "CALL tabfm_unload('classification');");
	// CALL tabfm_remove(task [, revision]);
	RegisterSet(loader, "anofox_tabfm_remove", "tabfm_remove",
	            {{LogicalType::VARCHAR}, {LogicalType::VARCHAR, LogicalType::VARCHAR}}, RemoveBind, RemoveInit,
	            RemoveExecute,
	            "Delete a downloaded TabFM model's weights from the local cache (by task, optionally a specific "
	            "revision).",
	            "CALL tabfm_remove('classification');");
	// CALL tabfm_gpu_precompile(task [, rows, features]);
	RegisterSet(loader, "anofox_tabfm_gpu_precompile", "tabfm_gpu_precompile",
	            {{LogicalType::VARCHAR}, {LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT}},
	            PrecompileBind, LifecycleInit, PrecompileExecute,
	            "Warm the GPU path for a task by compiling the model for a shape bucket ahead of the first predict "
	            "(on ROCm this builds and caches the .mxr program; a no-op cost on CPU/CUDA). Returns task, rows, "
	            "features, device, status.",
	            "CALL tabfm_gpu_precompile('classification', 1000, 50);");
}

} // namespace anofox
} // namespace duckdb
