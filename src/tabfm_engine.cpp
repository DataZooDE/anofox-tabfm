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
#include "tabfm_safetensors.hpp"
#include "tabfm_ort_engine.hpp"
#include "tabfm_state.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/database.hpp"

#include "yyjson.hpp"

#include <cmath>

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
	fs.Read(*handle, (void *)buffer.data(), NumericCast<int64_t>(size));
	return buffer;
}

//! The resolved on-disk artifacts for one (model, task).
struct ResolvedModel {
	ModelManifest manifest;
	string manifest_dir;
	string graph_path;
	string weights_path;
	unordered_map<string, string> tensor_map; // onnx initializer name -> st key
	string cache_key;
};

// Load {onnx -> safetensors} from the manifest: inline map, or the
// "initializers" object of the tensor-map JSON file, else identity.
unordered_map<string, string> LoadTensorMap(FileSystem &fs, const ModelManifest &manifest, const string &dir) {
	if (!manifest.tensor_map.empty()) {
		return manifest.tensor_map;
	}
	unordered_map<string, string> result;
	if (manifest.tensor_map_path.empty()) {
		return result; // identity mapping (handled at injection)
	}
	auto path = JoinPath(fs, dir, manifest.tensor_map_path);
	auto json = ReadWholeFile(fs, path);
	using namespace duckdb_yyjson; // NOLINT
	auto doc = yyjson_read(json.c_str(), json.size(), 0);
	if (!doc) {
		throw InvalidInputException("tabfm: cannot parse tensor map '%s'", path);
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
	if (!cache_dir.empty() && !manifest.repo.empty()) {
		auto slug = StringUtil::Replace(manifest.repo, "/", "__") + "@" + manifest.revision;
		candidates.push_back(fs.JoinPath(fs.JoinPath(cache_dir, slug), file.path));
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

ResolvedModel ResolveModel(FileSystem &fs, const PredictContext &ctx, TabFMTask task) {
	ResolvedModel resolved;
	const auto task_name = TabFMTaskName(task);
	if (!ctx.model_manifest_path.empty()) {
		resolved.manifest = LoadModelManifestFile(ctx.model_manifest_path);
		resolved.manifest_dir = DirName(ctx.model_manifest_path);
		if (resolved.manifest.task != task) {
			throw InvalidInputException(
			    "tabfm: the configured model manifest is for task '%s', but a '%s' prediction was requested. "
			    "Set anofox_tabfm_model_manifest to a %s model or use the matching tabfm_%s function.",
			    TabFMTaskName(resolved.manifest.task), task_name, task_name,
			    task == TabFMTask::CLASSIFICATION ? "classify" : "regress");
		}
	} else {
		resolved.manifest = BuiltinTabFMManifest(task);
		// built-in models live in the weights cache; graph beside the extension
		// (resources/) — resolved relative to the cache slug dir at load.
		resolved.manifest_dir = ctx.cache_dir;
	}
	// weights first: "not downloaded" is the common, actionable error (§5).
	resolved.weights_path =
	    ResolveWeightsPath(fs, resolved.manifest, resolved.manifest_dir, ctx.cache_dir, task_name);
	resolved.graph_path = ResolveGraphPath(fs, resolved.manifest, resolved.manifest_dir);
	resolved.tensor_map = LoadTensorMap(fs, resolved.manifest, resolved.manifest_dir);
	resolved.cache_key = resolved.manifest.model + ":" + task_name + "@" + resolved.manifest.revision;
	return resolved;
}

//===--------------------------------------------------------------------===//
// Session loading (cached in TabFMState)
//===--------------------------------------------------------------------===//

TabFMTensorDtype ToEngineDtype(SafetensorsDtype dtype) {
	return dtype == SafetensorsDtype::I64 ? TabFMTensorDtype::I64 : TabFMTensorDtype::F32;
}

// Build (or reuse) the ORT session for `resolved` and return a snapshot of the
// LoadedModel. The safetensors buffer + arena live only for the duration of
// CreateSession (S02: ORT copies), so they are scoped tightly here.
shared_ptr<LoadedModel> LoadOrGetSession(FileSystem &fs, TabFMState &state, const ResolvedModel &resolved,
                                         const PredictContext &ctx) {
	if (auto snapshot = state.Snapshot(resolved.cache_key)) {
		return snapshot;
	}

	// read the weight-free graph and the weights, inject by tensor-map name
	auto graph_bytes = ReadWholeFile(fs, resolved.graph_path);
	auto weights = ReadWholeFile(fs, resolved.weights_path);
	auto view = ParseSafetensors(const_data_ptr_cast(weights.data()), weights.size(), resolved.weights_path);
	auto arena = MaterializeF32Arena(view);

	vector<TabFMTensorRef> initializers;
	initializers.reserve(view.tensors.size());
	for (auto &entry : view.tensors) {
		// tensor map is onnx-initializer-name -> safetensors-key; invert it so
		// each safetensors tensor is injected under its graph name. When the map
		// is empty, inject under the safetensors key (identity).
		const string &st_key = entry.first;
		auto &materialized = arena.Get(st_key);
		TabFMTensorRef ref;
		ref.name = st_key; // default: identity
		ref.dtype = ToEngineDtype(materialized.dtype);
		ref.shape = materialized.shape;
		ref.data = materialized.data;
		ref.size_bytes = materialized.nbytes;
		initializers.push_back(std::move(ref));
	}
	// remap names onnx<-st using the tensor map (initializers currently named by st key)
	if (!resolved.tensor_map.empty()) {
		// invert: st_key -> onnx_name
		unordered_map<string, string> st_to_onnx;
		for (auto &kv : resolved.tensor_map) {
			st_to_onnx[kv.second] = kv.first;
		}
		for (auto &ref : initializers) {
			auto it = st_to_onnx.find(ref.name);
			if (it != st_to_onnx.end()) {
				ref.name = it->second;
			}
		}
	}

	TabFMSessionConfig config;
	config.intra_op_threads = MaxValue<int64_t>(1, ctx.threads);
	auto devices = DiscoverDevices();
	auto device = ResolveDevice(ctx.device, devices);
	config.device_id = device.device_id;
	config.device_ordinal = device.device_ordinal;
	config.model_tag = TabFMTaskName(resolved.manifest.task);

	auto session = CreateSession(graph_bytes.data(), graph_bytes.size(), initializers, config);
	// arena + weights + graph_bytes drop here (ORT owns copies, S02).

	auto model = make_shared_ptr<LoadedModel>();
	model->model_key = resolved.cache_key;
	model->session = shared_ptr<void>(std::move(session)); // implicit up-cast to void
	model->device_id = config.device_id;
	model->dtype = "f32";
	model->bytes = weights.size();
	state.Register(resolved.cache_key, model);
	return model;
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

		// 1. preprocess
		vector<PreprocessColumnSpec> columns;
		auto collection = BuildCollection(in.rows, in.row_type, columns, in.target_idx);
		auto pp_task = task == TabFMTask::CLASSIFICATION ? PreprocessTask::CLASSIFICATION : PreprocessTask::REGRESSION;
		auto batch = PreprocessBatch(collection, columns, pp_task);

		if (task == TabFMTask::CLASSIFICATION && batch.label_decoder.size() > 10) {
			throw InvalidInputException(
			    "target '%s' has %llu distinct labels; TabFM v1 supports at most 10. Consider grouping rare labels.",
			    in.target_name, static_cast<unsigned long long>(batch.label_decoder.size()));
		}

		// 2. resolve + load the session (serialized per device)
		auto resolved = ResolveModel(*fs, in.ctx, task);
		auto state = TabFMState::Get(*in.ctx.db);
		shared_ptr<LoadedModel> model;
		TabFMRunOutput out;
		{
			lock_guard<mutex> device_guard(state->DeviceMutex(in.ctx.device));
			model = LoadOrGetSession(*fs, *state, resolved, in.ctx);

			// 3. tensors (float32) + forward pass
			vector<float> x(batch.x.size());
			for (idx_t i = 0; i < batch.x.size(); i++) {
				x[i] = static_cast<float>(batch.x[i]);
			}
			vector<float> y(batch.y.size());
			for (idx_t i = 0; i < batch.y.size(); i++) {
				y[i] = static_cast<float>(batch.y[i]);
			}
			// std::vector<bool> is bit-packed (no .data()); materialize a real
			// bool array for the run input.
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
			auto *session = reinterpret_cast<TabFMSession *>(model->session.get());
			out = Run(*session, run_input);
		}

		// 4. decode logits[1,T,C] -> per-source-row predictions
		return Decode(in, batch, out, task);
	}

private:
	static TabFMPredictResult Decode(const PredictInput &in, const PreprocessedBatch &batch,
	                                 const TabFMRunOutput &out, TabFMTask task) {
		const idx_t T = batch.T;
		const idx_t n_rows = in.rows.size();
		const idx_t C = out.shape.empty() ? 0 : NumericCast<idx_t>(out.shape.back());

		TabFMPredictResult result;
		result.yhat.resize(n_rows);
		result.yhat_score.resize(n_rows);
		if (in.opts.detail && task == TabFMTask::CLASSIFICATION) {
			result.proba.resize(n_rows);
		}

		const idx_t n_classes = batch.label_decoder.size();
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

} // namespace anofox
} // namespace duckdb
