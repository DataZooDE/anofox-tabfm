#include "tabfm_predict.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "telemetry.hpp"

#include <cstdlib>

#include <cmath>

namespace duckdb {
namespace anofox {

// Predict core (WS-E, spec S04 / HLD §4.6 / SQL-API rev 4 §2):
//
//   anofox_tabfm_predict_agg(row STRUCT, target VARCHAR [, opts MAP])
//       → LIST(STRUCT(cols, yhat, yhat_score, is_training[, proba]))
//   anofox_tabfm_predict_win(row STRUCT, target VARCHAR [, opts MAP]) OVER (…)
//       → STRUCT(yhat, yhat_score[, proba])   (custom window callback; the
//         non-window path refuses — see tabfm_predict.hpp for the shape and
//         current-row rationale)
//
// The model behind both is the PredictEngine seam (tabfm_predict.hpp),
// implemented by the real TabFM engine (tabfm_engine.cpp).

namespace {

//===--------------------------------------------------------------------===//
// Bind data
//===--------------------------------------------------------------------===//
struct PredictBindData : public FunctionData {
	//! "tabfm_predict_agg" or "tabfm_predict_win" (error-text prefix)
	string function_name;
	string target;
	idx_t target_idx = 0;
	LogicalType row_type;
	LogicalType target_type;
	TabFMPredictOptions options;
	//! anofox_tabfm_max_rows snapshot (bind-time; enforced incrementally in update)
	idx_t max_rows = 10000;
	//! settings + DB handle captured at bind (finalize has no ClientContext)
	PredictContext context;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<PredictBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<PredictBindData>();
		return function_name == other.function_name && target == other.target &&
		       target_idx == other.target_idx && row_type == other.row_type &&
		       options.task == other.options.task && options.detail == other.options.detail &&
		       options.n_estimators == other.options.n_estimators && options.seed == other.options.seed &&
		       options.context_rows == other.options.context_rows && options.model == other.options.model &&
		       max_rows == other.max_rows;
	}

	bool EmitProba() const {
		return options.detail && options.task == TabFMTask::CLASSIFICATION;
	}
	LogicalType YhatType() const {
		return options.task == TabFMTask::CLASSIFICATION ? target_type : LogicalType::DOUBLE;
	}
	//! window return shape: STRUCT(yhat, yhat_score[, proba])
	LogicalType WindowStructType() const {
		child_list_t<LogicalType> fields;
		fields.emplace_back("yhat", YhatType());
		fields.emplace_back("yhat_score", LogicalType::DOUBLE);
		if (EmitProba()) {
			fields.emplace_back("proba", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE));
		}
		return LogicalType::STRUCT(std::move(fields));
	}
	//! aggregate return shape: LIST(STRUCT(cols, yhat, yhat_score, is_training[, proba]))
	LogicalType ListStructType() const {
		child_list_t<LogicalType> fields;
		fields.emplace_back("cols", row_type); // 'row' is a reserved word (S04)
		fields.emplace_back("yhat", YhatType());
		fields.emplace_back("yhat_score", LogicalType::DOUBLE);
		fields.emplace_back("is_training", LogicalType::BOOLEAN);
		if (EmitProba()) {
			fields.emplace_back("proba", LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE));
		}
		return LogicalType::LIST(LogicalType::STRUCT(std::move(fields)));
	}
};

//===--------------------------------------------------------------------===//
// Options parsing (bind time; opts must be a constant MAP, values VARCHAR)
//===--------------------------------------------------------------------===//
int64_t ParseIntegerOption(const string &fname, const string &key, const string &raw, const char *expectation) {
	try {
		size_t pos = 0;
		auto parsed = std::stoll(raw, &pos);
		if (pos != raw.size()) {
			throw std::invalid_argument(raw);
		}
		return parsed;
	} catch (const std::exception &) {
		throw BinderException("%s: %s %s, got '%s'", fname, key, expectation, raw);
	}
}

void ParseOneOption(const string &fname, PredictBindData &bind, const string &key_raw, const string &val) {
	auto key = StringUtil::Lower(key_raw);
	auto &opts = bind.options;
	if (key == "task") {
		auto task = StringUtil::Lower(val);
		if (task == "classification") {
			opts.task = TabFMTask::CLASSIFICATION;
		} else if (task == "regression") {
			if (!bind.target_type.IsNumeric()) {
				throw BinderException("%s: task 'regression' requires a numeric target; column '%s' is %s", fname,
				                      bind.target, bind.target_type.ToString());
			}
			opts.task = TabFMTask::REGRESSION;
		} else if (task != "auto") {
			throw BinderException("%s: task must be 'auto', 'classification' or 'regression', got '%s'", fname, val);
		}
	} else if (key == "n_estimators") {
		auto n = ParseIntegerOption(fname, "n_estimators", val, "must be an integer between 1 and 32");
		if (n < 1 || n > 32) {
			throw BinderException("%s: n_estimators must be an integer between 1 and 32, got '%s'", fname, val);
		}
		if (n > 1) {
			// parse stays complete; the ensemble of diverse views arrives with
			// the WS-F TabFM engine integration
			throw NotImplementedException(
			    "%s: n_estimators > 1 (ensemble) is not available yet — the ensemble port lands with the TabFM "
			    "engine integration (milestone M3); use n_estimators = '1'",
			    fname);
		}
		opts.n_estimators = NumericCast<idx_t>(n);
	} else if (key == "seed") {
		opts.seed = ParseIntegerOption(fname, "seed", val, "must be an integer");
	} else if (key == "output_mode") {
		auto mode = StringUtil::Lower(val);
		if (mode == "detail") {
			opts.detail = true;
		} else if (mode != "compact") {
			throw BinderException("%s: output_mode must be 'compact' or 'detail', got '%s'", fname, val);
		}
	} else if (key == "context_rows") {
		if (val.empty()) {
			opts.context_rows = 0; // '' = use the full context
		} else {
			auto n = ParseIntegerOption(fname, "context_rows", val, "must be '' or a non-negative integer");
			if (n < 0) {
				throw BinderException("%s: context_rows must be '' or a non-negative integer, got '%s'", fname, val);
			}
			opts.context_rows = NumericCast<idx_t>(n);
		}
	} else if (key == "softmax_temperature") {
		double temp;
		try {
			size_t pos = 0;
			temp = std::stod(val, &pos);
			if (pos != val.size()) {
				throw std::invalid_argument(val);
			}
		} catch (const std::exception &) {
			throw BinderException("%s: softmax_temperature must be a positive number, got '%s'", fname, val);
		}
		if (!(temp > 0) || !std::isfinite(temp)) {
			throw BinderException("%s: softmax_temperature must be a positive number, got '%s'", fname, val);
		}
		opts.softmax_temperature = temp;
	} else if (key == "model") {
		opts.model = val;
	} else {
		throw BinderException("%s: unknown option '%s'; valid options are task, n_estimators, seed, output_mode, "
		                      "context_rows, softmax_temperature, model",
		                      fname, key_raw);
	}
}

void ParseOptionsArgument(ClientContext &context, Expression &expr, PredictBindData &bind, const string &fname) {
	const auto type_id = expr.return_type.id();
	if (type_id == LogicalTypeId::SQLNULL || type_id == LogicalTypeId::UNKNOWN) {
		return; // NULL options → defaults
	}
	if (type_id == LogicalTypeId::LIST || type_id == LogicalTypeId::ARRAY) {
		// Macro overload dispatch is ARITY-ONLY (S04 3b/3c): a VARCHAR[]
		// feature list lands in the opts slot — redirect instead of a
		// confusing type error.
		throw BinderException(
		    "%s: options must be a MAP — for a feature list use tabfm_predict(tbl, target, features, opts)", fname);
	}
	if (type_id != LogicalTypeId::MAP && type_id != LogicalTypeId::STRUCT) {
		throw BinderException(
		    "%s: options must be a MAP — for a feature list use tabfm_predict(tbl, target, features, opts)", fname);
	}
	if (!expr.IsFoldable()) {
		throw BinderException("%s: options MAP must be a constant", fname);
	}
	auto opts_value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (opts_value.IsNull()) {
		return;
	}
	if (type_id == LogicalTypeId::MAP) {
		for (auto &kv : MapValue::GetChildren(opts_value)) {
			auto &entry = StructValue::GetChildren(kv);
			ParseOneOption(fname, bind, entry[0].ToString(), entry[1].IsNull() ? string() : entry[1].ToString());
		}
	} else {
		auto &child_types = StructType::GetChildTypes(opts_value.type());
		auto &children = StructValue::GetChildren(opts_value);
		for (idx_t i = 0; i < children.size(); i++) {
			ParseOneOption(fname, bind, child_types[i].first, children[i].IsNull() ? string() : children[i].ToString());
		}
	}
}

//===--------------------------------------------------------------------===//
// Bind (shared by the aggregate and the window variant)
//===--------------------------------------------------------------------===//
idx_t ReadSettingUBigint(ClientContext &context, const char *name, idx_t fallback) {
	Value value;
	if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) {
		return NumericCast<idx_t>(BigIntValue::Get(value.DefaultCastAs(LogicalType::BIGINT)));
	}
	return fallback;
}

string ExpandUserHome(const string &path) {
	if (path.empty() || path[0] != '~') {
		return path;
	}
	const char *home = std::getenv("HOME");
	if (!home) {
		return path;
	}
	return string(home) + path.substr(1);
}

unique_ptr<FunctionData> PredictBindInternal(ClientContext &context, AggregateFunction &function,
                                             vector<unique_ptr<Expression>> &arguments, const string &fname,
                                             bool is_window) {
	auto bind = make_uniq<PredictBindData>();
	bind->function_name = fname;

	// arg 0: the whole row as a STRUCT (the table alias from query_table)
	if (arguments[0]->return_type.id() != LogicalTypeId::STRUCT) {
		throw BinderException("%s: first argument must be the whole-row STRUCT — pass the table alias, e.g. "
		                      "SELECT %s(t, 'churned') FROM query_table('customers') t",
		                      fname, fname);
	}
	bind->row_type = arguments[0]->return_type;
	auto &fields = StructType::GetChildTypes(bind->row_type);

	// arg 1: constant target column name, resolved case-insensitively
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("%s: target must be a constant column name", fname);
	}
	auto target_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (target_value.IsNull()) {
		throw BinderException("%s: target must be a constant column name, not NULL", fname);
	}
	bind->target = StringValue::Get(target_value.DefaultCastAs(LogicalType::VARCHAR));

	bool found = false;
	for (idx_t i = 0; i < fields.size(); i++) {
		if (StringUtil::CIEquals(fields[i].first, bind->target)) {
			bind->target_idx = i;
			bind->target_type = fields[i].second;
			found = true;
			break;
		}
	}
	if (!found) {
		throw BinderException("%s: target column '%s' not found in the row struct (pass the bare column name, "
		                      "no quotes)",
		                      fname, bind->target);
	}

	// FR-3.2 task inference: non-numeric target (VARCHAR/ENUM/BOOL/…) →
	// classification; numeric → regression. Overridable via opts['task'].
	bind->options.task = bind->target_type.IsNumeric() ? TabFMTask::REGRESSION : TabFMTask::CLASSIFICATION;

	// arg 2: constant options MAP
	if (arguments.size() > 2) {
		ParseOptionsArgument(context, *arguments[2], *bind, fname);
	}

	// guardrails snapshot (FR-3.4): features checked here, rows incrementally
	bind->max_rows = ReadSettingUBigint(context, "anofox_tabfm_max_rows", 10000);
	const auto max_features = ReadSettingUBigint(context, "anofox_tabfm_max_features", 500);
	const auto feature_count = fields.size() - 1; // everything but the target
	if (feature_count > max_features) {
		throw BinderException("%s: %llu feature columns exceed anofox_tabfm_max_features (%llu). Raise it with "
		                      "SET anofox_tabfm_max_features = %llu; or pass a feature list",
		                      fname, static_cast<unsigned long long>(feature_count),
		                      static_cast<unsigned long long>(max_features),
		                      static_cast<unsigned long long>(feature_count));
	}

	function.return_type = is_window ? bind->WindowStructType() : bind->ListStructType();

	// capture engine settings + the DB handle now: finalize runs on a worker
	// thread with no ClientContext (tabfm_engine.cpp reads these).
	bind->context.db = &DatabaseInstance::GetDatabase(context);
	bind->context.threads = NumericCast<int64_t>(ReadSettingUBigint(context, "anofox_tabfm_threads", 1));
	Value setting;
	if (context.TryGetCurrentSetting("anofox_tabfm_model_manifest", setting) && !setting.IsNull()) {
		bind->context.model_manifest_path = setting.ToString();
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_cache_dir", setting) && !setting.IsNull()) {
		bind->context.cache_dir = ExpandUserHome(setting.ToString());
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_device", setting) && !setting.IsNull()) {
		bind->context.device = setting.ToString();
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_gpu_precision", setting) && !setting.IsNull()) {
		bind->context.gpu_precision = StringUtil::Lower(setting.ToString());
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_cpu_prepack", setting) && !setting.IsNull()) {
		bind->context.cpu_prepack = BooleanValue::Get(setting);
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_mxr_source", setting) && !setting.IsNull()) {
		bind->context.mxr_source = setting.ToString();
	}

	// once per query: bind runs once, update/finalize run per group/row
	PostHogTelemetry::Instance().CaptureFunctionExecution(fname);
	return std::move(bind);
}

unique_ptr<FunctionData> PredictAggBind(ClientContext &context, AggregateFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	return PredictBindInternal(context, function, arguments, "tabfm_predict_agg", false);
}

unique_ptr<FunctionData> PredictWinBind(ClientContext &context, AggregateFunction &function,
                                        vector<unique_ptr<Expression>> &arguments) {
	return PredictBindInternal(context, function, arguments, "tabfm_predict_win", true);
}

//===--------------------------------------------------------------------===//
// Aggregate state
//===--------------------------------------------------------------------===//

//! Heap-side row buffer for the aggregate path (S04 mechanics: the state slot
//! itself stays trivially movable; the destructor callback releases this).
struct PredictRowBuffer {
	explicit PredictRowBuffer(const LogicalType &row_type)
	    : collection(Allocator::DefaultAllocator(), vector<LogicalType> {row_type}) {
		append_chunk.Initialize(Allocator::DefaultAllocator(), vector<LogicalType> {row_type}, 1);
	}

	ColumnDataCollection collection;
	DataChunk append_chunk;

	void Append(const Value &row_value) {
		append_chunk.Reset();
		append_chunk.SetValue(0, 0, row_value);
		append_chunk.SetCardinality(1);
		collection.Append(append_chunk);
	}
};

//! Cursor cache for the window path (mode.cpp-style paged reads over the
//! partition's ColumnDataCollection).
struct WindowRowReader {
	const ColumnDataCollection *collection = nullptr;
	ColumnDataScanState scan;
	DataChunk page;

	Value Read(const WindowPartitionInput &partition, idx_t row_idx) {
		if (collection != partition.inputs) {
			collection = partition.inputs;
			vector<column_t> ids {partition.column_ids[0]}; // the row-struct argument
			page.Destroy();
			collection->InitializeScan(scan, std::move(ids));
			collection->InitializeScanChunk(scan, page);
			collection->Seek(0, scan, page);
		}
		if (row_idx < scan.current_row_index || row_idx >= scan.next_row_index) {
			collection->Seek(row_idx, scan, page);
		}
		return page.data[0].GetValue(row_idx - scan.current_row_index);
	}
};

struct PredictAggState {
	PredictRowBuffer *rows;
	WindowRowReader *reader;
};

void PredictStateInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<PredictAggState *>(state_ptr);
	state.rows = nullptr;
	state.reader = nullptr;
}

void FreeState(PredictAggState &state) {
	delete state.rows;
	state.rows = nullptr;
	delete state.reader;
	state.reader = nullptr;
}

void PredictStateDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<PredictAggState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		FreeState(*states[sdata.sel->get_index(i)]);
	}
}

//! DuckDB does not guarantee the Destroy callback runs when a query errors
//! mid-aggregation (ungrouped and grouped paths alike). Our state owns heap
//! buffers backed by a DuckDB Allocator, so an un-freed buffer trips the
//! allocation_count==0 assertion at shutdown. This guard frees every state in
//! the current state vector before the exception propagates.
struct StateVectorCleanup {
	Vector &state_vector;
	idx_t count;
	bool released = false;

	~StateVectorCleanup() {
		if (released) {
			return;
		}
		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);
		auto states = reinterpret_cast<PredictAggState **>(sdata.data);
		for (idx_t i = 0; i < count; i++) {
			FreeState(*states[sdata.sel->get_index(i)]);
		}
	}
};

idx_t PredictStateSize(const AggregateFunction &) {
	return sizeof(PredictAggState);
}

//===--------------------------------------------------------------------===//
// Aggregate update / combine / finalize
//===--------------------------------------------------------------------===//
void PredictAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector, idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<PredictBindData>();
	StateVectorCleanup guard {state_vector, count};

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<PredictAggState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (!state.rows) {
			state.rows = new PredictRowBuffer(bind.row_type);
		}
		// incremental guardrail: anofox_tabfm_max_rows per predict/group (FR-3.4)
		if (state.rows->collection.Count() >= bind.max_rows) {
			throw InvalidInputException(
			    "tabfm_predict: predict group exceeds anofox_tabfm_max_rows (%llu). Raise it with "
			    "SET anofox_tabfm_max_rows = <n>; or split the group",
			    static_cast<unsigned long long>(bind.max_rows));
		}
		state.rows->Append(inputs[0].GetValue(i));
	}
	guard.released = true;
}

void PredictAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                       idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<PredictBindData>();
	StateVectorCleanup source_guard {source_vector, count};
	StateVectorCleanup target_guard {target_vector, count};

	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<PredictAggState **>(source_data.data);
	auto targets = reinterpret_cast<PredictAggState **>(target_data.data);

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.rows) {
			continue;
		}
		if (!target.rows) {
			// transfer ownership; row order is not load-bearing (S04 #6)
			target.rows = source.rows;
			source.rows = nullptr;
			continue;
		}
		target.rows->collection.Combine(source.rows->collection);
		delete source.rows;
		source.rows = nullptr;
		if (target.rows->collection.Count() > bind.max_rows) {
			throw InvalidInputException(
			    "tabfm_predict: predict group exceeds anofox_tabfm_max_rows (%llu). Raise it with "
			    "SET anofox_tabfm_max_rows = <n>; or split the group",
			    static_cast<unsigned long long>(bind.max_rows));
		}
	}
	source_guard.released = true;
	target_guard.released = true;
}

//! Split a materialized row-struct Value into its field values.
vector<Value> RowChildren(const Value &row_value, const LogicalType &row_type) {
	if (row_value.IsNull()) {
		vector<Value> children;
		for (auto &field : StructType::GetChildTypes(row_type)) {
			children.emplace_back(field.second); // typed NULLs
		}
		return children;
	}
	return StructValue::GetChildren(row_value);
}

void PredictAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<PredictBindData>();
	StateVectorCleanup guard {state_vector, count};

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<PredictAggState **>(sdata.data);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t result_idx = i + offset;

		if (!state.rows || state.rows->collection.Count() == 0) {
			// e.g. FILTER (WHERE …) removed every row: an empty result list
			list_entries[result_idx] = list_entry_t(ListVector::GetListSize(result), 0);
			continue;
		}

		// materialize the group once
		vector<Value> row_values;
		row_values.reserve(state.rows->collection.Count());
		for (auto &chunk : state.rows->collection.Chunks()) {
			for (idx_t r = 0; r < chunk.size(); r++) {
				row_values.push_back(chunk.data[0].GetValue(r));
			}
		}
		vector<vector<Value>> rows;
		rows.reserve(row_values.size());
		bool has_context = false;
		for (auto &row_value : row_values) {
			rows.push_back(RowChildren(row_value, bind.row_type));
			has_context = has_context || !rows.back()[bind.target_idx].IsNull();
		}
		if (!has_context) {
			// FR-3.4 / SQL-API §5 (the no-NULL-targets case, in contrast, is a
			// valid result: every row comes back with is_training = true)
			throw InvalidInputException("tabfm_predict: target '%s' has no non-NULL rows to use as context",
			                            bind.target);
		}

		// ENGINE SEAM: one engine call per group (HLD §4.6)
		PredictInput engine_input {rows,           bind.row_type, bind.target_idx, bind.target_type,
		                           bind.target,     bind.options,  bind.context};
		auto predictions = GetPredictEngine().Predict(engine_input);

		const idx_t list_start = ListVector::GetListSize(result);
		const bool emit_proba = bind.EmitProba();
		for (idx_t j = 0; j < rows.size(); j++) {
			child_list_t<Value> struct_fields;
			struct_fields.emplace_back("cols", row_values[j]);
			struct_fields.emplace_back("yhat", predictions.yhat[j]);
			struct_fields.emplace_back("yhat_score", predictions.yhat_score[j]);
			struct_fields.emplace_back("is_training", Value::BOOLEAN(!rows[j][bind.target_idx].IsNull()));
			if (emit_proba) {
				struct_fields.emplace_back("proba", predictions.proba[j]);
			}
			// ListVector::PushBack grows the child vector as needed (10k+ safe)
			ListVector::PushBack(result, Value::STRUCT(std::move(struct_fields)));
		}
		list_entries[result_idx] = list_entry_t(list_start, rows.size());
	}
	guard.released = true;
}

//===--------------------------------------------------------------------===//
// Window variant: refuse the plain-aggregate path
//===--------------------------------------------------------------------===//
[[noreturn]] void ThrowPredictWinNeedsOver() {
	throw InvalidInputException(
	    "tabfm_predict_win requires an OVER clause with a moving frame, e.g. OVER (PARTITION BY sku ORDER BY ts "
	    "ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING); for whole-group prediction use tabfm_predict_agg");
}

void PredictWinUpdate(Vector[], AggregateInputData &, idx_t, Vector &, idx_t) {
	ThrowPredictWinNeedsOver();
}

void PredictWinCombine(Vector &, Vector &, AggregateInputData &, idx_t) {
	ThrowPredictWinNeedsOver();
}

void PredictWinFinalize(Vector &, AggregateInputData &, Vector &, idx_t, idx_t) {
	ThrowPredictWinNeedsOver();
}

//===--------------------------------------------------------------------===//
// Window callback (the real tabfm_predict_win)
//
// SubFrames = the context: every row in every subframe with a non-NULL target
// trains the in-context model. The scored row is the row immediately AFTER
// the frame (frames.back().end): under the documented usage
// `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` DuckDB computes the half-open
// frame end as exactly the current row's partition index (unclamped for every
// row, including partition starts) — see tabfm_predict.hpp for why `rid`
// cannot be used as a partition index in v1.5.4.
//===--------------------------------------------------------------------===//
void WriteWindowResult(Vector &result, idx_t rid, const PredictBindData &bind, const Value &yhat, const Value &score,
                       const Value &proba) {
	auto &entries = StructVector::GetEntries(result);
	entries[0]->SetValue(rid, yhat);
	entries[1]->SetValue(rid, score);
	if (bind.EmitProba()) {
		entries[2]->SetValue(rid, proba);
	}
}

void WriteWindowNull(Vector &result, idx_t rid, const PredictBindData &bind) {
	WriteWindowResult(result, rid, bind, Value(bind.YhatType()), Value(LogicalType::DOUBLE),
	                  Value(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)));
	FlatVector::SetNull(result, rid, true);
}

void PredictWinWindow(AggregateInputData &aggr_input_data, const WindowPartitionInput &partition, const_data_ptr_t,
                      data_ptr_t l_state, const SubFrames &frames, Vector &result, idx_t rid) {
	auto &bind = aggr_input_data.bind_data->Cast<PredictBindData>();
	auto &state = *reinterpret_cast<PredictAggState *>(l_state);
	// DuckDB may skip the Destroy callback when a query errors mid-window, so any
	// throw below (row read, session load, inference) would leak this state's heap
	// buffers. FreeState is idempotent (it nulls both pointers), so freeing here
	// and again at Destroy is safe. The normal path keeps `reader` for reuse
	// across window calls; only an exception releases it early.
	try {
	if (!state.reader) {
		state.reader = new WindowRowReader();
	}
	if (!partition.inputs || frames.empty()) {
		WriteWindowNull(result, rid, bind);
		return;
	}

	// the scored row sits right after the frame (must be inside the partition)
	const idx_t scored_row = frames.back().end;
	if (scored_row >= partition.count) {
		WriteWindowNull(result, rid, bind);
		return;
	}

	// collect the context rows from all subframes (FILTER-ed rows excluded)
	vector<vector<Value>> rows;
	for (auto &frame : frames) {
		for (idx_t r = frame.start; r < frame.end; r++) {
			if (!partition.filter_mask.AllValid() && !partition.filter_mask.RowIsValid(r)) {
				continue;
			}
			auto row_value = state.reader->Read(partition, r);
			auto children = RowChildren(row_value, bind.row_type);
			if (children[bind.target_idx].IsNull()) {
				continue; // NULL-target rows cannot train the context
			}
			rows.push_back(std::move(children));
		}
	}
	if (rows.empty()) {
		// empty frame / all-NULL-target frame: a soft NULL, not an error —
		// rolling predictions legitimately start with no history
		WriteWindowNull(result, rid, bind);
		return;
	}

	// append the scored row with its target hidden (never leaks into context)
	auto scored_children = RowChildren(state.reader->Read(partition, scored_row), bind.row_type);
	scored_children[bind.target_idx] = Value(bind.target_type);
	rows.push_back(std::move(scored_children));

	PredictInput engine_input {rows,       bind.row_type, bind.target_idx, bind.target_type,
	                           bind.target, bind.options,  bind.context};
	auto predictions = GetPredictEngine().Predict(engine_input);
	const auto last = rows.size() - 1;
	WriteWindowResult(result, rid, bind, predictions.yhat[last], predictions.yhat_score[last],
	                  bind.EmitProba() ? predictions.proba[last] : Value());
	} catch (...) {
		FreeState(state); // release heap buffers before the error propagates
		throw;
	}
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
AggregateFunction MakePredictFunction(const string &name, bool with_opts, bool is_window) {
	vector<LogicalType> arguments {LogicalType::ANY, LogicalType::VARCHAR};
	if (with_opts) {
		arguments.push_back(LogicalType::ANY); // MAP or STRUCT; LIST redirects at bind
	}
	AggregateFunction function(name, std::move(arguments), LogicalType::ANY, PredictStateSize,
	                           PredictStateInitialize, is_window ? PredictWinUpdate : PredictAggUpdate,
	                           is_window ? PredictWinCombine : PredictAggCombine,
	                           is_window ? PredictWinFinalize : PredictAggFinalize,
	                           /*simple_update=*/nullptr, is_window ? PredictWinBind : PredictAggBind,
	                           PredictStateDestroy);
	if (is_window) {
		function.SetWindowCallback(PredictWinWindow);
	}
	return function;
}

void RegisterPredictSet(ExtensionLoader &loader, const string &full_name, bool is_window) {
	AggregateFunctionSet set(full_name);
	set.AddFunction(MakePredictFunction(full_name, false, is_window));
	set.AddFunction(MakePredictFunction(full_name, true, is_window));

	CreateAggregateFunctionInfo info(set);
	loader.RegisterFunction(info);
}

} // anonymous namespace

// The predict aggregate is the engine boundary the tabfm_classify / tabfm_regress
// table macros build on (tabfm_macros.cpp). It is registered under an internal
// `__anofox_tabfm_*` name with no public alias: the user-facing surface in v1 is
// just classify/regress (SQL-API redesign 2026-07-04). The grouped/aggregate/
// windowed surfaces (tabfm_predict_by / _agg / _win) are added back on the same
// engine when needed; the windowed variant is registered (dormant) so its
// SetWindowCallback path stays compiled and covered.
void RegisterPredictAggFunction(ExtensionLoader &loader) {
	RegisterPredictSet(loader, "__anofox_tabfm_predict_agg", /*is_window=*/false);
	RegisterPredictSet(loader, "__anofox_tabfm_predict_win", /*is_window=*/true);
}

} // namespace anofox
} // namespace duckdb
