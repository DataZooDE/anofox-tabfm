//===----------------------------------------------------------------------===//
// tabfm_metrics_classification.cpp — WS-G classification metric aggregates
//
// Implements: anofox_tabfm_accuracy / tabfm_accuracy (CMET-01)
//             anofox_tabfm_precision / tabfm_precision (CMET-02)
//             anofox_tabfm_recall / tabfm_recall (CMET-02)
//             anofox_tabfm_f1 / tabfm_f1 (CMET-02)
//             anofox_tabfm_log_loss / tabfm_log_loss (CMET-03)
//             anofox_tabfm_roc_auc / tabfm_roc_auc (CMET-04)
//             anofox_tabfm_confusion_matrix / tabfm_confusion_matrix (CMET-05)
//             anofox_tabfm_ece / tabfm_ece (CMET-06)
//
// Aggregate triplet pattern mirrors tabfm_predict_agg.cpp (same state-size /
// init / update / combine / finalize layout, UnifiedVectorFormat NULL-skip,
// telemetry-at-bind convention).
//
// Spec ref: CMET-01..06, SQL-API §4 metric surface
// Research ref: 01-RESEARCH.md §Metric Formulas, 01-PATTERNS.md
//===----------------------------------------------------------------------===//

#include "tabfm_metrics_classification.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {
namespace anofox {

namespace {

//===----------------------------------------------------------------------===//
// tabfm_accuracy — CMET-01
//
// Computes correct / total as DOUBLE over (actual ANY, predicted ANY) pairs.
// NULL semantics: rows where actual OR predicted is NULL are skipped (not
// counted in either numerator or denominator). Empty / all-NULL input returns
// NULL (standard SQL aggregate NULL-on-empty). Alias: tabfm_accuracy.
//
// sklearn ref: sklearn.metrics.accuracy_score (normalize=True)
//===----------------------------------------------------------------------===//

struct AccuracyState {
	int64_t correct; // pairs where actual == predicted (as string)
	int64_t total;   // valid (non-NULL) pairs seen
};

idx_t AccuracyStateSize(const AggregateFunction &) {
	return sizeof(AccuracyState);
}

void AccuracyStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) AccuracyState{0, 0};
}

// Update — UnifiedVectorFormat NULL-skip (01-PATTERNS.md "NULL skip in Update")
// Signature: (inputs, aggr_input_data, input_count [discarded], states, row_count)
void AccuracyUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto states = reinterpret_cast<AccuracyState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		// NULL skip: matches standard SQL aggregate semantics (CONTEXT.md)
		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		auto &state = *states[sidx];
		state.total++;

		// Compare as string_t directly (zero allocation) so the aggregate works on
		// ANY type pair. Using raw UnifiedVectorFormat data avoids the per-row
		// Value heap allocation from GetValue(i) (WR-03).
		auto *actual_raw    = UnifiedVectorFormat::GetData<string_t>(actual_data);
		auto *predicted_raw = UnifiedVectorFormat::GetData<string_t>(predicted_data);
		if (actual_raw[aidx] == predicted_raw[pidx]) {
			state.correct++;
		}
	}
}

void AccuracyCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<AccuracyState **>(source_data.data);
	auto targets = reinterpret_cast<AccuracyState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.correct += src.correct;
		tgt.total += src.total;
	}
}

// Finalize — write correct/total as DOUBLE; SetNull when total == 0
// (T-01-01: division-by-zero guard)
void AccuracyFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                      idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<AccuracyState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.total == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    static_cast<double>(state.correct) / static_cast<double>(state.total);
		}
	}
}

unique_ptr<FunctionData> AccuracyBind(ClientContext &, AggregateFunction &,
                                       vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_accuracy");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_precision / tabfm_recall / tabfm_f1 — CMET-02
//
// Per-class TP/FP/FN accumulation; macro/micro/weighted averaging.
// The `avg` parameter is REQUIRED — there is NO silent default.
// Two overloads are registered:
//   - 2-arg (actual, predicted): bind always throws the named exception
//   - 3-arg (actual, predicted, avg): accepted; avg validated at Finalize
//
// NULL semantics: rows where actual OR predicted is NULL skipped.
// zero_division=0: undefined precision/recall return 0.0 (sklearn convention).
//
// sklearn ref: f1_score(average=..., zero_division=0)
//===----------------------------------------------------------------------===//

struct ClassCounts {
	int64_t tp = 0; // true positives
	int64_t fp = 0; // false positives
	int64_t fn = 0; // false negatives
};

// Fixed-size slot holding pointer to heap-allocated map (O(N) StateDestroy pattern)
struct F1StateSlot {
	std::unordered_map<std::string, ClassCounts> *data;
};

idx_t F1StateSize(const AggregateFunction &) {
	return sizeof(F1StateSlot);
}

void F1StateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &slot = *reinterpret_cast<F1StateSlot *>(state_ptr);
	slot.data  = nullptr;
}

void F1StateDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto slots = reinterpret_cast<F1StateSlot **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &slot = *slots[sdata.sel->get_index(i)];
		delete slot.data;
		slot.data = nullptr;
	}
}

// Bind data carries: avg mode ('macro'/'micro'/'weighted'), metric type ('p'/'r'/'f')
struct F1BindData : FunctionData {
	std::string avg;    // averaging mode
	char        metric; // 'p' precision, 'r' recall, 'f' f1

	F1BindData(std::string avg_mode, char m) : avg(std::move(avg_mode)), metric(m) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<F1BindData>(avg, metric);
	}
	bool Equals(const FunctionData &other) const override {
		auto &o = other.Cast<F1BindData>();
		return avg == o.avg && metric == o.metric;
	}
};

// safe_div returns 0.0 when denominator is 0 (sklearn zero_division=0 convention)
static inline double safe_div(double num, double denom) {
	return (denom == 0.0) ? 0.0 : num / denom;
}

// Bind factory for 3-arg metric functions. metric: 'p','r','f'. fn_name used in error messages.
// The avg value is read from arguments[2] at Finalize time via bind data.
// We read it early here if it is a constant expression (for validation).
static unique_ptr<FunctionData> BindF1Metric(ClientContext &context, const char *fn_name, char metric,
                                              vector<unique_ptr<Expression>> &arguments) {
	std::string avg_mode;
	if (arguments.size() >= 3) {
		try {
			Value v = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
			if (!v.IsNull()) {
				avg_mode = v.ToString();
			}
		} catch (...) {
			// Not a constant expression — avg will be validated at Finalize time
		}
		if (!avg_mode.empty() && avg_mode != "macro" && avg_mode != "micro" &&
		    avg_mode != "weighted") {
			throw InvalidInputException("%s: invalid avg '%s' — valid values: 'micro', 'macro', 'weighted'",
			                            fn_name, avg_mode.c_str());
		}
	}
	return make_uniq<F1BindData>(std::move(avg_mode), metric);
}

unique_ptr<FunctionData> PrecisionBind(ClientContext &ctx, AggregateFunction &,
                                        vector<unique_ptr<Expression>> &arguments) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_precision");
	return BindF1Metric(ctx, "tabfm_precision", 'p', arguments);
}

unique_ptr<FunctionData> RecallBind(ClientContext &ctx, AggregateFunction &,
                                     vector<unique_ptr<Expression>> &arguments) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_recall");
	return BindF1Metric(ctx, "tabfm_recall", 'r', arguments);
}

unique_ptr<FunctionData> F1ScoreBind(ClientContext &ctx, AggregateFunction &,
                                      vector<unique_ptr<Expression>> &arguments) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_f1");
	return BindF1Metric(ctx, "tabfm_f1", 'f', arguments);
}

void F1Update(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto slots = reinterpret_cast<F1StateSlot **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		auto &slot = *slots[sidx];
		if (!slot.data) {
			slot.data = new std::unordered_map<std::string, ClassCounts>();
		}
		auto &class_map = *slot.data;

		// Use raw string_t pointers from UnifiedVectorFormat to avoid per-row
		// Value heap allocation from GetValue(i) (WR-03).
		auto *actual_raw    = UnifiedVectorFormat::GetData<string_t>(actual_data);
		auto *predicted_raw = UnifiedVectorFormat::GetData<string_t>(predicted_data);
		std::string actual_str    = actual_raw[aidx].GetString();
		std::string predicted_str = predicted_raw[pidx].GetString();

		// Ensure both class entries exist before modifying them
		class_map[actual_str];    // default-insert if missing
		class_map[predicted_str]; // default-insert if missing

		if (actual_str == predicted_str) {
			class_map[actual_str].tp++;
		} else {
			class_map[actual_str].fn++;    // actual class missed
			class_map[predicted_str].fp++; // predicted class was wrong
		}
	}
}

void F1Combine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<F1StateSlot **>(source_data.data);
	auto targets = reinterpret_cast<F1StateSlot **>(target_data.data);

	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		if (!src.data) {
			continue;
		}
		if (!tgt.data) {
			tgt.data = new std::unordered_map<std::string, ClassCounts>();
		}
		for (auto &[cls, src_counts] : *src.data) {
			auto &tgt_counts = (*tgt.data)[cls];
			tgt_counts.tp += src_counts.tp;
			tgt_counts.fp += src_counts.fp;
			tgt_counts.fn += src_counts.fn;
		}
	}
}

void F1Finalize(Vector &state_vector, AggregateInputData &aggr_input, Vector &result, idx_t count,
                idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto slots = reinterpret_cast<F1StateSlot **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		auto &slot = *slots[sdata.sel->get_index(i)];
		if (!slot.data || slot.data->empty()) {
			FlatVector::SetNull(result, i + offset, true);
			continue;
		}

		std::string avg_mode;
		char        metric = 'f';
		if (aggr_input.bind_data) {
			auto &bd = aggr_input.bind_data->Cast<F1BindData>();
			avg_mode = bd.avg;
			metric   = bd.metric;
		}
		if (avg_mode.empty()) {
			// Non-constant avg expression was not resolved at bind time. Throw an
			// actionable error rather than silently returning NULL (SQL-API §5).
			std::string fn_name = (metric == 'p') ? "tabfm_precision"
			                    : (metric == 'r') ? "tabfm_recall"
			                                      : "tabfm_f1";
			throw InvalidInputException(
			    "%s: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'",
			    fn_name.c_str());
		}

		auto  &class_map = *slot.data;
		double value     = 0.0;

		if (avg_mode == "micro") {
			int64_t tp_total = 0, fp_total = 0, fn_total = 0;
			for (auto &[cls, cc] : class_map) {
				tp_total += cc.tp;
				fp_total += cc.fp;
				fn_total += cc.fn;
			}
			double p = safe_div(static_cast<double>(tp_total),
			                    static_cast<double>(tp_total + fp_total));
			double r = safe_div(static_cast<double>(tp_total),
			                    static_cast<double>(tp_total + fn_total));
			if (metric == 'p') {
				value = p;
			} else if (metric == 'r') {
				value = r;
			} else {
				value = safe_div(2.0 * p * r, p + r);
			}
		} else {
			// macro or weighted
			double  sum_val  = 0.0;
			int64_t total_n  = 0;
			int64_t n_classes = static_cast<int64_t>(class_map.size());

			for (auto &[cls, cc] : class_map) {
				double p_c = safe_div(static_cast<double>(cc.tp),
				                      static_cast<double>(cc.tp + cc.fp));
				double r_c = safe_div(static_cast<double>(cc.tp),
				                      static_cast<double>(cc.tp + cc.fn));
				double f_c = safe_div(2.0 * p_c * r_c, p_c + r_c);
				double v_c = (metric == 'p') ? p_c : (metric == 'r') ? r_c : f_c;

				int64_t support_c = cc.tp + cc.fn;
				if (avg_mode == "weighted") {
					sum_val += v_c * static_cast<double>(support_c);
				} else {
					sum_val += v_c;
				}
				total_n += support_c;
			}

			if (avg_mode == "weighted") {
				value = (total_n == 0) ? 0.0 : sum_val / static_cast<double>(total_n);
			} else {
				value = (n_classes == 0) ? 0.0 : sum_val / static_cast<double>(n_classes);
			}
		}

		FlatVector::GetData<double>(result)[i + offset] = value;
	}
}

//===----------------------------------------------------------------------===//
// tabfm_log_loss — CMET-03
//
// Computes -(1/N)*sum_i log(clip(proba[actual_i], eps, 1-eps))
// eps = 1e-15 (sklearn default).
// Input: (actual VARCHAR, proba MAP(VARCHAR, DOUBLE))
// MAP access via MapValue::GetChildren + StructValue::GetChildren.
// Missing key in MAP treated as p=0 -> clip to eps (T-01-02-02 mitigation).
// NULL actual or NULL proba rows skipped.
//
// O(N*C) update cost where C = number of classes; acceptable for C<=10.
// sklearn ref: log_loss (normalize=True, eps=1e-15)
//===----------------------------------------------------------------------===//

struct LogLossState {
	double  sum_ll = 0.0;
	int64_t n      = 0;
};

idx_t LogLossStateSize(const AggregateFunction &) {
	return sizeof(LogLossState);
}

void LogLossStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) LogLossState{0.0, 0};
}

void LogLossUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, proba_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, proba_data);
	auto states = reinterpret_cast<LogLossState **>(sdata.data);

	static constexpr double kEps = 1e-15;

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = proba_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !proba_data.validity.RowIsValid(pidx)) {
			continue;
		}

		Value actual_val = inputs[0].GetValue(i);
		Value proba_val  = inputs[1].GetValue(i);
		if (actual_val.IsNull() || proba_val.IsNull()) {
			continue;
		}

		std::string actual_str = actual_val.ToString();
		double      p          = 0.0; // default: missing key -> p=0 (T-01-02-02)

		// Iterate MAP(VARCHAR,DOUBLE): each entry is STRUCT({key VARCHAR, value DOUBLE})
		for (auto &kv : MapValue::GetChildren(proba_val)) {
			auto &entry = StructValue::GetChildren(kv);
			if (entry[0].ToString() == actual_str) {
				p = DoubleValue::Get(entry[1]);
				break;
			}
		}

		// Clip to [eps, 1-eps] to prevent -log(0) = +Inf (T-01-02-03)
		p = std::max(kEps, std::min(1.0 - kEps, p));

		auto &state = *states[sidx];
		state.sum_ll += -std::log(p);
		state.n++;
	}
}

void LogLossCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<LogLossState **>(source_data.data);
	auto targets = reinterpret_cast<LogLossState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.sum_ll += src.sum_ll;
		tgt.n += src.n;
	}
}

void LogLossFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                     idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<LogLossState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_ll / static_cast<double>(state.n);
		}
	}
}

unique_ptr<FunctionData> LogLossBind(ClientContext &, AggregateFunction &,
                                      vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_log_loss");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_roc_auc — CMET-04
//
// Computes ROC-AUC with correct tie handling (Mann-Whitney U / rank-sum form).
// Multiclass: avg='ovr' (one-vs-rest) or avg='ovo' (one-vs-one).
// The `avg` parameter is REQUIRED — there is NO silent default.
//
// State: per-class O(N) vector of (score, is_positive) pairs
// Finalize: sort + tie-group + Mann-Whitney U formula = AUC
//
// sklearn ref: roc_auc_score(multi_class='ovr'/'ovo', average='macro')
// See: 01-RESEARCH.md §Pitfall 1 for tie-group correctness argument
//===----------------------------------------------------------------------===//

struct ScoreLabel {
	double score;
	int    label; // 1 = positive (actual == class), 0 = negative
};

struct AUCClassData {
	std::vector<ScoreLabel> pairs;
};

struct AUCStateSlot {
	std::unordered_map<std::string, AUCClassData> *data;
};

idx_t AUCStateSize(const AggregateFunction &) {
	return sizeof(AUCStateSlot);
}

void AUCStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &slot = *reinterpret_cast<AUCStateSlot *>(state_ptr);
	slot.data  = nullptr;
}

void AUCStateDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto slots = reinterpret_cast<AUCStateSlot **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &slot = *slots[sdata.sel->get_index(i)];
		delete slot.data;
		slot.data = nullptr;
	}
}

struct AUCBindData : FunctionData {
	std::string avg; // "ovr" or "ovo"

	explicit AUCBindData(std::string avg_mode) : avg(std::move(avg_mode)) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AUCBindData>(avg);
	}
	bool Equals(const FunctionData &other) const override {
		return avg == other.Cast<AUCBindData>().avg;
	}
};

unique_ptr<FunctionData> AUCBind(ClientContext &ctx, AggregateFunction &,
                                  vector<unique_ptr<Expression>> &arguments) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_roc_auc");

	// Read avg from arguments[2] if it's a constant
	std::string avg_mode;
	if (arguments.size() >= 3) {
		try {
			Value v = ExpressionExecutor::EvaluateScalar(ctx, *arguments[2]);
			if (!v.IsNull()) {
				avg_mode = v.ToString();
			}
		} catch (...) {
		}
		if (!avg_mode.empty() && avg_mode != "ovr" && avg_mode != "ovo") {
			throw InvalidInputException(
			    "tabfm_roc_auc: invalid avg '%s' — valid values: 'ovr', 'ovo'", avg_mode.c_str());
		}
	}
	return make_uniq<AUCBindData>(std::move(avg_mode));
}

void AUCUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	// inputs[0]=actual VARCHAR, inputs[1]=proba MAP(VARCHAR,DOUBLE), inputs[2]=avg VARCHAR
	UnifiedVectorFormat sdata, actual_data, proba_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, proba_data);
	auto slots = reinterpret_cast<AUCStateSlot **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = proba_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !proba_data.validity.RowIsValid(pidx)) {
			continue;
		}

		Value actual_val = inputs[0].GetValue(i);
		Value proba_val  = inputs[1].GetValue(i);
		if (actual_val.IsNull() || proba_val.IsNull()) {
			continue;
		}

		std::string actual_str = actual_val.ToString();

		auto &slot = *slots[sidx];
		if (!slot.data) {
			slot.data = new std::unordered_map<std::string, AUCClassData>();
		}
		auto &class_map = *slot.data;

		// For each class in proba MAP, record (class_score, is_positive)
		// is_positive = (actual_str == class)
		for (auto &kv : MapValue::GetChildren(proba_val)) {
			auto &entry     = StructValue::GetChildren(kv);
			std::string cls = entry[0].ToString();
			double      sc  = DoubleValue::Get(entry[1]);
			int         pos = (actual_str == cls) ? 1 : 0;
			class_map[cls].pairs.push_back({sc, pos});
		}
	}
}

void AUCCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<AUCStateSlot **>(source_data.data);
	auto targets = reinterpret_cast<AUCStateSlot **>(target_data.data);

	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		if (!src.data) {
			continue;
		}
		if (!tgt.data) {
			tgt.data = new std::unordered_map<std::string, AUCClassData>();
		}
		for (auto &[cls, src_cd] : *src.data) {
			auto &tgt_cd = (*tgt.data)[cls];
			tgt_cd.pairs.insert(tgt_cd.pairs.end(), src_cd.pairs.begin(), src_cd.pairs.end());
		}
	}
}

// Compute binary AUC via Mann-Whitney U / rank-sum with tie-group handling.
// Equal-score groups are advanced simultaneously: no optimistic or pessimistic
// tie interpolation (01-RESEARCH.md §Pitfall 1).
static double ComputeBinaryAUC(std::vector<ScoreLabel> &pairs) {
	int64_t n_pos = 0, n_neg = 0;
	for (auto &p : pairs) {
		if (p.label == 1) {
			n_pos++;
		} else {
			n_neg++;
		}
	}
	if (n_pos == 0 || n_neg == 0) {
		return std::numeric_limits<double>::quiet_NaN(); // undefined binary AUC
	}

	// Sort descending by score
	std::sort(pairs.begin(), pairs.end(),
	          [](const ScoreLabel &a, const ScoreLabel &b) { return a.score > b.score; });

	// Mann-Whitney U:
	//   U += n_pos_group * n_neg_below + n_pos_group * n_neg_group * 0.5
	// AUC = U / (n_pos * n_neg)
	double U = 0.0;
	idx_t  j = 0;
	while (j < static_cast<idx_t>(pairs.size())) {
		// Tie group [j, k) with the same score
		idx_t k = j;
		while (k < static_cast<idx_t>(pairs.size()) && pairs[k].score == pairs[j].score) {
			k++;
		}
		int64_t n_pos_group = 0, n_neg_group = 0;
		for (idx_t m = j; m < k; m++) {
			if (pairs[m].label == 1) {
				n_pos_group++;
			} else {
				n_neg_group++;
			}
		}
		// Count negatives with strictly lower score (= in remaining groups [k..end])
		int64_t n_neg_below = 0;
		for (idx_t m = k; m < static_cast<idx_t>(pairs.size()); m++) {
			if (pairs[m].label == 0) {
				n_neg_below++;
			}
		}
		U += static_cast<double>(n_pos_group) * static_cast<double>(n_neg_below) +
		     static_cast<double>(n_pos_group) * static_cast<double>(n_neg_group) * 0.5;
		j = k;
	}

	return U / (static_cast<double>(n_pos) * static_cast<double>(n_neg));
}

void AUCFinalize(Vector &state_vector, AggregateInputData &aggr_input, Vector &result, idx_t count,
                 idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto slots = reinterpret_cast<AUCStateSlot **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		auto &slot = *slots[sdata.sel->get_index(i)];
		if (!slot.data || slot.data->empty()) {
			FlatVector::SetNull(result, i + offset, true);
			continue;
		}

		std::string avg_mode;
		if (aggr_input.bind_data) {
			avg_mode = aggr_input.bind_data->Cast<AUCBindData>().avg;
		}
		if (avg_mode.empty()) {
			// Non-constant avg expression was not resolved at bind time. Throw an
			// actionable error rather than silently returning NULL (SQL-API §5).
			throw InvalidInputException(
			    "tabfm_roc_auc: 'avg' is required — pass avg := 'ovr' or 'ovo'");
		}

		auto &class_map = *slot.data;
		std::vector<std::string> classes;
		for (auto &[cls, _] : class_map) {
			classes.push_back(cls);
		}
		std::sort(classes.begin(), classes.end());

		double  auc_sum   = 0.0;
		int64_t auc_count = 0;

		if (avg_mode == "ovr") {
			// One-vs-rest: binary AUC for each class, macro-average
			for (auto &cls : classes) {
				auto pairs_copy = class_map[cls].pairs; // copy for sort (in-place sort)
				double auc      = ComputeBinaryAUC(pairs_copy);
				if (!std::isnan(auc)) {
					auc_sum += auc;
					auc_count++;
				}
			}
		} else {
			// ovo: one-vs-one, macro-average over all C*(C-1)/2 pairs
			// For each pair (c1, c2): AUC(c1 vs c2) averaged with AUC(c2 vs c1),
			// then averaged over all pairs.
			//
			// Row matching: since Update appends one entry per MAP class per row
			// in the same row-visitation order, pairs in class_map[c1] and
			// class_map[c2] are index-aligned (same source rows, same order).
			for (idx_t ci = 0; ci < classes.size(); ci++) {
				for (idx_t cj = ci + 1; cj < classes.size(); cj++) {
					const std::string &c1 = classes[ci];
					const std::string &c2 = classes[cj];

					auto &pairs1 = class_map[c1].pairs;
					auto &pairs2 = class_map[c2].pairs;
					if (pairs1.size() != pairs2.size()) {
						continue; // shouldn't happen
					}

					// OvO(c1,c2): use c1's score, keep rows where actual ∈ {c1,c2}
					std::vector<ScoreLabel> ovo12, ovo21;
					for (idx_t r = 0; r < static_cast<idx_t>(pairs1.size()); r++) {
						int lbl1 = pairs1[r].label; // 1 if actual=c1
						int lbl2 = pairs2[r].label; // 1 if actual=c2
						if (lbl1 == 1) {
							// actual=c1: positive for c1 vs c2; negative for c2 vs c1
							ovo12.push_back({pairs1[r].score, 1});
							ovo21.push_back({pairs2[r].score, 0});
						} else if (lbl2 == 1) {
							// actual=c2: negative for c1 vs c2; positive for c2 vs c1
							ovo12.push_back({pairs1[r].score, 0});
							ovo21.push_back({pairs2[r].score, 1});
						}
					}

					double auc_12 = ComputeBinaryAUC(ovo12);
					double auc_21 = ComputeBinaryAUC(ovo21);

					if (!std::isnan(auc_12) && !std::isnan(auc_21)) {
						auc_sum += (auc_12 + auc_21) / 2.0;
						auc_count++;
					}
				}
			}
		}

		if (auc_count == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    auc_sum / static_cast<double>(auc_count);
		}
	}
}

//===----------------------------------------------------------------------===//
// tabfm_ece — CMET-06
//
// Computes 10-bin equal-width Expected Calibration Error.
// Input: (actual VARCHAR, proba MAP(VARCHAR, DOUBLE))
// For each row, extracts argmax class and max_proba from the MAP.
// bin = floor(max_proba * 10), clamped to [0, 9].
// ECE = sum_m (bin_n[m]/N) * |accuracy(bin_m) - confidence(bin_m)|
//
// NULL actual or NULL proba rows skipped.
// sklearn/Guo2017 ref: ECE with 10 equal-width bins
//===----------------------------------------------------------------------===//

struct ECEState {
	int64_t bin_n[10];       // row count per bin
	int64_t bin_correct[10]; // correct predictions per bin
	double  bin_conf[10];    // sum of max_proba per bin
	int64_t total;
};

idx_t ECEStateSize(const AggregateFunction &) {
	return sizeof(ECEState);
}

void ECEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &state = *reinterpret_cast<ECEState *>(state_ptr);
	for (int m = 0; m < 10; m++) {
		state.bin_n[m]       = 0;
		state.bin_correct[m] = 0;
		state.bin_conf[m]    = 0.0;
	}
	state.total = 0;
}

void ECEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, proba_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, proba_data);
	auto states = reinterpret_cast<ECEState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = proba_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !proba_data.validity.RowIsValid(pidx)) {
			continue;
		}

		Value actual_val = inputs[0].GetValue(i);
		Value proba_val  = inputs[1].GetValue(i);
		if (actual_val.IsNull() || proba_val.IsNull()) {
			continue;
		}

		std::string actual_str = actual_val.ToString();
		std::string argmax_cls;
		double      max_prob = -1.0;

		for (auto &kv : MapValue::GetChildren(proba_val)) {
			auto &entry  = StructValue::GetChildren(kv);
			double score = DoubleValue::Get(entry[1]);
			if (score > max_prob) {
				max_prob   = score;
				argmax_cls = entry[0].ToString();
			}
		}

		if (max_prob < 0.0) {
			continue; // empty MAP — skip
		}

		int bin = static_cast<int>(max_prob * 10.0);
		bin     = std::min(std::max(bin, 0), 9);

		auto &state = *states[sidx];
		state.bin_n[bin]++;
		if (argmax_cls == actual_str) {
			state.bin_correct[bin]++;
		}
		state.bin_conf[bin] += max_prob;
		state.total++;
	}
}

void ECECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<ECEState **>(source_data.data);
	auto targets = reinterpret_cast<ECEState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		for (int m = 0; m < 10; m++) {
			tgt.bin_n[m]       += src.bin_n[m];
			tgt.bin_correct[m] += src.bin_correct[m];
			tgt.bin_conf[m]    += src.bin_conf[m];
		}
		tgt.total += src.total;
	}
}

void ECEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count, idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<ECEState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.total == 0) {
			FlatVector::SetNull(result, i + offset, true);
			continue;
		}
		double N   = static_cast<double>(state.total);
		double ece = 0.0;
		for (int m = 0; m < 10; m++) {
			if (state.bin_n[m] == 0) {
				continue;
			}
			double acc_m  = static_cast<double>(state.bin_correct[m]) /
			                static_cast<double>(state.bin_n[m]);
			double conf_m = state.bin_conf[m] / static_cast<double>(state.bin_n[m]);
			ece += (static_cast<double>(state.bin_n[m]) / N) * std::abs(acc_m - conf_m);
		}
		FlatVector::GetData<double>(result)[i + offset] = ece;
	}
}

unique_ptr<FunctionData> ECEBind(ClientContext &, AggregateFunction &,
                                  vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_ece");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_confusion_matrix — CMET-05
//
// Implemented as a TABLE MACRO wrapping a GROUP BY query over user-supplied
// table/column names.
//
// Security: column identifiers double-quote escaped with replace(col,'"','""')
// (T-01-02-01 mitigation — same pattern as tabfm_macros.cpp:93).
//===----------------------------------------------------------------------===//

// The macro body uses replace(col, '"', '""') to safely quote the column
// identifiers before interpolating them into a dynamically-constructed SELECT.
// This mirrors the pattern in tabfm_macros.cpp:93 for identifier safety.
static const char *CONFUSION_MACRO_BODY = R"(
    SELECT *
    FROM query(
        'SELECT '
        || '"' || replace(actual_col, '"', '""') || '"'
        || ' AS actual, '
        || '"' || replace(predicted_col, '"', '""') || '"'
        || ' AS predicted, COUNT(*) AS count'
        || ' FROM (FROM ' || data || ')'
        || ' GROUP BY 1, 2'
        || ' ORDER BY 1, 2'
    )
)";

static unique_ptr<MacroFunction> BuildConfusionMacroFunction() {
	Parser parser;
	parser.ParseQuery(CONFUSION_MACRO_BODY);
	if (parser.statements.size() != 1 ||
	    parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("tabfm_confusion_matrix macro body must be a single SELECT");
	}
	auto node     = std::move(parser.statements[0]->Cast<SelectStatement>().node);
	auto function = make_uniq<TableMacroFunction>(std::move(node));

	const char *params[] = {"data", "actual_col", "predicted_col", nullptr};
	for (int k = 0; params[k] != nullptr; k++) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(params[k]));
		function->types.push_back(LogicalType::UNKNOWN);
	}
	return std::move(function);
}

static unique_ptr<CreateMacroInfo> BuildConfusionMacroInfo(const std::string &name,
                                                            const std::string &alias_of) {
	auto info       = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->schema    = DEFAULT_SCHEMA;
	info->name      = name;
	info->temporary = true;
	info->internal  = true;
	info->alias_of  = alias_of;
	info->macros.push_back(BuildConfusionMacroFunction());

	FunctionDescription fd;
	fd.parameter_names = {"data", "actual_col", "predicted_col"};
	fd.description =
	    "Compute a tidy confusion matrix from a table, returning (actual, predicted, count) rows "
	    "for each observed (actual class, predicted class) pair. Column identifiers are safely "
	    "double-quote escaped (T-01-02-01). Returns one row per distinct (actual, predicted) "
	    "combination, ordered by actual then predicted.";
	fd.examples = {"SELECT * FROM tabfm_confusion_matrix('predictions', 'actual', 'predicted');"};
	info->descriptions.push_back(std::move(fd));
	return info;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterClassificationMetrics(ExtensionLoader &loader) {
	// --- tabfm_accuracy / anofox_tabfm_accuracy (CMET-01) ---
	{
		AggregateFunctionSet set("anofox_tabfm_accuracy");
		AggregateFunction fn("anofox_tabfm_accuracy",
		                     {LogicalType::ANY, LogicalType::ANY}, LogicalType::DOUBLE, AccuracyStateSize,
		                     AccuracyStateInit, AccuracyUpdate, AccuracyCombine, AccuracyFinalize,
		                     /*simple_update=*/nullptr, AccuracyBind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute classification accuracy: correct / total as DOUBLE over (actual, predicted) pairs. "
		    "Rows where actual OR predicted is NULL are skipped (SQL aggregate NULL semantics). "
		    "Returns NULL on empty or all-NULL input. Matches sklearn.metrics.accuracy_score(normalize=True).";
		fd.examples = {"SELECT tabfm_accuracy(actual, predicted) FROM predictions;",
		               "SELECT round(tabfm_accuracy(actual, predicted), 4) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_accuracy", {std::move(fd)});
	}

	// --- tabfm_precision / anofox_tabfm_precision (CMET-02) ---
	// Two overloads:
	//   3-arg (actual, predicted, avg): accepted
	//   2-arg (actual, predicted):      bind throws the named exception
	{
		AggregateFunctionSet set("anofox_tabfm_precision");
		AggregateFunction fn3("anofox_tabfm_precision",
		                      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
		                      LogicalType::DOUBLE, F1StateSize, F1StateInit, F1Update, F1Combine,
		                      F1Finalize, /*simple_update=*/nullptr, PrecisionBind, F1StateDestroy);
		set.AddFunction(fn3);
		AggregateFunction fn2(
		    "anofox_tabfm_precision", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
		    F1StateSize, F1StateInit, F1Update, F1Combine, F1Finalize,
		    /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_precision: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'");
			    return nullptr;
		    },
		    F1StateDestroy);
		set.AddFunction(fn2);
		FunctionDescription fd;
		fd.description =
		    "Compute precision with the specified averaging mode. avg is REQUIRED: pass 'macro', 'micro', "
		    "or 'weighted'. Undefined precision (no predicted positives for a class) returns 0.0 "
		    "(zero_division=0). NULL rows skipped. Matches sklearn.metrics.precision_score(zero_division=0).";
		fd.examples = {"SELECT tabfm_precision(actual, predicted, 'macro') FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_precision", {std::move(fd)});
	}

	// --- tabfm_recall / anofox_tabfm_recall (CMET-02) ---
	{
		AggregateFunctionSet set("anofox_tabfm_recall");
		AggregateFunction fn3("anofox_tabfm_recall",
		                      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
		                      LogicalType::DOUBLE, F1StateSize, F1StateInit, F1Update, F1Combine,
		                      F1Finalize, /*simple_update=*/nullptr, RecallBind, F1StateDestroy);
		set.AddFunction(fn3);
		AggregateFunction fn2(
		    "anofox_tabfm_recall", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
		    F1StateSize, F1StateInit, F1Update, F1Combine, F1Finalize,
		    /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_recall: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'");
			    return nullptr;
		    },
		    F1StateDestroy);
		set.AddFunction(fn2);
		FunctionDescription fd;
		fd.description =
		    "Compute recall with the specified averaging mode. avg is REQUIRED: pass 'macro', 'micro', "
		    "or 'weighted'. Undefined recall (no actual positives for a class) returns 0.0 "
		    "(zero_division=0). NULL rows skipped. Matches sklearn.metrics.recall_score(zero_division=0).";
		fd.examples = {"SELECT tabfm_recall(actual, predicted, 'macro') FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_recall", {std::move(fd)});
	}

	// --- tabfm_f1 / anofox_tabfm_f1 (CMET-02) ---
	{
		AggregateFunctionSet set("anofox_tabfm_f1");
		AggregateFunction fn3("anofox_tabfm_f1",
		                      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
		                      LogicalType::DOUBLE, F1StateSize, F1StateInit, F1Update, F1Combine,
		                      F1Finalize, /*simple_update=*/nullptr, F1ScoreBind, F1StateDestroy);
		set.AddFunction(fn3);
		AggregateFunction fn2(
		    "anofox_tabfm_f1", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
		    F1StateSize, F1StateInit, F1Update, F1Combine, F1Finalize,
		    /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_f1: 'avg' is required — pass avg := 'micro', 'macro', or 'weighted'");
			    return nullptr;
		    },
		    F1StateDestroy);
		set.AddFunction(fn2);
		FunctionDescription fd;
		fd.description =
		    "Compute F1-score with the specified averaging mode. avg is REQUIRED: pass 'macro', 'micro', "
		    "or 'weighted'. Undefined per-class precision/recall returns 0.0 (zero_division=0). "
		    "NULL rows skipped. Matches sklearn.metrics.f1_score(average=avg, zero_division=0).";
		fd.examples = {"SELECT tabfm_f1(actual, predicted, 'macro') FROM predictions;",
		               "SELECT tabfm_f1(actual, predicted, 'weighted') FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_f1", {std::move(fd)});
	}

	// --- tabfm_log_loss / anofox_tabfm_log_loss (CMET-03) ---
	{
		AggregateFunctionSet set("anofox_tabfm_log_loss");
		AggregateFunction fn("anofox_tabfm_log_loss",
		                     {LogicalType::VARCHAR,
		                      LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)},
		                     LogicalType::DOUBLE, LogLossStateSize, LogLossStateInit, LogLossUpdate,
		                     LogLossCombine, LogLossFinalize, /*simple_update=*/nullptr, LogLossBind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute log-loss (cross-entropy) from a per-class probability MAP(VARCHAR,DOUBLE). "
		    "Probabilities are clipped to [1e-15, 1-1e-15] (sklearn eps default) to avoid infinity. "
		    "Missing MAP keys are treated as p=0 then clipped. NULL rows skipped. "
		    "Matches sklearn.metrics.log_loss(normalize=True, eps=1e-15).";
		fd.examples = {"SELECT tabfm_log_loss(actual, proba) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_log_loss", {std::move(fd)});
	}

	// --- tabfm_roc_auc / anofox_tabfm_roc_auc (CMET-04) ---
	// Two overloads: 3-arg (actual, proba, avg) and 2-arg (always throws)
	{
		AggregateFunctionSet set("anofox_tabfm_roc_auc");
		AggregateFunction fn3("anofox_tabfm_roc_auc",
		                      {LogicalType::VARCHAR,
		                       LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE),
		                       LogicalType::VARCHAR},
		                      LogicalType::DOUBLE, AUCStateSize, AUCStateInit, AUCUpdate, AUCCombine,
		                      AUCFinalize, /*simple_update=*/nullptr, AUCBind, AUCStateDestroy);
		set.AddFunction(fn3);
		AggregateFunction fn2(
		    "anofox_tabfm_roc_auc",
		    {LogicalType::VARCHAR, LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)},
		    LogicalType::DOUBLE, AUCStateSize, AUCStateInit, AUCUpdate, AUCCombine, AUCFinalize,
		    /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_roc_auc: 'avg' is required — pass avg := 'ovr' or 'ovo'");
			    return nullptr;
		    },
		    AUCStateDestroy);
		set.AddFunction(fn2);
		FunctionDescription fd;
		fd.description =
		    "Compute ROC-AUC with multiclass support and correct rank-sum tie handling. "
		    "avg is REQUIRED: 'ovr' (one-vs-rest, macro average) or 'ovo' (one-vs-one, macro average). "
		    "Tied scores handled via the Mann-Whitney U rank-sum form (no optimistic bias). "
		    "NULL rows skipped. Matches sklearn.metrics.roc_auc_score(multi_class='ovr'/'ovo').";
		fd.examples = {"SELECT tabfm_roc_auc(actual, proba, 'ovr') FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_roc_auc", {std::move(fd)});
	}

	// --- tabfm_ece / anofox_tabfm_ece (CMET-06) ---
	{
		AggregateFunctionSet set("anofox_tabfm_ece");
		AggregateFunction fn("anofox_tabfm_ece",
		                     {LogicalType::VARCHAR,
		                      LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)},
		                     LogicalType::DOUBLE, ECEStateSize, ECEStateInit, ECEUpdate, ECECombine,
		                     ECEFinalize, /*simple_update=*/nullptr, ECEBind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute Expected Calibration Error (ECE) using 10 equal-width bins over [0,1]. "
		    "For each row, extracts argmax class and max probability from the proba MAP. "
		    "ECE = sum_m (|B_m|/N)*|accuracy(B_m) - confidence(B_m)|. NULL rows skipped. "
		    "Reference: Guo et al. 2017 (On Calibration of Modern Neural Networks).";
		fd.examples = {"SELECT tabfm_ece(actual, proba) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_ece", {std::move(fd)});
	}

	// --- tabfm_confusion_matrix / anofox_tabfm_confusion_matrix (CMET-05) ---
	// TABLE MACRO wrapping a GROUP BY query with safely-quoted column identifiers
	// (T-01-02-01 mitigation: replace(col, '"', '""') before interpolation).
	{
		auto primary = BuildConfusionMacroInfo("anofox_tabfm_confusion_matrix", string());
		loader.RegisterFunction(*primary);
		auto alias =
		    BuildConfusionMacroInfo("tabfm_confusion_matrix", "anofox_tabfm_confusion_matrix");
		loader.RegisterFunction(*alias);
	}
}

} // namespace anofox
} // namespace duckdb
