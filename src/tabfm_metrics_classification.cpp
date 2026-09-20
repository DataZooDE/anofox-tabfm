//===----------------------------------------------------------------------===//
// tabfm_metrics_classification.cpp — WS-G classification metric aggregates
//
// Implements: anofox_tabfm_accuracy / tabfm_accuracy (CMET-01)
//             Further metrics (F1, log-loss, ROC-AUC, ECE, confusion matrix)
//             added by plan 01-02.
//
// Aggregate triplet pattern mirrors tabfm_predict_agg.cpp (same state-size /
// init / update / combine / finalize layout, UnifiedVectorFormat NULL-skip,
// telemetry-at-bind convention).
//
// Spec ref: CMET-01..06, SQL-API §4 metric surface
//===----------------------------------------------------------------------===//

#include "tabfm_metrics_classification.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/common/vector_operations/general_cast.hpp"
#include "telemetry.hpp"

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
void AccuracyUpdate(Vector inputs[], AggregateInputData &, idx_t count, Vector &state_vector, idx_t) {
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

		// Compare as strings so the aggregate works on ANY type pair
		// (INTEGER vs INTEGER, VARCHAR vs VARCHAR, etc.)
		auto actual_val = inputs[0].GetValue(i);
		auto predicted_val = inputs[1].GetValue(i);
		if (actual_val.ToString() == predicted_val.ToString()) {
			state.correct++;
		}
	}
}

// Combine for parallel aggregate execution
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
// (T-01-01: division-by-zero guard; CONTEXT.md §Numerical Semantics)
void AccuracyFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                      idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<AccuracyState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.total == 0) {
			// Empty / all-NULL input → NULL (CONTEXT.md §NULL handling)
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    static_cast<double>(state.correct) / static_cast<double>(state.total);
		}
	}
}

// Bind — telemetry called ONCE here, never in Update (CLAUDE.md rule #3)
unique_ptr<FunctionData> AccuracyBind(ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_accuracy");
	return nullptr;
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
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_accuracy");
	}
	// Further classification metrics (F1, log-loss, ROC-AUC, ECE,
	// confusion matrix) registered here by plan 01-02.
}

} // namespace anofox
} // namespace duckdb
