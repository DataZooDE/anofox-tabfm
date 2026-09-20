//===----------------------------------------------------------------------===//
// tabfm_metrics_regression.cpp — WS-G regression metric aggregates
//
// Implements:
//   anofox_tabfm_rmse / tabfm_rmse     (RMET-01)
//   anofox_tabfm_mae / tabfm_mae       (RMET-02)
//   anofox_tabfm_r2 / tabfm_r2         (RMET-03)
//   anofox_tabfm_mape / tabfm_mape     (RMET-04)
//   anofox_tabfm_medae / tabfm_medae   (RMET-04)
//
// Aggregate triplet pattern mirrors tabfm_predict_agg.cpp and
// tabfm_metrics_classification.cpp (same state-size / init / update /
// combine / finalize layout, UnifiedVectorFormat NULL-skip,
// telemetry-at-bind convention).
//
// Inputs are DOUBLE (DuckDB implicit numeric cast applies).
// NULL semantics: rows where actual OR predicted is NULL are skipped.
// Empty / all-NULL input returns NULL (standard SQL aggregate NULL-on-empty).
//
// Spec ref: RMET-01..04, SQL-API §4 metric surface
// Research ref: 01-RESEARCH.md §Metric Formulas, 01-PATTERNS.md
// Context ref: 01-CONTEXT.md (R² constant-target convention, MAPE zero-actual skip)
//===----------------------------------------------------------------------===//

#include "tabfm_metrics_regression.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox {

namespace {

//===----------------------------------------------------------------------===//
// tabfm_rmse — RMET-01
//
// Root-mean-squared error over (actual DOUBLE, predicted DOUBLE) pairs.
// State: {double sum_sq_residuals; int64_t n}
// Finalize: sqrt(sum_sq / n)
// NULL semantics: rows where actual OR predicted is NULL are skipped.
// Empty / all-NULL input → NULL.
// sklearn ref: sklearn.metrics.root_mean_squared_error
//===----------------------------------------------------------------------===//

struct RMSEState {
	double  sum_sq; // sum of (predicted - actual)^2
	int64_t n;      // valid (non-NULL) pairs seen
};

idx_t RMSEStateSize(const AggregateFunction &) {
	return sizeof(RMSEState);
}

void RMSEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) RMSEState{0.0, 0};
}

// Update — UnifiedVectorFormat NULL-skip (01-PATTERNS.md "NULL skip in Update")
// Signature: (inputs, aggr_input_data, input_count [discarded], states, row_count)
void RMSEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto states = reinterpret_cast<RMSEState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		// NULL skip: matches standard SQL aggregate semantics (CONTEXT.md)
		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		auto &state      = *states[sidx];
		double a         = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];
		double p         = UnifiedVectorFormat::GetData<double>(predicted_data)[pidx];
		double residual  = p - a;
		state.sum_sq    += residual * residual;
		state.n++;
	}
}

void RMSECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<RMSEState **>(source_data.data);
	auto targets = reinterpret_cast<RMSEState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src  = *sources[source_data.sel->get_index(i)];
		auto &tgt  = *targets[target_data.sel->get_index(i)];
		tgt.sum_sq += src.sum_sq;
		tgt.n      += src.n;
	}
}

// Finalize — sqrt(sum_sq/n); SetNull when n==0 (T-01-03-01: division-by-zero guard)
void RMSEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                  idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<RMSEState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    std::sqrt(state.sum_sq / static_cast<double>(state.n));
		}
	}
}

unique_ptr<FunctionData> RMSEBind(ClientContext &, AggregateFunction &,
                                   vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_rmse");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_mae — RMET-02
//
// Mean absolute error over (actual DOUBLE, predicted DOUBLE) pairs.
// State: {double sum_abs; int64_t n}
// Finalize: sum_abs / n
// NULL semantics: rows where actual OR predicted is NULL are skipped.
// Empty / all-NULL input → NULL.
// sklearn ref: sklearn.metrics.mean_absolute_error
//===----------------------------------------------------------------------===//

struct MAEState {
	double  sum_abs; // sum of |predicted - actual|
	int64_t n;       // valid (non-NULL) pairs seen
};

idx_t MAEStateSize(const AggregateFunction &) {
	return sizeof(MAEState);
}

void MAEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) MAEState{0.0, 0};
}

void MAEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto states = reinterpret_cast<MAEState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		auto &state       = *states[sidx];
		double a          = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];
		double p          = UnifiedVectorFormat::GetData<double>(predicted_data)[pidx];
		state.sum_abs    += std::abs(p - a);
		state.n++;
	}
}

void MAECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<MAEState **>(source_data.data);
	auto targets = reinterpret_cast<MAEState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src     = *sources[source_data.sel->get_index(i)];
		auto &tgt     = *targets[target_data.sel->get_index(i)];
		tgt.sum_abs  += src.sum_abs;
		tgt.n        += src.n;
	}
}

void MAEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                 idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<MAEState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_abs / static_cast<double>(state.n);
		}
	}
}

unique_ptr<FunctionData> MAEBind(ClientContext &, AggregateFunction &,
                                  vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_mae");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_r2 — RMET-03
//
// Coefficient of determination R² = 1 - SS_res / SS_tot.
//
// Online state uses the computational form:
//   SS_tot = sum_y2 - n * y_bar^2  (Welford-free; adequate for realistic N)
//
// Constant-target convention (D: R² constant-target, sklearn-matching):
//   When |SS_tot| < 1e-12 (target variance effectively zero):
//     - SS_res == 0 (perfect prediction) → 1.0
//     - SS_res > 0 (imperfect prediction) → 0.0
//   Never returns NaN or Inf on constant-target data (T-01-03-01).
//
// NULL semantics: rows where actual OR predicted is NULL are skipped.
// Empty / all-NULL input → NULL.
// sklearn ref: sklearn.metrics.r2_score (see D: constant-target convention)
//===----------------------------------------------------------------------===//

struct R2State {
	double  sum_y;    // sum of actual
	double  sum_y2;   // sum of actual^2
	double  sum_res;  // sum of (actual - predicted)^2
	int64_t n;        // valid (non-NULL) pairs seen
};

idx_t R2StateSize(const AggregateFunction &) {
	return sizeof(R2State);
}

void R2StateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) R2State{0.0, 0.0, 0.0, 0};
}

void R2Update(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto states = reinterpret_cast<R2State **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		auto &state   = *states[sidx];
		double a      = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];
		double p      = UnifiedVectorFormat::GetData<double>(predicted_data)[pidx];
		double resid  = a - p;
		state.sum_y  += a;
		state.sum_y2 += a * a;
		state.sum_res += resid * resid;
		state.n++;
	}
}

void R2Combine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<R2State **>(source_data.data);
	auto targets = reinterpret_cast<R2State **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src      = *sources[source_data.sel->get_index(i)];
		auto &tgt      = *targets[target_data.sel->get_index(i)];
		tgt.sum_y     += src.sum_y;
		tgt.sum_y2    += src.sum_y2;
		tgt.sum_res   += src.sum_res;
		tgt.n         += src.n;
	}
}

// Finalize — R² = 1 - SS_res/SS_tot.
// Guards SS_tot == 0: returns 1.0 (perfect) or 0.0 (imperfect) per sklearn convention.
// Never divides by zero (Pitfall 3 / T-01-03-01).
void R2Finalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<R2State **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
			continue;
		}
		double n      = static_cast<double>(state.n);
		double y_bar  = state.sum_y / n;
		// Computational form: SS_tot = Σy² - n·ȳ²
		double ss_tot = state.sum_y2 - n * y_bar * y_bar;
		double ss_res = state.sum_res;
		double r2;
		if (std::abs(ss_tot) < 1e-12) {
			// Constant-target convention (sklearn): return 1.0 iff perfect prediction,
			// 0.0 otherwise. Never NaN or Inf (D: R² constant-target).
			r2 = (ss_res < 1e-12) ? 1.0 : 0.0;
		} else {
			r2 = 1.0 - ss_res / ss_tot;
		}
		FlatVector::GetData<double>(result)[i + offset] = r2;
	}
}

unique_ptr<FunctionData> R2Bind(ClientContext &, AggregateFunction &,
                                 vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_r2");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_mape — RMET-04 (zero-actual safe)
//
// Mean absolute percentage error, returned as a DIMENSIONLESS RATIO
// (matches sklearn.metrics.mean_absolute_percentage_error which returns
// a fraction, not a percentage; e.g. 0.076667 = ~7.67% error).
//
// Zero-actual skip: rows where actual == 0 are SKIPPED to avoid division
// by zero (documented behavior, Pitfall on MAPE). The skipped rows are
// neither counted in the numerator nor the denominator.
//
// All-zero-actual input → NULL (no rows contribute, n remains 0).
// NULL actual or predicted → skipped (standard NULL semantics).
// Empty / all-NULL input → NULL.
// sklearn ref: sklearn.metrics.mean_absolute_percentage_error (non-zero rows)
//===----------------------------------------------------------------------===//

struct MAPEState {
	double  sum_ape; // sum of |predicted - actual| / |actual|  (zero-actual rows excluded)
	int64_t n;       // contributing rows (actual != 0 AND non-NULL)
};

idx_t MAPEStateSize(const AggregateFunction &) {
	return sizeof(MAPEState);
}

void MAPEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) MAPEState{0.0, 0};
}

void MAPEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto states = reinterpret_cast<MAPEState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		// Standard NULL skip
		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		double a = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];
		double p = UnifiedVectorFormat::GetData<double>(predicted_data)[pidx];

		// Zero-actual skip: documented behavior; never divides by zero (T-01-03-01).
		// Row is silently excluded from both numerator and denominator.
		if (a == 0.0) {
			continue;
		}

		auto &state     = *states[sidx];
		state.sum_ape  += std::abs(p - a) / std::abs(a);
		state.n++;
	}
}

void MAPECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<MAPEState **>(source_data.data);
	auto targets = reinterpret_cast<MAPEState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src     = *sources[source_data.sel->get_index(i)];
		auto &tgt     = *targets[target_data.sel->get_index(i)];
		tgt.sum_ape  += src.sum_ape;
		tgt.n        += src.n;
	}
}

void MAPEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                  idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<MAPEState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			// No contributing rows (all-NULL, empty, or all-zero-actual)
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_ape / static_cast<double>(state.n);
		}
	}
}

unique_ptr<FunctionData> MAPEBind(ClientContext &, AggregateFunction &,
                                   vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_mape");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_medae — RMET-04 (heap-owning vector state)
//
// Median absolute error: median of |actual - predicted| over all valid pairs.
//
// State: heap-owning vector<double> of absolute residuals (O(N) growth,
// documented). StateDestroy frees the vector (01-PATTERNS.md StateDestroy pattern).
// Update appends |actual - predicted|; Combine merges the two vectors.
// Finalize sorts (std::sort) and returns the middle element for odd N, or the
// mean of the two middle elements for even N.
//
// NULL semantics: rows where actual OR predicted is NULL are skipped.
// Empty / all-NULL input → NULL.
// sklearn ref: sklearn.metrics.median_absolute_error
//===----------------------------------------------------------------------===//

// Fixed-size slot holding pointer to heap-allocated vector.
// NOTE: O(N) memory growth — documented (T-01-03-03 accepted risk).
struct MedAEStateSlot {
	std::vector<double> *data; // pointer to heap-allocated residuals vector
};

idx_t MedAEStateSize(const AggregateFunction &) {
	return sizeof(MedAEStateSlot);
}

void MedAEStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	auto &slot = *reinterpret_cast<MedAEStateSlot *>(state_ptr);
	slot.data  = nullptr;
}

// StateDestroy: free the heap-allocated vector (01-PATTERNS.md StateDestroy pattern)
void MedAEStateDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto slots = reinterpret_cast<MedAEStateSlot **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &slot = *slots[sdata.sel->get_index(i)];
		delete slot.data;
		slot.data = nullptr;
	}
}

void MedAEUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector,
                 idx_t count) {
	UnifiedVectorFormat sdata, actual_data, predicted_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	inputs[1].ToUnifiedFormat(count, predicted_data);
	auto slots = reinterpret_cast<MedAEStateSlot **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);
		idx_t pidx = predicted_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx) || !predicted_data.validity.RowIsValid(pidx)) {
			continue;
		}

		auto &slot = *slots[sidx];
		if (!slot.data) {
			slot.data = new std::vector<double>();
		}

		double a = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];
		double p = UnifiedVectorFormat::GetData<double>(predicted_data)[pidx];
		slot.data->push_back(std::abs(a - p));
	}
}

void MedAECombine(Vector &source_vector, Vector &target_vector, AggregateInputData &,
                  idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<MedAEStateSlot **>(source_data.data);
	auto targets = reinterpret_cast<MedAEStateSlot **>(target_data.data);

	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];

		if (!src.data || src.data->empty()) {
			continue; // nothing to merge from source
		}
		if (!tgt.data) {
			tgt.data = new std::vector<double>();
		}
		tgt.data->insert(tgt.data->end(), src.data->begin(), src.data->end());
	}
}

// Finalize: sort residuals, return median (middle for odd N, mean of two middles for even N).
void MedAEFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                   idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto slots = reinterpret_cast<MedAEStateSlot **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &slot = *slots[sdata.sel->get_index(i)];
		if (!slot.data || slot.data->empty()) {
			FlatVector::SetNull(result, i + offset, true);
			continue;
		}
		auto &residuals = *slot.data;
		std::sort(residuals.begin(), residuals.end());
		idx_t n = static_cast<idx_t>(residuals.size());
		double median;
		if (n % 2 == 1) {
			// Odd N: middle element
			median = residuals[n / 2];
		} else {
			// Even N: mean of two middle elements
			median = (residuals[n / 2 - 1] + residuals[n / 2]) / 2.0;
		}
		FlatVector::GetData<double>(result)[i + offset] = median;
	}
}

unique_ptr<FunctionData> MedAEBind(ClientContext &, AggregateFunction &,
                                    vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_medae");
	return nullptr;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void RegisterRegressionMetrics(ExtensionLoader &loader) {
	// --- tabfm_rmse / anofox_tabfm_rmse (RMET-01) ---
	{
		AggregateFunctionSet set("anofox_tabfm_rmse");
		AggregateFunction fn("anofox_tabfm_rmse",
		                     {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		                     RMSEStateSize, RMSEStateInit, RMSEUpdate, RMSECombine, RMSEFinalize,
		                     /*simple_update=*/nullptr, RMSEBind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute root-mean-squared error (RMSE) over (actual, predicted) pairs. "
		    "Rows where actual OR predicted is NULL are skipped (SQL aggregate NULL semantics). "
		    "Returns NULL on empty or all-NULL input. "
		    "Matches sklearn.metrics.root_mean_squared_error.";
		fd.examples = {
		    "SELECT tabfm_rmse(actual, predicted) FROM predictions;",
		    "SELECT round(tabfm_rmse(actual, predicted), 4) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_rmse", {std::move(fd)});
	}

	// --- tabfm_mae / anofox_tabfm_mae (RMET-02) ---
	{
		AggregateFunctionSet set("anofox_tabfm_mae");
		AggregateFunction fn("anofox_tabfm_mae",
		                     {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		                     MAEStateSize, MAEStateInit, MAEUpdate, MAECombine, MAEFinalize,
		                     /*simple_update=*/nullptr, MAEBind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute mean absolute error (MAE) over (actual, predicted) pairs. "
		    "Rows where actual OR predicted is NULL are skipped (SQL aggregate NULL semantics). "
		    "Returns NULL on empty or all-NULL input. "
		    "Matches sklearn.metrics.mean_absolute_error.";
		fd.examples = {
		    "SELECT tabfm_mae(actual, predicted) FROM predictions;",
		    "SELECT round(tabfm_mae(actual, predicted), 4) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_mae", {std::move(fd)});
	}

	// --- tabfm_r2 / anofox_tabfm_r2 (RMET-03) ---
	// Constant-target convention (sklearn): |SS_tot| < 1e-12 → 1.0 (perfect) or 0.0 (imperfect).
	// Never returns NaN or Inf (T-01-03-01).
	{
		AggregateFunctionSet set("anofox_tabfm_r2");
		AggregateFunction fn("anofox_tabfm_r2",
		                     {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		                     R2StateSize, R2StateInit, R2Update, R2Combine, R2Finalize,
		                     /*simple_update=*/nullptr, R2Bind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute the coefficient of determination R² = 1 - SS_res/SS_tot over (actual, predicted) pairs. "
		    "Constant-target convention (matches sklearn): when the target variance is 0, returns 1.0 for a "
		    "perfect prediction and 0.0 otherwise — never NaN or Inf. "
		    "Rows where actual OR predicted is NULL are skipped. "
		    "Returns NULL on empty or all-NULL input. "
		    "Matches sklearn.metrics.r2_score.";
		fd.examples = {
		    "SELECT tabfm_r2(actual, predicted) FROM predictions;",
		    "SELECT round(tabfm_r2(actual, predicted), 4) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_r2", {std::move(fd)});
	}

	// --- tabfm_mape / anofox_tabfm_mape (RMET-04) ---
	// Returns a DIMENSIONLESS RATIO (not a percentage), matching sklearn.
	// Zero-actual rows are SKIPPED (documented; avoids division by zero, T-01-03-01).
	// All-zero-actual input → NULL.
	{
		AggregateFunctionSet set("anofox_tabfm_mape");
		AggregateFunction fn("anofox_tabfm_mape",
		                     {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		                     MAPEStateSize, MAPEStateInit, MAPEUpdate, MAPECombine, MAPEFinalize,
		                     /*simple_update=*/nullptr, MAPEBind,
		                     /*state_destroy=*/nullptr);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute mean absolute percentage error (MAPE) as a DIMENSIONLESS RATIO over (actual, predicted) "
		    "pairs. Rows where actual == 0 are SKIPPED to avoid division by zero (documented behavior); "
		    "returns NULL when all actuals are zero. Rows where actual OR predicted is NULL are also skipped. "
		    "Returns NULL on empty or all-NULL input. "
		    "Matches sklearn.metrics.mean_absolute_percentage_error (fraction, not percent).";
		fd.examples = {
		    "SELECT tabfm_mape(actual, predicted) FROM predictions;",
		    "SELECT round(tabfm_mape(actual, predicted) * 100, 2) || '%' FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_mape", {std::move(fd)});
	}

	// --- tabfm_medae / anofox_tabfm_medae (RMET-04) ---
	// Heap-owning vector<double> state; O(N) memory growth (T-01-03-03 accepted risk).
	// StateDestroy frees the vector (01-PATTERNS.md StateDestroy pattern).
	{
		AggregateFunctionSet set("anofox_tabfm_medae");
		AggregateFunction fn("anofox_tabfm_medae",
		                     {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		                     MedAEStateSize, MedAEStateInit, MedAEUpdate, MedAECombine, MedAEFinalize,
		                     /*simple_update=*/nullptr, MedAEBind, MedAEStateDestroy);
		set.AddFunction(fn);
		FunctionDescription fd;
		fd.description =
		    "Compute median absolute error (MedAE) over (actual, predicted) pairs. "
		    "Accumulates |actual - predicted| residuals in a heap-allocated vector (O(N) memory). "
		    "For even N, returns the mean of the two middle residuals. "
		    "Rows where actual OR predicted is NULL are skipped. "
		    "Returns NULL on empty or all-NULL input. "
		    "Matches sklearn.metrics.median_absolute_error.";
		fd.examples = {
		    "SELECT tabfm_medae(actual, predicted) FROM predictions;",
		    "SELECT round(tabfm_medae(actual, predicted), 4) FROM predictions;"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_medae", {std::move(fd)});
	}
}

} // namespace anofox
} // namespace duckdb
