//===----------------------------------------------------------------------===//
// tabfm_scoring.cpp — WS-scoring (proper scoring rule aggregates)
//
// Implements PSR-01 and PSR-04:
//   anofox_tabfm_crps / tabfm_crps   (PSR-01)
//
// CRPS is the Continuous Ranked Probability Score — the proper scoring rule
// for predictive distributions. For the bar/histogram distribution that
// tabfm_regress emits under output_mode='distribution', CRPS has an analytical
// closed-form: the integral of (F(t) - 1{y<=t})^2 over the finite support
// [b_0, b_K] using the piecewise-linear CDF (piecewise-uniform density per bin).
//
// PSR-04 bind-gate: a 2-arg (DOUBLE, DOUBLE) overload always throws at bind
// pointing the user at output_mode := 'distribution'.
//
// Input shape for CRPS:
//   (actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[]))
// The STRUCT is the exact type tabfm_predict_agg.cpp emits at lines 100-103.
//
// Spec ref: PSR-01, PSR-04, SQL-API §5
// Research ref: 03-RESEARCH.md §Exact CRPS Math, §CRPS C++ Implementation Skeleton,
//               §STRUCT Reading in Aggregate Update, §Bind-Gating (PSR-04) Pattern
// Context ref: 03-CONTEXT.md (locked decisions: non-uniform bins, bind-gate, aliases)
//===----------------------------------------------------------------------===//

#include "tabfm_scoring.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "telemetry.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace duckdb {
namespace anofox {

namespace {

//===----------------------------------------------------------------------===//
// SoftmaxInPlace — temperature=1.0 numerically stable softmax.
//
// Copied verbatim from tabfm_engine.cpp (anonymous-namespace, no external
// linkage). The scoring aggregate Update must have its own copy.
// See 03-RESEARCH.md §Softmax in the Aggregate Update.
//===----------------------------------------------------------------------===//
void SoftmaxInPlace(std::vector<double> &v) {
	if (v.empty())
		return;
	double m = v[0];
	for (auto x : v) {
		m = std::max(m, x);
	}
	double sum = 0.0;
	for (auto &x : v) {
		x = std::exp(x - m);
		sum += x;
	}
	if (sum > 0.0) {
		for (auto &x : v) {
			x /= sum;
		}
	}
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ComputeCRPS — analytical CRPS over a piecewise-uniform bar distribution.
//
// Given: normalized probs p_0..p_{K-1} (after softmax), borders b_0..b_K,
// observation y.
//
// CDF F(t) is piecewise-linear with F(b_i) = C_i = sum_{j<i} p_j,
// and uniform density p_i / w_i within bin i=[b_i, b_{i+1}].
//
// Per-bin contribution is the integral of (F(t) - 1{t>=y})^2 dt:
//   Case B: y > b_{i+1}  → w_i * (C_i^2   + C_i*(C_i-1)*0  + C_i*p_i + p_i^2/3)
//           (full bin to LEFT of y; indicator=0 throughout)
//   Case A: y < b_i      → w_i * ((C_i-1)^2 + (C_i-1)*p_i  + p_i^2/3)
//           (full bin to RIGHT of y; indicator=1 throughout)
//   Case C: b_i <= y <= b_{i+1} (y is INSIDE this bin; split at y):
//     Left part  [b_i, y], indicator=0:
//       L = C_i^2*d + C_i*(p_i/w_i)*d^2 + (p_i/w_i)^2*d^3/3
//     Right part [y, b_{i+1}], indicator=1:
//       a2 = F(y) - 1 = (C_i + p_i*d/w_i) - 1
//       R  = a2^2*e + a2*(p_i/w_i)*e^2 + (p_i/w_i)^2*e^3/3
//     contribution_i = L + R
//
// Assumption A1 (03-RESEARCH.md): integrates only over [b_0, b_K].
// y outside [b_0, b_K] is well-defined: all K bins contribute via Case A or B.
// Zero-width bins (w_i <= 0) are skipped (cum accumulates pi).
//
// External linkage so Catch2 unit tests can call it directly.
// (Pattern: DistributionMean/DistributionQuantile in tabfm_engine.cpp, Phase 2.)
//===----------------------------------------------------------------------===//
double ComputeCRPS(double y, const std::vector<double> &probs, const std::vector<double> &borders) {
	const size_t K = probs.size();
	// Guard: K must match borders.size() - 1
	if (K == 0 || borders.size() != K + 1) {
		return 0.0;
	}

	double crps = 0.0;
	double cum = 0.0; // C_i = cumulative probability up to left edge of bin i

	for (size_t i = 0; i < K; i++) {
		const double b_lo = borders[i];
		const double b_hi = borders[i + 1];
		const double wi = b_hi - b_lo;
		const double pi = probs[i];

		// T-03-02: guard zero/negative bin width — skip (no divide-by-zero in Case C)
		if (wi <= 0.0) {
			cum += pi;
			continue;
		}

		if (y < b_lo) {
			// Case A: y is to the left of this bin; indicator = 1 throughout
			// integral = w_i * ((C_i - 1)^2 + (C_i - 1)*p_i + p_i^2/3)
			double a = cum - 1.0;
			crps += wi * (a * a + a * pi + pi * pi / 3.0);
		} else if (y > b_hi) {
			// Case B: y is to the right of this bin; indicator = 0 throughout
			// integral = w_i * (C_i^2 + C_i*p_i + p_i^2/3)
			crps += wi * (cum * cum + cum * pi + pi * pi / 3.0);
		} else {
			// Case C: y is inside this bin [b_lo, b_hi]
			const double d = y - b_lo;  // distance from left edge (>=0)
			const double e = b_hi - y;  // distance to right edge (>=0)
			const double c = pi / wi;   // density in this bin

			// Left part [b_lo, y]: indicator = 0
			// integral of (C_i + c*s)^2 ds from 0 to d:
			//   = C_i^2*d + C_i*c*d^2 + c^2*d^3/3
			double L = cum * cum * d + cum * c * d * d + c * c * d * d * d / 3.0;

			// Right part [y, b_hi]: indicator = 1
			// F(y) = C_i + p_i * d / w_i
			double fy = cum + pi * d / wi;
			double a2 = fy - 1.0; // F(y) - 1 (negative, since F(y) < 1 inside any bin)
			double R = a2 * a2 * e + a2 * c * e * e + c * c * e * e * e / 3.0;

			crps += L + R;
		}

		cum += pi;
	}
	return crps;
}

namespace {

//===----------------------------------------------------------------------===//
// tabfm_crps — PSR-01
//
// Mean CRPS over (actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[]))
// rows. CRPS is the proper scoring rule for predictive regression distributions.
//
// State: {double sum_crps; int64_t n}
// Finalize: sum_crps / n
// NULL semantics: rows where actual OR yhat_dist is NULL are skipped;
//   malformed distributions (K==0, size mismatch) are also skipped (T-03-01).
// Empty / all-NULL input → NULL.
//
// Research ref: 03-RESEARCH.md §Exact CRPS Math
// Spec ref: PSR-01
//===----------------------------------------------------------------------===//

struct CRPSState {
	double  sum_crps; // running sum of per-row CRPS values
	int64_t n;        // non-NULL valid rows
};

idx_t CRPSStateSize(const AggregateFunction &) {
	return sizeof(CRPSState);
}

void CRPSStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) CRPSState{0.0, 0};
}

// CRPSUpdate — per-row Update.
// inputs[0] = actual DOUBLE
// inputs[1] = yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])
//
// STRUCT argument: use inputs[1].GetValue(i) idiom (handles flat/constant/dictionary
// vectors uniformly). Do NOT use UnifiedVectorFormat for the STRUCT argument.
// See 03-RESEARCH.md §STRUCT Reading in Aggregate Update.
//
// T-03-01 bounds-checks: K==0 and border_vals.size()!=K+1 skip the row.
void CRPSUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
	UnifiedVectorFormat sdata, actual_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	auto states = reinterpret_cast<CRPSState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);

		// NULL skip for actual (T-03-01)
		if (!actual_data.validity.RowIsValid(aidx)) {
			continue;
		}
		double actual_val = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];

		// STRUCT argument: GetValue(i) handles all vector representations
		Value dist_val = inputs[1].GetValue(i);
		if (dist_val.IsNull()) {
			continue; // NULL yhat_dist row — skip (T-03-01)
		}

		// Extract logits LIST and borders LIST from the STRUCT children.
		// Field order from tabfm_predict_agg.cpp:100-103:
		//   dist_children[0] = logits  LIST(DOUBLE)
		//   dist_children[1] = borders LIST(DOUBLE)
		auto &dist_children = StructValue::GetChildren(dist_val);
		if (dist_children[0].IsNull() || dist_children[1].IsNull()) {
			continue; // NULL list children — skip
		}

		auto &logit_vals  = ListValue::GetChildren(dist_children[0]);
		auto &border_vals = ListValue::GetChildren(dist_children[1]);

		const size_t K = logit_vals.size();
		// T-03-01 bounds-check: K must be > 0 and borders must be K+1
		if (K == 0 || border_vals.size() != K + 1) {
			continue; // malformed distribution — skip
		}

		// Build double vectors for logits and borders
		std::vector<double> logits(K), borders(K + 1);
		for (size_t k = 0; k < K; k++) {
			logits[k]  = DoubleValue::Get(logit_vals[k]);
			borders[k] = DoubleValue::Get(border_vals[k]);
		}
		borders[K] = DoubleValue::Get(border_vals[K]);

		// Softmax logits → per-bin probabilities (temperature=1.0 per bar-distribution contract)
		SoftmaxInPlace(logits);

		auto &state = *states[sidx];
		state.sum_crps += ComputeCRPS(actual_val, logits, borders);
		state.n++;
	}
}

void CRPSCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<CRPSState **>(source_data.data);
	auto targets = reinterpret_cast<CRPSState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.sum_crps += src.sum_crps;
		tgt.n        += src.n;
	}
}

// CRPSFinalize — mean CRPS; NULL when n==0 (empty/all-NULL group)
void CRPSFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                  idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<CRPSState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_crps / static_cast<double>(state.n);
		}
	}
}

unique_ptr<FunctionData> CRPSBind(ClientContext &, AggregateFunction &,
                                   vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_crps");
	return nullptr;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// RegisterScoringFunctions — entry point called from LoadInternal
//===----------------------------------------------------------------------===//
void RegisterScoringFunctions(ExtensionLoader &loader) {
	// --- tabfm_crps / anofox_tabfm_crps (PSR-01, PSR-04) ---
	//
	// Two overloads in registration order (pitfall 6: STRUCT overload FIRST):
	//   1. (actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])) → DOUBLE
	//      Normal aggregate; bind emits telemetry.
	//   2. (actual DOUBLE, plain_estimate DOUBLE) → bind ALWAYS throws (PSR-04).
	//      Point-estimate guard: rejects tabfm_crps(actual, yhat) where yhat is scalar.
	{
		// yhat_dist STRUCT type — must exactly match tabfm_predict_agg.cpp:100-103
		// field order (logits first, borders second). DuckDB v1.5.4 does NOT
		// implicitly cast aggregate args; field order matters.
		LogicalType yhat_dist_type = LogicalType::STRUCT(
		    {{"logits", LogicalType::LIST(LogicalType::DOUBLE)},
		     {"borders", LogicalType::LIST(LogicalType::DOUBLE)}});

		AggregateFunctionSet set("anofox_tabfm_crps");

		// Overload 1 — STRUCT distribution argument (normal compute path)
		AggregateFunction fn_dist("anofox_tabfm_crps",
		                          {LogicalType::DOUBLE, yhat_dist_type}, LogicalType::DOUBLE,
		                          CRPSStateSize, CRPSStateInit, CRPSUpdate, CRPSCombine,
		                          CRPSFinalize, /*simple_update=*/nullptr, CRPSBind,
		                          /*state_destroy=*/nullptr);
		set.AddFunction(fn_dist);

		// Overload 2 — PSR-04 bind-gate: plain DOUBLE second argument always throws.
		// Registered AFTER fn_dist so DuckDB prefers the exact STRUCT match.
		// Error message names the remedy per SQL-API §5.
		AggregateFunction fn_point(
		    "anofox_tabfm_crps", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		    CRPSStateSize, CRPSStateInit, CRPSUpdate, CRPSCombine, CRPSFinalize,
		    /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_crps: second argument must be a distribution "
			        "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
			        "Use output_mode := 'distribution' in tabfm_regress.");
			    return nullptr;
		    },
		    /*state_destroy=*/nullptr);
		set.AddFunction(fn_point);

		FunctionDescription fd;
		fd.description =
		    "Compute the mean Continuous Ranked Probability Score (CRPS) over regression "
		    "predictions. CRPS is a proper scoring rule for predictive distributions — "
		    "lower is better. The second argument must be the yhat_dist STRUCT output of "
		    "tabfm_regress with output_mode := 'distribution'. "
		    "Non-uniform bin widths from borders[i+1]-borders[i] are used (PSR-01). "
		    "NULL rows are skipped; empty input returns NULL.";
		fd.examples = {
		    "SELECT tabfm_crps(actual, yhat_dist) FROM "
		    "tabfm_regress('tbl', 'y', opts := MAP{'output_mode':'distribution'});"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_crps", {std::move(fd)});
	}
}

} // namespace anofox
} // namespace duckdb
