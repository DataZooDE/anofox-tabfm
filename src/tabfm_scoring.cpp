//===----------------------------------------------------------------------===//
// tabfm_scoring.cpp — WS-scoring (proper scoring rule aggregates)
//
// Implements PSR-01, PSR-02, PSR-03, PSR-04:
//   anofox_tabfm_crps           / tabfm_crps            (PSR-01)
//   anofox_tabfm_log_score      / tabfm_log_score       (PSR-02)
//   anofox_tabfm_interval_score / tabfm_interval_score  (PSR-03)
//
// CRPS:           analytical closed-form over piecewise-uniform bar distribution.
// Log-score:      NLL = log(w_i) - log(max(eps, p_i)); out-of-support → max penalty.
// Interval-score: Gneiting & Raftery (2007); reuses DistributionQuantile for l/u.
//
// PSR-04 bind-gate: a 2-arg (DOUBLE, DOUBLE) overload always throws at bind
// pointing the user at output_mode := 'distribution'.
//
// Input shape for all three:
//   (actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[]))
// The STRUCT is the exact type tabfm_predict_agg.cpp emits at lines 100-103.
//
// tabfm_interval_score additionally accepts an optional 3rd DOUBLE arg `coverage`
// (default 0.9, validated in IScoreBind via IScoreBindData).
//
// Spec ref: PSR-01, PSR-02, PSR-03, PSR-04, SQL-API §5
// Research ref: 03-RESEARCH.md §Exact CRPS Math, §Log-Score Math,
//               §Interval Score Math, §STRUCT Reading in Aggregate Update,
//               §Bind-Gating (PSR-04) Pattern, §Aggregate State Design
// Context ref: 03-CONTEXT.md (locked decisions: non-uniform bins, bind-gate, aliases)
//===----------------------------------------------------------------------===//

#include "tabfm_scoring.hpp"
#include "anofox_function_alias.hpp"
#include "tabfm_registration.hpp"
#include "tabfm_predict.hpp" // for DistributionQuantile (external linkage, tabfm_predict.hpp:206)

#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/execution/expression_executor.hpp"
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

//===----------------------------------------------------------------------===//
// ComputeLogScore — NLL of y under a piecewise-uniform bar distribution.
//
// For y in bin i: log_score = log(w_i) - log(max(clip_eps, p_i))
// where clip_eps = 1e-10 (Assumption A2, 03-RESEARCH.md §Log-Score Math —
// avoids log(0) for near-zero bins; documented as a user-visible epsilon choice).
//
// Out-of-support: y < borders[0] or y > borders[K] → max penalty -log(clip_eps).
// Zero-width bin (w_i < 1e-300): → max penalty (T-03-04 mitigation).
//
// External linkage for Catch2 (same decision as ComputeCRPS, PSR-01-external-linkage).
// Spec ref: PSR-02
//===----------------------------------------------------------------------===//
double ComputeLogScore(double y, const std::vector<double> &probs,
                       const std::vector<double> &borders) {
	// Assumption A2: clip epsilon for log-score (1e-10). This value is intentionally
	// larger than typical classification log-loss epsilons (1e-15) because bar
	// distributions can have genuinely small mass in wide outer bins.
	static constexpr double kClipEps = 1e-10;
	static const double     kMaxPenalty = -std::log(kClipEps); // 23.025...

	const size_t K = probs.size();
	if (K == 0 || borders.size() != K + 1) {
		return kMaxPenalty; // malformed input → max penalty
	}

	// T-03-04: Pitfall 7 — y outside [b_0, b_K] → max penalty (no bin contains y)
	if (y < borders[0] || y > borders[K]) {
		return kMaxPenalty;
	}

	// Find the bin containing y and compute log-score
	for (size_t i = 0; i < K; i++) {
		const double b_lo = borders[i];
		const double b_hi = borders[i + 1];
		if (b_lo <= y && y <= b_hi) {
			const double wi = b_hi - b_lo;
			// T-03-04: Pitfall 4 — zero-width bin guard
			if (wi < 1e-300) {
				return kMaxPenalty;
			}
			const double p_clipped = std::max(kClipEps, probs[i]);
			return std::log(wi) - std::log(p_clipped);
		}
	}

	// This fallback is unreachable (IN-01): the loop covers all bins including the
	// last one (b_lo = borders[K-1], b_hi = borders[K]), so y == borders[K] is
	// caught by the last iteration. The only exit-without-return path is when every
	// bin has zero width (wi < 1e-300), but in that case b_lo == b_hi == y for every
	// bin and the zero-width guard fires first, returning kMaxPenalty. Mark as
	// unreachable for debugging rather than silently returning kMaxPenalty.
	D_ASSERT(false && "ComputeLogScore: unreachable post-loop fallback");
	return kMaxPenalty;
}

//===----------------------------------------------------------------------===//
// ComputeIntervalScore — Gneiting & Raftery (2007) interval score.
//
// IS = (u - l) + (2/alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]
// alpha = 1 - coverage
// l = DistributionQuantile(probs, borders, alpha/2)
// u = DistributionQuantile(probs, borders, 1 - alpha/2)
//
// DistributionQuantile reused from tabfm_predict.hpp:206 (external linkage).
// Validated pre-condition: 0 < coverage < 1 (enforced at bind by IScoreBind).
//
// External linkage for Catch2.
// Spec ref: PSR-03, 03-RESEARCH.md §Interval Score Math
//===----------------------------------------------------------------------===//
double ComputeIntervalScore(double y, const std::vector<double> &probs,
                             const std::vector<double> &borders, double coverage) {
	const double alpha = 1.0 - coverage;
	const double q_lo  = alpha / 2.0;
	const double q_hi  = 1.0 - alpha / 2.0;

	// DistributionQuantile (tabfm_predict.hpp:206, external linkage) takes
	// duckdb::vector<double> (the DuckDB typedef), not std::vector<double>.
	// Convert via construction — copy overhead is negligible for K<=1024.
	duckdb::vector<double> dv_probs(probs.begin(), probs.end());
	duckdb::vector<double> dv_borders(borders.begin(), borders.end());

	const double l = DistributionQuantile(dv_probs, dv_borders, q_lo);
	const double u = DistributionQuantile(dv_probs, dv_borders, q_hi);

	const double penalty_lo = (y < l) ? (l - y) : 0.0;
	const double penalty_hi = (y > u) ? (y - u) : 0.0;

	return (u - l) + (2.0 / alpha) * (penalty_lo + penalty_hi);
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

		// Build double vectors for logits and borders.
		// WR-02: check per-element nullability — DoubleValue::Get returns 0.0 for NULL
		// elements without raising an error (GetValueUnsafe bypasses the null flag).
		// Skip the entire row if any element is NULL, consistent with the aggregate
		// NULL-skip policy for the top-level actual and dist arguments.
		std::vector<double> logits(K), borders(K + 1);
		bool any_null = false;
		for (size_t k = 0; k < K; k++) {
			if (logit_vals[k].IsNull() || border_vals[k].IsNull()) {
				any_null = true;
				break;
			}
			logits[k]  = DoubleValue::Get(logit_vals[k]);
			borders[k] = DoubleValue::Get(border_vals[k]);
		}
		if (!any_null && border_vals[K].IsNull()) {
			any_null = true;
		}
		if (any_null) {
			continue; // NULL element in logits/borders — skip row
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

//===----------------------------------------------------------------------===//
// tabfm_log_score — PSR-02
//
// Mean negative log-likelihood (NLL) over (actual DOUBLE, yhat_dist STRUCT(…)) rows.
// Log-score is the proper scoring rule for distributional sharpness.
//
// State: {double sum_ls; int64_t n}
// Finalize: sum_ls / n  (NULL when n==0)
// NULL semantics: same as CRPS (skips NULL actual or NULL yhat_dist).
//
// PSR-04 extended: plain DOUBLE second argument throws at bind.
// Research ref: 03-RESEARCH.md §Log-Score Math
// Spec ref: PSR-02
//===----------------------------------------------------------------------===//

struct LogScoreState {
	double  sum_ls; // running sum of per-row NLL values
	int64_t n;      // non-NULL valid rows
};

idx_t LogScoreStateSize(const AggregateFunction &) {
	return sizeof(LogScoreState);
}

void LogScoreStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) LogScoreState{0.0, 0};
}

// LogScoreUpdate — same STRUCT-read + SoftmaxInPlace pattern as CRPSUpdate.
// inputs[0] = actual DOUBLE, inputs[1] = yhat_dist STRUCT(logits, borders)
void LogScoreUpdate(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector,
                    idx_t count) {
	UnifiedVectorFormat sdata, actual_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	auto states = reinterpret_cast<LogScoreState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx)) {
			continue;
		}
		double actual_val = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];

		Value dist_val = inputs[1].GetValue(i);
		if (dist_val.IsNull()) {
			continue;
		}

		auto &dist_children = StructValue::GetChildren(dist_val);
		if (dist_children[0].IsNull() || dist_children[1].IsNull()) {
			continue;
		}

		auto  &logit_vals  = ListValue::GetChildren(dist_children[0]);
		auto  &border_vals = ListValue::GetChildren(dist_children[1]);
		const size_t K     = logit_vals.size();
		if (K == 0 || border_vals.size() != K + 1) {
			continue;
		}

		// WR-02: per-element null check (see CRPSUpdate for rationale)
		std::vector<double> logits(K), borders(K + 1);
		bool any_null = false;
		for (size_t k = 0; k < K; k++) {
			if (logit_vals[k].IsNull() || border_vals[k].IsNull()) {
				any_null = true;
				break;
			}
			logits[k]  = DoubleValue::Get(logit_vals[k]);
			borders[k] = DoubleValue::Get(border_vals[k]);
		}
		if (!any_null && border_vals[K].IsNull()) {
			any_null = true;
		}
		if (any_null) {
			continue;
		}
		borders[K] = DoubleValue::Get(border_vals[K]);

		SoftmaxInPlace(logits);

		auto &state = *states[sidx];
		state.sum_ls += ComputeLogScore(actual_val, logits, borders);
		state.n++;
	}
}

void LogScoreCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &,
                     idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<LogScoreState **>(source_data.data);
	auto targets = reinterpret_cast<LogScoreState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.sum_ls += src.sum_ls;
		tgt.n      += src.n;
	}
}

void LogScoreFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                      idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<LogScoreState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_ls / static_cast<double>(state.n);
		}
	}
}

unique_ptr<FunctionData> LogScoreBind(ClientContext &, AggregateFunction &,
                                       vector<unique_ptr<Expression>> &) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_log_score");
	return nullptr;
}

//===----------------------------------------------------------------------===//
// tabfm_interval_score — PSR-03
//
// Mean interval score at a configurable coverage level.
// IS = (u-l) + (2/alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]
// Lower/upper quantiles l, u computed via DistributionQuantile (tabfm_predict.hpp:206).
//
// Coverage parameter: named `coverage`, default 0.9, range (0,1) exclusive.
// Captured in IScoreBindData at bind time (Pitfall 5: Update has no ClientContext).
// Validated at bind: throw InvalidInputException if coverage <= 0 or >= 1.
//
// State: {double sum_is; int64_t n}
// Finalize: sum_is / n  (NULL when n==0)
// NULL semantics: same as CRPS.
//
// PSR-04 extended: plain DOUBLE second argument throws at bind.
// Research ref: 03-RESEARCH.md §Interval Score Math, §Aggregate State Design
// Spec ref: PSR-03
//===----------------------------------------------------------------------===//

/// IScoreBindData — stores coverage resolved at bind time (Pitfall 5 guard).
/// Pattern: F1BindData in tabfm_metrics_classification.cpp:205-219.
struct IScoreBindData : FunctionData {
	double coverage;

	explicit IScoreBindData(double cov) : coverage(cov) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<IScoreBindData>(coverage);
	}

	bool Equals(const FunctionData &other) const override {
		return coverage == other.Cast<IScoreBindData>().coverage;
	}
};

struct IScoreState {
	double  sum_is; // running sum of per-row interval scores
	int64_t n;      // non-NULL valid rows
};

idx_t IScoreStateSize(const AggregateFunction &) {
	return sizeof(IScoreState);
}

void IScoreStateInit(const AggregateFunction &, data_ptr_t state_ptr) {
	new (state_ptr) IScoreState{0.0, 0};
}

// IScoreUpdate — reads coverage from bind_data->Cast<IScoreBindData>().coverage.
// Pattern: F1Finalize reads aggr_input.bind_data->Cast<F1BindData>() at
// tabfm_metrics_classification.cpp:335-364.
void IScoreUpdate(Vector inputs[], AggregateInputData &aggr_input, idx_t, Vector &state_vector,
                  idx_t count) {
	// Pitfall 5 resolution: read coverage from bind data (not arguments, which
	// are unavailable in Update's AggregateInputData without a ClientContext).
	auto &bd         = aggr_input.bind_data->Cast<IScoreBindData>();
	double coverage  = bd.coverage;

	UnifiedVectorFormat sdata, actual_data;
	state_vector.ToUnifiedFormat(count, sdata);
	inputs[0].ToUnifiedFormat(count, actual_data);
	auto states = reinterpret_cast<IScoreState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		idx_t sidx = sdata.sel->get_index(i);
		idx_t aidx = actual_data.sel->get_index(i);

		if (!actual_data.validity.RowIsValid(aidx)) {
			continue;
		}
		double actual_val = UnifiedVectorFormat::GetData<double>(actual_data)[aidx];

		Value dist_val = inputs[1].GetValue(i);
		if (dist_val.IsNull()) {
			continue;
		}

		auto &dist_children = StructValue::GetChildren(dist_val);
		if (dist_children[0].IsNull() || dist_children[1].IsNull()) {
			continue;
		}

		auto  &logit_vals  = ListValue::GetChildren(dist_children[0]);
		auto  &border_vals = ListValue::GetChildren(dist_children[1]);
		const size_t K     = logit_vals.size();
		if (K == 0 || border_vals.size() != K + 1) {
			continue;
		}

		// WR-02: per-element null check (see CRPSUpdate for rationale)
		std::vector<double> logits(K), borders(K + 1);
		bool any_null = false;
		for (size_t k = 0; k < K; k++) {
			if (logit_vals[k].IsNull() || border_vals[k].IsNull()) {
				any_null = true;
				break;
			}
			logits[k]  = DoubleValue::Get(logit_vals[k]);
			borders[k] = DoubleValue::Get(border_vals[k]);
		}
		if (!any_null && border_vals[K].IsNull()) {
			any_null = true;
		}
		if (any_null) {
			continue;
		}
		borders[K] = DoubleValue::Get(border_vals[K]);

		SoftmaxInPlace(logits);

		auto &state = *states[sidx];
		state.sum_is += ComputeIntervalScore(actual_val, logits, borders, coverage);
		state.n++;
	}
}

void IScoreCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &,
                   idx_t count) {
	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<IScoreState **>(source_data.data);
	auto targets = reinterpret_cast<IScoreState **>(target_data.data);
	for (idx_t i = 0; i < count; i++) {
		auto &src = *sources[source_data.sel->get_index(i)];
		auto &tgt = *targets[target_data.sel->get_index(i)];
		tgt.sum_is += src.sum_is;
		tgt.n      += src.n;
	}
}

void IScoreFinalize(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                    idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<IScoreState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (state.n == 0) {
			FlatVector::SetNull(result, i + offset, true);
		} else {
			FlatVector::GetData<double>(result)[i + offset] =
			    state.sum_is / static_cast<double>(state.n);
		}
	}
}

/// IScoreBind — resolves coverage from arg[2] (if present) or defaults to 0.9.
/// Validates 0 < coverage < 1; throws InvalidInputException otherwise (T-03-05).
/// Pattern: ExpressionExecutor::EvaluateScalar reading a constant at bind —
///   tabfm_metrics_classification.cpp:230 (F1BindData reading avg from arg[2]).
unique_ptr<FunctionData> IScoreBind(ClientContext &context, AggregateFunction &,
                                     vector<unique_ptr<Expression>> &arguments) {
	PostHogTelemetry::Instance().CaptureFunctionExecution("tabfm_interval_score");

	double coverage = 0.9; // default (locked: 03-CONTEXT.md "named coverage := 0.9")
	if (arguments.size() >= 3 && arguments[2]) {
		// T-03-05: read coverage from a constant expression at bind time
		// (Pitfall 5: coverage cannot be read in Update — no ClientContext).
		Value cov_val = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
		if (!cov_val.IsNull()) {
			coverage = cov_val.GetValue<double>();
		}
	}

	// T-03-05: validate 0 < coverage < 1 at bind time (named error per SQL-API §5).
	// std::isnan guard is required because NaN comparisons always return false under
	// IEEE 754 — without it, NaN passes the range check and silently propagates through
	// DistributionQuantile into an NaN output aggregate (CR-01, gsd-code-review 03).
	if (std::isnan(coverage) || coverage <= 0.0 || coverage >= 1.0) {
		throw InvalidInputException(
		    "tabfm_interval_score: coverage must be in (0, 1) exclusive — "
		    "got %g. Use coverage := 0.9 for 90%% nominal coverage.",
		    coverage);
	}

	return make_uniq<IScoreBindData>(coverage);
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

	// --- tabfm_log_score / anofox_tabfm_log_score (PSR-02, PSR-04) ---
	//
	// Mean NLL = mean(log(w_i) - log(max(1e-10, p_i))) over the bin containing actual.
	// Out-of-support or zero-width bin → max penalty -log(1e-10) (T-03-04).
	// Two overloads: STRUCT (normal) then DOUBLE (bind-gate, PSR-04).
	{
		LogicalType yhat_dist_type = LogicalType::STRUCT(
		    {{"logits", LogicalType::LIST(LogicalType::DOUBLE)},
		     {"borders", LogicalType::LIST(LogicalType::DOUBLE)}});

		AggregateFunctionSet set("anofox_tabfm_log_score");

		// Overload 1 — STRUCT distribution argument (normal compute path)
		AggregateFunction fn_dist("anofox_tabfm_log_score",
		                          {LogicalType::DOUBLE, yhat_dist_type}, LogicalType::DOUBLE,
		                          LogScoreStateSize, LogScoreStateInit, LogScoreUpdate,
		                          LogScoreCombine, LogScoreFinalize, /*simple_update=*/nullptr,
		                          LogScoreBind, /*state_destroy=*/nullptr);
		set.AddFunction(fn_dist);

		// Overload 2 — PSR-04 bind-gate (STRUCT overload is registered first, per Pitfall 6)
		AggregateFunction fn_point(
		    "anofox_tabfm_log_score", {LogicalType::DOUBLE, LogicalType::DOUBLE},
		    LogicalType::DOUBLE, LogScoreStateSize, LogScoreStateInit, LogScoreUpdate,
		    LogScoreCombine, LogScoreFinalize, /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_log_score: second argument must be a distribution "
			        "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
			        "Use output_mode := 'distribution' in tabfm_regress.");
			    return nullptr;
		    },
		    /*state_destroy=*/nullptr);
		set.AddFunction(fn_point);

		FunctionDescription fd;
		fd.description =
		    "Compute the mean negative log-likelihood (NLL / log-score) of actual values "
		    "under the predictive bar distribution. Log-score is a proper scoring rule — "
		    "lower is better. For y in bin i: NLL = log(w_i) - log(max(1e-10, p_i)). "
		    "Out-of-support or zero-width bins return the max penalty -log(1e-10) (T-03-04). "
		    "The second argument must be the yhat_dist STRUCT from tabfm_regress with "
		    "output_mode := 'distribution'. NULL rows are skipped; empty input returns NULL.";
		fd.examples = {
		    "SELECT tabfm_log_score(actual, yhat_dist) FROM "
		    "tabfm_regress('tbl', 'y', opts := MAP{'output_mode':'distribution'});"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_log_score", {std::move(fd)});
	}

	// --- tabfm_interval_score / anofox_tabfm_interval_score (PSR-03, PSR-04) ---
	//
	// Gneiting & Raftery (2007) interval score:
	//   IS = (u-l) + (2/alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]
	// alpha = 1 - coverage, l/u = q_{alpha/2} / q_{1-alpha/2} from DistributionQuantile.
	//
	// Four overloads (Pitfall 6: STRUCT overloads registered first):
	//   1. (actual DOUBLE, yhat_dist STRUCT)                  → 2-arg with default coverage=0.9
	//   2. (actual DOUBLE, yhat_dist STRUCT, coverage DOUBLE) → explicit coverage
	//   3. (actual DOUBLE, plain_estimate DOUBLE)             → bind-gate throws (PSR-04)
	//   4. (actual DOUBLE, plain_estimate DOUBLE, coverage DOUBLE) → bind-gate (IN-02, PSR-04)
	{
		LogicalType yhat_dist_type = LogicalType::STRUCT(
		    {{"logits", LogicalType::LIST(LogicalType::DOUBLE)},
		     {"borders", LogicalType::LIST(LogicalType::DOUBLE)}});

		AggregateFunctionSet set("anofox_tabfm_interval_score");

		// Overload 1 — 2-arg STRUCT (default coverage=0.9 resolved in IScoreBind)
		AggregateFunction fn_dist2("anofox_tabfm_interval_score",
		                           {LogicalType::DOUBLE, yhat_dist_type}, LogicalType::DOUBLE,
		                           IScoreStateSize, IScoreStateInit, IScoreUpdate, IScoreCombine,
		                           IScoreFinalize, /*simple_update=*/nullptr, IScoreBind,
		                           /*state_destroy=*/nullptr);
		set.AddFunction(fn_dist2);

		// Overload 2 — 3-arg STRUCT + explicit coverage (IScoreBind reads arg[2])
		AggregateFunction fn_dist3("anofox_tabfm_interval_score",
		                           {LogicalType::DOUBLE, yhat_dist_type, LogicalType::DOUBLE},
		                           LogicalType::DOUBLE, IScoreStateSize, IScoreStateInit,
		                           IScoreUpdate, IScoreCombine, IScoreFinalize,
		                           /*simple_update=*/nullptr, IScoreBind,
		                           /*state_destroy=*/nullptr);
		set.AddFunction(fn_dist3);

		// Overload 3 — PSR-04 bind-gate (2-arg DOUBLE always throws)
		AggregateFunction fn_point(
		    "anofox_tabfm_interval_score", {LogicalType::DOUBLE, LogicalType::DOUBLE},
		    LogicalType::DOUBLE, IScoreStateSize, IScoreStateInit, IScoreUpdate, IScoreCombine,
		    IScoreFinalize, /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_interval_score: second argument must be a distribution "
			        "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
			        "Use output_mode := 'distribution' in tabfm_regress.");
			    return nullptr;
		    },
		    /*state_destroy=*/nullptr);
		set.AddFunction(fn_point);

		// Overload 4 — PSR-04 bind-gate for 3-arg (DOUBLE, DOUBLE, DOUBLE) call (IN-02).
		// Without this overload, tabfm_interval_score(actual, scalar_yhat, 0.9) resolves
		// to a generic DuckDB "no function found" error rather than the named PSR-04 remedy.
		AggregateFunction fn_point3(
		    "anofox_tabfm_interval_score",
		    {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
		    IScoreStateSize, IScoreStateInit, IScoreUpdate, IScoreCombine, IScoreFinalize,
		    /*simple_update=*/nullptr,
		    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
		        -> unique_ptr<FunctionData> {
			    throw InvalidInputException(
			        "tabfm_interval_score: second argument must be a distribution "
			        "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
			        "Use output_mode := 'distribution' in tabfm_regress.");
			    return nullptr;
		    },
		    /*state_destroy=*/nullptr);
		set.AddFunction(fn_point3);

		FunctionDescription fd;
		fd.description =
		    "Compute the mean Gneiting & Raftery (2007) interval score at a given coverage "
		    "level. IS = (u-l) + (2/(1-coverage)) * underflow + overflow penalties. "
		    "Lower is better. The `coverage` parameter (default 0.9) is the nominal "
		    "coverage — must be strictly between 0 and 1. "
		    "The second argument must be the yhat_dist STRUCT from tabfm_regress with "
		    "output_mode := 'distribution'. NULL rows are skipped; empty input returns NULL.";
		fd.examples = {
		    "SELECT tabfm_interval_score(actual, yhat_dist, coverage := 0.9) FROM "
		    "tabfm_regress('tbl', 'y', opts := MAP{'output_mode':'distribution'});"};
		RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_interval_score", {std::move(fd)});
	}
}

} // namespace anofox
} // namespace duckdb
