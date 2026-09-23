#pragma once
//===----------------------------------------------------------------------===//
// tabfm_scoring.hpp — WS-scoring (proper scoring rule aggregates)
//
// Public interface for proper scoring rule evaluation aggregates (Phase 3):
//   anofox_tabfm_crps          / tabfm_crps           (PSR-01)
//   anofox_tabfm_log_score     / tabfm_log_score      (PSR-02)
//   anofox_tabfm_interval_score / tabfm_interval_score (PSR-03)
//
// Input shape: (actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[]))
// The yhat_dist field is the direct output of tabfm_regress with
// output_mode := 'distribution' (Phase 2 bar-distribution contract).
//
// PSR-04 bind-gate: all PSR aggregates are bind-gated on the yhat_dist
// STRUCT input type; passing a plain DOUBLE second argument fails at bind
// with a named error pointing at output_mode := 'distribution'.
//
// Spec ref: PSR-01, PSR-02, PSR-03, PSR-04, SQL-API §5
// Research ref: 03-RESEARCH.md §Exact CRPS Math, §Log-Score Math, §Interval Score Math,
//               §Bind-Gating (PSR-04)
// Called from: LoadInternal() in anofox_tabfm_extension.cpp
//===----------------------------------------------------------------------===//

#include "duckdb/main/extension/extension_loader.hpp"

#include <vector>

namespace duckdb {
namespace anofox {

/// Register all proper scoring rule aggregates (tabfm_crps, tabfm_log_score,
/// tabfm_interval_score). Called once from LoadInternal() in anofox_tabfm_extension.cpp.
void RegisterScoringFunctions(ExtensionLoader &loader);

/// CRPS computation over a piecewise-uniform bar distribution.
///
/// Implements the analytical closed-form CRPS:
///   CRPS = sum over K bins of per-bin integral contribution
/// where the CDF is piecewise-linear (uniform density p_i / w_i per bin).
///
/// Preconditions:
///   - probs.size() == borders.size() - 1  (K probs, K+1 borders)
///   - probs already normalized (sum to 1.0 after softmax)
///   - borders are strictly increasing (w_i = borders[i+1] - borders[i] > 0)
///
/// Returns 0.0 on malformed input (K==0 or size mismatch).
///
/// External linkage so Catch2 unit tests can call it directly
/// (same pattern as DistributionMean/DistributionQuantile in tabfm_engine.cpp,
/// Phase 2 decision 02-02).
///
/// Assumption A1 (03-RESEARCH.md §CRPS Math): integrates only over [b_0, b_K];
/// y outside this range still contributes via Case A/B of every bin —
/// no half-normal tail extension. Deferred to ASCR-01.
double ComputeCRPS(double y, const std::vector<double> &probs, const std::vector<double> &borders);

/// Log-score (negative log-likelihood) over a piecewise-uniform bar distribution.
///
/// For y in bin i: log_score = log(w_i) - log(max(clip_eps, p_i))
/// where clip_eps = 1e-10 (Assumption A2, 03-RESEARCH.md §Log-Score Math).
///
/// Out-of-support (y < borders[0] or y > borders[K]) and zero-width bins
/// (w_i < 1e-300) return the maximum penalty -log(clip_eps) = 23.025...
///
/// Preconditions:
///   - probs.size() == borders.size() - 1  (K probs, K+1 borders)
///   - probs already normalized (sum to 1.0 after softmax)
///
/// Returns max-penalty on malformed input (K==0 or size mismatch).
///
/// External linkage so Catch2 unit tests can call it directly.
/// Spec ref: PSR-02
double ComputeLogScore(double y, const std::vector<double> &probs, const std::vector<double> &borders);

/// Interval score at a specified coverage level over a piecewise-uniform bar distribution.
///
/// IS = (u - l) + (2 / alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]
/// where alpha = 1 - coverage, l = q_{alpha/2}, u = q_{1-alpha/2}
/// (Gneiting & Raftery 2007 interval score formula).
///
/// Quantiles l, u are computed via DistributionQuantile (tabfm_predict.hpp:206).
///
/// Preconditions:
///   - probs.size() == borders.size() - 1  (K probs, K+1 borders)
///   - probs already normalized (sum to 1.0 after softmax)
///   - 0.0 < coverage < 1.0 (validated at bind time; not re-validated here)
///
/// External linkage so Catch2 unit tests can call it directly.
/// Spec ref: PSR-03
double ComputeIntervalScore(double y, const std::vector<double> &probs,
                             const std::vector<double> &borders, double coverage);

} // namespace anofox
} // namespace duckdb
