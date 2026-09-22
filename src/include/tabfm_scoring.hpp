#pragma once
//===----------------------------------------------------------------------===//
// tabfm_scoring.hpp — WS-scoring (proper scoring rule aggregates)
//
// Public interface for proper scoring rule evaluation aggregates (Phase 3):
//   anofox_tabfm_crps / tabfm_crps    (PSR-01)
//
// Input shape: (actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[]))
// The yhat_dist field is the direct output of tabfm_regress with
// output_mode := 'distribution' (Phase 2 bar-distribution contract).
//
// PSR-04 bind-gate: all PSR aggregates are bind-gated on the yhat_dist
// STRUCT input type; passing a plain DOUBLE second argument fails at bind
// with a named error pointing at output_mode := 'distribution'.
//
// Spec ref: PSR-01, PSR-04, SQL-API §5
// Research ref: 03-RESEARCH.md §Exact CRPS Math, §Bind-Gating (PSR-04)
// Called from: LoadInternal() in anofox_tabfm_extension.cpp
//===----------------------------------------------------------------------===//

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

/// Register all proper scoring rule aggregates (tabfm_crps, etc.).
/// Called once from LoadInternal() in anofox_tabfm_extension.cpp.
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

} // namespace anofox
} // namespace duckdb
