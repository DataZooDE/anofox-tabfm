//===----------------------------------------------------------------------===//
// tabfm_metrics_regression.cpp — WS-G regression metric aggregates
//
// Implements: anofox_tabfm_rmse, anofox_tabfm_mae, anofox_tabfm_r2,
//             anofox_tabfm_mape, anofox_tabfm_medae (and tabfm_* aliases).
//
// Spec ref: RMET-01..04 (plan 01-03)
// Status: stub — register body filled by plan 01-03.
//===----------------------------------------------------------------------===//

#include "tabfm_metrics_regression.hpp"
#include "tabfm_registration.hpp"

namespace duckdb {
namespace anofox {

void RegisterRegressionMetrics(ExtensionLoader &loader) {
	// Filled by plan 01-03.
	(void)loader;
}

} // namespace anofox
} // namespace duckdb
