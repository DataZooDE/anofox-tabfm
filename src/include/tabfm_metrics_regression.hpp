#pragma once
//===----------------------------------------------------------------------===//
// tabfm_metrics_regression.hpp — WS-G (regression metric aggregates)
//
// Public interface for regression evaluation metrics:
//   anofox_tabfm_rmse / tabfm_rmse
//   anofox_tabfm_mae / tabfm_mae
//   anofox_tabfm_r2 / tabfm_r2
//   anofox_tabfm_mape / tabfm_mape
//   anofox_tabfm_medae / tabfm_medae
//   (metrics added by plan 01-03)
//
// Spec ref: RMET-01..04, SQL-API §4 metric surface
// Called from: LoadInternal() in anofox_tabfm_extension.cpp
//===----------------------------------------------------------------------===//

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

/// Register all regression metric aggregates.
/// Called once from LoadInternal() in anofox_tabfm_extension.cpp.
void RegisterRegressionMetrics(ExtensionLoader &loader);

} // namespace anofox
} // namespace duckdb
