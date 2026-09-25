#pragma once
//===----------------------------------------------------------------------===//
// tabfm_metrics_classification.hpp — WS-G (classification metric aggregates)
//
// Public interface for classification evaluation metrics:
//   anofox_tabfm_accuracy / tabfm_accuracy
//   (further metrics added by plan 01-02)
//
// Spec ref: CMET-01..06, SQL-API §4 metric surface
// Called from: LoadInternal() in anofox_tabfm_extension.cpp
//===----------------------------------------------------------------------===//

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

/// Register all classification metric aggregates and table functions.
/// Called once from LoadInternal() in anofox_tabfm_extension.cpp.
void RegisterClassificationMetrics(ExtensionLoader &loader);

} // namespace anofox
} // namespace duckdb
