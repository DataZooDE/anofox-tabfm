#pragma once
//===----------------------------------------------------------------------===//
// tabfm_crossval.hpp — WS-G (cross-validation macros)
//
// Public interface for cross-validation:
//   anofox_tabfm_cross_validate / tabfm_cross_validate
//   anofox_tabfm_fold_assign / tabfm_fold_assign
//   (macros added by plan 01-04)
//
// Spec ref: CV-01..04, SQL-API §4 CV surface
// Called from: LoadInternal() in anofox_tabfm_extension.cpp
//===----------------------------------------------------------------------===//

#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

/// Register all cross-validation SQL macros.
/// Called once from LoadInternal() in anofox_tabfm_extension.cpp.
void RegisterCrossValidateMacros(ExtensionLoader &loader);

} // namespace anofox
} // namespace duckdb
