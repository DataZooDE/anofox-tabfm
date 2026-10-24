#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace anofox {

// One registration entry point per module. Each module owns exactly one .cpp
// under src/ so parallel workstreams never touch the same file.
void RegisterTabfmSettings(ExtensionLoader &loader);       // tabfm_settings.cpp
void RegisterWeightsFunctions(ExtensionLoader &loader);    // tabfm_weights.cpp
void RegisterDevicesFunctions(ExtensionLoader &loader);    // tabfm_devices.cpp
void RegisterPredictAggFunction(ExtensionLoader &loader);  // tabfm_predict_agg.cpp
void RegisterPredictMacros(ExtensionLoader &loader);       // tabfm_macros.cpp
void RegisterGenerateAggFunctions(ExtensionLoader &loader);// tabfm_generate.cpp
void RegisterGenerateMacros(ExtensionLoader &loader);      // tabfm_generate_macros.cpp
void RegisterClassificationMetrics(ExtensionLoader &loader); // tabfm_metrics_classification.cpp
void RegisterRegressionMetrics(ExtensionLoader &loader);     // tabfm_metrics_regression.cpp
void RegisterCrossValidateMacros(ExtensionLoader &loader);   // tabfm_crossval.cpp
void RegisterScoringFunctions(ExtensionLoader &loader);      // tabfm_scoring.cpp

// The anofox integration contract (tabfm_contract.cpp): snapshot the function names before registering,
// register the producers, then tag what this extension added.
case_insensitive_set_t SnapshotSystemFunctionNames(ExtensionLoader &loader);
void RegisterContractFunctions(ExtensionLoader &loader);
void ApplyContractTags(ExtensionLoader &loader, const case_insensitive_set_t &preexisting);

} // namespace anofox
} // namespace duckdb
