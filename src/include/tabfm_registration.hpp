#pragma once

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

void RegisterClassificationMetrics(ExtensionLoader &loader); // tabfm_metrics_classification.cpp
void RegisterRegressionMetrics(ExtensionLoader &loader);     // tabfm_metrics_regression.cpp
void RegisterCrossValidateMacros(ExtensionLoader &loader);   // tabfm_crossval.cpp

//! Force-link the preprocessing-profile registry TUs so their static
//! self-registration initializers are not stripped by the linker (T-02-02).
//! Called from LoadInternal before any prediction-related registration.
void ForceProfileInit(); // tabfm_profile_registry.cpp

} // namespace anofox
} // namespace duckdb
