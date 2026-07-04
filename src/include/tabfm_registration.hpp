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

} // namespace anofox
} // namespace duckdb
