//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_plugin_backend.hpp — load a TabFMBackend from a shared library.
//
// Phase 1 of docs/DYNAMIC_BACKENDS.md. See tabfm_plugin_abi.h for the C
// contract the library must satisfy, and tabfm_plugin_backend.cpp for the
// failures this refuses (missing file, not a plugin, ABI mismatch).
//===----------------------------------------------------------------------===//

#pragma once

#include "tabfm_ort_engine.hpp" // TabFMBackend
#include "tabfm_plugin_abi.h"

namespace duckdb {
namespace anofox {

//! Load `library_path` and construct its backend with `params`.
//!
//! Throws IOException when the library cannot be loaded, does not export the
//! plugin entry point, or was built against a different ABI version; and
//! InvalidInputException when the backend itself refuses to initialise (no
//! device, unreadable graph, and so on). Every message names the fix.
//!
//! The library is deliberately never unloaded: a GPU backend leaves a driver
//! context and thread-locals behind it, and dlclose'ing underneath those turns
//! shutdown into a crash in an unrelated destructor.
unique_ptr<TabFMBackend> LoadPluginBackend(const string &library_path, const TabFMPluginCreateParams &params);

} // namespace anofox
} // namespace duckdb
