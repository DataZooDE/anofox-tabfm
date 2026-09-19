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

//! Can this library be loaded as a plugin of an ABI we speak? Loads it and
//! checks the entry point and abi_version -- and deliberately stops there,
//! never calling create().
//!
//! That boundary is the whole point. create() reaches the driver: it can take
//! minutes (a MIGraphX shape compile), demand a runtime that is not installed,
//! or fail on a card that is busy. This answers the cheap question 'is the
//! lane here at all', which is what device selection needs before it commits;
//! whether THIS model can actually be served is a separate question with a
//! separate answer (EvaluateGpuServability).
//!
//! Never throws: a missing or unreadable file is a false, not an error --
//! callers use it to decide, not to report.
bool PluginLoadable(const string &library_path);

} // namespace anofox
} // namespace duckdb
