//===----------------------------------------------------------------------===//
// tabfm_profile_registry.hpp — preprocessing-profile registry seam (MGEN-01/02)
//
// Maps a manifest `preprocessing_profile` string to a C++ preprocessing
// function. Profiles self-register via static initializers; the engine
// dispatches through this registry so new model families never require
// changes to tabfm_engine.cpp.
//
// Threat model:
//   T-02-01 (Tampering): an untrusted manifest.preprocessing_profile string
//   crosses into the engine. The registry performs an exact-match lookup and
//   throws InvalidInputException BEFORE any ORT run or file I/O when the
//   profile is not known. No silent fall-through.
//   T-02-02 (DoS): static-init stripping would leave the registry empty.
//   ForceProfileInit() is called from LoadInternal to give the linker a
//   referenced symbol so self-registering TUs are never stripped.
//===----------------------------------------------------------------------===//

#pragma once

#include "tabfm_preprocess.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

namespace duckdb {
namespace anofox {

//! Function pointer type for a preprocessing profile implementation.
//! Must match PreprocessBatch's signature exactly.
using PreprocessFn = PreprocessedBatch (*)(const ColumnDataCollection &data,
                                           const vector<PreprocessColumnSpec> &columns,
                                           PreprocessTask task);

//! Self-registration RAII handle. Construct a file-scoped static instance of
//! this struct to register a preprocessing profile at program startup:
//!
//!   static const ProfileRegistration kMyReg("my_profile", MyPreprocessBatch);
//!
//! The constructor inserts the (id, fn) pair into the global registry. It is
//! safe to call from static initializers because GetRegistry() uses a
//! function-local static (no static-init order fiasco).
struct ProfileRegistration {
	ProfileRegistration(const string &profile_id, PreprocessFn fn);
};

//! Look up `profile_id` in the registry and call the registered function.
//! Throws InvalidInputException naming the fix when the profile is unknown:
//!   "tabfm: preprocessing profile '<id>' is not registered.
//!    Verify the manifest's preprocessing_profile field.
//!    Run: SELECT * FROM tabfm_models();"
//! This is the sole dispatch point for all model families; called by
//! tabfm_engine.cpp instead of the old direct PreprocessBatch() call.
PreprocessedBatch DispatchPreprocess(const string &profile_id, const ColumnDataCollection &data,
                                     const vector<PreprocessColumnSpec> &columns, PreprocessTask task);

//! Called from LoadInternal() to prevent the linker from stripping the
//! self-registering TUs (tabfm_profile_registry.cpp,
//! tabfm_preprocess_tabpfn_v2.cpp). An empty function body is sufficient —
//! its mere existence gives the linker a referenced symbol. (RESEARCH Pitfall 2)
void ForceProfileInit();

} // namespace anofox
} // namespace duckdb
