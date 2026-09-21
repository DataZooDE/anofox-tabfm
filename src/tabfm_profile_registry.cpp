//===----------------------------------------------------------------------===//
// tabfm_profile_registry.cpp — preprocessing-profile registry (MGEN-01/02, WS-F)
//
// Implements the preprocessing-profile dispatch seam: manifest
// `preprocessing_profile` string → C++ function pointer. Profiles
// self-register via static ProfileRegistration instances; the registry is
// a function-local static unordered_map (mirrors GetPredictEngine at
// tabfm_engine.cpp:781 — no static-init order fiasco).
//
// ForceProfileInit() is called from LoadInternal to ensure this TU (and the
// tabpfn_v2 stub TU) are linked even if the linker would otherwise strip them
// as unreferenced (RESEARCH Pitfall 2, T-02-02).
//===----------------------------------------------------------------------===//

#include "tabfm_profile_registry.hpp"
#include "tabfm_preprocess.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace anofox {

namespace {

//===----------------------------------------------------------------------===//
// Registry singleton
//===----------------------------------------------------------------------===//

//! Returns the global registry map, constructed on first use (function-local
//! static: guaranteed by C++11 §6.7 to be initialized exactly once and
//! thread-safe). Mirrors GetPredictEngine() at tabfm_engine.cpp:781.
unordered_map<string, PreprocessFn> &GetRegistry() {
	static unordered_map<string, PreprocessFn> registry;
	return registry;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ProfileRegistration constructor
//===----------------------------------------------------------------------===//

ProfileRegistration::ProfileRegistration(const string &profile_id, PreprocessFn fn) {
	GetRegistry().emplace(profile_id, fn);
}

//===----------------------------------------------------------------------===//
// DispatchPreprocess — the sole dispatch point for all model families
//===----------------------------------------------------------------------===//

PreprocessedBatch DispatchPreprocess(const string &profile_id, const ColumnDataCollection &data,
                                     const vector<PreprocessColumnSpec> &columns, PreprocessTask task) {
	auto &registry = GetRegistry();
	auto it = registry.find(profile_id);
	if (it == registry.end()) {
		throw InvalidInputException(
		    "tabfm: preprocessing profile '%s' is not registered. "
		    "Verify the manifest's preprocessing_profile field. "
		    "Run: SELECT * FROM tabfm_models();",
		    profile_id);
	}
	return it->second(data, columns, task);
}

//===----------------------------------------------------------------------===//
// Self-registration of the tabfm_v1_minimal profile (MGEN-01)
//===----------------------------------------------------------------------===//
// Use kPreprocessProfileId constant (not a literal) so a rename stays consistent.
static const ProfileRegistration kTabFMV1Reg(kPreprocessProfileId, PreprocessBatch);

//===----------------------------------------------------------------------===//
// ForceProfileInit — prevents linker stripping (T-02-02, RESEARCH Pitfall 2)
//===----------------------------------------------------------------------===//
// Forward-declare the tabpfn_v2 stub's force-init (defined in
// tabfm_preprocess_tabpfn_v2.cpp). Calling it here chains the force-link.
void ForceTabPFNV2ProfileInit();

void ForceProfileInit() {
	// Empty body is intentional: the mere existence of this symbol in a
	// referenced TU prevents the linker from stripping this translation unit
	// and its static initializer (kTabFMV1Reg). Chaining the tabpfn_v2 stub
	// ensures that TU is also linked.
	ForceTabPFNV2ProfileInit();
}

} // namespace anofox
} // namespace duckdb
