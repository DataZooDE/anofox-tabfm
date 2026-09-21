//===----------------------------------------------------------------------===//
// tabfm_preprocess_tabpfn_v2.cpp — tabpfn_v2 preprocessing profile stub
//
// Placeholder stub for plan 02-03. Self-registers the "tabpfn_v2" profile
// with the profile registry so the build links and the registry is consistent
// across all Phase-2 plans.
//
// Plan 03 fills TabPFNV2PreprocessBatch with a fixture-scoped minimal
// implementation that produces a correctly-shaped PreprocessedBatch for the
// K=16 distribution fixture. For now the function throws NotImplementedException
// so any accidental invocation surfaces a clear message.
//
// ForceTabPFNV2ProfileInit() is called from ForceProfileInit() in
// tabfm_profile_registry.cpp to ensure this TU is linked (RESEARCH Pitfall 2).
//===----------------------------------------------------------------------===//

#include "tabfm_profile_registry.hpp"
#include "tabfm_preprocess.hpp"

#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// tabpfn_v2 preprocessing function (stub — plan 03 fills this body)
//===----------------------------------------------------------------------===//

PreprocessedBatch TabPFNV2PreprocessBatch(const ColumnDataCollection & /*data*/,
                                          const vector<PreprocessColumnSpec> & /*columns*/,
                                          PreprocessTask /*task*/) {
	throw NotImplementedException(
	    "tabfm: tabpfn_v2 preprocessing profile is not yet implemented "
	    "(fixture arrives in plan 02-03)");
}

//===----------------------------------------------------------------------===//
// Self-registration of the tabpfn_v2 profile
//===----------------------------------------------------------------------===//
static const ProfileRegistration kTabPFNV2Reg("tabpfn_v2", TabPFNV2PreprocessBatch);

//===----------------------------------------------------------------------===//
// ForceTabPFNV2ProfileInit — chained from ForceProfileInit() to keep this TU linked
//===----------------------------------------------------------------------===//
void ForceTabPFNV2ProfileInit() {
	// Empty body intentional (see ForceProfileInit comment in
	// tabfm_profile_registry.cpp). Existence prevents linker stripping.
}

} // namespace anofox
} // namespace duckdb
