#include "tabfm_registration.hpp"

namespace duckdb {
namespace anofox {

// Table macros tabfm_predict / tabfm_predict_by — sugar over
// anofox_tabfm_predict_agg, bodies validated in spike S04 (use
// unnest(res, max_depth := 3), NEVER recursive := true). Owned by WS-E.
//
// Phase-0 state: nothing registered yet; the aggregate stub carries the
// NotImplemented error surface.

void RegisterPredictMacros(ExtensionLoader &loader) {
	(void)loader;
}

} // namespace anofox
} // namespace duckdb
