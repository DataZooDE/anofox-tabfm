#include "tabfm_registration.hpp"
#include "anofox_function_alias.hpp"

#include "duckdb/function/table_function.hpp"

namespace duckdb {
namespace anofox {

// Device discovery: tabfm_devices() (HLD D9/§9). Owned by workstream WS-C.
//
// Phase-0 state: registered with its final signature but throws
// NotImplementedException at bind time.

void RegisterDevicesFunctions(ExtensionLoader &loader) {
	TableFunction func(
	    "anofox_tabfm_devices", {}, nullptr,
	    [](ClientContext &, TableFunctionBindInput &, vector<LogicalType> &,
	       vector<string> &) -> unique_ptr<FunctionData> {
		    throw NotImplementedException("anofox_tabfm_devices is not implemented yet");
	    });
	RegisterTableFunctionWithAlias(loader, std::move(func), "tabfm_devices");
}

} // namespace anofox
} // namespace duckdb
