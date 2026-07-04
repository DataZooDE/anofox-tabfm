#include "tabfm_registration.hpp"
#include "anofox_function_alias.hpp"

#include "duckdb/function/table_function.hpp"

namespace duckdb {
namespace anofox {

// Weights lifecycle: tabfm_download / tabfm_models / tabfm_load / tabfm_unload /
// tabfm_remove (FR-1, FR-2, FR-4). Owned by workstream WS-D.
//
// Phase-0 state: every function is registered with its final signature but
// throws NotImplementedException at bind time.

namespace {

unique_ptr<FunctionData> NotImplementedBind(const char *name) {
	throw NotImplementedException("%s is not implemented yet", name);
}

void RegisterStubTableFunction(ExtensionLoader &loader, const string &full_name, const string &alias,
                               vector<LogicalType> arguments) {
	TableFunction func(
	    full_name, std::move(arguments),
	    nullptr,
	    [](ClientContext &, TableFunctionBindInput &input, vector<LogicalType> &, vector<string> &) {
		    return NotImplementedBind(input.table_function.name.c_str());
	    });
	RegisterTableFunctionWithAlias(loader, std::move(func), alias);
}

} // anonymous namespace

void RegisterWeightsFunctions(ExtensionLoader &loader) {
	// CALL tabfm_download(task [, revision]) — one result row per file.
	RegisterStubTableFunction(loader, "anofox_tabfm_download", "tabfm_download",
	                          {LogicalType::VARCHAR, LogicalType::VARCHAR});
	// SELECT * FROM tabfm_models();
	RegisterStubTableFunction(loader, "anofox_tabfm_models", "tabfm_models", {});
	// CALL tabfm_load(task);
	RegisterStubTableFunction(loader, "anofox_tabfm_load", "tabfm_load", {LogicalType::VARCHAR});
	// CALL tabfm_unload([task]);
	RegisterStubTableFunction(loader, "anofox_tabfm_unload", "tabfm_unload", {LogicalType::VARCHAR});
	// CALL tabfm_remove(task [, revision]);
	RegisterStubTableFunction(loader, "anofox_tabfm_remove", "tabfm_remove",
	                          {LogicalType::VARCHAR, LogicalType::VARCHAR});
}

} // namespace anofox
} // namespace duckdb
