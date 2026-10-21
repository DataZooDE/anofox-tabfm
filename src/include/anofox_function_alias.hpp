#pragma once

#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

#include "duckdb_compat.hpp"

#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

// Alias registration helpers (#51).
//
// Aliases are full copies of their primary function with only the name changed
// (copy-then-rename). Reconstructing the function from a subset of fields would
// silently drop metadata such as bind_replace, statistics propagation, pushdown
// flags, serialization hooks, null handling and error modes — the alias must be
// behaviorally identical to the primary.
//
// The optional descriptions parameter attaches the same FunctionDescription
// metadata to both the primary and the alias catalog entries. On DuckDB 2.0 the
// description's parameter names also name the signature's parameters: scalar and
// aggregate arguments become passable by name (`model := 'x'`, as 1.x allowed),
// table function parameters keep their positional names.

// DuckDB 2.0: name each overload's parameters after its description, so callers can pass
// arguments by name (`model := 'x'`) as 1.x allowed. Descriptions are per documented usage, which
// need not be one per overload; then the overloads stay as they are.
template <class INFO>
void NameArgumentsFromDocs(INFO &info) {
	idx_t count = 0;
	duckdb_compat::ForEachFunction(info.functions, [&](auto &) { count++; });
	if (count == info.descriptions.size()) {
		duckdb_compat::NameArgumentsFromDescriptions(info);
	}
}

// Helper to register a scalar function with an alias
inline void RegisterScalarFunctionWithAlias(ExtensionLoader &loader, ScalarFunction func,
                                            const std::string &alias_name,
                                            vector<FunctionDescription> descriptions = {}) {
	// Register primary with metadata
	CreateScalarFunctionInfo primary_info(func);
	primary_info.descriptions = descriptions;
	NameArgumentsFromDocs(primary_info);
	loader.RegisterFunction(primary_info);

	// Register alias: copy-then-rename so all metadata survives
	auto primary_name = func.name;
	auto alias_func = std::move(func);
	alias_func.name = duckdb_compat::FunctionName(alias_name);
	CreateScalarFunctionInfo alias_info(std::move(alias_func));
	alias_info.descriptions = std::move(descriptions);
	NameArgumentsFromDocs(alias_info);
	alias_info.alias_of = primary_name;
	loader.RegisterFunction(alias_info);
}

// Helper to register a scalar function set with an alias
inline void RegisterScalarFunctionSetWithAlias(ExtensionLoader &loader, ScalarFunctionSet func_set,
                                               const std::string &alias_name,
                                               vector<FunctionDescription> descriptions = {}) {
	// Register primary with metadata
	CreateScalarFunctionInfo primary_info(func_set);
	primary_info.descriptions = descriptions;
	NameArgumentsFromDocs(primary_info);
	loader.RegisterFunction(primary_info);

	// Register alias: copy-then-rename every overload so all metadata survives
	ScalarFunctionSet alias_set(duckdb_compat::FunctionName(alias_name));
	duckdb_compat::ForEachFunction(func_set, [&](const ScalarFunction &func) {
		auto alias_func = func;
		alias_func.name = duckdb_compat::FunctionName(alias_name);
		alias_set.AddFunction(std::move(alias_func));
	});
	CreateScalarFunctionInfo alias_info(std::move(alias_set));
	alias_info.descriptions = std::move(descriptions);
	NameArgumentsFromDocs(alias_info);
	alias_info.alias_of = func_set.name;
	loader.RegisterFunction(alias_info);
}

// Helper to register a table function with an alias
inline void RegisterTableFunctionWithAlias(ExtensionLoader &loader, TableFunction func,
                                           const std::string &alias_name,
                                           vector<FunctionDescription> descriptions = {}) {
	// Register primary with metadata
	CreateTableFunctionInfo primary_info(func);
	primary_info.descriptions = descriptions;
	loader.RegisterFunction(primary_info);

	// Register alias: copy-then-rename so all metadata survives
	auto primary_name = func.name;
	auto alias_func = std::move(func);
	alias_func.name = duckdb_compat::FunctionName(alias_name);
	CreateTableFunctionInfo alias_info(std::move(alias_func));
	alias_info.descriptions = std::move(descriptions);
	alias_info.alias_of = primary_name;
	loader.RegisterFunction(alias_info);
}

// Helper to register a table function set with an alias
inline void RegisterTableFunctionSetWithAlias(ExtensionLoader &loader, TableFunctionSet func_set,
                                              const std::string &alias_name,
                                              vector<FunctionDescription> descriptions = {}) {
	// Register primary with metadata
	CreateTableFunctionInfo primary_info(func_set);
	primary_info.descriptions = descriptions;
	loader.RegisterFunction(primary_info);

	// Register alias: copy-then-rename every overload so all metadata survives
	TableFunctionSet alias_set(duckdb_compat::FunctionName(alias_name));
	duckdb_compat::ForEachFunction(func_set, [&](const TableFunction &func) {
		auto alias_func = func;
		alias_func.name = duckdb_compat::FunctionName(alias_name);
		alias_set.AddFunction(std::move(alias_func));
	});
	CreateTableFunctionInfo alias_info(std::move(alias_set));
	alias_info.descriptions = std::move(descriptions);
	alias_info.alias_of = func_set.name;
	loader.RegisterFunction(alias_info);
}

// Helper to register an aggregate function set with an alias
inline void RegisterAggregateFunctionSetWithAlias(ExtensionLoader &loader, AggregateFunctionSet func_set,
                                                  const std::string &alias_name,
                                                  vector<FunctionDescription> descriptions = {}) {
	// Register primary with metadata
	CreateAggregateFunctionInfo primary_info(func_set);
	primary_info.descriptions = descriptions;
	NameArgumentsFromDocs(primary_info);
	loader.RegisterFunction(primary_info);

	// Register alias: copy-then-rename every overload so all metadata survives
	AggregateFunctionSet alias_set(duckdb_compat::FunctionName(alias_name));
	duckdb_compat::ForEachFunction(func_set, [&](const AggregateFunction &func) {
		auto alias_func = func;
		alias_func.name = duckdb_compat::FunctionName(alias_name);
		alias_set.AddFunction(std::move(alias_func));
	});
	CreateAggregateFunctionInfo alias_info(std::move(alias_set));
	alias_info.descriptions = std::move(descriptions);
	NameArgumentsFromDocs(alias_info);
	alias_info.alias_of = func_set.name;
	loader.RegisterFunction(alias_info);
}

} // namespace anofox
} // namespace duckdb
