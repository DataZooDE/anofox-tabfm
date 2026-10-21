#pragma once

#include "duckdb.hpp"
#include "duckdb/catalog/default/default_table_functions.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"

#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

// The CreateMacroInfo of an internal table macro: `parameters` are the
// required positional names and `default_params` the trailing optionals as
// "name=sql_default" (both nullptr-terminated, at most 7 each). DuckDB's own
// DefaultTableFunctionGenerator parses the body and the defaults, so this
// builds the same macro on DuckDB 1.4 / 1.5 / 2.0, whose parser and
// MacroFunction APIs differ.
inline unique_ptr<CreateMacroInfo> MakeTableMacroInfo(const std::string &name, const char *const *parameters,
                                                      const char *const *default_params, const char *body) {
	DefaultTableMacro macro {};
	macro.schema = DEFAULT_SCHEMA;
	macro.name = name.c_str();
	for (idx_t i = 0; parameters[i] != nullptr; i++) {
		macro.parameters[i] = parameters[i];
	}
	std::vector<std::string> names, defaults;
	names.reserve(8);
	defaults.reserve(8);
	for (idx_t i = 0; default_params[i] != nullptr; i++) {
		std::string spec = default_params[i];
		auto eq = spec.find('=');
		names.push_back(spec.substr(0, eq));
		defaults.push_back(spec.substr(eq + 1));
		macro.named_parameters[i] = {names.back().c_str(), defaults.back().c_str()};
	}
	macro.macro = body;
	auto info = DefaultTableFunctionGenerator::CreateTableMacroInfo(macro);

	// As DuckDB's CREATE MACRO transformer does: one UNKNOWN type per
	// parameter, and each default aliased to its parameter name.
	auto &function = *info->macros.back();
	function.types.assign(function.parameters.size(), LogicalType::UNKNOWN);
	for (auto &entry : function.default_parameters) {
		entry.second->SetAlias(entry.first);
	}
	return info;
}

} // namespace anofox
} // namespace duckdb
