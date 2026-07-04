#include "tabfm_registration.hpp"

#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

namespace duckdb {
namespace anofox {

// Table macros tabfm_predict / tabfm_predict_by — sugar over
// anofox_tabfm_predict_agg (SQL-API §2 Levels 1-3). Owned by WS-E.
//
// The bodies below are the FINAL S04 RESULTS.md macro bodies with two
// integration deviations (documented at their sites): the row-struct alias is
// anofox_tabfm_row (not `t`, which a user column can shadow), and the feature
// filter uses the new `lambda c:` syntax (the S04 `c ->` arrow is deprecated
// and warns on DuckDB 1.5.4). Load-bearing details from the spike run:
//  - unnest(res, max_depth := 3), NEVER recursive := true — recursive unnest
//    recurses *into user STRUCT columns* and flattens them away; max_depth
//    stops exactly at list → result-struct → cols (S04 4f/4g).
//  - Overload dispatch is ARITY-ONLY (macro params are untyped, S04 3b):
//    arity-3 is always the opts slot; a VARCHAR[] landing there is caught by
//    the aggregate's bind with a redirecting error (S04 3c).
//  - lcase() on BOTH sides of the feature match: struct_extract resolves the
//    target case-insensitively while the raw COLUMNS lambda comparison is
//    case-sensitive (S04 2b). A feature name matching no column silently
//    drops out — documented behavior.
//  - The _by feature variant must keep grp in the projection or the
//    GROUP BY struct_extract(t, grp) fails after projection.
//  - If the input already has a yhat column, unnest auto-deduplicates: the
//    prediction lands in yhat_1 (S04 4e). No row-order guarantee (S04 #6).
//
// Registration route: one CreateMacroInfo per macro NAME carrying all arity
// overloads as separate MacroFunctions — exactly what a comma-overloaded
// CREATE MACRO produces. (Re-registering the same name via separate infos
// would conflict in the catalog; a single multi-function info is the
// supported shape.) Full names anofox_tabfm_predict[_by] are the primaries;
// tabfm_predict[_by] are catalog aliases of them (house style).

namespace {

struct PredictMacroDef {
	const char *parameters[8]; // nullptr-terminated
	const char *body;
};

// The whole-row struct is aliased anofox_tabfm_row, NOT the spike's `t`: a
// user table with a column named `t` would otherwise shadow the alias and make
// struct_extract(t, grp) bind to that column (regression found in
// tabfm_predict_by.test). The long alias is collision-resistant; a user column
// literally named anofox_tabfm_row is the only (documented) remaining clash.
// clang-format off
static const PredictMacroDef PREDICT_MACROS[] = {
    {{"tbl", "target", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT tabfm_predict_agg(anofox_tabfm_row, target) AS res
      FROM query_table(tbl) anofox_tabfm_row
    )
)"},
    {{"tbl", "target", "opts", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT tabfm_predict_agg(anofox_tabfm_row, target, opts) AS res
      FROM query_table(tbl) anofox_tabfm_row
    )
)"},
    {{"tbl", "target", "features", "opts", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT tabfm_predict_agg(anofox_tabfm_row, target, opts) AS res
      FROM (SELECT COLUMNS(lambda c: list_contains(list_transform(features, lambda f: lcase(f)), lcase(c))
                                OR lcase(c) = lcase(target))
            FROM query_table(tbl)) anofox_tabfm_row
    )
)"},
    {{nullptr}, nullptr}};

static const PredictMacroDef PREDICT_BY_MACROS[] = {
    {{"tbl", "grp", "target", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT tabfm_predict_agg(anofox_tabfm_row, target) AS res
      FROM query_table(tbl) anofox_tabfm_row
      GROUP BY struct_extract(anofox_tabfm_row, grp)
    )
)"},
    {{"tbl", "grp", "target", "opts", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT tabfm_predict_agg(anofox_tabfm_row, target, opts) AS res
      FROM query_table(tbl) anofox_tabfm_row
      GROUP BY struct_extract(anofox_tabfm_row, grp)
    )
)"},
    {{"tbl", "grp", "target", "features", "opts", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT tabfm_predict_agg(anofox_tabfm_row, target, opts) AS res
      FROM (SELECT COLUMNS(lambda c: list_contains(list_transform(features, lambda f: lcase(f)), lcase(c))
                                OR lcase(c) IN (lcase(target), lcase(grp)))
            FROM query_table(tbl)) anofox_tabfm_row
      GROUP BY struct_extract(anofox_tabfm_row, grp)
    )
)"},
    {{nullptr}, nullptr}};
// clang-format on

unique_ptr<MacroFunction> BuildTableMacroFunction(const PredictMacroDef &def) {
	Parser parser;
	parser.ParseQuery(def.body);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("tabfm predict macro body must be a single SELECT statement");
	}
	auto node = std::move(parser.statements[0]->Cast<SelectStatement>().node);
	auto function = make_uniq<TableMacroFunction>(std::move(node));
	for (idx_t i = 0; def.parameters[i] != nullptr; i++) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(def.parameters[i]));
	}
	return std::move(function);
}

//! One CreateMacroInfo with all arity overloads of one macro name.
unique_ptr<CreateMacroInfo> BuildMacroInfo(const string &name, const PredictMacroDef *defs, const string &alias_of) {
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = name;
	info->temporary = true;
	info->internal = true;
	info->alias_of = alias_of;
	for (idx_t i = 0; defs[i].body != nullptr; i++) {
		info->macros.push_back(BuildTableMacroFunction(defs[i]));
	}
	return info;
}

void RegisterMacroWithAlias(ExtensionLoader &loader, const string &full_name, const string &alias_name,
                            const PredictMacroDef *defs) {
	auto primary = BuildMacroInfo(full_name, defs, string());
	loader.RegisterFunction(*primary);
	auto alias = BuildMacroInfo(alias_name, defs, full_name);
	loader.RegisterFunction(*alias);
}

} // anonymous namespace

void RegisterPredictMacros(ExtensionLoader &loader) {
	RegisterMacroWithAlias(loader, "anofox_tabfm_predict", "tabfm_predict", PREDICT_MACROS);
	RegisterMacroWithAlias(loader, "anofox_tabfm_predict_by", "tabfm_predict_by", PREDICT_BY_MACROS);
}

} // namespace anofox
} // namespace duckdb
