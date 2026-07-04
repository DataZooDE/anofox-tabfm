#include "tabfm_registration.hpp"

#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

namespace duckdb {
namespace anofox {

// User-facing predict surface (SQL-API redesign 2026-07-04): two task-specific
// table functions mirroring upstream TabFMClassifier / TabFMRegressor
// (github.com/google-research/tabfm — fit(X_train, y_train).predict(X_test)):
//
//   tabfm_classify(data, target [, test] [, features] [, opts])
//   tabfm_regress (data, target [, test] [, features] [, opts])
//
//   * data     — the context ("training") relation: a table/view NAME or a
//                parenthesised SQL subquery (both are spliced after `FROM`).
//   * target   — the column to predict.
//   * test     — optional relation of rows to score (predict(X_test)). When
//                given, only those rows are returned, their `target` forced to
//                NULL so it never contaminates the context. When omitted
//                (NULL), the call is single-relation: rows of `data` whose
//                target IS NULL are the scored rows (and all rows come back,
//                with is_training).
//   * features — optional VARCHAR[] of feature columns (default: all others);
//                matched case-insensitively, the target is always kept.
//   * opts     — trailing MAP(VARCHAR,VARCHAR) of engine options (seed,
//                n_estimators, output_mode, context_rows, …). `task` is fixed
//                by the function and cannot be overridden.
//
// Named parameters work (DuckDB macro feature), so the readable form is
//   tabfm_classify('history', 'churned', test := 'prospects', opts := MAP{'seed':'42'})
//
// Mechanics (all validated live on DuckDB 1.5.4 before wiring):
//   * DuckDB table macros cannot take a relation/subquery argument, but the
//     `query(<sql string>)` table function can — so `data`/`test` are strings
//     and the relation is assembled as a SQL string switched by a CASE on
//     whether `test` was supplied, then executed with query().
//   * arity-only dispatch forces ONE signature per function with `test`,
//     `features`, `opts` as trailing defaults (a 2-arg single-relation form
//     and a 3-arg train/test form cannot coexist as separate overloads).
//   * unnest(res, max_depth := 3) — NEVER recursive := true (S04: recursive
//     flattens user STRUCT columns away).
//   * the whole-row struct is aliased `anofox_tabfm_row` (a user column named
//     `t` would shadow the spike's `t` alias).
//   * classification defaults output_mode to 'detail' so `proba` is always in
//     the result (user can override to 'compact'); the forced `task` is
//     concatenated LAST so it wins over any user-supplied task key.
//   * WHERE (test IS NULL) OR (NOT is_training): single-relation returns every
//     row (with is_training); train/test returns only the scored rows.
//
// The engine is the internal `__anofox_tabfm_predict_agg` aggregate
// (tabfm_predict_agg.cpp). Full names anofox_tabfm_* are the primaries;
// tabfm_* are catalog aliases (house style).

namespace {

struct PredictMacroDef {
	const char *parameters[8];       // positional param names, nullptr-terminated
	const char *default_params[8];   // "name=sql_default" for trailing optionals, nullptr-terminated
	const char *body;
};

// clang-format off
// The two bodies differ only in the forced task literal ('classification' vs
// 'regression'). `data`, `test`, `target`, `features`, `opts` are macro params.
static const PredictMacroDef CLASSIFY_MACRO = {
    {"data", "target", nullptr},
    {"test=NULL", "features=NULL", "opts=MAP{}", nullptr},
R"(
    SELECT * FROM (
      SELECT unnest(res, max_depth := 3) FROM (
        SELECT __anofox_tabfm_predict_agg(
                 anofox_tabfm_row, target,
                 map_concat(
                   map_concat(MAP{'output_mode':'detail'}, CAST(opts AS MAP(VARCHAR, VARCHAR))),
                   MAP{'task':'classification'})) AS res
        FROM (
          SELECT COLUMNS(lambda c: features IS NULL
                                   OR list_contains(list_transform(features, lambda f: lcase(f)), lcase(c))
                                   OR lcase(c) = lcase(target))
          FROM query(
            CASE WHEN test IS NULL
                 THEN 'FROM ' || data
                 ELSE 'SELECT * FROM (FROM ' || data || ') UNION ALL BY NAME '
                      || 'SELECT *, NULL AS "' || replace(target, '"', '""') || '" FROM (FROM ' || test || ')'
            END)
        ) anofox_tabfm_row
      )
    ) WHERE (test IS NULL) OR (NOT is_training)
)"};

static const PredictMacroDef REGRESS_MACRO = {
    {"data", "target", nullptr},
    {"test=NULL", "features=NULL", "opts=MAP{}", nullptr},
R"(
    SELECT * FROM (
      SELECT unnest(res, max_depth := 3) FROM (
        SELECT __anofox_tabfm_predict_agg(
                 anofox_tabfm_row, target,
                 map_concat(CAST(opts AS MAP(VARCHAR, VARCHAR)), MAP{'task':'regression'})) AS res
        FROM (
          SELECT COLUMNS(lambda c: features IS NULL
                                   OR list_contains(list_transform(features, lambda f: lcase(f)), lcase(c))
                                   OR lcase(c) = lcase(target))
          FROM query(
            CASE WHEN test IS NULL
                 THEN 'FROM ' || data
                 ELSE 'SELECT * FROM (FROM ' || data || ') UNION ALL BY NAME '
                      || 'SELECT *, NULL AS "' || replace(target, '"', '""') || '" FROM (FROM ' || test || ')'
            END)
        ) anofox_tabfm_row
      )
    ) WHERE (test IS NULL) OR (NOT is_training)
)"};
// clang-format on

unique_ptr<MacroFunction> BuildTableMacroFunction(const PredictMacroDef &def) {
	Parser parser;
	parser.ParseQuery(def.body);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("tabfm predict macro body must be a single SELECT statement");
	}
	auto node = std::move(parser.statements[0]->Cast<SelectStatement>().node);
	auto function = make_uniq<TableMacroFunction>(std::move(node));
	// Required positional parameters. Every parameter goes into both
	// `parameters` and `types` (matching DuckDB's own CREATE MACRO transformer).
	for (idx_t i = 0; def.parameters[i] != nullptr; i++) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(def.parameters[i]));
		function->types.push_back(LogicalType::UNKNOWN);
	}
	// Trailing optional parameters ("name=sql_default"): also a parameter +
	// type, PLUS an entry in default_parameters (aliased to the parameter name).
	for (idx_t i = 0; def.default_params[i] != nullptr; i++) {
		string spec = def.default_params[i];
		auto eq = spec.find('=');
		auto name = spec.substr(0, eq);
		auto default_sql = spec.substr(eq + 1);
		Parser expr_parser;
		auto expr = expr_parser.ParseExpressionList(default_sql);
		expr[0]->SetAlias(name);
		function->parameters.push_back(make_uniq<ColumnRefExpression>(name));
		function->types.push_back(LogicalType::UNKNOWN);
		function->default_parameters[name] = std::move(expr[0]);
	}
	return std::move(function);
}

unique_ptr<CreateMacroInfo> BuildMacroInfo(const string &name, const PredictMacroDef &def, const string &alias_of) {
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = name;
	info->temporary = true;
	info->internal = true;
	info->alias_of = alias_of;
	info->macros.push_back(BuildTableMacroFunction(def));
	return info;
}

void RegisterMacroWithAlias(ExtensionLoader &loader, const string &full_name, const string &alias_name,
                            const PredictMacroDef &def) {
	auto primary = BuildMacroInfo(full_name, def, string());
	loader.RegisterFunction(*primary);
	auto alias = BuildMacroInfo(alias_name, def, full_name);
	loader.RegisterFunction(*alias);
}

} // anonymous namespace

void RegisterPredictMacros(ExtensionLoader &loader) {
	RegisterMacroWithAlias(loader, "anofox_tabfm_classify", "tabfm_classify", CLASSIFY_MACRO);
	RegisterMacroWithAlias(loader, "anofox_tabfm_regress", "tabfm_regress", REGRESS_MACRO);
}

} // namespace anofox
} // namespace duckdb
