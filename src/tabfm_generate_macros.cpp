#include "tabfm_registration.hpp"

#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb_compat.hpp"
#include "tabfm_table_macro.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

namespace duckdb {
namespace anofox {

// User-facing WS-G surface — two task-specific table functions, built exactly
// like the classify/regress macros in tabfm_macros.cpp:
//
//   tabfm_generate(data, n [, features] [, opts] [, model])
//   tabfm_impute  (data [, columns] [, features] [, opts] [, model])
//
//   * data     — the relation to learn from: a table/view NAME or a
//                parenthesised SQL subquery (spliced after `FROM`).
//   * n        — how many synthetic rows to produce.
//   * columns  — VARCHAR[] of columns to fill (default: every column with NULLs).
//   * features — optional VARCHAR[] restricting which columns are modelled at
//                all; matched case-insensitively. The escape hatch for a column
//                that cannot be a target (high-cardinality ids, timestamps).
//   * opts     — MAP(VARCHAR,VARCHAR): seed, temperature, bins, column_order,
//                rounds, model.
//
// tabfm_generate returns the input columns plus `synthetic_id` (1..n);
// tabfm_impute returns exactly the input columns, so it round-trips:
//   CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw');
//
// Mechanics, inherited from the predict macros (all validated on DuckDB 1.5.x):
//   * table macros cannot take a relation argument, so `data` is a string and
//     the relation is assembled and executed with `query(<sql string>)`.
//   * arity-only dispatch forces ONE signature with trailing defaults.
//   * unnest(res, max_depth := 3) — NEVER recursive := true (S04: recursive
//     flattens user STRUCT columns away).
//   * the whole-row struct is aliased `anofox_tabfm_row` so a user column named
//     `t` cannot shadow it.

namespace {

struct GenerateMacroDef {
	const char *parameters[8];
	const char *default_params[8];
	const char *body;
	const char *description;
	const char *example;
	const char *parameter_names[8];
};

// clang-format off
static const GenerateMacroDef GENERATE_MACRO = {
    {"data", "n", nullptr},
    {"features=NULL", "opts=MAP{}", "model=NULL", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT __anofox_tabfm_generate_agg(
               anofox_tabfm_row, n,
               map_concat(
                 CAST(opts AS MAP(VARCHAR, VARCHAR)),
                 map_concat(
                   CASE WHEN model IS NULL THEN MAP{}::MAP(VARCHAR, VARCHAR)
                        ELSE MAP{'model': model::VARCHAR} END,
                   -- See the predict macros: the COLUMNS(lambda) filter drops a
                   -- name that matches nothing, so a misspelling would silently
                   -- generate FEWER columns than asked for. Forward the request.
                   CASE WHEN features IS NULL THEN MAP{}::MAP(VARCHAR, VARCHAR)
                        ELSE MAP{'__features': array_to_string(features, chr(31))} END))) AS res
      FROM (
        SELECT COLUMNS(lambda c: features IS NULL
                                 OR list_contains(list_transform(features, lambda f: lcase(f)), lcase(c)))
        FROM query('FROM ' || data)
      ) anofox_tabfm_row
    )
)",
    "Generate synthetic rows from the joint distribution of `data` using a tabular foundation model. Factorizes the "
    "table column by column (the chain rule) and samples each column conditioned on the ones already generated, so "
    "correlations between columns are preserved rather than each column being drawn independently. Returns `n` rows "
    "with the same columns as `data` plus `synthetic_id`. Continuous columns are sampled via quantile bins, so values "
    "stay inside the observed range. Costs one model call per column, run sequentially. Options: seed, temperature "
    "(higher = more diverse), bins, column_order, model.",
    "SELECT * FROM tabfm_generate('customers', 100);",
    {"data", "n", "features", "opts", "model", nullptr}};

static const GenerateMacroDef IMPUTE_MACRO = {
    {"data", nullptr},
    {"columns=NULL", "features=NULL", "opts=MAP{}", "model=NULL", nullptr},
R"(
    SELECT unnest(res, max_depth := 3) FROM (
      SELECT __anofox_tabfm_impute_agg(
               anofox_tabfm_row, columns,
               map_concat(
                 CAST(opts AS MAP(VARCHAR, VARCHAR)),
                 map_concat(
                   CASE WHEN model IS NULL THEN MAP{}::MAP(VARCHAR, VARCHAR)
                        ELSE MAP{'model': model::VARCHAR} END,
                   -- See the predict macros: the COLUMNS(lambda) filter drops a
                   -- name that matches nothing, so a misspelling would silently
                   -- generate FEWER columns than asked for. Forward the request.
                   CASE WHEN features IS NULL THEN MAP{}::MAP(VARCHAR, VARCHAR)
                        ELSE MAP{'__features': array_to_string(features, chr(31))} END))) AS res
      FROM (
        SELECT COLUMNS(lambda c: features IS NULL
                                 OR list_contains(list_transform(features, lambda f: lcase(f)), lcase(c)))
        FROM query('FROM ' || data)
      ) anofox_tabfm_row
    )
)",
    "Fill the NULL cells of `data` with a tabular foundation model, conditioning each missing value on the other "
    "columns of its row. A NaN in a FLOAT/DOUBLE column being filled counts as missing, like NULL; +/-Infinity there "
    "is an error. Returns the same columns as `data`, with every other cell untouched, so it round-trips: "
    "CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw'). Unlike tabfm_generate this does not sample — it takes "
    "the conditional best estimate (classification argmax, regression point estimate), so continuous columns keep "
    "full precision. Optional `columns` restricts which columns are filled; `opts` accepts seed, rounds (MICE-style "
    "refinement sweeps), model.",
    "SELECT * FROM tabfm_impute('customers', columns := ['income']);",
    {"data", "columns", "features", "opts", "model", nullptr}};
// clang-format on

unique_ptr<CreateMacroInfo> BuildMacroInfo(const string &name, const GenerateMacroDef &def, const string &alias_of) {
	auto info = MakeTableMacroInfo(name, def.parameters, def.default_params, def.body);
	info->temporary = true;
	info->internal = true;
	info->alias_of = duckdb_compat::Ident(alias_of);

	FunctionDescription fd;
	for (idx_t i = 0; def.parameter_names[i] != nullptr; i++) {
		fd.parameter_names.emplace_back(def.parameter_names[i]);
	}
	fd.description = def.description;
	if (def.example) {
		fd.examples = {def.example};
		fd.categories = {"tabfm", "generate"};
	}
	info->descriptions.push_back(std::move(fd));
	return info;
}

void RegisterMacroWithAlias(ExtensionLoader &loader, const string &full_name, const string &alias_name,
                            const GenerateMacroDef &def) {
	auto primary = BuildMacroInfo(full_name, def, string());
	loader.RegisterFunction(*primary);
	auto alias = BuildMacroInfo(alias_name, def, full_name);
	loader.RegisterFunction(*alias);
}

} // anonymous namespace

void RegisterGenerateMacros(ExtensionLoader &loader) {
	RegisterMacroWithAlias(loader, "anofox_tabfm_generate", "tabfm_generate", GENERATE_MACRO);
	RegisterMacroWithAlias(loader, "anofox_tabfm_impute", "tabfm_impute", IMPUTE_MACRO);
}

} // namespace anofox
} // namespace duckdb
