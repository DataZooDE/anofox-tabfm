//===----------------------------------------------------------------------===//
// tabfm_crossval.cpp — WS-G cross-validation macros
//
// Implements: anofox_tabfm_cross_validate / tabfm_cross_validate
//             anofox_tabfm_fold_assign   / tabfm_fold_assign
//
// Spec ref: CV-01..04 (plan 01-04, SQL-API §4 CV surface)
//
// Design notes:
//   * tabfm_fold_assign: a simple table macro that appends fold_id computed
//     as (hash(row_key, seed) % k)::INTEGER — deterministic, order-independent,
//     seed-sensitive, range [0, k-1] (CV-01, Pitfall 7).
//     hash() is stable within a DuckDB version (Assumption A6); fold
//     reassignment on upgrade is expected and non-security.
//
//   * tabfm_cross_validate: a SQL table macro that generates the full k-fold
//     UNION ALL query dynamically at execution time via string_agg over
//     generate_series(0, k-1), then executes the generated SQL with an outer
//     query() call. This avoids C++ loop execution while staying inside the
//     SQL macro form locked by CONTEXT.md (Assumption A5 resolved to macro).
//
//     Per-fold structure (CV-02, Pitfall 2 — leakage prevention):
//       train := SELECT * FROM folds WHERE fold_id != f
//       test  := SELECT * FROM folds WHERE fold_id = f  (NULL-label via two-table form)
//     Held-out rows NEVER appear in the training context; they arrive as
//     NULL-label test rows via tabfm_classify/tabfm_regress 'test :=' param.
//
//     Output schema (CV-03): (fold_id INTEGER, metric_value DOUBLE, metric_std DOUBLE)
//       per-fold rows : fold_id in [0, k-1], metric_std = NULL
//       aggregate row : fold_id = -1, metric_value = mean, metric_std = stddev_pop
//
//   * CV-04: EVERY interpolation of the target identifier into generated SQL
//     uses '"' || replace(target, '"', '""') || '"' double-quote wrapping,
//     matching the quoting pattern in tabfm_macros.cpp:93.
//
// T-01-04-01 mitigation: replace(target, '"', '""') at every interpolation.
// T-01-04-02 mitigation: two-table predict form per fold (test := held-out).
//===----------------------------------------------------------------------===//

#include "tabfm_crossval.hpp"
#include "tabfm_registration.hpp"

#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "telemetry.hpp"

namespace duckdb {
namespace anofox {

namespace {

// ── Macro definition struct ──────────────────────────────────────────────────

struct CVMacroDef {
	const char *parameters[8];     // positional param names, nullptr-terminated
	const char *default_params[8]; // "name=sql_default" for trailing optionals
	const char *body;
	const char *description;
	const char *example;
};

// ── tabfm_fold_assign ────────────────────────────────────────────────────────
//
// Parameters: data (required), k (required), row_key (required), seed (required)
//
// Body: Appends fold_id = (hash(row_key, seed) % CAST(k AS UBIGINT))::INTEGER
// to every row of the input relation (CV-01).
//
// The hash() function is variadic ANY->UBIGINT, using seed as a second argument
// so hash(row_key, seed) combines both for a seed-sensitive, order-independent
// result (Pitfall 7: explicit UBIGINT cast avoids signed-modulo surprises).

// clang-format off
static const CVMacroDef FOLD_ASSIGN_MACRO = {
    {"data", "k", "row_key", "seed", nullptr},
    {nullptr},
R"(
    SELECT *, (hash(_cv_rk, CAST(seed AS BIGINT)) % CAST(k AS UBIGINT))::INTEGER AS fold_id
    FROM query('SELECT *, (' || row_key || ') AS _cv_rk FROM (FROM ' || data || ')')
)",
    "Assign each row of `data` to one of k deterministic folds using "
    "(hash(row_key, seed) % k)::INTEGER. Output: all input columns plus fold_id INTEGER in [0, k-1]. "
    "Deterministic and order-independent — only the row_key expression determines the fold, not row position. "
    "hash() is stable within a DuckDB version; fold assignments change on DuckDB upgrade (Assumption A6). "
    "CV-01 compliant: not row_number()-based.",
    "SELECT id, label, fold_id FROM tabfm_fold_assign('my_table', 5, 'id', 42);"
};

// ── tabfm_cross_validate ─────────────────────────────────────────────────────
//
// Parameters (positional): data, target, row_key
// Optional (named with defaults): k=5, seed=42, task='classification',
//                                 metric=NULL (auto-selected based on task)
//
// The macro body builds the k-fold UNION ALL SQL at execution time:
//   1. For each fold f in generate_series(0, k-1):
//      train := (SELECT * FROM folds WHERE _cv_fold_id <> f)
//      test  := (SELECT * FROM folds WHERE _cv_fold_id = f)
//      → tabfm_classify(train, target, test := test) or tabfm_regress(...)
//      → aggregate: metric_fn(target_col, yhat) AS metric_value
//   2. UNION ALL per-fold rows with fold_id = f, metric_std = NULL
//   3. Append aggregate row: fold_id = -1, mean(metric_value), stddev_pop(metric_value)
//
// CV-02: leakage-safe — held-out rows arrive as NULL-label test rows, never
//        as training context (two-table form enforces this).
// CV-03: per-fold rows + aggregate mean±std row.
// CV-04: target identifier double-quoted and embedded quotes escaped everywhere.
//
// Implementation approach (from 01-RESEARCH.md §Pattern 5 + Open Question 1):
// DuckDB SQL macros cannot loop. We use query(string_agg(...)) to generate a
// UNION ALL for k folds via generate_series at execution time. The generated SQL
// contains nested string literals; single-quote boundaries are managed with ''''
// patterns (in SQL: '''' = one single-quote character). See comments inline.
//
// Quoting legend for the generated SQL strings (all inside string_agg):
//   ''''                         = one ' in output (start/end of SQL string arg)
//   '"' || replace(t,'"','""') || '"'  = "target" with embedded double-quotes escaped
//   '''' || expr || ''''         = 'expr_value' in output (SQL string literal)
//
// Security: T-01-04-01 (SQL injection via target identifier) mitigated by
// replace(target, '"', '""') at every interpolation (my"col test, CV-04).

static const CVMacroDef CROSS_VALIDATE_MACRO = {
    {"data", "target", "row_key", nullptr},
    {"k=5", "seed=42", "task='classification'", "metric=NULL", nullptr},
R"(
    SELECT * FROM query(
      -- Build the full k-fold UNION ALL SQL as a scalar string expression.
      -- No subquery — query() does not allow subquery arguments in DuckDB 1.5.4.
      --
      -- Strategy: list_transform(range(k), f -> per_fold_sql) generates k SQL
      -- strings (one per fold) joined with UNION ALL, wrapped in a CTE _cv_folds
      -- inside the generated SQL, then appended with an aggregate row.
      --
      -- Architecture — why the JOIN is needed (leakage / accuracy contract):
      --   tabfm_classify in two-table form forces the test-fold label to NULL
      --   (UNION ALL BY NAME SELECT *, NULL AS "target" FROM test).
      --   So the label column in the classify output is NULL for test rows and
      --   cannot be used directly with tabfm_accuracy.
      --   Fix: we JOIN the classify output back to the original fold data on
      --   row_key to recover the actual labels for the metric computation.
      --
      -- Per-fold SQL fragment (CV-02, leakage-safe, CV-03, CV-04):
      --   SELECT f AS fold_id, m AS metric_value, NULL AS metric_std
      --   FROM (
      --     SELECT metric_fn(orig.__cv_actual, p.yhat) AS m
      --     FROM tabfm_classify(
      --       '(SELECT * FROM fold_data WHERE fold_id <> f)',    -- train
      --       'target',
      --       test := '(SELECT * EXCLUDE (target) FROM fold_data WHERE fold_id = f)'
      --     ) p
      --     JOIN (
      --       SELECT (row_key) AS __cv_rk, "target" AS __cv_actual
      --       FROM fold_data WHERE fold_id = f
      --     ) orig ON p.(row_key) = orig.__cv_rk
      --   )
      --
      -- fold_data = (SELECT *, (hash(row_key, seed) % k)::INTEGER AS _cv_fold_id
      --              FROM (FROM data))
      --
      -- CV-04: replace(target, '"', '""') at EVERY interpolation.
      -- T-01-04-01: replace(target,'"','""') protects double-quote injection.
      -- T-01-04-02: test := held-out fold ensures no preprocessing leakage.
      'WITH _cv_folds AS ('
      || array_to_string(
           list_transform(
             range(CAST(CAST(k AS BIGINT) AS BIGINT)),
             f -> (
               'SELECT ' || CAST(f AS VARCHAR) || ' AS fold_id,'
               ' m AS metric_value, NULL::DOUBLE AS metric_std'
               ' FROM (SELECT ' ||
               -- metric function (explicit or task-defaulted)
               coalesce(nullif(CAST(metric AS VARCHAR), ''),
                 CASE WHEN CAST(task AS VARCHAR) = 'regression'
                      THEN 'tabfm_rmse' ELSE 'tabfm_accuracy' END) ||
               '(orig.__cv_actual, p.yhat) AS m'
               ' FROM ' ||
               -- predict function (classify or regress based on task)
               (CASE WHEN CAST(task AS VARCHAR) = 'regression'
                     THEN 'tabfm_regress' ELSE 'tabfm_classify' END) ||
               '(' ||
               -- first arg: training subquery (all folds EXCEPT the held-out fold f)
               -- Generates: '(SELECT * FROM (SELECT *, hash_expr AS _cv_fold_id FROM data) WHERE _cv_fold_id <> f)'
               -- Note: closing ')' is INSIDE the SQL string literal (before the closing '''').
               '''' ||
               '(SELECT * FROM (SELECT *, (hash(' || CAST(row_key AS VARCHAR)
               -- Emit CAST(seed AS BIGINT) in generated SQL to match tabfm_fold_assign
               -- which uses CAST(seed AS BIGINT) — DuckDB hash() is type-sensitive and
               -- hash(x,42::INTEGER) != hash(x,42::BIGINT) in general (WR-02).
               || ', CAST(' || CAST(CAST(seed AS BIGINT) AS VARCHAR) || ' AS BIGINT)'
               -- Emit CAST(k AS UBIGINT) to match fold_assign's modulo expression.
               || ') % CAST(' || CAST(CAST(k AS BIGINT) AS VARCHAR) || ' AS UBIGINT)'
               || ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'
               || ' WHERE _cv_fold_id <> ' || CAST(f AS VARCHAR) || ')' ||
               '''' ||
               -- second arg: target column name — bare identifier, tabfm_classify
               -- double-quotes it internally. Single-quotes escaped with ''''.
               ', ' || '''' || replace(CAST(target AS VARCHAR), '''', '''''') || '''' ||
               -- test arg: held-out fold WITHOUT target column (EXCLUDE prevents duplicate
               -- column in tabfm_classify's internal UNION ALL BY NAME).
               -- Actual labels recovered via JOIN below.
               ', test := ' ||
               '''' ||
               '(SELECT * EXCLUDE ("' || replace(CAST(target AS VARCHAR), '"', '""')
               || '") FROM (SELECT *, (hash(' || CAST(row_key AS VARCHAR)
               || ', CAST(' || CAST(CAST(seed AS BIGINT) AS VARCHAR) || ' AS BIGINT)'
               || ') % CAST(' || CAST(CAST(k AS BIGINT) AS VARCHAR) || ' AS UBIGINT)'
               || ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'
               || ' WHERE _cv_fold_id = ' || CAST(f AS VARCHAR) || ')' ||
               '''' ||
               ') p JOIN ('
               -- JOIN back to original fold data to recover actual labels (lost when
               -- tabfm_classify forces label=NULL for test rows in two-table form).
               -- row_key is the join key; orig.__cv_actual is the ground-truth label.
               ' SELECT (' || CAST(row_key AS VARCHAR) || ') AS __cv_rk,'
               ' "' || replace(CAST(target AS VARCHAR), '"', '""') || '" AS __cv_actual'
               ' FROM (SELECT *, (hash(' || CAST(row_key AS VARCHAR)
               || ', CAST(' || CAST(CAST(seed AS BIGINT) AS VARCHAR) || ' AS BIGINT)'
               || ') % CAST(' || CAST(CAST(k AS BIGINT) AS VARCHAR) || ' AS UBIGINT)'
               || ')::INTEGER AS _cv_fold_id FROM (FROM ' || CAST(data AS VARCHAR) || '))'
               || ' WHERE _cv_fold_id = ' || CAST(f AS VARCHAR) || ''
               ') orig ON p.' || CAST(row_key AS VARCHAR) || ' = orig.__cv_rk)'
             )
           ),
           ' UNION ALL '
         ) ||
      ') SELECT * FROM _cv_folds'
      || ' UNION ALL SELECT -1 AS fold_id,'
      || ' avg(metric_value) AS metric_value,'
      || ' stddev_pop(metric_value) AS metric_std'
      || ' FROM _cv_folds'
    )
)",
    "Run leakage-safe k-fold cross-validation in SQL. Assigns rows to k folds via "
    "hash(row_key, seed) % k, then for each fold trains tabfm_classify (or tabfm_regress) "
    "on the other k-1 folds and predicts the held-out fold via the two-table form. "
    "Returns k per-fold rows (fold_id 0..k-1) plus one aggregate row (fold_id=-1) with "
    "mean and stddev of the per-fold metric. "
    "Default metric: tabfm_accuracy (classification) or tabfm_rmse (regression). "
    "Target identifier is safely double-quoted to prevent SQL injection (CV-04). "
    "CV-01: hash-based fold assignment (deterministic, order-independent, seed-sensitive). "
    "CV-02: leakage-safe — held-out rows arrive as NULL-label test rows, never as training context. "
    "CV-03: per-fold rows + aggregate mean±std row. CV-04: safe target quoting.",
    "SELECT * FROM tabfm_cross_validate('patients', 'diagnosis', 'patient_id', k := 5, seed := 42);"
};
// clang-format on

// ── Macro registration helpers (exact analog of tabfm_macros.cpp) ────────────

unique_ptr<MacroFunction> BuildTableMacroFunction(const CVMacroDef &def) {
	Parser parser;
	parser.ParseQuery(def.body);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw InternalException("tabfm CV macro body must be a single SELECT statement");
	}
	auto node = std::move(parser.statements[0]->Cast<SelectStatement>().node);
	auto function = make_uniq<TableMacroFunction>(std::move(node));

	// Positional required parameters
	for (idx_t i = 0; def.parameters[i] != nullptr; i++) {
		function->parameters.push_back(make_uniq<ColumnRefExpression>(def.parameters[i]));
		function->types.push_back(LogicalType::UNKNOWN);
	}
	// Trailing optional parameters ("name=sql_default")
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

unique_ptr<CreateMacroInfo> BuildMacroInfo(const string &name, const CVMacroDef &def, const string &alias_of,
                                           const vector<string> &param_names) {
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->schema = DEFAULT_SCHEMA;
	info->name = name;
	info->temporary = true;
	info->internal = true;
	info->alias_of = alias_of;
	info->macros.push_back(BuildTableMacroFunction(def));

	// FunctionDescription required by tabfm_function_docs.test (CLAUDE.md rule #3)
	FunctionDescription fd;
	fd.parameter_names = param_names;
	fd.description = def.description;
	if (def.example) {
		fd.examples = {def.example};
	}
	info->descriptions.push_back(std::move(fd));
	return info;
}

void RegisterCVMacroWithAlias(ExtensionLoader &loader, const string &full_name, const string &alias_name,
                              const CVMacroDef &def, const vector<string> &param_names) {
	// Telemetry: once per registration (at bind/load time) — CLAUDE.md rule #3.
	// For macros, telemetry fires at load time via registration rather than per-bind
	// (macros don't have a separate bind callback; see tabfm_macros.cpp pattern).
	PostHogTelemetry::Instance().CaptureFunctionExecution(alias_name.c_str());

	auto primary = BuildMacroInfo(full_name, def, string(), param_names);
	loader.RegisterFunction(*primary);
	auto alias = BuildMacroInfo(alias_name, def, full_name, param_names);
	loader.RegisterFunction(*alias);
}

} // anonymous namespace

// ── Public entry point ───────────────────────────────────────────────────────

void RegisterCrossValidateMacros(ExtensionLoader &loader) {
	RegisterCVMacroWithAlias(loader, "anofox_tabfm_fold_assign", "tabfm_fold_assign", FOLD_ASSIGN_MACRO,
	                         {"data", "k", "row_key", "seed"});
	RegisterCVMacroWithAlias(loader, "anofox_tabfm_cross_validate", "tabfm_cross_validate", CROSS_VALIDATE_MACRO,
	                         {"data", "target", "row_key", "k", "seed", "task", "metric"});
}

} // namespace anofox
} // namespace duckdb
