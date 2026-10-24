//===----------------------------------------------------------------------===//
// tabfm_contract.cpp — the anofox integration contract
//
// anofox-visualization plots any table that follows the contract's long-form
// schemas (anofox-visualization docs/plans/integration-contract.md, v1). This
// module adds tabfm's producers and the contract's discovery surface:
//
//   tabfm_classification_metrics_by(data, group_col, actual_col, predicted_col, proba_col)
//                                                            -> summary
//   tabfm_regression_metrics_by(data, group_col, actual_col, predicted_col)
//                                                            -> summary
//   tabfm_confusion_matrix_by(data, group_col, actual_col, predicted_col)
//                                                            -> matrix
//   tabfm_augment_by(data, group_col, actual_col, predicted_col [, row_id_col])
//                                                            -> obs
//   tabfm_roc_curve_by(data, group_col, actual_col, proba_col
//                      [, pos_label, drop_intermediate := true])  -> curve
//   tabfm_calibration_curve_by(data, group_col, actual_col, proba_col
//                              [, pos_label, n_bins := 10])       -> curve
//   anofox_contract_version()                                -> '1'
//
// The producers are table macros in tabfm's style (tabfm_confusion_matrix):
// `data` is a table name or a '(SELECT …)' string, the column arguments are
// column names, and every identifier is double-quote escaped before it is
// interpolated into the query() text. They are plain SQL over DuckDB built-ins
// and tabfm's own metric aggregates, so they build the same on every DuckDB
// line and report exactly what the aggregates report.
//
// Discovery: every function this extension registers is tagged
// anofox.family = 'tabfm' in duckdb_functions(); the producers also carry
// anofox.contract = '1' and anofox.output = <schema>. LoadInternal snapshots
// the catalog's function names before registering, so the tagging pass leaves
// DuckDB's and other extensions' functions alone.
//
// Plan: docs/ANOFOX_CONTRACT_PLAN.md.
//===----------------------------------------------------------------------===//

#include "tabfm_registration.hpp"
#include "tabfm_table_macro.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/transaction/meta_transaction.hpp"
#include "duckdb_compat.hpp"

#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

namespace {

constexpr const char *ANOFOX_CONTRACT_VERSION = "1";

constexpr CatalogType FUNCTION_CATALOG_TYPES[] = {
    CatalogType::SCALAR_FUNCTION_ENTRY, CatalogType::AGGREGATE_FUNCTION_ENTRY, CatalogType::TABLE_FUNCTION_ENTRY,
    CatalogType::MACRO_ENTRY, CatalogType::TABLE_MACRO_ENTRY};

// Contract producers: primary name -> output schema (the tabfm_* aliases are tagged the same).
// scripts/visualization_pickup.py keeps its own copy on purpose.
const struct {
	const char *name;
	const char *output;
} CONTRACT_PRODUCERS[] = {{"anofox_tabfm_classification_metrics_by", "summary"},
                          {"anofox_tabfm_regression_metrics_by", "summary"},
                          {"anofox_tabfm_confusion_matrix_by", "matrix"},
                          {"anofox_tabfm_augment_by", "obs"},
                          {"anofox_tabfm_roc_curve_by", "curve"},
                          {"anofox_tabfm_calibration_curve_by", "curve"}};

// A catalog entry's name: a string member before DuckDB 2.0, GetName() from 2.0 on.
template <class T>
auto EntryName(T &x, int) -> decltype(x.GetName()) {
	return x.GetName();
}
template <class T>
auto EntryName(T &x, long) -> decltype((x.name)) {
	return x.name;
}
template <class T>
std::string NameOf(T &x) {
	return static_cast<const std::string &>(EntryName(x, 0));
}

SchemaCatalogEntry &SystemMainSchema(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	return Catalog::GetSystemCatalog(db).GetSchema(CatalogTransaction::GetSystemTransaction(db), DEFAULT_SCHEMA);
}

// ── the producers ────────────────────────────────────────────────────────────
//
// Each body is a query() over a string built from the macro's arguments. Q(p)
// is the SQL that double-quotes the identifier in parameter p.

std::string Q(const char *param) {
	return std::string("'\"' || replace(") + param + ", '\"', '\"\"') || '\"'";
}

// The rows of `data` with the group as _g and the actual / predicted values as
// numbers _y / _f. A value that is not a number is an error naming fn; NULL and
// non-finite values (a scored row, NaN) stay in and are skipped by the caller.
std::string NumericRows(const std::string &fn, const std::string &extra_columns) {
	// Both values go through the check: a filter on either one (pushed below the projection by the optimizer)
	// still evaluates it, so a non-numeric column cannot be filtered away silently.
	const std::string bad = "(_ya IS NOT NULL AND TRY_CAST(_ya AS DOUBLE) IS NULL) OR (_fa IS NOT NULL AND TRY_CAST(_fa AS "
	                        "DOUBLE) IS NULL)";
	const std::string error = "error(''" + fn +
	                          ": the actual and predicted columns must be numeric, got '' || _ya::VARCHAR || '' and '' "
	                          "|| _fa::VARCHAR || ''. Fix: pass the regression target and yhat (tabfm_regress); for "
	                          "class labels use tabfm_classification_metrics_by or tabfm_confusion_matrix_by.'')";
	return "'WITH _r AS (SELECT ' || " + Q("group_col") + " || ' AS _g, ' || " + Q("actual_col") +
	       " || ' AS _ya, ' || " + Q("predicted_col") + " || ' AS _fa' || " + extra_columns +
	       " || ' FROM (FROM ' || data || ')), '"
	       " || '_v AS (SELECT *, CASE WHEN " +
	       bad + " THEN " + error + " ELSE TRY_CAST(_ya AS DOUBLE) END AS _y, CASE WHEN " + bad + " THEN " + error +
	       " ELSE TRY_CAST(_fa AS DOUBLE) END AS _f FROM _r), '"
	       " || '_ok AS (SELECT * FROM _v WHERE _y IS NOT NULL AND _f IS NOT NULL AND isfinite(_y) AND isfinite(_f)) '";
}

std::string SummaryTail(const std::string &metrics) {
	return " || ', _s AS (SELECT _g, unnest([" + metrics +
	       "]) AS _x FROM _ok GROUP BY _g) '"
	       " || 'SELECT _g AS ' || " +
	       Q("group_col") +
	       " || ', _g::VARCHAR AS model_id, _x.metric AS metric, _x.value AS value, NULL::DOUBLE AS conf_low, "
	       "NULL::DOUBLE AS conf_high FROM _s WHERE _x.value IS NOT NULL AND isfinite(_x.value) ORDER BY 1, 3'";
}

std::string Metric(const char *name, const char *aggregate) {
	return std::string("struct_pack(metric := ''") + name + "'', value := " + aggregate + ")";
}

// Dyn(sql): the SQL expression that builds `sql` as query() text. Plain SQL in, with placeholders:
//   {data}            the data argument as given (a table name or a '(SELECT …)' string)
//   {g} {a} {pr}      the group / actual / proba column, double-quoted
//   {s:param}         the parameter as a SQL string literal, or NULL
//   {v:param}         the parameter's value as SQL text (a number or boolean)
// Single quotes in `sql` are doubled; the template itself must not contain braces.
std::string DynExpr(const std::string &token) {
	if (token == "data") {
		return "data";
	}
	if (token == "g") {
		return Q("group_col");
	}
	if (token == "a") {
		return Q("actual_col");
	}
	if (token == "pr") {
		return Q("proba_col");
	}
	const auto param = token.substr(2);
	if (token.rfind("s:", 0) == 0) {
		return "CASE WHEN " + param + " IS NULL THEN 'NULL' ELSE '''' || replace(" + param +
		       "::VARCHAR, '''', '''''') || '''' END";
	}
	if (token.rfind("v:", 0) == 0) {
		return "coalesce(" + param + "::VARCHAR, 'NULL')";
	}
	throw InternalException("tabfm_contract: unknown placeholder {%s}", token);
}

std::string Dyn(const std::string &sql) {
	std::string out = "'";
	for (size_t i = 0; i < sql.size(); i++) {
		if (sql[i] == '{') {
			const auto end = sql.find('}', i);
			out += "' || " + DynExpr(sql.substr(i + 1, end - i - 1)) + " || '";
			i = end;
		} else {
			out += sql[i] == '\'' ? std::string("''") : std::string(1, sql[i]);
		}
	}
	return out + "'";
}

// The one-vs-rest rows of a classification table: one row per (input row, class) with _g, the class _k, its
// probability _p and _y = 1 when the row's label is that class. The classes drawn: pos_label when given;
// otherwise the greater label of a two-class group (scikit-learn's classes_[1]; the other curve is its mirror
// image) and every class of a larger one.
std::string OneVsRestRows(const std::string &fn) {
	return "WITH _r AS (SELECT {g} AS _g, {a}::VARCHAR AS _a, CAST({pr} AS MAP(VARCHAR, DOUBLE)) AS _m "
	       "FROM (FROM {data})), "
	       "_v AS (SELECT _g, _a, CASE WHEN list_min(map_values(_m)) < 0 OR list_max(map_values(_m)) > 1 "
	       "OR NOT isfinite(list_sum(map_values(_m))) THEN error('" +
	       fn +
	       ": every probability must be a finite number in [0, 1], got ' || _m::VARCHAR || '. Fix: pass the proba "
	       "column tabfm_classify returns.') ELSE _m END AS _m FROM _r WHERE _a IS NOT NULL AND _m IS NOT NULL), "
	       "_e AS (SELECT _g, _a, unnest(map_entries(_m)) AS _kv FROM _v), "
	       "_l AS (SELECT _g, _kv.key AS _k, _kv.value AS _p, (_a = _kv.key)::INTEGER AS _y FROM _e "
	       "WHERE _kv.value IS NOT NULL), "
	       "_cls AS (SELECT _g, _k, count(*) OVER (PARTITION BY _g) AS _nk, max(_k) OVER (PARTITION BY _g) AS _kmax "
	       "FROM (SELECT DISTINCT _g, _k FROM _l)), "
	       "_sel AS (SELECT _l.* FROM _l JOIN _cls ON _l._g IS NOT DISTINCT FROM _cls._g AND _l._k = _cls._k "
	       "WHERE CASE WHEN {s:pos_label} IS NOT NULL THEN _l._k = {s:pos_label} WHEN _nk = 2 THEN _l._k = _kmax "
	       "ELSE true END) ";
}

// scikit-learn roc_curve per (group, class): tied probabilities are one point, the curve starts at (0, 0)
// with threshold +inf, and drop_intermediate drops the points on straight segments. A class without
// positives or without negatives in a group has no curve.
std::string RocCurveBody() {
	return "SELECT * FROM query(" +
	       Dyn(OneVsRestRows("tabfm_roc_curve_by") +
	           ", _pts AS (SELECT _g, _k, _thr, _i, _tps, _fps, _n_pos, _n_neg, _i = 1 OR _i = _n "
	           "OR lag(_fps) OVER _w - 2 * _fps + lead(_fps) OVER _w <> 0 "
	           "OR lag(_tps) OVER _w - 2 * _tps + lead(_tps) OVER _w <> 0 AS _keep "
	           "FROM (SELECT _g, _k, _thr, row_number() OVER _c AS _i, count(*) OVER (PARTITION BY _g, _k) AS _n, "
	           "sum(_tp) OVER _c AS _tps, sum(_fp) OVER _c AS _fps, sum(_tp) OVER (PARTITION BY _g, _k) AS _n_pos, "
	           "sum(_fp) OVER (PARTITION BY _g, _k) AS _n_neg "
	           "FROM (SELECT _g, _k, _p AS _thr, sum(_y) AS _tp, sum(1 - _y) AS _fp FROM _sel GROUP BY _g, _k, _p) "
	           "WINDOW _c AS (PARTITION BY _g, _k ORDER BY _thr DESC ROWS UNBOUNDED PRECEDING)) "
	           "WINDOW _w AS (PARTITION BY _g, _k ORDER BY _i)) "
	           "SELECT _g AS {g}, _g::VARCHAR AS model_id, _k AS series, 'roc'::VARCHAR AS curve_type, x, y, "
	           "NULL::DOUBLE AS y_low, NULL::DOUBLE AS y_high, threshold, NULL::DOUBLE AS weight, "
	           "NULL::BIGINT AS n_censor FROM ("
	           "SELECT _g, _k, 0 AS _i, 0::DOUBLE AS x, 0::DOUBLE AS y, 'inf'::DOUBLE AS threshold FROM _pts "
	           "WHERE _i = 1 AND _n_pos > 0 AND _n_neg > 0 "
	           "UNION ALL SELECT _g, _k, _i, _fps / _n_neg, _tps / _n_pos, _thr FROM _pts "
	           "WHERE _n_pos > 0 AND _n_neg > 0 AND (_keep OR NOT {v:drop_intermediate})) ORDER BY 1, 3, _i") +
	       ")";
}

// The reliability curve per (group, class): n_bins equal-width bins on the class probability (tabfm_ece's
// edge rule: floor(p * n_bins), p = 1 in the last bin); x is the bin's mean probability, y its fraction of
// rows with that label, y_low / y_high the 95% Wilson interval, weight the bin's row count. Empty bins are
// left out.
std::string CalibrationCurveBody() {
	return "SELECT * FROM query(" +
	       Dyn(OneVsRestRows("tabfm_calibration_curve_by") +
	           "SELECT _g AS {g}, _g::VARCHAR AS model_id, _k AS series, 'calibration'::VARCHAR AS curve_type, "
	           "_x AS x, _f AS y, greatest(0::DOUBLE, _c - _h) AS y_low, least(1::DOUBLE, _c + _h) AS y_high, "
	           "NULL::DOUBLE AS threshold, _n::DOUBLE AS weight, NULL::BIGINT AS n_censor "
	           "FROM (SELECT *, (_f + _z2 / (2 * _n)) / (1 + _z2 / _n) AS _c, "
	           "sqrt(_z2 * (_f * (1 - _f) / _n + _z2 / (4 * _n * _n))) / (1 + _z2 / _n) AS _h "
	           "FROM (SELECT _g, _k, _b, avg(_p) AS _x, avg(_y) AS _f, count(*) AS _n, "
	           "pow(1.959963984540054::DOUBLE, 2) AS _z2 "
	           "FROM (SELECT *, least(floor(_p * _nb), _nb - 1)::BIGINT AS _b FROM _sel, "
	           "(SELECT CASE WHEN {v:n_bins}::BIGINT < 1 THEN error('tabfm_calibration_curve_by: n_bins must be at "
	           "least 1, got {v:n_bins}. Fix: pass n_bins := 10 (the default) or another positive count.') "
	           "ELSE {v:n_bins}::BIGINT END AS _nb)) "
	           "GROUP BY _g, _k, _b)) ORDER BY 1, 3, _b") +
	       ")";
}

std::string ClassificationMetricsBody() {
	const std::string fn = "tabfm_classification_metrics_by";
	const std::string metrics = Metric("accuracy", "tabfm_accuracy(_a, _p)") + ", " +
	                            Metric("ece", "tabfm_ece(_a, _m)") + ", " +
	                            Metric("f1_macro", "tabfm_f1(_a, _p, ''macro'')") + ", " +
	                            Metric("log_loss", "tabfm_log_loss(_a, _m)") + ", " +
	                            Metric("roc_auc_ovr", "tabfm_roc_auc(_a, _m, ''ovr'')");
	return "SELECT * FROM query('WITH _r AS (SELECT ' || " + Q("group_col") + " || ' AS _g, ' || " +
	       Q("actual_col") + " || '::VARCHAR AS _a, ' || " + Q("predicted_col") +
	       " || '::VARCHAR AS _p, CAST(' || " + Q("proba_col") +
	       " || ' AS MAP(VARCHAR, DOUBLE)) AS _m FROM (FROM ' || data || ')), '"
	       " || '_ok AS (SELECT _g, _a, _p, CASE WHEN list_min(map_values(_m)) < 0 OR list_max(map_values(_m)) > 1 "
	       "OR NOT isfinite(list_sum(map_values(_m))) THEN error(''" +
	       fn +
	       ": every probability must be a finite number in [0, 1], got '' || _m::VARCHAR || ''. Fix: pass the "
	       "proba column tabfm_classify returns.'') ELSE _m END AS _m "
	       "FROM _r WHERE _a IS NOT NULL AND _p IS NOT NULL AND _m IS NOT NULL)'" +
	       SummaryTail(metrics) + ")";
}

std::string RegressionMetricsBody() {
	const std::string metrics = Metric("mae", "tabfm_mae(_y, _f)") + ", " + Metric("mape", "tabfm_mape(_y, _f)") +
	                            ", " + Metric("medae", "tabfm_medae(_y, _f)") + ", " +
	                            Metric("r2", "tabfm_r2(_y, _f)") + ", " + Metric("rmse", "tabfm_rmse(_y, _f)");
	return "SELECT * FROM query(" + NumericRows("tabfm_regression_metrics_by", "''") + SummaryTail(metrics) + ")";
}

std::string ConfusionMatrixBody() {
	return "SELECT * FROM query('WITH _r AS (SELECT ' || " + Q("group_col") + " || ' AS _g, ' || " +
	       Q("actual_col") + " || '::VARCHAR AS _a, ' || " + Q("predicted_col") +
	       " || '::VARCHAR AS _p FROM (FROM ' || data || ')), '"
	       " || '_ok AS (SELECT * FROM _r WHERE _a IS NOT NULL AND _p IS NOT NULL), '"
	       " || '_l AS (SELECT _g, _a AS _lab FROM _ok UNION SELECT _g, _p FROM _ok), '"
	       " || '_c AS (SELECT _g, _a, _p, count(*) AS _n FROM _ok GROUP BY ALL) '"
	       " || 'SELECT r._g AS ' || " +
	       Q("group_col") +
	       " || ', r._g::VARCHAR AS model_id, r._lab AS \"row\", c._lab AS \"col\", coalesce(_c._n, 0)::BIGINT AS "
	       "value FROM _l r JOIN _l c ON r._g = c._g LEFT JOIN _c ON _c._g = r._g AND _c._a = r._lab AND _c._p = "
	       "c._lab ORDER BY 1, 3, 4')";
}

std::string AugmentBody() {
	// Without row_id_col the id is the row's position in its group (the input order).
	const std::string row_id = "' , ' || CASE WHEN row_id_col IS NULL THEN 'row_number() OVER (PARTITION BY ' || " +
	                           Q("group_col") + " || ')' ELSE " + Q("row_id_col") + " END || ' AS _id'";
	return "SELECT * FROM query(" + NumericRows("tabfm_augment_by", row_id) +
	       " || 'SELECT _g AS ' || " + Q("group_col") +
	       " || ', _g::VARCHAR AS model_id, _id::VARCHAR AS row_id, _y AS y, _f AS fitted, _y - _f AS residual "
	       "FROM _ok ORDER BY 1, _id')";
}

struct ProducerMacro {
	const char *name; // without the anofox_tabfm_ / tabfm_ prefix
	const char *const *parameters;
	const char *const *defaults;
	std::string body;
	vector<std::string> parameter_names;
	const char *description;
	const char *example;
};

unique_ptr<CreateMacroInfo> TableMacro(const std::string &name, const std::string &alias_of, const ProducerMacro &m) {
	auto info = MakeTableMacroInfo(name, m.parameters, m.defaults, m.body.c_str());
	info->temporary = true;
	info->internal = true;
	info->alias_of = duckdb_compat::Ident(alias_of);

	FunctionDescription fd;
	fd.parameter_names = m.parameter_names;
	fd.description = m.description;
	fd.examples = {m.example};
	fd.categories = {"tabfm", "metrics"};
	info->descriptions.push_back(std::move(fd));
	return info;
}

void ContractVersionFunction(DataChunk &, ExpressionState &, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result.SetValue(0, Value(ANOFOX_CONTRACT_VERSION));
}

} // namespace

case_insensitive_set_t SnapshotSystemFunctionNames(ExtensionLoader &loader) {
	case_insensitive_set_t names;
	auto &schema = SystemMainSchema(loader);
	for (auto type : FUNCTION_CATALOG_TYPES) {
		schema.Scan(type, [&](CatalogEntry &entry) { names.insert(NameOf(entry)); });
	}
	return names;
}

void RegisterContractFunctions(ExtensionLoader &loader) {
	// Shared by the whole anofox family: whichever extension loads first registers it (a second identical
	// overload would make this load fail).
	if (!loader.TryGetFunction("anofox_contract_version")) {
		ScalarFunction func("anofox_contract_version", {}, LogicalType::VARCHAR, ContractVersionFunction);
		CreateScalarFunctionInfo info(func);
		FunctionDescription desc;
		desc.description = "Returns the version of the anofox integration contract (output schemas and "
		                   "duckdb_functions() tags) this extension implements.";
		desc.examples = {"SELECT anofox_contract_version();"};
		desc.categories = {"metadata"};
		info.descriptions.push_back(std::move(desc));
		loader.RegisterFunction(std::move(info));
	}

	static const char *const four[] = {"data", "group_col", "actual_col", "predicted_col", nullptr};
	static const char *const five[] = {"data", "group_col", "actual_col", "predicted_col", "proba_col", nullptr};
	static const char *const none[] = {nullptr};
	static const char *const row_id[] = {"row_id_col=NULL", nullptr};
	static const char *const curve[] = {"data", "group_col", "actual_col", "proba_col", nullptr};
	static const char *const roc[] = {"pos_label=NULL", "drop_intermediate=true", nullptr};
	static const char *const calibration[] = {"pos_label=NULL", "n_bins=10", nullptr};

	const ProducerMacro producers[] = {
	    {"classification_metrics_by", five, none, ClassificationMetricsBody(),
	     {"data", "group_col", "actual_col", "predicted_col", "proba_col"},
	     "Table macro: classification metrics per group of data, one row per metric (the anofox integration "
	     "contract's `summary` schema, for anofox_plot): the group column, model_id (the group as text), metric "
	     "(accuracy, ece, f1_macro, log_loss, roc_auc_ovr), value, conf_low, conf_high. The values are tabfm's "
	     "own aggregates (tabfm_accuracy, tabfm_ece, tabfm_f1 with avg 'macro', tabfm_log_loss, tabfm_roc_auc "
	     "with avg 'ovr'). data is a table name or a '(SELECT ...)' string; the column arguments are column "
	     "names. proba_col is the MAP(label, probability) column tabfm_classify returns. A metric that is "
	     "undefined for a group (roc_auc_ovr with one class) is left out. Rows with a NULL label or proba are "
	     "skipped.",
	     "SELECT * FROM tabfm_classification_metrics_by('scored', 'model', 'actual', 'yhat', 'proba');"},
	    {"regression_metrics_by", four, none, RegressionMetricsBody(),
	     {"data", "group_col", "actual_col", "predicted_col"},
	     "Table macro: regression metrics per group of data, one row per metric (the anofox integration "
	     "contract's `summary` schema, for anofox_plot): the group column, model_id (the group as text), metric "
	     "(mae, mape, medae, r2, rmse), value, conf_low, conf_high. The values are tabfm's own aggregates "
	     "(mape skips zero actuals). data is a table name or a '(SELECT ...)' string; the column arguments are "
	     "column names. Rows with a NULL or non-finite actual or prediction (the scored rows of a single-"
	     "relation tabfm_regress) are skipped.",
	     "SELECT * FROM tabfm_regression_metrics_by('fitted', 'model', 'price', 'yhat');"},
	    {"confusion_matrix_by", four, none, ConfusionMatrixBody(),
	     {"data", "group_col", "actual_col", "predicted_col"},
	     "Table macro: the confusion matrix per group of data (the anofox integration contract's `matrix` "
	     "schema, for anofox_plot): the group column, model_id (the group as text), row (actual label), col "
	     "(predicted label), value (row count). Every (actual, predicted) pair of the group's labels is listed, "
	     "zero cells included. Labels are compared as text. data is a table name or a '(SELECT ...)' string; "
	     "the column arguments are column names. tabfm_confusion_matrix gives the observed pairs of one table.",
	     "SELECT * FROM tabfm_confusion_matrix_by('scored', 'model', 'actual', 'yhat');"},
	    {"augment_by", four, row_id, AugmentBody(),
	     {"data", "group_col", "actual_col", "predicted_col", "row_id_col"},
	     "Table macro: actual against predicted per row and group of data (the anofox integration contract's "
	     "`obs` schema, for anofox_plot's predicted-vs-actual and residual plots): the group column, model_id "
	     "(the group as text), row_id, y, fitted, residual (y - fitted). row_id is row_id_col as text, or the "
	     "row's position in its group when row_id_col is not given. data is a table name or a '(SELECT ...)' "
	     "string; the column arguments are column names. Rows with a NULL or non-finite actual or prediction "
	     "are skipped.",
	     "SELECT * FROM tabfm_augment_by('fitted', 'model', 'price', 'yhat', row_id_col := 'id');"},
	    {"roc_curve_by", curve, roc, RocCurveBody(),
	     {"data", "group_col", "actual_col", "proba_col", "pos_label", "drop_intermediate"},
	     "Table macro: the ROC curve of classification probabilities per group of data, one-vs-rest per class "
	     "(scikit-learn roc_curve on actual = class; the anofox integration contract's `curve` schema, curve_type "
	     "'roc', for anofox_plot). One row per point: the group column, model_id (the group as text), series (the "
	     "class), curve_type, x (false positive rate), y (true positive rate), y_low, y_high, threshold (the "
	     "probability; +inf for the (0, 0) start), weight, n_censor. Two classes give one curve, for the greater "
	     "label; more classes give one per class; pos_label := picks one class. The mean area of a group's curves "
	     "is tabfm_roc_auc(avg := 'ovr'). Tied probabilities form one point; drop_intermediate (default true) "
	     "drops the points on straight segments. A class without positives or negatives in a group has no "
	     "curve. data is a table name or a '(SELECT ...)' string; proba_col is the MAP(label, probability) "
	     "column tabfm_classify returns.",
	     "SELECT * FROM tabfm_roc_curve_by('scored', 'model', 'actual', 'proba');"},
	    {"calibration_curve_by", curve, calibration, CalibrationCurveBody(),
	     {"data", "group_col", "actual_col", "proba_col", "pos_label", "n_bins"},
	     "Table macro: the calibration (reliability) curve of classification probabilities per group of data, "
	     "one-vs-rest per class (the anofox integration contract's `curve` schema, curve_type 'calibration', for "
	     "anofox_plot). The class probabilities fall in n_bins equal-width bins (default 10); one row per "
	     "non-empty bin: the group column, model_id (the group as text), series (the class), curve_type, x (mean "
	     "probability), y (fraction of rows with that label), y_low / y_high (95% Wilson interval), threshold, "
	     "weight (rows in the bin), n_censor. Classes are drawn as in tabfm_roc_curve_by. tabfm_ece bins the "
	     "top-label confidence instead, so the curve does not reproduce the ECE number.",
	     "SELECT * FROM tabfm_calibration_curve_by('scored', 'model', 'actual', 'proba', n_bins := 10);"},
	};
	for (auto &m : producers) {
		std::string primary = std::string("anofox_tabfm_") + m.name;
		loader.RegisterFunction(*TableMacro(primary, std::string(), m));
		loader.RegisterFunction(*TableMacro(std::string("tabfm_") + m.name, primary, m));
	}
}

void ApplyContractTags(ExtensionLoader &loader, const case_insensitive_set_t &preexisting) {
	case_insensitive_map_t<std::string> outputs;
	for (auto &p : CONTRACT_PRODUCERS) {
		outputs[p.name] = p.output;
		outputs[std::string(p.name).substr(std::string("anofox_").size())] = p.output;
	}
	auto &schema = SystemMainSchema(loader);
	for (auto type : FUNCTION_CATALOG_TYPES) {
		schema.Scan(type, [&](CatalogEntry &entry) {
			auto name = NameOf(entry);
			if (preexisting.find(name) != preexisting.end() || name == "anofox_contract_version") {
				return;
			}
			entry.tags["anofox.family"] = "tabfm";
			auto output = outputs.find(name);
			if (output != outputs.end()) {
				entry.tags["anofox.contract"] = ANOFOX_CONTRACT_VERSION;
				entry.tags["anofox.output"] = output->second;
			}
		});
	}
}

} // namespace anofox
} // namespace duckdb
