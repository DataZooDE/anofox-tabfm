// test/cpp/test_tabfm_metrics_regression.cpp — Catch2 unit tests for regression
// metric aggregates (tabfm_rmse, tabfm_mae, tabfm_r2, tabfm_mape, tabfm_medae)
// exercised via DuckDB SQL execution.
//
// Spec ref: RMET-01..04, plan 01-03
// Golden values match sklearn (see tools/golden/generate_metric_fixtures.py §RMET-*).

#include "catch.hpp"
#include "duckdb.hpp"

using namespace duckdb;

// Helper: open a fresh DB, load the extension, run a query, return the first
// column of the first row as double.
static double run_scalar_double(Connection &con, const std::string &sql) {
	auto result = con.Query(sql);
	REQUIRE(!result->HasError());
	REQUIRE(result->RowCount() == 1);
	REQUIRE(!result->GetValue(0, 0).IsNull());
	return result->GetValue(0, 0).GetValue<double>();
}

static bool is_null(Connection &con, const std::string &sql) {
	auto result = con.Query(sql);
	REQUIRE(!result->HasError());
	REQUIRE(result->RowCount() == 1);
	return result->GetValue(0, 0).IsNull();
}

TEST_CASE("tabfm_rmse: golden value matches sklearn root_mean_squared_error", "[tabfm][metrics_regression]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES "
	                   "(1.0, 1.1),(2.0, 2.2),(3.0, 2.9),(4.0, 3.8),(5.0, 5.5)"
	                   ") v(actual, predicted)")->HasError());

	double rmse = run_scalar_double(con, "SELECT tabfm_rmse(actual, predicted) FROM t");
	// sklearn root_mean_squared_error = 0.264575...
	REQUIRE(std::abs(rmse - 0.264575) < 1e-5);

	// Alias parity
	double rmse_alias = run_scalar_double(con, "SELECT anofox_tabfm_rmse(actual, predicted) FROM t");
	REQUIRE(rmse == rmse_alias);
}

TEST_CASE("tabfm_rmse: NULL skip and empty input returns NULL", "[tabfm][metrics_regression]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// NULL rows skipped — 3 valid pairs: (1.0,1.1),(2.0,2.2),(5.0,5.5)
	REQUIRE(!con.Query("CREATE OR REPLACE TABLE t_null AS SELECT * FROM (VALUES "
	                   "(1.0, 1.1),(2.0, 2.2),(NULL, 2.9),(4.0, NULL),(5.0, 5.5)"
	                   ") v(actual, predicted)")->HasError());
	double rmse = run_scalar_double(con, "SELECT tabfm_rmse(actual, predicted) FROM t_null");
	REQUIRE(std::abs(rmse - 0.316228) < 1e-5);

	// Empty input → NULL
	REQUIRE(is_null(con, "SELECT tabfm_rmse(actual, predicted) FROM t_null WHERE 1=0"));
}

TEST_CASE("tabfm_mae: golden value matches sklearn mean_absolute_error", "[tabfm][metrics_regression]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES "
	                   "(1.0, 1.1),(2.0, 2.2),(3.0, 2.9),(4.0, 3.8),(5.0, 5.5)"
	                   ") v(actual, predicted)")->HasError());

	double mae = run_scalar_double(con, "SELECT tabfm_mae(actual, predicted) FROM t");
	// sklearn mean_absolute_error = 0.22
	REQUIRE(std::abs(mae - 0.22) < 1e-9);

	// Alias parity
	double mae_alias = run_scalar_double(con, "SELECT anofox_tabfm_mae(actual, predicted) FROM t");
	REQUIRE(mae == mae_alias);
}

TEST_CASE("tabfm_r2: golden value and constant-target edge cases", "[tabfm][metrics_regression]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES "
	                   "(1.0, 1.1),(2.0, 2.2),(3.0, 2.9),(4.0, 3.8),(5.0, 5.5)"
	                   ") v(actual, predicted)")->HasError());

	double r2 = run_scalar_double(con, "SELECT tabfm_r2(actual, predicted) FROM t");
	// sklearn r2_score = 0.965
	REQUIRE(std::abs(r2 - 0.965) < 1e-9);

	// Constant target, perfect prediction → 1.0 (not NaN, not Inf)
	double r2_perfect = run_scalar_double(
	    con, "SELECT tabfm_r2(actual, predicted) FROM (VALUES "
	         "(5.0,5.0),(5.0,5.0),(5.0,5.0)) v(actual,predicted)");
	REQUIRE(r2_perfect == 1.0);
	REQUIRE(std::isfinite(r2_perfect));

	// Constant target, imperfect prediction → 0.0 (not NaN, not Inf)
	double r2_imperfect = run_scalar_double(
	    con, "SELECT tabfm_r2(actual, predicted) FROM (VALUES "
	         "(5.0,4.0),(5.0,5.0),(5.0,6.0)) v(actual,predicted)");
	REQUIRE(r2_imperfect == 0.0);
	REQUIRE(std::isfinite(r2_imperfect));

	// Empty input → NULL
	REQUIRE(is_null(con, "SELECT tabfm_r2(actual, predicted) FROM t WHERE 1=0"));
}

TEST_CASE("tabfm_mape: zero-actual skip and golden value", "[tabfm][metrics_regression]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES "
	                   "(1.0, 1.1),(2.0, 2.2),(3.0, 2.9),(4.0, 3.8),(5.0, 5.5)"
	                   ") v(actual, predicted)")->HasError());

	double mape = run_scalar_double(con, "SELECT tabfm_mape(actual, predicted) FROM t");
	// sklearn mean_absolute_percentage_error = 0.076667 (dimensionless ratio)
	REQUIRE(std::abs(mape - 0.076667) < 1e-5);

	// Zero-actual rows skipped; MAPE over (1.0,1.5),(2.0,2.5) = 0.375
	double mape_skip =
	    run_scalar_double(con, "SELECT tabfm_mape(actual, predicted) FROM (VALUES "
	                           "(0.0,0.5),(1.0,1.5),(2.0,2.5)) v(actual,predicted)");
	REQUIRE(std::abs(mape_skip - 0.375) < 1e-9);

	// All-zero actuals → NULL
	REQUIRE(is_null(con, "SELECT tabfm_mape(actual, predicted) FROM (VALUES "
	                     "(0.0,1.0),(0.0,2.0)) v(actual,predicted)"));

	// Empty → NULL
	REQUIRE(is_null(con, "SELECT tabfm_mape(actual, predicted) FROM t WHERE 1=0"));
}

TEST_CASE("tabfm_medae: golden value and even-N mean-of-middles", "[tabfm][metrics_regression]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE OR REPLACE TABLE t AS SELECT * FROM (VALUES "
	                   "(1.0, 1.1),(2.0, 2.2),(3.0, 2.9),(4.0, 3.8),(5.0, 5.5)"
	                   ") v(actual, predicted)")->HasError());

	// Odd N=5: sorted residuals=[0.1,0.1,0.2,0.2,0.5] → median=0.2
	double medae = run_scalar_double(con, "SELECT tabfm_medae(actual, predicted) FROM t");
	REQUIRE(std::abs(medae - 0.2) < 1e-9);

	// Even N=4: actual=[1,2,3,4] pred=[1.1,2.2,2.9,3.8]
	// sorted residuals=[0.1,0.1,0.2,0.2] → median=(0.1+0.2)/2=0.15
	double medae_even =
	    run_scalar_double(con, "SELECT tabfm_medae(actual, predicted) FROM (VALUES "
	                           "(1.0,1.1),(2.0,2.2),(3.0,2.9),(4.0,3.8)) v(actual,predicted)");
	REQUIRE(std::abs(medae_even - 0.15) < 1e-9);

	// Alias parity
	double medae_alias =
	    run_scalar_double(con, "SELECT anofox_tabfm_medae(actual, predicted) FROM t");
	REQUIRE(medae == medae_alias);

	// Empty → NULL
	REQUIRE(is_null(con, "SELECT tabfm_medae(actual, predicted) FROM t WHERE 1=0"));
}
