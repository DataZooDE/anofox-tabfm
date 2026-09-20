// test/cpp/test_tabfm_metrics_regression.cpp — Catch2 unit tests for regression
// metric aggregates (RMSEState, MAEState, R2State, etc.) via DuckDB SQL execution.
//
// Scaffold placeholder: one passing case so the OBJECT library source list
// compiles. Plan 01-03 adds real test cases.

#include "catch.hpp"
#include "duckdb.hpp"

TEST_CASE("tabfm_rmse: scaffold placeholder", "[tabfm][metrics_regression][scaffold]") {
	// Minimal sanity — DuckDB opens and computes.
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	auto result = con.Query("SELECT 2 * 3");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 6);
}
