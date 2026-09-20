// test/cpp/test_tabfm_metrics.cpp — Catch2 unit tests for classification metric
// aggregates (AccuracyState, F1State, etc.) via DuckDB SQL execution.
//
// Scaffold placeholder: one passing case so the OBJECT library source list
// compiles. Plan 01-01 (Task 2) and plan 01-02 add real test cases.

#include "catch.hpp"
#include "duckdb.hpp"

TEST_CASE("tabfm_accuracy: scaffold placeholder", "[tabfm][metrics][scaffold]") {
	// Minimal sanity — DuckDB opens and computes.
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	auto result = con.Query("SELECT 1 + 1");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 2);
}
