// test/cpp/test_tabfm_crossval.cpp — Catch2 unit tests for cross-validation
// macros (tabfm_cross_validate, tabfm_fold_assign) via DuckDB SQL execution.
//
// Scaffold placeholder: one passing case so the OBJECT library source list
// compiles. Plan 01-04 adds real test cases.

#include "catch.hpp"
#include "duckdb.hpp"

TEST_CASE("tabfm_cross_validate: scaffold placeholder", "[tabfm][crossval][scaffold]") {
	// Minimal sanity — DuckDB opens and computes.
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	auto result = con.Query("SELECT 7 - 3");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 4);
}
