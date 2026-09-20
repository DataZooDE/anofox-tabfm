// test/cpp/test_tabfm_metrics.cpp — Catch2 unit tests for classification metric
// aggregates via DuckDB SQL execution.
//
// Tests tabfm_accuracy / anofox_tabfm_accuracy (CMET-01).
// Plan 01-02 adds test cases for F1, log-loss, ROC-AUC, ECE.

#include "catch.hpp"
#include "duckdb.hpp"

TEST_CASE("tabfm_accuracy: golden value 2/3 matches sklearn", "[tabfm][metrics]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);

	// Load extension
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// 3-class fixture: 2 correct out of 3 = 0.666667 (sklearn: accuracy_score)
	REQUIRE(!con.Query("CREATE TABLE preds AS SELECT * FROM (VALUES "
	                   "  ('cat', 'cat'), ('dog', 'cat'), ('cat', 'cat')"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT round(tabfm_accuracy(actual, predicted), 6) FROM preds");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == 0.666667);
}

TEST_CASE("tabfm_accuracy: NULL rows skipped", "[tabfm][metrics]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// Row 2 (NULL actual) and row 4 (NULL predicted) are skipped.
	// Valid: ('cat','cat') and ('dog','dog') => 2/2 = 1.0
	REQUIRE(!con.Query("CREATE TABLE preds_null AS SELECT * FROM (VALUES "
	                   "  ('cat', 'cat'), (NULL::VARCHAR, 'cat'), ('dog', 'dog'), ('cat', NULL::VARCHAR)"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_accuracy(actual, predicted) FROM preds_null");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == 1.0);
}

TEST_CASE("tabfm_accuracy: all-NULL input returns NULL", "[tabfm][metrics]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE TABLE preds_empty AS SELECT * FROM (VALUES "
	                   "  (NULL::VARCHAR, NULL::VARCHAR)"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_accuracy(actual, predicted) FROM preds_empty");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).IsNull());
}

TEST_CASE("tabfm_accuracy: alias parity with anofox_tabfm_accuracy", "[tabfm][metrics]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE TABLE preds_alias AS SELECT * FROM (VALUES "
	                   "  ('x', 'x'), ('y', 'z'), ('z', 'z')"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto r1 = con.Query("SELECT round(tabfm_accuracy(actual, predicted), 10) FROM preds_alias");
	auto r2 = con.Query("SELECT round(anofox_tabfm_accuracy(actual, predicted), 10) FROM preds_alias");
	REQUIRE(!r1->HasError());
	REQUIRE(!r2->HasError());
	REQUIRE(r1->GetValue(0, 0).GetValue<double>() == r2->GetValue(0, 0).GetValue<double>());
}
