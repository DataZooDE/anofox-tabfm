// test/cpp/test_tabfm_scoring.cpp — Catch2 unit tests for tabfm_scoring
// (proper scoring rules: CRPS, PSR-01)
//
// RED phase stub — test cases created before implementation to follow TDD.
// Spec ref: PSR-01, PSR-04, plan 03-01
// Research ref: 03-RESEARCH.md §Exact CRPS Math, §Golden Testing Strategy

#include "catch.hpp"
#include "tabfm_scoring.hpp"
#include "duckdb.hpp"

#include <cmath>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

// ---------------------------------------------------------------------------
// ComputeCRPS — direct unit tests for the math helper
//
// Reference values computed by tools/parity/src/parity/crps_reference.py
// crps_bar() function (Python softmax + per-bin Case A/B/C integral).
// ---------------------------------------------------------------------------

TEST_CASE("tabfm_scoring: CRPS synthetic K=4 uniform bins, y inside range",
          "[tabfm_scoring]") {
	// K=4 equal logits → uniform probs 0.25 each
	// borders=[0,1,2,3,4], y=1.5 (inside bin 1)
	//
	// Reference (crps_reference.py): 0.395833333333333
	// Hand-verified:
	//   bin 0 [0,1]: Case B → 1*(0^2 + 0*0.25 + 0.25^2/3) = 0.020833...
	//   bin 1 [1,2]: Case C (y=1.5, d=0.5, e=0.5, c=0.25)
	//     L = 0.25^2*0.5 + 0.25*0.25*0.25 + 0.25^2*0.125/3 = 0.049479...
	//     F(y)=0.375, a2=-0.625; R = 0.625^2*0.5 - 0.625*0.25*0.25 + 0.25^2*0.125/3 = 0.158854...
	//     contribution = 0.208333...
	//   bin 2 [2,3]: Case A (y<b_lo) → 1*((-0.5)^2 + (-0.5)*0.25 + 0.25^2/3) = 0.145833...
	//   bin 3 [3,4]: Case A → 1*((-0.25)^2 + (-0.25)*0.25 + 0.25^2/3) = 0.020833...
	//   total = 0.395833...
	std::vector<double> probs   = {0.25, 0.25, 0.25, 0.25};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeCRPS(1.5, probs, borders);
	// crps_reference.py: 0.395833333333333
	REQUIRE(std::abs(result - 0.395833333333333) < 1e-9);
}

TEST_CASE("tabfm_scoring: CRPS synthetic K=4 uniform bins, y outside range (left tail)",
          "[tabfm_scoring]") {
	// K=4, probs=0.25 each, borders=[0,1,2,3,4], y=-1.0 (all bins Case A)
	//
	// Reference (crps_reference.py): 1.33333333333333
	// All 4 bins have y < b_lo → Case A: sum wi*((C_i-1)^2 + (C_i-1)*pi + pi^2/3)
	// bin 0: wi=1, C=0:  (-1)^2 + (-1)*0.25 + 0.25^2/3 = 1 - 0.25 + 0.0208 = 0.770833
	// bin 1: wi=1, C=0.25: (-0.75)^2 + (-0.75)*0.25 + 0.0208 = 0.5625 - 0.1875 + 0.0208 = 0.395833
	// bin 2: wi=1, C=0.5:  (-0.5)^2 + (-0.5)*0.25 + 0.0208 = 0.25 - 0.125 + 0.0208 = 0.145833
	// bin 3: wi=1, C=0.75: (-0.25)^2 + (-0.25)*0.25 + 0.0208 = 0.0625 - 0.0625 + 0.0208 = 0.020833
	// total = 0.770833 + 0.395833 + 0.145833 + 0.020833 = 1.333333
	std::vector<double> probs   = {0.25, 0.25, 0.25, 0.25};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeCRPS(-1.0, probs, borders);
	// crps_reference.py: 1.33333333333333
	REQUIRE(std::abs(result - 1.33333333333333) < 1e-9);
	REQUIRE(std::isfinite(result));
	REQUIRE(result >= 0.0);
}

TEST_CASE("tabfm_scoring: CRPS returns 0 on empty probs", "[tabfm_scoring]") {
	std::vector<double> probs   = {};
	std::vector<double> borders = {0.0};
	// K==0 → guard returns 0.0
	REQUIRE(ComputeCRPS(1.0, probs, borders) == 0.0);
}

TEST_CASE("tabfm_scoring: CRPS returns 0 on size mismatch", "[tabfm_scoring]") {
	std::vector<double> probs   = {0.5, 0.5};
	std::vector<double> borders = {0.0, 1.0}; // should be 3 borders for 2 probs
	REQUIRE(ComputeCRPS(0.5, probs, borders) == 0.0);
}

// ---------------------------------------------------------------------------
// SQL-level tests: tabfm_crps aggregate via DuckDB connection
// ---------------------------------------------------------------------------

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

TEST_CASE("tabfm_scoring: tabfm_crps empty group returns NULL", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// Build a tiny synthetic distribution table
	REQUIRE(!con.Query(
	            "CREATE TABLE t_empty AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist "
	            "WHERE 1=0")
	             ->HasError());

	REQUIRE(is_null(con, "SELECT tabfm_crps(actual, dist) FROM t_empty"));
}

TEST_CASE("tabfm_scoring: tabfm_crps PSR-04 bind-gate rejects plain DOUBLE", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE TABLE t AS SELECT 1.5::DOUBLE AS actual, 1.2::DOUBLE AS yhat")
	             ->HasError());

	// PSR-04: tabfm_crps(actual, plain_double) must fail at bind
	auto result = con.Query("SELECT tabfm_crps(actual, yhat) FROM t");
	REQUIRE(result->HasError());
	// Error message must name the remedy (SQL-API §5)
	std::string err = result->GetError();
	REQUIRE(err.find("output_mode") != std::string::npos);
}

TEST_CASE("tabfm_scoring: tabfm_crps synthetic K=4 via SQL", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// Create a single row with K=4 uniform distribution, y=1.5
	// probs after softmax([0,0,0,0]) = [0.25, 0.25, 0.25, 0.25]
	// Expected CRPS = 0.395833333333333 (crps_reference.py)
	REQUIRE(!con.Query(
	            "CREATE TABLE t_k4 AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist")
	             ->HasError());

	double crps = run_scalar_double(con, "SELECT tabfm_crps(actual, dist) FROM t_k4");
	// crps_reference.py: 0.395833333333333
	REQUIRE(std::abs(crps - 0.395833333333333) < 1e-9);

	// Alias parity (PSR-01 requirement)
	double crps_alias =
	    run_scalar_double(con, "SELECT anofox_tabfm_crps(actual, dist) FROM t_k4");
	REQUIRE(std::abs(crps - crps_alias) < 1e-15);
}

TEST_CASE("tabfm_scoring: tabfm_crps NULL rows are skipped", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// Three rows: one valid (y=1.5, K=4 uniform), one NULL actual, one NULL dist
	REQUIRE(!con.Query(
	            "CREATE TABLE t_nulls AS SELECT * FROM (VALUES "
	            "(1.5::DOUBLE, {'logits': [0.0,0.0,0.0,0.0]::DOUBLE[], 'borders': [0.0,1.0,2.0,3.0,4.0]::DOUBLE[]}), "
	            "(NULL::DOUBLE, {'logits': [0.0,0.0,0.0,0.0]::DOUBLE[], 'borders': [0.0,1.0,2.0,3.0,4.0]::DOUBLE[]}), "
	            "(1.5::DOUBLE, NULL::{logits DOUBLE[], borders DOUBLE[]})"
	            ") v(actual, dist)")
	             ->HasError());

	// Only the first row contributes; result equals single-row CRPS
	double crps = run_scalar_double(con, "SELECT tabfm_crps(actual, dist) FROM t_nulls");
	REQUIRE(std::abs(crps - 0.395833333333333) < 1e-9);
}
