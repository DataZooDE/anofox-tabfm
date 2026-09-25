// test/cpp/test_tabfm_scoring.cpp — Catch2 unit tests for tabfm_scoring
// (proper scoring rules: CRPS PSR-01, log-score PSR-02, interval-score PSR-03)
//
// RED phase stubs — test cases created before implementation to follow TDD.
// Spec ref: PSR-01, PSR-02, PSR-03, PSR-04, plan 03-01/03-02
// Research ref: 03-RESEARCH.md §Log-Score Math, §Interval Score Math, §Golden Testing Strategy

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

// ---------------------------------------------------------------------------
// ComputeLogScore — direct unit tests for the math helper (PSR-02)
//
// Reference values computed by log_score_bar() added to
// tools/parity/src/parity/crps_reference.py
// ---------------------------------------------------------------------------

TEST_CASE("tabfm_scoring: ComputeLogScore synthetic K=4, y inside bin 1", "[tabfm_scoring]") {
	// logits=[0,0,0,0] → probs=[0.25,0.25,0.25,0.25]
	// borders=[0,1,2,3,4], y=1.5 → bin 1 [1,2], w_1=1, p_1=0.25
	// log_score = log(w_1) - log(p_1) = log(1) - log(0.25) = ln(4) = 1.38629436...
	// Reference (log_score_bar): 1.38629436111989
	std::vector<double> probs   = {0.25, 0.25, 0.25, 0.25};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeLogScore(1.5, probs, borders);
	REQUIRE(std::abs(result - 1.38629436111989) < 1e-9);
	REQUIRE(std::isfinite(result));
}

TEST_CASE("tabfm_scoring: ComputeLogScore out-of-support y → max penalty", "[tabfm_scoring]") {
	// y=-1.0 is outside [0,4] → returns -log(1e-10) = 23.0258509...
	// T-03-04: out-of-support guard (no log(0), no divide-by-zero)
	std::vector<double> probs   = {0.25, 0.25, 0.25, 0.25};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeLogScore(-1.0, probs, borders);
	// -log(1e-10) = ln(1e10) = 10*ln(10) = 23.0258509...
	const double max_penalty = -std::log(1e-10);
	REQUIRE(std::abs(result - max_penalty) < 1e-9);
	REQUIRE(std::isfinite(result));
}

TEST_CASE("tabfm_scoring: ComputeLogScore zero-width bin → max penalty", "[tabfm_scoring]") {
	// Single zero-width bin: borders=[1.0, 1.0], K=1, y=1.0 falls in it.
	// w_0=0 < 1e-300 → returns max penalty (T-03-04).
	// Note: borders are [1.0, 1.0] so borders[0]=1.0 and y=1.0 is NOT out-of-support
	// (y == borders[0] == borders[K] is in-support, degenerate bin guard fires).
	std::vector<double> probs   = {1.0};
	std::vector<double> borders = {1.0, 1.0}; // zero-width only bin
	double result = ComputeLogScore(1.0, probs, borders);
	const double max_penalty = -std::log(1e-10);
	REQUIRE(std::abs(result - max_penalty) < 1e-9);
}

// ---------------------------------------------------------------------------
// ComputeIntervalScore — direct unit tests for the math helper (PSR-03)
//
// Reference values from interval_score_bar() in crps_reference.py
// ---------------------------------------------------------------------------

TEST_CASE("tabfm_scoring: ComputeIntervalScore synthetic K=4, y=1.5, coverage=0.9",
          "[tabfm_scoring]") {
	// logits=[0,0,0,0] → probs=[0.25,0.25,0.25,0.25]
	// borders=[0,1,2,3,4], y=1.5, coverage=0.9, alpha=0.1
	// q_lo = 0.05 → bin 0 [0,1]: p_0=0.25, q_lo=0.05 < 0.25
	//   l = 0 + (0.05/0.25)*1 = 0.2
	// q_hi = 0.95 → cum after bins 0,1,2=0.75, bin 3 [3,4]: p_3=0.25
	//   u = 3 + ((0.95-0.75)/0.25)*1 = 3 + 0.8 = 3.8
	// y=1.5 in [0.2, 3.8] → no penalty
	// IS = (3.8 - 0.2) + 0 = 3.6
	// Reference (interval_score_bar): 3.6
	std::vector<double> probs   = {0.25, 0.25, 0.25, 0.25};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeIntervalScore(1.5, probs, borders, 0.9);
	REQUIRE(std::abs(result - 3.6) < 1e-9);
	REQUIRE(std::isfinite(result));
}

TEST_CASE("tabfm_scoring: ComputeIntervalScore synthetic K=4, y=1.5, coverage=0.5",
          "[tabfm_scoring]") {
	// probs=[0.25,0.25,0.25,0.25], borders=[0,1,2,3,4], y=1.5, coverage=0.5, alpha=0.5
	// q_lo = 0.25 → boundary between bin 0 and bin 1: l = 1.0
	// q_hi = 0.75 → boundary between bin 2 and bin 3: u = 3.0
	// y=1.5 in [1.0, 3.0] → no penalty
	// IS = (3.0 - 1.0) + 0 = 2.0
	// Reference (interval_score_bar): 2.0
	std::vector<double> probs   = {0.25, 0.25, 0.25, 0.25};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeIntervalScore(1.5, probs, borders, 0.5);
	REQUIRE(std::abs(result - 2.0) < 1e-9);
	REQUIRE(std::isfinite(result));
}

TEST_CASE("tabfm_scoring: ComputeIntervalScore penalty when y below lower quantile",
          "[tabfm_scoring]") {
	// Concentrated distribution: probs=[1,0,0,0] → p_0=1, all others=0
	// borders=[0,1,2,3,4], y=3.0, coverage=0.9, alpha=0.1
	// q_lo=0.05 → l = 0 + (0.05/1)*1 = 0.05
	// q_hi=0.95 → q_hi > p_0=1.0? No: cum_after_0=1.0 >= 0.95 so u in bin 0
	//   u = 0 + (0.95/1)*1 = 0.95
	// y=3.0 > u=0.95 → penalty_hi = (3.0 - 0.95) = 2.05
	// IS = (0.95 - 0.05) + (2/0.1)*2.05 = 0.9 + 20*2.05 = 0.9 + 41.0 = 41.9
	std::vector<double> probs   = {1.0, 0.0, 0.0, 0.0};
	std::vector<double> borders = {0.0, 1.0, 2.0, 3.0, 4.0};
	double result = ComputeIntervalScore(3.0, probs, borders, 0.9);
	REQUIRE(std::abs(result - 41.9) < 1e-9);
	REQUIRE(result > 0.0);
	REQUIRE(std::isfinite(result));
}

// ---------------------------------------------------------------------------
// SQL-level: tabfm_log_score aggregate (PSR-02) via DuckDB connection
// ---------------------------------------------------------------------------

TEST_CASE("tabfm_scoring: tabfm_log_score empty group returns NULL", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query(
	            "CREATE TABLE tls_empty AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist "
	            "WHERE 1=0")
	             ->HasError());

	REQUIRE(is_null(con, "SELECT tabfm_log_score(actual, dist) FROM tls_empty"));
}

TEST_CASE("tabfm_scoring: tabfm_log_score PSR-04 bind-gate rejects plain DOUBLE",
          "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE TABLE tls_pt AS SELECT 1.5::DOUBLE AS actual, 1.2::DOUBLE AS yhat")
	             ->HasError());

	// PSR-04 extended to log-score: plain DOUBLE second arg must fail at bind
	auto result = con.Query("SELECT tabfm_log_score(actual, yhat) FROM tls_pt");
	REQUIRE(result->HasError());
	std::string err = result->GetError();
	REQUIRE(err.find("output_mode") != std::string::npos);
}

TEST_CASE("tabfm_scoring: tabfm_log_score synthetic K=4 via SQL", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// logits=[0,0,0,0] → probs=[0.25,0.25,0.25,0.25], y=1.5 → bin 1
	// mean log_score (single row) = ln(4) ≈ 1.38629436
	REQUIRE(!con.Query(
	            "CREATE TABLE tls_k4 AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist")
	             ->HasError());

	double ls = run_scalar_double(con, "SELECT tabfm_log_score(actual, dist) FROM tls_k4");
	// Reference: ln(4) = 1.38629436111989
	REQUIRE(std::abs(ls - 1.38629436111989) < 1e-9);

	// Alias parity
	double ls_alias = run_scalar_double(
	    con, "SELECT anofox_tabfm_log_score(actual, dist) FROM tls_k4");
	REQUIRE(std::abs(ls - ls_alias) < 1e-15);
}

// ---------------------------------------------------------------------------
// SQL-level: tabfm_interval_score aggregate (PSR-03) via DuckDB connection
// ---------------------------------------------------------------------------

TEST_CASE("tabfm_scoring: tabfm_interval_score empty group returns NULL", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query(
	            "CREATE TABLE tis_empty AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist "
	            "WHERE 1=0")
	             ->HasError());

	REQUIRE(is_null(con, "SELECT tabfm_interval_score(actual, dist) FROM tis_empty"));
}

TEST_CASE("tabfm_scoring: tabfm_interval_score PSR-04 bind-gate rejects plain DOUBLE",
          "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query("CREATE TABLE tis_pt AS SELECT 1.5::DOUBLE AS actual, 1.2::DOUBLE AS yhat")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_interval_score(actual, yhat) FROM tis_pt");
	REQUIRE(result->HasError());
	std::string err = result->GetError();
	REQUIRE(err.find("output_mode") != std::string::npos);
}

TEST_CASE("tabfm_scoring: tabfm_interval_score coverage validation at bind", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query(
	            "CREATE TABLE tis_v AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist")
	             ->HasError());

	// coverage=0.0 must fail at bind with named coverage range error
	auto r0 = con.Query("SELECT tabfm_interval_score(actual, dist, 0.0) FROM tis_v");
	REQUIRE(r0->HasError());
	REQUIRE(r0->GetError().find("coverage") != std::string::npos);

	// coverage=1.0 must fail at bind too
	auto r1 = con.Query("SELECT tabfm_interval_score(actual, dist, 1.0) FROM tis_v");
	REQUIRE(r1->HasError());
	REQUIRE(r1->GetError().find("coverage") != std::string::npos);

	// coverage=0.9 (default-like explicit) must succeed
	auto rok = con.Query("SELECT tabfm_interval_score(actual, dist, 0.9) FROM tis_v");
	REQUIRE(!rok->HasError());
}

TEST_CASE("tabfm_scoring: tabfm_interval_score synthetic K=4 via SQL (default coverage=0.9)",
          "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	REQUIRE(!con.Query(
	            "CREATE TABLE tis_k4 AS SELECT "
	            "1.5::DOUBLE AS actual, "
	            "{'logits': [0.0, 0.0, 0.0, 0.0]::DOUBLE[], "
	            " 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist")
	             ->HasError());

	// Default coverage=0.9: expected IS = 3.6 (reference: interval_score_bar)
	double is9 = run_scalar_double(
	    con, "SELECT tabfm_interval_score(actual, dist) FROM tis_k4");
	REQUIRE(std::abs(is9 - 3.6) < 1e-9);

	// Explicit coverage=0.5: expected IS = 2.0
	double is5 = run_scalar_double(
	    con, "SELECT tabfm_interval_score(actual, dist, 0.5) FROM tis_k4");
	REQUIRE(std::abs(is5 - 2.0) < 1e-9);

	// Alias parity
	double is9_alias = run_scalar_double(
	    con, "SELECT anofox_tabfm_interval_score(actual, dist) FROM tis_k4");
	REQUIRE(std::abs(is9 - is9_alias) < 1e-15);
}

TEST_CASE("tabfm_scoring: tabfm_crps NULL rows are skipped", "[tabfm_scoring]") {
	DuckDB     db(nullptr);
	Connection con(db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());

	// Three rows: one valid (y=1.5, K=4 uniform), one NULL actual, one NULL dist
	REQUIRE(!con.Query(
	            "CREATE TABLE t_nulls AS "
	            "SELECT 1.5::DOUBLE AS actual, "
	            "{'logits': [0.0,0.0,0.0,0.0]::DOUBLE[], 'borders': [0.0,1.0,2.0,3.0,4.0]::DOUBLE[]} AS dist "
	            "UNION ALL "
	            "SELECT NULL::DOUBLE AS actual, "
	            "{'logits': [0.0,0.0,0.0,0.0]::DOUBLE[], 'borders': [0.0,1.0,2.0,3.0,4.0]::DOUBLE[]} AS dist "
	            "UNION ALL "
	            "SELECT 1.5::DOUBLE AS actual, "
	            "NULL::STRUCT(logits DOUBLE[], borders DOUBLE[]) AS dist")
	             ->HasError());

	// Only the first row contributes; result equals single-row CRPS
	double crps = run_scalar_double(con, "SELECT tabfm_crps(actual, dist) FROM t_nulls");
	REQUIRE(std::abs(crps - 0.395833333333333) < 1e-9);
}
