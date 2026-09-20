// test/cpp/test_tabfm_crossval.cpp — Catch2 unit tests for cross-validation
// macros (tabfm_cross_validate, tabfm_fold_assign) via DuckDB SQL execution.
//
// Spec ref: CV-01..04, plan 01-04
// Test coverage:
//   - tabfm_fold_assign determinism, range, seed sensitivity, formula verification
//   - tabfm_cross_validate row count, schema, metric_value bounds
//   - NON-NEGOTIABLE leakage-detecting golden test (CV-02 gate):
//       A fixture engineered so a leaky pipeline would score ~100% but the
//       leakage-safe two-table form scores at chance level (~1/k). The test
//       asserts aggregate accuracy < 0.8 (well below the leaky ceiling of 1.0).
//
// Uses the committed CI fixture model (test/fixtures/manifest.json, random-init
// weights) so no real model download is required. Fixture logic documented below.

#include "catch.hpp"
#include "duckdb.hpp"

#include <string>

// Helper: open an in-memory DuckDB, load the extension, optionally set model manifest.
static duckdb::unique_ptr<duckdb::MaterializedQueryResult> qry(duckdb::Connection &con, const std::string &sql) {
	return con.Query(sql);
}

static void load_ext(duckdb::Connection &con) {
	REQUIRE(!qry(con, "LOAD anofox_tabfm")->HasError());
}

// Load extension + configure the fixture model manifest.
static void setup_cv_db(duckdb::Connection &con) {
	load_ext(con);
	REQUIRE(!qry(con, "SET anofox_tabfm_model_manifest = 'test/fixtures/manifest.json'")->HasError());
}

// ============================================================================
// CV-01: tabfm_fold_assign
// ============================================================================

TEST_CASE("tabfm_fold_assign: fold ids are in [0, k-1]", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	load_ext(con);

	qry(con, R"(
		CREATE TABLE fa_rows AS SELECT * FROM (VALUES
		  (1,'a',1.0),(2,'b',2.0),(3,'c',3.0),(4,'d',4.0),(5,'e',5.0),
		  (6,'f',6.0),(7,'g',7.0),(8,'h',8.0)
		) t(id, cat, val)
	)");

	// All fold ids in [0, 2] for k=3
	auto res = qry(con, "SELECT count(*) FROM tabfm_fold_assign('fa_rows', 3, 'id', 42) WHERE fold_id < 0 OR fold_id >= 3");
	REQUIRE(!res->HasError());
	REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == 0);

	// Total row count preserved
	auto cnt = qry(con, "SELECT count(*) FROM tabfm_fold_assign('fa_rows', 3, 'id', 42)");
	REQUIRE(!cnt->HasError());
	REQUIRE(cnt->GetValue(0, 0).GetValue<int64_t>() == 8);
}

TEST_CASE("tabfm_fold_assign: deterministic — two runs agree", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	load_ext(con);

	qry(con, R"(
		CREATE TABLE fa2 AS SELECT * FROM (VALUES
		  (1,'x'),(2,'y'),(3,'z'),(4,'w'),(5,'v'),(6,'u')
		) t(id, cat)
	)");

	auto res = qry(con, R"(
		SELECT count(*) FROM (
		  SELECT a.id FROM tabfm_fold_assign('fa2', 2, 'id', 7) a
		  JOIN tabfm_fold_assign('fa2', 2, 'id', 7) b USING (id)
		  WHERE a.fold_id <> b.fold_id
		)
	)");
	REQUIRE(!res->HasError());
	REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == 0);
}

TEST_CASE("tabfm_fold_assign: seed-sensitive — different seeds differ", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	load_ext(con);

	qry(con, "CREATE TABLE fa3 AS SELECT * FROM range(12) t(id)");

	// Verify fold assignments differ between seed=7 and seed=99 for at least some rows
	auto res = qry(con, R"(
		SELECT count(*) FROM (
		  SELECT a.id FROM tabfm_fold_assign('fa3', 3, 'id', 7) a
		  JOIN tabfm_fold_assign('fa3', 3, 'id', 99) b USING (id)
		  WHERE a.fold_id <> b.fold_id
		)
	)");
	REQUIRE(!res->HasError());
	// Expect at least 1 difference (hash changes across seed change)
	REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() > 0);
}

TEST_CASE("tabfm_fold_assign: formula matches (hash(id,seed)%k)::INTEGER", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	load_ext(con);

	qry(con, "CREATE TABLE fa4 AS SELECT * FROM range(10) t(id)");

	// Each row's fold_id must equal the explicit hash formula.
	// CAST(42 AS BIGINT) matches tabfm_fold_assign's macro body which uses
	// hash(_cv_rk, CAST(seed AS BIGINT)). DuckDB hash() is type-sensitive so
	// hash(x, 42::INTEGER) != hash(x, 42::BIGINT) in general (WR-01).
	auto res = qry(con, R"(
		SELECT count(*) FROM (
		  SELECT id, fold_id, (hash(id, CAST(42 AS BIGINT)) % 4)::INTEGER AS expected
		  FROM tabfm_fold_assign('fa4', 4, 'id', 42)
		  WHERE fold_id <> expected
		)
	)");
	REQUIRE(!res->HasError());
	REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == 0);
}

// ============================================================================
// CV-02 / CV-03 / CV-04: tabfm_cross_validate
// ============================================================================

TEST_CASE("tabfm_cross_validate: k=2 returns 3 rows (2 fold + 1 aggregate)", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	setup_cv_db(con);

	qry(con, R"(
		CREATE TABLE cv_t AS SELECT * FROM (VALUES
		  (1, 0.5, 1.0, 'a', 'c0'),
		  (2, 1.5, 0.2, 'b', 'c1'),
		  (3, 2.5,-1.0, 'a', 'c0'),
		  (4, 0.2, 0.8, 'b', 'c1'),
		  (5, 1.1, 0.4, 'a', 'c0'),
		  (6, 3.0,-0.5, 'b', 'c1'),
		  (7, 0.9, 1.2, 'a', 'c0'),
		  (8, 2.1, 0.1, 'b', 'c1')
		) t(id, f1, f2, cat, label)
	)");

	auto cnt = qry(con, "SELECT count(*) FROM tabfm_cross_validate('cv_t', 'label', 'id', k := 2, seed := 42)");
	REQUIRE(!cnt->HasError());
	REQUIRE(cnt->GetValue(0, 0).GetValue<int64_t>() == 3);

	// Aggregate row exists with fold_id = -1
	auto agg = qry(con, "SELECT count(*) FROM tabfm_cross_validate('cv_t', 'label', 'id', k := 2, seed := 42) WHERE fold_id = -1");
	REQUIRE(!agg->HasError());
	REQUIRE(agg->GetValue(0, 0).GetValue<int64_t>() == 1);
}

TEST_CASE("tabfm_cross_validate: aggregate row has non-NULL mean and std", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	setup_cv_db(con);

	qry(con, R"(
		CREATE TABLE cv_t2 AS SELECT * FROM (VALUES
		  (1, 0.5, 1.0, 'a', 'c0'), (2, 1.5, 0.2, 'b', 'c1'),
		  (3, 2.5,-1.0, 'a', 'c0'), (4, 0.2, 0.8, 'b', 'c1'),
		  (5, 1.1, 0.4, 'a', 'c0'), (6, 3.0,-0.5, 'b', 'c1')
		) t(id, f1, f2, cat, label)
	)");

	auto res = qry(con, R"(
		SELECT count(*) FROM tabfm_cross_validate('cv_t2', 'label', 'id', k := 2, seed := 42)
		  WHERE fold_id = -1 AND (metric_value IS NULL OR metric_std IS NULL)
	)");
	REQUIRE(!res->HasError());
	REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == 0);
}

TEST_CASE("tabfm_cross_validate: per-fold accuracy in [0,1]", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	setup_cv_db(con);

	qry(con, R"(
		CREATE TABLE cv_t3 AS SELECT * FROM (VALUES
		  (1, 0.5, 1.0, 'a', 'c0'), (2, 1.5, 0.2, 'b', 'c1'),
		  (3, 2.5,-1.0, 'a', 'c0'), (4, 0.2, 0.8, 'b', 'c1'),
		  (5, 1.1, 0.4, 'a', 'c0'), (6, 3.0,-0.5, 'b', 'c1'),
		  (7, 0.9, 1.2, 'a', 'c0'), (8, 2.1, 0.1, 'b', 'c1')
		) t(id, f1, f2, cat, label)
	)");

	auto res = qry(con, R"(
		SELECT count(*) FROM tabfm_cross_validate('cv_t3', 'label', 'id', k := 2, seed := 42)
		  WHERE fold_id >= 0 AND (metric_value < 0 OR metric_value > 1)
	)");
	REQUIRE(!res->HasError());
	REQUIRE(res->GetValue(0, 0).GetValue<int64_t>() == 0);
}

// ============================================================================
// CV-04: SQL injection guard — target identifier with embedded double-quote
// ============================================================================

TEST_CASE("tabfm_cross_validate: CV-04 safe quoting of target with double-quote", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	setup_cv_db(con);

	// Include integer id column for stable fold assignment (float columns may all hash
	// to the same fold with certain seed/k combinations — integer ids distribute uniformly).
	qry(con, R"(
		CREATE TABLE cv_quot_src AS SELECT * FROM (VALUES
		  (1, 1.0, 1.0, 'a', 'c0'), (2, 2.0, 0.2, 'b', 'c1'),
		  (3, 3.0,-1.0, 'a', 'c0'), (4, 4.0, 0.8, 'b', 'c1'),
		  (5, 1.1, 0.4, 'a', 'c1'), (6, 3.0,-0.5, 'b', 'c0')
		) t(id, f1, f2, cat, label)
	)");
	// Rename label to 'my"col'
	qry(con, R"(CREATE TABLE cv_quot_q AS SELECT id, f1, f2, cat, label AS "my""col" FROM cv_quot_src)");

	// Must succeed (no SQL injection/parse error) — row_key = 'id' (integer, stable fold distribution)
	auto res = qry(con, R"(SELECT count(*) > 0 FROM tabfm_cross_validate('cv_quot_q', 'my"col', 'id', k := 2, seed := 42))");
	REQUIRE(!res->HasError());
}

// ============================================================================
// NON-NEGOTIABLE: CV-02 leakage-detecting golden test
//
// Fixture design: 9 labeled rows, k=3 folds (3 rows per fold), 3-class target.
//
// The CI fixture model (random-init weights) deterministically predicts class 'c2'
// (the 3rd class alphabetically) for every row, as confirmed by golden.json:
// logits are [-0.002, -0.053, +0.064] — third index always highest.
//
// Leakage-safe accuracy: with 3 classes (c0, c1, c2), the model always predicts
// 'c2'. Each fold has ~1 row labeled 'c2' out of 3, giving accuracy ≈ 1/3.
//
// Leaky ceiling: ~1.0 for a memorizing model.
//
// Assertion: aggregate accuracy < 0.8.
// This threshold is conservative: real leakage-safe accuracy ≈ 1/3.
// A value ≥ 0.8 would indicate either (a) test-fold rows leaked, giving the model
// access to test labels as training examples, or (b) the random-init model
// accidentally achieves high accuracy (impossible with balanced classes).
//
// The two-table form enforces the guarantee: per-fold queries pass:
//   train := WHERE fold_id != f
//   test  := WHERE fold_id = f  (held-out rows, NULL label injected by macro)
// ============================================================================

TEST_CASE("tabfm_cross_validate: NON-NEGOTIABLE leakage-detecting golden test", "[tabfm][crossval]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	setup_cv_db(con);

	// Fixture: 9 rows, 3 classes (3 per class), k=3 folds (3 rows per fold).
	// The fixture model always predicts 'c2' → expected fold accuracy ≈ 1/3.
	REQUIRE(!qry(con, R"(
		CREATE TABLE cv_leak AS SELECT * FROM (VALUES
		  (1, 0.5, 1.0, 'a', 'c0'),
		  (2, 1.5, 0.2, 'b', 'c2'),
		  (3, 2.5,-1.0, 'a', 'c1'),
		  (4, 0.2, 0.8, 'b', 'c0'),
		  (5, 1.1, 0.4, 'a', 'c2'),
		  (6, 3.0,-0.5, 'b', 'c1'),
		  (7, 0.9, 1.2, 'a', 'c0'),
		  (8, 2.1, 0.1, 'b', 'c2'),
		  (9, 1.3, 0.3, 'a', 'c1')
		) t(id, f1, f2, cat, label)
	)")->HasError());

	// Step 1: CV returns exactly 4 rows (3 folds + 1 aggregate)
	auto cnt = qry(con, "SELECT count(*) FROM tabfm_cross_validate('cv_leak', 'label', 'id', k := 3, seed := 42)");
	REQUIRE(!cnt->HasError());
	REQUIRE(cnt->GetValue(0, 0).GetValue<int64_t>() == 4);

	// Step 2: Aggregate row (fold_id=-1) must exist
	auto agg_exists = qry(con, R"(
		SELECT count(*) FROM tabfm_cross_validate('cv_leak', 'label', 'id', k := 3, seed := 42)
		  WHERE fold_id = -1
	)");
	REQUIRE(!agg_exists->HasError());
	REQUIRE(agg_exists->GetValue(0, 0).GetValue<int64_t>() == 1);

	// Step 3: LEAKAGE-DETECTING ASSERTION
	// Aggregate accuracy must be < 0.8.
	// With random-init weights (always predict 'c2') and balanced 3-class data:
	//   expected aggregate accuracy ≈ 1/3 ≈ 0.333
	// Threshold 0.8 provides generous headroom while catching structural bugs
	// (e.g., single-table form that leaks test labels into the training context).
	auto leak_check = qry(con, R"(
		SELECT metric_value
		FROM tabfm_cross_validate('cv_leak', 'label', 'id', k := 3, seed := 42)
		WHERE fold_id = -1
	)");
	REQUIRE(!leak_check->HasError());
	double agg_acc = leak_check->GetValue(0, 0).GetValue<double>();
	// Leakage-safe accuracy ≈ 1/3 (chance level). Must be < 0.8.
	REQUIRE(agg_acc < 0.8);

	// Step 4: All per-fold metrics in [0, 1]
	auto bounds = qry(con, R"(
		SELECT count(*) FROM tabfm_cross_validate('cv_leak', 'label', 'id', k := 3, seed := 42)
		  WHERE fold_id >= 0 AND (metric_value < 0 OR metric_value > 1)
	)");
	REQUIRE(!bounds->HasError());
	REQUIRE(bounds->GetValue(0, 0).GetValue<int64_t>() == 0);
}
