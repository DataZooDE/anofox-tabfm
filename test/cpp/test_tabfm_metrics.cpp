// test/cpp/test_tabfm_metrics.cpp — Catch2 unit tests for classification metric
// aggregates via DuckDB SQL execution.
//
// Tests: tabfm_accuracy / anofox_tabfm_accuracy (CMET-01)
//        tabfm_f1 / tabfm_precision / tabfm_recall (CMET-02)
//        tabfm_log_loss (CMET-03)
//        tabfm_roc_auc with tie-handling (CMET-04)
//        tabfm_ece (CMET-06)
//
// Plan 01-02 adds CMET-02..06 test cases.

#include "catch.hpp"
#include "duckdb.hpp"

// Helper: load extension on a fresh connection
static duckdb::Connection make_con() {
	// Constructed once, DELIBERATELY NEVER DESTROYED.
	//
	// As a plain `static duckdb::DuckDB db(nullptr)` this is torn down by
	// __cxa_finalize_ranges at process exit, and on macOS/arm64 that teardown
	// faults every time:
	//
	//   ~DuckDB -> ~DatabaseInstance -> ~DBConfig -> ~BlockAllocator
	//            -> BlockAllocatorThreadLocalState::Initialize
	//   EXC_BAD_ACCESS (SIGSEGV), KERN_INVALID_ADDRESS, pointer-auth failure
	//
	// Every test passes before that runs, so the binary prints "All tests
	// passed" and *then* exits 139 -- which is exactly how it looks in CI, and
	// why it reads as a mystery rather than a test failure. It surfaced only on
	// macOS: a release build there crashed 8/8, and 0/8 with this change.
	//
	// It is also load-order sensitive, which is why it looked flaky. The same
	// commit passed on a machine where the MLX plugin was built and failed
	// where it was not -- an unrelated dylib shifting teardown order was enough
	// to hide it. Hosted macOS runners have no MLX, so CI always lost.
	//
	// The pointer stays reachable from .bss, so this is "still reachable"
	// rather than a leak: LeakSanitizer does not report it, and the process is
	// exiting anyway. Sibling suites (test_tabfm_crossval.cpp) avoid the
	// problem differently, by giving each TEST_CASE its own stack-local DuckDB
	// that is destroyed while the runtime is still fully alive.
	static auto *db = new duckdb::DuckDB(nullptr);
	duckdb::Connection con(*db);
	REQUIRE(!con.Query("LOAD anofox_tabfm")->HasError());
	return con;
}

// ============================================================
// CMET-01: tabfm_accuracy
// ============================================================

TEST_CASE("tabfm_accuracy: golden value 2/3 matches sklearn", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE preds AS SELECT * FROM (VALUES "
	                   "  ('cat', 'cat'), ('dog', 'cat'), ('cat', 'cat')"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT round(tabfm_accuracy(actual, predicted), 6) FROM preds");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == 0.666667);
}

TEST_CASE("tabfm_accuracy: NULL rows skipped", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE preds_null AS SELECT * FROM (VALUES "
	                   "  ('cat', 'cat'), (NULL::VARCHAR, 'cat'), ('dog', 'dog'), ('cat', NULL::VARCHAR)"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_accuracy(actual, predicted) FROM preds_null");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == 1.0);
}

TEST_CASE("tabfm_accuracy: all-NULL input returns NULL", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE preds_empty AS SELECT * FROM (VALUES "
	                   "  (NULL::VARCHAR, NULL::VARCHAR)"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_accuracy(actual, predicted) FROM preds_empty");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).IsNull());
}

TEST_CASE("tabfm_accuracy: alias parity with anofox_tabfm_accuracy", "[tabfm][metrics]") {
	auto con = make_con();

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

// ============================================================
// CMET-02: tabfm_f1 — required avg, golden values, NULL skip
// ============================================================

TEST_CASE("tabfm_f1: missing avg throws named exception", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE pf AS SELECT * FROM (VALUES ('a','a'),('b','a')) v(actual, predicted)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_f1(actual, predicted) FROM pf");
	REQUIRE(result->HasError());
	REQUIRE(result->GetErrorObject().Message().find("tabfm_f1") != std::string::npos);
	REQUIRE(result->GetErrorObject().Message().find("avg") != std::string::npos);
}

TEST_CASE("tabfm_f1: macro/weighted/micro on 3-class fixture match sklearn", "[tabfm][metrics]") {
	// Fixture: actual=['cat','dog','cat'], predicted=['cat','cat','cat']
	// F1 macro=0.4, weighted=0.533333, micro=0.666667
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE f1_preds AS SELECT * FROM (VALUES "
	                   "  ('cat','cat'), ('dog','cat'), ('cat','cat')"
	                   ") v(actual, predicted)")
	             ->HasError());

	auto r_macro    = con.Query("SELECT round(tabfm_f1(actual, predicted, 'macro'), 6) FROM f1_preds");
	auto r_weighted = con.Query("SELECT round(tabfm_f1(actual, predicted, 'weighted'), 6) FROM f1_preds");
	auto r_micro    = con.Query("SELECT round(tabfm_f1(actual, predicted, 'micro'), 6) FROM f1_preds");

	REQUIRE(!r_macro->HasError());
	REQUIRE(!r_weighted->HasError());
	REQUIRE(!r_micro->HasError());

	REQUIRE(r_macro->GetValue(0, 0).GetValue<double>() == Approx(0.4).epsilon(1e-6));
	REQUIRE(r_weighted->GetValue(0, 0).GetValue<double>() == Approx(0.533333).epsilon(1e-5));
	REQUIRE(r_micro->GetValue(0, 0).GetValue<double>() == Approx(0.666667).epsilon(1e-6));
}

// ============================================================
// CMET-04: tabfm_roc_auc — tie handling
// ============================================================

TEST_CASE("tabfm_roc_auc: tie handling matches sklearn rank-sum (OvR)", "[tabfm][metrics]") {
	// 6-row 3-class fixture WITH tied cat score=0.6 in rows 2 and 3:
	//   actual=['cat','cat','dog','dog','fish','fish']
	//   proba[cat] = [0.8, 0.6, 0.6, 0.2, 0.1, 0.1]  <- tie at 0.6
	// sklearn roc_auc_score(multi_class='ovr', average='macro') = 0.958333
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE auc_preds AS SELECT * FROM (VALUES "
	                   "  ('cat',  MAP {'cat': 0.8, 'dog': 0.1, 'fish': 0.1}),"
	                   "  ('cat',  MAP {'cat': 0.6, 'dog': 0.3, 'fish': 0.1}),"
	                   "  ('dog',  MAP {'cat': 0.6, 'dog': 0.3, 'fish': 0.1}),"
	                   "  ('dog',  MAP {'cat': 0.2, 'dog': 0.7, 'fish': 0.1}),"
	                   "  ('fish', MAP {'cat': 0.1, 'dog': 0.1, 'fish': 0.8}),"
	                   "  ('fish', MAP {'cat': 0.1, 'dog': 0.2, 'fish': 0.7})"
	                   ") v(actual, proba)")
	             ->HasError());

	auto result = con.Query("SELECT round(tabfm_roc_auc(actual, proba, 'ovr'), 6) FROM auc_preds");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == Approx(0.958333).epsilon(1e-6));
}

TEST_CASE("tabfm_roc_auc: OvO matches sklearn", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE auc_ovo AS SELECT * FROM (VALUES "
	                   "  ('cat',  MAP {'cat': 0.8, 'dog': 0.1, 'fish': 0.1}),"
	                   "  ('cat',  MAP {'cat': 0.6, 'dog': 0.3, 'fish': 0.1}),"
	                   "  ('dog',  MAP {'cat': 0.6, 'dog': 0.3, 'fish': 0.1}),"
	                   "  ('dog',  MAP {'cat': 0.2, 'dog': 0.7, 'fish': 0.1}),"
	                   "  ('fish', MAP {'cat': 0.1, 'dog': 0.1, 'fish': 0.8}),"
	                   "  ('fish', MAP {'cat': 0.1, 'dog': 0.2, 'fish': 0.7})"
	                   ") v(actual, proba)")
	             ->HasError());

	auto result = con.Query("SELECT round(tabfm_roc_auc(actual, proba, 'ovo'), 6) FROM auc_ovo");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == Approx(0.958333).epsilon(1e-6));
}

TEST_CASE("tabfm_roc_auc: missing avg throws named exception", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE auc_noavg AS SELECT * FROM (VALUES "
	                   "  ('cat', MAP {'cat': 0.8, 'dog': 0.2}),"
	                   "  ('dog', MAP {'cat': 0.3, 'dog': 0.7})"
	                   ") v(actual, proba)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_roc_auc(actual, proba) FROM auc_noavg");
	REQUIRE(result->HasError());
	REQUIRE(result->GetErrorObject().Message().find("tabfm_roc_auc") != std::string::npos);
	REQUIRE(result->GetErrorObject().Message().find("avg") != std::string::npos);
}

// ============================================================
// CMET-03: tabfm_log_loss — clipping, no infinity
// ============================================================

TEST_CASE("tabfm_log_loss: clipped p=0 never returns infinity", "[tabfm][metrics]") {
	auto con = make_con();

	// actual=fish but MAP has no 'fish' key -> p=0 -> clip to 1e-15 -> finite loss
	REQUIRE(!con.Query("CREATE TABLE ll_clip AS SELECT * FROM (VALUES "
	                   "  ('fish', MAP {'cat': 0.7, 'dog': 0.3})"
	                   ") v(actual, proba)")
	             ->HasError());

	auto result = con.Query("SELECT tabfm_log_loss(actual, proba) FROM ll_clip");
	REQUIRE(!result->HasError());
	double v = result->GetValue(0, 0).GetValue<double>();
	REQUIRE(std::isfinite(v));
	REQUIRE(v > 30.0); // -log(1e-15) ≈ 34.5
	REQUIRE(v < 40.0);
}

TEST_CASE("tabfm_log_loss: golden value matches sklearn on 3-row proba fixture", "[tabfm][metrics]") {
	auto con = make_con();

	REQUIRE(!con.Query("CREATE TABLE ll_preds AS SELECT * FROM (VALUES "
	                   "  ('cat', MAP {'cat': 0.7, 'dog': 0.2, 'fish': 0.1}),"
	                   "  ('dog', MAP {'cat': 0.1, 'dog': 0.8, 'fish': 0.1}),"
	                   "  ('cat', MAP {'cat': 0.5, 'dog': 0.3, 'fish': 0.2})"
	                   ") v(actual, proba)")
	             ->HasError());

	auto result = con.Query("SELECT round(tabfm_log_loss(actual, proba), 6) FROM ll_preds");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<double>() == Approx(0.424322).epsilon(1e-5));
}
