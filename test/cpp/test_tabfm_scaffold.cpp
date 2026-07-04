// Scaffold sanity test: proves the Catch2 in-unittest registration mechanism
// works end to end (CMake OBJECT lib -> deferred target_sources(unittest)).
// Real C++ tests (safetensors, manifest, engine, preprocess, ensemble) are
// added by their workstreams.

#include "catch.hpp"

#include "duckdb.hpp"

TEST_CASE("anofox_tabfm scaffold: DuckDB opens and answers", "[tabfm]") {
	duckdb::DuckDB db(nullptr);
	duckdb::Connection con(db);
	auto result = con.Query("SELECT 21 * 2");
	REQUIRE(!result->HasError());
	REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
}
