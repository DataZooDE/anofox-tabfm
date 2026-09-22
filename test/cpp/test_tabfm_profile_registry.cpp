//===----------------------------------------------------------------------===//
// test_tabfm_profile_registry.cpp — Catch2 registry unit tests (MGEN-01, MGEN-02)
//
// Tests the preprocessing-profile registry module:
//   Test 1: Unknown profile throws InvalidInputException containing the profile
//           id substring and the tabfm_models() SELECT hint.
//   Test 2: Registered tabfm_v1_minimal profile dispatches correctly and returns
//           a PreprocessedBatch with expected T/train_size for a tiny hand-built
//           ColumnDataCollection.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_profile_registry.hpp"
#include "tabfm_preprocess.hpp"
#include "tabfm_manifest.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types.hpp"

#include <string>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

//===----------------------------------------------------------------------===//
// Helpers: build a minimal ColumnDataCollection
//===----------------------------------------------------------------------===//

//! Returns a tiny ColumnDataCollection suitable for dispatching through
//! tabfm_v1_minimal. Layout: two columns — numeric feature (DOUBLE) + target
//! (DOUBLE, 2 non-NULL train rows and 1 NULL test row).
ColumnDataCollection BuildTinyCollection() {
	// Schema: feat (DOUBLE), target (DOUBLE)
	vector<LogicalType> types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	ColumnDataCollection col(Allocator::DefaultAllocator(), types);

	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);

	// Row 0: feat=1.0, target=10.0 (train)
	chunk.SetValue(0, 0, Value::DOUBLE(1.0));
	chunk.SetValue(1, 0, Value::DOUBLE(10.0));
	// Row 1: feat=2.0, target=20.0 (train)
	chunk.SetValue(0, 1, Value::DOUBLE(2.0));
	chunk.SetValue(1, 1, Value::DOUBLE(20.0));
	// Row 2: feat=3.0, target=NULL (test)
	chunk.SetValue(0, 2, Value::DOUBLE(3.0));
	chunk.SetValue(1, 2, Value(LogicalType::DOUBLE));
	chunk.SetCardinality(3);

	col.Append(chunk);
	return col;
}

vector<PreprocessColumnSpec> BuildTinyColumnSpecs() {
	vector<PreprocessColumnSpec> specs(2);
	specs[0].name = "feat";
	specs[0].type = LogicalType::DOUBLE;
	specs[0].is_feature = true;
	specs[0].is_target = false;
	specs[1].name = "target";
	specs[1].type = LogicalType::DOUBLE;
	specs[1].is_feature = false;
	specs[1].is_target = true;
	return specs;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Test 1: Unknown profile throws with profile id and SELECT hint
//===----------------------------------------------------------------------===//

TEST_CASE("profile registry — unknown profile throws named error", "[tabfm][profile_registry]") {
	auto coll = BuildTinyCollection();
	auto specs = BuildTinyColumnSpecs();

	REQUIRE_THROWS_AS(
	    DispatchPreprocess("no_such_profile", coll, specs, PreprocessTask::REGRESSION),
	    InvalidInputException);

	// Message must contain the profile id and the tabfm_models() hint
	// (substring check — not full sentence, as per CLAUDE.md comment discipline)
	REQUIRE_THROWS_WITH(
	    DispatchPreprocess("no_such_profile", coll, specs, PreprocessTask::REGRESSION),
	    Catch::Contains("no_such_profile") && Catch::Contains("tabfm_models()"));
}

//===----------------------------------------------------------------------===//
// Test 2: tabfm_v1_minimal profile dispatches and returns expected batch shape
//===----------------------------------------------------------------------===//

TEST_CASE("profile registry — tabfm_v1_minimal dispatches correctly", "[tabfm][profile_registry]") {
	auto coll = BuildTinyCollection();
	auto specs = BuildTinyColumnSpecs();

	// Should NOT throw — "tabfm_v1_minimal" is self-registered at static init.
	PreprocessedBatch batch;
	REQUIRE_NOTHROW(batch = DispatchPreprocess(kPreprocessProfileId, coll, specs, PreprocessTask::REGRESSION));

	// 3 total rows, 2 train rows (non-NULL target), 1 test row
	CHECK(batch.T == 3);
	CHECK(batch.train_size == 2);
	// Feature matrix must be non-empty (H >= 1 feature)
	CHECK(batch.H >= 1);
	CHECK(batch.x.size() == batch.T * batch.H);
}

//===----------------------------------------------------------------------===//
// Test 3: Built-in manifests use the registered profile id (CR-01 regression)
//
// Guards against a future "tabfm-v1" vs "tabfm_v1_minimal" mismatch: if the
// built-in manifest JSON is ever edited to use a different profile id string,
// every production predict call would immediately throw "unknown preprocessing
// profile" while CI tests pass (since all fixtures override the manifest).
// This test exercises the BUILT-IN manifest path directly.
//===----------------------------------------------------------------------===//

TEST_CASE("built-in manifests use the registered preprocessing profile id (CR-01)", "[tabfm][profile_registry]") {
	// Parse both built-in manifests through the same validation path as production.
	const ModelManifest cls_manifest = BuiltinTabFMManifest(TabFMTask::CLASSIFICATION);
	const ModelManifest reg_manifest = BuiltinTabFMManifest(TabFMTask::REGRESSION);

	// The preprocessing_profile field must exactly equal kPreprocessProfileId
	// (the registry key used in tabfm_profile_registry.cpp).
	CHECK(cls_manifest.preprocessing_profile == kPreprocessProfileId);
	CHECK(reg_manifest.preprocessing_profile == kPreprocessProfileId);

	// Also verify that the profile id from each built-in manifest actually
	// dispatches through the registry without throwing (end-to-end production
	// path for both tasks — IN-01: use matching task enum per manifest).
	auto coll = BuildTinyCollection();
	auto specs = BuildTinyColumnSpecs();
	REQUIRE_NOTHROW(DispatchPreprocess(cls_manifest.preprocessing_profile, coll, specs, PreprocessTask::CLASSIFICATION));
	REQUIRE_NOTHROW(DispatchPreprocess(reg_manifest.preprocessing_profile, coll, specs, PreprocessTask::REGRESSION));
}
