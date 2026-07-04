// Catch2 tests for tabfm_manifest — WS-B.
//
// Covers: valid manifest round-trip, every validation error, engine-profile
// defaulting, URL derivation, the built-in TabFM v1 manifests, file loading,
// and the optional WS-A fixture test/fixtures/manifest.json.

#include "catch.hpp"

#include "tabfm_manifest.hpp"

#include "duckdb/common/exception.hpp"

#include <cstdio>
#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

const char *kValidManifest = R"({
	"model": "fixture-v1",
	"task": "classification",
	"repo": "anofox/tabfm-fixture",
	"revision": "v2",
	"files": [
		{"path": "classification/model.safetensors", "bytes": 1234},
		{"path": "extra.json", "url": "https://example.org/extra.json", "bytes": 9}
	],
	"graph": "graph_classification",
	"tensor_map": {"onnx.w1": "st.w1", "onnx.b1": "st.b1"},
	"preprocessing_profile": "tabfm-v1",
	"license": "apache-2.0",
	"engine_profiles": {"cuda": {"dtype": "bf16"}, "rocm": {"dtype": "fp16"}}
})";

} // anonymous namespace

TEST_CASE("manifest: valid manifest round-trips", "[tabfm][manifest]") {
	auto manifest = ParseModelManifest(kValidManifest, "fixture.json");
	REQUIRE(manifest.model == "fixture-v1");
	REQUIRE(manifest.task == TabFMTask::CLASSIFICATION);
	REQUIRE(manifest.repo == "anofox/tabfm-fixture");
	REQUIRE(manifest.revision == "v2");
	REQUIRE(manifest.files.size() == 2);
	REQUIRE(manifest.files[0].path == "classification/model.safetensors");
	REQUIRE(manifest.files[0].bytes == 1234);
	REQUIRE(manifest.files[0].url.empty());
	REQUIRE(manifest.files[1].url == "https://example.org/extra.json");
	REQUIRE(manifest.graph == "graph_classification");
	REQUIRE(manifest.tensor_map_path.empty());
	REQUIRE(manifest.tensor_map.size() == 2);
	REQUIRE(manifest.tensor_map.at("onnx.w1") == "st.w1");
	REQUIRE(manifest.preprocessing_profile == "tabfm-v1");
	REQUIRE(manifest.license == "apache-2.0");
	// explicit profiles kept, cpu/f32 defaulted in
	REQUIRE(manifest.engine_profiles.size() == 3);
	REQUIRE(manifest.engine_profiles.at("cpu").dtype == "f32");
	REQUIRE(manifest.engine_profiles.at("cuda").dtype == "bf16");
	REQUIRE(manifest.engine_profiles.at("rocm").dtype == "fp16");
}

TEST_CASE("manifest: tensor_map may be a file path", "[tabfm][manifest]") {
	auto manifest = ParseModelManifest(R"({
		"model": "m", "task": "regression", "repo": "a/b",
		"files": [{"path": "regression/model.safetensors", "bytes": 10}],
		"graph": "graph_regression", "tensor_map": "tensor_map_regression.json",
		"preprocessing_profile": "p", "license": "l"
	})");
	REQUIRE(manifest.task == TabFMTask::REGRESSION);
	REQUIRE(manifest.tensor_map_path == "tensor_map_regression.json");
	REQUIRE(manifest.tensor_map.empty());
}

TEST_CASE("manifest: defaults", "[tabfm][manifest]") {
	auto manifest = ParseModelManifest(R"({
		"model": "m", "task": "classification", "repo": "a/b",
		"files": [{"path": "f.safetensors", "bytes": 1}],
		"graph": "g", "preprocessing_profile": "p", "license": "l"
	})");
	SECTION("revision defaults to main") {
		REQUIRE(manifest.revision == "main");
	}
	SECTION("engine_profiles defaults to cpu/f32") {
		REQUIRE(manifest.engine_profiles.size() == 1);
		REQUIRE(manifest.engine_profiles.at("cpu").dtype == "f32");
	}
	SECTION("tensor_map defaults to empty (identity mapping)") {
		REQUIRE(manifest.tensor_map.empty());
		REQUIRE(manifest.tensor_map_path.empty());
	}
}

TEST_CASE("manifest: URL resolution", "[tabfm][manifest]") {
	auto manifest = ParseModelManifest(kValidManifest);
	SECTION("derived from repo/revision/path when no explicit url") {
		REQUIRE(ResolveManifestFileUrl(manifest, manifest.files[0]) ==
		        "https://huggingface.co/anofox/tabfm-fixture/resolve/v2/classification/model.safetensors");
	}
	SECTION("explicit url wins") {
		REQUIRE(ResolveManifestFileUrl(manifest, manifest.files[1]) == "https://example.org/extra.json");
	}
}

TEST_CASE("manifest: validation errors", "[tabfm][manifest]") {
	auto parse = [](const string &json) {
		return ParseModelManifest(json, "bad_manifest.json");
	};
	SECTION("malformed JSON") {
		REQUIRE_THROWS_AS(parse("{nope"), InvalidInputException);
		REQUIRE_THROWS_WITH(parse("{nope"), Contains("bad_manifest.json"));
	}
	SECTION("root is not an object") {
		REQUIRE_THROWS_AS(parse("[1,2]"), InvalidInputException);
	}
	SECTION("unknown task names the valid values") {
		auto json = string(R"({"model":"m","task":"ranking","repo":"a/b",)"
		                   R"("files":[{"path":"f","bytes":1}],"graph":"g",)"
		                   R"("preprocessing_profile":"p","license":"l"})");
		REQUIRE_THROWS_AS(parse(json), InvalidInputException);
		REQUIRE_THROWS_WITH(parse(json),
		                    Contains("ranking") && Contains("classification") && Contains("regression"));
	}
	SECTION("missing required fields name the field and the manifest path") {
		// drop one required field at a time
		const char *required[] = {"model", "task", "files", "graph", "preprocessing_profile", "license"};
		const char *full = R"({"model":"m","task":"classification","repo":"a/b",)"
		                   R"("files":[{"path":"f","bytes":1}],"graph":"g",)"
		                   R"("preprocessing_profile":"p","license":"l"})";
		for (auto field : required) {
			// rebuild the JSON without this field
			string json = "{";
			bool first = true;
			auto add = [&](const string &key, const string &raw_value) {
				if (key == field) {
					return;
				}
				if (!first) {
					json += ",";
				}
				first = false;
				json += "\"" + key + "\":" + raw_value;
			};
			add("model", "\"m\"");
			add("task", "\"classification\"");
			add("repo", "\"a/b\"");
			add("files", R"([{"path":"f","bytes":1}])");
			add("graph", "\"g\"");
			add("preprocessing_profile", "\"p\"");
			add("license", "\"l\"");
			json += "}";
			(void)full;
			INFO("missing field: " << field);
			REQUIRE_THROWS_AS(parse(json), InvalidInputException);
			REQUIRE_THROWS_WITH(parse(json), Contains(field) && Contains("bad_manifest.json"));
		}
	}
	SECTION("files must be a non-empty array") {
		REQUIRE_THROWS_AS(parse(R"({"model":"m","task":"classification","repo":"a/b","files":[],)"
		                        R"("graph":"g","preprocessing_profile":"p","license":"l"})"),
		                  InvalidInputException);
	}
	SECTION("file entry missing path") {
		REQUIRE_THROWS_AS(parse(R"({"model":"m","task":"classification","repo":"a/b",)"
		                        R"("files":[{"bytes":1}],)"
		                        R"("graph":"g","preprocessing_profile":"p","license":"l"})"),
		                  InvalidInputException);
	}
	SECTION("file entry missing bytes") {
		REQUIRE_THROWS_AS(parse(R"({"model":"m","task":"classification","repo":"a/b",)"
		                        R"("files":[{"path":"f"}],)"
		                        R"("graph":"g","preprocessing_profile":"p","license":"l"})"),
		                  InvalidInputException);
	}
	SECTION("file entry with negative bytes") {
		REQUIRE_THROWS_AS(parse(R"({"model":"m","task":"classification","repo":"a/b",)"
		                        R"("files":[{"path":"f","bytes":-5}],)"
		                        R"("graph":"g","preprocessing_profile":"p","license":"l"})"),
		                  InvalidInputException);
	}
	SECTION("no repo and no explicit url") {
		auto json = string(R"({"model":"m","task":"classification",)"
		                   R"("files":[{"path":"f","bytes":1}],"graph":"g",)"
		                   R"("preprocessing_profile":"p","license":"l"})");
		REQUIRE_THROWS_AS(parse(json), InvalidInputException);
		REQUIRE_THROWS_WITH(parse(json), Contains("repo") && Contains("url"));
	}
	SECTION("tensor_map of the wrong type") {
		REQUIRE_THROWS_AS(parse(R"({"model":"m","task":"classification","repo":"a/b",)"
		                        R"("files":[{"path":"f","bytes":1}],"graph":"g","tensor_map":42,)"
		                        R"("preprocessing_profile":"p","license":"l"})"),
		                  InvalidInputException);
	}
	SECTION("engine profile with invalid dtype names the valid values") {
		auto json = string(R"({"model":"m","task":"classification","repo":"a/b",)"
		                   R"("files":[{"path":"f","bytes":1}],"graph":"g",)"
		                   R"("preprocessing_profile":"p","license":"l",)"
		                   R"("engine_profiles":{"cpu":{"dtype":"int8"}}})");
		REQUIRE_THROWS_AS(parse(json), InvalidInputException);
		REQUIRE_THROWS_WITH(parse(json), Contains("int8") && Contains("f32"));
	}
	SECTION("engine profile missing dtype") {
		REQUIRE_THROWS_AS(parse(R"({"model":"m","task":"classification","repo":"a/b",)"
		                        R"("files":[{"path":"f","bytes":1}],"graph":"g",)"
		                        R"("preprocessing_profile":"p","license":"l",)"
		                        R"("engine_profiles":{"cuda":{}}})"),
		                  InvalidInputException);
	}
}

TEST_CASE("manifest: built-in TabFM v1 manifests", "[tabfm][manifest]") {
	SECTION("classification") {
		auto manifest = BuiltinTabFMManifest(TabFMTask::CLASSIFICATION);
		REQUIRE(manifest.model == "tabfm-v1");
		REQUIRE(manifest.task == TabFMTask::CLASSIFICATION);
		REQUIRE(manifest.repo == "google/tabfm-1.0.0-pytorch");
		REQUIRE(manifest.revision == "main");
		REQUIRE(manifest.files.size() == 1);
		REQUIRE(manifest.files[0].path == "classification/model.safetensors");
		REQUIRE(manifest.files[0].bytes == 6557888408ULL);
		REQUIRE(manifest.graph == "graph_classification");
		REQUIRE(manifest.license == "tabfm-non-commercial-v1.0");
		REQUIRE(manifest.engine_profiles.at("cpu").dtype == "f32");
		REQUIRE(manifest.engine_profiles.at("cuda").dtype == "bf16");
		REQUIRE(manifest.engine_profiles.at("rocm").dtype == "fp16");
		REQUIRE(ResolveManifestFileUrl(manifest, manifest.files[0]) ==
		        "https://huggingface.co/google/tabfm-1.0.0-pytorch/resolve/main/"
		        "classification/model.safetensors");
	}
	SECTION("regression") {
		auto manifest = BuiltinTabFMManifest(TabFMTask::REGRESSION);
		REQUIRE(manifest.model == "tabfm-v1");
		REQUIRE(manifest.task == TabFMTask::REGRESSION);
		REQUIRE(manifest.files.size() == 1);
		REQUIRE(manifest.files[0].path == "regression/model.safetensors");
		REQUIRE(manifest.graph == "graph_regression");
		REQUIRE(manifest.license == "tabfm-non-commercial-v1.0");
	}
	SECTION("raw JSON literals are exposed for registration") {
		REQUIRE(string(BuiltinTabFMManifestJson(TabFMTask::CLASSIFICATION)).find("6557888408") != string::npos);
		REQUIRE(string(BuiltinTabFMManifestJson(TabFMTask::REGRESSION)).find("regression") != string::npos);
	}
}

TEST_CASE("manifest: load from file", "[tabfm][manifest]") {
	auto path = string("duckdb_unittest_tempdir/tabfm_wsb_manifest_test.json");
	{
		std::ofstream out(path, std::ios::trunc);
		if (!out.good()) {
			WARN("cannot write temp manifest under duckdb_unittest_tempdir - skipping file test");
			SUCCEED();
			return;
		}
		out << kValidManifest;
	}
	auto manifest = LoadModelManifestFile(path);
	REQUIRE(manifest.model == "fixture-v1");
	REQUIRE(manifest.files[0].bytes == 1234);
	std::remove(path.c_str());

	SECTION("missing file raises an IO error naming the path") {
		REQUIRE_THROWS_AS(LoadModelManifestFile("/nonexistent/tabfm/manifest.json"), IOException);
		REQUIRE_THROWS_WITH(LoadModelManifestFile("/nonexistent/tabfm/manifest.json"),
		                    Contains("/nonexistent/tabfm/manifest.json"));
	}
}

TEST_CASE("manifest: WS-A fixture manifest parses when present", "[tabfm][manifest]") {
	const char *candidates[] = {"test/fixtures/manifest.json", "../test/fixtures/manifest.json"};
	string path;
	for (auto candidate : candidates) {
		std::ifstream probe(candidate);
		if (probe.good()) {
			path = candidate;
			break;
		}
	}
	if (path.empty()) {
		WARN("test/fixtures/manifest.json not present (WS-A not landed yet) - skipping");
		SUCCEED();
		return;
	}
	auto manifest = LoadModelManifestFile(path);
	REQUIRE(!manifest.model.empty());
	REQUIRE(!manifest.files.empty());
	REQUIRE(manifest.engine_profiles.count("cpu") == 1);
}
