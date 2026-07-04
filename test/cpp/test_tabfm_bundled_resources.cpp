//===----------------------------------------------------------------------===//
// Catch2 tests for the bundled weight-free resources (tabfm_bundled_resources):
// the .onnx graphs and tensor maps from resources/ compiled into the binary so
// the built-in model flow works with no companion files on disk.
//
// The resources are Google-weight-free (license wall): the graphs are the real
// weight-free computation graphs, the tensor maps are {onnx name -> st key}.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_bundled_resources.hpp"
#include "tabfm_ort_engine.hpp"

#include "../../duckdb/third_party/yyjson/include/yyjson.hpp"

#include <cstring>
#include <fstream>
#include <iterator>

using namespace duckdb;
using namespace duckdb::anofox;
using namespace duckdb_yyjson;

namespace {

string ThisDir() {
	string file = __FILE__;
	return file.substr(0, file.find_last_of("/\\"));
}

string ResourcesDir() {
	return ThisDir() + "/../../resources";
}

string ReadDiskFile(const string &path) {
	std::ifstream file(path, std::ios::binary);
	REQUIRE(file.good());
	return string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("tabfm_bundled_resources: graphs are embedded byte-exact", "[tabfm][bundled]") {
	struct Case {
		const char *id;
		const char *file;
	};
	for (auto &c : {Case {"graph_classification", "graph_classification.onnx"},
	                Case {"graph_regression", "graph_regression.onnx"}}) {
		auto disk = ReadDiskFile(ResourcesDir() + "/" + c.file);
		REQUIRE(!disk.empty());

		// bundled under both the bare id and the ".onnx" filename
		for (auto *id : {c.id, c.file}) {
			auto res = GetBundledResource(id);
			INFO("resource id: " << id);
			REQUIRE(res.data != nullptr);
			REQUIRE(res.size == disk.size());
			REQUIRE(std::memcmp(res.data, disk.data(), disk.size()) == 0);
		}
	}
}

TEST_CASE("tabfm_bundled_resources: tensor maps are embedded and parse", "[tabfm][bundled]") {
	for (auto *id : {"tensor_map_classification.json", "tensor_map_regression.json"}) {
		auto res = GetBundledResource(id);
		INFO("resource id: " << id);
		REQUIRE(res.data != nullptr);
		REQUIRE(res.size > 0);

		auto disk = ReadDiskFile(ResourcesDir() + "/" + id);
		REQUIRE(res.size == disk.size());
		REQUIRE(std::memcmp(res.data, disk.data(), disk.size()) == 0);

		// parses as JSON with an "initializers" object carrying the real model's
		// many tensors ({onnx name -> st key})
		auto *doc = yyjson_read(res.data, res.size, 0);
		REQUIRE(doc != nullptr);
		auto *root = yyjson_doc_get_root(doc);
		auto *inits = yyjson_obj_get(root, "initializers");
		REQUIRE(inits != nullptr);
		REQUIRE(yyjson_is_obj(inits));
		REQUIRE(yyjson_obj_size(inits) > 100);
		yyjson_doc_free(doc);
	}
}

TEST_CASE("tabfm_bundled_resources: unknown ids and paths do not match", "[tabfm][bundled]") {
	REQUIRE(GetBundledResource("nope").data == nullptr);
	REQUIRE(GetBundledResource("").data == nullptr);
	// a filesystem path must fall through to disk resolution, never the bundle
	REQUIRE(GetBundledResource("/some/dir/graph_classification.onnx").data == nullptr);
	REQUIRE(GetBundledResource("resources/graph_classification.onnx").data == nullptr);
}

TEST_CASE("tabfm_bundled_resources: bundled graph is a valid weight-free ONNX graph", "[tabfm][bundled]") {
	// Loading the real bundled graph through ORT with no injected weights must
	// fail cleanly with the weights-not-loaded remediation (S01/S02): this both
	// proves the embedded bytes are a loadable graph and that the built-in flow
	// will surface the right error when weights are missing.
	auto graph = GetBundledResource("graph_classification");
	REQUIRE(graph.data != nullptr);

	TabFMSessionConfig config;
	config.device_id = "cpu";
	config.model_tag = "classification";
	try {
		CreateSession(graph.data, graph.size, {}, config);
		FAIL("expected an exception (weight-free graph, no injection)");
	} catch (std::exception &error) {
		string message = error.what();
		REQUIRE(message.find("model weights are not loaded") != string::npos);
		REQUIRE(message.find("tabfm_download") != string::npos);
	}
}
