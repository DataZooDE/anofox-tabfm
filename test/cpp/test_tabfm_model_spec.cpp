// Catch2 tests for tabfm_model_spec — the multi-model registry unit (P1).
//
// ParseModelSpec must accept BOTH the v1 manifest (flat task/files/graph/
// license-string) and the v2 manifest (schema_version:2, weights/graph keyed by
// task, license object, capabilities, tensor_contract, size_regime) — v1 → a
// single-task spec, v2 → a multi-task spec — with total back-compat.

#include "catch.hpp"

#include "tabfm_model_spec.hpp"

#include "duckdb/common/exception.hpp"

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

// A v1 manifest (the shape every current manifest/fixture uses).
const char *kV1 = R"({
	"model": "tabfm-v1",
	"task": "classification",
	"repo": "google/tabfm-1.0.0-pytorch",
	"revision": "main",
	"files": [{"path": "classification/model.safetensors", "bytes": 6557888408}],
	"graph": "graph_classification",
	"tensor_map": "tensor_map_classification.json",
	"preprocessing_profile": "tabfm-v1",
	"license": "tabfm-non-commercial-v1.0",
	"engine_profiles": {"cuda": {"dtype": "bf16"}, "rocm": {"dtype": "fp16"}}
})";

// A v2 manifest (Mitra-shaped: one model, two task checkpoints, license object,
// capabilities, tensor_contract, size_regime).
const char *kV2 = R"json({
	"schema_version": 2,
	"id": "mitra",
	"display_name": "Mitra (AWS AutoGluon)",
	"family": "icl-transformer",
	"license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": null},
	"weights": {
		"classification": {
			"repo": "autogluon/mitra-classifier", "revision": "main",
			"files": [{"path": "model.safetensors", "bytes": 317000000}]
		},
		"regression": {
			"repo": "autogluon/mitra-regressor",
			"files": [{"path": "model.safetensors", "bytes": 0}]
		}
	},
	"graph": {
		"classification": "resources/mitra/mitra_clf.onnx",
		"regression": "resources/mitra/mitra_reg.onnx",
		"tensor_map": "resources/mitra/mitra_tensor_map.json"
	},
	"tensor_contract": {
		"inputs": {
			"x":          {"name": "x",          "shape": ["1","T","H"], "dtype": "f32"},
			"y":          {"name": "y",          "shape": ["1","T"],     "dtype": "f32"},
			"train_size": {"name": "train_size", "shape": ["1"],         "dtype": "i64"},
			"cat_mask":   {"name": "cat_mask",   "shape": ["1","H"],     "dtype": "bool"},
			"d":          {"name": "d",          "shape": ["1"],         "dtype": "i64"}
		},
		"outputs": {"logits": {"name": "logits", "shape": ["1","T","C"], "dtype": "f32"}}
	},
	"preprocessing_profile": "tabpfn-minimal-v1",
	"capabilities": ["classify", "regress"],
	"size_regime": {"max_rows": 5000, "max_features": 100, "max_classes": 10},
	"compute": {"cpu": "f32", "gpu_precision_default": "bf16"}
})json";

} // anonymous namespace

TEST_CASE("model_spec: v1 manifest parses to a single-task spec", "[tabfm][model_spec]") {
	auto spec = ParseModelSpec(kV1, "v1.json");
	REQUIRE(spec.schema_version == 1);
	REQUIRE(spec.id == "tabfm-v1");
	REQUIRE(spec.license.id == "tabfm-non-commercial-v1.0");
	REQUIRE(spec.license.commercial == false); // conservative default for v1
	REQUIRE(spec.tasks.size() == 1);
	REQUIRE(spec.HasTask(TabFMTask::CLASSIFICATION));
	REQUIRE_FALSE(spec.HasTask(TabFMTask::REGRESSION));
	auto &clf = spec.tasks.at(TabFMTask::CLASSIFICATION);
	REQUIRE(clf.repo == "google/tabfm-1.0.0-pytorch");
	REQUIRE(clf.files.size() == 1);
	REQUIRE(clf.files[0].bytes == 6557888408);
	REQUIRE(clf.graph == "graph_classification");
	REQUIRE(clf.tensor_map_path == "tensor_map_classification.json");
	REQUIRE(clf.preprocessing_profile == "tabfm-v1");
	// capability inferred from the single task
	REQUIRE(spec.HasCapability("classify"));
	REQUIRE_FALSE(spec.HasCapability("regress"));
	// engine profiles: explicit kept + cpu/f32 defaulted in
	REQUIRE(spec.engine_profiles.at("cpu").dtype == "f32");
	REQUIRE(spec.engine_profiles.at("cuda").dtype == "bf16");
	// no declared tensor contract → engine falls back to built-in
	REQUIRE(spec.tensor_contract.empty());
}

TEST_CASE("model_spec: v2 manifest parses to a multi-task spec", "[tabfm][model_spec]") {
	auto spec = ParseModelSpec(kV2, "mitra.json");
	REQUIRE(spec.schema_version == 2);
	REQUIRE(spec.id == "mitra");
	REQUIRE(spec.display_name == "Mitra (AWS AutoGluon)");
	REQUIRE(spec.family == "icl-transformer");
	// license object
	REQUIRE(spec.license.id == "apache-2.0");
	REQUIRE(spec.license.commercial == true);
	REQUIRE(spec.license.redistributable == true);
	REQUIRE(spec.license.gate_setting.empty()); // null → no gate
	// two task checkpoints
	REQUIRE(spec.tasks.size() == 2);
	REQUIRE(spec.HasTask(TabFMTask::CLASSIFICATION));
	REQUIRE(spec.HasTask(TabFMTask::REGRESSION));
	REQUIRE(spec.tasks.at(TabFMTask::CLASSIFICATION).repo == "autogluon/mitra-classifier");
	REQUIRE(spec.tasks.at(TabFMTask::CLASSIFICATION).graph == "resources/mitra/mitra_clf.onnx");
	REQUIRE(spec.tasks.at(TabFMTask::REGRESSION).repo == "autogluon/mitra-regressor");
	REQUIRE(spec.tasks.at(TabFMTask::REGRESSION).revision == "main"); // defaulted
	// model-wide preprocessing mirrored into each task
	REQUIRE(spec.preprocessing_profile == "tabpfn-minimal-v1");
	REQUIRE(spec.tasks.at(TabFMTask::REGRESSION).preprocessing_profile == "tabpfn-minimal-v1");
	// tensor map (shared) applied to each task
	REQUIRE(spec.tasks.at(TabFMTask::CLASSIFICATION).tensor_map_path == "resources/mitra/mitra_tensor_map.json");
	// capabilities explicit
	REQUIRE(spec.HasCapability("classify"));
	REQUIRE(spec.HasCapability("regress"));
	REQUIRE_FALSE(spec.HasCapability("impute"));
	// size regime
	REQUIRE(spec.size_regime.max_rows == 5000);
	REQUIRE(spec.size_regime.max_features == 100);
	REQUIRE(spec.size_regime.max_classes == 10);
	// tensor contract declared
	REQUIRE_FALSE(spec.tensor_contract.empty());
	REQUIRE(spec.tensor_contract.inputs.size() == 5);
	REQUIRE(spec.tensor_contract.outputs.size() == 1);
	REQUIRE(spec.tensor_contract.outputs[0].logical == "logits");
	REQUIRE(spec.tensor_contract.outputs[0].dtype == "f32");
	// compute → engine profiles (cpu f32 always present)
	REQUIRE(spec.engine_profiles.at("cpu").dtype == "f32");
}

TEST_CASE("model_spec: unknown model id / missing fields error clearly", "[tabfm][model_spec]") {
	// v2 without an id
	REQUIRE_THROWS_WITH(ParseModelSpec(R"({"schema_version":2,"weights":{}})", "x.json"),
	                    Contains("id"));
	// v2, otherwise valid, but with no task checkpoints → errors on weights
	REQUIRE_THROWS_WITH(
	    ParseModelSpec(R"({"schema_version":2,"id":"m","license":{"id":"x"},"preprocessing_profile":"p","weights":{}})",
	                   "x.json"),
	    Contains("weights"));
}
