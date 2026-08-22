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

// kV2 plus a model-provided GPU graph for one task (docs/GPU_HARDENING_PLAN.md
// P3): "ext_graph" mirrors "graph"'s task keying and is optional per task.
const char *kV2ExtGraph = R"json({
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
		"regression": "resources/mitra/mitra_reg.onnx"
	},
	"ext_graph": {
		"classification": "resources/mitra/mitra_clf_ext.onnx"
	},
	"migraphx_graph": {
		"classification": "resources/mitra/mitra_clf_migraphx.onnx"
	},
	"preprocessing_profile": "tabpfn-minimal-v1",
	"capabilities": ["classify", "regress"],
	"compute": {"cpu": "f32"}
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

TEST_CASE("model_spec: v2 ext_graph is optional and carried per task", "[tabfm][model_spec]") {
	// The model-provided GPU graph (docs/GPU_HARDENING_PLAN.md P3). Declared for
	// classification only, so regression must stay empty — and a manifest that
	// never mentions ext_graph must parse exactly as before (back-compat).
	auto spec = ParseModelSpec(kV2ExtGraph, "mitra_ext.json");
	REQUIRE(spec.tasks.at(TabFMTask::CLASSIFICATION).ext_graph == "resources/mitra/mitra_clf_ext.onnx");
	REQUIRE(spec.tasks.at(TabFMTask::REGRESSION).ext_graph.empty());
	// migraphx_graph is its own field, not a fallback of ext_graph: U1 measured
	// that MIGraphX cannot run a plain external-data ONNX, so conflating them
	// would trade a clear "no GPU graph" for a runtime MIGraphX error.
	REQUIRE(spec.tasks.at(TabFMTask::CLASSIFICATION).migraphx_graph == "resources/mitra/mitra_clf_migraphx.onnx");
	REQUIRE(spec.tasks.at(TabFMTask::REGRESSION).migraphx_graph.empty());

	auto plain = ParseModelSpec(kV2, "mitra.json");
	REQUIRE(plain.tasks.at(TabFMTask::CLASSIFICATION).ext_graph.empty());
	REQUIRE(plain.tasks.at(TabFMTask::REGRESSION).ext_graph.empty());
}

TEST_CASE("model_spec: SelectGpuGraph — model-provided wins, bundled needs its header match", "[tabfm][model_spec]") {
	// The pure gate behind GPU dispatch (docs/GPU_HARDENING_PLAN.md P3). This
	// exact gate — in its old, implicit form — is what locked every model but
	// tabfm-v1 out of the GPUs, so it gets the CanReuseSession treatment:
	// exhaustive, hardware-free, and mutation-honest (each case pins one rule).
	using G = GpuGraphSource;
	// A model-provided graph wins unconditionally: its offsets match its own
	// weights by construction, so the bundled header check is irrelevant to it.
	REQUIRE(SelectGpuGraph(true, false, false) == G::MODEL_PROVIDED);
	REQUIRE(SelectGpuGraph(true, true, false) == G::MODEL_PROVIDED);
	REQUIRE(SelectGpuGraph(true, true, true) == G::MODEL_PROVIDED);
	REQUIRE(SelectGpuGraph(true, false, true) == G::MODEL_PROVIDED);
	// The bundled tabfm-v1 graph needs BOTH: present and header-matched.
	REQUIRE(SelectGpuGraph(false, true, true) == G::BUNDLED);
	REQUIRE(SelectGpuGraph(false, true, false) == G::NONE);
	// A header match without a bundled graph is meaningless.
	REQUIRE(SelectGpuGraph(false, false, true) == G::NONE);
	REQUIRE(SelectGpuGraph(false, false, false) == G::NONE);
}

TEST_CASE("model_spec: NoGpuGraphMessage names the model and the real fix, not ep_path", "[tabfm][model_spec]") {
	// GPU_HARDENING_PLAN follow-up, found running the examples on a pod: with
	// ep_path SET and device='cuda', a model without a CUDA-servable graph
	// errored with "SET anofox_tabfm_ep_path" — the one thing that was already
	// configured. The message must name the model, the missing artifact, and
	// the two real fixes (register a graph, or use cpu).
	auto msg = duckdb::anofox::NoGpuGraphMessage("cuda", "mitra", "classification", "ext_graph");
	REQUIRE(msg.find("mitra") != std::string::npos);
	REQUIRE(msg.find("ext_graph") != std::string::npos);
	REQUIRE(msg.find("classification") != std::string::npos);
	REQUIRE(msg.find("device 'cuda'") != std::string::npos);
	REQUIRE(msg.find("anofox_tabfm_device='cpu'") != std::string::npos);
	// The one anti-requirement: it must NOT send the user to ep_path.
	REQUIRE(msg.find("ep_path") == std::string::npos);

	auto rocm = duckdb::anofox::NoGpuGraphMessage("rocm", "my-model", "regression", "migraphx_graph");
	REQUIRE(rocm.find("migraphx_graph") != std::string::npos);
	REQUIRE(rocm.find("regression_migraphx_graph") != std::string::npos);
}

TEST_CASE("model_spec: IsExplicitGpuRequest — only the named device (or its alias) throws", "[tabfm][model_spec]") {
	using duckdb::anofox::IsExplicitGpuRequest;
	// Explicit requests hard-error on a missing graph; 'auto' must stay a
	// graceful CPU fallback — that asymmetry is the whole point.
	REQUIRE(IsExplicitGpuRequest("cuda", "cuda"));
	REQUIRE_FALSE(IsExplicitGpuRequest("auto", "cuda"));
	REQUIRE_FALSE(IsExplicitGpuRequest("cpu", "cuda"));
	REQUIRE(IsExplicitGpuRequest("rocm", "rocm"));
	REQUIRE(IsExplicitGpuRequest("migraphx", "rocm")); // documented alias
	REQUIRE_FALSE(IsExplicitGpuRequest("auto", "rocm"));
	REQUIRE_FALSE(IsExplicitGpuRequest("cuda", "rocm"));
}

TEST_CASE("model_spec: bundled GPU graph ids are model-qualified, tabfm-v1 keeps legacy names", "[tabfm][model_spec]") {
	using duckdb::anofox::BundledGpuGraphId;
	// tabfm-v1's graphs predate multi-model bundling and keep their unqualified
	// resource ids — renaming them would orphan the embedded resources.
	REQUIRE(BundledGpuGraphId("tabfm-v1", "ext", "classification") == "graph_ext_classification");
	REQUIRE(BundledGpuGraphId("tabfm-v1", "migraphx", "regression") == "graph_migraphx_regression");
	// Every other model gets a model-qualified id.
	REQUIRE(BundledGpuGraphId("mitra", "ext", "classification") == "graph_ext_mitra_classification");
	REQUIRE(BundledGpuGraphId("mitra", "migraphx", "regression") == "graph_migraphx_mitra_regression");
}

TEST_CASE("model_spec: weights-header table is (model, task)-keyed", "[tabfm][model_spec]") {
	using duckdb::anofox::ExpectedWeightsHeaderShaFor;
	// tabfm-v1's hashes (tools/make_external_graph.py output, unchanged).
	REQUIRE(ExpectedWeightsHeaderShaFor("tabfm-v1", "classification") ==
	        "534d6d38b49b323bb38682858f232573c254689df03d3d9f17e7504716a31d96");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabfm-v1", "regression") ==
	        "35c346e4e29f61b493a9e601e66bf0ae241d0fb76623a3336c61408cfc3e88d0");
	// mitra's, from the same tool against the HF-downloaded safetensors.
	REQUIRE(ExpectedWeightsHeaderShaFor("mitra", "classification") ==
	        "cb1a261bb3d9ca505e0db66e21df85bec6777f7e104247d4a71a8eb8d8b3b96a");
	REQUIRE(ExpectedWeightsHeaderShaFor("mitra", "regression") ==
	        "44f9293fa2d81ccf56dffab09600ec4f775c5c30bbc2af551cf4f6b2cf889f01");
	// Unknown (model, task) means "no bundled GPU graph": empty, never a guess.
	// (tabpfn-v2 gained a real entry when the catalog landed; use a model that
	// genuinely has none.)
	REQUIRE(ExpectedWeightsHeaderShaFor("mitra", "generate") == "");
	REQUIRE(ExpectedWeightsHeaderShaFor("some-user-model", "classification") == "");
}

TEST_CASE("model_spec: the whole catalog has (model, task) header hashes for ext graphs", "[tabfm][model_spec]") {
	using duckdb::anofox::ExpectedWeightsHeaderShaFor;
	// tools/make_external_graph.py output against each model's HF-downloaded
	// safetensors. These gate the CUDA + CPU-low-memory ext path; the
	// single_eval_pos family (x, y only — no train_size input) deliberately
	// bundles NO migraphx variant: MIGraphX compiles per fixed shape, and for
	// those models y's LENGTH is the train/test split, so bucketing would need
	// a compile per distinct train_size.
	REQUIRE(ExpectedWeightsHeaderShaFor("tabpfn-v2", "classification") ==
	        "e97043c10b4572d6011cb1e389db2c7d57425213c761288f935188e25e953362");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabpfn-v2", "regression") ==
	        "ca4435a405cbd17afc8cf08346954705d7cc01fc6d08bdcf9c06363faed867a9");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabicl-v2", "classification") ==
	        "085731de6a7b33e6fcbda7e1b3cba725d798d30f93ec3cdc91c3ac2c2c762d3f");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabicl-v2", "regression") ==
	        "d792dd9433bdf78773eddcd4bda3e0e49550aec0a2df1a0bfa36afebf320e8ae");
	REQUIRE(ExpectedWeightsHeaderShaFor("orion-bix", "classification") ==
	        "c2b7ff39add2b0c1c2d3ddabbaf413e8c15f433b620f3091f0562f376255d166");
	REQUIRE(ExpectedWeightsHeaderShaFor("orion-bix", "regression") == "");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabpfn-v2-5", "classification") ==
	        "b230477af81d4ac5bff856b2f9dcc281d5b9a04d659a5dee335553f0f49897ea");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabpfn-v2-5", "regression") ==
	        "8865ee281d0172e31e1a03d1d43057ac8e69b88a27b2ce0a93ee77b865f45737");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabpfn-v3", "classification") ==
	        "c0f3a23322d1ec039356b618565e5e1c62378613e3081167881220968933f04b");
	REQUIRE(ExpectedWeightsHeaderShaFor("tabpfn-v3", "regression") ==
	        "92cf58d84bf968d6e5de90f7ccc29c0d1e79e8f0e9a04c7e0ab66274185d061c");
}

TEST_CASE("model_spec: catalog bundled ids use the resource stems, not the registry ids", "[tabfm][model_spec]") {
	using duckdb::anofox::BundledGpuGraphId;
	// Registry ids (tabpfn-v2, tabicl-v2, ...) differ from the resource file
	// stems (tabpfn, tabicl, ...) — the id function owns that mapping so the
	// engine never string-mangles.
	REQUIRE(BundledGpuGraphId("tabpfn-v2", "ext", "classification") == "graph_ext_tabpfn_classification");
	REQUIRE(BundledGpuGraphId("tabpfn-v2-5", "ext", "regression") == "graph_ext_tabpfn25_regression");
	REQUIRE(BundledGpuGraphId("tabpfn-v3", "ext", "classification") == "graph_ext_tabpfn3_classification");
	REQUIRE(BundledGpuGraphId("tabicl-v2", "ext", "regression") == "graph_ext_tabicl_regression");
	REQUIRE(BundledGpuGraphId("orion-bix", "ext", "classification") == "graph_ext_orion_bix_classification");
}
