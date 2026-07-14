// Catch2 tests for the model registry (P2). Uses real temp manifest files on
// disk (no mocks) — the built-in TabFM v1 plus user manifests from a file or a
// directory, with the back-compat resolution rules.

#include "catch.hpp"

#include "tabfm_registry.hpp"

#include "duckdb/common/exception.hpp"

#include <filesystem>
#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

const char *kMitraV2 = R"json({
	"schema_version": 2,
	"id": "acme-clf",
	"family": "icl-transformer",
	"license": {"id": "apache-2.0", "commercial": true, "redistributable": true},
	"preprocessing_profile": "tabpfn-minimal-v1",
	"weights": {"classification": {"repo": "autogluon/mitra-classifier",
	                               "files": [{"path": "model.safetensors"}]}},
	"graph": {"classification": "mitra_clf.onnx"},
	"capabilities": ["classify"]
})json";

std::filesystem::path UniqueTmp(const char *stem) {
	// A per-test-case unique dir under the system temp.
	auto base = std::filesystem::temp_directory_path() / "tabfm_reg_test";
	std::filesystem::create_directories(base);
	return base / stem;
}

void WriteFile(const std::filesystem::path &path, const string &content) {
	std::ofstream f(path);
	f << content;
}

} // anonymous namespace

TEST_CASE("registry: built-in TabFM is always present", "[tabfm][registry]") {
	auto reg = ModelRegistry::Build("");
	REQUIRE(reg.Has("tabfm-v1"));
	REQUIRE(reg.ImplicitDefault().empty());
	auto &g = reg.Get("tabfm-v1");
	REQUIRE(g.HasTask(TabFMTask::CLASSIFICATION));
	REQUIRE(g.HasTask(TabFMTask::REGRESSION));
	REQUIRE(g.HasCapability("classify"));
	REQUIRE(g.HasCapability("regress"));
	REQUIRE(g.license.commercial == false);
	REQUIRE(g.license.gate_setting == "accept_hf_license"); // stays gated
	// the built-in catalog also ships the commercial-clean models
	REQUIRE(reg.Has("mitra"));
	REQUIRE(reg.Has("tabpfn-v2"));
	REQUIRE(reg.Has("tabicl-v2"));
	REQUIRE(reg.Get("mitra").license.commercial == true);
	// several models + no selection → actionable ambiguity error; explicit works
	REQUIRE_THROWS_WITH(reg.Resolve("", ""), Contains("registered"));
	REQUIRE(reg.Resolve("tabfm-v1", "").id == "tabfm-v1");
	// unknown id → actionable error naming the id + the registered ids
	REQUIRE_THROWS_WITH(reg.Get("nope"), Contains("nope") && Contains("tabfm-v1"));
	REQUIRE_THROWS_WITH(reg.Resolve("nope", ""), Contains("nope"));
}

TEST_CASE("registry: a manifest FILE adds a model and becomes the default", "[tabfm][registry]") {
	auto path = UniqueTmp("mitra_file.json");
	WriteFile(path, kMitraV2);

	auto reg = ModelRegistry::Build(path.string());
	REQUIRE(reg.Has("tabfm-v1"));
	REQUIRE(reg.Has("acme-clf"));
	REQUIRE(reg.Models().size() == 5);
	REQUIRE(reg.ImplicitDefault() == "acme-clf");
	REQUIRE(reg.Get("acme-clf").license.commercial == true);

	// back-compat: SET model_manifest = file → that model is active by default
	REQUIRE(reg.Resolve("", "").id == "acme-clf");
	// anofox_tabfm_default_model overrides the implicit default
	REQUIRE(reg.Resolve("", "tabfm-v1").id == "tabfm-v1");
	// per-call model := wins over everything
	REQUIRE(reg.Resolve("tabfm-v1", "acme-clf").id == "tabfm-v1");

	std::filesystem::remove(path);
}

TEST_CASE("registry: a DIRECTORY of manifests merges without an implicit default",
          "[tabfm][registry]") {
	auto dir = UniqueTmp("manifest_dir");
	std::filesystem::create_directories(dir);
	WriteFile(dir / "mitra.json", kMitraV2);

	auto reg = ModelRegistry::Build(dir.string());
	REQUIRE(reg.Models().size() == 5);
	REQUIRE(reg.ImplicitDefault().empty());
	// no selection + >1 model → an actionable ambiguity error
	REQUIRE_THROWS_WITH(reg.Resolve("", ""), Contains("registered"));
	// explicit selection still works
	REQUIRE(reg.Resolve("acme-clf", "").id == "acme-clf");

	std::filesystem::remove_all(dir);
}

TEST_CASE("registry: two manifests in a dir with the same id is an error", "[tabfm][registry]") {
	auto dir = UniqueTmp("dup_id_dir");
	std::filesystem::create_directories(dir);
	WriteFile(dir / "a.json", kMitraV2);
	WriteFile(dir / "b.json", kMitraV2); // both declare id "acme-clf"

	REQUIRE_THROWS_WITH(ModelRegistry::Build(dir.string()), Contains("acme-clf") && Contains("unique"));

	std::filesystem::remove_all(dir);
}
