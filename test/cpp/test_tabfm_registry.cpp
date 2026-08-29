// Catch2 tests for the model registry: the built-in catalog plus models
// registered in SQL (CALL tabfm_register_model → ModelRegistry::Build(specs)).

#include "catch.hpp"

#include "tabfm_registry.hpp"
#include "tabfm_model_spec.hpp"

#include "duckdb/common/exception.hpp"

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

const char *kAcmeClf = R"json({
	"schema_version": 2,
	"id": "acme-clf",
	"family": "icl-transformer",
	"license": {"id": "apache-2.0", "commercial": true, "redistributable": true},
	"preprocessing_profile": "tabpfn-minimal-v1",
	"weights": {"classification": {"repo": "acme/clf", "files": [{"path": "model.safetensors"}]}},
	"graph": {"classification": "clf.onnx"},
	"capabilities": ["classify"]
})json";

const char *kAcmeReg = R"json({
	"schema_version": 2,
	"id": "acme-reg",
	"family": "icl-transformer",
	"license": {"id": "apache-2.0", "commercial": true, "redistributable": true},
	"preprocessing_profile": "tabpfn-minimal-v1",
	"weights": {"regression": {"repo": "acme/reg", "files": [{"path": "model.safetensors"}]}},
	"graph": {"regression": "reg.onnx"},
	"capabilities": ["regress"]
})json";

// A SQL-registered model carries a (non-empty) source_dir, marking it a
// disk-resolved user model rather than a built-in.
ModelSpec Registered(const char *json) {
	auto s = ParseModelSpec(json, "(test)");
	s.source_dir = ".";
	return s;
}

//! Size of the built-in catalog. Derived, not hard-coded, so onboarding a model
//! does not break every registry test — the assertions below are about
//! registration *arithmetic* (added / shadowed), not the catalog's size.
size_t BuiltinCount() {
	return BuiltinModelSpecs().size();
}

} // anonymous namespace

TEST_CASE("registry: the built-in catalog is always present", "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("tabfm-v1"));
	REQUIRE(reg.ImplicitDefault().empty());
	auto &g = reg.Get("tabfm-v1");
	REQUIRE(g.HasTask(TabFMTask::CLASSIFICATION));
	REQUIRE(g.HasTask(TabFMTask::REGRESSION));
	REQUIRE(g.HasCapability("classify"));
	REQUIRE(g.HasCapability("regress"));
	REQUIRE(g.license.commercial == false);
	REQUIRE(g.license.gate_setting == "accept_hf_license"); // stays gated
	// the commercial-clean models ship built in too
	REQUIRE(reg.Has("mitra"));
	REQUIRE(reg.Has("tabpfn-v2"));
	REQUIRE(reg.Has("tabicl-v2"));
	REQUIRE(reg.Get("mitra").license.commercial == true);
	REQUIRE(reg.Has("orion-bix"));
	// classify-only built-in: capability gating is real, not hypothetical
	REQUIRE(reg.Get("orion-bix").HasCapability("classify"));
	REQUIRE_FALSE(reg.Get("orion-bix").HasCapability("regress"));
	REQUIRE(reg.Models().size() == BuiltinCount());
	// several models + no selection → actionable ambiguity error; explicit works
	REQUIRE_THROWS_WITH(reg.Resolve("", ""), Contains("registered"));
	REQUIRE(reg.Resolve("tabfm-v1", "").id == "tabfm-v1");
	// unknown id → actionable error naming the id + the registered ids
	REQUIRE_THROWS_WITH(reg.Get("nope"), Contains("nope") && Contains("tabfm-v1"));
	REQUIRE_THROWS_WITH(reg.Resolve("nope", ""), Contains("nope"));
}

TEST_CASE("registry: a single SQL-registered model is the implicit default", "[tabfm][registry]") {
	auto reg = ModelRegistry::Build({Registered(kAcmeClf)});
	REQUIRE(reg.Has("tabfm-v1"));
	REQUIRE(reg.Has("acme-clf"));
	REQUIRE(reg.Models().size() == BuiltinCount() + 1);
	REQUIRE(reg.ImplicitDefault() == "acme-clf");
	REQUIRE(reg.Get("acme-clf").license.commercial == true);

	// one registration → bare resolve picks it
	REQUIRE(reg.Resolve("", "").id == "acme-clf");
	// anofox_tabfm_default_model overrides the implicit default
	REQUIRE(reg.Resolve("", "tabfm-v1").id == "tabfm-v1");
	// per-call model := wins over everything
	REQUIRE(reg.Resolve("tabfm-v1", "acme-clf").id == "tabfm-v1");
}

TEST_CASE("registry: several registered models → selection is explicit", "[tabfm][registry]") {
	auto reg = ModelRegistry::Build({Registered(kAcmeClf), Registered(kAcmeReg)});
	REQUIRE(reg.Models().size() == BuiltinCount() + 2);
	REQUIRE(reg.ImplicitDefault().empty());
	// no selection + >1 model → an actionable ambiguity error
	REQUIRE_THROWS_WITH(reg.Resolve("", ""), Contains("registered"));
	REQUIRE(reg.Resolve("acme-clf", "").id == "acme-clf");
	REQUIRE(reg.Resolve("acme-reg", "").id == "acme-reg");
}

TEST_CASE("registry: a registered id shadows a built-in of the same id", "[tabfm][registry]") {
	// register a model whose id collides with the built-in 'mitra'
	auto spec = Registered(kAcmeClf);
	spec.id = "mitra";
	auto reg = ModelRegistry::Build({spec});
	REQUIRE(reg.Models().size() == BuiltinCount()); // unchanged — the registration replaced the built-in
	// the registered model won: it has a source_dir (built-ins do not)
	REQUIRE_FALSE(reg.Get("mitra").source_dir.empty());
	REQUIRE(reg.Get("mitra").HasCapability("classify"));
}

TEST_CASE("registry: RealTabPFN-2.5 is a sibling entry that cannot clobber 2.5's cache",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("tabpfn-v2-5"));
	REQUIRE(reg.Has("tabpfn-v2-5-real"));

	auto &base = reg.Get("tabpfn-v2-5");
	auto &real = reg.Get("tabpfn-v2-5-real");

	// Same licence line — 2.5-real is the same release under the same terms.
	REQUIRE(real.license.id == base.license.id);
	REQUIRE(real.license.commercial == false);
	REQUIRE(real.license.gate_setting == "accept_hf_license");
	REQUIRE(real.HasCapability("classify"));
	REQUIRE(real.HasCapability("regress"));
	// Same architecture ⇒ same preprocessing profile and the same bundled graph.
	REQUIRE(real.preprocessing_profile == base.preprocessing_profile);

	// The cache slug is keyed by HF REPO (WeightsManifest::CacheSlug), and both
	// checkpoints live in Prior-Labs/tabpfn_2_5. Sharing files[].path would make
	// the two entries download over each other's weights and silently serve the
	// wrong checkpoint — the failure this assertion exists to prevent.
	REQUIRE(real.tasks.at(TabFMTask::CLASSIFICATION).files[0].path !=
	        base.tasks.at(TabFMTask::CLASSIFICATION).files[0].path);
	REQUIRE(real.tasks.at(TabFMTask::REGRESSION).files[0].path !=
	        base.tasks.at(TabFMTask::REGRESSION).files[0].path);
	// ... while still pointing at the same repo (so this is a real collision risk,
	// not one avoided by accident).
	REQUIRE(real.tasks.at(TabFMTask::CLASSIFICATION).repo ==
	        base.tasks.at(TabFMTask::CLASSIFICATION).repo);
	// The URLs must differ too: same path + same repo but a different URL would
	// still collide; different path + same URL would download the default twice.
	REQUIRE(real.tasks.at(TabFMTask::CLASSIFICATION).files[0].url !=
	        base.tasks.at(TabFMTask::CLASSIFICATION).files[0].url);
	REQUIRE(real.tasks.at(TabFMTask::REGRESSION).files[0].url !=
	        base.tasks.at(TabFMTask::REGRESSION).files[0].url);
}
