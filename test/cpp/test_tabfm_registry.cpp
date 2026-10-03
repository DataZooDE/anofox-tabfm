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


TEST_CASE("registry: TabDPT is commercially clean and shares one file across tasks",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("tabdpt"));
	auto &spec = reg.Get("tabdpt");

	// Apache-2.0 and ungated. Alongside mitra this is the catalog's only
	// permissively-licensed entry that does BOTH tasks, which is the main reason
	// it was worth onboarding.
	REQUIRE(spec.license.id == "apache-2.0");
	REQUIRE(spec.license.commercial == true);
	REQUIRE(spec.license.gate_setting.empty());
	REQUIRE(spec.HasCapability("classify"));
	REQUIRE(spec.HasCapability("regress"));

	// One checkpoint, one head: class logits followed by regression bins. Both
	// tasks therefore point at the SAME cached file, so a user pays one download
	// rather than two. This is the exact opposite of tabpfn-v2-5 vs -real, where
	// a shared path would silently serve the wrong weights -- there the repo is
	// shared and the checkpoints differ; here it is literally one checkpoint.
	auto &clf = spec.tasks.at(TabFMTask::CLASSIFICATION);
	auto &reg_task = spec.tasks.at(TabFMTask::REGRESSION);
	REQUIRE(clf.repo == reg_task.repo);
	REQUIRE(clf.files[0].path == reg_task.files[0].path);
	REQUIRE(clf.files[0].url == reg_task.files[0].url);

	// Layer 6 publish safetensors, so there is no .ckpt to convert -- the file
	// the manifest names is the one the engine injects.
	REQUIRE(clf.files[0].path == "model.safetensors");
}


TEST_CASE("registry: Orion-MSP is classify-only alongside its Orion-BiX sibling",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("orion-msp"));
	auto &msp = reg.Get("orion-msp");

	// MIT and ungated. With orion-bix, mitra and tabdpt this is the
	// commercially-clean half of the catalog.
	REQUIRE(msp.license.id == "mit");
	REQUIRE(msp.license.commercial == true);
	REQUIRE(msp.license.gate_setting.empty());

	// Upstream ships sklearn/classifier.py and no regressor, so the capability
	// gate must actually deny regression rather than let the engine try.
	REQUIRE(msp.HasCapability("classify"));
	REQUIRE_FALSE(msp.HasCapability("regress"));
	REQUIRE(msp.HasTask(TabFMTask::CLASSIFICATION));
	REQUIRE_FALSE(msp.HasTask(TabFMTask::REGRESSION));

	// Sibling, not alias: same vendor and licence, different checkpoint, and it
	// must not have inherited orion-bix's weights URL.
	auto &bix = reg.Get("orion-bix");
	REQUIRE(msp.license.id == bix.license.id);
	REQUIRE(msp.tasks.at(TabFMTask::CLASSIFICATION).files[0].url !=
	        bix.tasks.at(TabFMTask::CLASSIFICATION).files[0].url);
	REQUIRE(msp.tasks.at(TabFMTask::CLASSIFICATION).repo !=
	        bix.tasks.at(TabFMTask::CLASSIFICATION).repo);
}

TEST_CASE("registry: LimiX-2M ships built in, commercial under the Stable AI licence, gated, pinned to the release it was exported from",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("limix-2m"));
	auto &m = reg.Get("limix-2m");

	// The weights repo's LICENSE.txt (Stable AI Technology Co., Ltd. License v1.0, 2026-09) is
	// Apache-2.0 plus one added Section 10, so commercial use is permitted. The gate is therefore the
	// licence ACKNOWLEDGEMENT, not a commercial restriction. (The older upstream README says "academic,
	// commercial with authorization" and the model card is internally inconsistent; the newest
	// LICENSE.txt governs and docs/REAL_MODELS.md says so.)
	REQUIRE(m.license.id == "limix-2m-license-v1.0");
	REQUIRE(m.license.commercial == true);
	REQUIRE(m.license.redistributable == true);
	REQUIRE(m.license.gate_setting == "accept_hf_license");
	// Section 10's two obligations must be quoted where a user sees the model, verbatim.
	REQUIRE_THAT(m.license.attribution, Contains("Built with StableAI LimiX"));
	REQUIRE_THAT(m.license.attribution, Contains("Section 10"));
	REQUIRE_THAT(m.license.attribution, Contains("\"LimiX\" at the beginning of the name"));
	REQUIRE_THAT(m.license.attribution, Contains("Stable AI Technology Co., Ltd."));

	REQUIRE(m.HasCapability("classify"));
	REQUIRE(m.HasCapability("regress"));
	// the model standardises its features itself and the wrapper standardises the regression target
	// in-graph, so the engine must do neither
	REQUIRE(m.preprocessing_profile == "limix_v1_raw");
	REQUIRE(m.size_regime.max_classes == 10);

	// ONE checkpoint serves both tasks, so both tasks name the same file (downloaded once), pinned to the
	// commit the graphs were exported from. Its LFS sha256 is identical across every commit since release.
	for (auto task : {TabFMTask::CLASSIFICATION, TabFMTask::REGRESSION}) {
		auto &t = m.tasks.at(task);
		REQUIRE(t.repo == "stable-ai/LimiX-2M");
		REQUIRE(t.revision == "641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e");
		REQUIRE(t.files.size() == 1);
		REQUIRE_THAT(t.files[0].url, Contains("/resolve/641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e/LimiX-2M.ckpt"));
		REQUIRE(t.files[0].bytes == 9558253);
	}
	REQUIRE(m.tasks.at(TabFMTask::CLASSIFICATION).files[0].path ==
	        m.tasks.at(TabFMTask::REGRESSION).files[0].path);
}

TEST_CASE("registry: Causilo ships built in, non-commercial and gated, pinned to the release it was exported from",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("causilo"));
	auto &m = reg.Get("causilo");

	// Weights are under the Causilo License v1.0: non-commercial research/evaluation, and
	// hosted/API/SaaS use needs a separate license. The attribution must say so where a user
	// sees it, and the model must be gated like the other non-commercial entries.
	REQUIRE(m.license.id == "causilo-license-v1.0");
	REQUIRE(m.license.commercial == false);
	REQUIRE(m.license.redistributable == false);
	REQUIRE(m.license.gate_setting == "accept_hf_license");
	REQUIRE_THAT(m.license.attribution, Contains("Nums AI"));
	REQUIRE_THAT(m.license.attribution, Contains("hosted"));
	REQUIRE_THAT(m.license.attribution, Contains("contact@nums.world"));

	REQUIRE(m.HasCapability("classify"));
	REQUIRE(m.HasCapability("regress"));
	// the graph normalises in-graph (train-prefix z-score, tail bounds), so the engine must not
	REQUIRE(m.preprocessing_profile == "causilo_v1_raw");
	// the head is 10 classes wide; the engine's ceiling comes from here
	REQUIRE(m.size_regime.max_classes == 10);

	// The graph and tensor map were exported against ONE checkpoint, and the safetensors header
	// is not stable across upstream's later commits (HF main has moved past this one). A moving
	// ref would silently pair a new checkpoint with an old graph, so the revision is a commit.
	for (auto task : {TabFMTask::CLASSIFICATION, TabFMTask::REGRESSION}) {
		auto &t = m.tasks.at(task);
		REQUIRE(t.repo == "nums-ai/causilo");
		REQUIRE(t.revision == "94f2bd91db0737d4da59f347910662905ecb5a09");
		REQUIRE(t.files.size() == 1);
		REQUIRE_THAT(t.files[0].url, Contains("/resolve/94f2bd91db0737d4da59f347910662905ecb5a09/"));
		REQUIRE_THAT(t.files[0].path, Catch::Matchers::EndsWith("model.safetensors"));
	}
	// Both tasks live in one HF repo, so their cache paths must differ (see the 2.5-real test).
	REQUIRE(m.tasks.at(TabFMTask::CLASSIFICATION).files[0].path !=
	        m.tasks.at(TabFMTask::REGRESSION).files[0].path);
	// Exact sizes: the engine treats a size mismatch as "not downloaded".
	REQUIRE(m.tasks.at(TabFMTask::CLASSIFICATION).files[0].bytes == 144385448);
	REQUIRE(m.tasks.at(TabFMTask::REGRESSION).files[0].bytes == 148417316);
}
