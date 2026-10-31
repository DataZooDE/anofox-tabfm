// Catch2 tests for the model registry: the built-in catalog plus models
// registered in SQL (CALL tabfm_register_model → ModelRegistry::Build(specs)).

#include "catch.hpp"

#include "tabfm_registry.hpp"
#include "tabfm_model_spec.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#include <set>

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

TEST_CASE("registry: Mitra-v2 is a new entry beside mitra, its classifier reusing v1's graph",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("mitra"));
	REQUIRE(reg.Has("mitra-v2"));
	auto &v1 = reg.Get("mitra");
	auto &v2 = reg.Get("mitra-v2");

	// Apache-2.0 and ungated, like v1 (autogluon/mitra-{classifier,regressor}-2 model cards).
	REQUIRE(v2.license.id == "apache-2.0");
	REQUIRE(v2.license.commercial == true);
	REQUIRE(v2.license.gate_setting.empty());
	REQUIRE(v2.HasCapability("classify"));
	REQUIRE(v2.HasCapability("regress"));

	auto &c1 = v1.tasks.at(TabFMTask::CLASSIFICATION);
	auto &c2 = v2.tasks.at(TabFMTask::CLASSIFICATION);
	auto &r2 = v2.tasks.at(TabFMTask::REGRESSION);
	// Separate HF repos, so separate cache slugs: v1 and v2 can never overwrite each other.
	REQUIRE(c2.repo == "autogluon/mitra-classifier-2");
	REQUIRE(r2.repo == "autogluon/mitra-regressor-2");
	REQUIRE(c2.repo != c1.repo);
	REQUIRE(c2.files[0].bytes == 302717904);
	REQUIRE(r2.files[0].bytes == 306776952);

	// The v2 classifier checkpoint has v1's exact safetensors header (same names, shapes, dtypes,
	// offsets), so its graph and tensor map are v1's verbatim -- no new embedded bytes.
	REQUIRE(c2.graph == c1.graph);
	REQUIRE(c2.tensor_map_path == c1.tensor_map_path);
	// The v2 REGRESSOR is not drop-in: a 1000-bin distributional head (final_layer [1000, 512]) and a
	// bin-id label embedding (y_embedding [1000, 512], no bias) replace v1's scalar head, so it needs its
	// own graph.
	REQUIRE(r2.graph == "graph_mitra2_regression");
	REQUIRE(r2.graph != v1.tasks.at(TabFMTask::REGRESSION).graph);

	// The regression graph min-max scales the target and bins it itself (raw in, raw out), so the
	// profile is a *_raw one. Mitra's feature embedding is rank based, so raw features are exactly
	// what upstream feeds it.
	REQUIRE(StringUtil::EndsWith(v2.preprocessing_profile, "_raw"));

	// Wider than v1 (10000 / 100): Mitra-v2 predicts with up to 32,768 support rows and a 256-feature
	// budget (Mitra-v2 Technical Report, arXiv 2609.04540, sec. 3.1 / 3.5).
	REQUIRE(v2.size_regime.max_rows == 32768);
	REQUIRE(v2.size_regime.max_features == 256);
	REQUIRE(v2.size_regime.max_classes == 10);
}

TEST_CASE("registry: TabDPT v1.3 is a new entry in tabdpt's repo that cannot clobber v1.2's cache",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("tabdpt"));
	REQUIRE(reg.Has("tabdpt-v1-3"));
	auto &v12 = reg.Get("tabdpt");
	auto &v13 = reg.Get("tabdpt-v1-3");

	REQUIRE(v13.license.id == "apache-2.0");
	REQUIRE(v13.license.commercial == true);
	REQUIRE(v13.license.gate_setting.empty());
	REQUIRE(v13.HasCapability("classify"));
	REQUIRE(v13.HasCapability("regress"));
	REQUIRE(v13.preprocessing_profile == v12.preprocessing_profile);
	REQUIRE(v13.size_regime.max_features == 128);
	REQUIRE(v13.size_regime.max_classes == 16);

	auto &clf = v13.tasks.at(TabFMTask::CLASSIFICATION);
	auto &rgr = v13.tasks.at(TabFMTask::REGRESSION);
	// One checkpoint serves both tasks, as in v1.2.
	REQUIRE(clf.files[0].path == rgr.files[0].path);
	REQUIRE(clf.files[0].url == rgr.files[0].url);
	REQUIRE_THAT(clf.files[0].url, Catch::Matchers::EndsWith("/tabdpt1_3.safetensors"));
	REQUIRE(clf.files[0].bytes == 252233296);
	// Same HF repo as v1.2, so the cache slug is shared: the path must differ or the two entries
	// would download over each other (the tabpfn-v2-5-real lesson).
	REQUIRE(clf.repo == v12.tasks.at(TabFMTask::CLASSIFICATION).repo);
	REQUIRE(clf.files[0].path != v12.tasks.at(TabFMTask::CLASSIFICATION).files[0].path);
	// New y-encoders (per-layer class embeddings + a narrower regression MLP), so new graphs.
	REQUIRE(clf.graph == "graph_tabdpt13_classification");
	REQUIRE(rgr.graph == "graph_tabdpt13_regression");
}

TEST_CASE("registry: Nori-30M ships built in, Apache-2.0 and ungated, REGRESSION ONLY, read natively from nori.pt",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("nori-30m"));
	auto &m = reg.Get("nori-30m");

	// Code and weights are Apache-2.0 (model card front matter and body agree) and the HF repo is
	// ungated, so there is no licence gate at all, as for tabdpt.
	REQUIRE(m.license.id == "apache-2.0");
	REQUIRE(m.license.commercial == true);
	REQUIRE(m.license.redistributable == true);
	REQUIRE(m.license.gate_setting.empty());
	REQUIRE_THAT(m.license.attribution, Contains("Synthefy"));

	// Regression only: upstream's build_model refuses any task_type but "reg"
	REQUIRE(m.HasCapability("regress"));
	REQUIRE_FALSE(m.HasCapability("classify"));
	REQUIRE(m.HasTask(TabFMTask::REGRESSION));
	REQUIRE_FALSE(m.HasTask(TabFMTask::CLASSIFICATION));
	REQUIRE(m.preprocessing_profile == "nori_v1_raw");
	// measured caps (registry comment + docs/REAL_MODELS.md): 5000x20 peaks at 15.4 GB on the CPU
	REQUIRE(m.size_regime.max_rows == 5000);
	REQUIRE(m.size_regime.max_features == 100);

	// the released nori.pt itself, pinned to the commit the graph was exported from; no converter
	auto &t = m.tasks.at(TabFMTask::REGRESSION);
	REQUIRE(t.repo == "Synthefy/Nori-30M");
	REQUIRE(t.revision == "63c9f7facf9fb32c37ce3fc2fba331d524696318");
	REQUIRE(t.files.size() == 1);
	REQUIRE_THAT(t.files[0].url, Contains("/resolve/63c9f7facf9fb32c37ce3fc2fba331d524696318/nori.pt"));
	REQUIRE(t.files[0].bytes == 117093952);
	REQUIRE_THAT(t.files[0].path, Catch::Matchers::EndsWith(".ckpt"));

	// the unsupported-task error names the models that CAN classify, and nori-30m is not among them
	auto classify = reg.IdsWithTask(TabFMTask::CLASSIFICATION);
	REQUIRE_THAT(classify, Contains("mitra"));
	REQUIRE(classify.find("nori-30m") == string::npos);
	REQUIRE_THAT(reg.IdsWithTask(TabFMTask::REGRESSION), Contains("nori-30m"));
}

TEST_CASE("registry: EXAONE Tabular ships built in, non-commercial (EXAONE NC 1.2) and gated, one checkpoint per task",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	REQUIRE(reg.Has("exaone-tabular"));
	auto &m = reg.Get("exaone-tabular");

	// EXAONE AI Model License Agreement 1.2 - NC: research/educational use only, no commercial use of the
	// Model, Derivatives or Output. Gated like the other non-commercial entries, and the attribution says so.
	REQUIRE(m.license.id == "exaone-nc-1.2");
	REQUIRE(m.license.commercial == false);
	REQUIRE(m.license.redistributable == false);
	REQUIRE(m.license.gate_setting == "accept_hf_license");
	REQUIRE_THAT(m.license.attribution, Contains("LG AI Research"));
	REQUIRE_THAT(m.license.attribution, Contains("NO commercial use"));
	REQUIRE_THAT(m.license.attribution, Contains("LG-AI-Research/EXAONE-Tabular/blob/main/LICENSE"));

	REQUIRE(m.HasCapability("classify"));
	REQUIRE(m.HasCapability("regress"));
	REQUIRE(m.preprocessing_profile == "exaone_v1_raw");
	REQUIRE(m.size_regime.max_classes == 10);
	// measured caps, not upstream's 100000-row support limit (its chunked CUDA runtime): the engine runs the
	// dense forward, and 3000x20 already peaks at 13.1 GB on the CPU
	REQUIRE(m.size_regime.max_rows == 3000);
	REQUIRE(m.size_regime.max_features == 100);

	for (auto task : {TabFMTask::CLASSIFICATION, TabFMTask::REGRESSION}) {
		auto &t = m.tasks.at(task);
		REQUIRE(t.repo == "LG-AI-Research/EXAONE-Tabular");
		REQUIRE(t.revision == "093ad1c2613e1e95d26df8610451d2e02ed53cbe");
		REQUIRE(t.files.size() == 1);
		REQUIRE_THAT(t.files[0].url, Contains("/resolve/093ad1c2613e1e95d26df8610451d2e02ed53cbe/"));
	}
	// the classifier and the regressor are DIFFERENT checkpoints in one repo: their cache paths must differ
	REQUIRE_THAT(m.tasks.at(TabFMTask::CLASSIFICATION).files[0].url, Contains("classifier-v1_default"));
	REQUIRE_THAT(m.tasks.at(TabFMTask::REGRESSION).files[0].url, Contains("regressor-v1_default"));
	REQUIRE(m.tasks.at(TabFMTask::CLASSIFICATION).files[0].path !=
	        m.tasks.at(TabFMTask::REGRESSION).files[0].path);
}

TEST_CASE("registry: Kumo-Tabular ships built in at three sizes, commercial and ungated under OpenMDW-1.1, pinned to v1.0.1",
          "[tabfm][registry]") {
	auto reg = ModelRegistry::Build();
	struct Size {
		const char *id;
		const char *dir;
		int64_t clf_bytes;
		int64_t reg_bytes;
	};
	// Exact LFS sizes at the pinned revision (identical on main): the engine treats a size mismatch as
	// "not downloaded".
	const Size sizes[] = {{"kumo-tabular-small", "small", 109936978, 113968523},
	                      {"kumo-tabular-medium", "medium", 246117940, 250144651},
	                      {"kumo-tabular-large", "large", 854850034, 862909259}};
	for (auto &size : sizes) {
		INFO(size.id);
		REQUIRE(reg.Has(size.id));
		auto &m = reg.Get(size.id);
		// OpenMDW-1.1 grants use "without restriction" (commercial included) and the HF repo is not gated, so
		// there is no licence acknowledgement to require. Its one distribution duty -- keep a copy of the
		// agreement and NVIDIA's copyright notices -- must be quoted where a user sees the model.
		REQUIRE(m.license.id == "openmdw-1.1");
		REQUIRE(m.license.commercial == true);
		REQUIRE(m.license.redistributable == true);
		REQUIRE(m.license.gate_setting.empty());
		REQUIRE_THAT(m.license.attribution, Contains("Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES"));
		REQUIRE_THAT(m.license.attribution, Contains("retain a copy of the OpenMDW-1.1 agreement"));
		REQUIRE_THAT(m.license.attribution, Contains("https://huggingface.co/nvidia/Kumo-Tabular/blob/main/LICENSE"));

		REQUIRE(m.HasCapability("classify"));
		REQUIRE(m.HasCapability("regress"));
		// Kumo's single-estimator recipe IS the engine's standardising profile, so not *_raw
		REQUIRE(m.preprocessing_profile == "kumo_tabular_v1_minimal");
		REQUIRE(m.size_regime.max_classes == 10);
		REQUIRE(m.size_regime.max_features == 500);
		// measured cap: the masked graph attends over all rows; 10000 rows needed ~30 GB on the SMALL model
		REQUIRE(m.size_regime.max_rows == 5000);

		for (auto task : {TabFMTask::CLASSIFICATION, TabFMTask::REGRESSION}) {
			auto &t = m.tasks.at(task);
			REQUIRE(t.repo == "nvidia/Kumo-Tabular");
			// the tag upstream's own loader pins (KumoTabular._load_from_pretrained, revision="v1.0.1")
			REQUIRE(t.revision == "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f");
			REQUIRE(t.files.size() == 1);
			const string pt = task == TabFMTask::CLASSIFICATION ? "classifier.pt" : "regressor.pt";
			REQUIRE_THAT(t.files[0].url, Contains("/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/" +
			                                      string(size.dir) + "/" + pt));
			// declared as .ckpt so the converted sibling model.safetensors is preferred (GPU graphs)
			REQUIRE(t.files[0].path == string(size.dir) + "/" + TabFMTaskName(task) + "/model.ckpt");
			REQUIRE(t.files[0].bytes == (task == TabFMTask::CLASSIFICATION ? size.clf_bytes : size.reg_bytes));
		}
	}
	// All six checkpoints live in ONE HF repo, so every (size, task) needs its own cache path.
	std::set<string> paths;
	for (auto &size : sizes) {
		for (auto task : {TabFMTask::CLASSIFICATION, TabFMTask::REGRESSION}) {
			paths.insert(reg.Get(size.id).tasks.at(task).files[0].path);
		}
	}
	REQUIRE(paths.size() == 6);
}
