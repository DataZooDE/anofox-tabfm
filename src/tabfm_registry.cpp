// tabfm_registry.cpp — the model registry. Single source of truth for the set of
// models: the built-ins compiled into the binary + models registered in SQL
// (CALL tabfm_register_model). Schema: tabfm_registry.hpp.

#include "tabfm_registry.hpp"
#include "tabfm_manifest.hpp" // BuiltinTabFMManifestJson

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <set>

namespace duckdb {
namespace anofox {

// Built-in catalog (pure-SQL surface): the models we ship are baked in as v2
// manifests referencing BUNDLED graphs + tensor-maps (embedded in the binary via
// cmake/embed_resources.cmake), so they are usable by name with no JSON file.
// Weight bytes are still the user's own HF download — nothing licensed is here.
static const char *const BUILTIN_MITRA = R"json({
  "schema_version": 2, "id": "mitra", "display_name": "Mitra (AWS AutoGluon)",
  "family": "icl-transformer",
  "license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": null},
  "preprocessing_profile": "mitra_v1_minimal",
  "weights": {
    "classification": {"repo": "autogluon/mitra-classifier", "revision": "main",
      "files": [{"path": "model.safetensors", "bytes": 302717904,
                 "url": "https://huggingface.co/autogluon/mitra-classifier/resolve/main/model.safetensors"}]},
    "regression": {"repo": "autogluon/mitra-regressor", "revision": "main",
      "files": [{"path": "model.safetensors", "bytes": 302683140,
                 "url": "https://huggingface.co/autogluon/mitra-regressor/resolve/main/model.safetensors"}]}
  },
  "graph": {"classification": "graph_mitra_classification", "regression": "graph_mitra_regression",
    "tensor_map": {"classification": "tensor_map_mitra_classification.json",
                   "regression": "tensor_map_mitra_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"},
                                 "train_size": {"name": "train_size", "dtype": "i64"}, "n_features": {"name": "d", "dtype": "i64"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 100, "max_classes": 10}
})json";

static const char *const BUILTIN_TABPFN = R"json({
  "schema_version": 2, "id": "tabpfn-v2", "display_name": "TabPFN v2 (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": "accept_hf_license",
              "attribution": "TabPFN v2 by Prior Labs GmbH (Prior Labs License v1.1 = Apache-2.0 + attribution)."},
  "preprocessing_profile": "tabpfn_v2_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/TabPFN-v2-clf", "revision": "main",
      "files": [{"path": "classification/model.ckpt",
                 "url": "https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier.ckpt"}]},
    "regression": {"repo": "Prior-Labs/TabPFN-v2-reg", "revision": "main",
      "files": [{"path": "regression/model.ckpt",
                 "url": "https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt"}]}
  },
  "graph": {"classification": "graph_tabpfn_classification", "regression": "graph_tabpfn_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn_classification.json",
                   "regression": "tensor_map_tabpfn_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 500, "max_classes": 10}
})json";

// TabPFN-2.5 relicensed STRICTLY vs v2: `tabpfn-2.5-license-v1.1` forbids any
// commercial or production use of the weights (and of their outputs), so unlike
// `tabpfn-v2` this entry is commercial:false + gated. Only the `_default`
// checkpoint variant is shipped. Classifier and regressor are different depths
// (24 vs 18 layers) with different encoders, hence per-task graphs AND per-task
// tensor maps.
static const char *const BUILTIN_TABPFN25 = R"json({
  "schema_version": 2, "id": "tabpfn-v2-5", "display_name": "TabPFN 2.5 (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "tabpfn-2.5-license-v1.1", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "TabPFN-2.5 by Prior Labs GmbH. Weights are research/internal-evaluation only — NO commercial or production use. Contact sales@priorlabs.ai for a commercial license."},
  "preprocessing_profile": "tabpfn_v2_5_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/tabpfn_2_5", "revision": "main",
      "files": [{"path": "classification/model.ckpt", "bytes": 42935499,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/tabpfn-v2.5-classifier-v2.5_default.ckpt"}]},
    "regression": {"repo": "Prior-Labs/tabpfn_2_5", "revision": "main",
      "files": [{"path": "regression/model.ckpt", "bytes": 40831995,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/tabpfn-v2.5-regressor-v2.5_default.ckpt"}]}
  },
  "graph": {"classification": "graph_tabpfn25_classification", "regression": "graph_tabpfn25_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn25_classification.json",
                   "regression": "tensor_map_tabpfn25_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 500, "max_classes": 10}
})json";

// TabPFN-3 (released 2026-05-12). Same licensing posture as 2.5 —
// `tabpfn-3-license-v1.0` allows testing/evaluation/internal benchmarking only,
// so commercial:false + gated. Architecturally it is NOT 2.5 scaled up: a
// distribution-embedding stack with inducing points feeds a feature-aggregation
// stack, and RoPE replaces the pre-generated column-embedding table. It exports
// through the same pipeline nonetheless (tools/export_tabpfn, `--arch v3`);
// parity 2.4e-07 classification / 2.4e-06 regression.
//
// Like the other .ckpt models, the downloadable artifact needs the one-time
// `convert_weights.py` pass; the engine then prefers the `model.safetensors` it
// writes next to the checkpoint.
static const char *const BUILTIN_TABPFN3 = R"json({
  "schema_version": 2, "id": "tabpfn-v3", "display_name": "TabPFN 3 (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "tabpfn-3-license-v1.0", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "TabPFN-3 by Prior Labs GmbH. Weights are testing/evaluation/internal-benchmarking only — NO commercial or production use. Contact sales@priorlabs.ai for a commercial license."},
  "preprocessing_profile": "tabpfn_v3_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/tabpfn_3", "revision": "main",
      "files": [{"path": "classification/model.ckpt", "bytes": 212804803,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-classifier-v3_default.ckpt"}]},
    "regression": {"repo": "Prior-Labs/tabpfn_3", "revision": "main",
      "files": [{"path": "regression/model.ckpt",
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-regressor-v3_default.ckpt"}]}
  },
  "graph": {"classification": "graph_tabpfn3_classification", "regression": "graph_tabpfn3_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn3_classification.json",
                   "regression": "tensor_map_tabpfn3_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 500, "max_classes": 10}
})json";

// RealTabPFN-2.5: the SAME 2.5 architecture, continued pre-training on real
// tabular data (TabArena v0.1.4 ranks it above TabICLv2). It ships as its own
// catalog entry rather than a flag on `tabpfn-v2-5` because the two score
// differently and users pick between them by name.
//
// Everything graph-shaped is REUSED from 2.5 — same bundled graph, same tensor
// map, same ext-graph header sha — because the two checkpoints have identical
// `config` blocks and identical state_dict signatures (154 clf / 121 reg
// tensors: same names, shapes, dtypes; only the values differ). A safetensors
// JSON header is a function of names/shapes/dtypes, so converting either
// checkpoint yields a byte-identical header. Verified against the real weights;
// test_tabfm_model_spec.cpp pins it so a divergent future checkpoint fails loud.
//
// What is NOT shared is the cache path. `CacheSlug` keys on the HF REPO, and
// both checkpoints live in Prior-Labs/tabpfn_2_5 — so identical files[].path
// values would make the two entries download over each other and silently serve
// the wrong weights. Hence the `-real` task directories, which
// tools/export_tabpfn/convert_weights.py --variant=real writes to.
static const char *const BUILTIN_TABPFN25_REAL = R"json({
  "schema_version": 2, "id": "tabpfn-v2-5-real", "display_name": "RealTabPFN 2.5 (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "tabpfn-2.5-license-v1.1", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "RealTabPFN-2.5 by Prior Labs GmbH (TabPFN-2.5 continued pre-training on real data). Weights are research/internal-evaluation only — NO commercial or production use. Contact sales@priorlabs.ai for a commercial license."},
  "preprocessing_profile": "tabpfn_v2_5_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/tabpfn_2_5", "revision": "main",
      "files": [{"path": "classification-real/model.ckpt", "bytes": 42929707,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/tabpfn-v2.5-classifier-v2.5_real.ckpt"}]},
    "regression": {"repo": "Prior-Labs/tabpfn_2_5", "revision": "main",
      "files": [{"path": "regression-real/model.ckpt", "bytes": 40831868,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/tabpfn-v2.5-regressor-v2.5_real.ckpt"}]}
  },
  "graph": {"classification": "graph_tabpfn25_classification", "regression": "graph_tabpfn25_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn25_classification.json",
                   "regression": "tensor_map_tabpfn25_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 500, "max_classes": 10}
})json";

// TabPFN-2.6 (Prior Labs). A 2.5-LINE architecture — same emsize/nhead/
// features_per_group/thinking-row layout and the same pre-generated
// column-embedding table — so it exports through the 2.5 patch surface with NO
// new patches (tools/export_tabpfn --config real26; parity 8.9e-08
// classification / 1.4e-06 regression). What changed is layernorm_type="rmsnorm"
// and a deeper per-layer parameterisation: 322 clf / 324 reg mapped
// initializers against 2.5's 250 / 192, which is why it needs its own graphs.
//
// Unlike 2.5 (24-layer classifier, 18-layer regressor) both 2.6 heads are 24
// layers and differ only in encoder_type, but they still get per-task graphs and
// per-task tensor maps like every other model here.
//
// Licensing matches the rest of the post-v2 Prior Labs line: tabpfn-2.6-license
// -v1.0 permits testing/evaluation/internal benchmarking only, and the HF repo
// is gated — so commercial:false + accept_hf_license.
//
// The size regime is genuinely wider than 2.5's: Prior Labs document <=50k
// samples and <=2000 features (2.5: 10k / 500).
static const char *const BUILTIN_TABPFN26 = R"json({
  "schema_version": 2, "id": "tabpfn-v2-6", "display_name": "TabPFN 2.6 (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "tabpfn-2.6-license-v1.0", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "TabPFN-2.6 by Prior Labs GmbH. Weights are testing/evaluation/internal-benchmarking only — NO commercial or production use. Contact sales@priorlabs.ai for a commercial license."},
  "preprocessing_profile": "tabpfn_v2_6_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/tabpfn_2_6", "revision": "main",
      "files": [{"path": "classification/model.ckpt", "bytes": 43044699,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_2_6/resolve/main/tabpfn-v2.6-classifier-v2.6_default.ckpt"}]},
    "regression": {"repo": "Prior-Labs/tabpfn_2_6", "revision": "main",
      "files": [{"path": "regression/model.ckpt", "bytes": 51955328,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_2_6/resolve/main/tabpfn-v2.6-regressor-v2.6_default.ckpt"}]}
  },
  "graph": {"classification": "graph_tabpfn26_classification", "regression": "graph_tabpfn26_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn26_classification.json",
                   "regression": "tensor_map_tabpfn26_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 50000, "max_features": 2000, "max_classes": 10}
})json";

static const char *const BUILTIN_TABICL = R"json({
  "schema_version": 2, "id": "tabicl-v2", "display_name": "TabICL v2 (soda-inria)",
  "family": "icl-transformer",
  "license": {"id": "bsd-3-clause", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "TabICL (soda-inria), BSD-3-Clause. Checkpoints: HF jingang/TabICL."},
  "preprocessing_profile": "tabicl_v2_raw",
  "weights": {
    "classification": {"repo": "jingang/TabICL", "revision": "main",
      "files": [{"path": "classification/model.ckpt",
                 "url": "https://huggingface.co/jingang/TabICL/resolve/main/tabicl-classifier-v2-20260212.ckpt"}]},
    "regression": {"repo": "jingang/TabICL", "revision": "main",
      "files": [{"path": "regression/model.ckpt",
                 "url": "https://huggingface.co/jingang/TabICL/resolve/main/tabicl-regressor-v2-20260212.ckpt"}]}
  },
  "graph": {"classification": "graph_tabicl_classification", "regression": "graph_tabicl_regression",
    "tensor_map": {"classification": "tensor_map_tabicl_classification.json",
                   "regression": "tensor_map_tabicl_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 100000, "max_features": 512, "max_classes": 10}
})json";

// Orion-BiX is the first built-in with a SINGLE capability: upstream ships a
// classifier only (orion_bix/sklearn/classifier.py, no regressor), so
// `tabfm_regress(..., model := 'orion-bix')` raises the unsupported-task error.
// Unlike TabPFN/TabICL it does NOT normalize internally — its sklearn wrapper
// standardizes externally — hence a "_minimal" (engine-standardizes) profile
// rather than a "_raw" one.
// TabDPT (Layer 6 AI), 1459 Elo on the TabArena v0.1.4 board and the catalog's
// second permissively-licensed model with BOTH tasks (Apache-2.0, ungated).
//
// docs/MULTI_MODEL_PLAN.md deferred TabDPT behind a hypothetical
// `RetrievalOnnxBackend`. That premise was wrong: retrieval is not part of the
// model. TabDPTEstimator defaults to context_reduction="subsample" and only
// reaches for FAISS on request; either way the reduction lives in the sklearn
// wrapper and merely picks WHICH context rows to hand over. TabDPTModel.forward
// takes the whole context and derives the split from the label length
// (`eval_pos = y_src.shape[0]`) — precisely the engine's single_eval_pos family.
// So it needed no engine change at all, only an exporter (tools/export_tabdpt).
//
// It is also the first built-in with NO ckpt->safetensors conversion step:
// Layer 6 publish safetensors whose keys are already the model's state_dict
// namespace, so the downloaded file is injected as-is against the committed
// tensor map.
//
// BOTH TASKS SHARE ONE WEIGHTS FILE. TabDPT has a single head whose output is
// class logits followed by regression bins, so the two graphs map the same 647
// initializers and both tasks declare the SAME files[].path — one 254 MB
// download serves both, and ExpectedWeightsHeaderShaFor returns the same sha
// for both tasks because it is literally the same file. (Contrast tabpfn-v2-5
// vs -real, where a shared path would have been a BUG; here it is the point.)
//
// max_features is a hard 128: the export wrapper pads x up to the model's fixed
// num_features, so a wider table would make that pad negative.
static const char *const BUILTIN_TABDPT = R"json({
  "schema_version": 2, "id": "tabdpt", "display_name": "TabDPT v1.2 (Layer 6 AI)",
  "family": "icl-transformer",
  "license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "TabDPT by Layer 6 AI, Apache-2.0. Checkpoint: HF Layer6/TabDPT (tabdpt1_2.safetensors)."},
  "preprocessing_profile": "tabdpt_v1_raw",
  "weights": {
    "classification": {"repo": "Layer6/TabDPT", "revision": "main",
      "files": [{"path": "model.safetensors", "bytes": 254098072,
                 "url": "https://huggingface.co/Layer6/TabDPT/resolve/main/tabdpt1_2.safetensors"}]},
    "regression": {"repo": "Layer6/TabDPT", "revision": "main",
      "files": [{"path": "model.safetensors", "bytes": 254098072,
                 "url": "https://huggingface.co/Layer6/TabDPT/resolve/main/tabdpt1_2.safetensors"}]}
  },
  "graph": {"classification": "graph_tabdpt_classification", "regression": "graph_tabdpt_regression",
    "tensor_map": {"classification": "tensor_map_tabdpt_classification.json",
                   "regression": "tensor_map_tabdpt_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 100000, "max_features": 128, "max_classes": 10}
})json";

static const char *const BUILTIN_ORION_BIX = R"json({
  "schema_version": 2, "id": "orion-bix", "display_name": "Orion-BiX v1.1 (Lexsi Labs)",
  "family": "icl-transformer",
  "license": {"id": "mit", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "Orion-BiX by Lexsi Labs, MIT. Checkpoint: HF Lexsi/Orion-BiX."},
  "preprocessing_profile": "orion_bix_v1_minimal",
  "weights": {
    "classification": {"repo": "Lexsi/Orion-BiX", "revision": "main",
      "files": [{"path": "classification/model.ckpt", "bytes": 314653437,
                 "url": "https://huggingface.co/Lexsi/Orion-BiX/resolve/main/Orion-BiX-v1.1.ckpt"}]}
  },
  "graph": {"classification": "graph_orion_bix_classification",
    "tensor_map": {"classification": "tensor_map_orion_bix_classification.json"}},
  "capabilities": ["classify"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 100000, "max_features": 512, "max_classes": 10}
})json";

vector<ModelSpec> BuiltinModelSpecs() {
	// Merge the two per-task built-in TabFM v1 manifests into one multi-task
	// model spec (id "tabfm-v1"). Each task keeps its own graph/tensor-map/repo.
	auto spec = ParseModelSpec(BuiltinTabFMManifestJson(TabFMTask::CLASSIFICATION),
	                           "(built-in tabfm-v1 classification)");
	auto reg_spec =
	    ParseModelSpec(BuiltinTabFMManifestJson(TabFMTask::REGRESSION), "(built-in tabfm-v1 regression)");
	spec.tasks.emplace(TabFMTask::REGRESSION, reg_spec.tasks.at(TabFMTask::REGRESSION));
	spec.capabilities = {"classify", "regress"};
	spec.display_name = "Google TabFM v1";
	spec.family = "icl-transformer";
	// Non-commercial weights → the license gate fires (unchanged behavior).
	spec.license.commercial = false;
	spec.license.redistributable = false;
	spec.license.gate_setting = "accept_hf_license";

	vector<ModelSpec> specs;
	specs.push_back(std::move(spec));
	// The commercial-clean catalog, baked in (bundled graphs/tensor-maps).
	specs.push_back(ParseModelSpec(BUILTIN_MITRA, "(built-in mitra)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN, "(built-in tabpfn-v2)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN25, "(built-in tabpfn-v2-5)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN25_REAL, "(built-in tabpfn-v2-5-real)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN26, "(built-in tabpfn-v2-6)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN3, "(built-in tabpfn-v3)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABICL, "(built-in tabicl-v2)"));
	specs.push_back(ParseModelSpec(BUILTIN_ORION_BIX, "(built-in orion-bix)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABDPT, "(built-in tabdpt)"));
	return specs;
}

string ModelRegistry::RegisteredIds() const {
	string out; // models_ is a std::map → keys already sorted
	for (auto &kv : models_) {
		if (!out.empty()) {
			out += ", ";
		}
		out += kv.first;
	}
	return out;
}

const ModelSpec &ModelRegistry::Get(const string &id) const {
	auto it = models_.find(id);
	if (it == models_.end()) {
		throw InvalidInputException("tabfm: unknown model '%s'. Registered: %s. See tabfm_list_models().", id,
		                            RegisteredIds());
	}
	return it->second;
}

const ModelSpec &ModelRegistry::Resolve(const string &requested, const string &default_setting) const {
	if (!requested.empty()) {
		return Get(requested);
	}
	if (!default_setting.empty()) {
		return Get(default_setting);
	}
	if (!implicit_default_.empty()) {
		return Get(implicit_default_);
	}
	if (models_.size() == 1) {
		return models_.begin()->second;
	}
	throw InvalidInputException(
	    "tabfm: %llu models are registered (%s) and none is selected. Choose one with "
	    "model := '<id>' or SET anofox_tabfm_default_model = '<id>'. See tabfm_list_models().",
	    static_cast<unsigned long long>(models_.size()), RegisteredIds());
}

ModelRegistry ModelRegistry::Build(const vector<ModelSpec> &registered) {
	ModelRegistry registry;
	for (auto &spec : BuiltinModelSpecs()) {
		registry.models_[spec.id] = std::move(spec);
	}
	// SQL-registered models (CALL tabfm_register_model) merge last and shadow a
	// built-in of the same id. A single SQL registration also becomes the implicit
	// default (a bare call resolves to it); with several, selection is explicit.
	for (auto &spec : registered) {
		registry.models_[spec.id] = spec;
	}
	if (registered.size() == 1) {
		registry.implicit_default_ = registered.front().id;
	}
	return registry;
}

} // namespace anofox
} // namespace duckdb
