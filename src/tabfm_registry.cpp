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

// TabPFN-3.5 and TabPFN-3.5-Fast (Prior Labs, released 2026-09-09; HF
// Prior-Labs/tabpfn_3_5). A new architecture, not v3 retrained: every cell gets
// a midrank-ECDF channel against the train rows of its column, a Fourier cell
// embedding, q/k RMSNorm in every attention, and multi-task heads. Upstream's
// ECDF uses torch.searchsorted (no ONNX operator), so the graphs come from a
// re-driven forward with a fixed-step binary search in its place
// (tools/export_tabpfn35, tabpfn 9.2.0); parity with upstream's own forward on
// the released weights 1.7e-05 clf / 3.8e-06 reg (3.5), 1.5e-04 / 6.2e-06 (Fast).
//
// Licensing: tabpfn-3-5-license-v1.0 is word for word tabpfn-3-license-v1.0
// (only the model name and revision date differ) -- testing / evaluation /
// research only, no commercial or production use, no hosted/API/SaaS service.
// Hence commercial:false + accept_hf_license, exactly like tabpfn-v3.
//
// Weights: ONE published safetensors per variant holds BOTH heads (a 160-class
// many-class decoder and a 5000-bucket regression head) under the model's own
// state_dict names, so -- like tabdpt -- both tasks read the same downloaded file
// and there is no conversion step. The two variants live in the SAME repo, and
// CacheSlug keys on the repo, so each gets its own directory (v3.5/, v3.5-fast/):
// identical paths would download over each other and serve the wrong weights.
//
// The two variants differ only in depth (24 vs 8 ICL layers), which is still a
// different graph, so they get separate stems (tabpfn35 / tabpfn35fast).
// Not shipped: the experimental `_multiclass` checkpoint (a third, different
// architecture variant) and the API-only "Thinking" model.
static const char *const BUILTIN_TABPFN35 = R"json({
  "schema_version": 2, "id": "tabpfn-v3-5", "display_name": "TabPFN 3.5 (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "tabpfn-3-5-license-v1.0", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "TabPFN-3.5 by Prior Labs GmbH. Weights are testing/evaluation/research only — NO commercial or production use, and no hosted/API/SaaS service. Contact sales@priorlabs.ai for a commercial license."},
  "preprocessing_profile": "tabpfn_v3_5_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/tabpfn_3_5", "revision": "main",
      "files": [{"path": "v3.5/model.safetensors", "bytes": 876027932,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-20260909.safetensors"}]},
    "regression": {"repo": "Prior-Labs/tabpfn_3_5", "revision": "main",
      "files": [{"path": "v3.5/model.safetensors", "bytes": 876027932,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-20260909.safetensors"}]}
  },
  "graph": {"classification": "graph_tabpfn35_classification", "regression": "graph_tabpfn35_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn35_classification.json",
                   "regression": "tensor_map_tabpfn35_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 500, "max_classes": 10}
})json";

static const char *const BUILTIN_TABPFN35_FAST = R"json({
  "schema_version": 2, "id": "tabpfn-v3-5-fast", "display_name": "TabPFN 3.5 Fast (Prior Labs)",
  "family": "icl-transformer",
  "license": {"id": "tabpfn-3-5-license-v1.0", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "TabPFN-3.5-Fast by Prior Labs GmbH. Weights are testing/evaluation/research only — NO commercial or production use, and no hosted/API/SaaS service. Contact sales@priorlabs.ai for a commercial license."},
  "preprocessing_profile": "tabpfn_v3_5_raw",
  "weights": {
    "classification": {"repo": "Prior-Labs/tabpfn_3_5", "revision": "main",
      "files": [{"path": "v3.5-fast/model.safetensors", "bytes": 334181708,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-fast-20260909.safetensors"}]},
    "regression": {"repo": "Prior-Labs/tabpfn_3_5", "revision": "main",
      "files": [{"path": "v3.5-fast/model.safetensors", "bytes": 334181708,
                 "url": "https://huggingface.co/Prior-Labs/tabpfn_3_5/resolve/main/tabpfn-v3.5-fast-20260909.safetensors"}]}
  },
  "graph": {"classification": "graph_tabpfn35fast_classification", "regression": "graph_tabpfn35fast_regression",
    "tensor_map": {"classification": "tensor_map_tabpfn35fast_classification.json",
                   "regression": "tensor_map_tabpfn35fast_regression.json"}},
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
//
// max_classes is 16, the checkpoint's actual head width (cfg.model
// max_num_classes), NOT the catalog's usual 10. The engine reads this ceiling
// straight off the registry entry (`resolved.max_classes` in tabfm_engine.cpp),
// so declaring 10 here would refuse an 11-16 class task that the graph is
// perfectly able to serve.
// Orion-MSP (Lexsi Labs, MIT) — the sibling of orion-bix, and like it
// CLASSIFICATION ONLY (upstream ships sklearn/classifier.py and no regressor).
//
// Two things about the RELEASED checkpoint are narrower than the paper, and
// tools/export_orion_msp asserts both so a future release cannot silently change
// the traced architecture: `row_scales = (1,)` (the "multi-scale" row stage
// ships with a SINGLE scale) and `perc_num_latents = 0` (Perceiver memory off).
//
// `row_num_random = 2` IS live, and it is the interesting part. Upstream redraws
// the BigBird random attention links with `torch.randperm` on EVERY forward, so
// upstream inference is not deterministic and the mask cannot be traced. The
// exporter freezes one seeded draw into a table and slices it — deterministic
// across runs and machines, structurally faithful (each non-special query keeps
// exactly `num_random` extra links), but reproducing ONE particular upstream
// draw rather than any specific one. See docs/REAL_MODELS.md.
static const char *const BUILTIN_ORION_MSP = R"json({
  "schema_version": 2, "id": "orion-msp", "display_name": "Orion-MSP v1.5 (Lexsi Labs)",
  "family": "icl-transformer",
  "license": {"id": "mit", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "Orion-MSP by Lexsi Labs, MIT. Checkpoint: HF Lexsi/Orion-MSP."},
  "preprocessing_profile": "orion_msp_v1_minimal",
  "weights": {
    "classification": {"repo": "Lexsi/Orion-MSP", "revision": "main",
      "files": [{"path": "classification/model.ckpt", "bytes": 335545028,
                 "url": "https://huggingface.co/Lexsi/Orion-MSP/resolve/main/OrionMSP-classifier-v1.5-202603.ckpt"}]}
  },
  "graph": {"classification": "graph_orion_msp_classification",
    "tensor_map": {"classification": "tensor_map_orion_msp_classification.json"}},
  "capabilities": ["classify"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 100000, "max_features": 512, "max_classes": 10}
})json";

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
  "size_regime": {"max_rows": 100000, "max_features": 128, "max_classes": 16}
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

// Causilo (Nums AI Inc.; TabArena Elo 1785), a TabICL-style column-then-row model with an extra
// row-refinement stage: 36.1M (classifier) / 37.1M (regressor) parameters, plain torch, native
// 10-class head. Code is Apache-2.0; the WEIGHTS are under the Causilo License v1.0 --
// non-commercial research/evaluation, and "Commercial or production use, and hosted/API/SaaS
// services whether paid or free, require separate licenses" -- so it is registered like the other
// non-commercial entries (commercial:false + accept_hf_license) with that clause in the
// attribution. The HF repo itself is ungated; no token is needed.
//
// Its graph is (x, y)-only like TabICL's: the split is the length of y, so the engine needs no
// model-specific code (tools/export_causilo). The graph normalises its own features, hence a
// *_raw profile. Weights are already safetensors with the keys the tensor map names, so there is
// no converter, and they are pinned to the commit the graph was exported from: HF main has moved
// past it, and a moving ref would pair a newer checkpoint with this graph.
//
// What this does NOT reproduce, so the docs do not over-claim: the engine runs ONE estimator, while
// the leaderboard figure is an 8-member ensemble (n_estimators=1 is upstream's own single-member
// path), and the engine mean-imputes missing cells before the graph, so Causilo's missing-value
// embedding is unused.
static const char *const BUILTIN_CAUSILO = R"json({
  "schema_version": 2, "id": "causilo", "display_name": "Causilo (Nums AI)",
  "family": "icl-transformer",
  "license": {"id": "causilo-license-v1.0", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "Causilo by Nums AI Inc. Code Apache-2.0; weights Causilo License v1.0 — non-commercial research/evaluation only. Commercial or production use, and hosted/API/SaaS services whether paid or free, require a separate license: contact@nums.world. Checkpoints: HF nums-ai/causilo."},
  "preprocessing_profile": "causilo_v1_raw",
  "weights": {
    "classification": {"repo": "nums-ai/causilo", "revision": "94f2bd91db0737d4da59f347910662905ecb5a09",
      "files": [{"path": "classification/model.safetensors", "bytes": 144385448,
                 "url": "https://huggingface.co/nums-ai/causilo/resolve/94f2bd91db0737d4da59f347910662905ecb5a09/classifier/model.safetensors"}]},
    "regression": {"repo": "nums-ai/causilo", "revision": "94f2bd91db0737d4da59f347910662905ecb5a09",
      "files": [{"path": "regression/model.safetensors", "bytes": 148417316,
                 "url": "https://huggingface.co/nums-ai/causilo/resolve/94f2bd91db0737d4da59f347910662905ecb5a09/regressor/model.safetensors"}]}
  },
  "graph": {"classification": "graph_causilo_classification", "regression": "graph_causilo_regression",
    "tensor_map": {"classification": "tensor_map_causilo_classification.json",
                   "regression": "tensor_map_causilo_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 10000, "max_features": 500, "max_classes": 10}
})json";

// LimiX-2M (StableAI), 2.38M parameters, 12 layers, embedding 96: axis-wise sample/feature attention with
// a decimal (sign, exponent, mantissa) numeric tokenizer. Upstream CODE is Apache-2.0. The WEIGHTS carry the
// Stable AI Technology Co., Ltd. License v1.0 (stable-ai/LimiX-2M LICENSE.txt, 2026-09), which is Apache-2.0
// plus one added Section 10: commercial use is permitted, subject to the attribution "Built with StableAI
// LimiX" when the Work or a product/service incorporating it is distributed or made available, and to a
// "LimiX" prefix on the name of any AI model created from the weights. Internal evaluation is exempt.
//
// The sources disagree and the registry follows the newest: the older upstream README (vendor/limix)
// says "academic research; commercial use upon authorization", and the Hugging Face model card is
// internally inconsistent (front matter `license: other`, body "Apache 2.0"). LICENSE.txt, added
// 2026-09-15, is the governing text, so this is commercial:true gated by the licence ACKNOWLEDGEMENT
// (accept_hf_license). Whether shipping weight-free graphs that load the weights at runtime counts as
// "distributing the Work" under Section 10 has NOT been confirmed with StableAI; docs/REAL_MODELS.md says so.
//
// Its graph is (x, y)-only like TabICL's and Causilo's: the split is the length of y, so the engine needs
// no model-specific code (tools/export_limix), and ROCm refuses it by name (docs/ROCM_SINGLE_EVAL_POS.md).
// The model standardises its own features and the export wrapper standardises the regression target
// in-graph, hence a *_raw profile. ONE checkpoint serves both tasks, so both name the same file.
//
// What this does NOT reproduce, so the docs do not over-claim: the engine runs ONE estimator, while
// upstream's leaderboard figures come from a feature-view / retrieval ensemble; and the engine
// mean-imputes missing cells before the graph, so LimiX's own missing-value encoding is unused.
//
// size_regime is an engine cap chosen from MEASUREMENT, not an upstream claim (upstream states none).
// CPU, default threads, 200 query rows (rows x features -> wall, peak RSS): 500x20 7 s; 1000x20 16 s /
// 1.3 GB; 1000x50 24 s; 1000x100 39 s / 4.6 GB; 2000x20 39 s / 3.7 GB; 3000x20 52 s / 6.2 GB;
// 3000x50 142 s / 14.3 GB; 5000x20 132 s / 15.3 GB. Memory grows roughly with rows^1.7 and linearly in
// features, so the corner of the cap (5000x100) would need on the order of 70 GB: the two limits are
// independent, and docs/REAL_MODELS.md gives the table. The first guess (10000 x 500) was wrong.
static const char *const BUILTIN_LIMIX2M = R"json({
  "schema_version": 2, "id": "limix-2m", "display_name": "LimiX-2M (StableAI)",
  "family": "icl-transformer",
  "license": {"id": "limix-2m-license-v1.0", "commercial": true, "redistributable": true,
              "gate_setting": "accept_hf_license",
              "attribution": "LimiX-2M by Stable AI Technology Co., Ltd. Code Apache-2.0; weights under the Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 plus Section 10), which permits commercial use. Section 10 requires, if you distribute or make available the Work, a Derivative Work, or any product or service that incorporates it: (a) a copy of or link to the License, and (b) the attribution \"Built with StableAI LimiX\". An AI model created, trained, fine-tuned or distilled from the weights and made available to a third party must include \"LimiX\" at the beginning of the name of such AI model. Internal research, evaluation, benchmarking or testing is exempt. License: https://huggingface.co/stable-ai/LimiX-2M/blob/main/LICENSE.txt. Checkpoint: HF stable-ai/LimiX-2M."},
  "preprocessing_profile": "limix_v1_raw",
  "weights": {
    "classification": {"repo": "stable-ai/LimiX-2M", "revision": "641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e",
      "files": [{"path": "model.ckpt", "bytes": 9558253,
                 "url": "https://huggingface.co/stable-ai/LimiX-2M/resolve/641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e/LimiX-2M.ckpt"}]},
    "regression": {"repo": "stable-ai/LimiX-2M", "revision": "641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e",
      "files": [{"path": "model.ckpt", "bytes": 9558253,
                 "url": "https://huggingface.co/stable-ai/LimiX-2M/resolve/641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e/LimiX-2M.ckpt"}]}
  },
  "graph": {"classification": "graph_limix2m_classification", "regression": "graph_limix2m_regression",
    "tensor_map": {"classification": "tensor_map_limix2m_classification.json",
                   "regression": "tensor_map_limix2m_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 5000, "max_features": 100, "max_classes": 10}
})json";

// Google TabFM 1.1.0 (HF google/tabfm-1.1.0-pytorch, 2026-08-01). The SAME architecture code as
// tabfm-v1 (vendor/tabfm model.py at the pinned submodule needs no change, and upstream's public
// repo has no 1.1-specific code at all), but a different checkpoint, so it is its own catalog entry
// rather than a flag on tabfm-v1 -- the two score differently and v1 must keep working unchanged:
//   * embed_dim 256 -> 128 (icl_dim 2048 -> 1024): 410 M params instead of 1.64 B, 0.82 GB per task
//     instead of 6.56 GB, because the dims are 2x smaller AND the tensors ship as BF16 (v1: F32).
//     The engine's safetensors reader upcasts BF16 -> F32 exactly (a 16-bit shift), so the CPU
//     graph computes in f32 on the bf16 values, which is what upstream's torch load does too.
//   * decoder_hidden is explicit (2048) where v1 has null; 2048 is also v1's fallback rule
//     (icl_dim * 2), so the graph needs nothing new, but tools/export_onnx now carries the value
//     through instead of relying on the coincidence.
// Same licence text as v1 (the LICENSE files are byte-identical): TabFM Non-Commercial License
// v1.0, so commercial:false + accept_hf_license. The HF repo itself is NOT access-gated (no token).
// Tensor maps are REUSED from tabfm-v1: same 913 / 918 state_dict keys, and the exporter writes
// byte-identical maps. The graphs are new (dims are baked into Reshape constants). Pinned to the
// revision the graphs and the GPU-graph header sha were made against.
static const char *const BUILTIN_TABFM_V1_1 = R"json({
  "schema_version": 2, "id": "tabfm-v1-1", "display_name": "Google TabFM 1.1",
  "family": "icl-transformer",
  "license": {"id": "tabfm-non-commercial-v1.0", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "TabFM 1.1.0 by Google Research. Code Apache-2.0; weights TabFM Non-Commercial License v1.0 -- non-commercial testing, evaluation and research only, no production use, no redistribution. Checkpoints: HF google/tabfm-1.1.0-pytorch."},
  "preprocessing_profile": "tabfm-v1",
  "weights": {
    "classification": {"repo": "google/tabfm-1.1.0-pytorch", "revision": "d7cfd3b8d184d7c8fed4210aeba9e1c967d45f2a",
      "files": [{"path": "classification/model.safetensors", "bytes": 820550772,
                 "url": "https://huggingface.co/google/tabfm-1.1.0-pytorch/resolve/d7cfd3b8d184d7c8fed4210aeba9e1c967d45f2a/classification/model.safetensors"}]},
    "regression": {"repo": "google/tabfm-1.1.0-pytorch", "revision": "d7cfd3b8d184d7c8fed4210aeba9e1c967d45f2a",
      "files": [{"path": "regression/model.safetensors", "bytes": 824695714,
                 "url": "https://huggingface.co/google/tabfm-1.1.0-pytorch/resolve/d7cfd3b8d184d7c8fed4210aeba9e1c967d45f2a/regression/model.safetensors"}]}
  },
  "graph": {"classification": "graph_tabfm_v1_1_classification", "regression": "graph_tabfm_v1_1_regression",
    "tensor_map": {"classification": "tensor_map_classification.json",
                   "regression": "tensor_map_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"},
                                 "train_size": {"name": "train_size", "dtype": "i64"}, "n_features": {"name": "d", "dtype": "i64"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 100000, "max_features": 500, "max_classes": 10}
})json";

// Mitra-v2 (AWS AutoGluon, Apache-2.0, ungated; arXiv 2609.04540). A NEW entry: `mitra` (v1) is
// unchanged. The two tasks are not equally new:
//
//  * autogluon/mitra-classifier-2 is a drop-in. config.json is byte-identical to v1's and the
//    safetensors JSON header is byte-identical too (sha cb1a261b..., 392 tensors -- same names,
//    shapes, dtypes and offsets; only the values differ), so classification reuses mitra's graph,
//    tensor map and GPU graphs verbatim (see BundledGpuGraphId) and embeds no new bytes.
//  * autogluon/mitra-regressor-2 is NOT: dim_output 1 -> 1000. Regression is cast as classification
//    over 1,000 target bins: final_layer is [1000, 512], and each support target is embedded by its
//    BIN (y_embedding becomes an Embedding [1000, 512] with no bias; 393 -> 392 tensors). Its graph
//    (tools/export_mitra --config real2) min-max scales the raw target over the context rows, bins it
//    on linspace(-0.5, 1.5, 1001), and decodes the mean of the bin distribution back to raw units --
//    raw in, raw out, hence the "_raw" profile. Features are passed raw too: Mitra's quantile
//    embedding is rank-based, i.e. invariant to the engine's monotone scaling either way, and raw is
//    what upstream feeds it.
//
// The size regime is v2's documented operating range rather than v1's: prediction draws up to 32,768
// support rows and the deployment feature budget is 256 (Mitra-v2 Technical Report sec. 3.1, 3.5;
// pre-trained on 1-50 features, 160-5,120 support rows -- "10x longer context, 3x features"). The
// card's headline numbers use 50-step fine-tuning with 8-fold bagging; this entry is in-context only.
static const char *const BUILTIN_MITRA2 = R"json({
  "schema_version": 2, "id": "mitra-v2", "display_name": "Mitra-v2 (AWS AutoGluon)",
  "family": "icl-transformer",
  "license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "Mitra-v2 by AWS AutoGluon, Apache-2.0. Checkpoints: HF autogluon/mitra-classifier-2, autogluon/mitra-regressor-2."},
  "preprocessing_profile": "mitra_v2_raw",
  "weights": {
    "classification": {"repo": "autogluon/mitra-classifier-2", "revision": "main",
      "files": [{"path": "model.safetensors", "bytes": 302717904,
                 "url": "https://huggingface.co/autogluon/mitra-classifier-2/resolve/main/model.safetensors"}]},
    "regression": {"repo": "autogluon/mitra-regressor-2", "revision": "main",
      "files": [{"path": "model.safetensors", "bytes": 306776952,
                 "url": "https://huggingface.co/autogluon/mitra-regressor-2/resolve/main/model.safetensors"}]}
  },
  "graph": {"classification": "graph_mitra_classification", "regression": "graph_mitra2_regression",
    "tensor_map": {"classification": "tensor_map_mitra_classification.json",
                   "regression": "tensor_map_mitra2_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"},
                                 "train_size": {"name": "train_size", "dtype": "i64"}, "n_features": {"name": "d", "dtype": "i64"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 32768, "max_features": 256, "max_classes": 10}
})json";

// TabDPT v1.3 (Layer 6 AI, Apache-2.0, ungated). A NEW entry: `tabdpt` stays on v1.2. Same HF repo
// (Layer6/TabDPT), new file tabdpt1_3.safetensors; TabArena 1545 Elo (v1.3 default) vs 1457 (v1.2).
//
// The checkpoint cfg carries v1.2's exact dims, but the y-encoders changed: v1.2's per-layer MLP
// (y_encoders.*) became a per-TASK pair -- cls_y_encoders.* (Embedding(16, 128) over the class id) and
// reg_y_encoders.* (Linear(1,128) -> GELU -> Linear(128,128)); 647 -> 679 tensors, every other tensor
// unchanged, head (16 logits + 2048 regression bins) unchanged. So it has its own graphs
// (tools/export_tabdpt --config real13) under the stem "tabdpt13". v1.3's "full probabilistic
// regression outputs" are an estimator API over the same bins; the engine returns the point estimate
// (the bar-distribution mean), as for v1.2.
//
// The cache slug is keyed by repo and is SHARED with `tabdpt`, so the file lives under tabdpt1_3/ --
// a shared files[].path would make the two entries overwrite each other (the tabpfn-v2-5-real rule).
static const char *const BUILTIN_TABDPT13 = R"json({
  "schema_version": 2, "id": "tabdpt-v1-3", "display_name": "TabDPT v1.3 (Layer 6 AI)",
  "family": "icl-transformer",
  "license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "TabDPT by Layer 6 AI, Apache-2.0. Checkpoint: HF Layer6/TabDPT (tabdpt1_3.safetensors)."},
  "preprocessing_profile": "tabdpt_v1_raw",
  "weights": {
    "classification": {"repo": "Layer6/TabDPT", "revision": "main",
      "files": [{"path": "tabdpt1_3/model.safetensors", "bytes": 252233296,
                 "url": "https://huggingface.co/Layer6/TabDPT/resolve/main/tabdpt1_3.safetensors"}]},
    "regression": {"repo": "Layer6/TabDPT", "revision": "main",
      "files": [{"path": "tabdpt1_3/model.safetensors", "bytes": 252233296,
                 "url": "https://huggingface.co/Layer6/TabDPT/resolve/main/tabdpt1_3.safetensors"}]}
  },
  "graph": {"classification": "graph_tabdpt13_classification", "regression": "graph_tabdpt13_regression",
    "tensor_map": {"classification": "tensor_map_tabdpt13_classification.json",
                   "regression": "tensor_map_tabdpt13_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 100000, "max_features": 128, "max_classes": 16}
})json";

// Nori-30M (Synthefy), 29.2M parameters: a LimiX fork (upstream NOTICE: "originated as a fork of LimiX")
// grown to 28 pre-norm `smf` layers with RMSNorm, SwiGLU, DeepNorm residual scaling, a query-adaptive
// softmax temperature (QASSMax) in the sample attention and a target-aware embedding of the context labels.
// CODE AND WEIGHTS are Apache-2.0 (model card front matter and body agree; repo LICENSE + NOTICE) and the
// HF repo is ungated, so it is registered like tabdpt: commercial, redistributable, no licence gate.
//
// REGRESSION ONLY. The checkpoint config carries a 10-class head, but upstream's build_model refuses any
// task_type other than "reg" and the model card calls it a regression model; `capabilities` is
// ["regress"], so tabfm_classify is refused before any download with the actionable unsupported-task error
// (which names the models that do classify).
//
// Its graph is (x, y)-only like LimiX-2M's: the split is the length of y, so the engine needs no
// model-specific code (tools/export_nori). For ROCm a second, MASKED graph (x, y_full, train_size, d, n_rows;
// tools/export_nori --contract mask) is bundled; it reads the safetensors tabfm_download writes beside
// nori.pt, and without that file ROCm reports the bundled graph as not matching the weights. The
// model standardises its own features and the export wrapper standardises the target in-graph and inverts
// it, hence a *_raw profile. The 999-quantile pinball head is reduced IN-GRAPH to the mean of the implied
// quantile function, upstream's default point estimate (NoriRegressor quantile_collapse="mean"); on the
// uniform tau grid that is a closed form without a sort (tools/export_nori nori_patches.quantile_mean).
//
// The engine reads nori.pt natively (a torch.save archive {model_config, ema_state_dict}; the reader
// prefers ema_state_dict, as upstream's load_model does), so the CPU needs no converter. Pinned to the commit
// the graph was exported from.
//
// What this does NOT reproduce, so the docs do not over-claim: upstream's default NoriRegressor runs an
// 8-member preprocessing ensemble (quantile / adaptive feature transforms, polynomial interactions,
// fingerprint features, feature shuffles); the engine runs ONE forward on the raw table, which is
// upstream's own model.forward. The engine mean-imputes missing cells before the graph, so Nori's
// missing-value embedding is unused.
//
// size_regime is an engine cap chosen from MEASUREMENT (CPU, default threads, 200 query rows; rows x
// features -> wall, peak RSS): 500x20 5.6 s / 0.85 GB; 1000x20 10.8 s / 1.5 GB; 2000x20 28 s / 3.3 GB;
// 3000x20 53 s / 6.1 GB; 5000x20 127 s / 15.4 GB; 1000x100 40 s / 5.0 GB. The two limits are independent
// (their corner is far larger); docs/REAL_MODELS.md gives the table.
static const char *const BUILTIN_NORI30M = R"json({
  "schema_version": 2, "id": "nori-30m", "display_name": "Nori-30M (Synthefy)",
  "family": "icl-transformer",
  "license": {"id": "apache-2.0", "commercial": true, "redistributable": true, "gate_setting": null,
              "attribution": "Nori-30M by Synthefy, Apache-2.0 (code and weights). Checkpoint: HF Synthefy/Nori-30M (nori.pt). Regression only."},
  "preprocessing_profile": "nori_v1_raw",
  "weights": {
    "regression": {"repo": "Synthefy/Nori-30M", "revision": "63c9f7facf9fb32c37ce3fc2fba331d524696318",
      "files": [{"path": "model.ckpt", "bytes": 117093952,
                 "url": "https://huggingface.co/Synthefy/Nori-30M/resolve/63c9f7facf9fb32c37ce3fc2fba331d524696318/nori.pt"}]}
  },
  "graph": {"regression": "graph_nori30m_regression",
    "tensor_map": {"regression": "tensor_map_nori30m_regression.json"}},
  "capabilities": ["regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 5000, "max_features": 100, "max_classes": 10}
})json";

// EXAONE Tabular v1 (LG AI Research; TabArena v0.1.4 classification Elo 1759, regression 1883 per the model
// card), a Cross-Axis Summary Transformer: width 192, 12 blocks, one column per feature token, 3 item-summary
// and 32 feature-summary tokens, SSMax attention; 20.8M (classifier) / 21.1M (regressor) parameters.
// Upstream CODE is BSD-3-Clause-LG. The WEIGHTS are under the EXAONE AI Model License Agreement 1.2 - NC:
// research and educational use only (s. 2.1a: "evaluation, testing, academic research, experimentation"), and
// "the Licensee is expressly prohibited from using the Model, Derivatives, or Output for any commercial
// purposes" (s. 3.1). Registered like the other non-commercial entries (commercial:false, redistributable:false,
// gated by accept_hf_license); the engine never redistributes the weights (the user downloads them from LG's
// Hugging Face repo, which is ungated). docs/REAL_MODELS.md quotes the clauses and the reasoning.
//
// Its graph is (x, y)-only like TabICL's: the split is the length of y, so the engine needs no model-specific
// code (tools/export_exaone). For ROCm a second, MASKED graph per task (x, y_full, train_size, d, n_rows;
// tools/export_exaone --contract mask) is bundled, keyed to LG's released files by header sha. The model standardises
// its own features and the wrapper standardises the regression target in-graph, hence a *_raw profile. The
// 999-quantile regression head is reduced in-graph to the MEAN of the bank, the released regressor's own
// point_estimate="mean". Upstream pins part of the attention to FlashAttention on CUDA (fp16/bf16 only);
// the export writes the plain softmax(q k^T / sqrt(d)) v path, the same function in float32.
//
// The released files are safetensors keyed by the module namespace, so there is no converter; they are pinned to
// the commit the graphs were exported against. The classifier and the regressor are DIFFERENT checkpoints in the
// same repo, hence per-task paths (classification/ and regression/), as for causilo.
//
// What this does NOT reproduce, so the docs do not over-claim: the TabArena figures come from upstream's
// 8-member ensemble (four numeric transforms, sign flips, feature permutations, and for >10 classes
// error-correcting output codes); the engine runs ONE forward on the raw table, upstream's own model.forward.
// The engine mean-imputes missing cells before the graph, so EXAONE's non-finite code channel never sees a NaN.
//
// size_regime is an engine cap chosen from MEASUREMENT, not upstream's support_row_limit (100000, which its
// cached, chunked CUDA runtime serves): the engine runs the dense forward on the whole table. CPU, default
// threads, 200 query rows (rows x features -> wall, peak RSS): 500x20 4.3 s / 1.2 GB; 1000x20 9.7 s / 2.5 GB;
// 2000x20 26 s / 6.7 GB; 3000x20 49 s / 13.1 GB; 1000x100 37 s / 10.5 GB. Memory grows roughly with rows^1.7
// and linearly in features; the limits are independent, and docs/REAL_MODELS.md gives the table.
static const char *const BUILTIN_EXAONE_TABULAR = R"json({
  "schema_version": 2, "id": "exaone-tabular", "display_name": "EXAONE Tabular v1 (LG AI Research)",
  "family": "icl-transformer",
  "license": {"id": "exaone-nc-1.2", "commercial": false, "redistributable": false,
              "gate_setting": "accept_hf_license",
              "attribution": "EXAONE Tabular by LG AI Research (LG Management Development Institute Co., Ltd.). Code BSD-3-Clause-LG; weights under the EXAONE AI Model License Agreement 1.2 - NC: research and educational use only, NO commercial use of the Model, Derivatives or Output (s. 3.1); a commercial license is a separate agreement with LG. Publications must attribute the Model's name and version (s. 4.3). License: https://huggingface.co/LG-AI-Research/EXAONE-Tabular/blob/main/LICENSE. Checkpoints: HF LG-AI-Research/EXAONE-Tabular."},
  "preprocessing_profile": "exaone_v1_raw",
  "weights": {
    "classification": {"repo": "LG-AI-Research/EXAONE-Tabular", "revision": "093ad1c2613e1e95d26df8610451d2e02ed53cbe",
      "files": [{"path": "classification/model.safetensors", "bytes": 83286800,
                 "url": "https://huggingface.co/LG-AI-Research/EXAONE-Tabular/resolve/093ad1c2613e1e95d26df8610451d2e02ed53cbe/exaone-tabular-classifier-v1_default.safetensors"}]},
    "regression": {"repo": "LG-AI-Research/EXAONE-Tabular", "revision": "093ad1c2613e1e95d26df8610451d2e02ed53cbe",
      "files": [{"path": "regression/model.safetensors", "bytes": 84502160,
                 "url": "https://huggingface.co/LG-AI-Research/EXAONE-Tabular/resolve/093ad1c2613e1e95d26df8610451d2e02ed53cbe/exaone-tabular-regressor-v1_default.safetensors"}]}
  },
  "graph": {"classification": "graph_exaone_tabular_classification", "regression": "graph_exaone_tabular_regression",
    "tensor_map": {"classification": "tensor_map_exaone_tabular_classification.json",
                   "regression": "tensor_map_exaone_tabular_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 3000, "max_features": 100, "max_classes": 10}
})json";

// NVIDIA Kumo-Tabular (TabArena Elo, as published by NVIDIA with num_estimators=8: large 2017, medium
// 1936, small 1820), three sizes of one TabFM/TabICL-lineage architecture: Fourier cell embeddings,
// induced column attention, RoPE row attention over 4 readout tokens, then an ICL transformer with
// test-time GQA. Code Apache-2.0 (NVIDIA/structured-data-models); WEIGHTS OpenMDW-1.1, which grants use
// "without restriction" (commercial included) and asks one thing of a distributor: keep a copy of the
// agreement and NVIDIA's copyright notices. The HF repo is not gated, so there is no licence gate here.
// Whether shipping weight-free graphs of the architecture is "distributing a portion of the Model
// Materials" is not settled; the attribution carries the notice either way (docs/REAL_MODELS.md).
//
// Graph contract: the tabfm-v1 five inputs (x, y[1,T], train_size, cat_mask, d) -- upstream's own
// padded-context / padded-column path (_KumoTabular.forward_train), so the split and the real width
// are VALUES and one graph family serves CPU, CUDA (ext) and bucket-padded ROCm (migraphx). The engine
// standardises features and target: Kumo's single-estimator recipe is the engine's profile step for
// step (tools/export_kumo/src/export_kumo/engine_ref.py), hence *_minimal, not *_raw.
//
// Weights are the published .pt (a plain float32 state_dict), declared as model.ckpt so the engine's
// native reader injects it on the CPU with no conversion; tabfm_download writes the sibling
// model.safetensors the ext/migraphx graphs need (ExpectedWeightsHeaderShaFor pins its header). The
// classifier .pt stores both y_emb.weight tensors as transposed views, which the reader gathers in order.
// Pinned to tag v1.0.1, the revision upstream's loader requests; the LFS hashes are identical on main.
//
// What this does NOT reproduce: the engine runs ONE estimator (the Elo figures above are an 8-member
// ensemble of random sign flips, column shuffles, class shifts and four numeric transforms); more than
// 10 classes (upstream: error-correcting output codes) is refused by the head width; category codes
// are the engine's first-appearance ordinals, not upstream's sort-by-value codes, and the
// category-count columns upstream adds for >= 50 categories are not added; NULLs are mean-imputed
// before the graph, so Kumo's missing-value projection is unused.
//
// size_regime is an engine cap from MEASUREMENT (upstream states none; 500 features is its own
// SelectColumns(500)). CPU, 8 threads, 80% context (rows x features -> wall, peak RSS): small 2000x20
// 5.4 s / 1.8 GB, 5000x20 19 s / 5.0 GB, 10000x50 73 s / 30 GB; large 2000x20 26 s / 4.5 GB, 5000x20
// 123 s / 14.7 GB. The masked graph attends over all T rows, so memory grows faster than linearly in
// rows; 10000 rows would already need ~30 GB on the SMALL model, hence 5000 for all three.

static const char *const BUILTIN_KUMO_SMALL = R"json({
  "schema_version": 2, "id": "kumo-tabular-small", "display_name": "Kumo-Tabular small (NVIDIA)",
  "family": "icl-transformer",
  "license": {"id": "openmdw-1.1", "commercial": true, "redistributable": true, "gate_setting": null,
              "text_resource": "license_openmdw_1_1_nvidia.txt",
              "attribution": "NVIDIA Kumo-Tabular (small; 27.5 M / 28.5 M parameters classifier / regressor). Code Apache-2.0 (NVIDIA/structured-data-models); weights OpenMDW-1.1, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Commercial use is permitted. If you distribute any portion of the Model Materials, retain a copy of the OpenMDW-1.1 agreement and NVIDIA's copyright and origin notices (both follow, as this extension carries them): https://huggingface.co/nvidia/Kumo-Tabular/blob/main/LICENSE. Checkpoints: HF nvidia/Kumo-Tabular v1.0.1 (small/)."},
  "preprocessing_profile": "kumo_tabular_v1_minimal",
  "weights": {
    "classification": {"repo": "nvidia/Kumo-Tabular", "revision": "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f",
      "files": [{"path": "small/classification/model.ckpt", "bytes": 109936978,
                 "url": "https://huggingface.co/nvidia/Kumo-Tabular/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/small/classifier.pt"}]},
    "regression": {"repo": "nvidia/Kumo-Tabular", "revision": "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f",
      "files": [{"path": "small/regression/model.ckpt", "bytes": 113968523,
                 "url": "https://huggingface.co/nvidia/Kumo-Tabular/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/small/regressor.pt"}]}
  },
  "graph": {"classification": "graph_kumo_small_classification", "regression": "graph_kumo_small_regression",
    "tensor_map": {"classification": "tensor_map_kumo_small_classification.json",
                   "regression": "tensor_map_kumo_small_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"},
                                 "train_size": {"name": "train_size", "dtype": "i64"}, "n_features": {"name": "d", "dtype": "i64"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 5000, "max_features": 500, "max_classes": 10}
})json";

static const char *const BUILTIN_KUMO_MEDIUM = R"json({
  "schema_version": 2, "id": "kumo-tabular-medium", "display_name": "Kumo-Tabular medium (NVIDIA)",
  "family": "icl-transformer",
  "license": {"id": "openmdw-1.1", "commercial": true, "redistributable": true, "gate_setting": null,
              "text_resource": "license_openmdw_1_1_nvidia.txt",
              "attribution": "NVIDIA Kumo-Tabular (medium; 61.5 M / 62.5 M parameters classifier / regressor). Code Apache-2.0 (NVIDIA/structured-data-models); weights OpenMDW-1.1, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Commercial use is permitted. If you distribute any portion of the Model Materials, retain a copy of the OpenMDW-1.1 agreement and NVIDIA's copyright and origin notices (both follow, as this extension carries them): https://huggingface.co/nvidia/Kumo-Tabular/blob/main/LICENSE. Checkpoints: HF nvidia/Kumo-Tabular v1.0.1 (medium/)."},
  "preprocessing_profile": "kumo_tabular_v1_minimal",
  "weights": {
    "classification": {"repo": "nvidia/Kumo-Tabular", "revision": "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f",
      "files": [{"path": "medium/classification/model.ckpt", "bytes": 246117940,
                 "url": "https://huggingface.co/nvidia/Kumo-Tabular/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/medium/classifier.pt"}]},
    "regression": {"repo": "nvidia/Kumo-Tabular", "revision": "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f",
      "files": [{"path": "medium/regression/model.ckpt", "bytes": 250144651,
                 "url": "https://huggingface.co/nvidia/Kumo-Tabular/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/medium/regressor.pt"}]}
  },
  "graph": {"classification": "graph_kumo_medium_classification", "regression": "graph_kumo_medium_regression",
    "tensor_map": {"classification": "tensor_map_kumo_medium_classification.json",
                   "regression": "tensor_map_kumo_medium_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"},
                                 "train_size": {"name": "train_size", "dtype": "i64"}, "n_features": {"name": "d", "dtype": "i64"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 5000, "max_features": 500, "max_classes": 10}
})json";

static const char *const BUILTIN_KUMO_LARGE = R"json({
  "schema_version": 2, "id": "kumo-tabular-large", "display_name": "Kumo-Tabular large (NVIDIA)",
  "family": "icl-transformer",
  "license": {"id": "openmdw-1.1", "commercial": true, "redistributable": true, "gate_setting": null,
              "text_resource": "license_openmdw_1_1_nvidia.txt",
              "attribution": "NVIDIA Kumo-Tabular (large; 213.7 M / 215.7 M parameters classifier / regressor). Code Apache-2.0 (NVIDIA/structured-data-models); weights OpenMDW-1.1, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Commercial use is permitted. If you distribute any portion of the Model Materials, retain a copy of the OpenMDW-1.1 agreement and NVIDIA's copyright and origin notices (both follow, as this extension carries them): https://huggingface.co/nvidia/Kumo-Tabular/blob/main/LICENSE. Checkpoints: HF nvidia/Kumo-Tabular v1.0.1 (large/)."},
  "preprocessing_profile": "kumo_tabular_v1_minimal",
  "weights": {
    "classification": {"repo": "nvidia/Kumo-Tabular", "revision": "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f",
      "files": [{"path": "large/classification/model.ckpt", "bytes": 854850034,
                 "url": "https://huggingface.co/nvidia/Kumo-Tabular/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/large/classifier.pt"}]},
    "regression": {"repo": "nvidia/Kumo-Tabular", "revision": "b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f",
      "files": [{"path": "large/regression/model.ckpt", "bytes": 862909259,
                 "url": "https://huggingface.co/nvidia/Kumo-Tabular/resolve/b2f5a9d6404e3574df6c30cc6ae66a30b8e4f06f/large/regressor.pt"}]}
  },
  "graph": {"classification": "graph_kumo_large_classification", "regression": "graph_kumo_large_regression",
    "tensor_map": {"classification": "tensor_map_kumo_large_classification.json",
                   "regression": "tensor_map_kumo_large_regression.json"}},
  "capabilities": ["classify", "regress"],
  "tensor_contract": {"inputs": {"features": {"name": "x", "dtype": "f32"}, "labels": {"name": "y", "dtype": "f32"},
                                 "train_size": {"name": "train_size", "dtype": "i64"}, "n_features": {"name": "d", "dtype": "i64"}},
                      "outputs": {"logits": {"name": "logits", "dtype": "f32"}}},
  "size_regime": {"max_rows": 5000, "max_features": 500, "max_classes": 10}
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
	specs.push_back(ParseModelSpec(BUILTIN_ORION_MSP, "(built-in orion-msp)"));
	specs.push_back(ParseModelSpec(BUILTIN_CAUSILO, "(built-in causilo)"));
	specs.push_back(ParseModelSpec(BUILTIN_LIMIX2M, "(built-in limix-2m)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABFM_V1_1, "(built-in tabfm-v1-1)"));
	specs.push_back(ParseModelSpec(BUILTIN_MITRA2, "(built-in mitra-v2)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABDPT13, "(built-in tabdpt-v1-3)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN35, "(built-in tabpfn-v3-5)"));
	specs.push_back(ParseModelSpec(BUILTIN_TABPFN35_FAST, "(built-in tabpfn-v3-5-fast)"));
	specs.push_back(ParseModelSpec(BUILTIN_NORI30M, "(built-in nori-30m)"));
	specs.push_back(ParseModelSpec(BUILTIN_EXAONE_TABULAR, "(built-in exaone-tabular)"));
	specs.push_back(ParseModelSpec(BUILTIN_KUMO_SMALL, "(built-in kumo-tabular-small)"));
	specs.push_back(ParseModelSpec(BUILTIN_KUMO_MEDIUM, "(built-in kumo-tabular-medium)"));
	specs.push_back(ParseModelSpec(BUILTIN_KUMO_LARGE, "(built-in kumo-tabular-large)"));
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

string ModelRegistry::IdsWithTask(TabFMTask task) const {
	string out;
	for (auto &kv : models_) {
		if (!kv.second.HasTask(task)) {
			continue;
		}
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
