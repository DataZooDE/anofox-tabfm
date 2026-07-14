# Real tabular foundation models in `anofox_tabfm`

Status of onboarding real (non-fixture) tabular foundation models behind the
extension's fixed ONNX engine contract. The registry proves FR-5.1 / M4 ("a
second model is a manifest, not new C++") — but "not new C++" only holds when a
model's forward maps onto the engine's existing input-feeding + preprocessing.
That is exactly what separates Mitra (drop-in) from TabPFN v2 / TabICL (need
engine work).

The engine feeds a fixed signature and preprocessing:
`x[1,T,H] f32, y[1,T] f32, train_size[1] i64, cat_mask[1,H] bool, d[1] i64 →
logits[1,T,C]`, with `tabfm_v1_minimal` preprocessing (z-score + first-appearance
ordinal). It feeds inputs by name and only feeds names the graph declares.

## Summary

| model | license | status | runs today | notes |
|---|---|---|---|---|
| **Google TabFM** (`tabfm-v1`) | non-commercial (gated) | ✅ shipped | yes | the original; 1.6 B params / 6.56 GB |
| **Mitra** (`mitra`) | Apache-2.0 (commercial) | ✅ shipped | **yes — zero C++ changes** | 72 M / ~303 MB; iris 0.962 in ~2.4 s |
| **TabPFN v2** (`tabpfn-v2`) | Prior Labs (Apache-2.0 + attribution) | 🧪 export-proven | no | needs engine + preprocessing + weight-conversion work |
| **TabICL v2** (`tabicl-v2`) | BSD-3-Clause (commercial) | 🧪 export-proven | no | same shared blockers as TabPFN |

## Mitra — done

Real Tab2D transformer, exported weight-free with exact parity (graph + injected
real weights vs upstream: 4.6e-6 classification, 1.5e-5 regression, 100% argmax).
It maps onto the engine with **no C++ change** because:
- Its exported contract is `x, y, train_size, d → logits` — a subset of what the
  engine already feeds (it simply doesn't declare `cat_mask`).
- It **rank/quantile-normalizes inside the graph**, so the engine's z-score
  preprocessing is harmless (a rank transform is invariant to monotonic external
  transforms).

Registered via `examples/mitra.json`; offline fixture `test/sql/tabfm_mitra.test`;
head-to-head in `examples/compare_models.sql`. Downloadable from HF
(`autogluon/mitra-{classifier,regressor}`, per-file URLs in the manifest),
Apache-2.0, ungated.

## TabPFN v2 and TabICL v2 — export-proven, integration pending

Both export cleanly to weight-free ONNX with strong parity on the real
architecture (TabPFN real-weight parity 1.18e-4 / 100% argmax; TabICL
random-weight parity 1.5e-7 — real-weight parity not yet run). The export
tooling (`tools/export_tabpfn/`, `tools/export_tabicl/`), weight-free graphs
(`resources/graph_{tabpfn,tabicl}_*.onnx`), random-init fixtures
(`test/fixtures/{tabpfn,tabicl}/`), and prototype manifests
(`examples/{tabpfn,tabicl}.json`) are all committed and reproducible. They do
**not** run through the engine yet. Shared blockers:

1. **`y` is the train-prefix, not the full column.** Both architectures derive
   the train/test split from `len(y)` (`single_eval_pos`), so their graphs take
   `y` as `[1, train_size]` (training labels only) — a data-dependent
   `y[:, :train_size]` slice with a runtime `train_size` tensor is *not*
   ONNX-exportable. The engine currently feeds `y` as the full `[1, T]`. **Fix:**
   let a model declare "feed `y` as the training prefix" (a manifest capability),
   and slice in the engine's input-feeding path. This single change unblocks
   *both* classification paths.
2. **Preprocessing must be raw-ish, not z-scored.** TabPFN normalizes inside the
   model and its docs say "don't scale yourself"; TabICL z-scores on train rows
   inside the graph. The engine's `tabfm_v1_minimal` z-score would *double-scale*
   and degrade accuracy. **Fix:** a `raw`/passthrough preprocessing profile
   selected by the manifest's `preprocessing_profile`.
3. **Distributional regression heads.** TabPFN regression = a 5000-bucket
   `FullSupportBarDistribution` (borders live in the criterion, not the graph);
   TabICL regression = 999 quantile logits. Neither is a `[1,T,1]` point
   estimate — the engine (WS-E) must reduce the distribution to a scalar. **So
   the near-term target for both is classification only.**
4. **Weights are pickle `.ckpt`, not safetensors.** HF ships PyTorch checkpoints;
   the engine injects from safetensors. **Fix:** a ckpt→safetensors conversion
   (keyed by the committed tensor map) at download-prep time — the export tools
   already know the mapping.

### Recommended next step

One focused engine feature — **manifest-declared "y-as-train-prefix" feeding +
a `raw` preprocessing profile** — plus a ckpt→safetensors download-prep step
makes **TabPFN v2 and TabICL v2 classification** run with real weights (both are
commercial/attribution-permissive, and TabPFN's ~29 MB weights are ideal for an
embedded extension). Regression follows once the distributional heads are decoded.

## License wall (all models)

Every shipped graph is weight-free (all checkpoint initializers externalized,
`.onnx.data` deleted); every fixture is seeded random-init. No real weight bytes
from any model are committed. Real weights are the user's own HF download behind
the manifest's license gate.
