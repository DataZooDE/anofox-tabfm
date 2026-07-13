# Multi-model registry — implementation plan (my own, after critiquing the proposal)

Source proposal: `/home/jr/Projects/tmp/research/2026-07-13-tabfm-multi-model/`
(REPORT, INTERFACE, INTEGRATION-CHECKLIST, manifests/mitra.json). This is **not**
a restatement of it — it is a scoped, code-grounded plan that keeps the sound
core and defers/rejects the speculative parts with reasons.

## What the proposal gets right (implement)

The proposal's premise checks out against the code:
- `TabFMPredictOptions.model` exists and is parsed (`tabfm_predict_agg.cpp:172`)
  but is **inert** — `ResolveModel` (`tabfm_engine.cpp:186`) selects only
  built-in-vs-`model_manifest_path`, never by id. The seam is real and unwired.
- A built-in manifest already exists (`BuiltinTabFMManifest(task)`), and manifests
  are strictly parsed/validated (`tabfm_manifest.cpp`).
- So a **model registry (id → manifest, N models)** is the genuine FR-5.1/M4
  deliverable ("a second model is a manifest, not new C++"), and it is the
  highest-value, most self-contained slice.

## What I am changing vs the proposal (judgement calls)

1. **REJECT the `SidecarBackend` (Python worker over a socket).** It directly
   contradicts the extension's core invariant — a **self-contained single-file**
   `.duckdb_extension` (we just merged PR #4 to static-link ORT precisely so
   there is *no* companion runtime). A Python sidecar reintroduces a runtime
   dependency, IPC, a security surface, and version hell. "A model is usable
   before its ONNX export exists" is a bug, not a feature: the ONNX-export bar is
   the moat. Documented as rejected; not built.
2. **DEFER the real Mitra integration to an offline export spike.** Mitra needs a
   Python `torch.onnx.export` spike (trace the forward pass, externalize weights,
   tensor map, parity vs AutoGluon) — the manifest's `tensor_contract` is itself
   flagged PLACEHOLDER ("MUST be confirmed by tracing"). That is ML/export work,
   not C++ TDD, and it cannot be done "no mocks, real tests" without real weights
   and a working autogluon export. It gets a checklist, not code, this round.
3. **DEFER `RetrievalOnnxBackend` (TabDPT) and the extra-task functions**
   (`impute`/`anomaly`/`generate`/`embed`, which need TabPFN v2). No shipped model
   supports them; building the surface now is dead code. The **capability flag +
   gating** is the only foundation needed now.
4. **PROVE multi-model with a second _real_ fixture model, not a mock.** Following
   the committed S06 CI-fixture pattern (random-init weights + weight-free graph),
   a second fixture model is a *genuine* model in the registry: real v2 manifest,
   real safetensors, a real ORT forward pass, real (meaningless-but-valid)
   predictions — zero external bytes, zero mocks. This is exactly what
   `test/fixtures/manifest.json` already is for TabFM. It lets every registry
   behaviour (`model :=`, list, per-model download/load, commercial-vs-gated,
   capability gating, unknown-model/unsupported-task errors) be tested with real
   protocol tests offline.

Net: implement the **registry foundation + manifest-driven tensor contract**, the
actual M4 slice, and prove it end-to-end with a real second fixture model. Defer
the model-acquisition ML spikes and reject the sidecar.

## Ground rules
- **Red/green TDD.** Every step: failing test first (Catch2 for manifest/registry
  units, sqllogictest for the SQL surface running the *real* engine), then code.
- **No mocks.** Manifest parsing tested against real JSON; the SQL surface tested
  against the real ORT engine driving real fixture graphs.
- **Total back-compat.** Every existing v1 call/manifest/test keeps passing; the
  first registry step is a no-op refactor validated against the current suite.

## Phases

**P1 — Manifest schema v2 (backward-compatible).** Extend `ModelManifest` +
`ParseModelManifest` to accept v2 (`schema_version`, `id`, `display_name`,
`family`, `license` object with `commercial`/`redistributable`/`gate_setting`,
`weights` keyed by task, `graph` object keyed by task, `tensor_contract`,
`capabilities`, `size_regime`, `compute`) **and** v1 (flat `task`/`files`/`graph`/
`license`-string) unchanged. Tests: v1 fixtures still parse; a v2 fixture parses;
license-object vs string; capabilities/tensor_contract/size_regime.

**P2 — ModelRegistry.** One registry: built-in `google-tabfm-v1` + user manifests
from `anofox_tabfm_model_manifest` (now a dir *or* file, merged). Resolution
precedence: `model :=` → `anofox_tabfm_default_model` → sole model (else an error
listing ids). **Unify the duplicate `WeightsManifest`** onto the registry so
predict and weights share one source of truth (retires the `tabfm_weights.cpp`
parser — the flagged duplication). Highest-risk refactor → prime review target.
Tests: registry with 1 then 2 models; precedence; unknown-model error.

**P3 — SQL surface.** `model :=` named arg on `tabfm_classify`/`tabfm_regress`
(promote from `opts`; `opts['model']` = deprecated alias); `tabfm_list_models()`;
extend `tabfm_models()` with `model`/`family`/`commercial`; leading model id on
`tabfm_download`/`load`/`unload`/`remove` (v1 forms → resolved default);
capability gating + per-model license gate + per-model `max_classes`; settings
`anofox_tabfm_default_model` + `model_manifest` dir/file merge; the §9 error
catalog. Tests: one sqllogictest per behaviour, driven by the fixtures.

**P4 — Manifest-driven tensor contract.** Drive `Run()` I/O from the manifest
`tensor_contract` (default = today's `x/y/train_size/cat_mask/d`), so a
different-shaped model is data-only. Tests: fixture predict still passes through
the contract-driven path; a contract/graph mismatch errors cleanly.

**P5 — Second real fixture model (the M4 proof).** Generate a second fixture
(v2 manifest, `commercial:true` icl-transformer, random weights, weight-free
graph) via the S06 generator; register it; full end-to-end sqllogictests:
`model := 'fixture-b'` runs a real forward pass, `tabfm_list_models()` shows both,
per-model download/load, commercial model needs no gate while `google-tabfm-v1`
still does, unknown-model + unsupported-task errors.

## Deferred / rejected (tracked, not built)
- **Real Mitra** → offline export spike (INTEGRATION-CHECKLIST B/C/D). `mitra.json`
  stays in research until the spike yields real graphs + parity.
- **Extra tasks (TabPFN v2), retrieval backend (TabDPT)** → later stages on this
  seam; capability flags reserve the surface.
- **SidecarBackend** → rejected (see judgement call #1).

## Review checkpoints (codex + antigravity)
Run an external review at three gates; feed findings back before moving on:
1. **After P2** (registry + the WeightsManifest unification) — the riskiest
   design change. Ask each reviewer for: correctness bugs (lifetime/ownership of
   registry entries, resolution edge cases), **remaining duplication**, and API
   design smells.
2. **After P4** (tensor-contract engine change) — ask for: ORT I/O correctness,
   dtype/shape handling, back-compat regressions, buffer lifetime.
3. **After P5** (full surface) — ask for: **usability** of `model :=` /
   `tabfm_list_models` / per-model weights, error-message actionability, and any
   back-compat break in the v1 call/manifest paths.
Each gate: `codex exec < /dev/null` and an antigravity pass on the diff since the
last gate; trisect/confirm each finding against a real test before acting on it
(the Windows-crash discipline: reproduce, don't trust the claim).

## Definition of done (this round)
Registry live with `google-tabfm-v1` + a real second fixture model; `model :=`,
`tabfm_list_models()`, per-model weights, capability gating, tensor-contract
engine all green under real (no-mock) tests on all CI platforms; the two manifest
parsers unified; docs updated ("TabFM = category"); Mitra/extra-tasks/retrieval/
sidecar dispositions recorded. Real Mitra remains a documented follow-up spike.
</content>
