# Phase 2: Model Generalization + Distribution Output (fixture-backed) - Context

**Gathered:** 2026-09-21
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous), post-spike rescope

<domain>
## Phase Boundary

Generalize the model seam so families beyond `tabfm-v1` are first-class, and carry
a regression predictive distribution end-to-end against the **confirmed TabPFN v2
tensor contract**, proven with a committed **weight-free random-init fixture
family**. Real TabPFN v2 ONNX *inference* export and all of TabICL are
**deferred** (upstream-blocked — see `.planning/spikes/`).

In scope: MGEN-01, MGEN-02, MGEN-03, RDIST-01, RDIST-02, MODL-01 (fixture-scoped),
MODL-03, MODL-04 (fixture parity).
Deferred (v2): MODL-02 (TabICL), real TabPFN v2 inference export, MODL-04-for-TabICL.

</domain>

<spike_outcomes>
## Spike Outcomes (2026-09-21)

**TabPFN v2 tensor contract — CONFIRMED** (`.planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md`):
- K = 5000 bins (`num_buckets`).
- Logits tensor `[n_test, K]` float32 (post-ensemble; raw single-estimator `[n_test, 1, K]`).
- Borders tensor `[K+1]` float32 — **checkpoint-fixed and HIGHLY NON-UNIFORM**
  (central bins ~0.0024 wide, outer bins ~67 wide, ratio ~28,000:1). Borders are
  NOT data-dependent; only the affine transform `raw_borders = znorm_borders *
  y_std + y_mean` is per-dataset.
- CRPS/quantiles MUST use the actual bucket widths — a uniform-bin assumption is
  wrong (this is why `yhat_dist` must carry borders).
- Real ONNX export blocked: data-dependent preprocessing + chunked attention fail
  `torch.export`. `tools/export_onnx` currently emits Google TabFM (scalar C=1),
  an incompatible contract.

**TabICL ONNX export — INFEASIBLE** on `tabicl 2.2.0`
(`.planning/spikes/SPIKE-tabicl-onnx-export.md`): data-dependent Stage-1 branches
(`SkippableLinear` if-branch; `num_classes = y_train.max().item()`). Deferred.

</spike_outcomes>

<decisions>
## Implementation Decisions

### Distribution output (RDIST-01/02)
- `output_mode='distribution'` adds a `yhat_dist STRUCT(logits DOUBLE[], borders
  DOUBLE[])` field to the predict detail STRUCT (alongside the existing yhat /
  yhat_score). Carrying borders in the struct keeps the non-uniform-bin
  information with each prediction so Phase 3 scoring rules are correct.
- Also emit `yhat_quantiles DOUBLE[]` at documented fixed levels (Claude's
  discretion on the exact level set; e.g. 0.1…0.9), computed from logits + borders.
- Point-estimate output stays the backward-compatible default; `yhat_dist` is
  absent (or empty) unless `output_mode='distribution'` AND the model emits a
  distribution. `RDIST-01`: the ORT run output can carry the distribution, empty
  when a model does not emit one.
- The C++ decoder handles **arbitrary K** (real K=5000 works at runtime); the
  committed fixture uses a small **K=16** to stay tiny and license-wall-clean.

### Model generalization (MGEN-01/02/03)
- New module `src/tabfm_profile_registry.cpp` (+ header): maps a manifest
  `preprocessing_profile` string → a C++ preprocessing function. Profiles
  self-register via static initializers; the existing `tabfm-v1` profile
  self-registers (MGEN-01). Model loading dispatches preprocessing through the
  registry; an unknown profile throws a named, actionable error naming the fix
  (MGEN-02).
- Model-output validation (shape/rank/class-count, and for distribution models
  the `[n,K]` logits + `[K+1]` borders contract) runs before decode for every
  family and rejects contract violations with a named error (MGEN-03; extends the
  existing P0 gap).

### tabpfn_v2 family (MODL-01 fixture-scoped, MODL-04 fixture parity)
- `tabpfn_v2` preprocessing profile is a **fixture-scoped minimal, correct-shape**
  profile — enough to exercise the registry + distribution path end-to-end on the
  fixture; documented as fixture-scoped (full-fidelity port deferred with the
  real-export work).
- Committed weight-free random-init ONNX fixture whose outputs match the confirmed
  contract (`[n,K]` logits + `[K+1]` borders, small K=16). `tools/parity`
  validates the fixture family's ONNX output contract (both distribution tensors)
  before the C++ decoder is trusted (MODL-04, fixture).
- A golden fixture verifies the distribution decode (mean/quantiles) from the
  known random-init logits+borders.

### License gate (MODL-03)
- Generic **manifest-license-keyed** gate: the manifest names its license id; the
  download is gated on a per-license `SET anofox_tabfm_accept_<id> = true`
  acceptance, extending the existing `anofox_tabfm_accept_hf_license` pattern
  generically so future families need no new settings code. Error message names
  the exact SET to run (SQL-API §5).

### Claude's Discretion
- Exact quantile level set; internal decoder math (softmax over logits → bin
  probabilities → CDF over non-uniform borders → mean/quantiles); the exact
  STRUCT/field wiring in the predict bind; fixture generation tooling location
  (extend `tools/make_fixture` or a small dedicated script). Grounded in the
  confirmed contract and existing predict/decode patterns.

</decisions>

<code_context>
## Existing Code Insights

### Reusable assets / seams
- Predict output shape: `LIST(STRUCT(cols, yhat, yhat_score, is_training[,
  proba]))`; `output_mode='detail'` adds `proba MAP(VARCHAR,DOUBLE)`
  (`src/include/tabfm_predict.hpp`, `src/tabfm_predict_agg.cpp`). Add the
  `yhat_dist` STRUCT field under `output_mode='distribution'` here.
- Decode step in `src/tabfm_engine.cpp` (classification argmax/softmax; regression
  inverse-transform) — the distribution decode (logits+borders → mean/quantiles)
  extends this.
- License gate: `anofox_tabfm_accept_hf_license` extension option
  (`src/tabfm_settings.cpp:77`) + manifest `license` field + gate check
  (`src/tabfm_weights.cpp:63,97,291`). Generalize keyed on manifest license id.
- Manifest `preprocessing_profile` field already parsed
  (`src/include/tabfm_manifest.hpp:82`). The registry consumes it.
- Preprocessing lives in `src/tabfm_preprocess.cpp` (WS-F); the registry is a new
  module that dispatches to profile functions.
- Fixtures committed in `test/fixtures/` with deterministic sha256 (CI verifies,
  never regenerates); tooling in `tools/make_fixture`, `tools/export_onnx` (S01
  TabFM exporter — NOT TabPFN v2), `tools/parity`.

### Scaffold-owned (coordinate — CLAUDE.md rule #2)
- `CMakeLists.txt` source list, `src/anofox_tabfm_extension.cpp`,
  `src/include/tabfm_registration.hpp`, `src/tabfm_settings.cpp` — batch edits for
  the new registry module + any new SET options.

### Conventions
- Full `anofox_tabfm_*` + short alias; telemetry once per user-facing function;
  errors name the fixing SET/CALL; red-green TDD; cpu-flavor-clean; no weight
  bytes (random-init fixtures only).
- DuckDB v1.5.4: aggregates do NOT implicitly cast arg types — see
  [[duckdb-aggregate-any-cast]] (relevant if any new aggregate is added).

</code_context>

<specifics>
## Specific Ideas

- The confirmed contract is the source of truth for RDIST/MODL-01: logits `[n,K]`
  f32, borders `[K+1]` f32 non-uniform, K dynamic (fixture K=16).
- `tools/parity` must assert the fixture's two distribution tensors match the
  declared contract before the C++ decoder is trusted.
- Everything must stay license-clean: fixture graphs are weight-free random-init;
  no TabPFN v2 / Google weight bytes in the repo.

</specifics>

<deferred>
## Deferred Ideas

- Real TabPFN v2 ONNX *inference* export (blocked: data-dependent preprocessing +
  chunked attention) — v2; needs a cleaned export path / custom traced backbone.
- TabICL as a first-class family (MODL-02) + its parity — v2; needs upstream PRs
  to `soda-inria/tabicl`.
- Full-fidelity tabpfn_v2 preprocessing port — with the real-export work.
- Proper scoring rules (CRPS/log-score/interval) + cross-model comparison — Phase
  3 (built against this phase's confirmed contract + fixture).

</deferred>
