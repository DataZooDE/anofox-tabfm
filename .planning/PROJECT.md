# anofox-tabfm

## What This Is

A DuckDB C++ extension that embeds tabular foundation models (TabPFN-style
in-context learners) via ONNX Runtime, exposing zero-shot classification and
regression as SQL — no Python, no training loops. This milestone extends it from
a single-model prediction surface into an **evaluation-and-comparison platform**:
composable, model-agnostic metric primitives and cross-validation in SQL, plus
first-class support for multiple foundation-model families so users can measure
and compare model quality on their own tables.

## Core Value

Users can trust and compare tabular-foundation-model predictions directly in
SQL — computing standard metrics and cross-validation on their own data, across
more than one model family — without leaving DuckDB.

## Requirements

### Validated

<!-- Inferred from existing code (see .planning/codebase/). These ship today. -->

- ✓ Zero-shot classification & regression as SQL (`tabfm_classify` / `tabfm_regress`) — existing
- ✓ Predict aggregate producing `yhat`, `yhat_score`, and per-class `proba` MAP (detail mode) — existing
- ✓ Weight download / cache / license gate from Hugging Face (`tabfm_download`, secrets) — existing
- ✓ Preprocessing pipeline (encode → filter → scale → outlier-clip), C++ port of TabFM logic — existing
- ✓ ONNX Runtime engine with cpu/cuda/rocm/coreml flavors + device discovery (`tabfm_devices`) — existing
- ✓ Custom model support via `SET anofox_tabfm_model_manifest` (single family: `tabfm-v1`) — existing
- ✓ Composable, model-agnostic **classification metrics** as SQL aggregates (accuracy, precision/recall/F1 with required `avg`, log-loss, ROC-AUC, ECE, confusion matrix) over `(actual, predicted[, proba])` — Phase 1
- ✓ Composable, model-agnostic **regression metrics** as SQL aggregates (RMSE, MAE, R², MAPE, median abs error) over `(actual, predicted)` — Phase 1
- ✓ **k-fold cross-validation** macro (`tabfm_cross_validate` + `tabfm_fold_assign`) — leakage-safe two-table predict, deterministic `hash(row_key, seed) % k` folds, per-fold + aggregate (mean ± std) output, safe target-identifier quoting — Phase 1
- ✓ **Generalized model seam** — self-registering `preprocessing_profile` registry + dispatch; unknown profiles fail with a named error; existing `tabfm-v1` self-registers (backward-compatible) — Phase 2
- ✓ **Regression predictive-distribution output** — `output_mode='distribution'` emits `yhat_dist STRUCT(logits[], borders[])` + `yhat_quantiles[]` decoded over model-provided **non-uniform** bin borders (FullSupportBarDistribution), backward-compatible default — Phase 2
- ✓ **Model-output validation** before decode (rank/shape/borders-length/logits-count) rejecting contract violations with named errors — Phase 2
- ✓ **TabPFN v2 as a first-class family (fixture-scoped)** — manifest + `tabpfn_v2` profile + committed **weight-free** random-init ONNX fixture matching the confirmed `[n,K]`/`[K+1]` contract + `tools/parity` contract validator — Phase 2
- ✓ **Generic per-family license gate** — manifest-license-keyed acceptance (`SET anofox_tabfm_accept_<id>`), backward-compatible with the existing HF-license flag — Phase 2

- ✓ **Proper scoring rules** — `tabfm_crps` (analytical closed-form over non-uniform bins), `tabfm_log_score` (NLL), `tabfm_interval_score` (configurable coverage); all bind-gated on distribution input with a named remedy — Phase 3
- ✓ **Cross-model comparison** — `tabfm_compare_models` runs the Phase-1 metrics + Phase-3 scores across model families on the user's own table (no bundled datasets) — Phase 3

### Active

<!-- This milestone. Hypotheses until shipped and validated. -->

All milestone requirements delivered. (See Deferred below for upstream-blocked items carried to v2.)

### Deferred (upstream-blocked — see `.planning/spikes/`)

- **Real TabPFN v2 ONNX *inference* export** — blocked (data-dependent preprocessing + chunked attention fail `torch.export`); Phase 2 ships the fixture-scoped family, real inference needs a cleaned export path
- **Add TabICL** — ONNX export infeasible on `tabicl 2.2.0` (data-dependent Stage-1 branches); needs upstream PRs

### Out of Scope

- **Bundled / auto-downloaded benchmark dataset suite** (ScoringBench-style ~104-dataset leaderboard) — comparison runs on the *user's own tables*; keeps the extension lean and avoids dataset-licensing burden
- **Google TabFM weight bytes in the repo** — license wall (S06); fixtures stay random-init, real graphs stay weight-free stubs
- **GPU-only code paths in the cpu flavor** — community-extension eligibility (PR #2181) requires cpu build to have zero GPU dependencies
- **Training / fine-tuning** — these are in-context (zero-shot) models; training is not the product
- **Reproducing ScoringBench's ranking machinery** (`autorank`, critical-difference diagrams) inside the extension — statistical ranking is a downstream/notebook concern, not SQL

## Context

- **Brownfield.** Mature codebase mapped in `.planning/codebase/` (2026-09-19). Layered architecture with strict module-per-file ownership; see `ARCHITECTURE.md`.
- **No evaluation exists today.** Confirmed by grep: the extension is a prediction surface only. Metrics, CV, and scoring are all net-new. Classification emits a full `proba` MAP (a foundation for calibration/log-loss); **regression emits only a point estimate** — the key gap for proper scoring rules.
- **One model family today.** Only `tabfm-v1` (`google/tabfm-1.0.0-pytorch`); `preprocessing_profile` is effectively hardcoded. Custom manifests exist but new families need their own preprocessing profile + graph handling.
- **ScoringBench** (github.com/jonaslandsgesell/ScoringBench) is the inspiration: probabilistic-regression benchmark using proper scoring rules (CRPS, log-score, interval/energy/variogram) with 5-fold CV; models plug in via `ProbabilisticWrapper` (TabPFN as reference). We adopt its *evaluation philosophy*, not its dataset suite.
- **Community-extension submission** (PR #2181 to duckdb/community-extensions) is in flight; ORT must link statically (vcpkg `ort-vcpkg`). New work must not jeopardize eligibility.
- **Tooling seams** exist for model onboarding: `tools/export_onnx`, `tools/make_fixture`, `tools/parity` (WS-A, uv projects); `vendor/tabfm` is the upstream Apache-2.0 submodule (never edited).
- Known correctness debt (P0/P1) that touches this work: missing model-output validation, NaN-in-features, datetime precision, macro identifier interpolation — see `CONCERNS.md`.

## Constraints

- **Tech stack**: DuckDB v1.5.4 (pinned submodule), ONNX Runtime 1.23.2, C++ extension via extension-ci-tools v1.5-variegata — new code follows existing module conventions (CLAUDE.md)
- **Community-extension eligibility**: cpu flavor must have zero GPU code; metric/eval code must be flavor-independent and license-clean
- **License wall**: no Google/vendor weight bytes anywhere in the repo; fixtures random-init (S06)
- **Model-agnostic metrics**: eval primitives operate on plain `(actual, predicted)` columns so they're useful independent of tabfm output
- **File ownership**: one module = one `src/tabfm_*.cpp` (+ headers); scaffold-owned shared files require coordination (CLAUDE.md rule #2)
- **TDD**: red-green — failing sqllogictest/Catch2 test first, then implement; error-path tests are first-class
- **Telemetry**: every user-facing function calls `CaptureFunctionExecution` once per execution
- **Dependency**: proper scoring rules for regression are blocked until a model emits predictive distributions (TabPFN v2 provides this natively)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Evaluation framework before adding models | Need a way to measure quality before it's meaningful to compare families | ✓ Phase 1 metrics/CV, then Phase 2 model seam |
| Evaluation delivered phased (point metrics + CV → proper scoring rules) | Point metrics are achievable with current outputs (fast win); scoring rules need regression distributions | ✓ Phase 1: point metrics + CV shipped |
| Composable, model-agnostic metric primitives (not one monolithic eval macro) | Most DuckDB-idiomatic; reusable outside tabfm; CV macro composes them | ✓ Phase 1: 12 metric aggregates + confusion-matrix/CV macros, all `{ANY,…}` model-agnostic |
| Metric aggregates accept ANY-typed labels (not VARCHAR-only) | DuckDB v1.5.4 does not implicitly cast INTEGER→VARCHAR for aggregate args; VARCHAR-only registration broke integer/float label columns | ✓ Phase 1: registered `{ANY, ANY}` with type-safe `Value` comparison |
| Comparison on user's own tables, no bundled datasets | Keeps extension lean, avoids dataset licensing, consistent with community-extension goal | — Pending |
| Proper scoring rules gated on distribution output | Regression previously emitted only a point estimate; PSR need a predictive distribution | ✓ Phase 2 shipped distribution output → Phase 3 CRPS/log-score/interval built + golden-tested against the confirmed contract via the fixture |
| Prioritize TabPFN v2 + TabICL, plus generalized custom-model support | TabPFN v2 is the field reference and brings regression distributions; TabICL scales; generalization future-proofs onboarding | ⚠ Phase 2: generalization + TabPFN v2 distribution shipped fixture-scoped; real TabPFN v2 export + TabICL deferred (ONNX export blocked upstream — spikes) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-22 after Phase 3 (milestone complete)*
