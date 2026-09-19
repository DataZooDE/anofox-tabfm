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

### Active

<!-- This milestone. Hypotheses until shipped and validated. -->

**Evaluation framework (first — phased):**
- [ ] Composable, model-agnostic **classification metrics** as SQL aggregates (accuracy, F1, AUC, log-loss, …) over `(actual, predicted[, proba])`
- [ ] Composable, model-agnostic **regression metrics** as SQL aggregates (RMSE, MAE, R², …) over `(actual, predicted)`
- [ ] **k-fold cross-validation** macro that orchestrates predict + metrics across folds and returns per-fold and aggregate results
- [ ] **Proper scoring rules** (CRPS, log-score, interval score) + calibration — *gated on regression emitting predictive distributions/quantiles*

**Multi-model support (second):**
- [ ] **Generalize model support** — make `preprocessing_profile` + graph handling pluggable so families beyond `tabfm-v1` are first-class (not just raw custom manifests)
- [ ] **Add TabPFN v2** — including its native regression predictive distribution (unlocks proper scoring rules)
- [ ] **Add TabICL** — classification, scales to larger tables
- [ ] **Cross-model comparison** path that runs the eval primitives across models **on the user's own tables**

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
| Evaluation framework before adding models | Need a way to measure quality before it's meaningful to compare families | — Pending |
| Evaluation delivered phased (point metrics + CV → proper scoring rules) | Point metrics are achievable with current outputs (fast win); scoring rules need regression distributions | — Pending |
| Composable, model-agnostic metric primitives (not one monolithic eval macro) | Most DuckDB-idiomatic; reusable outside tabfm; CV macro composes them | — Pending |
| Comparison on user's own tables, no bundled datasets | Keeps extension lean, avoids dataset licensing, consistent with community-extension goal | — Pending |
| Prioritize TabPFN v2 + TabICL, plus generalized custom-model support | TabPFN v2 is the field reference and brings regression distributions; TabICL scales; generalization future-proofs onboarding | — Pending |

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
*Last updated: 2026-09-19 after initialization*
