# Project Research Summary

**Project:** anofox-tabfm — evaluation framework + multi-model support
**Domain:** DuckDB C++ extension embedding tabular foundation models via ONNX Runtime
**Researched:** 2026-09-19
**Confidence:** MEDIUM-HIGH

## Executive Summary

This milestone extends a mature DuckDB extension (zero-shot tabular classification/regression over one model family, `tabfm-v1`) into an evaluation-and-comparison platform. Research across four dimensions (stack, features, architecture, pitfalls) converges on a clear, dependency-driven build order with one dominant gate.

The gate: **point-estimate metrics and k-fold cross-validation are achievable today with existing model outputs** — classification already emits `yhat` + a per-class `proba` MAP, and regression emits a point `yhat`. That makes the whole point-metrics + CV layer a fast, low-risk first win with zero changes to the core predict path. **Proper scoring rules (CRPS, log-score, interval score) are hard-gated** on a model emitting a regression *predictive distribution* — which `tabfm-v1` does not, but **TabPFN v2's bar-distribution head does** (per-bin logits over a fixed grid + bin borders). So the two user-requested threads reinforce each other: adding TabPFN v2 is what unlocks ScoringBench-style probabilistic evaluation for regression.

Key risks are (1) **CV data leakage** — the existing predict aggregate fits preprocessing statistics over its whole context window, so a "full table then filter by fold" CV design silently leaks; the existing two-table (train context + NULL-labelled test) form must be used per fold; (2) **metric correctness subtleties** — multiclass AUC averaging modes, ROC-AUC tie handling, PR-AUC (average-precision, not trapezoidal), log-loss clipping — all invisible without golden tests against scikit-learn; and (3) **model onboarding unknowns** — no official ONNX artifacts exist for TabPFN v2 or TabICL, so tensor contracts must be validated via `tools/export_onnx` + `tools/parity` before any C++ decoder is written, and TabICL's three-stage architecture needs a dynamo-traceability spike. All are preventable with TDD and spikes sequenced ahead of implementation.

## Key Findings

### Recommended Stack

New evaluation code lands as **DuckDB C++ aggregate functions** (verified against DuckDB core_functions API) plus **SQL table macros** for orchestration — model-agnostic, operating on plain `(actual, predicted[, proba])` columns. Model onboarding reuses the existing `tools/export_onnx` pipeline (`torch.dynamo`, opset 18, `ExportWrapper` pattern) and the already-present `preprocessing_profile` manifest seam. Proper-scoring-rule formulas are taken from primary sources (Gneiting & Raftery 2007; ScoringBench arXiv). See `STACK.md`.

**Core technologies:**
- DuckDB C++ aggregate API — metric primitives (accuracy/F1/AUC/log-loss/RMSE/MAE/R²) — idiomatic, composable, model-agnostic
- DuckDB SQL table macros — k-fold CV orchestration (fold assignment + per-fold/aggregate output) — no C++ executor fighting
- `tools/export_onnx` (torch.dynamo, opset 18) — reusable for TabPFN v2; adapt `ExportWrapper` + tensor map per family
- Energy-score identity for CRPS over bar distributions — exact for finite support, simpler than full integral
- CRPS + log-score + interval score cover the ScoringBench core; **defer** CRLS (log singularity) and variogram (multivariate, irrelevant to scalar regression)

### Expected Features

See `FEATURES.md`. The composable-aggregate design gives per-class breakdowns (`GROUP BY label`) and per-fold results (`GROUP BY fold_id`) for free — usage patterns, not separate features.

**Must have (table stakes) — achievable with today's outputs:**
- Classification: accuracy, precision/recall, F1 (micro/macro/weighted), ROC-AUC, log-loss, confusion matrix
- Regression: RMSE, MAE, R² (MAPE/MedAE low-effort complements)
- k-fold cross-validation macro (deterministic, leakage-safe)

**Should have (competitive):**
- Expected Calibration Error (ECE) for classification — uses existing `proba`, can front-load into phase 1
- Proper scoring rules: CRPS, log-score, interval score — gated on regression distributions
- Cross-model comparison over user tables

**Defer / out of scope:**
- Bundled benchmark dataset suite, `autorank`/critical-difference diagrams in-engine, post-hoc calibration fitting, training/fine-tuning (all confirmed against PROJECT.md Out of Scope)

### Architecture Approach

Four self-contained new modules integrate at existing seams without redesigning the pipeline (see `ARCHITECTURE.md`). Two scaffold-owned files (`tabfm_registration.hpp`, `anofox_tabfm_extension.cpp`) need coordinated edits — batch all new `Register*` declarations in one PR at phase-1 start with stub bodies.

**Major components:**
1. `tabfm_metrics.cpp` — model-agnostic metric aggregates; no engine header dependency
2. `tabfm_cv.cpp` — k-fold CV table macro; fold loop unrolled into `UNION ALL` subqueries; deterministic hash-based fold assignment; composes existing two-table predict form
3. `tabfm_profile_registry.cpp` — singleton mapping `preprocessing_profile` → C++ preprocess function; `tabfm_v1_minimal` self-registers; single dispatch call-site change in `tabfm_engine.cpp`
4. `tabfm_scoring.cpp` — CRPS/log-score/interval score; bind-time gated on distribution-type input, fails with named remedy on point input
- Distribution path: extend `TabFMRunOutput` with `bar_logits`/`bar_shape`/quantile fields (empty = not emitted); decode conditionally emits `yhat_dist DOUBLE[]` / `yhat_quantiles MAP` under `output_mode='distribution'`

### Critical Pitfalls

Top items from `PITFALLS.md`:

1. **CV data leakage** — full-table-then-filter fits preprocessing stats on test rows. *Avoid:* use the existing two-table predict form (training rows as context, test rows as NULL-labelled queries), one call per fold. Highest-risk correctness item.
2. **Multiclass AUC ambiguity** — averaging mode (OvR/OvO/weighted) must be an explicit required parameter (no silent default); micro-averaging is undefined for OvO; ties need the rank-sum form. *Avoid:* golden tests vs sklearn on a 3-class fixture.
3. **PR-AUC interpolation** — trapezoidal is provably wrong on imbalanced data. *Avoid:* implement average precision (step function), matching `average_precision_score`.
4. **TabPFN v2 bar-distribution decode** — must export *both* per-bin logits and bin boundaries as named ONNX outputs; assuming uniform bins yields plausible-but-wrong quantiles/CRPS. *Avoid:* `tools/parity` validates both tensors before the C++ decoder exists.
5. **Per-family licensing + P0 output validation** — TabPFN v2 has Apache-2.0 code but a separate (non-commercial for v2.5+) weights license; reusing the existing HF-license flag is legally ambiguous → separate per-family gate. Extend the existing P0 model-output shape validation to the new tensor contract before wiring predictions.

## Implications for Roadmap

Research suggests **5 phases** (granularity is set to *coarse* in config, so the roadmapper may merge these into 3–5 broader phases). Point-metrics + CV first; proper scoring rules last, gated on TabPFN v2.

### Phase 1: Point-Estimate Metrics + k-fold CV
**Rationale:** Achievable with current outputs; highest immediate user value; locks the composable-aggregate foundation everything else builds on.
**Delivers:** Classification metrics (accuracy, P/R/F1, log-loss, ROC-AUC, confusion matrix, ECE) + regression metrics (RMSE, MAE, R², MAPE, MedAE) + leakage-safe k-fold CV macro.
**Addresses:** All table-stakes evaluation features.
**Avoids:** CV leakage (two-table form), AUC averaging ambiguity (explicit param + golden tests), zero-division edge cases; fix the P1 macro identifier-interpolation bug (`tabfm_macros.cpp:91`) here.

### Phase 2: Preprocessing Profile Registry + Model Generalization
**Rationale:** Small refactor of an existing seam; unblocks parallel model onboarding.
**Delivers:** Singleton profile registry; preprocessing dispatch rerouted through it; `tabfm_v1_minimal` self-registers.
**Implements:** `tabfm_profile_registry.cpp`; one coordinated WS-C edit in `tabfm_engine.cpp`.

### Phase 3: Regression Distribution Output (Decode + SQL Surface)
**Rationale:** Minimal, backward-compatible; the enabler for proper scoring rules.
**Delivers:** `TabFMRunOutput` distribution fields; bar-distribution decode (softmax + quantile extraction); `output_mode='distribution'` emitting `yhat_dist` / `yhat_quantiles`.
**Uses:** TabPFN v2 bar-distribution contract (must be finalized by a WS-A spike first).

### Phase 4: Multi-Model Onboarding — TabPFN v2 + TabICL
**Rationale:** Longest pole. TabPFN v2 brings regression distributions; TabICL scales classification.
**Delivers:** `tabpfn_v2` + `tabicl_v2` preprocessing profiles, manifests, weight-free ONNX fixtures, per-family license gate.
**Avoids:** Preprocessing mismatch (mandatory `tools/parity` per family), P0 output-validation gap (extend to new contracts). Recommend a **TabICL export spike immediately** (three-stage dynamo traceability is unconfirmed).

### Phase 5: Proper Scoring Rules
**Rationale:** Formulas are HIGH-confidence; implementation is straightforward once distributions exist.
**Delivers:** CRPS, log-score, interval score aggregates (bind-gated on distribution input); cross-model comparison over user tables.

### Phase Ordering Rationale

- Phase 1 depends on nothing → ship first for immediate value and to lock the aggregate foundation.
- Proper scoring rules (5) depend on distribution output (3), which depends on TabPFN v2's tensor contract (4/spike) and the profile registry (2).
- Phases 2 and the TabICL export spike can run in parallel with Phase 1.
- ECE can be front-loaded into Phase 1 (uses existing `proba`).

### Research Flags

Phases likely needing deeper research/spikes during planning:
- **Phase 3/4 (TabPFN v2):** exact ONNX tensor names/shapes for logits **and** bin borders; K (bin count) — needs a checkpoint inspection + `tools/export_onnx` run.
- **Phase 4 (TabICL):** whether the three-stage Transformer is `torch.dynamo`-traceable with dynamic shapes; whether upstream supports ONNX export at all — **spike immediately**.
- **Phase 4 (licensing):** per-family license-acceptance flag design before wiring TabPFN v2 downloads.

Phases with standard patterns (lighter research):
- **Phase 1 metrics:** scikit-learn is the reference; well-documented (still needs golden tests).
- **Phase 5 scoring rules:** published formulas (Gneiting & Raftery; ScoringBench), verified.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Scoring-rule formulas HIGH; DuckDB aggregate API HIGH; ONNX export paths MEDIUM (no official artifacts) |
| Features | MEDIUM | Standard ML feature set; scikit-learn parity; distribution-gating well understood |
| Architecture | MEDIUM-HIGH | Integrates cleanly with mapped layers; scaffold coordination identified |
| Pitfalls | HIGH | Metrics/CV/scoring-rule pitfalls well-documented; model-onboarding pitfalls MEDIUM |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **TabPFN v2 ONNX tensor contract** (logits + bin borders, K): validate via `tools/export_onnx` + `tools/parity` before Phase 3 decode is written.
- **TabICL ONNX export feasibility**: dynamo-traceability spike before committing Phase 4 scope/timeline.
- **CV leakage test design**: write the leakage-detecting golden test before Phase 1 CV ships (non-negotiable).
- **Per-family license gate**: design before TabPFN v2 download is wired.
- **AUC state memory**: O(N) row-storing state may hit limits on large eval sets — document a row-count recommendation.

## Sources

### Primary (HIGH confidence)
- Gneiting & Raftery 2007, "Strictly Proper Scoring Rules, Prediction, and Estimation" — CRPS, log-score, interval score definitions
- ScoringBench (github.com/jonaslandsgesell/ScoringBench + arXiv) — proper-scoring-rule benchmark methodology, 5-fold CV, ProbabilisticWrapper pattern
- scikit-learn metrics docs — averaging modes, average-precision vs trapezoidal, log-loss clipping
- DuckDB core_functions README — C++ aggregate function API

### Secondary (MEDIUM confidence)
- Prior Labs TabPFN v2 docs / issues / PRs — bar-distribution regression head; Apache-2.0 code + separate weights license
- Existing `tools/export_onnx` (torch.dynamo, opset 18) — reusable export pattern
- anofox-tabfm codebase map (`.planning/codebase/`) — existing seams, `preprocessing_profile` field, P0 output-validation gap

### Tertiary (LOW confidence)
- TabICL paper — three-stage architecture; no published ONNX export path (spike required)
- TabPFN v3 (post-cutoff) — tensor contract unknown; likely shares bar-distribution pattern

---
*Research completed: 2026-09-19*
*Ready for roadmap: yes*
