# Feature Research

**Domain:** In-SQL tabular-ML evaluation framework (metrics, cross-validation, proper scoring rules) inside a DuckDB C++ extension
**Researched:** 2026-09-19
**Confidence:** MEDIUM (web sources, cross-checked against ScoringBench paper and PROJECT.md)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features a tabular-ML evaluation toolkit must have. Missing any of these makes the evaluation surface feel broken.

| Feature | Why Expected | Distribution Required? | Complexity | Notes |
|---------|--------------|------------------------|------------|-------|
| **Accuracy** | Baseline classification metric; first thing every user asks | No — point predicted class | LOW | `tabfm_accuracy(actual, predicted)` scalar agg. Binary and multiclass. |
| **Precision / Recall (per class + averaged)** | F1 meaningless without them; confusion matrix foundation | No — point predicted class | LOW | Requires accumulating TP/FP/FN per class in state. `macro`, `micro`, `weighted` average modes. |
| **F1 score (macro / micro / weighted)** | De-facto summary metric in ML papers and Kaggle; scikit-learn's `classification_report` is the mental model | No — point predicted class | LOW | Depends on precision+recall state; can share one aggregate returning a STRUCT. |
| **Log-loss (cross-entropy)** | Penalizes overconfident wrong predictions; only metric that uses `proba` in phase-1; rewards calibrated probabilities | Yes — class probabilities (proba MAP) | LOW | Existing `yhat_score` / `proba` map from `tabfm_classify` is the input. `-mean(log(p[actual]))`. |
| **RMSE** | Universal regression benchmark metric; MSE is the training objective for most regressors | No — point predicted value | LOW | `sqrt(avg((actual - predicted)^2))`. Standard agg. |
| **MAE** | Complement to RMSE; linear, interpretable in original units | No — point predicted value | LOW | `avg(abs(actual - predicted))`. |
| **R² (coefficient of determination)** | Scale-free; the standard "goodness of fit" number quoted in papers | No — point predicted value | LOW | Requires accumulating sum of squared residuals and total variance. Single-pass with online Welford OK. |
| **Confusion matrix** | Required to compute per-class metrics; diagnostic for class imbalance | No — point predicted class | MEDIUM | Returns a 2-D result. Most natural as a table-valued function `tabfm_confusion_matrix(actual, predicted)`. Complexity is in the multi-class dense representation, not the math. |
| **ROC-AUC (binary; OvR multiclass)** | Standard classification ranking metric; used when threshold matters less than ranking | Yes — class probabilities (proba MAP) | MEDIUM | Requires sorting by probability; AUC via trapezoidal rule. One-vs-Rest for multiclass. Needs all rows in state before finalize (O(n log n)). |

### Differentiators (Competitive Advantage)

Features that make anofox-tabfm's evaluation surface more powerful than a hand-rolled Python script. These align with the ScoringBench philosophy and the project's core value of in-SQL model comparison.

| Feature | Value Proposition | Distribution Required? | Complexity | Notes |
|---------|-------------------|------------------------|------------|-------|
| **PR-AUC (Precision-Recall AUC)** | Better than ROC-AUC on imbalanced datasets; standard in biomedical ML | Yes — class probabilities (proba MAP) | MEDIUM | Same state structure as ROC-AUC; sort by proba, accumulate precision/recall curve. |
| **Per-class metric breakdown** | `GROUP BY actual` gives per-class precision/recall/F1 without extra code — only possible because metrics are composable aggregates | No | LOW | Emerges for free from composable aggregate design. Document as a usage pattern; no extra implementation. |
| **k-fold cross-validation macro (`tabfm_cv`)** | Orchestrates fold assignment + predict + metric aggregation in SQL; eliminates the Python notebook step | No | HIGH | Fold assignment: `NTILE(k) OVER (PARTITION BY label ORDER BY RANDOM())` for stratified. Macro calls `tabfm_classify`/`tabfm_regress` per fold, unions results, aggregates metrics. Complexity: macro identifier interpolation (see CONCERNS.md P1 bug), fold-level row tagging. |
| **Stratified fold assignment (`tabfm_assign_folds`)** | Ensures class balance per fold; critical for imbalanced datasets | No | LOW | Window function: `(ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) - 1) % k` as fold id. Expose as a scalar or table macro so users can inspect the fold mapping. |
| **MedAE (Median Absolute Error)** | Robust regression metric; complement to RMSE+MAE for outlier-heavy data | No | LOW | `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs(actual - predicted))`. Can be delegated to DuckDB built-in if exposed via macro. |
| **MAPE (Mean Absolute Percentage Error)** | Business-interpretable %; requested by practitioners with non-technical stakeholders | No | LOW | `avg(abs((actual - predicted) / actual))`. Must guard against `actual = 0` (emit NULL or error with clear message). |
| **ECE — Expected Calibration Error** | Scalar calibration summary; measures if `proba[class] = 0.7` actually wins 70% of the time | Yes — class probabilities (proba MAP) | MEDIUM | Bin predictions into B buckets (10 is standard), compute weighted `|mean_confidence - accuracy|`. Requires full proba vector. Complements log-loss. |
| **CRPS (Continuous Ranked Probability Score)** | The flagship proper scoring rule for probabilistic regression; strictly proper — rewards honest uncertainty | Yes — FULL predictive distribution (quantiles or PMF grid) | HIGH | **BLOCKED until TabPFN v2 integration** (which outputs distribution natively). Formula: `mean((F(y) - 1{y <= actual})^2)` over the CDF. Requires quantile vector in prediction output. |
| **Log-score (regression)** | Proper scoring rule; strict; penalizes overconfident distributions harshly | Yes — FULL predictive distribution | HIGH | **BLOCKED until TabPFN v2**. `-log(p(actual))` where p is the predicted density. Requires density evaluation at the observed point. |
| **Interval score** | Penalizes both overconfident narrow intervals AND underconfident wide intervals | Yes — quantile pairs (e.g., q5/q95) | MEDIUM | **BLOCKED until TabPFN v2**. `(upper - lower) + (2/alpha) * max(0, lower - actual) + (2/alpha) * max(0, actual - upper)`. Input: two quantile columns + alpha. |
| **Coverage at alpha% (calibration diagnostic)** | Empirical check: does the 90% interval actually contain actual ~90% of the time? | Yes — quantile pairs | LOW | **BLOCKED until TabPFN v2**. `avg(actual BETWEEN q_lo AND q_hi)`. Very simple once quantile output exists. |
| **Cross-model comparison** | Run the metric aggregates over unioned predictions from multiple models; the killer feature for model selection | No | MEDIUM | Emerges from composable aggregate design + model label column in prediction output. No extra C++ required; document as SQL pattern. |

### Anti-Features (Deliberately NOT Build)

These are features that seem related but create problems. Cross-checked against PROJECT.md Out of Scope.

| Feature | Why Requested | Why Not | What to Do Instead |
|---------|---------------|---------|-------------------|
| **Bundled benchmark dataset suite** | "I want to see how the model does on 100 standard datasets" | Dataset licensing burden; bloats the extension; conflicts with community-extension goal; out-of-scope per PROJECT.md | Users bring their own tables; `tabfm_cv` runs on them. Document links to OpenML for users who want standard datasets |
| **autorank / critical-difference diagrams in-engine** | "Tell me which model is statistically better" | Statistical ranking is a downstream/notebook concern; CD diagrams are visual artifacts; in-engine they require R/Python dependencies; out-of-scope per PROJECT.md | Emit per-fold metric tables as DuckDB relations; users export to Python/R for `autorank`, `scipy.stats.wilcoxon`, etc. |
| **Post-hoc calibration (Platt scaling, isotonic regression)** | "Calibrate the probabilities for me" | Training/fitting step; violates "no training" principle of in-context models; adds significant model lifecycle complexity | Surface ECE so users can see miscalibration; they can apply calibration externally |
| **Training / fine-tuning** | "Can I update the model on my data?" | Antithetical to zero-shot in-context learning; no gradient infrastructure | Document that TabPFN/TabICL are in-context learners: the context IS the training set |
| **Multivariate regression targets** | "I have multiple regression outputs" | Not supported by current model families or ScoringBench (listed as future work); adds major tensor shape complexity | Evaluate each target column independently; document as limitation |
| **Energy score / variogram score** | ScoringBench includes them | Require joint distribution over multiple targets (multivariate); not applicable to scalar regression | Implement CRPS + interval score first (scalar); revisit if multivariate support is added |
| **Streaming / online metric computation** | "Update metrics as rows arrive" | DuckDB aggregates finalize at query end; streaming requires stateful connections and an event loop; out of scope for a SQL extension | Use DuckDB's `tabsample` or window functions for large-dataset approximations |
| **In-extension leaderboard / ranking tables** | "Show me a ranked comparison table" | Static HTML/markdown artifacts cannot live in a SQL extension; ranking logic belongs in the query layer | Users write `ORDER BY metric DESC` on the metric aggregate output; no special support needed |

---

## Feature Dependencies

```
[Confusion matrix]
    └──required by──> [Precision / Recall per class]
                          └──required by──> [F1 score (macro/micro/weighted)]
                          └──required by──> [Per-class breakdown (GROUP BY pattern)]

[Class probabilities (proba MAP) — existing output]
    └──required by──> [Log-loss]
    └──required by──> [ROC-AUC]
    └──required by──> [PR-AUC]
    └──required by──> [ECE]

[Point predicted value — existing output]
    └──required by──> [RMSE, MAE, R², MedAE, MAPE]

[Stratified fold assignment]
    └──required by──> [k-fold CV macro (tabfm_cv)]
                          └──required by──> [Cross-model comparison via CV]

[TabPFN v2 integration — BLOCKED phase]
    └──required by──> [CRPS]
    └──required by──> [Log-score (regression proper scoring rule)]
    └──required by──> [Interval score]
    └──required by──> [Coverage at alpha%]

[RMSE + MAE + R²] ──sufficient for──> [Phase-1 regression eval (no distribution)]
[CRPS] ──supplements──> [RMSE + MAE + R²] when distribution available

[k-fold CV macro] ──uses──> [tabfm_classify / tabfm_regress] (existing)
[k-fold CV macro] ──uses──> [metric aggregates] (phase-1)
```

### Dependency Notes

- **ROC-AUC / PR-AUC require proba MAP**: The existing `tabfm_classify` with `detail := true` already emits `proba` as a MAP column. These metrics are unlocked by phase-1 — no new model work needed.
- **Proper scoring rules are hard-blocked on distribution output**: `tabfm-v1` (current) emits only a point estimate for regression. CRPS, log-score, and interval score cannot be computed. This unblocks only after TabPFN v2 is integrated.
- **ECE straddles the phases**: It uses `proba` from classification (already available), so it is implementable in phase-1 for classification. Regression calibration requires distribution output (phase-2).
- **CV macro depends on macro identifier interpolation fix**: CONCERNS.md flags the P1 bug in `tabfm_macros.cpp:91` where target column names are not quoted. The CV macro must either fix this or work around it before shipping.
- **Per-class breakdown is not a separate feature**: It emerges from GROUP BY over composable metric aggregates. Document as a usage pattern only.

---

## MVP Definition

### Phase 1: Point-estimate metrics + CV (launch with)

These are achievable with current model outputs (point class prediction + proba MAP for classification; point value for regression). No model changes required.

- [ ] **Classification: Accuracy** — simplest table-stakes metric; validates the aggregate skeleton
- [ ] **Classification: Precision / Recall / F1 (macro/micro/weighted)** — the core classification evaluation trio
- [ ] **Classification: Log-loss** — uses existing `proba` MAP; rewards calibrated probabilities
- [ ] **Classification: ROC-AUC (OvR)** — uses existing `proba`; ranking metric
- [ ] **Classification: Confusion matrix (table-valued function)** — diagnostic foundation
- [ ] **Regression: RMSE, MAE, R²** — the minimum regression eval trio; achievable with current point output
- [ ] **Regression: MAPE, MedAE** — low-effort complements; MAPE needs zero-guard
- [ ] **k-fold CV macro (`tabfm_cv`)** — orchestration in SQL; depends on macro interpolation fix
- [ ] **Stratified fold assignment helper (`tabfm_assign_folds`)** — prerequisite for CV macro

### Phase 2: Calibration + proper scoring rules (add after TabPFN v2 integration)

Blocked until regression predictive distribution is available in the extension.

- [ ] **Classification: ECE** — calibration scalar; uses classification proba (could actually ship in phase-1)
- [ ] **Classification: PR-AUC** — low extra cost once ROC-AUC state exists
- [ ] **Regression: CRPS** — flagship proper scoring rule; requires quantile vector output
- [ ] **Regression: Interval score** — requires quantile pair output; lower complexity than CRPS
- [ ] **Regression: Coverage at alpha%** — trivially simple once quantile pairs exist
- [ ] **Regression: Log-score** — requires density evaluation; most complex of the three

### Deferred / Future Consideration

- [ ] **Energy score / variogram score** — multivariate targets; not in scope for v1
- [ ] **Weighted CRPS (wCRPS)** — asymmetric cost weighting; advanced use case
- [ ] **Streaming metric updates** — requires stateful connection; major architectural change

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Accuracy | HIGH | LOW | P1 |
| Precision / Recall / F1 | HIGH | LOW | P1 |
| Log-loss | HIGH | LOW | P1 |
| RMSE | HIGH | LOW | P1 |
| MAE | HIGH | LOW | P1 |
| R² | HIGH | LOW | P1 |
| Confusion matrix | HIGH | MEDIUM | P1 |
| ROC-AUC | HIGH | MEDIUM | P1 |
| k-fold CV macro | HIGH | HIGH | P1 |
| Stratified fold assignment | MEDIUM | LOW | P1 (prereq for CV) |
| MAPE | MEDIUM | LOW | P2 |
| MedAE | MEDIUM | LOW | P2 |
| PR-AUC | MEDIUM | MEDIUM | P2 |
| ECE | MEDIUM | MEDIUM | P2 |
| CRPS | HIGH | HIGH | P2 (blocked on TabPFN v2) |
| Interval score | MEDIUM | MEDIUM | P2 (blocked on TabPFN v2) |
| Coverage at alpha% | LOW | LOW | P2 (blocked on TabPFN v2) |
| Log-score | MEDIUM | HIGH | P3 (blocked on TabPFN v2) |
| Energy score | LOW | HIGH | P3 (multivariate scope) |

**Priority key:**
- P1: Must have for phase-1 launch
- P2: Add in phase-2 (after TabPFN v2 or as polish to phase-1)
- P3: Nice to have, future milestone

---

## Competitor Feature Analysis

| Feature | scikit-learn | mlflow / wandb | Our Approach |
|---------|--------------|----------------|--------------|
| Classification metrics | Functions over arrays | Logged scalars | SQL aggregate functions over `(actual, predicted[, proba])` — composable with GROUP BY |
| Regression metrics | Functions over arrays | Logged scalars | Same composable aggregate pattern |
| Cross-validation | `cross_val_score` Python | External loop | SQL macro; fold assignment in SQL; no Python required |
| Proper scoring rules | `properscoring` library | Not native | Native SQL aggregates (phase-2); input = quantile columns from TabPFN v2 |
| Calibration | `CalibratedClassifierCV`, `calibration_curve` | Not native | ECE as aggregate (classification); calibration curve = separate table-valued function |
| Model comparison | External; Pandas DataFrame | Experiment tracking UI | SELECT + GROUP BY over metric aggregates with model_name column; no extra infrastructure |

---

## Implementation Notes for DuckDB Aggregate Design

Each metric is a `DuckDB::AggregateFunction` registered as `tabfm_<metric_name>(actual, predicted[, proba][, average_mode])`:

- **State struct** accumulates sufficient statistics (e.g., TP/FP/FN counts per class for F1; sum of squared errors for RMSE; sorted score list for AUC).
- **Initialize / Update / Combine / Finalize** pattern — `Combine` enables parallel aggregation; this is critical because DuckDB can parallelize aggregation across threads.
- **Return type**: scalars return `DOUBLE`; confusion matrix returns a `STRUCT` or table-valued function.
- **Averaging mode**: pass as `VARCHAR` parameter (`'macro'`, `'micro'`, `'weighted'`) or separate registered functions per mode.
- **Composability**: aggregates naturally compose with `GROUP BY fold_id` for per-fold CV results and `GROUP BY model_name` for model comparison — no extra C++ required.
- **AUC state size**: ROC-AUC requires storing all `(proba, actual)` pairs in the state for sorting at finalize — O(n) memory. For very large evaluation sets this is a concern; document a row-count limit recommendation (~100K rows safe).

---

## Sources

- ScoringBench paper (arxiv.org/abs/2603.29928) — proper scoring rules, CV setup, model output requirements
- scikit-learn documentation — classification_report, ROC-AUC, calibration_curve reference patterns
- TabPFN v2 / Prior Labs docs (docs.priorlabs.ai) — distribution output format, quantile interface
- TabICL paper (arxiv.org/abs/2502.05564) — scale-to-large-data classification, 999-quantile regression output
- DuckDB GitHub discussions — AggregateFunction C++ API patterns
- Analytics Vidhya / NVIDIA — regression metric reference
- PROJECT.md Out of Scope section — anti-feature cross-check (bundled datasets, autorank, training)
- CONCERNS.md — macro interpolation P1 bug affects CV macro design

---

*Feature research for: anofox-tabfm evaluation framework milestone*
*Researched: 2026-09-19*
