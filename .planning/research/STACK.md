# Stack Research

**Domain:** DuckDB C++ extension — tabular foundation model evaluation + multi-model ONNX onboarding
**Researched:** 2026-09-19
**Confidence:** MEDIUM (ONNX export paths for TabPFN v2 and TabICL are not officially published; formulas for scoring rules are HIGH from primary sources)

---

## Scope

This document covers only **net-new stack decisions** for this milestone. The existing stack (DuckDB 1.5.4, ONNX Runtime 1.23.2, torch, safetensors, vcpkg, uv, extension-ci-tools) is already in `.planning/codebase/STACK.md` and is not repeated here.

---

## Part A: ONNX Export — TabPFN v2

### A.1 Feasibility Verdict

**ACHIEVABLE with medium effort.** No official ONNX artifact exists from PriorLabs. The `pyproject.toml` for `tabpfn` includes `onnx>=1.19.0` only in the `ci` extras group with the explicit comment _"We run onnx export in the tests, but not in the production package."_ This means the codebase is ONNX-export-tested internally but the artifact is not distributed. The existing `tools/export_onnx` pipeline (torch.dynamo + opset 18) that works for Google TabFM v1 is directly applicable to TabPFN v2 — same `torch.onnx.export(dynamo=True, opset_version=18)` call, same `ExportWrapper` pattern, same tensor-map + safetensors split.

### A.2 TabPFN v2 ONNX Tensor Contract

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `x` (input features) | `[1, T, H]` | float32 | T = total rows (train+test), H = padded feature count |
| `y` (labels) | `[1, T]` | float32 | Train labels; test rows filled with padding sentinel |
| `train_size` | `[1]` | int64 | Number of context/train rows |
| `cat_mask` | `[1, H]` | bool | Per-column categorical flag |
| `d` | `[1]` | int64 | Active (non-padded) feature count |
| `logits` (classification output) | `[1, T, 10]` | float32 | 10 classes; for C<10 the last (10-C) logits are masked/ignored; mixed-radix ensembling for C>10 |
| `logits` (regression output) | `[1, T, K]` | float32 | K = bar-distribution bin count; K is model-version-specific (verify by loading `FullSupportBarDistribution` from checkpoint) |

Dynamic dims: `T` (rows, min=4, max=100_000) and `H` (features, min=2, max=500). These match the existing `export_onnx` `DIM_ROWS`/`DIM_FEATURES` constants and only need range adjustments for TabPFN's 10K row limit.

**Regression bar-distribution detail (HIGH confidence from ScoringBench paper + TabPFN issue #120 + PR #1215):**
- The model outputs K logits per test row representing log-probability masses over K contiguous bins covering the response range.
- Bins have fixed borders `[b_0, b_1, ..., b_K]` stored alongside the model weights.
- The left tail (below `b_0`) and right tail (above `b_{K-1}`) use half-normal distributions (FullSupportBarDistribution), not zero mass — critical for CRPS computation.
- `softmax(logits)` gives per-bin probability masses `p_0..p_{K-1}`.
- CDF: `F(x) = sum_{b_i <= x} p_i` (discrete, then linearly interpolated within bins).
- Mean: `sum_i p_i * center_i` where `center_i = (b_i + b_{i+1}) / 2`.
- Quantile at level q: invert CDF by linear interpolation.
- The regression output tensor must ship the bin borders as a companion tensor (either a separate ONNX initializer or a side-channel JSON in the tensor map).

### A.3 TabPFN v2 Preprocessing Profile

TabPFN v2 preprocessing **differs significantly from TabFM v1** (`tabfm_v1_minimal`). A new preprocessing profile `tabpfn_v2` is required. Key differences:

| Step | TabFM v1 (`tabfm_v1_minimal`) | TabPFN v2 (`tabpfn_v2`) |
|------|-------------------------------|--------------------------|
| Categorical encoding | CategoricalOrdinalEncoder (first-appearance, min_frequency=2, unknown=-1) | Label encoding; NaN/missing → -1; NaN numerical → 0 |
| Numerical imputation | Mean imputation on TRAIN only | Mean imputation; NaN → 0 in numeric |
| Normalization | CustomStandardScaler (z-score, ddof=0, clip [-100,100]) + OutlierRemover (4-sigma log1p) | Z-score per feature on TRAIN; every second ensemble member gets PowerTransform |
| Column order | Categorical first, then numeric, then datetime×5 | Categorical first, then numeric (no datetime expansion) |
| Datetime | 5-column expansion (epoch_ns, year, month, day, dayofweek) | Not natively supported — treat as numeric or categorical |
| Feature cap | 512 (H) | 500 features (H); rows capped at 10K train |
| Regression target | StandardScaler on y, then inverse after predict | StandardScaler on y; inverse after bar-distribution shift |

**Recommended approach:** introduce `kPreprocessProfileId = "tabpfn_v2"` in a new `tabfm_preprocess_tabpfn.cpp` module, following the same interface as `tabfm_preprocess.cpp`. The manifest's `preprocessing_profile` field dispatches to the correct module at runtime.

### A.4 Required Python Versions for Export Tool

| Dependency | Version | Source |
|------------|---------|--------|
| `tabpfn` | `>=2.0` (v2.5 current as of research date) | PyPI `tabpfn` from PriorLabs |
| `torch` | `>=2.5` | Required by tabpfn pyproject.toml |
| `onnx` | `>=1.19.0` | From tabpfn ci extras |
| `onnxruntime` | `>=1.23.2` | Parity check matches runtime version |
| Python | `>=3.9` | tabpfn model card |

The existing `tools/export_onnx/pyproject.toml` uses `uv` — add a new `[tool.uv.sources]` entry for `tabpfn` when building the TabPFN v2 export config.

---

## Part B: ONNX Export — TabICL v2

### B.1 Feasibility Verdict

**ACHIEVABLE with higher effort than TabPFN v2.** TabICL v2 (`pip install tabicl`, `github.com/soda-inria/tabicl`) has no official ONNX export path. The three-stage architecture (column-wise Transformer → row-wise Transformer → dataset-wise ICL Transformer) requires a custom `ExportWrapper` with three sets of dynamic dimensions:

- Stage 1: column-wise — dynamic over `m` (features) 
- Stage 2: row-wise — dynamic over `n` (rows) and `m` (features)
- Stage 3: dataset-wise — dynamic over `n_train` + `n_test` combined

The internal representation uses fixed-width row embeddings (dimension `d`), so stages can be exported independently if they are called sequentially in the wrapper. Alternatively, the full three-stage forward pass can be traced with `dynamo=True` and dynamic shapes on `(n_rows, n_features)`.

**Key risk:** TabICL v2's dataset-wise Transformer processes the entire table in one forward pass including mixed-radix ensembling for C>10 classes; this may require Python-level control flow that is harder to trace with dynamo. Recommend an initial spike to assess traceability.

### B.2 TabICL v2 Tensor Contract

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `x` (features) | `[n, m]` | float32 | n=total rows, m=features after preprocessing |
| `y` (labels) | `[n_train]` | int64 or float32 | Train labels only |
| `logits` (output) | `[n_test, C]` | float32 | C classes; or `[n_test, 1]` for regression |

Dynamic dims: `n` (rows, large range — up to 500K), `m` (features, up to 500).

### B.3 TabICL v2 Preprocessing Profile (`tabicl_v2`)

| Step | TabICL v2 |
|------|-----------|
| Categorical detection | Auto-detects and ordinal-encodes |
| Numerical imputation | Mean imputation |
| Missing numeric | Mean-fill |
| Normalization | Standardization; random rescaling in pretraining (not at inference) |
| Feature grouping | Repeated feature grouping (internal to architecture) |
| Target | Classification: ordinal labels; Regression: raw float |
| Row limit | Scales to 500K+ (vs TabPFN's 10K cap) |

The key difference from `tabpfn_v2`: TabICL processes feature groups in column-wise attention, so the preprocessing wrapper must preserve column ordering consistently.

### B.4 Required Python Versions for Export Tool

| Dependency | Version | Source |
|------------|---------|--------|
| `tabicl` | `>=1.0` (v2 series current) | PyPI `tabicl` from soda-inria |
| `torch` | `>=2.1` | inferred from tabicl architecture |
| `onnx` | `>=1.19.0` | same as TabPFN v2 |
| Python | `>=3.9` | inferred |

---

## Part C: Metric Aggregates — Classification

All metrics implemented as DuckDB C++ aggregate functions with the standard state/initialize/update/combine/finalize API.

### C.1 Point Metrics

#### Accuracy

```
accuracy = (1/N) * sum_i 1{yhat_i == y_i}
```

State: `{int64 correct, int64 total}`. Update: increment `correct` if match, always increment `total`. Finalize: `correct / total` (DOUBLE). Inputs: `(ACTUAL, PREDICTED)` — same type.

**Confidence: HIGH** — standard definition, no ambiguity.

#### F1 (Macro / Micro / Weighted)

For multiclass, compute per-class precision = TP/(TP+FP), recall = TP/(TP+FN), F1_c = 2*P*R/(P+R).

- **Macro F1:** `(1/C) * sum_c F1_c` — unweighted mean over classes.
- **Weighted F1:** `sum_c (support_c / N) * F1_c` — support-weighted.
- **Micro F1:** global TP, FP, FN summed across classes; F1 = 2*TP/(2*TP+FP+FN).

State: `MAP<VARCHAR, {int64 tp, int64 fp, int64 fn}>` keyed by class label string. Update: increment per class. Finalize: compute per-class then aggregate.

Inputs: `(ACTUAL VARCHAR, PREDICTED VARCHAR)`. Return type: DOUBLE.

**Confidence: HIGH** — definition matches scikit-learn exactly.

#### Log-Loss (Binary and Multiclass)

Binary: `LL = -(1/N) * sum_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]`

Multiclass: `LL = -(1/N) * sum_i sum_c 1{y_i=c} * log(p_{i,c})`

Clip probabilities to `[epsilon, 1-epsilon]` where `epsilon = 1e-15` to avoid -inf. Standard `epsilon` is scikit-learn's default.

State: `{double sum_ll, int64 n}`. Update: sum negative log of predicted probability for true class. Finalize: `sum_ll / n`.

Inputs: `(ACTUAL VARCHAR, proba MAP(VARCHAR, DOUBLE))`. Return: DOUBLE.

**Confidence: HIGH** — formula directly from scikit-learn docs + cross-entropy literature.

#### AUC-ROC (Binary)

Uses the trapezoidal rule over the ROC curve. Requires sorting `(score, label)` pairs by descending score.

State: `{LIST<STRUCT(score DOUBLE, label INT)>}` — accumulate all pairs, sort at finalize.

Finalize:
1. Sort by `score` descending.
2. Walk list: for each point compute `(FPR_new - FPR_prev) * (TPR_new + TPR_prev) / 2`.
3. Handle ties by grouping equal-score observations before stepping (avoids the ClickHouse PR #65840 duplicate-score bug).

Formula: `AUC = sum_i (x_{i+1} - x_i) * (y_i + y_{i+1}) / 2` where x=FPR, y=TPR.

For multiclass OvR macro-AUC: compute per-class binary AUC and average.

**Confidence: HIGH** — standard trapezoidal rule; tie-handling confirmed from ClickHouse discussion #65192.

---

## Part D: Metric Aggregates — Regression

### D.1 Point Metrics

#### RMSE

```
RMSE = sqrt((1/N) * sum_i (yhat_i - y_i)^2)
```

State: `{double sum_sq, int64 n}`. Finalize: `sqrt(sum_sq / n)`.

#### MAE

```
MAE = (1/N) * sum_i |yhat_i - y_i|
```

State: `{double sum_abs, int64 n}`. Finalize: `sum_abs / n`.

#### R² (Coefficient of Determination)

```
SS_res = sum_i (y_i - yhat_i)^2
SS_tot = sum_i (y_i - y_bar)^2
R2 = 1 - SS_res / SS_tot
```

Two-pass algorithm (or Welford online variance for SS_tot):

State: `{double sum_y, double sum_y2, double sum_res, int64 n}`. Finalize:
```
y_bar = sum_y / n
SS_tot = sum_y2 - n * y_bar^2
R2 = 1.0 - sum_res / SS_tot
```

Can be negative (model worse than constant mean predictor). Return type DOUBLE.

**Confidence: HIGH** — textbook formula; scikit-learn `r2_score` uses same derivation.

#### MedAE (Median Absolute Error)

Cannot be computed as an online aggregate without storing all residuals. State: `{LIST<double> residuals}`. Finalize: sort and take median. Because this requires O(N) memory and a sort, implement as a separate aggregate clearly marked as non-streaming.

---

## Part E: Proper Scoring Rules (Regression with Predictive Distributions)

**Gate:** These metrics require a model that outputs a predictive distribution, not just a point estimate. As of the start of this milestone, only TabPFN v2 in regression mode provides this (bar distribution). TabFM v1 regression outputs a single logit. TabICL v2 classification outputs class probabilities (usable for Brier score only).

### E.1 CRPS (Continuous Ranked Probability Score)

**Formula (Gneiting & Raftery 2007):**
```
CRPS(F, y) = integral_{-inf}^{inf} (F(x) - 1{x >= y})^2 dx
```

**For bar distribution (discrete CDF):** Because the CDF is piecewise-constant over bins with linear interpolation within bins, the integral decomposes into a sum over bins:

```
CRPS = sum_{k=0}^{K-1} integral_{b_k}^{b_{k+1}} (F(x) - 1{x >= y})^2 dx
```

For interior bins where F is constant at `F_k = sum_{j<=k} p_j`:
```
integral_{b_k}^{b_{k+1}} (F_k - 1{x >= y})^2 dx = (b_{k+1} - b_k) * (F_k - 1{b_k >= y})^2
```

For the bin containing y, the indicator switches mid-bin, requiring splitting at y.

**Tails (FullSupportBarDistribution):** Left tail below `b_0` uses half-normal; right tail above `b_{K-1}` uses half-normal. The tail integral has a closed form involving the normal CDF and PDF. If implementing without tail support initially, truncate to `[b_0, b_{K-1}]` and document as approximation.

**Alternative energy-score identity:** `CRPS(F, y) = E_F|X - y| - 0.5 * E_F|X - X'|`. For bar distributions, E_F|X - y| = sum_k p_k * |center_k - y|. Use this for a simpler implementation that avoids bin-splitting.

State: `{double sum_crps, int64 n}`. Update: compute per-row CRPS from bin borders + probabilities + actual y. Finalize: mean.

**Confidence: HIGH** — formula from Gneiting & Raftery 2007 + ScoringBench paper 2603.29928.

### E.2 Log-Score (Negative Log-Density)

```
LogS(F, y) = -log(f(y))
```

where `f(y)` is the predictive density at observed y.

For bar distribution: `f(y) = p_k / (b_{k+1} - b_k)` where k is the bin containing y (density = probability mass / bin width). For tail regions, use the half-normal density.

Clip to avoid `-inf`: if `f(y) < epsilon`, report `min_log = -log(epsilon)`.

State: `{double sum_nll, int64 n}`. Update: find bin for y, compute density, accumulate `-log(density)`. Finalize: mean.

**Confidence: HIGH** — formula from ScoringBench 2603.29928.

### E.3 Interval Score (IS)

```
IS_alpha(F, y) = (u - l) + (2/alpha) * (l - y) * 1{y < l} + (2/alpha) * (y - u) * 1{y > u}
```

where `[l, u]` is the central (1-alpha) prediction interval (i.e., l = quantile at alpha/2, u = quantile at 1-alpha/2).

Standard choice: `alpha = 0.1` (90% interval). Expose alpha as a parameter.

Quantiles from bar distribution: invert CDF by linear interpolation within bins.

State: `{double sum_is, int64 n}`. Update: extract l and u from distribution, compute IS.

**Confidence: HIGH** — formula from Gneiting & Raftery 2007, confirmed in ScoringBench paper.

### E.4 Energy Score / Beta-Energy Score

For univariate regression (scalar y):
```
ES_beta(F, y) = E_F|X - y|^beta - 0.5 * E_F|X - X'|^beta
```
where beta in (0, 2). Standard choice: beta=1.

For bar distribution with beta=1 this is equivalent to CRPS (CRPS is the energy score with beta=1 for univariate case). Implement as a generalization of CRPS with configurable beta.

**Confidence: MEDIUM** — formula from Gneiting & Raftery 2007; univariate case matches CRPS at beta=1.

### E.5 CRLS (Continuous Ranked Logarithmic Score)

```
CRLS(F, y) = -integral_{-inf}^{inf} log|F(x) + 1{y <= x} - 1| dx
```

This is more complex to implement numerically (integrand has a log singularity at x=y). ScoringBench implements it; defer to Phase 2 of proper scoring rules unless TabPFN v2 CRLS is a requested output.

**Confidence: MEDIUM** — formula confirmed from ScoringBench paper; implementation complexity is HIGH.

---

## Part F: In-SQL Input Format for Distributional Metrics

Because DuckDB aggregate functions receive one row at a time, the bar distribution from the predict aggregate must be passed as a structured column. Two design options:

**Option A — Quantile LIST:** Pass pre-computed quantiles `(q05, q25, q50, q75, q95)` as a STRUCT. Simpler for interval score and CRPS approximation but loses full distribution shape.

**Option B — Raw bin representation:** Pass `{logits: FLOAT[], borders: FLOAT[]}` as a STRUCT. Enables exact CRPS and log-score computation. The borders array must be the same for all rows of the same model run (model-level constant). Prefer this for correctness.

**Recommendation:** Use Option B. Extend `TabFMPredictResult` to carry a `dist_logits: vector<float>` and `dist_borders: vector<float>` field for regression predictions when `preprocessing_profile` declares `distribution_output: true`. The predict aggregate exposes this as `STRUCT(logits FLOAT[], borders FLOAT[])` in detail mode.

---

## Part G: Cross-Validation Architecture

### G.1 Fold Assignment

k-fold CV is implemented as a **table macro** `tabfm_kfold_assign(tbl, k, stratify_by, seed)` that adds a `fold_id INTEGER` column. Algorithm:

For non-stratified: `fold_id = floor(row_number() OVER (ORDER BY hash(rowid, seed)) * k / n)`.

For stratified (classification): sort by class, then assign folds cyclically within each class to preserve class proportions per fold. Algorithm:
1. Rank rows within each class: `rank_in_class = row_number() OVER (PARTITION BY target ORDER BY hash(rowid, seed))`.
2. `fold_id = (rank_in_class - 1) % k`.

Expose as a table macro returning the original table with an additional `fold_id` column.

### G.2 CV Macro

`tabfm_crossval(tbl, target, model, k, metrics)` is a table macro that:
1. Calls `tabfm_kfold_assign` to get fold assignments.
2. For each fold i in 0..k-1: runs `tabfm_predict_agg` with `fold_id != i` as train rows and `fold_id = i` as test rows.
3. Applies the requested metric aggregates to the predictions of fold i.
4. Returns a table with columns `fold_id, metric_name, metric_value`.

**Implementation path:** DuckDB table macros can call aggregate functions but cannot loop. The loop must be unrolled by generating k separate queries and UNIONing them. Alternatively, implement as a C++ table function that generates query text and executes it via `context.db->Query(...)`. The C++ table function path is cleaner but requires implementing a `TableFunction` — use this.

**Confidence: HIGH** for the algorithm; MEDIUM for exact DuckDB API for executing sub-queries from a table function.

---

## Part H: Calibration

### H.1 Expected Calibration Error (ECE)

```
ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|
```

where M = number of bins (standard: 10 equal-width bins over [0,1]), `|B_m|` = number of predictions in bin m, `acc(B_m)` = fraction of correct predictions in bin m, `conf(B_m)` = mean predicted confidence in bin m.

State: `{int64 bin_n[10], int64 bin_correct[10], double bin_conf_sum[10]}`. Update: assign prediction to bin, update counts. Finalize: compute weighted mean.

Inputs: `(actual BOOLEAN or VARCHAR, predicted_proba DOUBLE)` — the maximum class probability paired with whether the top-class prediction is correct.

**Confidence: HIGH** — standard definition from Guo et al. 2017; confirmed in multiple calibration papers.

---

## Part I: SQL Engine Precedent

### BigQuery ML

`ML.EVALUATE` returns `{mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, explained_variance}` for linear regression; `{precision, recall, accuracy, f1_score, log_loss, roc_auc}` for classification. Metrics are exposed as table-function outputs, not as SQL aggregates — they require a trained model object.

**DuckDB approach is more powerful:** implementing metrics as composable SQL aggregates means they work over any `(actual, predicted)` pair without a model object. This is the correct design for anofox-tabfm.

---

## Recommended Stack Summary

| Component | Tool/Version | Confidence |
|-----------|-------------|------------|
| ONNX export toolchain (Python) | `tabpfn>=2.0, torch>=2.5, onnx>=1.19.0` via uv | MEDIUM |
| TabICL export toolchain (Python) | `tabicl>=1.0, torch>=2.1, onnx>=1.19.0` via uv | MEDIUM |
| DuckDB aggregate C++ API | DuckDB 1.5.4 — `AggregateFunction` + `state/initialize/update/combine/finalize` | HIGH |
| CRPS implementation base | Energy-score identity: `E_F|X-y| - 0.5*E_F|X-X'|` over bar-distribution bins | HIGH |
| Log-score implementation | Density from bar distribution: `p_k / (b_{k+1}-b_k)` for bin k | HIGH |
| Interval score | Quantile inversion of bar CDF at alpha/2, 1-alpha/2 | HIGH |
| Fold assignment | DuckDB table macro using `hash(rowid, seed) % k` | HIGH |
| CV orchestration | C++ `TableFunction` generating and executing sub-queries | MEDIUM |
| ECE calibration | 10-bin equal-width aggregate | HIGH |
| AUC-ROC | Sort-at-finalize trapezoidal with tie-grouping | HIGH |
| Preprocessing profile dispatch | `preprocessing_profile` field in manifest JSON → C++ dispatch enum | HIGH (existing pattern) |

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| Energy-score CRPS identity for bar distribution | Full integral over bins | Energy identity is simpler to implement and equally exact for finite-support distributions; the full integral needs special-casing for the bin containing y |
| C++ TableFunction for CV macro | DuckDB SQL table macro with k-union | Table macro cannot loop; unrolling requires string generation at the SQL layer which is fragile and k-dependent |
| Bar distribution as `{logits, borders}` STRUCT | Pre-computed quantile LIST | Quantile list loses distribution shape needed for log-score and CRPS; full representation is strictly more capable |
| Per-profile preprocessing C++ module | Single branching function in preprocess.cpp | One module per profile preserves file ownership rule and prevents merge conflicts across model onboarding workstreams |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `recursive := true` in DuckDB unnest | Breaks on arbitrary nesting depth; disallowed in CLAUDE.md | `unnest(x, max_depth := 3)` |
| CRLS in Phase 1 | Log-singularity at y makes numerical integration difficult; adds implementation risk without proportional value | CRPS + LogS cover the same information theoretically |
| Variogram score for univariate targets | Variogram score is a multivariate extension; for scalar y it reduces to energy score | Beta-energy score (= CRPS at beta=1) |
| Bundling rank statistics (autorank, CD diagrams) | Out of scope per PROJECT.md; R/Python ecosystem already has this | Use scoringutils (R) or autorank (Python) downstream |

---

## Sources

- [ScoringBench paper (arXiv 2603.29928)](https://arxiv.org/html/2603.29928v1) — scoring rule formulas (CRPS, LogS, CRLS, IS, Energy), 5-fold CV design; confidence MEDIUM (verified)
- [ScoringBench GitHub (jonaslandsgesell)](https://github.com/jonaslandsgesell/ScoringBench) — ProbabilisticWrapper pattern, compute_metrics() API; confidence MEDIUM
- [TabPFN v2 PriorLabs/tabpfn pyproject.toml](https://github.com/PriorLabs/TabPFN/blob/main/pyproject.toml) — onnx>=1.19.0 in ci extras; torch>=2.5; confidence HIGH (direct source)
- [TabPFN v2 issue #120](https://github.com/PriorLabs/TabPFN/issues/120) — bar distribution output format, bin borders, logits shape; confidence MEDIUM
- [TabPFN v2 PR #1215](https://github.com/PriorLabs/TabPFN/pull/1215) — FullSupportBarDistribution tail treatment (half-normal); confidence MEDIUM
- [Prior Labs regression docs](https://docs.priorlabs.ai/capabilities/regression) — output_type API (mean/median/mode/quantiles/main/full); confidence HIGH
- [TabICL GitHub (soda-inria)](https://github.com/soda-inria/tabicl) — architecture, preprocessing, no ONNX export; confidence MEDIUM
- [TabICLv2 paper (arXiv 2602.11139)](https://arxiv.org/html/2602.11139v1) — three-stage architecture, input/output shapes, mixed-radix ensembling; confidence MEDIUM (verified)
- [Gneiting & Raftery 2007 (JASA)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) — IS, energy score formulas; confidence HIGH (primary source)
- [DuckDB aggregate function README](https://github.com/duckdb/duckdb/blob/main/extension/core_functions/aggregate/README.md) — initialize/update/combine/finalize API; confidence HIGH
- [anofox-tabfm community extension page](https://duckdb.org/community_extensions/extensions/anofox_tabfm) — eleven built-in models confirmed; confidence HIGH
- `tools/export_onnx/src/export_onnx/export.py` (local codebase) — existing TabFM v1 ONNX export pipeline with ExportWrapper pattern; HIGH

---

*Stack research for: anofox-tabfm evaluation framework + multi-model ONNX onboarding milestone*
*Researched: 2026-09-19*
