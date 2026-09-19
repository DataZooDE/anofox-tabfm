# Pitfalls Research

**Domain:** Tabular-ML evaluation metrics, cross-validation, proper scoring rules, and ONNX model onboarding inside a DuckDB C++ extension
**Researched:** 2026-09-19
**Confidence:** HIGH (metrics/scoring-rules domain); MEDIUM (ONNX model-specific output contracts for TabPFN v2 / TabICL)

---

## Critical Pitfalls

### Pitfall 1: Multiclass AUC Averaging Mode Mismatch

**What goes wrong:**
Implementing a single `tabfm_auc` aggregate that defaults to the wrong averaging mode for the caller's task. OvR (one-vs-rest) macro-average and OvO (one-vs-one) macro-average produce different numbers for the same predictions on the same data; neither is "correct" in isolation. A user comparing results to a Python reference will get a different number if the modes don't match, and the discrepancy is silent — both values are valid AUC scores, they just measure different things. Additionally, micro-averaging AUC is undefined for OvO; accepting that combination produces mathematical nonsense.

**Why it happens:**
Developers port a binary AUC implementation and extend it naively to multiclass by averaging per-class AUCs without exposing (or documenting) which averaging convention they used.

**How to avoid:**
- Expose `averaging` as a required named parameter (`'macro_ovr'`, `'macro_ovo'`, `'weighted_ovr'`) with no default that silently picks one.
- Reject `averaging='micro'` with `'ovo'` with an explicit error naming the SQL fix (`SET anofox_tabfm_auc_averaging`).
- Golden-test each averaging mode against sklearn's `roc_auc_score(multiclass='ovr'/'ovo', average='macro'/'weighted')` on the same fixture data — the fixture is already in `test/fixtures/` and can be extended with a known 3-class problem.

**Warning signs:**
- AUC results that differ between SQL and Python by more than floating-point tolerance.
- No test that specifies an averaging mode and checks against a known reference.
- The aggregate accepts any `averaging` string without rejecting invalid combinations.

**Phase to address:**
Evaluation framework — Phase 1 (classification metrics). Write the failing Catch2 golden test for each averaging mode before implementing.

---

### Pitfall 2: ROC-AUC Tie Handling Produces Incorrect Results

**What goes wrong:**
When multiple rows share the same predicted score, the AUC computation depends on how ties are broken in the rank ordering. The correct approach (consistent with the trapezoidal rule over a sorted threshold sweep) requires averaging tie-group contributions, not breaking ties arbitrarily. An implementation that sorts by score and uses a simple running sum will produce an AUC that varies depending on the sort stability of the implementation — a non-deterministic, data-order-dependent bug.

**Why it happens:**
The Wilcoxon-Mann-Whitney form of AUC (`AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)`) makes tie handling explicit, but most tutorial implementations omit the tie term, passing the linear sort path. In SQL, the sort order of equal-scored rows is undefined unless `ORDER BY score, row_id` is specified.

**How to avoid:**
- Use the rank-sum form: `AUC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)` where ties are broken by averaging ranks — this is mathematically equivalent to the correct AUC for tied predictions.
- Fix the sort order inside the aggregate by including a stable secondary key (row insertion order / `rowid`).
- Add a golden test with a deliberately tie-heavy fixture (e.g., a 3-class model that emits the same probability for half the rows).

**Warning signs:**
- AUC differs between two runs on the same data when the physical row order of the input changes.
- No test with tied predicted scores.

**Phase to address:**
Evaluation framework — Phase 1 (classification metrics).

---

### Pitfall 3: PR-AUC Uses Trapezoidal Interpolation Instead of Step Function

**What goes wrong:**
The area under the precision-recall curve computed via trapezoidal interpolation (connect adjacent PR points with straight lines) is provably overly optimistic. Davis and Goadrich (2006) showed that the correct interpolation between two PR operating points follows a hyperbolic path, not a linear one. sklearn's `auc(recall, precision)` uses the trapezoidal rule; sklearn's `average_precision_score` uses right-end step function interpolation — the two functions return different numbers for the same curve and neither is universally "correct." If a user compares the extension's PR-AUC to sklearn without knowing which convention was used, they will conclude the implementation is wrong.

**How to avoid:**
- Implement PR-AUC as a step-function (average precision) by default: `AP = sum_k (R_k - R_{k-1}) * P_k`. This matches sklearn's `average_precision_score`.
- Document clearly that this is average precision, not trapezoidal PR-AUC.
- If trapezoidal is offered as an option, emit a warning or note that it is optimistic on imbalanced data.
- Golden-test against `sklearn.metrics.average_precision_score` with the same fixture.

**Warning signs:**
- PR-AUC values are higher than expected on imbalanced datasets.
- No docstring or SQL comment clarifying interpolation convention.
- Test uses `sklearn.metrics.auc(recall, precision)` as the reference (that is the trapezoidal version, not average precision).

**Phase to address:**
Evaluation framework — Phase 1 (classification metrics).

---

### Pitfall 4: Log-Loss Clipping Epsilon Not Matched to Float Precision

**What goes wrong:**
Log-loss clips probabilities to `[eps, 1-eps]` to avoid `log(0)`. If `eps` is too large (e.g., `1e-7` when the model emits values closer to the boundary) it distorts the loss for confident correct predictions. If `eps` is too small (e.g., `1e-15` with F32 inputs), it has no effect because F32 probabilities cannot represent values below `~1.2e-7` anyway — the clip is silently a no-op and `log(0.0f)` still produces `-inf` on exact-zero inputs. An `-inf` in any row poisons the aggregate mean to `-inf` with no useful diagnostic.

**Why it happens:**
Epsilon values are copied from Python `float64` implementations (sklearn uses `1e-15`) without adjusting for F32 precision. The TabFM predict aggregate emits probabilities as F32 MAP values.

**How to avoid:**
- Use `eps = 1e-7f` for F32 inputs (the smallest F32 above zero is ~`1.18e-38`, but `1e-7f` keeps clipping far from the boundary and matches sklearn's effective F32 behavior).
- Explicitly check for `-inf` after log and emit an `InvalidInputException` naming the row, rather than silently averaging infinity.
- Add a Catch2 test with exactly-zero and exactly-one probability inputs and verify the clip fires and produces a finite loss.

**Warning signs:**
- Log-loss aggregate returns `-inf` or `NaN` on any dataset.
- No test for extreme probability values (0.0, 1.0, probabilities below `1e-6`).
- Epsilon value copied from a Python `float64` reference without adjustment.

**Phase to address:**
Evaluation framework — Phase 1 (classification metrics).

---

### Pitfall 5: F1 Zero-Division on All-Negative or Absent Classes

**What goes wrong:**
For a class that never appears in the ground truth (support = 0) or that the model never predicts, the per-class F1 is `0/0`. Macro F1 (which averages unweighted per-class F1) becomes `NaN` or inflated depending on whether absent classes are included in the denominator. Weighted F1 (weighted by support) silently drops absent classes — which is numerically correct but means macro and weighted F1 are computed over different class counts, which is confusing when cross-validating across folds where class support varies by fold.

**Why it happens:**
Fold-based CV with stratified assignment can still produce folds where a rare class has zero test instances. Metrics computed naively on that fold produce `NaN`, which then propagates into the fold-average.

**How to avoid:**
- For zero-division, follow sklearn's behavior: treat absent-class F1 as 0 by default and emit a warning, with a configurable parameter `zero_division=0|1|warn`.
- In CV orchestration, report per-fold metrics alongside fold sizes and class distributions so users can see which folds triggered zero-division.
- Add a sqllogictest with a binary classification test where all predictions are one class and verify the result is 0 (not NaN) with a warning logged.

**Warning signs:**
- Metric aggregate returns `NaN` on any CV fold.
- No test with a class that never appears in ground truth.
- CV macro-average silently drops folds that produced `NaN`.

**Phase to address:**
Evaluation framework — Phase 1 (classification metrics) and Phase 2 (CV orchestration).

---

### Pitfall 6: R² Undefined for Single-Sample or Constant-Target Folds

**What goes wrong:**
R² = `1 - SS_res / SS_tot`. When all ground-truth values in a fold are identical (`SS_tot = 0`), the formula produces `0/0`. When there is only one test sample, R² is also undefined. sklearn defaults to replacing `NaN` with `1.0` (for perfect predictions) or `0.0` (for imperfect), but this default is misleading — a single-sample fold getting R² = 1.0 is not a valid performance estimate. In CV, this silently inflates the aggregated R².

**How to avoid:**
- Detect `SS_tot == 0` and return `NULL` (not 0 or 1) so the CV orchestrator can aggregate only over well-defined folds.
- Detect single-sample test folds and return `NULL` with a warning.
- In CV orchestration, count and report folds where R² was `NULL`; exclude them from the fold-average with a visible warning, not silently.
- Add a Catch2 test with a constant-target vector.

**Warning signs:**
- R² aggregate returns `NaN`, `Inf`, or `1.0` for a constant target.
- CV fold summary shows all-1.0 R² for a small dataset where some folds have constant targets.
- No test for degenerate input.

**Phase to address:**
Evaluation framework — Phase 1 (regression metrics) and Phase 2 (CV orchestration).

---

### Pitfall 7: MAPE Silently Returns Infinity or NaN When Actuals Contain Zero

**What goes wrong:**
MAPE = `mean(|actual - predicted| / |actual|)`. When any `actual == 0`, the term is `∞`. One zero in a 1,000-row evaluation set makes MAPE `+∞` with no diagnostic. Some implementations silently drop zero-actual rows (biasing the metric) or replace `0` with `epsilon` (makes the reported value meaningless). Both hide the fact that MAPE is the wrong metric for zero-containing targets.

**How to avoid:**
- Detect zero actual values and raise an `InvalidInputException` that names the alternative (`WAPE`, `MASE`, or `MAE`) and the SQL fix (`WHERE actual != 0` or a different metric function).
- Do not silently substitute epsilon or drop rows.
- Document that `tabfm_mape` requires non-zero actuals and that it is provided for completeness, not as a recommended default.
- Add a test that passes a zero actual and verifies the exception message.

**Warning signs:**
- MAPE returns `Inf` or `NaN` without any exception or log message.
- No test with zero actual values.
- MAPE used as the primary CV metric for a regression target that might include zero (e.g., sales data).

**Phase to address:**
Evaluation framework — Phase 1 (regression metrics).

---

### Pitfall 8: CV Data Leakage via Preprocessing Statistics Computed Over the Full Table

**What goes wrong:**
The existing preprocessing pipeline (`tabfm_preprocess.cpp`) computes mean, std, and outlier-clip bounds from the full context window — which in a CV scenario includes both training and test rows for the current fold. This leaks test-set statistics into the training-set fit. For a proper k-fold CV, the scaler must be fit on training rows only and applied (without refitting) to test rows. If the SQL CV macro passes the entire table to the predict aggregate and then filters the output by fold assignment, the leakage has already occurred inside the aggregate.

**Why it happens:**
The predict aggregate is designed for full-table in-context learning where all rows are "context" — that is correct for inference. For CV, the semantic changes: only training-fold rows should contribute to context statistics. This is a fundamentally different call pattern.

**How to avoid:**
- The CV macro must emit two separate calls: `tabfm_classify(training_rows, target)` and `tabfm_classify(test_rows, target, context := training_rows)` using the two-table form that already exists in the API. The two-table form correctly separates context (training) from query (test) rows.
- Never fold-mask inside the single-table aggregate and then filter — that leaks statistics.
- Add a CV correctness test: compare CV accuracy computed with the two-table form vs. with the full-table + filter form on a dataset where leakage would produce a measurably different result.
- Document this constraint prominently in the CV macro's SQL docstring.

**Warning signs:**
- CV macro uses `tabfm_classify(full_table, target) WHERE fold != k` — this is wrong.
- CV accuracy is higher than held-out accuracy on fresh data by more than a few percent.
- No test that verifies the two-table separation is maintained per fold.

**Phase to address:**
Evaluation framework — Phase 2 (CV orchestration). This is the highest-risk correctness pitfall in the CV work.

---

### Pitfall 9: Non-Deterministic or Non-Stratified Fold Assignment

**What goes wrong:**
If fold assignment is not deterministic (seeded), re-running the CV macro returns different per-fold results. If fold assignment is not stratified, small datasets or imbalanced classes produce folds with very different class distributions, making per-fold metrics incomparable and the aggregate misleading. Both problems make debugging impossible: a user who gets a different AUC on two runs of the same query cannot tell whether the model changed or the fold assignment changed.

**How to avoid:**
- Fold assignment must use a deterministic hash: `fold_id = hash(primary_key || seed) % k`, not `random()`. The `seed` parameter must be exposed to the user and documented.
- Stratified fold assignment must be supported for classification: within each class, distribute rows across folds proportionally. The SQL implementation can achieve this with `ROW_NUMBER() OVER (PARTITION BY target ORDER BY hash(rowid || seed)) % k`.
- Add a determinism test (analogous to the existing `tabfm_classify` determinism test) that runs the CV macro twice and asserts identical per-fold metrics.
- Add a stratification test: verify that fold class distributions are approximately equal.

**Warning signs:**
- CV macro uses `random()` without a seed for fold assignment.
- Re-running the CV macro returns different per-fold metrics.
- No test for determinism.
- Fold sizes vary by more than ±1 row for a balanced dataset.

**Phase to address:**
Evaluation framework — Phase 2 (CV orchestration).

---

### Pitfall 10: TabFM Context-Row Limit Exceeded Silently During CV

**What goes wrong:**
The existing predict aggregate has a context-row limit (the in-context learning window `T`). During CV, each training fold has `(k-1)/k * N` rows — more rows than a typical single predict call. If `N` is large enough, the training fold exceeds the model's context limit. The current extension likely handles this via the ensemble / cobatch path, but during CV the fold size is unknown at query-plan time. If the aggregate silently truncates to the context limit, the CV estimates are computed on a different-sized training set than the user expects, without any warning.

**How to avoid:**
- The CV macro must check fold sizes against `anofox_tabfm_max_context_rows` (or equivalent) before running and emit a warning if any training fold exceeds the limit.
- Document the effective context limit in the CV macro's documentation.
- For large datasets, recommend stratified subsampling within each training fold rather than silent truncation.
- Add a test that passes a dataset larger than the context limit and verifies the warning fires.

**Warning signs:**
- CV on a large table runs without error but produces lower accuracy than expected.
- No warning when training fold size exceeds context limit.
- No test for oversized folds.

**Phase to address:**
Evaluation framework — Phase 2 (CV orchestration).

---

### Pitfall 11: CRPS Sample Estimator Bias for Small Prediction Ensembles

**What goes wrong:**
CRPS for a predictive distribution represented as samples (`energy score` form) has a known downward bias when the number of samples is small. The bias is `O(1/M)` where `M` is the ensemble size. For TabPFN v2's bar-distribution output (a discrete CDF over ~100 bins), CRPS can be computed in closed form from the bin probabilities — which is exact and has no sample bias. Teams that convert the bar-distribution to a set of quantile samples and then apply the sample estimator introduce unnecessary bias. Similarly, the energy-score form `CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|` requires `O(M^2)` pairwise distances, which is expensive if M is large.

**How to avoid:**
- For TabPFN v2's bar-distribution output, implement CRPS in closed form: sum over bins of `(F(y_k) - 1{y <= y_k})^2 * delta_k` where `F` is the predicted CDF and `delta_k` is the bin width. This is exact and `O(B)` where `B` is the number of bins (~100).
- Only fall back to the sample estimator if the model emits samples rather than a parametric distribution.
- Document which CRPS variant is used in the aggregate's SQL docstring.
- Add a Catch2 test that computes closed-form CRPS and sample-CRPS for the same distribution and verifies they agree within tolerance.

**Warning signs:**
- CRPS is implemented by sampling quantiles from the predicted distribution and applying the energy score formula.
- CRPS values are systematically lower than a Python reference (bias signature).
- No documentation of which CRPS form is implemented.

**Phase to address:**
Evaluation framework — Phase 3 (proper scoring rules, gated on TabPFN v2 landing).

---

### Pitfall 12: Interval Score Applied to a Point-Estimate Regression Model

**What goes wrong:**
The interval score requires a `(lower, upper)` prediction interval, not a point estimate. If the current `tabfm_regress` path is used (which emits only a point `yhat`), the interval score cannot be meaningfully computed — yet it is tempting to approximate it by using `(yhat - delta, yhat + delta)` with an arbitrary delta, which produces an interval score that measures the choice of delta, not the model's calibration. This is a methodologically unsound result that looks correct in SQL.

**Why it happens:**
The PROJECT.md explicitly flags this dependency: proper scoring rules for regression are blocked until a model emits predictive distributions. The pitfall is implementing interval score before TabPFN v2 lands and using a synthetic interval as a stopgap.

**How to avoid:**
- Gate the `tabfm_interval_score` aggregate behind a check that the model output includes quantile columns. If invoked on a result set without quantile columns, throw an `InvalidInputException` naming the fix.
- Do not implement a synthetic-interval fallback.
- Document the dependency in the function's SQL docstring: "Requires a model that emits predictive quantiles (e.g., TabPFN v2)."

**Warning signs:**
- Interval score is implemented before TabPFN v2 ONNX export is complete.
- The function accepts a `(yhat, yhat)` degenerate interval without error.
- No guard that checks for non-degenerate intervals.

**Phase to address:**
Evaluation framework — Phase 3 (proper scoring rules), only after multi-model Phase 4 (TabPFN v2 onboarding).

---

### Pitfall 13: TabPFN v2 Bar-Distribution Output Layout Misread in the Decoder

**What goes wrong:**
TabPFN v2's regression head outputs logits over a fixed grid of `B` bins that partition the target space. The bins are not uniformly spaced — they are quantile-based over the training target distribution, computed during the forward pass. The ONNX graph must export both the logits tensor (`[1, T, B]`) and the bin boundary tensor (`[1, B+1]` or `[B+1]`). If the decoder reads only the logits and assumes uniform bins (e.g., `[0, 1/B, 2/B, ...]`), every quantile and CRPS computation is wrong in a non-obvious way — they will be plausible-looking numbers, just systematically off. This is particularly insidious because the existing `tabfm_engine.cpp` output validation code (which is already flagged P0-missing in CONCERNS.md) does not yet check tensor names or shapes against a model family contract.

**How to avoid:**
- The `tools/export_onnx` WS-A tool must export both `logits` and `bin_boundaries` tensors as named outputs; verify both are present in the ONNX graph using `onnxruntime.InferenceSession.get_outputs()` before committing the fixture.
- The `tabfm_manifest.json` for TabPFN v2 must declare the output tensor names and their roles so the C++ decoder can locate them by name, not by index.
- `ValidateOutput` (the P0 fix from CONCERNS.md) must be extended for the TabPFN v2 family to check for both output tensors and reject with an actionable error if either is missing.
- Golden-test the bin boundaries against the reference Python output on the same fixture data.

**Warning signs:**
- The ONNX export graph has only one output tensor for regression (logits only).
- The decoder hardcodes bin boundaries as `linspace(0, 1, B)`.
- CRPS or quantile values differ from the Python reference by more than 5%.
- The manifest's `preprocessing_profile` for TabPFN v2 is set to `tabfm-v1` (copy-paste from existing manifest).

**Phase to address:**
Multi-model Phase 4 (TabPFN v2 onboarding), specifically the `tools/export_onnx` and manifest work.

---

### Pitfall 14: TabPFN v2 / TabICL Preprocessing Profile Mismatch

**What goes wrong:**
The existing `tabfm_preprocess.cpp` implements the exact preprocessing pipeline for `tabfm-v1` (Google's model): ordinal encoding, mean imputation, outlier clipping, z-score scaling with specific epsilon values and clipping ranges. TabPFN v2 (PriorLabs) uses a different preprocessing: it applies its own internal normalization inside the forward pass after receiving raw ordinal-encoded features — applying the existing `tabfm-v1` scaler before the TabPFN v2 graph double-normalizes the data. TabICL has yet another convention (ordinal encoding with separate category-for-NaN treatment). Using the wrong preprocessing profile produces predictions that are in-distribution for the scaler but out-of-distribution for the model, which produces confident but systematically wrong predictions with no error.

**Why it happens:**
`preprocessing_profile` is currently effectively hardcoded to `tabfm-v1` in the engine, as noted in the PROJECT.md. Adding a new model family without adding its preprocessing profile produces silent mismatch.

**How to avoid:**
- The `preprocessing_profile` field in the manifest must be parsed and dispatched to the correct preprocessing strategy before the predict call. The dispatch table must reject unknown profile names with an error.
- The `tools/parity` WS-A tool must run a parity check between the Python reference preprocessing output and the C++ output for each new model family before the ONNX fixture is committed.
- Add a manifest validation step in `tabfm_manifest.cpp` that rejects manifests where `model_family` and `preprocessing_profile` are inconsistent (e.g., family `tabpfn-v2` with profile `tabfm-v1`).

**Warning signs:**
- A new model family's manifest sets `preprocessing_profile = "tabfm-v1"`.
- The parity tool (`tools/parity`) is not run before committing the new model fixture.
- Predictions from the extension differ from the Python reference by more than the expected floating-point tolerance.

**Phase to address:**
Multi-model Phase 4 (generalize model support) — this is the prerequisite step before adding TabPFN v2 or TabICL.

---

### Pitfall 15: TabPFN v2 / TabICL ONNX Dynamic Shape Export Misses the Column-Count Dimension

**What goes wrong:**
The existing `tabfm-v1` graph is exported with a fixed feature-count dimension `H` (number of columns), handled by shape-bucket padding. TabPFN v2 and TabICL support arbitrary column counts — but the ONNX export must declare the column dimension as dynamic (e.g., `dynamic_axes={'x_cont': {0: 'batch', 1: 'T', 2: 'H'}}`). If only the batch and row dimensions are marked dynamic and `H` is frozen at export-time, the ONNX graph will fail at runtime whenever the user's table has a different number of features than the export fixture. This error appears as an ORT shape mismatch, not a preprocessing error, making it hard to diagnose.

**How to avoid:**
- The `tools/export_onnx` export script must mark `H` (feature count) as dynamic in the `dynamic_axes` specification.
- After export, validate the ONNX graph by running inference with two different feature counts and verify both succeed.
- Update `tabfm_ort_engine.cpp`'s shape-bucket logic to handle variable `H` — or document that `H` is padded to the nearest bucket and verify the padding is consistent with how the new model handles padding.

**Warning signs:**
- ONNX export uses a single dummy input with fixed feature count and does not test a second feature count.
- ORT error message includes "got shape [1, T, H_new] but expected [1, T, H_export]" during integration testing.
- The fixture manifest hard-codes `"n_features"` as a static integer.

**Phase to address:**
Multi-model Phase 4 (TabPFN v2 / TabICL onboarding), specifically the `tools/export_onnx` step.

---

### Pitfall 16: TabPFN v2 Weight License Not Gated Separately from Code License

**What goes wrong:**
The TabPFN v2 code is Apache 2.0, but the model weights retain the Prior Labs License (Apache 2.0 + attribution requirement for commercial use; v2.5, v2.6, v3 are non-commercial only). If the extension downloads TabPFN v2 weights using the same license-gate mechanism as `tabfm-v1` (which checks `anofox_tabfm_accept_hf_license`), users who accepted the `tabfm-v1` (Google) license may think they have also accepted the Prior Labs license, which has different terms. Mixing the acceptance flags creates a legal ambiguity.

**How to avoid:**
- Add a per-model-family license acceptance flag: `anofox_tabfm_accept_priorlabs_license` distinct from `anofox_tabfm_accept_hf_license`.
- The download path must check the correct per-family flag; the manifest must declare which license family it belongs to.
- Error messages must cite the specific license URL, not a generic HuggingFace license.
- Never auto-accept a license by inferring it from another family's flag.

**Warning signs:**
- The TabPFN v2 download path reuses `anofox_tabfm_accept_hf_license` without modification.
- The manifest for TabPFN v2 declares `"license": "hf"` (copied from `tabfm-v1`).
- No test for the TabPFN v2-specific license gate rejection.

**Phase to address:**
Multi-model Phase 4 (TabPFN v2 onboarding). Must be resolved before any download logic is wired for the new family.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reuse `tabfm-v1` preprocessing profile for TabPFN v2 | Avoids implementing a second dispatch path | Silent prediction corruption; impossible to detect without parity test | Never |
| Default AUC averaging to macro without exposing parameter | Simpler SQL surface | Silent mismatch with user's Python reference; misleading metric comparisons | Never |
| Use trapezoidal rule for PR-AUC (matches ROC-AUC implementation pattern) | Single interpolation code path | Overly optimistic PR-AUC on imbalanced data; contradicts sklearn's `average_precision_score` | Never for imbalanced tasks |
| Implement CRPS via sample estimator even when a closed-form exists | Simpler code; avoids bin-boundary parsing | Systematic downward bias; O(M^2) cost | Only if model provably emits samples, not a parametric distribution |
| Single license gate for all HuggingFace models | Simpler UX | Legal ambiguity between Google and Prior Labs license terms | Never |
| CV fold assignment without stratification | Simpler SQL macro | Misleading per-fold metrics for imbalanced classes; NaN propagation from absent classes | Only for balanced-class regression tasks |
| Implement interval score before TabPFN v2 lands using synthetic intervals | Can ship the function earlier | Produces methodologically meaningless results that look correct | Never |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| TabPFN v2 ONNX export | Exporting only the logits tensor, omitting bin boundaries | Export both `logits` and `bin_boundaries` as named outputs; validate both in `tools/parity` |
| TabPFN v2 preprocessing | Passing the `tabfm-v1` scaler output to the TabPFN v2 graph | Dispatch on `preprocessing_profile` in the manifest; TabPFN v2 expects raw ordinal-encoded features, not z-scored |
| TabICL categorical encoding | Treating NaN as a missing numerical value | TabICL creates a separate category for NaN; the C++ preprocessing must replicate this — NaN must map to a dedicated category id, not to the mean |
| DuckDB two-table predict in CV | Passing the full table to the predict aggregate and filtering output by fold | Use the two-table form `tabfm_classify(train_rows, target)` with `context := train_rows` so that only training rows contribute to preprocessing statistics |
| ONNX Runtime dtype | Passing F64 probabilities (DuckDB DOUBLE) to an F32 ONNX input | Cast to F32 at the boundary; ORT will silently misread F64 bytes as F32 without an error on some backends |
| Multiclass AUC | Using the OvR implementation for OvO averaging | Implement both paths and reject invalid combinations (`micro + OvO`) at parse time |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| CRPS energy-score with large sample count | O(M^2) pairwise distances; SQL metric query hangs | Use closed-form CRPS from bar-distribution bins; reserve sample estimator for non-parametric cases | M > 500 samples per prediction |
| CV macro runs full predict per fold sequentially | CV for k=5 takes 5× single predict time with no parallelism | If DuckDB connection allows, pipeline fold calls; document expected runtime | k ≥ 5, dataset N > 1,000 |
| AUC computed over the full table without an index | Sorting cost is O(N log N) per call; slow for large N | Acceptable for evaluation (not hot path); document expected complexity | N > 100,000 rows |
| F1/AUC recomputed per fold without caching intermediate state | Each fold re-scans full result set | Use CTEs to compute fold predictions once, then join metric aggregates | k ≥ 10, N > 10,000 |

---

## "Looks Done But Isn't" Checklist

- [ ] **Multiclass AUC:** Implemented for binary only — verify the `averaging` parameter is accepted and dispatches correctly for `n_classes > 2`.
- [ ] **PR-AUC vs. Average Precision:** Verify the implementation matches `sklearn.metrics.average_precision_score`, not `sklearn.metrics.auc(recall, precision)` — both exist in sklearn, they are different.
- [ ] **Log-Loss with F32 proba:** Verify that exactly-zero probabilities are clipped (not producing `-inf`) — test with a fixture where the model output includes 0.0 probability for a class.
- [ ] **R² for constant target:** Verify the aggregate returns `NULL` (not `1.0` or `NaN`) for a constant ground-truth vector.
- [ ] **CV fold assignment determinism:** Run the CV macro twice with the same seed and verify identical per-fold metric values.
- [ ] **CV two-table separation:** Verify by inspection of the SQL macro that training-fold preprocessing statistics do not include test-fold rows.
- [ ] **TabPFN v2 ONNX export:** Verify `tools/parity` shows agreement between Python reference and C++ decoder on the full set of outputs: logits, bin boundaries, quantile(0.5), CRPS.
- [ ] **License gates:** Verify that `tabfm_download` for TabPFN v2 checks `anofox_tabfm_accept_priorlabs_license`, not `anofox_tabfm_accept_hf_license`.
- [ ] **Model output validation:** The P0 fix from CONCERNS.md (`ValidateOutput`) must be extended for the TabPFN v2 output tensor contract before the model is wired to predictions.
- [ ] **MAPE with zero actuals:** Verify the aggregate raises an exception on zero actuals, not returns `Inf`.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong AUC averaging mode shipped | MEDIUM | Add `averaging` parameter with backward-compatible default; document breaking change in changelog; add golden tests |
| Preprocessing profile mismatch discovered post-ship | HIGH | Invalidate all cached predictions from the affected model family; rerun parity; issue a patch release with the correct profile |
| CV leakage discovered post-ship | HIGH | Re-document the correct two-table calling pattern; provide a migration guide; re-run all CV benchmarks with the corrected macro |
| Log-loss returning `-inf` in production | LOW | Add F32-aware epsilon clip; patch release; the fix is a one-line change in the aggregate finalize |
| TabPFN v2 ONNX export missing bin boundaries | MEDIUM | Re-export with the correct output spec; update the manifest and fixture sha256; rebuild CI |
| License gate using wrong flag | LOW | Add the correct per-family flag; the old flag remains functional for its family |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Multiclass AUC averaging mode mismatch | Phase 1: Classification metrics | Golden test each mode against sklearn reference on 3-class fixture |
| ROC-AUC tie handling | Phase 1: Classification metrics | Golden test with tie-heavy fixture (50% tied scores) |
| PR-AUC trapezoidal vs. step function | Phase 1: Classification metrics | Golden test against `sklearn.metrics.average_precision_score` |
| Log-loss epsilon / F32 clip | Phase 1: Classification metrics | Catch2 test with exactly-zero and exactly-one probabilities |
| F1 zero-division on absent classes | Phase 1 (impl) + Phase 2 (CV) | sqllogictest with all-one-class predictions; CV test with rare class |
| R² undefined cases | Phase 1: Regression metrics | Catch2 test with constant target and single-sample input |
| MAPE zero actuals | Phase 1: Regression metrics | sqllogictest verifying exception message on zero actual |
| CV data leakage via preprocessing | Phase 2: CV orchestration | Correctness test comparing two-table vs. full-table+filter AUC on a dataset with detectable leakage |
| Non-deterministic / non-stratified fold assignment | Phase 2: CV orchestration | Determinism test (two runs, same seed → identical metrics); stratification check |
| Context-row limit in CV folds | Phase 2: CV orchestration | Test with dataset larger than max context; verify warning fires |
| CRPS sample bias | Phase 3: Proper scoring rules | Catch2 test comparing closed-form vs. sample CRPS on bar-distribution fixture |
| Interval score on point-estimate model | Phase 3: Proper scoring rules | Guard clause + test that verifies rejection when quantile columns absent |
| TabPFN v2 bar-distribution layout misread | Phase 4: TabPFN v2 onboarding | `tools/parity` comparison on all output tensors before C++ decoder is written |
| Preprocessing profile mismatch | Phase 4: Generalize model support | `tools/parity` run per new model family before fixture commit |
| ONNX dynamic shape missing H dimension | Phase 4: TabPFN v2 / TabICL onboarding | Export validation with two different feature counts |
| License gate mismatch | Phase 4: TabPFN v2 onboarding | sqllogictest verifying per-family license rejection error message |

---

## Sources

- scikit-learn docs: `roc_auc_score` multiclass parameter semantics (OvR/OvO, micro undefined for OvO): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- "The wrong and right way to approximate AUPRC" (Davis & Goadrich 2006 revisited): https://towardsdatascience.com/the-wrong-and-right-way-to-approximate-area-under-precision-recall-curve-auprc-8fd9ca409064/
- scikit-learn `f1_score` zero_division parameter: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- scikit-learn `r2_score` constant-target behavior: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
- MAPE zero denominator pitfalls and alternatives (WAPE, MASE): https://openforecast.org/2024/04/17/avoid-using-mape/
- scoringRules R package: CRPS closed-form parametric vs. sample estimator: https://arxiv.org/pdf/1709.04743
- Interval score as proper scoring rule, coverage-vs-width tradeoff: https://epiforecasts.io/scoringutils/articles/scoring-rules.html
- TabPFN v2 bar-distribution regression head and quantile decoding: https://arxiv.org/pdf/2605.13986
- TabPFN v2 license: Apache 2.0 code, Prior Labs License weights (non-commercial for v2.5+): https://github.com/PriorLabs/TabPFN/pull/1271
- TabICL categorical encoding and ordinal strategy: https://pypi.org/project/tabicl/
- ONNX dynamic axes and dtype mismatch pitfalls: https://onnxruntime.ai/docs/api/python/auto_examples/plot_common_errors.html
- TabPFN preprocessing error handling (upstream issue): https://github.com/PriorLabs/TabPFN/issues/183
- ONNX shape mismatch runtime errors and prevention: https://theneuralbase.com/onnx/errors/onnx-input-shape-mismatch-runtime-error/
- Data leakage in cross-validation (preprocessing must be fit on training fold only): https://machinelearningmastery.com/data-preparation-without-data-leakage/
- Project internal: `.planning/codebase/CONCERNS.md` — P0 missing model output validation, NaN features, preprocessing fragility
- Project internal: `.planning/PROJECT.md` — confirmed regression emits point estimate only; proper scoring rules gated on TabPFN v2

---

*Pitfalls research for: tabular-ML evaluation metrics, CV, proper scoring rules, ONNX model onboarding in DuckDB C++ extension*
*Researched: 2026-09-19*
