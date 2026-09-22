# Phase 3: Proper Scoring Rules + Cross-Model Comparison — Research

**Researched:** 2026-09-22
**Domain:** DuckDB C++ aggregate functions over bar-distribution predictive output; proper scoring rules math; STRUCT/LIST reading in aggregate Update
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Scoring-rule aggregates (PSR-01/02/03/04)**
- New module `src/tabfm_scoring.cpp` (+ header). All three are SQL aggregates
  registered with full `anofox_tabfm_*` + short `tabfm_*` aliases; telemetry once
  per bind.
- **Input shape:** each takes `(actual DOUBLE, yhat_dist STRUCT(logits DOUBLE[],
  borders DOUBLE[]))` — the exact shape Phase 2 emits under
  `output_mode='distribution'`, so users pass `yhat_dist` straight through.
- **`tabfm_crps` (PSR-01):** analytical closed-form CRPS over the bar/histogram
  distribution — integral of `(F(x) − 1{y ≤ x})²` with a piecewise-linear
  (or piecewise-constant-density) CDF across the **non-uniform** bins; use the
  actual bucket widths (uniform assumption is wrong — spike). Softmax the logits
  to per-bin probability mass first. Golden-tested against a reference computed in
  the fixture/parity tooling.
- **`tabfm_log_score` (PSR-02):** negative log-likelihood of `actual` under the
  bar distribution — `−log(density in the bin containing actual)`, density =
  `p_bin / bin_width`; clip to avoid `log(0)` (document the epsilon).
- **`tabfm_interval_score` (PSR-03):** interval score at a configurable coverage;
  derive the lower/upper quantiles for `(1−coverage)/2` and `1−(1−coverage)/2`
  from the distribution and apply the standard interval-score formula.
- **Coverage parameter:** named `coverage := 0.9` with a documented **default of
  0.9** (configurable per success criterion).
- **Bind-gating (PSR-04):** all PSR aggregates are bind-gated on the `yhat_dist`
  STRUCT input type; given a point estimate (plain DOUBLE) they fail at bind with
  a named error pointing at `output_mode := 'distribution'` (SQL-API §5).

**Cross-model comparison (CMP-01)**
- Surfaced as a **documented pattern + a thin helper macro** that runs the Phase-1
  metrics and Phase-3 scores across model families on the **user's own table**
  (no bundled datasets). Composes the existing model-agnostic primitives; the
  macro quotes identifiers safely (carry forward the Phase-1 pattern). Verified by
  a test that compares `tabpfn_v2` (fixture) distribution output and `tabfm-v1` on
  a small user-style table.

### Claude's Discretion
- Exact CDF interpolation convention within a bin (piecewise-uniform density is
  the natural choice for a bar distribution) and the log-score epsilon; the
  aggregate state layout; the precise helper-macro shape for CMP-01; whether CRPS
  reference goldens are generated in `tools/parity` or hard-coded from a
  documented computation. Grounded in the confirmed contract and Phase 1/2
  patterns.

### Deferred Ideas (OUT OF SCOPE)
- CRLS (log-score with singularity handling) and multivariate scores
  (energy/variogram) — v2 (ASCR-01).
- Calibration curves / reliability diagrams as data output — v2 (ASCR-02).
- Real TabPFN v2 inference export + TabICL comparison — deferred (Phase 2 notes);
  CMP-01 compares the fixture-scoped tabpfn_v2 vs tabfm-v1.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PSR-01 | CRPS over regression predictive distribution via SQL aggregate | §Exact CRPS Math; §STRUCT Reading in Update |
| PSR-02 | Log-score (NLL) over regression predictive distribution | §Log-Score Math; §STRUCT Reading in Update |
| PSR-03 | Interval score at configurable coverage level | §Interval Score Math; §STRUCT Reading in Update |
| PSR-04 | PSR functions bind-gated on distribution input with named remedy | §Bind-Gating (PSR-04) Pattern |
| CMP-01 | Evaluation metrics across multiple model families on user tables | §CMP-01 Macro Pattern |
</phase_requirements>

---

## Summary

Phase 3 adds three proper-scoring-rule aggregates (`tabfm_crps`, `tabfm_log_score`, `tabfm_interval_score`) over the `yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])` output that Phase 2 produced, plus a thin cross-model comparison macro. All math operates on the confirmed Phase 2 bar-distribution contract: K bins, non-uniform borders in raw space, z-space logits stored pre-softmax. The aggregates follow the same bind/update/combine/finalize pattern as Phase 1 regression metrics, with one new wrinkle: the second argument is a STRUCT containing two nested LISTs, which must be extracted in Update via `StructValue::GetChildren` + `ListValue::GetChildren` (the same idiom as the MAP reader in `tabfm_metrics_classification.cpp:476`).

The CRPS closed-form over a piecewise-uniform-density bar distribution is analytically tractable and fully derivable from the bin borders. The interval score reuses the existing `DistributionQuantile` helper (external linkage in `tabfm_predict.hpp`). The bind-gate for PSR-04 is the 2-arg overload that always throws, identical to the Phase 1 `tabfm_precision`/`tabfm_recall`/`tabfm_f1` pattern. The CMP-01 macro is a thin SQL table macro using the same `query()` + safe identifier quoting as `tabfm_cross_validate`.

**Primary recommendation:** Start with `tabfm_log_score` (simplest state: one `double sum`, one `int64_t n`) to validate the STRUCT-reading idiom, then add `tabfm_crps` (per-bin integral loop), then `tabfm_interval_score` (reuses quantile helper), then PSR-04 bind overloads. Implement CMP-01 macro in the same plan.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| CRPS/log-score/interval score computation | SQL aggregate (C++ finalize-less — per-row running state) | — | Scoring rules are streaming aggregates over `(actual, yhat_dist)` pairs; finalize divides by count |
| STRUCT input reading (yhat_dist) | C++ aggregate Update | — | Per-row extraction of logits LIST + borders LIST from the STRUCT argument |
| Quantile computation for interval score | Shared helper `DistributionQuantile` (tabfm_predict.hpp) | — | Already external-linkage; reuse directly in scoring Update |
| Bind-gating (PSR-04) | Aggregate Bind callback | — | 2-arg overload always throws at bind time; same pattern as Phase 1 required-avg gate |
| CMP-01 cross-model comparison | SQL table macro (string-building, query() executor) | — | Composes existing model-agnostic primitives; no new C++ needed |
| Golden test reference values | tools/parity Python script | test/sql golden assertions | Computed from golden.json fixture; Python reference generates expected values for hard-coded C++ test assertions |

---

## Standard Stack

This phase introduces no new library dependencies. It uses only:

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| DuckDB Extension API (bundled) | v1.5.4 (pinned) | Aggregate registration, `AggregateFunctionSet`, `StructValue::GetChildren`, `ListValue::GetChildren`, `Value::LIST` | Project-mandated; same as Phase 1/2 |
| ONNX Runtime (bundled) | v1.23.2 | Already used; no new calls needed | Project-mandated |
| Catch2 (bundled) | embedded | C++ unit tests for CRPS/log-score/interval math with known logit/border inputs | Project-mandated |
| PostHog telemetry (bundled) | embedded | `CaptureFunctionExecution` once per bind | CLAUDE.md rule #3 |

**Installation:** None — no new packages. All capabilities are in the existing build.

---

## Package Legitimacy Audit

No new external packages are introduced in this phase.

| Package | Verdict | Disposition |
|---------|---------|-------------|
| (none) | — | No new packages |

---

## Architecture Patterns

### System Architecture Diagram

```
User SQL:
  SELECT tabfm_crps(actual, yhat_dist) FROM predictions

                        │
                        ▼
        ┌─────────────────────────────┐
        │  AggregateFunctionSet       │
        │  2-arg overload: throws     │ ◄── PSR-04 bind gate
        │  (actual DOUBLE, yhat_dist  │
        │   STRUCT): normal bind      │
        └──────────┬──────────────────┘
                   │ Bind: telemetry, validate arg type
                   ▼
        ┌─────────────────────────────┐
        │  Update (per row)           │
        │  StructValue::GetChildren   │
        │    → {logits LIST, borders  │
        │       LIST}                 │
        │  ListValue::GetChildren     │
        │    → vector<Value>          │
        │  softmax(logits) → probs    │
        │  score += compute(y,probs,  │
        │    borders)                 │
        └──────────┬──────────────────┘
                   ▼
        ┌─────────────────────────────┐
        │  Combine (parallel merge)   │
        │  sum_score += src.sum_score │
        │  n += src.n                 │
        └──────────┬──────────────────┘
                   ▼
        ┌─────────────────────────────┐
        │  Finalize                   │
        │  return sum_score / n       │
        │  (NULL on n==0)             │
        └─────────────────────────────┘
```

### Recommended Project Structure

```
src/
├── tabfm_scoring.cpp          # NEW — PSR-01/02/03/04 aggregates
├── include/
│   └── tabfm_scoring.hpp      # NEW — RegisterScoringFunctions declaration
test/
├── sql/
│   └── tabfm_scoring.test     # NEW — sqllogictest PSR-01..04 + bind-gate
├── cpp/
│   └── test_tabfm_scoring.cpp # NEW — Catch2 unit tests for math helpers
tools/
└── parity/
    └── src/parity/
        └── crps_reference.py  # NEW — golden CRPS/log-score computation
```

---

## STRUCT Reading in Aggregate Update — Exact DuckDB Idiom

[VERIFIED: src/tabfm_metrics_classification.cpp:466-482]

The Phase 1 `tabfm_log_loss` Update reads a MAP(VARCHAR, DOUBLE) argument via:
```cpp
Value proba_val = inputs[1].GetValue(i);
for (auto &kv : MapValue::GetChildren(proba_val)) {
    auto &entry = StructValue::GetChildren(kv);
    // entry[0] = key, entry[1] = value
}
```

For the PSR aggregates, `inputs[1]` is `STRUCT(logits DOUBLE[], borders DOUBLE[])`. The reading idiom follows the same `GetValue(i)` → `StructValue::GetChildren` → `ListValue::GetChildren` path:

```cpp
Value dist_val = inputs[1].GetValue(i);
if (dist_val.IsNull()) { continue; }
auto &dist_children = StructValue::GetChildren(dist_val);
// dist_children[0] = logits LIST(DOUBLE)  (field "logits")
// dist_children[1] = borders LIST(DOUBLE) (field "borders")

auto &logit_vals   = ListValue::GetChildren(dist_children[0]);
auto &border_vals  = ListValue::GetChildren(dist_children[1]);

const size_t K = logit_vals.size();
if (K == 0 || border_vals.size() != K + 1) { continue; }

std::vector<double> logits(K), borders(K + 1);
for (size_t k = 0; k < K; k++) {
    logits[k]  = DoubleValue::Get(logit_vals[k]);
    borders[k] = DoubleValue::Get(border_vals[k]);
}
borders[K] = DoubleValue::Get(border_vals[K]);
```

**Why `GetValue(i)` not `UnifiedVectorFormat`:** The Phase 1 pattern for complex-typed arguments (MAP, STRUCT) uses `inputs[N].GetValue(i)` which handles all DuckDB vector representations (flat, constant, dictionary) uniformly. Using raw `UnifiedVectorFormat` on a STRUCT/LIST column and then manually indexing struct/list child vectors is more verbose and fragile for the value-extraction path. The MAP/STRUCT approach via `GetValue` is the established idiom in this codebase.

[VERIFIED: src/tabfm_predict_agg.cpp:596-619] The finalize code in Phase 2 confirms the STRUCT field order: `dist_fields.emplace_back("logits", ...)` first, then `("borders", ...)`. `StructValue::GetChildren` returns children in field-declaration order, so `dist_children[0]` is logits and `dist_children[1]` is borders.

**NULL handling:** Check `dist_val.IsNull()` before calling `StructValue::GetChildren`. Also check that `dist_children[0].IsNull()` (logits) and `dist_children[1].IsNull()` (borders) before calling `ListValue::GetChildren` — a NULL-targeted training row in Phase 2 can have a NULL `yhat_dist` field even when `output_mode='distribution'` is set.

---

## Exact CRPS Math for a Bar Distribution

### Derivation (Piecewise-Uniform Density Convention)

[VERIFIED: .planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md:159-167]

The CRPS is defined as:

```
CRPS(F, y) = integral_{-inf}^{+inf} (F(t) - 1{t >= y})^2 dt
```

For the bar distribution, the CDF F(t) is piecewise-linear (each bin has uniform density p_i / w_i where w_i = b_{i+1} - b_i):

```
F(b_0) = 0
F(t)   = F(b_i) + p_i * (t - b_i) / w_i   for t in [b_i, b_{i+1}]
F(b_K) = 1
```

Let C_i = F(b_i) = sum_{j<i} p_j (cumulative probability up to left edge of bin i).

**Integral within bin i = [b_i, b_{i+1}]:**

Case A — y < b_i (y is to the left of this bin): the indicator is 1 throughout this bin.

```
integral_{b_i}^{b_{i+1}} (F(t) - 1)^2 dt
= integral (C_i + p_i*(t-b_i)/w_i - 1)^2 dt
```

Let a = C_i - 1 (a < 0), c = p_i / w_i. Then:
```
= integral_0^{w_i} (a + c*s)^2 ds
= [a^2*s + a*c*s^2 + c^2*s^3/3]_0^{w_i}
= a^2*w_i + a*c*w_i^2 + c^2*w_i^3/3
= w_i * (a^2 + a*c*w_i + c^2*w_i^2/3)
= w_i * ((C_i-1)^2 + (C_i-1)*p_i + p_i^2/3)
```

Case B — y > b_{i+1} (y is to the right of this bin): the indicator is 0 throughout.

```
integral_{b_i}^{b_{i+1}} F(t)^2 dt
= integral_0^{w_i} (C_i + c*s)^2 ds
= w_i * (C_i^2 + C_i*p_i + p_i^2/3)
```

Case C — y in [b_i, b_{i+1}] (y is inside this bin): split at y.

Left part [b_i, y]: indicator = 0.

```
integral_{b_i}^{y} F(t)^2 dt
```

Let d = y - b_i (d in [0, w_i]). Using same substitution:
```
= d * (C_i^2 + C_i * p_i * d/w_i + p_i^2 * (d/w_i)^2 / 3)
```

Actually the integral of (C_i + c*s)^2 from 0 to d = C_i^2*d + C_i*c*d^2 + c^2*d^3/3 where c = p_i/w_i.

Right part [y, b_{i+1}]: indicator = 1.

```
integral_y^{b_{i+1}} (F(t) - 1)^2 dt
```

Let e = b_{i+1} - y = w_i - d. With substitution s from 0 to e:
F(y + s) = C_i + p_i*(d+s)/w_i, so (F(y+s)-1) = (C_i - 1 + p_i*(d+s)/w_i).
Let a2 = C_i - 1 + p_i*d/w_i (the CDF at y, minus 1), c = p_i/w_i:
```
= integral_0^{e} (a2 + c*s)^2 ds
= a2^2*e + a2*c*e^2 + c^2*e^3/3
```

### Closed-Form Summary — Per-Bin Contribution

Given: probs p_i (softmax of logits), borders b_0..b_K, observation y, cumulative sums C_i = sum_{j<i} p_j:

```
w_i = b_{i+1} - b_i      (bin width, non-uniform)
c_i = p_i / w_i          (density in bin i)

If y < b_i  (bin is entirely to the right of y):
    contribution_i = w_i * (C_i^2 + C_i * p_i + p_i^2 / 3)

If y > b_{i+1} (bin is entirely to the left of y):
    contribution_i = w_i * ((C_i - 1)^2 + (C_i - 1) * p_i + p_i^2 / 3)

If b_i <= y <= b_{i+1} (y falls inside this bin):
    d = y - b_i                         (distance from left edge)
    e = b_{i+1} - y  = w_i - d          (distance to right edge)
    F_y = C_i + p_i * d / w_i           (CDF at y, between C_i and C_{i+1})

    Left part [b_i, y], indicator=0:
        L = C_i^2*d + C_i*(p_i/w_i)*d^2 + (p_i/w_i)^2*d^3/3

    Right part [y, b_{i+1}], indicator=1:
        a2 = F_y - 1                    (negative: F_y < 1 inside any bin)
        R = a2^2*e + a2*(p_i/w_i)*e^2 + (p_i/w_i)^2*e^3/3

    contribution_i = L + R
```

**CRPS = sum over all bins i of contribution_i**

**Edge cases:**
- y < b_0: all bins are to the right of y; all use Case B with C_i = cumulative sum.
  Equivalently, there's an implicit left tail bin from -inf to b_0 that contributes `(0 - 1)^2 * 0 = 0` because its width is "captured" by the half-normal tail. However, for the finite-support scoring rule (integrating only over the explicit border range), we DO sum over all K bins using their indicator state relative to y.
- y > b_K: all bins use Case A (y is to the right).
- Empty or zero-width bin (w_i = 0): skip (contribution = 0). Do NOT divide by w_i in this case.

**FullSupportBarDistribution note:** The outer-bin half-normal tails affect the MEAN and QUANTILE decode (via `DistributionMean`), but for CRPS over the bar distribution's explicit CDF (the piecewise-linear piece defined by the borders), the tails extend the integration range to (-inf, b_0) and (b_K, +inf). The contribution from [b_0, +inf) left tail when y < b_0, and from (-inf, b_K] right tail when y > b_K, is:

For y <= b_0, left tail integral: `integral_{-inf}^{b_0} (0 - 1)^2 dt` diverges unless we use the half-normal extension. However, in practice the Phase 2 CONTEXT.md says "use actual bucket widths" and integrates over the explicit bin range. The prudent approach: **integrate only over [b_0, b_K]** using the piecewise-linear CDF, and note that:
- For y < b_0: add a penalty term for the mass "outside" the left tail: `1.0 * (b_0 - y)` would be appropriate if we extend, but the standard approach for bar distributions is to use the **energy-score identity** which avoids tail extrapolation.

[ASSUMED] The exact tail treatment for y outside [b_0, b_K] — whether to clip y to the border range or to compute a truncated integral — is a discretion area. **Recommended safe choice:** treat y < b_0 and y > b_K as edge cases where all K bins contribute using their respective Case B or Case A formulas (same formula; no additional tail integral needed). This integrates only over [b_0, b_K] and gives a well-defined finite CRPS for all y values.

### Alternative via Energy Score Identity (Cross-Check)

The CRPS for any predictive distribution F equals:
```
CRPS(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
```

where X, X' are independent draws from F. For a discrete categorical distribution over K bins, this becomes a weighted sum that is O(K^2) per row vs O(K) for the per-bin integral above. The per-bin integral is preferred for performance.

---

## Log-Score Math (PSR-02)

Log-score = NLL = −log(density at y).

For y in bin i: density = p_i / w_i, so log_score = −log(p_i / w_i) = log(w_i) - log(p_i).

```
clip_eps = 1e-10   (chosen; avoids log(0) = -inf)
p_clipped = max(clip_eps, p_i)

log_score(y, probs, borders) =
    if y not in any bin [b_0, b_K]:
        return -log(clip_eps)    // out-of-support: maximum penalty
    else:
        i = bin containing y
        w_i = b_{i+1} - b_i
        if w_i < 1e-300: return -log(clip_eps)   // degenerate bin
        return -(log(p_clipped) - log(w_i))
        = log(w_i) - log(p_clipped)
```

The aggregate computes the mean: `avg_log_score = (1/n) * sum_i log_score(y_i, ...)`.

**Epsilon choice:** `1e-10` is appropriate (larger than `1e-15` used for classification log-loss since the bar distribution can have genuinely small but non-zero bin mass in wide outer bins). Document the epsilon in code comments.

---

## Interval Score Math (PSR-03)

The interval score at coverage level c uses the (1-c)/2 and 1-(1-c)/2 quantiles:

```
alpha = 1.0 - coverage     (e.g., alpha=0.1 for coverage=0.9)
q_lo  = alpha / 2          (e.g., 0.05 for coverage=0.9)
q_hi  = 1.0 - alpha / 2   (e.g., 0.95 for coverage=0.9)

l = DistributionQuantile(probs, borders, q_lo)
u = DistributionQuantile(probs, borders, q_hi)

penalty_lo = (l - y) * (y < l ? 1.0 : 0.0)
penalty_hi = (y - u) * (y > u ? 1.0 : 0.0)

interval_score = (u - l) + (2.0 / alpha) * (penalty_lo + penalty_hi)
```

This is the Gneiting & Raftery (2007) interval score definition, lower is better. The `2/alpha` sharpness penalty makes under-coverage costly.

**Coverage parameter:** named `coverage`, default 0.9, valid range (0, 1) exclusive. Validate at bind time: `throw BinderException` if coverage <= 0.0 or coverage >= 1.0.

**State:** `{double sum_score; int64_t n}` — same shape as RMSE state. No heap allocation.

**Bind data:** stores the resolved coverage value (extracted at bind time from a constant expression, same idiom as `AUCBind`/`BindF1Metric`).

---

## Bind-Gating (PSR-04) Pattern

[VERIFIED: src/tabfm_metrics_classification.cpp:1077-1087]

The exact pattern from Phase 1 for a 2-arg overload that always throws at bind:

```cpp
// 2-arg overload — rejects plain DOUBLE second argument (PSR-04)
AggregateFunction fn_point(
    "anofox_tabfm_crps",
    {LogicalType::DOUBLE, LogicalType::DOUBLE},   // (actual, plain_estimate)
    LogicalType::DOUBLE,
    CRPSStateSize, CRPSStateInit, CRPSUpdate, CRPSCombine, CRPSFinalize,
    /*simple_update=*/nullptr,
    [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
        -> unique_ptr<FunctionData> {
        throw InvalidInputException(
            "tabfm_crps: second argument must be a distribution "
            "STRUCT(logits DOUBLE[], borders DOUBLE[]) — use "
            "output_mode := 'distribution' in tabfm_regress.");
        return nullptr;
    },
    /*state_destroy=*/nullptr);
set.AddFunction(fn_point);
```

The 2-arg point-estimate overload is registered AFTER the full STRUCT overload so DuckDB's overload resolution prefers the exact-type match for STRUCT arguments and falls through to the point-estimate overload only when the second argument is a plain DOUBLE.

**STRUCT type for bind:** The exact STRUCT type to register for the 2-arg scored overload:

```cpp
LogicalType yhat_dist_type = LogicalType::STRUCT({
    {"logits",  LogicalType::LIST(LogicalType::DOUBLE)},
    {"borders", LogicalType::LIST(LogicalType::DOUBLE)}
});
// Register with {LogicalType::DOUBLE, yhat_dist_type} as input types
```

[VERIFIED: src/tabfm_predict_agg.cpp:99-103] This is exactly the type declared in `ListStructType()` for the `yhat_dist` field.

**DuckDB v1.5.4 does NOT implicitly cast arg types for aggregates** (confirmed in CONTEXT.md and CLAUDE.md). The STRUCT type in the registration must match exactly what Phase 2 emits — field names `"logits"` and `"borders"`, both `LIST(DOUBLE)`.

---

## Aggregate State Design

### CRPS State

```cpp
struct CRPSState {
    double  sum_crps; // running sum of per-row CRPS values
    int64_t n;        // non-NULL valid rows
};
// sizeof(CRPSState) = 16 bytes — fits in fixed-size state slot, no StateDestroy needed
```

### Log-Score State

```cpp
struct LogScoreState {
    double  sum_ls; // running sum of per-row log-scores
    int64_t n;
};
```

### Interval Score State

```cpp
struct IScoreState {
    double  sum_is;   // running sum of per-row interval scores
    int64_t n;
    double  coverage; // snapshot from bind data (needed in Update via bind_data)
};
// Alternative: store coverage in BindData and read via aggr_input.bind_data in Update.
// Preferred: read coverage from bind data in Update, keep state small.
```

**For interval score, coverage is available in Update via `aggr_input.bind_data`:**

```cpp
void IScoreUpdate(Vector inputs[], AggregateInputData &aggr_input, idx_t,
                  Vector &state_vector, idx_t count) {
    auto &bd = aggr_input.bind_data->Cast<IScoreBindData>();
    double coverage = bd.coverage;
    // ...
}
```

[VERIFIED: src/tabfm_metrics_classification.cpp:335-364] This is exactly how `F1Finalize` reads `aggr_input.bind_data->Cast<F1BindData>()`. The same pattern applies to Update.

---

## Softmax in the Aggregate Update

`SoftmaxInPlace` is defined inside an anonymous namespace in `tabfm_engine.cpp` and is NOT in external scope. The scoring aggregate Update must implement its own softmax or copy the function locally within the anonymous namespace in `tabfm_scoring.cpp`.

[VERIFIED: src/tabfm_engine.cpp:649-665] `SoftmaxInPlace` is at line 649, inside `namespace { ... }` at line 667 — NOT given external linkage. It is NOT declared in any header.

**Recommended:** Copy the identical implementation into an anonymous namespace helper in `tabfm_scoring.cpp`. It is 17 lines; duplication is acceptable for this utility.

```cpp
// In anonymous namespace of tabfm_scoring.cpp:
void SoftmaxInPlace(std::vector<double> &v) {
    // temperature = 1.0 per bar distribution contract
    if (v.empty()) return;
    double m = v[0];
    for (auto x : v) { m = std::max(m, x); }
    double sum = 0.0;
    for (auto &x : v) { x = std::exp(x - m); sum += x; }
    if (sum > 0.0) { for (auto &x : v) { x /= sum; } }
}
```

For interval score, `DistributionQuantile` IS externally linked (declared in `tabfm_predict.hpp`), so include `"tabfm_predict.hpp"` in `tabfm_scoring.cpp`. [VERIFIED: src/include/tabfm_predict.hpp:206]

---

## CRPS C++ Implementation Skeleton

```cpp
// In anonymous namespace of tabfm_scoring.cpp:
double ComputeCRPS(double y, const std::vector<double> &probs,
                   const std::vector<double> &borders) {
    const size_t K = probs.size();
    // Guard: K must match borders.size() - 1
    if (K == 0 || borders.size() != K + 1) return 0.0;

    double crps = 0.0;
    double cum = 0.0; // C_i = cumulative probability up to left edge of bin i

    for (size_t i = 0; i < K; i++) {
        const double b_lo = borders[i];
        const double b_hi = borders[i + 1];
        const double wi   = b_hi - b_lo;
        const double pi   = probs[i];

        if (wi <= 0.0) {
            cum += pi;
            continue; // degenerate bin — skip
        }

        if (y < b_lo) {
            // Case A: y is to the left of this bin; indicator = 1 throughout
            // integral = w_i * ((C_i - 1)^2 + (C_i - 1)*p_i + p_i^2/3)
            double a = cum - 1.0;
            crps += wi * (a * a + a * pi + pi * pi / 3.0);
        } else if (y > b_hi) {
            // Case B: y is to the right of this bin; indicator = 0 throughout
            // integral = w_i * (C_i^2 + C_i*p_i + p_i^2/3)
            crps += wi * (cum * cum + cum * pi + pi * pi / 3.0);
        } else {
            // Case C: y is inside this bin
            const double d   = y - b_lo;      // distance from left edge
            const double e   = b_hi - y;      // distance to right edge
            const double c   = pi / wi;       // density

            // Left part [b_lo, y]: indicator = 0
            // integral = C_i^2*d + C_i*c*d^2 + c^2*d^3/3
            double L = cum * cum * d + cum * c * d * d + c * c * d * d * d / 3.0;

            // Right part [y, b_hi]: indicator = 1
            // a2 = CDF(y) - 1 = (C_i + p_i*d/w_i) - 1
            double fy = cum + pi * d / wi;
            double a2 = fy - 1.0;
            double R  = a2 * a2 * e + a2 * c * e * e + c * c * e * e * e / 3.0;

            crps += L + R;
        }

        cum += pi;
    }
    return crps;
}
```

**Numerical notes:**
- `cum` accumulates floating-point error over K iterations. For K=16 (fixture) this is negligible; for K=5000 (real TabPFN v2) consider Kahan summation in the cumulative — [ASSUMED] the fixture K=16 is the only tested case in this phase; K=5000 is deferred.
- The `pi * pi / 3.0` term: divide by 3, not multiply by 0.333... (avoid precision loss).
- Check `wi <= 0.0` not `wi == 0.0` in case of floating-point underflow.

---

## Golden Testing Strategy

### Option A (Recommended): Hard-coded values derived from documented computation

The fixture `test/fixtures/tabpfn_v2/golden.json` provides exact logits (K=16), borders, y_mean, y_std, and decoded raw_borders for 2 test rows. CRPS reference values can be computed once in Python (in `tools/parity/src/parity/crps_reference.py`) and hard-coded in the C++ Catch2 test alongside the computation they represent.

**Python reference for CRPS:**

```python
import numpy as np

def crps_bar(y: float, logits: list, borders: list) -> float:
    """CRPS over bar distribution with piecewise-uniform density."""
    probs = np.array(logits, dtype=np.float64)
    probs -= probs.max()
    probs = np.exp(probs)
    probs /= probs.sum()
    borders = np.array(borders, dtype=np.float64)
    K = len(probs)
    assert len(borders) == K + 1
    crps = 0.0
    cum = 0.0
    for i in range(K):
        b_lo, b_hi = borders[i], borders[i+1]
        wi = b_hi - b_lo
        pi = probs[i]
        if wi <= 0:
            cum += pi
            continue
        if y < b_lo:
            a = cum - 1.0
            crps += wi * (a*a + a*pi + pi*pi/3)
        elif y > b_hi:
            crps += wi * (cum*cum + cum*pi + pi*pi/3)
        else:
            d = y - b_lo
            e = b_hi - y
            c = pi / wi
            L = cum*cum*d + cum*c*d*d + c*c*d*d*d/3
            fy = cum + pi*d/wi
            a2 = fy - 1.0
            R = a2*a2*e + a2*c*e*e + c*c*e*e*e/3
            crps += L + R
        cum += pi
    return crps
```

Run this on the golden.json test rows (using `raw_borders` for y-space scoring) to produce a reference CRPS value that is hard-coded in `test/cpp/test_tabfm_scoring.cpp`.

**golden.json test row 0:** logits = [0.30400875..., 0.03311837..., ...] (16 values at line 99-116), actual y = 0.3129126727581024 (first training row) or some test row y. For the C++ test, use a simple synthetic case with K=4 bins where the CRPS can be hand-verified, then use the golden fixture for the end-to-end SQL test.

### Option B: Add to tools/parity

Add `crps_reference.py` to `tools/parity/src/parity/` computing reference CRPS/log-score/interval values from the golden.json fixture and printing them for inclusion in the test. This matches the MODL-04 pattern.

**Recommendation:** Use Option A (hard-coded in C++ test) for the Catch2 unit test (simple synthetic K=4 case). For the SQL test, use the golden.json raw_borders and a specific y value with a tolerance of 1e-6.

---

## CMP-01 Macro Pattern

[VERIFIED: src/tabfm_crossval.cpp:128-250]

CMP-01 is a thin SQL table macro that runs Phase-1 metrics AND Phase-3 scores for two model families on the user's table. It follows the same `CVMacroDef` pattern as `tabfm_cross_validate`.

**Design:** A macro `tabfm_compare_models(data, target, actual_col)` that:
1. Runs `tabfm_regress(data, target, opts := MAP{'output_mode':'distribution', 'model':'tabfm-v1'})` and collects `(actual, yhat, yhat_dist)`.
2. Runs the same with `model='tabpfn_v2'` (fixture-scoped, requires `SET anofox_tabfm_model_manifest`).
3. Computes `tabfm_rmse`, `tabfm_crps`, `tabfm_log_score` for each model.
4. Returns a UNION of metric rows with a `model` column.

**Simpler alternative (also valid for CMP-01):** A documented SQL pattern (no C++ macro) showing the user how to compose two `SELECT tabfm_rmse(...), tabfm_crps(...) FROM tabfm_regress(...)` queries UNION-ed with a model label. This avoids a new C++ macro registration.

**CONTEXT.md decision:** "a thin helper macro" — the macro form is locked. Use the same `PredictMacroDef` / `CVMacroDef` struct pattern in `tabfm_crossval.cpp` or `tabfm_macros.cpp`.

**Safe identifier quoting in macro body:** carry forward `replace(target, '"', '""')` quoting wherever the target column is interpolated into a SQL string. [VERIFIED: src/tabfm_crossval.cpp:209] shows `replace(CAST(target AS VARCHAR), '"', '""')`.

**Macro registration pattern:** same as `RegisterCrossValidateMacros` — register in new `RegisterScoringFunctions` or extend `RegisterCrossValidateMacros`. Both are planner decisions.

---

## Scaffold-Owned Files (Coordinate — CLAUDE.md Rule #2)

[VERIFIED: src/include/tabfm_registration.hpp:1-26]

The following files require batch coordination:

| File | Required Change | Risk |
|---|---|---|
| `CMakeLists.txt` | Add `src/tabfm_scoring.cpp` to `EXTENSION_SOURCES`; add `test/cpp/test_tabfm_scoring.cpp` to `TABFM_CPP_TEST_SOURCES` | Build breaks if omitted |
| `src/anofox_tabfm_extension.cpp` | Add `RegisterScoringFunctions(loader)` call in `LoadInternal` | Functions not registered at runtime |
| `src/include/tabfm_registration.hpp` | Add `void RegisterScoringFunctions(ExtensionLoader &loader);` declaration | Compile error if missing |

No changes needed to `src/tabfm_settings.cpp` (no new settings for scoring functions — coverage is a function argument, not a global setting).

---

## Registration Pattern (verbatim from Phase 1)

[VERIFIED: src/tabfm_metrics_classification.cpp:1159-1176]

The `tabfm_log_loss` registration (simplest 1-overload aggregate with a MAP argument) is the closest analog for PSR-02. The PSR aggregate set adds a second (2-arg DOUBLE) overload for the bind-gate:

```cpp
void RegisterScoringFunctions(ExtensionLoader &loader) {
    // --- tabfm_crps / anofox_tabfm_crps (PSR-01) ---
    {
        LogicalType yhat_dist_type = LogicalType::STRUCT({
            {"logits",  LogicalType::LIST(LogicalType::DOUBLE)},
            {"borders", LogicalType::LIST(LogicalType::DOUBLE)}
        });

        AggregateFunctionSet set("anofox_tabfm_crps");

        // Normal 2-arg overload: (actual DOUBLE, yhat_dist STRUCT(logits,borders))
        AggregateFunction fn_dist("anofox_tabfm_crps",
            {LogicalType::DOUBLE, yhat_dist_type}, LogicalType::DOUBLE,
            CRPSStateSize, CRPSStateInit, CRPSUpdate, CRPSCombine, CRPSFinalize,
            /*simple_update=*/nullptr, CRPSBind, /*state_destroy=*/nullptr);
        set.AddFunction(fn_dist);

        // Bind-gate: (actual DOUBLE, plain_estimate DOUBLE) → always throws (PSR-04)
        AggregateFunction fn_point("anofox_tabfm_crps",
            {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
            CRPSStateSize, CRPSStateInit, CRPSUpdate, CRPSCombine, CRPSFinalize,
            /*simple_update=*/nullptr,
            [](ClientContext &, AggregateFunction &, vector<unique_ptr<Expression>> &)
                -> unique_ptr<FunctionData> {
                throw InvalidInputException(
                    "tabfm_crps: second argument must be a distribution "
                    "STRUCT(logits DOUBLE[], borders DOUBLE[]). "
                    "Use output_mode := 'distribution' in tabfm_regress.");
                return nullptr;
            }, /*state_destroy=*/nullptr);
        set.AddFunction(fn_point);

        FunctionDescription fd;
        fd.description = "...";
        fd.examples = {"SELECT tabfm_crps(actual, yhat_dist) FROM tabfm_regress('tbl', 'y', opts := MAP{'output_mode':'distribution'});"};
        RegisterAggregateFunctionSetWithAlias(loader, set, "tabfm_crps", {std::move(fd)});
    }
    // tabfm_log_score and tabfm_interval_score follow the same pattern
}
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Quantile computation for interval score | Custom bisection/binary search | `DistributionQuantile` (tabfm_predict.hpp:206, external linkage) | Already implements CDF cumsum search + linear interpolation over non-uniform bins; tested against golden fixture |
| Softmax | Custom exp/sum loop | Copy `SoftmaxInPlace` pattern from tabfm_engine.cpp:649 | Numerically stable (subtract max before exp); 17 lines; the exact same operation as Phase 2 |
| AggregateFunctionSet + alias registration | Custom registration glue | `RegisterAggregateFunctionSetWithAlias` (anofox_function_alias.hpp) | Handles full name + short alias atomically; FunctionDescription attached |
| CDF construction | Manual cumsum array | Accumulate inline in `ComputeCRPS` loop | No separate array needed; `cum` variable tracks cumulative sum per bin |

---

## Common Pitfalls

### Pitfall 1: Using Uniform Bin Width Assumption

**What goes wrong:** Computing CRPS or log-score with `w_i = 1/K` instead of actual `borders[i+1] - borders[i]`. Gives plausible-looking but wrong values for any non-fixture model. [VERIFIED: .planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md:89] "Assuming uniform bins yields plausible-but-wrong quantiles/CRPS." Width ratio is ~28,000:1 for real TabPFN v2 borders.

**How to avoid:** Always use `w_i = borders[i+1] - borders[i]`. Never hardcode `1.0/K`.

### Pitfall 2: Calling StructValue::GetChildren on a NULL Value

**What goes wrong:** Crash or UB if `dist_val.IsNull()` is true when `StructValue::GetChildren` is called.

**How to avoid:** Check `if (dist_val.IsNull()) { continue; }` before calling `StructValue::GetChildren`. Also check that the nested LIST children are not NULL.

### Pitfall 3: STRUCT Type Mismatch in Registration

**What goes wrong:** Registering the aggregate with `LogicalType::ANY` or a different STRUCT type causes DuckDB to either not match the Phase 2 `yhat_dist` column type or to silently cast. DuckDB v1.5.4 does NOT implicitly cast aggregate args.

**How to avoid:** Use the exact STRUCT type `{{"logits", LIST(DOUBLE)}, {"borders", LIST(DOUBLE)}}` matching Phase 2's `dist_struct_type` at `src/tabfm_predict_agg.cpp:594-596`. Field order matters.

### Pitfall 4: Zero-Width Bins

**What goes wrong:** Division by zero in log-score (`−log(p_i / w_i)` with `w_i = 0`) or in CRPS Case C (`d = 0; e = 0`).

**How to avoid:** In `ComputeCRPS`, check `if (wi <= 0.0) { cum += pi; continue; }`. In log-score, check `if (w_i < 1e-300) { return -log(clip_eps); }`.

### Pitfall 5: Coverage Parameter Not Stored in Bind Data

**What goes wrong:** The coverage parameter for `tabfm_interval_score` is a constant at bind time. If it is not captured in bind data and carried to Update, it cannot be accessed in the per-row Update callback (which has no `ClientContext`).

**How to avoid:** Use a `IScoreBindData : FunctionData` struct (same pattern as `F1BindData` at `src/tabfm_metrics_classification.cpp:205-219`) to store the coverage value. Read it in Update via `aggr_input.bind_data->Cast<IScoreBindData>().coverage`.

### Pitfall 6: Registering 2-arg Bind-Gate BEFORE 3-arg Normal Overload

**What goes wrong:** DuckDB resolves overloads in registration order; if the DOUBLE overload is added first and a STRUCT is passed, DuckDB may fail to resolve to the STRUCT overload.

**How to avoid:** Add the STRUCT overload (fn_dist) first, then the DOUBLE/point overload (fn_point). [VERIFIED: src/tabfm_metrics_classification.cpp:1071-1088] Phase 1 adds fn3 (3-arg) before fn2 (2-arg throw).

### Pitfall 7: y Outside [b_0, b_K] — Log-Score

**What goes wrong:** If y < b_0 or y > b_K, no bin contains y. A naive "find bin containing y" loop exits without setting `i`, and an uninitialized `p_i` or `w_i` is used.

**How to avoid:** Before the bin search loop, check if y < b_0 or y > b_K and return the max-penalty value (`−log(clip_eps)`) immediately.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Catch2 (embedded in DuckDB build) |
| Config file | `CMakeLists.txt` (`TABFM_CPP_TEST_SOURCES`) |
| Quick run command | `./build/debug/test/unittest test/cpp/test_tabfm_scoring.cpp` |
| SQL test run | `./build/debug/test/unittest test/sql/tabfm_scoring.test` |
| Full suite command | `make test_debug` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PSR-01 | CRPS over bar distribution (K=16 fixture + synthetic K=4) | unit + SQL | `./build/debug/test/unittest test/cpp/test_tabfm_scoring.cpp` | ❌ Wave 0 |
| PSR-02 | Log-score NLL, out-of-support penalty, epsilon clip | unit + SQL | `./build/debug/test/unittest test/cpp/test_tabfm_scoring.cpp` | ❌ Wave 0 |
| PSR-03 | Interval score at coverage=0.9 and coverage=0.5 | unit + SQL | `./build/debug/test/unittest test/cpp/test_tabfm_scoring.cpp` | ❌ Wave 0 |
| PSR-04 | Bind-gate: plain DOUBLE → throw with named remedy | SQL | `./build/debug/test/unittest test/sql/tabfm_scoring.test` | ❌ Wave 0 |
| CMP-01 | Cross-model macro, tabpfn_v2 fixture vs tabfm-v1 | SQL | `./build/debug/test/unittest test/sql/tabfm_scoring.test` | ❌ Wave 0 |

### Wave 0 Gaps

- [ ] `test/cpp/test_tabfm_scoring.cpp` — covers PSR-01/02/03 unit math
- [ ] `test/sql/tabfm_scoring.test` — covers PSR-01..04 end-to-end + CMP-01

---

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Validate coverage in (0,1) at bind; validate K>0 and borders.size()==K+1 in Update; clip epsilon for log-score |
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Division by zero (zero-width bin, empty distribution) | Denial of Service | Guard `wi <= 0.0` in ComputeCRPS; guard `n==0` in Finalize |
| Log of zero (zero-probability bin) | Denial of Service | Clip: `p_clipped = max(clip_eps, p_i)` |
| SQL injection via target in CMP-01 macro | Tampering | `replace(target, '"', '""')` at every interpolation (CV-04 pattern) |
| Out-of-support y (log-score, interval score) | Incorrect results | Return max-penalty for y outside [b_0, b_K]; clamp q to (0,1) in DistributionQuantile |

---

## Environment Availability

Step 2.6: SKIPPED — Phase 3 is code-only (new C++ aggregates + SQL macros). No new external tools, services, or runtimes beyond the existing build chain.

---

## Runtime State Inventory

Step 2.5: SKIPPED — Phase 3 is greenfield new functionality. No rename/refactor/migration.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Proper scoring rules only in Python (properscoring, scoringrules) | SQL aggregates over the predictive distribution column | Phase 3 (this) | Users can compute CRPS/NLL in-database without exporting data |
| Uniform bin width assumption for CRPS | Non-uniform borders from checkpoint (`borders` field in yhat_dist) | Phase 2 confirmed | Correct CRPS for TabPFN v2's highly non-uniform bin structure |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | CRPS tail treatment: integrating only over [b_0, b_K] using the piecewise-linear CDF (no half-normal tail extension) is acceptable for the scoring rule | CRPS Math | If tails matter for specific y values, CRPS will underestimate the true score; deferred to ASCR-01 |
| A2 | log-score epsilon = 1e-10 is appropriate for bar distributions | Log-Score Math | Different epsilon could change behavior on near-zero bins; documented so it is reviewable |
| A3 | `StructValue::GetChildren` returns children in field-declaration order (logits first, borders second) for the yhat_dist STRUCT | STRUCT Reading | If DuckDB does not guarantee order, index must use field-name lookup; current codebase evidence (predict_agg.cpp:617-618) shows order-based access is used |

**If this table is empty:** N/A — 3 assumptions logged above.

---

## Open Questions

1. **Tail treatment for CRPS when y is outside [b_0, b_K]**
   - What we know: the per-bin integral covers [b_0, b_K] exactly; y outside this range still has a valid CRPS contribution from all K bins (all bins are on one side of y)
   - What's unclear: whether to add a half-normal tail integral for the FullSupportBarDistribution outer bins when y is far outside the border range
   - Recommendation: Implement without tail extension first (A1); the golden test with y in [b_0, b_K] verifies the core logic; document the limitation

2. **Coverage parameter at bind time vs. runtime**
   - What we know: DuckDB allows reading constant expressions at bind via `ExpressionExecutor::EvaluateScalar`; if coverage is not a constant, it cannot be validated early
   - What's unclear: whether users will want to pass a non-constant coverage (e.g., a column)
   - Recommendation: Require constant coverage (same pattern as `avg` in F1 — validate at bind, throw if not constant)

---

## Sources

### Primary (HIGH confidence)
- `src/tabfm_metrics_classification.cpp` (read this session) — 2-arg bind-gate pattern lines 1077-1088; MAP/STRUCT reading lines 466-482; AggregateFunctionSet registration lines 1065-1211
- `src/tabfm_metrics_regression.cpp` (read this session) — aggregate state/update/combine/finalize pattern lines 53-130
- `src/tabfm_predict_agg.cpp` (read this session) — STRUCT type declaration lines 99-103; yhat_dist assembly lines 594-626; bind data pattern lines 36-107
- `src/tabfm_engine.cpp` (read this session) — SoftmaxInPlace lines 649-665; DistributionMean lines 675-701; DistributionQuantile lines 703-738; DecodeDistribution lines 896-976
- `src/include/tabfm_predict.hpp` (read this session) — DistributionMean/DistributionQuantile declarations lines 193/206
- `src/include/tabfm_registration.hpp` (read this session) — registration pattern lines 1-26
- `src/tabfm_crossval.cpp` (read this session) — CMP-01 macro pattern, quoting conventions lines 128-250
- `src/tabfm_macros.cpp` (read this session) — predict macro pattern lines 61-104
- `.planning/spikes/SPIKE-tabpfn-v2-tensor-contract.md` (read this session) — confirmed distribution contract, CRPS requirement for non-uniform borders
- `test/fixtures/tabpfn_v2/golden.json` (read this session) — K=16 fixture logits, borders, raw_borders, decoded values
- `test/sql/tabfm_distribution.test` (read this session) — SQL test pattern for distribution output
- `.planning/phases/02-model-generalization-distribution-output-fixture-backed/02-PATTERNS.md` (read this session) — Phase 2 patterns map

### Secondary (MEDIUM confidence)
- Gneiting & Raftery (2007) interval score formula — `(u-l) + (2/alpha) * [(l-y)*1{y<l} + (y-u)*1{y>u}]` [ASSUMED from training knowledge; formula is standard in probabilistic forecasting literature]
- Properscoring literature for CRPS identity over bar distributions [ASSUMED from training knowledge; cross-verified against the per-bin integral derivation above]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages, all patterns in codebase
- Architecture: HIGH — verified from Phase 1/2 source files
- Scoring rule math (CRPS/log-score/interval): HIGH for log-score and interval score; MEDIUM for exact CRPS tail treatment (documented as A1 assumption)
- Bind-gate pattern: HIGH — verbatim from Phase 1 source
- STRUCT reading idiom: HIGH — confirmed from Phase 1 MAP reading + Phase 2 STRUCT assembly

**Research date:** 2026-09-22
**Valid until:** Stable (no external dependencies change); valid indefinitely for this pinned DuckDB v1.5.4 codebase
