# Phase 3: Proper Scoring Rules + Cross-Model Comparison - Context

**Gathered:** 2026-09-22
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous)

<domain>
## Phase Boundary

Users can score probabilistic regression predictions with proper scoring rules
(CRPS, log-score, interval score) over the Phase 2 predictive-distribution output,
and compare model families head-to-head on their own tables using the Phase 1
evaluation primitives. Delivers PSR-01, PSR-02, PSR-03, PSR-04, CMP-01.

The scoring rules operate on `(actual, yhat_dist)` columns and are therefore
built and golden-tested against the CONFIRMED TabPFN v2 distribution contract via
the Phase 2 weight-free fixture — they do not require real TabPFN v2 weights.

</domain>

<decisions>
## Implementation Decisions

### Scoring-rule aggregates (PSR-01/02/03/04)
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

### Cross-model comparison (CMP-01)
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

</decisions>

<code_context>
## Existing Code Insights

### Reusable assets / seams
- **Distribution output (Phase 2):** `output_mode='distribution'` emits
  `yhat_dist STRUCT(logits DOUBLE[], borders DOUBLE[])` + `yhat_quantiles DOUBLE[]`
  (`src/tabfm_predict_agg.cpp`, `src/tabfm_engine.cpp` DecodeDistribution). The
  decode helpers `DistributionMean`/`DistributionQuantile` (external linkage in
  `tabfm_engine.cpp`) and the softmax/CDF-over-non-uniform-borders math are the
  reference for the scoring math.
- **Aggregate pattern:** the Phase-1 metric aggregates (`tabfm_metrics_*.cpp`) are
  the template — bind/update/combine/finalize, `AggregateFunctionSet` +
  `RegisterAggregateFunctionSetWithAlias`, telemetry-at-bind, NULL-skip,
  heap-owning state + StateDestroy where needed. Reading a STRUCT(list,list)
  column in Update: use `StructValue::GetChildren` + `ListValue::GetChildren`
  (cf. the proba-MAP reading pattern).
- **Required-param bind-gate pattern:** the Phase-1 required-`avg` 2-arg overload
  that throws at bind is the model for PSR-04's point-estimate rejection.
- **Comparison / macro pattern:** `tabfm_cross_validate` / `tabfm_macros.cpp`
  (query() composition, safe identifier quoting) is the analog for the CMP-01
  helper macro.
- **Fixture:** `test/fixtures/tabpfn_v2/` gives a real distribution output to
  golden-test the scoring rules end-to-end.

### Scaffold-owned (coordinate — CLAUDE.md rule #2)
- `CMakeLists.txt` source + test lists, `src/anofox_tabfm_extension.cpp` Load(),
  `src/include/tabfm_registration.hpp` — batch the new scoring module registration.

### Conventions
- Full `anofox_tabfm_*` + short `tabfm_*` alias; telemetry once per user-facing
  function; errors name the fixing SET/CALL/param; red-green TDD; cpu-flavor-clean;
  DuckDB v1.5.4 aggregates do NOT implicitly cast arg types — see
  [[duckdb-aggregate-any-cast]] (bind on the exact STRUCT type).

</code_context>

<specifics>
## Specific Ideas

- Function names pinned by ROADMAP success criteria: `tabfm_crps(actual,
  yhat_dist)`, `tabfm_log_score(actual, yhat_dist)`,
  `tabfm_interval_score(actual, yhat_dist, coverage := 0.9)`.
- CRPS/log-score/interval must use the model-provided **non-uniform** borders.
- PSR functions bind-gate on distribution input and fail with a named remedy
  (point at `output_mode := 'distribution'`) when given point estimates.
- CMP-01 comparison runs on the user's own table across `tabpfn_v2` (fixture) and
  `tabfm-v1` — no bundled datasets.

</specifics>

<deferred>
## Deferred Ideas

- CRLS (log-score with singularity handling) and multivariate scores
  (energy/variogram) — v2 (ASCR-01).
- Calibration curves / reliability diagrams as data output — v2 (ASCR-02).
- Real TabPFN v2 inference export + TabICL comparison — deferred (Phase 2 notes);
  CMP-01 compares the fixture-scoped tabpfn_v2 vs tabfm-v1.

</deferred>
