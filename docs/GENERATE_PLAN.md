# Synthetic data generation + imputation (WS-G)

Status: **implemented**. Kept as the design record, like `MULTI_MODEL_PLAN.md`.
User-facing documentation is `docs/GENERATE.md`; this file is the reasoning
behind it. Owner: WS-G (`src/tabfm_generate.cpp`).

## Deviations from the plan as approved

Four things changed while building, all of them small and all deliberate:

1. **No `context_rows` option.** The plan mirrored it from the predict surface.
   It turns out `context_rows` is parsed and validated by `tabfm_predict_agg`
   but **never read** — neither `tabfm_preprocess.cpp` nor `tabfm_engine.cpp`
   consumes it, so it is a no-op today. Mirroring it would have shipped a
   documented knob that does nothing. (The pre-existing no-op in the predict
   surface is untouched and worth a separate look.)
2. **A relation with only one modellable column is allowed**, where the plan
   said "fewer than 2 columns → error". For a single column the marginal *is*
   the joint distribution, so sampling it is correct and complete, not a
   degraded fallback. Only a relation where *every* column is constant errors,
   since there is genuinely nothing to sample.
3. **`test/sql/tabfm_examples.test` guards signatures, not files.** The plan
   said it would "execute the SQL shipped in `examples/`". The shipped samples
   need real weights (multi-GB downloads), so CI cannot run them. The test
   instead replays every call form the samples use against the fixture model —
   a renamed parameter or option key still breaks the build, which was the
   point, but the claim in the test header is accurate about what it does.
4. **RNG moved to `src/include/tabfm_random.hpp`.** Not in the original file
   list: a shared `TabFMRandom` over `duckdb::RandomEngine` now serves both WS-G
   and the ensemble layer, replacing the removed `PyRandom` MT19937 port.
5. **`bins` is clamped to the model's class-head width**, which the plan did not
   anticipate. The plan assumed the cap was a flat 10 (the engine's label limit).
   It is not: every generation step is a *classification* problem, so `bins` can
   never exceed the model's actual output-head width. Found the hard way — the
   first end-to-end run died with `model output has 3 class column(s) but
   classification needs at least 10`, a tensor-shape error that says nothing
   about `bins`. Bind now reads `size_regime.max_classes` from the resolved model
   and clamps both `bins` and the categorical cardinality limit to it. The CI
   fixtures declare their true widths (3 for the base fixture, 4 for TabPFN),
   which is also what makes the clamp path covered by tests.
6. **Within-bin sampling is empirical, not uniform.** The plan said "draw
   uniformly inside the sampled bin". That shipped, and the breast-cancer
   benchmark caught it: for skewed columns the outermost bins are wide while the
   real mass hugs one end, so a uniform fill pushes values into the sparse tail.
   Measured cost on `area_se`: mean inflated 67% (38.99 -> 65.21), average SD
   error 0.52 across the 30 features, and visible correlation attenuation
   (0.998 -> 0.949 on the strongest pair). `ExpandBin` now samples a uniform
   position among the bin's observations and interpolates between neighbouring
   order statistics. Novelty is preserved — a draw lands strictly between two
   real values. Found only because the plots were rendered and looked at; no
   unit test would have flagged it, since every value was still legal and in
   range.

Adds two user-facing table functions on top of the existing predict engine:

```sql
tabfm_generate(data, n := 100, features := NULL, opts := MAP{}, model := NULL)
tabfm_impute  (data, columns := NULL, features := NULL, opts := MAP{}, model := NULL)
```

Both generalize to **every** model in the registry: they are wrappers over
`PredictEngine::Predict`, so they never touch ONNX, tensor contracts, graphs or
preprocessing. Gated by the `capabilities` array already present in `ModelSpec`.

---

## 1. What we are porting

`https://docs.priorlabs.ai/cookbook/generate_synthetic_data` —
`TabPFNUnsupervisedModel` treats a table as a joint density via the
**autoregressive chain rule over columns**:

```
p(c_1, …, c_H) = p(c_π(1)) · p(c_π(2) | c_π(1)) · … · p(c_π(H) | c_π(1..H-1))
```

For a column permutation π: for step *i*, fit on the real rows with
`c_π(1..i-1)` as features and `c_π(i)` as target, predict on the partially
filled synthetic rows, and **sample** from the predictive distribution at
temperature `temp` (rather than taking the argmax). Discrete column →
classifier, continuous → regressor. Step 1 has no features, so it is drawn from
the empirical marginal.

This is a wrapper over predict, not a new model. That is why it generalizes.

## 2. Why it fits this codebase

* **The seam already exists.** `__anofox_tabfm_predict_agg` materializes a group
  into `vector<vector<Value>>` and computes its `LIST(STRUCT(...))` return type
  at bind (`tabfm_predict_agg.cpp:79` `ListStructType`). A sibling aggregate has
  the same shape.
* **The engine takes an arbitrary row schema.** `PredictInput` is
  `{rows, row_type, target_idx, target_type, target_name, opts, ctx}`
  (`tabfm_predict.hpp:129`). Each chain-rule step just builds a **narrowed**
  `row_type` — only `c_π(1..i-1)` plus the step's target — and calls
  `GetPredictEngine().Predict()`. No engine change.
* **Context vs. query rows are already the right contract.** Rows with a
  non-NULL target are context, NULL-target rows are scored. Real rows always
  have the step target; synthetic rows never do. Exactly the existing semantics.
* **Deterministic RNG comes from DuckDB.** All draws use
  `duckdb::RandomEngine` (`duckdb/common/random_engine.hpp`), constructed with
  `opts['seed']`.

  **Not `PyRandom`.** That class (`tabfm_ensemble.hpp:74`) is a bit-exact
  CPython MT19937 port whose entire purpose is matching the
  `golden_preprocess.json` member configs bit-for-bit, and whose header warns
  that any extension of it is a parity hazard. Generation has no upstream
  numbers to reproduce, so it has no parity requirement — and `PyRandom`'s
  deliberately minimal surface (`GetRandBits`/`RandBelow`/`SampleRange`/
  `Shuffle`) has **no uniform double**, which uniform-within-bin expansion
  needs. Using it would mean adding `random()` to a parity-critical WS-F file
  for a WS-G feature.

  `RandomEngine` covers the whole need: `NextRandom(lo, hi)` for
  uniform-within-bin, `NextRandom()` + a cumulative scan for the weighted
  categorical draw over `proba`, and `NextRandomInteger(0, i + 1)` for
  Fisher–Yates — a drop-in for `RandBelow(i+1)` since **`NextRandomInteger` is
  half-open**, `max` exclusive despite the name.

  Reproducibility holds: `RandomEngine(int64_t seed)` seeds a vendored `pcg32`
  directly, which is pure specified integer arithmetic (no platform or
  standard-library divergence, unlike `std::uniform_int_distribution`), and
  `NextRandom()` is `ldexp(uint64, -64)`, IEEE754. Same seed → same rows on
  every platform. Caveat: `random_engine.hpp` is a DuckDB *internal* header, so
  a submodule bump could in principle remap seed → output. The README therefore
  promises reproducibility **within a build**, which is what a user needs from a
  query; if cross-version stability is ever required, vendor ~30 lines of pcg32
  into WS-G and own the mapping.
* **Classification sampling is exact today.** `output_mode := 'detail'` returns a
  `proba` MAP and `softmax_temperature` is already applied at decode
  (`tabfm_engine.cpp:832`), so upstream's `temp` maps 1:1 onto an existing
  option.
* **`generate` was already reserved.** `tabfm_model_spec.hpp:98` lists
  `"impute"` as a capability string and `docs/MULTI_MODEL_PLAN.md:36` defers
  `impute`/`anomaly`/`generate`/`embed` explicitly. This is that deferred slice.

## 3. The one real gap: no continuous predictive distribution

Our regression path returns a **point estimate only**. The exporter reduces the
5000-bucket bar distribution to its mean *inside the ONNX graph*:

```
resources/export_report_tabpfn_regression.json
  "n_out": 5000
  "output_signature": {"logits": "f32[1,T,1] RAW-space point estimate …
                                  (bar-distribution mean, de-standardized)"}
  "regression_contract": {"reduction": "softmax(bucket_logits) . bucket_means
                                        over criterion.borders (z-space) …"}
```

and `tabfm_engine.cpp:851` reads `out.logits[t * C]`. Same story for TabFM,
Mitra and TabICL. Orion-BiX has no regression head at all (`capabilities:
["classify"]`, `tabfm_registry.cpp:140`).

**Decision: quantile-bin → classify.** For a continuous column, bin it into
K ≤ 10 quantile bins over the real data, run the step as a **classification**
problem on bin labels, sample a bin from `proba` at temperature, then draw
uniformly inside the sampled bin's edges (outer bins clamped to the observed
min/max).

* Gives a genuine, multi-modal conditional density — which the point estimate
  cannot.
* Works on classify-only models (Orion-BiX), so *every* registry model can
  generate.
* Uniform-within-bin means synthetic values are novel, never verbatim copies of
  real rows (matters for the privacy story).
* Cost: quantized to ≤10 levels, no tail extrapolation past the observed range,
  and the model sees an unordered categorical so ordinal structure is lost.
* K is capped by the model's `size_regime.max_classes` (10 for every shipped
  model) and by the ≤10-label engine guard (`tabfm_engine.cpp:758`).

**Roadmap (not v1):** re-export the graphs with `bucket_logits` + the
`regression_borders` initializer as a second ONNX output and sample the true bar
distribution. That is per-model-family WS-A export work with its own parity
spike, and it leaves Orion-BiX unserved. Tracked separately.

**Imputation needs none of this.** Imputation wants the conditional *mode/mean*,
not a sample — so it uses the regression point estimate and the classification
argmax directly, at full precision, with no binning. `tabfm_impute` is therefore
both simpler and higher-fidelity than `tabfm_generate`, and this is the crisp
semantic split between the two functions:

| | sampling | continuous column |
|---|---|---|
| `tabfm_generate` | yes, at temperature | quantile-bin → classify |
| `tabfm_impute` | no (mode / mean) | regress, point estimate |

## 4. SQL surface

### `tabfm_generate`

```sql
tabfm_generate(
  data,                -- VARCHAR: relation name or a parenthesised subquery
  n := 100,            -- BIGINT: rows to synthesize
  features := NULL,    -- VARCHAR[]: restrict the columns modelled
  opts := MAP{},       -- MAP(VARCHAR,VARCHAR): seed, temperature, bins, …
  model := NULL)       -- VARCHAR: registry model id
```

Returns **the input columns plus `synthetic_id BIGINT`** (1..n, in generation
order — joinable, reproducible, and it keeps `SELECT * FROM tabfm_generate(...)`
honest about what these rows are).

```sql
SELECT * FROM tabfm_generate('customers', 100);

SELECT * FROM tabfm_generate(
  '(SELECT * FROM customers WHERE region = ''EU'')',
  n := 500,
  opts := MAP{'temperature':'1.2','seed':'7','bins':'10'},
  model := 'tabpfn-v2');
```

### `tabfm_impute`

```sql
tabfm_impute(
  data,
  columns := NULL,     -- VARCHAR[]: which columns to fill (default: all with NULLs)
  features := NULL,
  opts := MAP{},
  model := NULL)
```

Returns **exactly the input columns**, same names and types, NULLs in `columns`
replaced. Round-trips: `CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw')`.

```sql
SELECT * FROM tabfm_impute('customers');
SELECT * FROM tabfm_impute('customers', columns := ['income','plan']);
```

### Options (`opts`)

| key | default | meaning |
|---|---|---|
| `seed` | `42` | MT19937 seed for permutation + all draws |
| `temperature` | `1.0` | sampling temperature (generate only); passed through as `softmax_temperature` |
| `bins` | `10` | quantile bins per continuous column (generate only), 2..`max_classes` |
| `column_order` | `random` | `random` \| `natural` \| `missingness` — chain-rule permutation |
| `context_rows` | `0` | subsample the real context, as in predict |
| `rounds` | `1` | impute only: MICE-style refinement passes |

`task` is not accepted — it is decided per column, per step.

### Idiom compliance

* Built as **table macros** in the existing `PredictMacroDef` style
  (`tabfm_macros.cpp:63`): `data` is a string spliced after `FROM` inside
  `query(...)`, `features` filtered with the same `COLUMNS(lambda c: …)` trick,
  results flattened with `unnest(res, max_depth := 3)` — **never**
  `recursive := true` (S04).
* Full name `anofox_tabfm_generate` primary + `tabfm_generate` alias, via
  `RegisterMacroWithAlias`; likewise for impute.
* Whole-row struct aliased `anofox_tabfm_row`.
* `FunctionDescription` with parameter names, description and a runnable example
  so `duckdb_functions()` documents both (asserted in
  `test/sql/tabfm_function_docs.test`).

## 5. Implementation

### New files (no shared-file contention beyond the two scaffold seams)

| file | contents |
|---|---|
| `src/tabfm_generate.cpp` | the `__anofox_tabfm_generate_agg` / `__anofox_tabfm_impute_agg` aggregates + the chain-rule driver |
| `src/include/tabfm_generate.hpp` | `GenerateOptions`, `ColumnPlan`, `RunChainRule`, `RunImpute` declarations |
| `src/tabfm_generate_macros.cpp` | the two table macros (mirrors `tabfm_macros.cpp`; keeps WS-E's file untouched) |
| `test/sql/tabfm_generate.test` | end-to-end against the CI fixture model |
| `test/sql/tabfm_impute.test` | end-to-end |
| `test/sql/tabfm_generate_errors.test` | the SQL-API §5 error catalog |
| `test/cpp/test_tabfm_generate.cpp` | binning, marginal draw, permutation determinism, type coercion |

Scaffold edits (coordinate per CLAUDE.md rule 2): `CMakeLists.txt`
(`EXTENSION_SOURCES` + `TABFM_CPP_TEST_SOURCES`), `tabfm_registration.hpp`
(two `Register…` declarations), `anofox_tabfm_extension.cpp` (two calls).

### The generate aggregate

`__anofox_tabfm_generate_agg(row_struct, n, opts)`:

* **Bind** — reads the row STRUCT type, validates `opts`, snapshots the settings
  into a `PredictContext` (finalize runs off-thread and has no `ClientContext`),
  and computes `return_type = LIST(STRUCT(<input fields…>, synthetic_id BIGINT))`.
  Rejects a model without the required capability here, with the §5 remediation
  text.
* **Update** — accumulates rows exactly like the predict aggregate, enforcing
  `anofox_tabfm_max_rows` incrementally.
* **Finalize** — runs the chain rule:

```
plan   = ColumnPlan per column: KIND ∈ {categorical, continuous, temporal}
         + label set (categorical) or quantile edges (continuous)
π      = Fisher-Yates(0..H-1, RandomEngine(seed))  # or natural / missingness
synth  = n rows, all cells NULL

# step 1 — no features, empirical marginal
synth[:, π(1)] ← draw n values from the marginal of column π(1)

for i in 2..H:
    sub_type   = STRUCT(c_π(1..i-1)…, c_π(i))        # narrowed row schema
    rows       = real rows (target known)  ++  synth rows (target NULL)
    result     = GetPredictEngine().Predict({rows, sub_type, target_idx=i-1, …})
    synth[:, π(i)] ← sample from result.proba[row] at temperature
```

* Continuous columns are pre-binned into `bins` quantile labels before the call
  and the sampled bin is expanded back to a value; the mapping lives entirely in
  WS-G, so the engine only ever sees a normal classification problem.
* `H - 1` forward passes, each with `T = n_real + n` rows. **Sequential** — the
  chain rule cannot be parallelized across columns. Documented in the function
  description so the cost is not a surprise.

### The impute aggregate

`__anofox_tabfm_impute_agg(row_struct, columns, opts)`. Same skeleton, different
loop: order the target columns by **ascending null count**; for each, context =
rows where it is non-NULL, query = rows where it IS NULL, features = all other
columns (residual NULLs in the features are handled by the preprocessor's
existing mean/ordinal imputation, `tabfm_preprocess.cpp:326`). Take the argmax /
point estimate — no sampling. `opts['rounds'] > 1` repeats the sweep using the
previous round's fills, MICE-style.

### Type handling

* **Categorical** (VARCHAR / BOOL / ENUM / small-cardinality INTEGER) — used as
  the classification target directly; `label_decoder` returns the original typed
  `Value`, so output types match the input exactly.
* **Continuous** (all numeric types) — sampled as DOUBLE, then cast back to the
  column's type; integer columns round, and the observed min/max clamp keeps the
  result in range.
* **Temporal** (DATE / TIMESTAMP) — the preprocessor expands these into 5
  features (`golden_preprocess.json`), which is fine when they are *inputs*, but
  as a *target* they need an epoch round-trip. **v1 errors** with the §5-style
  message naming `features :=` as the way to exclude them; the epoch route is a
  follow-up.
* Columns with a single distinct value, or all-NULL columns, are copied /
  emitted constant without a model call.

### Guardrails and errors (SQL-API §5 style — every message names the fix)

| condition | message shape |
|---|---|
| model lacks `generate`-required capability | `tabfm_generate: model 'orion-bix' cannot regress …; it is still usable — continuous columns are binned. Use model := …` |
| fewer than 2 modellable columns | `… needs at least 2 columns to model a joint distribution; got 1` |
| `n_real + n > anofox_tabfm_max_rows` | `… Raise it with SET anofox_tabfm_max_rows = <n>; or lower n` |
| `bins` out of 2..max_classes | `… bins must be an integer between 2 and 10, got '…'` |
| temporal target column | `… cannot generate into TIMESTAMP column 'ts'; exclude it with features := [...]` |
| unknown opts key | reuses the predict wording, with the generate key list |

### Telemetry

`PostHogTelemetry::Instance().RecordFunctionCall("tabfm_generate")` /
`"tabfm_impute"` once per execution, in bind (per CLAUDE.md rule 3; note the
shipped API is `RecordFunctionCall`, cf. `tabfm_devices.cpp:510`, not the
`CaptureFunctionExecution` name in CLAUDE.md).

## 6. Milestones — red-green TDD, real SQL tests as DoD

Every milestone is **red first**: the named test file is written and committed
failing, then the code makes it pass. No milestone is done on a green C++ unit
test alone — the **definition of done for G2–G6 is a passing sqllogictest run
against the real engine and the real fixture model** (`make test_debug`, no
mocks, no stubs). Predictions from the fixture model are meaningless (random
weights), so tests assert the *contract* — shape, types, ranges, determinism —
exactly as `test/sql/tabfm_classify.test` already does.

| # | Milestone | Red-first test | Definition of done |
|---|---|---|---|
| G1 | Column planning + samplers (pure C++, no ORT) | `test/cpp/test_tabfm_generate.cpp` | `./build/debug/test/unittest "[tabfm][generate]"` green |
| G2 | `__anofox_tabfm_generate_agg` | `test/sql/tabfm_generate.test` | that file green via `./build/debug/test/unittest test/sql/tabfm_generate.test` |
| G3 | Macro surface + error catalog | `test/sql/tabfm_generate.test`, `test/sql/tabfm_generate_errors.test` | both green |
| G4 | `tabfm_impute` | `test/sql/tabfm_impute.test` | green |
| G5 | Multi-model coverage | `test/sql/tabfm_generate_multimodel.test` | green, offline |
| G6 | Docs + samples | `test/sql/tabfm_function_docs.test`, `test/sql/tabfm_examples.test` | green; every shipped sample runs |

**G1 — column planning + samplers.** `test_tabfm_generate.cpp` (added to
`TABFM_CPP_TEST_SOURCES`), tagged `[tabfm][generate]`: quantile-edge computation
(including ties and <K distinct values), empirical marginal draw, weighted
categorical draw at temperature, bin→value expansion, integer/bool/decimal
coercion, constant- and all-NULL-column shortcuts. Pure functions, no ORT.

RNG tests pin the contract rather than specific numbers: a fixed seed produces
an identical permutation and identical draw sequence across two `RandomEngine`
instances; different seeds diverge; `NextRandomInteger(0, n)` never returns `n`
(the half-open boundary, easy to get wrong); uniform-within-bin values always
land inside their bin edges; and a weighted draw over a degenerate `proba` (one
label at probability 1) always returns that label.

**G2 — the generate aggregate.** `test/sql/tabfm_generate.test`, driven by the
committed CI fixture model via `CALL tabfm_register_model(...)` exactly as
`tabfm_classify.test` does. Asserts:

* `count(*) = n` for several `n`, including `n = 1`;
* the output schema equals the input schema plus `synthetic_id BIGINT`, checked
  through `DESCRIBE` — column names *and* types;
* `synthetic_id` is exactly `1..n`, no gaps, no duplicates;
* every categorical value is one of the observed labels of that column;
* every numeric value lies within the observed `[min, max]` of that column;
* no output cell is NULL for a column that had no NULLs in the input;
* **determinism**: two calls with the same `seed` agree row-for-row (self-join,
  the `tabfm_classify.test` idiom); two different seeds do not;
* `INSERT INTO <source table> SELECT * EXCLUDE (synthetic_id) FROM
  tabfm_generate(...)` succeeds — the round-trip contract, asserted for real.

**G3 — the macro surface.** In `tabfm_generate.test`: named parameters in any
order, the parenthesised-subquery form of `data`, `features := [...]` narrowing
the modelled columns, `model := 'fixture'`, and a table name needing quoting.
`test/sql/tabfm_generate_errors.test` covers the §5 catalog from the table in §5
above — one `statement error` block per row, each asserting the remediation
text, plus the unknown-`opts`-key and out-of-range-`bins` messages.

**G4 — impute.** `test/sql/tabfm_impute.test`: NULLs filled; **non-NULL cells
identical to the input, asserted by an `EXCEPT` in both directions on the
non-NULL projection**; types and column order preserved; `columns := [...]`
leaves other columns' NULLs alone; deterministic across two calls;
`opts := MAP{'rounds':'2'}` runs and stays within observed ranges; a column with
no NULLs is a no-op; an all-NULL column raises the §5 error naming `columns :=`.

**G5 — multi-model.** `test/sql/tabfm_generate_multimodel.test`, following
`tabfm_multimodel_v2.test`: register two fixture models, assert `model :=`
selects between them, that a model advertising only `classify` still generates
continuous columns (the binning route), and that requesting a model without the
needed capability produces the §5 message. Offline, fixtures only.

**G6 — documentation and samples.** See §7 — this milestone ships user-facing
docs and runnable samples, and it is gated by tests too: the
`tabfm_function_docs.test` count moves from 9 to 11 and asserts that
`tabfm_generate`/`tabfm_impute` carry both a description and a runnable
`examples` entry, and a new `test/sql/tabfm_examples.test` executes the SQL
shipped in `examples/` against the fixture model so the samples cannot rot.

## 7. Documentation and samples (part of the work, not a follow-up)

Nothing merges without these. They are written in G6 but drafted alongside G2–G5
so the API is documented as it is designed.

### README

A new top-level section **"Generate synthetic data and fill gaps"**, placed
after *"We support classification and regression"*, matching that section's
shape: what it does in two sentences, the two signatures, one worked
copy-pasteable example each, an **Output columns** subsection (mirroring
`README.md:138`), and the options table from §4. Cross-linked from *Quickstart*
and from *Status & scope*.

### `docs/`

* `docs/GENERATE.md` — the user guide: the chain-rule explanation in plain
  language, when synthetic data is and is not appropriate, the temperature knob,
  choosing `bins`, cost (`H-1` sequential passes at `T = n_real + n`), and a
  **fidelity section** stating plainly what quantile binning preserves
  (marginals, conditional structure, multi-modality) and what it does not (tail
  extrapolation beyond the observed range, resolution finer than `bins`,
  ordinal structure inside the model). Includes the explicit non-claim: this is
  **not** a differential-privacy mechanism.
* `docs/REAL_MODELS.md` — capability table gains `generate` / `impute` columns.
* `docs/GENERATE_PLAN.md` (this file) — marked implemented, kept as the design
  record like `MULTI_MODEL_PLAN.md`.
* `CHANGELOG.md` — a feature entry naming both functions.

### `examples/`

Following the existing `examples/*.sql` + `examples/README.md` convention:

* `examples/generate_synthetic.sql` — build a small table, generate 2× its rows,
  and show them side by side.
* `examples/impute_missing.sql` — a table with scattered NULLs, imputed, with a
  before/after diff query.
* `examples/generate_fidelity.sql` — the cookbook's validation step ported to
  pure SQL: per-column marginal comparison (quantiles for numerics,
  `count(*) GROUP BY` shares for categoricals) and a pairwise `corr()` matrix of
  real vs. synthetic, so a user can *check* fidelity rather than trust it.
* `examples/generate_conditional.sql` — the practical conditional-generation
  recipe available in v1: generate from a filtered subquery
  (`'(SELECT * FROM customers WHERE churned)'`), and use `tabfm_impute` to
  complete partially-specified rows.
* `examples/README.md` — index entries for all four.

Each sample is exercised by `test/sql/tabfm_examples.test` against the fixture
model, so a signature change breaks CI rather than the docs.

## 8. Explicitly out of scope for v1

* True bar-distribution sampling (needs the ONNX re-export; §3).
* Conditional generation from a seed relation (`tabfm_impute` covers the
  conditional case for *existing* rows; generating *new* rows conditioned on
  fixed cells is a natural v2 and reuses the same driver).
* Temporal target columns.
* Ensembling (`n_estimators > 1` still throws, `tabfm_engine.cpp:733`).
* Differential-privacy guarantees. Uniform-within-bin avoids verbatim copying,
  but this is **not** a DP mechanism and the docs must not imply it is.
