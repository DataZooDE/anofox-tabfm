# Generating synthetic data and filling gaps

Two functions, both built on the same idea and the same engine:

```sql
tabfm_generate(data, n [, features] [, opts] [, model])   -- sample new rows
tabfm_impute  (data [, columns] [, features] [, opts] [, model])  -- fill NULLs
```

Neither one trains anything. They call the same in-context prediction engine as
`tabfm_classify` / `tabfm_regress`, repeatedly, with a different column playing
the target each time.

## How it works

A table is a joint probability distribution over its columns. Any joint
distribution factorizes by the chain rule:

```
p(c₁, …, c_H) = p(c_π₁) · p(c_π₂ | c_π₁) · … · p(c_πH | c_π₁ … c_π(H-1))
```

So to sample a row, pick an order for the columns and fill them one at a time,
each conditioned on the ones already filled. `tabfm_generate` does exactly that:
at step *i* it fits on the real rows using columns `π₁…π(i-1)` as features and
`π(i)` as the target, predicts on the partially-filled synthetic rows, and
**samples** from the predictive distribution. The first column has nothing to
condition on, so it comes from the empirical marginal.

This is why the correlations survive. Sampling each column independently would
reproduce every marginal perfectly and destroy every relationship between
columns — `examples/generate_fidelity.sql` includes the correlation test that
catches exactly that failure.

**It works with every model in the registry.** The driver sits above the engine
boundary and never touches ONNX graphs, tensor contracts, or preprocessing, so
anything that can classify can generate.

## Continuous columns, and what binning costs you

Sampling needs a predictive *distribution*. Classification gives us one — the
`proba` map. Regression does not: the ONNX exporters reduce the model's bar
distribution to its mean inside the graph itself
(`resources/export_report_tabpfn_regression.json` describes the output as
`f32[1,T,1] … bar-distribution mean, de-standardized`), so all the engine can
return is a single number per row. You cannot sample from a point estimate.

So a continuous column is cut into `bins` quantile buckets, the step runs as a
**classification** problem over bucket labels, a bucket is sampled at
temperature, and a value is drawn from the bucket's **empirical distribution** —
a uniform position among the observations that fell in it, interpolated between
neighbouring order statistics.

That last part is not a detail. Drawing uniformly across the bucket's *span*
instead is the obvious implementation and it is wrong for skewed columns: the
outermost buckets are wide, but the real mass hugs one end, so a uniform fill
pushes values out into the sparse tail. On the breast-cancer benchmark
(`examples/generate_breast_cancer.sql`) that inflated `area_se`'s mean by 67%
and bent the top 15% of its CDF away from the real curve. Interpolating between
observed neighbours keeps the bucket's internal shape and still emits novel
values, since a draw lands strictly between two real ones.

What you get:

- a genuine, potentially **multi-modal** conditional density — something a point
  estimate cannot express at all;
- it works on **classify-only models** (Orion-BiX has no regression head), so
  every model in the registry can generate;
- values are **novel**, not copied: a draw is interpolated strictly between two
  observations, so a synthetic number is essentially never a verbatim real one
  (the exception is tied neighbours, where the interpolation has nowhere to go).

What it costs:

- resolution is capped at `bins` levels per column — and `bins` is itself capped
  by **how wide the model's class head is**, not just by the engine's limit of
  10. `tabfm_generate` reads the selected model's `size_regime.max_classes` and
  clamps `bins` to it, because every step is a classification problem: asking a
  4-class model for 10 bins would otherwise fail deep in the engine with a
  tensor-shape error that never mentions `bins`. The same cap applies to
  categorical columns — a column with more distinct values than the head is wide
  cannot be a generation target. All six shipped models declare 10; narrower
  models (and CI fixtures) declare less;
- **no tail extrapolation** — nothing is ever generated outside the observed
  `[min, max]` of a column;
- the model sees an unordered categorical, so the ordering between adjacent
  buckets is not information it can use.

Exact sampling from the bar distribution needs the graphs re-exported with the
bucket logits and borders as a second output. That is per-model-family export
work, tracked separately.

## Imputation is the deterministic sibling

`tabfm_impute` wants the *most likely* value, not a sample. So it skips all of
the above: categorical columns take the classification argmax, continuous
columns take the regression point estimate directly, at full precision. It is
both simpler and more accurate than generation — and it needs regression weights
for numeric columns, which generation never does.

|  | samples? | continuous columns | needs regression weights |
|---|---|---|---|
| `tabfm_generate` | yes, at `temperature` | quantile bins | no |
| `tabfm_impute` | no (mode / mean) | full-precision point estimate | yes, for numeric columns |

Non-NULL cells are never modified. `tabfm_impute` returns exactly the input
columns, so it round-trips:

```sql
CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw');
```

Columns are filled in ascending order of missingness, so the sparsest ones get
the most context. `rounds` re-runs the sweep MICE-style, letting columns filled
later inform ones filled earlier. Within a sweep, a cell being filled is always
presented to the model as NULL — a guessed value is never fed back as ground
truth.

## Options

| key | default | applies to | meaning |
|---|---|---|---|
| `seed` | `42` | both | RNG seed for the column order and every draw |
| `temperature` | `1.0` | generate | sampling diversity; higher explores more, lower hugs the observed data |
| `bins` | `10` | generate | quantile buckets per continuous column, 2–10, further clamped to the model's `max_classes` |
| `column_order` | `random` | both | `random` \| `natural` \| `missingness` — the chain-rule order |
| `rounds` | `1` | impute | MICE-style refinement sweeps, 1–16 |
| `model` | — | both | registry model id (or use the `model :=` argument) |

There is no `context_rows`: the predict surface accepts that key but the engine
never reads it, and shipping a knob that does nothing is worse than not having
one.

## Cost

One model call per column, **sequentially** — step *i* needs what step *i-1*
produced, so this cannot be parallelized across columns. Each call runs over
`(real rows + n)` rows for generation, or `(real rows)` for imputation. Cost
therefore scales with the number of **columns**, not with `n`: generating 10,000
rows from a 6-column table is 5 calls, the same as generating 10.

`anofox_tabfm_max_rows` bounds `real rows + n`. Constant columns are emitted
directly and cost nothing.

## Columns that cannot be generated

- **High-cardinality categoricals** — more distinct values than the model's
  class head is wide (at most 10, often less; see above).
- **Temporal columns** (`DATE`, `TIMESTAMP`, …) are fine as *features* — the
  preprocessor expands them into numeric parts — but generating *into* one needs
  an epoch round-trip that v1 does not do.

Both raise an error naming the column and the fix, which is to exclude it:

```sql
SELECT * FROM tabfm_generate('events', 100, features := ['amount', 'category']);
```

Constant and all-NULL columns are emitted as-is with no model call.

## Reproducibility

Same seed, same rows — on every platform, because the RNG is DuckDB's `pcg32`
(pure specified integer arithmetic) rather than anything from `<random>`, whose
output is not portable across standard libraries. The guarantee is scoped to a
**single build**: `random_engine.hpp` is a DuckDB internal header, so a
submodule bump could remap seed → output.

## What this is not

Uniform-within-bin expansion means synthetic values are not verbatim copies of
real ones, and `examples/generate_fidelity.sql` checks for exact row reuse. That
is a useful sanity check and **not a privacy guarantee**. This is not a
differential-privacy mechanism, it carries no formal disclosure bound, and it
should not be described as anonymization. A model conditioned on real rows can
reproduce recognizable combinations of them, especially from small or highly
distinctive source tables.
