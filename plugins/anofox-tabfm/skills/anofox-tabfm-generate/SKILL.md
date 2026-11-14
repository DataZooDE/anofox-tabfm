---
name: anofox-tabfm-generate
description: >
  Synthetic tabular data and missing-value imputation with the anofox_tabfm DuckDB
  extension: `tabfm_generate` samples new rows that keep the relationships between
  columns, `tabfm_impute` fills NULL cells conditioned on the rest of each row, both
  zero-shot from a tabular foundation model. Use when creating synthetic or
  augmented rows, upsampling a rare segment, filling NULLs in a table, choosing
  generation options (seed, temperature, bins, column_order, rounds), or checking
  the fidelity and limits of generated data.
version: 2026.11.01
user-invocable: false
---

# Anofox TabFM — Generate and Impute Cheat Sheet

**Extension:** anofox_tabfm v2026.11.01 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `tabfm_*` and `anofox_tabfm_*` (identical)

The in-context engine used backwards: instead of predicting one column it models the table as a joint
distribution, one column at a time (the chain rule), so relationships between columns survive, not only each
column's marginal.

## Critical gotchas

1. **Weights.** Generation needs only the model's **classification** weights, so classify-only models work.
   Imputing a **numeric** column also needs its **regression** weights. `nori-30m` is regress-only:
   `tabfm_generate` refuses it, `tabfm_impute` fills numeric columns only.
2. **Values never leave the observed range.** Continuous columns are sampled through quantile bins
   (`bins`, at most 10): resolution is capped and extrapolation is impossible.
3. **Leave keys out.** A numeric id is generated like any number: plausible values inside the observed
   range, so duplicates of real keys, and every other column is conditioned on it. Free-text ids,
   timestamps and categoricals with more levels than the model's class head are errors. Name the columns to
   model with `features := [...]` or pass a subquery without them, and mint fresh keys yourself.
4. **Cost: one model call per column, sequentially, each over real rows + `n`.** `real rows + n` must stay
   within `anofox_tabfm_max_rows` (default 10000): for more synthetic rows, sample the input or make several
   calls with different seeds. Impute costs `rounds` x (columns with NULLs) calls; `columns := [...]` limits it.
5. **Not a privacy mechanism.** Synthetic rows are conditioned on the real ones and can reproduce them.
6. **`opts` values are strings**; an unknown key or an out-of-range value is an error.
7. **Reproducible within a build**: the same `seed` gives the same rows.

## Signatures

```sql
tabfm_generate(data, n [, features] [, opts] [, model])           -- sample n new rows
tabfm_impute  (data [, columns] [, features] [, opts] [, model])  -- fill NULLs
```

| argument | meaning |
|---|---|
| `data` | table name or `'(SELECT ...)'`: the rows to model (generate) or to fill (impute) |
| `n` | a positive constant: how many rows to generate |
| `columns` | impute only: the columns to fill (default: every column with a NULL) |
| `features` | the columns used at all; the others are dropped from the output |
| `opts` | `MAP(VARCHAR, VARCHAR)`, below |
| `model` | model id; default `anofox_tabfm_default_model` |

## Options

| key | default | applies to | meaning |
|---|---|---|---|
| `seed` | `42` | both | RNG seed |
| `temperature` | `1.0` | generate | diversity: higher explores more, lower hugs the data |
| `bins` | `10` | generate | quantile buckets per continuous column, 2–10 (clamped to the model's class head) |
| `column_order` | `random` | generate | `random`, `natural` or `missingness`; impute ignores it and always fills the column with the fewest NULLs first |
| `rounds` | `1` | impute | MICE-style refinement sweeps, 1–16 |

## Output

- `tabfm_generate`: every column of `data` (or only `features`), plus `synthetic_id` (`1..n`, generation
  order). With `features` omitted, `SELECT * EXCLUDE (synthetic_id)` has the shape of `data`.
- `tabfm_impute`: exactly the columns of `data` (or only `features`), so it round-trips. Non-NULL cells are
  never changed; fills are the conditional best estimate (deterministic), at full precision for continuous
  columns. A `NaN` in a `FLOAT` / `DOUBLE` column being filled counts as missing; `±Infinity` is an error.

## Generate

```sql
SELECT * FROM tabfm_generate('(SELECT * EXCLUDE (id) FROM customers)', 100, model := 'mitra');

-- more diverse, reproducible
SELECT * FROM tabfm_generate('(SELECT * EXCLUDE (id) FROM customers)', 100,
                             opts := MAP{'seed': '7', 'temperature': '1.3'}, model := 'mitra');

-- only some columns
SELECT * FROM tabfm_generate('customers', 50, features := ['age', 'plan', 'income', 'churned'],
                             model := 'mitra');
```

Append to the source table, with fresh keys:

```sql
INSERT INTO customers
SELECT (SELECT max(id) FROM customers) + synthetic_id AS id, * EXCLUDE (synthetic_id)
FROM tabfm_generate('(SELECT * EXCLUDE (id) FROM customers)', 20, model := 'mitra');
```

**Conditional generation** is a filter on the input: generate from the segment you want more of.

```sql
SELECT * FROM tabfm_generate('(SELECT * EXCLUDE (id) FROM customers WHERE churned)', 50, model := 'mitra');
```

## Impute

```sql
-- every NULL cell
CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw', model := 'mitra');

-- only some columns, with refinement sweeps
SELECT * FROM tabfm_impute('raw', columns := ['income'], opts := MAP{'rounds': '3'}, model := 'mitra');
```

## Checking fidelity

Compare marginals and a relationship between the real and the synthetic table:

```sql
CREATE TABLE synth AS
SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('(SELECT * EXCLUDE (id) FROM customers)', 200,
                                                    model := 'mitra');

SELECT 'real' AS src, avg(income) AS mean_income, avg(churned::INT) AS churn_rate,
       corr(age, income) AS corr_age_income
FROM customers
UNION ALL
SELECT 'synthetic', avg(income), avg(churned::INT), corr(age, income) FROM synth;

-- category frequencies
SELECT plan, count(*) FILTER (WHERE src = 'real') AS real_n, count(*) FILTER (WHERE src = 'synth') AS synth_n
FROM (SELECT 'real' AS src, plan FROM customers UNION ALL SELECT 'synth', plan FROM synth)
GROUP BY plan ORDER BY plan;
```

Train on synthetic, test on real (TSTR) is the stronger check: generate from one part of the real data
and score a held-out part the generator never saw, with the synthetic table as context
(`anofox-tabfm-predict` and `anofox-tabfm-evaluate`). Compare with the same holdout scored from the real
training part (TRTR).

```sql
CREATE TABLE real_train AS SELECT * FROM customers WHERE hash(id) % 10 < 7;
CREATE TABLE real_hold  AS SELECT * FROM customers WHERE hash(id) % 10 >= 7;
CREATE TABLE synth_train AS
SELECT * EXCLUDE (synthetic_id)
FROM tabfm_generate('(SELECT * EXCLUDE (id) FROM real_train)', 100, model := 'mitra');

-- keep := ['id'] brings the holdout's key back for the join (the synthetic context has none);
-- the holdout's labels in `test` are ignored
SELECT 'tstr' AS setup, tabfm_accuracy(h.churned, p.yhat) AS accuracy
FROM tabfm_classify('synth_train', 'churned', test := 'real_hold', keep := ['id'], model := 'mitra') p
JOIN real_hold h USING (id)
UNION ALL
SELECT 'trtr', tabfm_accuracy(h.churned, p.yhat)
FROM tabfm_classify('(SELECT * EXCLUDE (id) FROM real_train)', 'churned',
                    test := 'real_hold', keep := ['id'], model := 'mitra') p
JOIN real_hold h USING (id);
```

## Canonical pipeline

```sql
-- 0. once (needs httpfs): CALL tabfm_download('classification', model := 'mitra');
--    and, to impute numeric columns: CALL tabfm_download('regression', model := 'mitra');
SET anofox_tabfm_default_model = 'mitra';

-- 1. fill the gaps first, then model the clean table
CREATE TABLE filled AS SELECT * FROM tabfm_impute('raw');

-- 2. sample, keeping the source's shape
CREATE TABLE augmented AS
SELECT * FROM filled
UNION ALL
SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('filled', 40, opts := MAP{'seed': '1'});

SELECT count(*) FROM augmented;
```

More on the method and what binning preserves: `docs/GENERATE.md` in the repository.
