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
   (`bins`, at most 10): resolution is capped and extrapolation is impossible. A wide empty stretch
   (iris petal length 1.9-3.0) is always a bin edge and never filled in.
3. **Exclude high-cardinality and temporal columns** (IDs, free text, timestamps) with `features := [...]`:
   they cannot be generation targets.
4. **Cost scales with columns, not rows.** One model call per column, sequentially; `n` is cheap.
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
| `column_order` | `random` | both | `random`, `natural` or `missingness` |
| `rounds` | `1` | impute | MICE-style refinement sweeps, 1–16 |

## Output

- `tabfm_generate`: every column of `data` (or only `features`), plus `synthetic_id` (`1..n`, generation
  order). With `features` omitted, `SELECT * EXCLUDE (synthetic_id)` has the shape of `data`.
- `tabfm_impute`: exactly the columns of `data` (or only `features`), so it round-trips. Non-NULL cells are
  never changed; fills are the conditional best estimate (deterministic), at full precision for continuous
  columns. A `NaN` in a `FLOAT` / `DOUBLE` column being filled counts as missing; `±Infinity` is an error.

## Generate

```sql
SELECT * FROM tabfm_generate('customers', 100, model := 'mitra');

-- more diverse, reproducible
SELECT * FROM tabfm_generate('customers', 100,
                             opts := MAP{'seed': '7', 'temperature': '1.3'}, model := 'mitra');

-- only some columns (e.g. leave the id out)
SELECT * FROM tabfm_generate('customers', 50, features := ['age', 'plan', 'income', 'churned'],
                             model := 'mitra');
```

Append to the source table:

```sql
INSERT INTO customers
SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('customers', 20, model := 'mitra');

INSERT INTO customers (age, plan, income)
SELECT age, plan, income
FROM tabfm_generate('customers', 20, features := ['age', 'plan', 'income'], model := 'mitra');
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
SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('customers', 200, model := 'mitra');

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

Train on synthetic, test on real (TSTR) is the stronger check: score real held-out rows with the synthetic
table as context (`anofox-tabfm-predict` and `anofox-tabfm-evaluate`).

```sql
-- id is NULL throughout the synthetic context, so it carries no signal there, yet it comes back
-- with every scored real row for the join
SELECT tabfm_accuracy(c.churned, p.yhat) AS tstr_accuracy
FROM tabfm_classify('(SELECT * REPLACE (NULL::BIGINT AS id) FROM synth)', 'churned',
                    test := '(SELECT * EXCLUDE (churned) FROM customers)', model := 'mitra') p
JOIN customers c USING (id);
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
