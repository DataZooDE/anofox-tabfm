# Generation and imputation

How to use them, and what to check: [Generate and impute](../guides/05-generate-and-impute.md). The method,
what binning preserves and why this is not a privacy mechanism: [docs/GENERATE.md](../GENERATE.md).

```sql
tabfm_generate(data, n [, features] [, opts] [, model])           -- sample new rows
tabfm_impute  (data [, columns] [, features] [, opts] [, model])  -- fill NULLs
```

```sql
SELECT * FROM tabfm_generate('customers', 500, opts := MAP{'seed': '7', 'temperature': '0.8'}, model := 'mitra');
SELECT * FROM tabfm_impute('raw', columns := ['income'], opts := MAP{'rounds': '3'}, model := 'mitra');
```

`n` is a positive constant. `columns` limits which columns are filled (default: every column with a NULL);
`features` limits which columns are used at all. Generation needs only classification weights, so it works
with classify-only models; imputing a numeric column also needs the model's regression weights. A `NaN` in
a `FLOAT`/`DOUBLE` column being filled counts as missing, like `NULL`; `+/-Infinity` there is an error.

## Output columns

With `features` omitted, `tabfm_generate` returns every column of `data`, plus the column below; with
`features`, only the selected columns plus it:

| column | meaning |
|---|---|
| `synthetic_id` | `1..n` in generation order — joinable and reproducible |

With `features` omitted, `tabfm_impute` returns **exactly** the columns of `data`, so it round-trips; with
`features`, only the selected columns.

## Options

`opts` is a constant `MAP` with `VARCHAR` values; an unknown key or a value out of range is an error.

| key | default | applies to | meaning |
|---|---|---|---|
| `seed` | `42` | both | RNG seed; same seed → same rows within a build |
| `temperature` | `1.0` | generate | diversity; higher explores more, lower hugs the data |
| `bins` | `10` | generate | quantile buckets per continuous column, 2–10 (clamped to the model's class-head width) |
| `column_order` | `random` | both | `random` \| `natural` \| `missingness` |
| `rounds` | `1` | impute | MICE-style refinement sweeps, 1–16 |

## Cost and limits

Cost is one model call per column, run sequentially — it scales with the number
of **columns**, not with `n`. Continuous columns are sampled through quantile
bins, so generated values never leave the observed range and resolution is
capped at `bins` levels; high-cardinality and temporal columns cannot be
generation targets and must be excluded with `features :=`.
