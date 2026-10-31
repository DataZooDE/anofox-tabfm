# Generation and imputation

```sql
tabfm_generate(data, n [, features] [, opts] [, model])           -- sample new rows
tabfm_impute  (data [, columns] [, features] [, opts] [, model])  -- fill NULLs
```

## Output columns

`tabfm_generate` returns every column of `data`, plus:

| column | meaning |
|---|---|
| `synthetic_id` | `1..n` in generation order — joinable and reproducible |

`tabfm_impute` returns **exactly** the columns of `data`, so it round-trips.

## Options

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
