# Generate and impute

The same in-context engine also works *backwards*: instead of predicting one
column, it can model the whole table as a joint distribution and sample from it.
Signatures, options and output columns: [Generation](../api/03-generation.md).

## Before you start

Generation needs only the **classification** weights. Imputing a numeric column also needs the
regression weights:

```sql
INSTALL httpfs; LOAD httpfs;
CALL tabfm_download('classification', model := 'mitra');
CALL tabfm_download('regression', model := 'mitra');       -- for tabfm_impute on numeric columns
```

## Generate synthetic rows

```sql
-- 500 synthetic customers that look like the real ones
SELECT * FROM tabfm_generate('customers', 500, model := 'mitra');

-- ...and straight back into the table, since the shape matches
INSERT INTO customers SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('customers', 500, model := 'mitra');
```

Generation works column by column: each column is sampled conditioned on the
ones already generated (the chain rule), so the **relationships between columns
survive** — not just each column's marginal. It works with every model in the
registry, including classify-only ones.

## Fill NULL cells

```sql
-- fill every NULL cell, conditioned on the rest of each row
CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw', model := 'mitra');
SELECT * FROM tabfm_impute('raw', columns := ['income', 'plan'], model := 'mitra');
```

Imputation is the deterministic sibling: it takes the conditional best estimate
rather than sampling, so continuous fills keep full precision. Non-NULL cells
are never modified.

## Check the output

Continuous columns are sampled through quantile bins, so generated values never leave the observed range;
high-cardinality and temporal columns cannot be generation targets and must be excluded with
`features :=`. Synthetic rows are not a privacy mechanism.

**[`docs/GENERATE.md`](../GENERATE.md)** explains the method, what the binning
does and does not preserve, and why this is *not* a privacy mechanism.
Runnable samples: [`examples/generate_synthetic.sql`](../../examples/generate_synthetic.sql),
[`examples/impute_missing.sql`](../../examples/impute_missing.sql),
[`examples/generate_fidelity.sql`](../../examples/generate_fidelity.sql),
[`examples/generate_conditional.sql`](../../examples/generate_conditional.sql).
