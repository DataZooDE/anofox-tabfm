# Generate and impute

## Generate synthetic data and fill gaps

The same in-context engine also works *backwards*: instead of predicting one
column, it can model the whole table as a joint distribution and sample from it.

```sql
-- 500 synthetic customers that look like the real ones
SELECT * FROM tabfm_generate('customers', 500);

-- ...and straight back into the table, since the shape matches
INSERT INTO customers SELECT * EXCLUDE (synthetic_id) FROM tabfm_generate('customers', 500);

-- fill every NULL cell, conditioned on the rest of each row
CREATE TABLE clean AS SELECT * FROM tabfm_impute('raw');
SELECT * FROM tabfm_impute('raw', columns := ['income', 'plan']);
```

Generation works column by column: each column is sampled conditioned on the
ones already generated (the chain rule), so the **relationships between columns
survive** — not just each column's marginal. It needs only classification
weights, and works with every model in the registry, including classify-only
ones.

Imputation is the deterministic sibling: it takes the conditional best estimate
rather than sampling, so continuous fills keep full precision. Non-NULL cells
are never modified.

**[`docs/GENERATE.md`](../GENERATE.md)** explains the method, what the binning
does and does not preserve, and why this is *not* a privacy mechanism.
Runnable samples: [`examples/generate_synthetic.sql`](../../examples/generate_synthetic.sql),
[`examples/impute_missing.sql`](../../examples/impute_missing.sql),
[`examples/generate_fidelity.sql`](../../examples/generate_fidelity.sql),
[`examples/generate_conditional.sql`](../../examples/generate_conditional.sql).
