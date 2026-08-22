-- Conditional generation, pure SQL: getting synthetic rows that satisfy a
-- constraint, with the two mechanisms v1 actually gives you.
--
--   (a) FILTER THE SOURCE. Pass a subquery as `data`; the model then learns the
--       conditional distribution directly, which is exactly p(rest | filter).
--   (b) COMPLETE PARTIAL ROWS. Write the cells you want fixed, leave the rest
--       NULL, and let tabfm_impute fill them in.
--
-- Run:  duckdb :memory: < examples/generate_conditional.sql
-- Needs: CALL tabfm_download('classification', model := 'mitra');
--        CALL tabfm_download('regression', model := 'mitra');   -- for (b) on numeric columns

LOAD anofox_tabfm;
-- With several models registered, one must be chosen explicitly
-- (generate_breast_cancer.sql does the same):
SET anofox_tabfm_default_model = 'mitra';

CREATE TABLE customers AS
SELECT
    (10 + 60 * random())::INT                AS tenure_months,
    round(20 + 4 * (10 + 60 * random()), 2)  AS monthly_spend,
    CASE WHEN random() < 0.5 THEN 'basic' ELSE 'premium' END AS plan,
    (random() < 0.3)                         AS churned
FROM range(250);

-- (a) More churned customers: filter the source, and every synthetic row
--     inherits the constraint. `churned` becomes a constant column, so it is
--     emitted directly without a model call — and the OTHER columns are drawn
--     from the churned-only distribution, which is the point.
CREATE TABLE synthetic_churned AS
SELECT * FROM tabfm_generate(
    '(SELECT * FROM customers WHERE churned)',
    200,
    opts := MAP{'seed': '42'});

SELECT 'all churned?' AS check,
       count(*) FILTER (NOT churned) = 0 AS ok,
       count(*) AS n
FROM synthetic_churned;

-- The conditional really differs from the marginal: compare the spend profile
-- of churned-only synthetic rows against unconditional ones.
SELECT 'conditional (churned only)' AS sample, round(avg(monthly_spend), 2) AS mean_spend
FROM synthetic_churned
UNION ALL
SELECT 'unconditional', round(avg(monthly_spend), 2)
FROM tabfm_generate('customers', 200, opts := MAP{'seed': '42'});

-- Several constraints at once — any SQL predicate works, because `data` is
-- just a relation.
SELECT count(*) AS n, round(min(tenure_months), 1) AS min_tenure
FROM tabfm_generate(
    '(SELECT * FROM customers WHERE plan = ''premium'' AND tenure_months > 40)',
    50);

-- (b) Complete partially-specified rows: pin the columns you care about, NULL
--     the ones you want the model to choose. This is imputation, so the fills
--     are the conditional BEST estimate rather than a sample — use it when you
--     want the most likely completion, not a diverse set.
CREATE TABLE skeleton AS SELECT * FROM (VALUES
    (48, NULL, 'premium', NULL),
    (12, NULL, 'basic',   NULL),
    (60, NULL, 'premium', NULL))
    v(tenure_months, monthly_spend, plan, churned);

-- Give the model the real rows as context, then read back only the skeleton.
SELECT tenure_months, plan, monthly_spend, churned
FROM tabfm_impute('(SELECT * FROM customers UNION ALL BY NAME SELECT * FROM skeleton)')
WHERE tenure_months IN (48, 12, 60)
ORDER BY tenure_months;

-- NOTE on what v1 does NOT do: generating brand-new rows conditioned on fixed
-- cells (sampled, not best-estimate) is not a single call yet. Use (a) when the
-- constraint can be expressed as a filter over real data — which is the common
-- case — and (b) when you have specific rows to complete.
