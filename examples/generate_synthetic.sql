-- Synthetic tabular data, pure SQL: sample new rows from a table's joint
-- distribution with a tabular foundation model.
--
-- The model is never "trained". tabfm_generate factorizes the table column by
-- column (the chain rule) and samples each column conditioned on the columns
-- already generated, using the real rows as in-context examples. That is what
-- keeps the correlations: generating each column independently would preserve
-- every marginal and destroy every relationship between them.
--
-- Run:  duckdb :memory: < examples/generate_synthetic.sql
-- Needs: any model with classification weights, e.g.
--        CALL tabfm_download('classification');

LOAD anofox_tabfm;
-- With several models registered, one must be chosen explicitly
-- (generate_breast_cancer.sql does the same):
SET anofox_tabfm_default_model = 'mitra';

-- 1. A small source table with real structure worth reproducing: tenure and
--    monthly_spend rise together, and `plan` tracks both.
CREATE TABLE customers AS
SELECT
    (10 + 60 * random())::INT                       AS tenure_months,
    round(20 + 4 * (10 + 60 * random()), 2)::DOUBLE AS monthly_spend,
    CASE WHEN random() < 0.55 THEN 'basic'
         WHEN random() < 0.80 THEN 'plus'
         ELSE 'premium' END                         AS plan,
    (random() < 0.25)                               AS churned
FROM range(200);

SELECT 'source rows' AS what, count(*) AS n FROM customers;

-- 2. Generate twice as many synthetic rows as we have real ones.
--    One model call per column, run sequentially — cost scales with the number
--    of COLUMNS, not with n.
CREATE TABLE synthetic AS
SELECT * FROM tabfm_generate('customers', 400, opts := MAP{'seed': '42'});

SELECT 'synthetic rows' AS what, count(*) AS n FROM synthetic;

-- 3. What came back: the same columns as the source, plus synthetic_id (1..n).
DESCRIBE SELECT * FROM synthetic;

SELECT * FROM synthetic ORDER BY synthetic_id LIMIT 10;

-- 4. Temperature controls diversity. Lower stays closer to the observed data,
--    higher explores more of the distribution's tails.
SELECT 'temperature 0.5' AS setting,
       round(avg(monthly_spend), 2) AS mean_spend,
       round(stddev_pop(monthly_spend), 2) AS sd_spend
FROM tabfm_generate('customers', 200, opts := MAP{'seed': '1', 'temperature': '0.5'})
UNION ALL
SELECT 'temperature 2.0',
       round(avg(monthly_spend), 2),
       round(stddev_pop(monthly_spend), 2)
FROM tabfm_generate('customers', 200, opts := MAP{'seed': '1', 'temperature': '2.0'});

-- 5. The round trip: synthetic rows are ordinary rows of the same shape, so
--    they insert straight back into the source table.
INSERT INTO customers
SELECT * EXCLUDE (synthetic_id) FROM synthetic LIMIT 50;

SELECT 'after append' AS what, count(*) AS n FROM customers;

-- 6. Reproducibility: the same seed gives byte-identical output within a build.
SELECT count(*) AS rows_that_differ FROM (
    SELECT * FROM tabfm_generate('customers', 20, opts := MAP{'seed': '7'})
    EXCEPT ALL
    SELECT * FROM tabfm_generate('customers', 20, opts := MAP{'seed': '7'})
);
