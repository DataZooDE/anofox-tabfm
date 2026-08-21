-- Filling gaps, pure SQL: replace NULLs with a tabular foundation model's
-- conditional best estimate.
--
-- tabfm_impute is the deterministic sibling of tabfm_generate. It does NOT
-- sample: each missing cell gets the conditional mode (categorical columns) or
-- the conditional mean (continuous columns), so continuous fills keep full
-- precision rather than being binned. Non-NULL cells are never touched.
--
-- Run:  duckdb :memory: < examples/impute_missing.sql
-- Needs: a model with BOTH classification and regression weights for mixed
--        tables, e.g. CALL tabfm_download('classification');
--                     CALL tabfm_download('regression');

LOAD anofox_tabfm;
-- With several models registered, one must be chosen explicitly
-- (generate_breast_cancer.sql does the same):
SET anofox_tabfm_default_model = 'mitra';

-- 1. A table where the columns genuinely inform each other, then punch holes
--    in it. `keep` remembers the truth so we can score the fills.
CREATE TABLE truth AS
SELECT
    range                                    AS id,
    (10 + 60 * random())::INT                AS tenure_months,
    round(20 + 4 * (10 + 60 * random()), 2)  AS monthly_spend,
    CASE WHEN random() < 0.5 THEN 'basic' ELSE 'premium' END AS plan
FROM range(150);

CREATE TABLE gappy AS
SELECT
    id,
    tenure_months,
    CASE WHEN id % 7 = 0 THEN NULL ELSE monthly_spend END AS monthly_spend,
    CASE WHEN id % 11 = 0 THEN NULL ELSE plan END         AS plan
FROM truth;

SELECT 'missing monthly_spend' AS what, count(*) FILTER (monthly_spend IS NULL) AS n FROM gappy
UNION ALL
SELECT 'missing plan', count(*) FILTER (plan IS NULL) FROM gappy;

-- 2. Fill every column that has NULLs. Same columns back, so it round-trips
--    into a repaired table.
CREATE TABLE repaired AS SELECT * FROM tabfm_impute('gappy');

SELECT 'still missing' AS what,
       count(*) FILTER (monthly_spend IS NULL OR plan IS NULL) AS n
FROM repaired;

-- 3. Known cells must be untouched. Both directions of EXCEPT over the rows
--    that had no gaps: this has to be zero.
SELECT count(*) AS changed_known_cells FROM (
    (SELECT id, monthly_spend, plan FROM repaired
     WHERE id NOT IN (SELECT id FROM gappy WHERE monthly_spend IS NULL OR plan IS NULL))
    EXCEPT
    (SELECT id, monthly_spend, plan FROM gappy WHERE monthly_spend IS NOT NULL AND plan IS NOT NULL)
);

-- 4. How good were the fills? Compare against the held-out truth.
SELECT
    'categorical accuracy' AS metric,
    round(avg(CASE WHEN r.plan = t.plan THEN 1.0 ELSE 0.0 END), 3) AS value
FROM repaired r JOIN truth t USING (id)
WHERE t.id % 11 = 0
UNION ALL
SELECT
    'continuous MAE',
    round(avg(abs(r.monthly_spend - t.monthly_spend)), 3)
FROM repaired r JOIN truth t USING (id)
WHERE t.id % 7 = 0;

-- 5. Fill only specific columns and leave the rest alone.
SELECT count(*) FILTER (plan IS NULL) AS plan_still_missing
FROM tabfm_impute('gappy', columns := ['monthly_spend']);

-- 6. `rounds` runs extra MICE-style sweeps, so columns filled later inform the
--    ones filled earlier. Worth it when several columns are sparse at once.
SELECT count(*) FILTER (monthly_spend IS NULL OR plan IS NULL) AS still_missing
FROM tabfm_impute('gappy', opts := MAP{'rounds': '3'});
