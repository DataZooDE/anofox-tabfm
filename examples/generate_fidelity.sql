-- Checking synthetic data instead of trusting it, pure SQL.
--
-- The PriorLabs cookbook validates synthetic data three ways: marginal
-- distributions, correlation structure, and a PCA projection. The first two
-- port directly to SQL and are the ones that actually catch problems — this
-- script computes both, so you can decide whether a sample is good enough for
-- your use rather than taking the model's word for it.
--
-- Run:  duckdb :memory: < examples/generate_fidelity.sql
-- Needs: CALL tabfm_download('classification');

LOAD anofox_tabfm;
-- With several models registered, one must be chosen explicitly
-- (generate_breast_cancer.sql does the same):
SET anofox_tabfm_default_model = 'mitra';

-- A source table with deliberate structure: x and y strongly correlated, and
-- `grp` unevenly distributed. Both should survive into the synthetic sample.
CREATE TABLE real_data AS
SELECT
    round(x, 3)                                     AS x,
    round(2.0 * x + 1.0 + 0.4 * (random() - 0.5), 3) AS y,
    CASE WHEN random() < 0.7 THEN 'common' ELSE 'rare' END AS grp
FROM (SELECT random() * 10 AS x FROM range(300));

CREATE TABLE fake_data AS
SELECT * EXCLUDE (synthetic_id)
FROM tabfm_generate('real_data', 300, opts := MAP{'seed': '42'});

-- 1. MARGINALS, numeric: compare the quantiles side by side. Quantile binning
--    caps resolution at `bins` levels and never extrapolates past the observed
--    min/max, so expect the extremes to match closely and the interior to be
--    slightly coarser than the real data.
SELECT
    'x' AS column,
    round(q_real[1], 2) AS real_p10, round(q_fake[1], 2) AS fake_p10,
    round(q_real[2], 2) AS real_p50, round(q_fake[2], 2) AS fake_p50,
    round(q_real[3], 2) AS real_p90, round(q_fake[3], 2) AS fake_p90
FROM (SELECT quantile_cont(x, [0.1, 0.5, 0.9]) AS q_real FROM real_data),
     (SELECT quantile_cont(x, [0.1, 0.5, 0.9]) AS q_fake FROM fake_data)
UNION ALL
SELECT
    'y',
    round(q_real[1], 2), round(q_fake[1], 2),
    round(q_real[2], 2), round(q_fake[2], 2),
    round(q_real[3], 2), round(q_fake[3], 2)
FROM (SELECT quantile_cont(y, [0.1, 0.5, 0.9]) AS q_real FROM real_data),
     (SELECT quantile_cont(y, [0.1, 0.5, 0.9]) AS q_fake FROM fake_data);

-- 2. MARGINALS, categorical: the class shares should track. A big gap here
--    means the chain-rule order put this column somewhere unhelpful.
SELECT
    coalesce(r.grp, f.grp)              AS grp,
    round(coalesce(r.share, 0), 3)      AS real_share,
    round(coalesce(f.share, 0), 3)      AS synthetic_share,
    round(abs(coalesce(r.share, 0) - coalesce(f.share, 0)), 3) AS abs_diff
FROM (SELECT grp, count(*) / (SELECT count(*) FROM real_data)::DOUBLE AS share
      FROM real_data GROUP BY grp) r
FULL OUTER JOIN
     (SELECT grp, count(*) / (SELECT count(*) FROM fake_data)::DOUBLE AS share
      FROM fake_data GROUP BY grp) f USING (grp)
ORDER BY grp;

-- 3. CORRELATION STRUCTURE — the test that separates real joint modelling from
--    independent per-column sampling. Independent draws would score ~0 here
--    while still passing every marginal check above.
SELECT
    'corr(x, y)' AS statistic,
    round((SELECT corr(x, y) FROM real_data), 3) AS real_value,
    round((SELECT corr(x, y) FROM fake_data), 3) AS synthetic_value;

-- 4. And the conditional structure: does `grp` still shift the mean of y?
SELECT
    'real' AS source, grp, round(avg(y), 2) AS mean_y, count(*) AS n
FROM real_data GROUP BY grp
UNION ALL
SELECT 'synthetic', grp, round(avg(y), 2), count(*)
FROM fake_data GROUP BY grp
ORDER BY grp, source;

-- 5. PRIVACY SANITY CHECK: a continuous draw is interpolated between two
--    observed values rather than copied, so exact row reuse should be
--    rare-to-absent. NOTE: a smoke test, NOT a differential-privacy guarantee.
SELECT count(*) AS synthetic_rows_identical_to_a_real_row
FROM (SELECT x, y, grp FROM fake_data INTERSECT SELECT x, y, grp FROM real_data);
