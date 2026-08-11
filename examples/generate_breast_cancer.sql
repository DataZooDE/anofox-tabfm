-- Reproduction of the Prior Labs synthetic-data cookbook, in pure SQL.
--   https://docs.priorlabs.ai/cookbook/generate_synthetic_data
--
-- Upstream: load_breast_cancer(), train_test_split(test_size=0.33,
-- random_state=42), TabPFNUnsupervisedModel over all 30 features, temp=1.0,
-- n_samples = X_train.shape[0] * 2, then three fidelity checks
-- (plot_distributions, plot_correlation_difference, plot_pca).
--
-- Here: the same dataset (UCI wdbc.data — what load_breast_cancer wraps), the
-- same split proportions, the same 30 features, the same 2x sample count, and
-- the same three checks expressed as NUMBERS instead of matplotlib figures,
-- because a terminal cannot eyeball a scatter plot. The PCA panel — a visual
-- "do the clouds overlap" test — is replaced by the quantitative version of the
-- same question: train a model on the synthetic data, score it on held-out REAL
-- data, and see whether the signal survived.
--
-- Run:  duckdb :memory: < examples/generate_breast_cancer.sql
-- Needs: any classify-capable model's weights. Generation never needs the
--        regression head, so classification alone is enough.
-- Takes: ~15 min on CPU — 29 sequential model calls over 1143 rows each.
--
-- The numbers quoted in examples/README.md were produced with `mitra`
-- (Apache-2.0, ~303 MB). Any registered model works; set it below.

LOAD anofox_tabfm;
LOAD httpfs;
SET anofox_tabfm_max_rows = 20000;
SET anofox_tabfm_threads = 8;
SET anofox_tabfm_default_model = 'mitra';

-- 1. The dataset. UCI's wdbc.data is exactly what sklearn's load_breast_cancer
--    packages: 569 samples, id + diagnosis (M/B) + 30 features, being the
--    mean / standard-error / worst of ten cell-nucleus measurements.
CREATE TABLE wdbc AS
SELECT
    column00 AS id,               column01 AS diagnosis,
    column02 AS radius_mean,      column03 AS texture_mean,
    column04 AS perimeter_mean,   column05 AS area_mean,
    column06 AS smoothness_mean,  column07 AS compactness_mean,
    column08 AS concavity_mean,   column09 AS concave_points_mean,
    column10 AS symmetry_mean,    column11 AS fractal_dimension_mean,
    column12 AS radius_se,        column13 AS texture_se,
    column14 AS perimeter_se,     column15 AS area_se,
    column16 AS smoothness_se,    column17 AS compactness_se,
    column18 AS concavity_se,     column19 AS concave_points_se,
    column20 AS symmetry_se,      column21 AS fractal_dimension_se,
    column22 AS radius_worst,     column23 AS texture_worst,
    column24 AS perimeter_worst,  column25 AS area_worst,
    column26 AS smoothness_worst, column27 AS compactness_worst,
    column28 AS concavity_worst,  column29 AS concave_points_worst,
    column30 AS symmetry_worst,   column31 AS fractal_dimension_worst
FROM read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
    header = false);

-- 2. A 67/33 split, deterministic by id hash. Not bit-identical to sklearn's
--    random_state=42 permutation — that would mean reimplementing NumPy's
--    Mersenne Twister — but the same proportions and reproducible here.
CREATE TABLE train AS
    SELECT * EXCLUDE (id) FROM wdbc WHERE hash(id) % 100 < 67;
CREATE TABLE test_real AS
    SELECT * EXCLUDE (id) FROM wdbc WHERE hash(id) % 100 >= 67;
CREATE TABLE test_features AS SELECT * EXCLUDE (diagnosis) FROM test_real;

SELECT 'split' AS step,
       (SELECT count(*) FROM train)     AS train_rows,
       (SELECT count(*) FROM test_real) AS test_rows;

-- 3. Generate 2x the training rows, as upstream does. Every one of the 30
--    columns is modelled: 29 sequential model calls, each conditioned on the
--    columns already generated.
CREATE TABLE synthetic AS
SELECT * EXCLUDE (synthetic_id)
FROM tabfm_generate('train', 762, opts := MAP{'seed': '42', 'temperature': '1.0'});

SELECT 'generated' AS step, count(*) AS rows FROM synthetic;

--------------------------------------------------------------------------------
-- CHECK 1 — MARGINALS  (upstream: table_evaluator.plot_distributions)
--------------------------------------------------------------------------------
-- Long form once, reused by checks 1 and 2.
CREATE TABLE real_long AS
SELECT * FROM (
    UNPIVOT (SELECT row_number() OVER () AS row_id, * EXCLUDE (diagnosis) FROM train)
    ON COLUMNS(* EXCLUDE (row_id)) INTO NAME feature VALUE value);
CREATE TABLE fake_long AS
SELECT * FROM (
    UNPIVOT (SELECT row_number() OVER () AS row_id, * EXCLUDE (diagnosis) FROM synthetic)
    ON COLUMNS(* EXCLUDE (row_id)) INTO NAME feature VALUE value);

-- Per-feature agreement, summarized over all 30 features. `nrmse_*` is the gap
-- between real and synthetic, scaled by the real feature's own spread, so the
-- 30 features (which range over wildly different units) are comparable.
SELECT 'marginals' AS check,
       round(avg(abs(r.mu - f.mu) / nullif(r.sd, 0)), 4)         AS mean_shift_in_sds,
       round(max(abs(r.mu - f.mu) / nullif(r.sd, 0)), 4)         AS worst_mean_shift,
       round(avg(abs(r.sd - f.sd) / nullif(r.sd, 0)), 4)         AS avg_sd_ratio_error,
       round(avg(abs(r.p50 - f.p50) / nullif(r.sd, 0)), 4)       AS avg_median_shift
FROM (SELECT feature, avg(value) mu, stddev_pop(value) sd, quantile_cont(value, 0.5) p50
      FROM real_long GROUP BY feature) r
JOIN (SELECT feature, avg(value) mu, stddev_pop(value) sd, quantile_cont(value, 0.5) p50
      FROM fake_long GROUP BY feature) f USING (feature);

-- The five worst-matching features, so a bad column cannot hide in an average.
SELECT 'worst marginals' AS check, feature,
       round(r.mu, 3) AS real_mean, round(f.mu, 3) AS synthetic_mean,
       round(abs(r.mu - f.mu) / nullif(r.sd, 0), 3) AS shift_in_sds
FROM (SELECT feature, avg(value) mu, stddev_pop(value) sd FROM real_long GROUP BY feature) r
JOIN (SELECT feature, avg(value) mu FROM fake_long GROUP BY feature) f USING (feature)
ORDER BY shift_in_sds DESC LIMIT 5;

--------------------------------------------------------------------------------
-- CHECK 2 — CORRELATION STRUCTURE  (upstream: plot_correlation_difference)
--------------------------------------------------------------------------------
-- The test that separates real joint modelling from independent per-column
-- sampling. Sampling each column on its own would pass CHECK 1 perfectly and
-- fail here: all 435 feature pairs would decorrelate toward zero.
CREATE TABLE corr_real AS
SELECT a.feature AS f1, b.feature AS f2, corr(a.value, b.value) AS r
FROM real_long a JOIN real_long b USING (row_id)
WHERE a.feature < b.feature GROUP BY 1, 2;

CREATE TABLE corr_fake AS
SELECT a.feature AS f1, b.feature AS f2, corr(a.value, b.value) AS r
FROM fake_long a JOIN fake_long b USING (row_id)
WHERE a.feature < b.feature GROUP BY 1, 2;

SELECT 'correlation' AS check,
       count(*)                                    AS feature_pairs,
       round(avg(abs(cr.r - cf.r)), 4)             AS mean_abs_corr_diff,
       round(max(abs(cr.r - cf.r)), 4)             AS max_abs_corr_diff,
       -- How much of the real correlation structure is retained overall.
       round(corr(cr.r, cf.r), 4)                  AS corr_of_corrs,
       -- A per-column-independent generator would score ~0 here.
       round(avg(abs(cf.r)), 4)                    AS synthetic_mean_abs_corr,
       round(avg(abs(cr.r)), 4)                    AS real_mean_abs_corr
FROM corr_real cr JOIN corr_fake cf USING (f1, f2);

-- The pairs the real data cares about most, and what survived.
SELECT 'strongest pairs' AS check, f1, f2,
       round(cr.r, 3) AS real_corr, round(cf.r, 3) AS synthetic_corr
FROM corr_real cr JOIN corr_fake cf USING (f1, f2)
ORDER BY abs(cr.r) DESC LIMIT 8;

--------------------------------------------------------------------------------
-- CHECK 3 — UTILITY: train on synthetic, test on real
--   (upstream: plot_pca — a visual "do the two clouds overlap" check; this is
--    the same question answered with a number instead of a scatter plot)
--------------------------------------------------------------------------------
-- Both runs score the SAME held-out real rows. The only thing that changes is
-- what the model gets as in-context examples: real rows, or synthetic ones.
CREATE TABLE pred_real AS
SELECT * FROM tabfm_classify('train', 'diagnosis', test := 'test_features');

CREATE TABLE pred_synth AS
SELECT * FROM tabfm_classify('synthetic', 'diagnosis', test := 'test_features');

SELECT 'utility (TSTR)' AS check, source,
       round(accuracy, 4) AS accuracy, n
FROM (
    SELECT 'context = real train' AS source,
           avg(CASE WHEN p.yhat = t.diagnosis THEN 1.0 ELSE 0.0 END) AS accuracy,
           count(*) AS n
    FROM pred_real p JOIN test_real t USING (radius_mean, texture_mean, perimeter_mean)
    UNION ALL
    SELECT 'context = synthetic',
           avg(CASE WHEN p.yhat = t.diagnosis THEN 1.0 ELSE 0.0 END),
           count(*)
    FROM pred_synth p JOIN test_real t USING (radius_mean, texture_mean, perimeter_mean)
) ORDER BY source;

--------------------------------------------------------------------------------
-- CHECK 4 — did it just memorize the training set?
--------------------------------------------------------------------------------
-- Uniform-within-bin expansion draws fresh values rather than copying, so exact
-- reuse should be absent. A sanity check, NOT a privacy guarantee.
SELECT 'memorization' AS check,
       (SELECT count(*) FROM (SELECT * FROM synthetic INTERSECT SELECT * FROM train))
           AS synthetic_rows_identical_to_a_real_row,
       (SELECT count(DISTINCT radius_mean) FROM synthetic) AS distinct_radius_values,
       (SELECT count(DISTINCT radius_mean) FROM train)     AS distinct_radius_in_train;

-- Class balance should track the source, since diagnosis is modelled like any
-- other column rather than being conditioned on.
SELECT 'class balance' AS check, diagnosis,
       round(count(*) / (SELECT count(*) FROM train)::DOUBLE, 3) AS real_share
FROM train GROUP BY diagnosis ORDER BY diagnosis;

SELECT 'class balance' AS check, diagnosis,
       round(count(*) / (SELECT count(*) FROM synthetic)::DOUBLE, 3) AS synthetic_share
FROM synthetic GROUP BY diagnosis ORDER BY diagnosis;
