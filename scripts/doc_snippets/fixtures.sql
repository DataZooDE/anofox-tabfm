-- Tables the documentation and the skills query, run before every checked ```sql block
-- (scripts/check_doc_snippets.py). Small, deterministic, and shaped like the docs say:
-- in a train / test pair both sides expose the same feature columns.

-- customers: fully labelled; single-relation predict, CV, generation, segments
CREATE TABLE customers AS
SELECT i AS id, 20 + (i * 7) % 50 AS age, ['basic', 'plus', 'premium'][1 + i % 3] AS plan,
       30000.0 + (i * 1373) % 70000 AS income, round(((i * 37) % 100) / 3.0, 1) AS usage_gb,
       CASE WHEN i % 2 = 0 THEN 'north' ELSE 'south' END AS region, 2023 + i % 3 AS signup_year,
       (i * 13) % 5 < 2 AS churned
FROM range(1, 41) t(i);

-- history (labelled context) and prospects (rows to score): the explicit train / test form
CREATE TABLE history AS
SELECT id AS customer_id, age, plan, usage_gb, signup_year, churned FROM customers WHERE id <= 30;
CREATE TABLE prospects AS
SELECT id AS customer_id, age, plan, usage_gb, signup_year FROM customers WHERE id > 30;

-- regression pair
CREATE TABLE sold_homes AS
SELECT 50.0 + i * 7 AS sqft, 1 + i % 4 AS bedrooms, 100000.0 + i * 5300 + (i % 4) * 9000 AS price
FROM range(1, 31) t(i);
CREATE TABLE listings AS
SELECT 55.0 + i * 9 AS sqft, 1 + i % 3 AS bedrooms FROM range(1, 9) t(i);

-- raw: NULL cells to impute
CREATE TABLE raw AS
SELECT age, CASE WHEN id % 7 = 0 THEN NULL ELSE plan END AS plan,
       CASE WHEN id % 5 = 0 THEN NULL ELSE income END AS income
FROM customers;

-- scored: classification predictions of two models, for the metrics and the _by macros
CREATE TABLE scored AS
SELECT m AS model, a AS actual, p AS predicted, p AS yhat,
       MAP {'yes': s, 'no': 1 - s} AS proba
FROM (
  SELECT m, CASE WHEN (i * 13) % 5 < 2 THEN 'yes' ELSE 'no' END AS a,
         CASE WHEN (i * (CASE m WHEN 'mitra' THEN 13 ELSE 11 END)) % 5 < 2 THEN 'yes' ELSE 'no' END AS p,
         CASE WHEN (i * (CASE m WHEN 'mitra' THEN 13 ELSE 11 END)) % 5 < 2
              THEN 0.55 + (i % 9) / 20.0 ELSE 0.45 - (i % 9) / 20.0 END AS s
  FROM range(1, 21) t(i), (VALUES ('mitra'), ('tabicl-v2')) v(m));

-- scored_regression: regression predictions, for the regression metrics
CREATE TABLE scored_regression AS
SELECT 'mitra' AS model, 10.0 + i AS actual, 10.0 + i + ((i % 5) - 2) * 0.3 AS predicted
FROM range(1, 21) t(i);

-- predictions: a predictive distribution per row, for the scoring rules
CREATE TABLE predictions AS
SELECT 0.5 + (i % 4) AS actual,
       {'logits': [0.1, 0.4, 1.2, 0.3]::DOUBLE[], 'borders': [0.0, 1.0, 2.0, 3.0, 4.0]::DOUBLE[]} AS dist
FROM range(1, 21) t(i);

-- iris: the getting-started table (the docs read it from Hugging Face)
CREATE TABLE iris AS
SELECT 4.5 + (i % 7) * 0.4 AS sepal_len, 2.5 + (i % 5) * 0.2 AS sepal_wid,
       1.0 + (i % 3) * 2.2 AS petal_len, 0.2 + (i % 3) * 0.9 AS petal_wid,
       CASE WHEN i % 5 = 0 THEN NULL
            ELSE ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'][1 + i % 3] END AS species
FROM range(1, 46) t(i);

-- churn, train, test, test_features, preds: the evaluate-and-compare walk-through
CREATE TABLE churn AS
SELECT 'C' || lpad(i::VARCHAR, 4, '0') AS customerID, 1 + (i * 7) % 70 AS tenure,
       ['Month-to-month', 'One year', 'Two year'][1 + i % 3] AS Contract,
       20.0 + (i * 3) % 90 AS MonthlyCharges, 100.0 + (i * 53) % 5000 AS TotalCharges,
       (i * 13) % 5 < 2 AS Churn, (i * 37) % 100 AS bucket
FROM range(1, 61) t(i);
CREATE TABLE train AS SELECT * EXCLUDE (bucket) FROM churn WHERE bucket < 70;
CREATE TABLE test AS SELECT * EXCLUDE (bucket) FROM churn WHERE bucket >= 70;
CREATE TABLE test_features AS SELECT * EXCLUDE (Churn) FROM test;
CREATE TABLE preds AS
SELECT customerID, NOT Churn = (tenure % 4 = 0) AS pred,
       MAP {'true': 0.3 + (tenure % 5) / 10.0, 'false': 0.7 - (tenure % 5) / 10.0} AS proba
FROM test;
