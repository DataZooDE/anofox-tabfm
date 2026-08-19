-- A workflow shaped like something a user would actually run, rather than a
-- synthetic range() table: read a CSV with mixed types, missing values in both
-- features and label, categorical columns, then join predictions back to the
-- source and aggregate them.
-- prepend: SET anofox_tabfm_ep_path='<dir holding the backend plugin>';
LOAD anofox_tabfm;
SET anofox_tabfm_default_model='tabfm-v1';

-- 1. ingest: types inferred, NULLs from empty CSV fields
CREATE TABLE customers AS SELECT * FROM read_csv_auto('tools/gpu_test/scenarios/customers.csv');
.mode list
SELECT 'STEP1_ROWS=' || count(*) FROM customers;
SELECT 'STEP1_UNLABELLED=' || count(*) FROM customers WHERE churned IS NULL;
SELECT 'STEP1_NULL_FEATURES=' || count(*) FROM customers
 WHERE plan IS NULL OR monthly_spend IS NULL OR support_tickets IS NULL
    OR last_login_days IS NULL OR satisfaction IS NULL;
.mode duckbox

-- 2. predict churn for the rows that have no label, on cpu
SET anofox_tabfm_device='cpu';
CREATE TABLE pred_cpu AS
SELECT customer_id, yhat FROM tabfm_classify('customers', 'churned');
.mode list
SELECT 'STEP2_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded;
SELECT 'STEP2_PREDICTED=' || count(*) FROM pred_cpu WHERE yhat IS NOT NULL;
SELECT 'STEP2_LABELS=' || string_agg(DISTINCT yhat, ',') FROM pred_cpu;
.mode duckbox

-- 3. the same thing on the GPU
SET anofox_tabfm_device='rocm';
CREATE TABLE pred_gpu AS
SELECT customer_id, yhat FROM tabfm_classify('customers', 'churned');
.mode list
SELECT 'STEP3_SERVED_BY=' || coalesce(max(device),'NONE') FROM tabfm_models() WHERE loaded;
SELECT 'STEP3_DISAGREEMENTS=' || count(*)
  FROM pred_cpu c JOIN pred_gpu g USING (customer_id)
 WHERE c.yhat IS DISTINCT FROM g.yhat;
.mode duckbox

-- 4. what a user does next: join back and aggregate by a business dimension
CREATE TABLE scored AS
SELECT c.customer_id, c.region, c.plan, c.monthly_spend,
       c.churned AS known, p.yhat AS predicted,
       coalesce(c.churned, p.yhat) AS churn_final
FROM customers c JOIN pred_gpu p USING (customer_id);

SELECT region,
       count(*) AS customers,
       sum(CASE WHEN churn_final = 'yes' THEN 1 ELSE 0 END) AS churn_risk,
       round(avg(monthly_spend), 2) AS avg_spend
FROM scored GROUP BY region ORDER BY region;

.mode list
SELECT 'STEP4_JOINED=' || count(*) FROM scored;
SELECT 'STEP4_NO_NULL_FINAL=' || count(*) FROM scored WHERE churn_final IS NULL;
.mode duckbox
