-- One model's regression on the CPU and on CUDA (see cuda_catalog.sql). Expects:
--   LOAD anofox_tabfm; SET VARIABLE model = '<id>';
-- and anofox_tabfm_cache_dir / anofox_tabfm_ep_path pointing at the weights and the CUDA plugin.
SET anofox_tabfm_accept_hf_license = true;
CREATE OR REPLACE TEMP TABLE r_cpu (row_id BIGINT, yhat DOUBLE);
CREATE OR REPLACE TEMP TABLE r_gpu (row_id BIGINT, yhat DOUBLE);
CREATE OR REPLACE TEMP TABLE r_gpu2 (row_id BIGINT, yhat DOUBLE);

SET anofox_tabfm_device = 'cpu';
SET VARIABLE t0 = epoch_ms(now());
INSERT INTO r_cpu SELECT row_id, yhat
FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := getvariable('model'));
SET VARIABLE t1 = epoch_ms(now());

SET anofox_tabfm_device = 'cuda';
SET VARIABLE t2 = epoch_ms(now());
INSERT INTO r_gpu SELECT row_id, yhat
FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := getvariable('model'));
SET VARIABLE t3 = epoch_ms(now());
INSERT INTO r_gpu2 SELECT row_id, yhat
FROM tabfm_regress('ctx_r', 'tgt', test := 'qry', model := getvariable('model'));
SET VARIABLE t4 = epoch_ms(now());

INSERT INTO report
SELECT getvariable('model'), 'regression',
       (SELECT coalesce(string_agg(DISTINCT device, ','), 'NONE') FROM tabfm_models()
         WHERE loaded AND model = getvariable('model') AND task = 'regression' AND device <> 'cpu'),
       (SELECT count(*) FROM r_cpu), (SELECT count(*) FROM r_gpu),
       NULL,
       (SELECT max(abs(a.yhat - b.yhat)) FROM r_cpu a JOIN r_gpu b USING (row_id)),
       (SELECT max(abs(a.yhat - b.yhat)) / nullif((SELECT max(yhat) - min(yhat) FROM r_cpu), 0)
          FROM r_cpu a JOIN r_gpu b USING (row_id)),
       (SELECT CASE WHEN count(*) > 0 THEN count(*) FILTER (WHERE abs(a.yhat - b.yhat) > 1e-6) END
          FROM r_gpu a JOIN r_gpu2 b USING (row_id)),
       (SELECT 1 - sum(power(a.yhat - t.tgt, 2)) / sum(power(t.tgt - (SELECT avg(tgt) FROM truth), 2))
          FROM r_cpu a JOIN truth t USING (row_id)),
       (SELECT 1 - sum(power(b.yhat - t.tgt, 2)) / sum(power(t.tgt - (SELECT avg(tgt) FROM truth), 2))
          FROM r_gpu b JOIN truth t USING (row_id)),
       getvariable('t1') - getvariable('t0'),
       CASE WHEN (SELECT count(*) FROM r_gpu) > 0 THEN getvariable('t3') - getvariable('t2') END,
       CASE WHEN (SELECT count(*) FROM r_gpu2) > 0 THEN getvariable('t4') - getvariable('t3') END;

.mode line
SELECT * FROM report WHERE model = getvariable('model') AND task = 'regression';
CALL tabfm_unload();
