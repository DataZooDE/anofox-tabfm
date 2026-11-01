-- One model's classification on the CPU and on CUDA (see cuda_catalog.sql). Expects:
--   LOAD anofox_tabfm; SET VARIABLE model = '<id>';
-- and anofox_tabfm_cache_dir / anofox_tabfm_ep_path pointing at the weights and the CUDA plugin.
SET anofox_tabfm_accept_hf_license = true;
CREATE OR REPLACE TEMP TABLE c_cpu (row_id BIGINT, yhat VARCHAR, yhat_score DOUBLE);
CREATE OR REPLACE TEMP TABLE c_gpu (row_id BIGINT, yhat VARCHAR, yhat_score DOUBLE);
CREATE OR REPLACE TEMP TABLE c_gpu2 (row_id BIGINT, yhat VARCHAR, yhat_score DOUBLE);

SET anofox_tabfm_device = 'cpu';
SET VARIABLE t0 = epoch_ms(now());
INSERT INTO c_cpu SELECT row_id, yhat::VARCHAR, yhat_score
FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := getvariable('model'));
SET VARIABLE t1 = epoch_ms(now());

SET anofox_tabfm_device = 'cuda';
SET VARIABLE t2 = epoch_ms(now());
INSERT INTO c_gpu SELECT row_id, yhat::VARCHAR, yhat_score
FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := getvariable('model'));
SET VARIABLE t3 = epoch_ms(now());
INSERT INTO c_gpu2 SELECT row_id, yhat::VARCHAR, yhat_score
FROM tabfm_classify('ctx_c', 'y', test := 'qry', model := getvariable('model'));
SET VARIABLE t4 = epoch_ms(now());

INSERT INTO report
SELECT getvariable('model'), 'classification',
       (SELECT coalesce(string_agg(DISTINCT device, ','), 'NONE') FROM tabfm_models()
         WHERE loaded AND model = getvariable('model') AND task = 'classification' AND device <> 'cpu'),
       (SELECT count(*) FROM c_cpu), (SELECT count(*) FROM c_gpu),
       (SELECT CASE WHEN count(*) > 0 THEN count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) END
          FROM c_cpu a JOIN c_gpu b USING (row_id)),
       (SELECT max(abs(a.yhat_score - b.yhat_score)) FROM c_cpu a JOIN c_gpu b USING (row_id)),
       NULL,
       (SELECT CASE WHEN count(*) > 0 THEN count(*) FILTER (WHERE a.yhat IS DISTINCT FROM b.yhat) END
          FROM c_gpu a JOIN c_gpu2 b USING (row_id)),
       (SELECT avg((a.yhat = t.y)::INT) FROM c_cpu a JOIN truth t USING (row_id)),
       (SELECT avg((b.yhat = t.y)::INT) FROM c_gpu b JOIN truth t USING (row_id)),
       getvariable('t1') - getvariable('t0'),
       CASE WHEN (SELECT count(*) FROM c_gpu) > 0 THEN getvariable('t3') - getvariable('t2') END,
       CASE WHEN (SELECT count(*) FROM c_gpu2) > 0 THEN getvariable('t4') - getvariable('t3') END;

.mode line
SELECT * FROM report WHERE model = getvariable('model') AND task = 'classification';
-- the class spread on both sides: a degenerate one-class GPU answer cannot hide behind agreement
.mode list
SELECT 'CLASS_SPREAD cpu=' || (SELECT string_agg(yhat || ':' || n, ' ' ORDER BY yhat)
                               FROM (SELECT yhat, count(*) n FROM c_cpu GROUP BY yhat))
    || ' cuda=' || coalesce((SELECT string_agg(yhat || ':' || n, ' ' ORDER BY yhat)
                             FROM (SELECT yhat, count(*) n FROM c_gpu GROUP BY yhat)), 'NONE');
CALL tabfm_unload();
