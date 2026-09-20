-- Tier 5: 'auto' resolves PER MODEL, and only ever to a device that model can
-- actually be served on.
--
-- Run on a machine with a real GPU and its plugin installed:
--   SET anofox_tabfm_ep_path='<dir>';  -- prepended by the harness
--
-- What makes this tier 5 rather than another equivalence check: it asserts
-- that two models in ONE session land on DIFFERENT devices, each the best its
-- own architecture supports. Nine of eleven catalog models have no MIGraphX
-- graph (docs/ROCM_SINGLE_EVAL_POS.md), so on ROCm the correct behaviour is a
-- split, not a uniform answer. A run where everything says cpu is the old
-- behaviour; a run where everything says rocm:0 means the servability gate is
-- not being consulted. Both are failures, and only printing SERVED_BY per
-- model can tell them apart.
--
-- Never trust "cpu and gpu agree" (CLAUDE.md): read the device column.

SET anofox_tabfm_device = 'auto';

CREATE TABLE ctx AS
SELECT i AS f1, (i % 7)::DOUBLE AS f2, ((i * 13) % 5)::DOUBLE AS f3,
       CASE WHEN i < 80 THEN ('c' || (i % 3)) ELSE NULL END AS label
FROM range(100) s(i);

-- 1. A model the GPU CAN serve (train_size-scalar family). Under auto this
--    must leave the CPU.
SELECT count(*) AS mitra_rows FROM tabfm_classify('ctx', 'label', model := 'mitra');
SELECT 'MITRA_SERVED_BY=' || coalesce(max(device), 'NONE')
FROM tabfm_models() WHERE loaded AND model = 'mitra';

-- 2. A model the GPU CANNOT serve, in the SAME session. This must say cpu --
--    and must still return answers rather than erroring, because auto never
--    promised the GPU for it.
SELECT count(*) AS tabdpt_rows FROM tabfm_classify('ctx', 'label', model := 'tabdpt');
SELECT 'TABDPT_SERVED_BY=' || coalesce(max(device), 'NONE')
FROM tabfm_models() WHERE loaded AND model = 'tabdpt';

-- 3. The two must DIFFER. If this prints false on a GPU box, per-model auto is
--    not happening: either everything fell back to cpu, or the servability
--    gate was skipped and tabdpt was sent somewhere it cannot run.
SELECT 'AUTO_IS_PER_MODEL=' ||
       ((SELECT max(device) FROM tabfm_models() WHERE loaded AND model = 'mitra') <>
        (SELECT max(device) FROM tabfm_models() WHERE loaded AND model = 'tabdpt'))::VARCHAR;

-- 4. The machine-checkable invariant: no loaded model sits on a device its own
--    capability row calls unsupported.
SELECT 'AUTO_NEVER_UNSUPPORTED=' || (count(*) = 0)::VARCHAR
FROM tabfm_models() m
-- Join on TASK as well. Without it a loaded mitra/classification matched the
-- mitra/REGRESSION capability row too, whose weights were never downloaded --
-- so the invariant reported a violation that had not happened. Caught on CUDA,
-- where only the classification weights had been fetched; it passes locally
-- only because the fixture model has a single task.
JOIN tabfm_backends() b
  ON b.model = m.model AND b.task = m.task AND b.device = m.device
WHERE m.loaded AND b.supported IS NOT TRUE;

-- 5. And the reason is queryable rather than folklore.
SELECT 'TABDPT_REASON=' || coalesce(max(reason), 'NONE')
FROM tabfm_backends() WHERE model = 'tabdpt' AND backend = 'rocm' AND NOT supported;

-- 6. An explicit request still hard-errors instead of degrading. Uncomment on
--    a ROCm box: this must raise, naming the model and the analysis, NOT
--    return rows on the CPU.
-- SET anofox_tabfm_device = 'rocm';
-- SELECT count(*) FROM tabfm_classify('ctx', 'label', model := 'tabdpt');
