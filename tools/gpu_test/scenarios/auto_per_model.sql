-- 'auto' resolves PER MODEL, and only ever to a device that model can actually be served on.
--
-- Run on a machine with a real GPU and its plugin installed:
--   SET anofox_tabfm_ep_path='<dir>';  -- prepended by the harness
--
-- Two models in ONE session must land on DIFFERENT devices. Every built-in model now has a GPU graph (the ggml
-- GPU plugin serves the whole catalog on CUDA, ROCm and MLX), so the negative example is a model REGISTERED
-- without an ext graph: the committed fixture with only its plain graph. A run where everything says cpu means
-- auto never left the CPU; a run where everything says a GPU means the servability gate was not consulted.
-- Both are failures, and only printing SERVED_BY per model can tell them apart.
--
-- Never trust "cpu and gpu agree" (CLAUDE.md): read the device column.

SET anofox_tabfm_device = 'auto';

CREATE TABLE ctx AS
SELECT i AS f1, (i % 7)::DOUBLE AS f2, ((i * 13) % 5)::DOUBLE AS f3,
       CASE WHEN i < 80 THEN ('c' || (i % 3)) ELSE NULL END AS label
FROM range(100) s(i);

-- 1. A built-in model the GPU serves: under auto this must leave the CPU. (Needs its weights.)
SELECT count(*) AS mitra_rows FROM tabfm_classify('ctx', 'label', model := 'mitra');
SELECT 'MITRA_SERVED_BY=' || coalesce(max(device), 'NONE')
FROM tabfm_models() WHERE loaded AND model = 'mitra';

-- 2. A model the GPU CANNOT serve, in the SAME session: registered without an ext graph. This must say cpu,
--    and must still return answers rather than erroring, because auto never promised the GPU for it.
CALL tabfm_register_model(id := 'cpu-only-fixture', base_dir := 'test/fixtures',
  classification_graph := 'graph_fixture.onnx',
  classification_weights := 'model.safetensors',
  classification_tensor_map := 'tensor_map_fixture.json',
  license := 'fixture-mit', preprocessing_profile := 'tabfm_v1_minimal');
SELECT count(*) AS fixture_rows FROM tabfm_classify('ctx', 'label', model := 'cpu-only-fixture');
SELECT 'FIXTURE_SERVED_BY=' || coalesce(max(device), 'NONE')
FROM tabfm_models() WHERE loaded AND model = 'cpu-only-fixture';

-- 3. The two must DIFFER.
SELECT 'AUTO_IS_PER_MODEL=' ||
       ((SELECT max(device) FROM tabfm_models() WHERE loaded AND model = 'mitra') <>
        (SELECT max(device) FROM tabfm_models() WHERE loaded AND model = 'cpu-only-fixture'))::VARCHAR;

-- 4. The machine-checkable invariant: no loaded model sits on a device its own capability row calls
--    unsupported. Joined on TASK as well: without it a loaded mitra/classification matched the
--    mitra/regression row too, whose weights were never downloaded.
SELECT 'AUTO_NEVER_UNSUPPORTED=' || (count(*) = 0)::VARCHAR
FROM tabfm_models() m
JOIN tabfm_backends() b
  ON b.model = m.model AND b.task = m.task AND b.device = m.device
WHERE m.loaded AND b.supported IS NOT TRUE;

-- 5. The reason is queryable rather than folklore (it names the missing ext graph).
SELECT 'FIXTURE_REASON=' || coalesce(max(reason), 'NONE')
FROM tabfm_backends()
WHERE model = 'cpu-only-fixture' AND task = 'classification' AND device <> 'cpu' AND NOT supported;

-- 6. An explicit request still hard-errors instead of degrading. Uncomment with the GPU's device name: this must
--    raise, naming the model and the missing graph, NOT return rows on the CPU.
-- SET anofox_tabfm_device = 'rocm';
-- SELECT count(*) FROM tabfm_classify('ctx', 'label', model := 'cpu-only-fixture');
