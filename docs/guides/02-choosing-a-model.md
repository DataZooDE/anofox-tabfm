# Choosing a model

## Multiple models (the registry)

`anofox_tabfm` is one entrypoint for many **tabular foundation models** — "TabFM"
is the *category*, not a single model. Twelve models are **built in** and usable by
name with no config, no manifest file:

```sql
SELECT * FROM tabfm_list_models();          -- the registry: every known model
```

Pick a model per call (a first-class argument, promoted out of `opts`), or set a
session default; precedence is **per-call → `anofox_tabfm_default_model` → a
single SQL-registered model**:

```sql
SELECT * FROM tabfm_classify('customers', 'churned', model := 'mitra');
SET anofox_tabfm_default_model = 'mitra';            -- session-wide
```

Register your **own** model entirely in SQL — no JSON file — with
`CALL tabfm_register_model(id := 'my', classification_graph := '<path|url>', …)`
(and `tabfm_unregister_model('my')`). The only external thing is the weight-free
ONNX graph blob; every other field is a named argument. An unknown `model :=`, or
a model that lacks the requested task, is a clean error naming the alternatives.
`tabfm_download` / `tabfm_load` / `tabfm_unload` / `tabfm_remove` /
`tabfm_gpu_precompile` each take the same `model :=` argument (or fall back to the
default) so the whole lifecycle is per-model.

## How they compare

On this dataset `tabfm-v1` reaches **F1 0.667 / accuracy 0.827** zero-shot. The
same recipe generalizes across models and tasks — swap `model :=` to compare: on
iris, `mitra` / `tabpfn-v2` / `tabicl-v2` all reach **0.962** vs `tabfm-v1`'s
0.943, at a fraction of the size and ~40–60× the speed. The runnable scripts
(including `compare_models.sql`, a real head-to-head) and full numbers are in
[`examples/`](../../examples/README.md).
