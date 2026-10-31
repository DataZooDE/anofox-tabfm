# Choosing a model

`anofox_tabfm` is one entrypoint for many **tabular foundation models** — "TabFM"
is the *category*, not a single model. Every built-in model is usable by name with
no config and no manifest file:

```sql
SELECT * FROM tabfm_list_models();          -- the registry: every known model, downloaded or not
```

The [Models table](../../README.md#models) lists each one with its tasks, licence, whether commercial use is
allowed and whether the download is gated. Per-model detail (sizes, size limits, attribution texts):
[Models and weights](../api/02-models-and-weights.md).

## Which one to start with

| You want | Start with |
|---|---|
| commercial use, no licence gate | `mitra` (about 303 MB per task), `tabicl-v2`, `tabdpt`; `orion-bix` and `orion-msp` for classification only |
| commercial use, attribution is fine | `tabpfn-v2`, `limix-2m` (about 10 MB, one checkpoint for both tasks) |
| research or evaluation only | the non-commercial models: `tabpfn-v2-5`, `tabpfn-v2-5-real`, `tabpfn-v2-6`, `tabpfn-v3`, `causilo`, `tabfm-v1` (6.56 GB) |
| a GPU | check `SELECT * FROM tabfm_backends();`: not every model runs on every backend ([support matrix](../api/04-gpu.md#backend-support-matrix)) |

## Selecting a model

Pick a model per call (a first-class argument, promoted out of `opts`), or set a
session default; precedence is **per-call → `anofox_tabfm_default_model` → a
single SQL-registered model**:

```sql
SELECT * FROM tabfm_classify('customers', 'churned', model := 'mitra');
SET anofox_tabfm_default_model = 'mitra';            -- session-wide
```

Downloading a model does not make it the default. With several models registered (always the case: the
built-ins alone are several) and neither `model :=` nor the setting given, a call fails and names both
remedies. An unknown `model :=`, or a model that lacks the requested task, is a clean error naming the
alternatives. `tabfm_download` / `tabfm_load` / `tabfm_unload` / `tabfm_remove` /
`tabfm_gpu_precompile` each take the same `model :=` argument (or fall back to the
default) so the whole lifecycle is per-model.

## Your own model

Register your **own** model entirely in SQL — no JSON file — with
`CALL tabfm_register_model(id := 'my', classification_graph := '<path|url>', …)`
(and `tabfm_unregister_model('my')`). The only external thing is the weight-free
ONNX graph blob; every other field is a named argument
([the full list](../api/02-models-and-weights.md#registering-your-own-model)). A single SQL-registered model
becomes the implicit default, so bare calls resolve to it.

## How they compare

Swap `model :=` to compare models on the same split. On iris, `mitra` / `tabpfn-v2` / `tabicl-v2` all reach
**0.962** accuracy against `tabfm-v1`'s 0.943, at a fraction of the size and ~40–60× the speed. On the churn
dataset of [Evaluate and compare](03-evaluate-and-compare.md) `tabfm-v1` reaches **F1 0.667 / accuracy
0.827** zero-shot. The runnable scripts (including `compare_models.sql`, a real head-to-head) and full
numbers are in [`examples/`](../../examples/README.md). Compare on your own data before you choose: these
are small public datasets.
