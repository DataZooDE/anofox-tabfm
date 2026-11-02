# Models and weights

- [The built-in models](#the-built-in-models)
- [How weights work](#how-weights-work)
- [Managing weights](#managing-weights)
- [Registering your own model](#registering-your-own-model)

## The built-in models

The canonical licence, commercial and gate table is the [Models table in the README](../../README.md#models);
this page repeats it with sizes and notes so it stands on its own. The registry builds every model below in; `SELECT * FROM tabfm_list_models();` returns the same facts
(`model, family, capabilities, license, commercial, max_rows, max_features, max_classes, downloaded,
attribution`; `attribution` is the notice the model's licence asks you to keep, and for Kumo-Tabular the
OpenMDW-1.1 agreement the binary carries).
*Gated* means the download needs `SET anofox_tabfm_accept_hf_license = true` first
([licence acknowledgement](../guides/01-getting-started.md#licence-acknowledgement-and-gated-repositories)).
Sizes are the registry's per-task weight files. The `max_rows` / `max_features` that `tabfm_list_models()`
reports are measured or recommended limits and are **not enforced**; only `max_classes` is. The enforced
row and feature caps are the global `anofox_tabfm_max_rows` / `anofox_tabfm_max_features` settings.

| `model` | by | tasks | licence id | `commercial` | gated | weights | notes |
|---|---|---|---|---|---|---|---|
| `mitra` | AWS AutoGluon | classify, regress | `apache-2.0` | `true` | no | ~303 MB per task | |
| `tabdpt` | Layer 6 AI | classify, regress | `apache-2.0` | `true` | no | ~254 MB, one file for both tasks | |
| `tabicl-v2` | Inria (soda-inria) | classify, regress | `bsd-3-clause` | `true` | no | | |
| `orion-bix` | Lexsi Labs | classify | `mit` | `true` | no | ~315 MB | classify only |
| `orion-msp` | Lexsi Labs | classify | `mit` | `true` | no | ~336 MB | classify only |
| `tabpfn-v2` | Prior Labs | classify, regress | `apache-2.0` | `true` | yes | | Prior Labs License v1.1 = Apache-2.0 + attribution |
| `limix-2m` | StableAI | classify, regress | `limix-2m-license-v1.0` | `true` | yes | ~10 MB, one checkpoint for both tasks | Apache-2.0 + Section 10: "Built with StableAI LimiX"; recommended at most 5,000 rows and 100 features (not enforced: `SET anofox_tabfm_max_rows = 5000`) |
| `tabpfn-v2-5` | Prior Labs | classify, regress | `tabpfn-2.5-license-v1.1` | `false` | yes | ~43 MB / ~41 MB | |
| `tabpfn-v2-5-real` | Prior Labs | classify, regress | `tabpfn-2.5-license-v1.1` | `false` | yes | ~43 MB / ~41 MB | |
| `tabpfn-v2-6` | Prior Labs | classify, regress | `tabpfn-2.6-license-v1.0` | `false` | yes | ~43 MB / ~52 MB | |
| `tabpfn-v3` | Prior Labs | classify, regress | `tabpfn-3-license-v1.0` | `false` | yes | ~213 MB (classification) | |
| `causilo` | Nums AI | classify, regress | `causilo-license-v1.0` | `false` | yes | ~144 MB / ~148 MB | hosted, API or SaaS use needs a separate licence |
| `tabfm-v1` | Google | classify, regress | `tabfm-non-commercial-v1.0` | `false` | yes | 6.56 GB (classification) | non-commercial, no redistribution |

Some repositories are also gated by Hugging Face itself (account approval plus a token); that is a
separate step, described in
[Getting Started](../guides/01-getting-started.md#licence-acknowledgement-and-gated-repositories). Per-model
background (how each was ported, what is verified, licences in full, LimiX's attribution duties):
[docs/REAL_MODELS.md](../REAL_MODELS.md). Which backend serves which model:
[GPU](04-gpu.md#backend-support-matrix).

## How weights work

The extension ships only **weight-free** ONNX graphs, compiled into the extension; the weights are your own
Hugging Face download, cached under `anofox_tabfm_cache_dir` (default `~/.cache/anofox-tabfm`). After
`tabfm_download` a model works with no companion files. The engine (preprocess → ONNX Runtime forward →
decode) is checked against each model's PyTorch reference ([docs/REAL_MODELS.md](../REAL_MODELS.md)); for TabFM v1 per-row outputs match to ~1e-5. A **model
registry** (`model :=`, `tabfm_list_models()`, `tabfm_register_model`) makes another model a data change,
not new C++. For an air-gapped machine, pre-seed the cache directory.

## Managing weights

Every lifecycle function takes the task first and `model :=` (or falls back to
`anofox_tabfm_default_model`):

| Call | What it does |
|---|---|
| `CALL tabfm_download(task [, revision], model := '<id>')` | fetch the weights into the cache; one row per file (`file, url, bytes, status`). For `nori-30m` and the Kumo models a second row (`url` NULL) is the `model.safetensors` it writes from the checkpoint for the GPU graphs: `converted`, or `cached` when already there |
| `CALL tabfm_load(task, model := '<id>')` | check that the weights are downloaded and fail with the fix if not; the session itself is built on the first predict |
| `CALL tabfm_unload([task], model := '<id>')` | free a loaded session (all of them without a task); memory returns to baseline |
| `CALL tabfm_remove(task [, revision], model := '<id>')` | delete the weights from the cache |
| `SELECT * FROM tabfm_models()` | what is cached and loaded, which device serves it, and its [context cache](#context-cache) |

```sql
CALL tabfm_download('classification', model := 'mitra');
CALL tabfm_load('classification', model := 'mitra');      -- verify the weights are present
CALL tabfm_unload();                                      -- free every loaded session
CALL tabfm_remove('regression', model := 'mitra');        -- delete from the cache
SELECT * FROM tabfm_models();
```
```
┌─────────┬────────────────┬──────────┬──────────────┬───────────┬─────────┬────────────┬─────────┬───────────────┬───────────────┬─────────────────┬──────────────┬────────────────┐
│  model  │      task      │ revision │     path     │   bytes   │ loaded  │  license   │ device  │ context_cache │ context_bytes │ context_entries │ context_hits │ context_misses │
├─────────┼────────────────┼──────────┼──────────────┼───────────┼─────────┼────────────┼─────────┼───────────────┼───────────────┼─────────────────┼──────────────┼────────────────┤
│ mitra   │ classification │ main     │ …/model.saf… │ 302717904 │ false   │ apache-2.0 │         │ false         │             0 │               0 │            0 │              0 │
└─────────┴────────────────┴──────────┴──────────────┴───────────┴─────────┴────────────┴─────────┴───────────────┴───────────────┴─────────────────┴──────────────┴────────────────┘
```

(Illustrative. After a predict the row shows `loaded = true` and the serving device, e.g. `cpu` or `rocm:0`.
The `context_*` columns are `false` / zero for a session that does not use the [context cache](#context-cache).)

`device` is empty until a predict warms a session; it is how you check that a GPU is used
([GPU acceleration](../guides/04-gpu-acceleration.md#2-check-what-serves-each-model)). To warm a ROCm
session ahead of the first query, use `tabfm_gpu_precompile` ([GPU](04-gpu.md#rocm-compile-time)).

For a gated or private HF repo, add a bearer token — no custom credential store:

```sql
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_…', SCOPE 'https://huggingface.co');
```

## Context cache

Most of a `tabicl-v2` call re-encodes the labelled context: 83-98% of its CPU time at a 375- or 1000-row
context ([CONTEXT_CACHE_PLAN.md](../CONTEXT_CACHE_PLAN.md)). The context cache encodes it once and reuses it,
transparently: there is no new function, and every existing query benefits, including repeated scoring
against a fixed training table, cross-validation folds and per-segment `UNION ALL` calls.

- **Which models.** Models that ship an *exact* prepare/query graph pair, on the CPU. Today that is
  `tabicl-v2`, both tasks. Every other model, and every GPU session, runs the combined graph exactly as before.
- **Exact.** A cached context is keyed by the preprocessed support rows and labels themselves: a 64-bit hash
  finds a candidate, and a full compare accepts it. A changed value, label, row or column is a different key,
  so it is a miss, never a stale answer. Every output row, the context rows' fitted values included, equals the
  combined graph's (on the real `tabicl-v2` weights: identical labels, |Δlogit| ≤ 1.9e-5).
- **Cost.** The first call against a new context costs about what it costs without the cache. Later calls
  run the query rows only.
- **Memory.** `anofox_tabfm_context_cache_memory` (default `256MB`) bounds the cached contexts per loaded model;
  the least recently used goes first. The budget is instance-wide (one cache per loaded model serves every
  connection) and read when a statement runs. `tabfm_unload()` drops the contexts with the session. With the
  cache on, a model also holds a second ORT session (prepare + query): about +300 MB for `tabicl-v2`.
- **Batches within one call.** `opts := MAP{'query_batch': 'N'}` scores the test rows N at a time against one
  encoded context, so it bounds the peak memory of a large call without re-encoding the context per batch.
  Each pass has a fixed cost that grows with the context (about 60 ms at 2000 support rows), so keep N in the
  hundreds: at S = Q = 2000, `N = 256` cost 0.9x the unbatched call, `N = 16` 1.3x and `N = 1` 8x.
- **Kill switch.** `SET anofox_tabfm_context_cache = false` for the connection (statements bound after it);
  `SET GLOBAL anofox_tabfm_context_cache = false` also stops statements already prepared.
- **Sessions.** The cached session sits beside the combined one, so generation, imputation and connections
  with the kill switch on use the combined graph without evicting the cache.

```sql
SELECT * FROM tabfm_classify('history', 'churned', test := 'prospects', model := 'tabicl-v2',
                             opts := MAP{'query_batch': '128'});
SELECT model, task, context_cache, context_bytes, context_entries, context_hits, context_misses
FROM tabfm_models() WHERE loaded;
```

`context_cache` marks the cached session's row. `context_hits` and `context_misses` count lookups, one per
call (or per group of an aggregate), however many `query_batch` passes the call makes. `context_bytes` is the
payload (state tensors, the context rows' outputs and the verbatim key), not allocator overhead. The four
counters are read without a common snapshot, so a row read while a call is running can be momentarily
inconsistent. The support rows have to arrive in the same order to hit; a different order is a miss (safe,
just slower).

## Registering your own model

Register a model entirely in SQL, then use it with `model := '<id>'`:

```sql
CALL tabfm_register_model(id := 'my', classification_graph := '/p/g.onnx',
                          classification_weights := '/p/w.safetensors', tensor_map := '/p/map.json',
                          license := 'apache-2.0');
SELECT * FROM tabfm_classify('customers', 'churned', model := 'my');
CALL tabfm_unregister_model('my');
```

The only external thing is the weight-free ONNX graph; every other field is a named argument:
`id`, `display_name`, `family`, `classification_graph` / `regression_graph` (path or url),
`classification_weights` / `regression_weights`, `classification_weights_url` / `regression_weights_url`,
`classification_weights_bytes` / `regression_weights_bytes`, `weights_repo`, `weights_revision`,
`tensor_map` (or `classification_tensor_map` / `regression_tensor_map`), `license`, `commercial`,
`gate_setting`, `preprocessing_profile`, `max_rows` / `max_features` / `max_classes`, `base_dir`,
`contract_inputs` / `contract_outputs`, the GPU graph variants (`classification_ext_graph`,
`regression_ext_graph`, `classification_migraphx_graph`, `regression_migraphx_graph`) and, for several files
per task, the parallel lists `classification_files` / `classification_file_urls` /
`classification_file_bytes` (and the `regression_` equivalents). A registration is kept in memory and is gone when the process ends. A
model registered with the id of a built-in shadows it; a single SQL-registered model becomes the implicit
default.
