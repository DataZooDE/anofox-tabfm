# Models and weights

- [The built-in models](#the-built-in-models)
- [How weights work](#how-weights-work)
- [Managing weights](#managing-weights)
- [Registering your own model](#registering-your-own-model)

## The built-in models

The registry builds every model below in; `SELECT * FROM tabfm_list_models();` returns the same facts
(`model, family, capabilities, license, commercial, max_rows, max_features, max_classes, downloaded`).
*Gated* means the download needs `SET anofox_tabfm_accept_hf_license = true` first
([licence acknowledgement](../guides/01-getting-started.md#licence-acknowledgement-and-gated-repositories)).
Sizes are the registry's per-task weight files.

| `model` | by | tasks | licence id | `commercial` | gated | weights | notes |
|---|---|---|---|---|---|---|---|
| `mitra` | AWS AutoGluon | classify, regress | `apache-2.0` | `true` | no | ~303 MB per task | |
| `tabdpt` | Layer 6 AI | classify, regress | `apache-2.0` | `true` | no | ~254 MB, one file for both tasks | |
| `tabicl-v2` | Inria (soda-inria) | classify, regress | `bsd-3-clause` | `true` | no | | |
| `orion-bix` | Lexsi Labs | classify | `mit` | `true` | no | ~315 MB | classify only |
| `orion-msp` | Lexsi Labs | classify | `mit` | `true` | no | ~336 MB | classify only |
| `tabpfn-v2` | Prior Labs | classify, regress | `apache-2.0` | `true` | yes | | Prior Labs License v1.1 = Apache-2.0 + attribution |
| `limix-2m` | StableAI | classify, regress | `limix-2m-license-v1.0` | `true` | yes | ~10 MB, one checkpoint for both tasks | Apache-2.0 + Section 10: "Built with StableAI LimiX"; at most 5,000 rows and 100 features |
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
| `CALL tabfm_download(task [, revision], model := '<id>')` | fetch the weights into the cache; one row per file (`file, url, bytes, status`) |
| `CALL tabfm_load(task, model := '<id>')` | check that the weights are downloaded and fail with the fix if not; the session itself is built on the first predict |
| `CALL tabfm_unload([task], model := '<id>')` | free a loaded session (all of them without a task); memory returns to baseline |
| `CALL tabfm_remove(task [, revision], model := '<id>')` | delete the weights from the cache |
| `SELECT * FROM tabfm_models()` | what is cached and loaded, and which device serves it |

```sql
CALL tabfm_download('classification', model := 'mitra');
CALL tabfm_load('classification', model := 'mitra');      -- verify the weights are present
CALL tabfm_unload();                                      -- free every loaded session
CALL tabfm_remove('regression', model := 'mitra');        -- delete from the cache
SELECT * FROM tabfm_models();
```
```
┌───────────┬────────────────┬──────────┬──────────────┬────────────┬─────────┬───────────────────────────┬─────────┐
│   model   │      task      │ revision │     path     │   bytes    │ loaded  │          license          │ device  │
├───────────┼────────────────┼──────────┼──────────────┼────────────┼─────────┼───────────────────────────┼─────────┤
│ tabfm-v1  │ classification │ main     │ …/model.saf… │ 6557888408 │ true    │ tabfm-non-commercial-v1.0 │ rocm:0  │
└───────────┴────────────────┴──────────┴──────────────┴────────────┴─────────┴───────────────────────────┴─────────┘
```

`device` is empty until a predict warms a session; it is how you check that a GPU is used
([GPU acceleration](../guides/04-gpu-acceleration.md#2-check-what-serves-each-model)). To warm a ROCm
session ahead of the first query, use `tabfm_gpu_precompile` ([GPU](04-gpu.md#rocm-compile-time)).

For a gated or private HF repo, add a bearer token — no custom credential store:

```sql
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_…', SCOPE 'https://huggingface.co');
```

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
