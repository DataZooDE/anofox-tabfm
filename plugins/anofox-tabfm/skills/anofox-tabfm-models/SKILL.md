---
name: anofox-tabfm-models
description: >
  Choosing, downloading and managing models for the anofox_tabfm DuckDB extension:
  the 23 built-in tabular foundation models (Mitra, TabPFN 2 / 2.5 / 2.6 / 3 / 3.5,
  TabICL v2, TabDPT, LimiX-2M, Kumo-Tabular, EXAONE, Nori, Orion, Causilo, Google
  TabFM), their licences and commercial use, the licence gate, Hugging Face
  tokens, and registering your own ONNX model. Use when picking a model,
  running `tabfm_download` / `tabfm_load` / `tabfm_unload` / `tabfm_remove`,
  reading `tabfm_list_models()` or `tabfm_models()`, setting
  `anofox_tabfm_default_model` or `anofox_tabfm_accept_hf_license`, calling
  `tabfm_register_model`, tuning the context cache, or checking whether a model
  may be used commercially.
version: 2026.11.01
user-invocable: false
---

# Anofox TabFM — Models and Weights Cheat Sheet

**Extension:** anofox_tabfm v2026.11.01 | **DuckDB:** v1.5.6 / v2.0 | **Dual naming:** `tabfm_*` and `anofox_tabfm_*` (identical)

## Critical gotchas

1. **No weights ship with the extension.** Each model's weights are your own Hugging Face download, per task:
   `CALL tabfm_download('classification', model := '<id>')` and, separately, `'regression'`. `INSTALL httpfs;
   LOAD httpfs;` first. Weights are cached under `anofox_tabfm_cache_dir` (default `~/.cache/anofox-tabfm`).
2. **Downloading does not select a model.** Every call needs `model := '<id>'` or
   `SET anofox_tabfm_default_model = '<id>'`; `tabfm_cross_validate` takes **only** the default.
3. **Check `commercial` before building on a model.** Ten models are non-commercial. Licences are as
   published upstream; `tabfm_list_models()` has `license`, `commercial` and the `attribution` text to keep.
4. **Two separate gates.** A *gated* model needs `SET anofox_tabfm_accept_hf_license = true;` (your
   acknowledgement of its licence). Some Hugging Face repositories also need a token and an accepted licence
   on the model page: `CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_...', SCOPE 'https://huggingface.co');`.
   HTTP 401 usually means no token, 403 a token whose account has not accepted the licence.
5. **The per-model `max_rows` / `max_features` are not enforced**; the global settings
   `anofox_tabfm_max_rows` (10000) and `anofox_tabfm_max_features` (500) are. `max_classes` is enforced.
6. **Settings are not persisted.** Repeat `SET` statements in every new process.

## The registry

```sql
SELECT model, capabilities, license, commercial, max_rows, max_features, max_classes, downloaded
FROM tabfm_list_models()
ORDER BY model;
```

Columns: `model, family, capabilities, license, commercial, max_rows, max_features, max_classes, downloaded,
attribution`. `capabilities` is `'classify, regress'`, `'classify'` or `'regress'`.

## Model catalogue (23 built-in)

Sizes are per task (classification / regression) unless one file serves both. "rows" is the recommended
context size from `tabfm_list_models()`.

| model | by | tasks | licence | commercial | gated | weights | rows / features | notes |
|---|---|---|---|---|---|---|---|---|
| `mitra` | AWS AutoGluon | classify, regress | `apache-2.0` | yes | no | ~303 MB | 10k / 100 | the safe first choice |
| `mitra-v2` | AWS AutoGluon | classify, regress | `apache-2.0` | yes | no | ~303 / ~307 MB | 32k / 256 | larger tables than `mitra` |
| `tabdpt` | Layer 6 AI | classify, regress | `apache-2.0` | yes | no | ~254 MB, both tasks | 100k / 128 | TabDPT v1.2, up to 16 classes |
| `tabdpt-v1-3` | Layer 6 AI | classify, regress | `apache-2.0` | yes | no | ~252 MB, both tasks | 100k / 128 | TabDPT v1.3, up to 16 classes |
| `tabicl-v2` | Inria (soda-inria) | classify, regress | `bsd-3-clause` | yes | no | ~110 / ~114 MB | 100k / 512 | context cache on the CPU: fastest for repeated scoring |
| `orion-bix` | Lexsi Labs | classify | `mit` | yes | no | ~315 MB | 100k / 512 | classify only |
| `orion-msp` | Lexsi Labs | classify | `mit` | yes | no | ~336 MB | 100k / 512 | classify only |
| `nori-30m` | Synthefy | regress | `apache-2.0` | yes | no | ~117 MB | 5k / 100 | regress only; `tabfm_generate` refuses it |
| `kumo-tabular-small` | NVIDIA | classify, regress | `openmdw-1.1` | yes | no | ~110 / ~114 MB | 5k / 500 | keep the OpenMDW agreement and NVIDIA's notice when distributing |
| `kumo-tabular-medium` | NVIDIA | classify, regress | `openmdw-1.1` | yes | no | ~246 / ~250 MB | 5k / 500 | as small |
| `kumo-tabular-large` | NVIDIA | classify, regress | `openmdw-1.1` | yes | no | ~855 / ~863 MB | 5k / 500 | as small |
| `tabpfn-v2` | Prior Labs | classify, regress | `apache-2.0` | yes | yes | ~29 / ~44 MB | 10k / 500 | Prior Labs License v1.1 = Apache-2.0 + attribution |
| `limix-2m` | StableAI | classify, regress | `limix-2m-license-v1.0` | yes | yes | ~10 MB, both tasks | 5k / 100 | display "Built with StableAI LimiX"; smallest model |
| `tabpfn-v2-5` | Prior Labs | classify, regress | `tabpfn-2.5-license-v1.1` | **no** | yes | ~43 / ~41 MB | 10k / 500 | |
| `tabpfn-v2-5-real` | Prior Labs | classify, regress | `tabpfn-2.5-license-v1.1` | **no** | yes | ~43 / ~41 MB | 10k / 500 | RealTabPFN 2.5 |
| `tabpfn-v2-6` | Prior Labs | classify, regress | `tabpfn-2.6-license-v1.0` | **no** | yes | ~43 / ~52 MB | 50k / 2000 | widest tables |
| `tabpfn-v3` | Prior Labs | classify, regress | `tabpfn-3-license-v1.0` | **no** | yes | ~213 / ~233 MB | 10k / 500 | |
| `tabpfn-v3-5` | Prior Labs | classify, regress | `tabpfn-3-5-license-v1.0` | **no** | yes | ~876 MB, both tasks | 10k / 500 | no hosted service |
| `tabpfn-v3-5-fast` | Prior Labs | classify, regress | `tabpfn-3-5-license-v1.0` | **no** | yes | ~334 MB, both tasks | 10k / 500 | 8 of 3.5's 24 layers |
| `causilo` | Nums AI | classify, regress | `causilo-license-v1.0` | **no** | yes | ~144 / ~148 MB | 10k / 500 | hosted, API or SaaS use needs a separate licence |
| `exaone-tabular` | LG AI Research | classify, regress | `exaone-nc-1.2` | **no** | yes | ~83 / ~85 MB | 3k / 100 | CPU and ROCm only |
| `tabfm-v1` | Google | classify, regress | `tabfm-non-commercial-v1.0` | **no** | yes | 6.56 GB (classification) | — | the original; slow and large |
| `tabfm-v1-1` | Google | classify, regress | `tabfm-non-commercial-v1.0` | **no** | yes | ~821 / ~825 MB | 100k / 500 | v1's licence, 4x fewer parameters |

## Which model

| Situation | Pick |
|---|---|
| commercial, no licence gate, general purpose | `mitra`; `mitra-v2` for larger tables; compare with `tabicl-v2`, `tabdpt-v1-3` |
| scoring repeatedly against the same training table | `tabicl-v2` (context cache) |
| more than 10 classes | `tabdpt` / `tabdpt-v1-3` (16 classes); otherwise decompose |
| very wide tables | `tabpfn-v2-6` (2,000 features recommended), `tabicl-v2` / Orion (512) |
| tiny download, commercial with attribution | `limix-2m` (about 10 MB) |
| classification only | also `orion-bix`, `orion-msp` |
| research and evaluation | the TabPFN 2.5 / 2.6 / 3 / 3.5 family, `causilo`, `exaone-tabular`, `tabfm-v1-1` |
| a GPU | `SELECT * FROM tabfm_backends();` (the `anofox-tabfm-gpu` skill) |

There is no universal winner: swap `model :=` and compare on your own data with the
`anofox-tabfm-evaluate` skill. The rows / features above are recommendations; to hold a call to one, set the
guardrail (`SET anofox_tabfm_max_rows = 5000;` for `limix-2m`).

## Weight lifecycle

```sql
-- needs httpfs: INSTALL httpfs; LOAD httpfs;
CALL tabfm_download(task [, revision], model := '<id>');   -- rows: file, url, bytes, status
CALL tabfm_load(task, model := '<id>');                    -- verify the weights are present
CALL tabfm_unload([task,] model := '<id>');                -- free a loaded session
CALL tabfm_unload();                                       -- free every loaded session
CALL tabfm_remove(task [, revision], model := '<id>');     -- delete from the cache
```

`task` is `'classification'` or `'regression'`. A gated model:

```sql
SET anofox_tabfm_accept_hf_license = true;   -- after reading the licence
-- CALL tabfm_download('classification', model := 'tabpfn-v2');
```

`tabfm_load` does not build the session; the first predict does. `tabfm_models()` shows what is cached and
loaded:

```sql
SELECT model, task, revision, bytes, loaded, device, context_cache, context_hits, context_misses
FROM tabfm_models();
```

Columns: `model, task, revision, path, bytes, loaded, license, device, context_cache, context_bytes,
context_entries, context_hits, context_misses, context_first_seen`. `device` is empty until a predict warms a
session, then `cpu`, `cuda:0`, `rocm:0` or `mlx:0`.

Air-gapped machines: pre-seed `anofox_tabfm_cache_dir` with a cache copied from a connected machine.

## Selecting a model

Precedence: `model :=` in the call → `anofox_tabfm_default_model` → a single SQL-registered model.

```sql
SET anofox_tabfm_default_model = 'mitra';
SELECT customer_id, yhat FROM tabfm_classify('history', 'churned', test := 'prospects', keep := ['customer_id']);
SELECT customer_id, yhat
FROM tabfm_classify('history', 'churned', test := 'prospects', keep := ['customer_id'], model := 'tabicl-v2');
```

## Registering your own model

The only file outside SQL is the weight-free ONNX graph; every field is a named argument. The registration
lives in memory until the process ends; an id equal to a built-in shadows it.

```sql
CALL tabfm_register_model(
    id := 'my', classification_graph := '/models/my/graph.onnx',
    classification_weights := '/models/my/model.safetensors',
    tensor_map := '/models/my/tensor_map.json', license := 'apache-2.0');
-- SELECT * FROM tabfm_classify('customers', 'churned', model := 'my');
-- CALL tabfm_unregister_model('my');
```

Other arguments: `display_name`, `family`, `regression_graph`, `regression_weights`,
`classification_weights_url` / `regression_weights_url` (download instead of a local file) with their
`_bytes`, `weights_repo`, `weights_revision`, per-task `classification_tensor_map` / `regression_tensor_map`,
`commercial`, `gate_setting`, `preprocessing_profile`, `max_rows` / `max_features` / `max_classes`,
`fitted_second_pass` (the tasks whose context rows' own outputs are not fitted values, e.g. `['regression']`), `base_dir`
(relative paths resolve against it), `contract_inputs` / `contract_outputs`, the GPU graph variants
(`classification_ext_graph`, `regression_ext_graph`, `classification_migraphx_graph`,
`regression_migraphx_graph`) and multi-file lists (`classification_files`, `classification_file_urls`,
`classification_file_bytes` and the `regression_` equivalents).

## Context cache (`tabicl-v2` on the CPU)

The encoded labelled context is cached and reused across calls, exactly: the key is the preprocessed support
rows and labels, in order, so a changed or reordered table is a miss, never a stale answer.

- On by default. A context is kept on its **second** sighting (`anofox_tabfm_context_cache_admit = 'second'`),
  so the third identical call is the first fast one; `'first'` keeps every context at once.
- Budget: `anofox_tabfm_context_cache_memory` (default `256MB`, per loaded model, instance-wide: `SET SESSION`
  is refused). `CALL tabfm_unload()` drops the contexts.
- Kill switch: `SET anofox_tabfm_context_cache = false;` (`SET GLOBAL` also stops prepared statements).
- GPU sessions and every other model run the combined graph, unaffected.

```sql
SET GLOBAL anofox_tabfm_context_cache_memory = '512MB';
SELECT model, context_bytes, context_entries, context_hits, context_misses, context_first_seen
FROM tabfm_models() WHERE context_cache;
```

## Related settings

| setting | default | meaning |
|---|---|---|
| `anofox_tabfm_default_model` | `''` | model for calls without `model :=` |
| `anofox_tabfm_accept_hf_license` | `false` | acknowledge gated models' licences |
| `anofox_tabfm_cache_dir` | `~/.cache/anofox-tabfm` | weight cache root |
| `anofox_tabfm_max_rows` | `10000` | enforced rows per call or group |
| `anofox_tabfm_max_features` | `500` | enforced features per call |
| `anofox_tabfm_max_sessions` | `4` | cached sessions across models, devices and precisions |
| `anofox_tabfm_max_memory` | `''` | refuse a call when resident memory is at or above this |
| `anofox_tabfm_threads` | cores / 2 | ONNX Runtime CPU threads |
| `anofox_tabfm_cpu_prepack` | `true` | faster CPU matmuls for about 16% more memory |
| `anofox_tabfm_trace_level` | `warn` | `error` / `warn` / `info` / `debug` / `trace` |

## Errors and fixes

| You see | Do this |
|---|---|
| `N models are registered (...) and none is selected` | `model := '<id>'` or `SET anofox_tabfm_default_model` |
| `unknown model '<id>'. Registered: ...` | pick an id from the message or `tabfm_list_models()` |
| `model '<id>' does not support task '<task>'` | check `capabilities` |
| `has no downloaded weights for task '<task>'` | `CALL tabfm_download('<task>', model := '<id>');` |
| `weights in '<repo>' are licensed '<licence>' ... Run: SET anofox_tabfm_accept_hf_license = true;` | read the licence, set it, download again |
| `access denied (HTTP 401 Unauthorized)` | create the `hf` secret with a token |
| `access denied (HTTP 403 Forbidden)` | accept the licence on the Hugging Face model page |
| `size mismatch for ...` | a truncated download: run it again |
| `This checkpoint needs the one-time conversion step` (`tabpfn-v2`, `tabpfn-v2-5`, `tabpfn-v2-5-real`) | run `tools/export_tabpfn/convert_weights.py` from the repository once; it writes a `model.safetensors` the engine then prefers |
