# Models and weights

## The built-in models

| `model` | family | license | `commercial` |
|---|---|---|---|
| `tabfm-v1` | Google TabFM | non-commercial (gated) | `false` |
| `mitra` | AWS AutoGluon | Apache-2.0 | `true` |
| `tabpfn-v2` | Prior Labs | Apache-2.0 + attribution (gated) | `true` |
| `tabicl-v2` | Inria | BSD-3-Clause | `true` |
| `orion-bix` | Lexsi Labs | MIT | `true` (classify only) |
| `tabdpt` | Layer 6 AI | Apache-2.0 | `true` |
| `orion-msp` | Lexsi Labs | MIT | `true` (classify only) |
| `tabpfn-v2-5` | Prior Labs | TabPFN-2.5 (non-commercial, gated) | `false` |
| `tabpfn-v2-5-real` | Prior Labs | TabPFN-2.5 (non-commercial, gated) | `false` |
| `tabpfn-v2-6` | Prior Labs | TabPFN-2.6 (non-commercial, gated) | `false` |
| `tabpfn-v3` | Prior Labs | TabPFN-3 (non-commercial, gated) | `false` |
| `causilo` | Nums AI | Causilo License v1.0 (non-commercial, gated; hosted/API/SaaS use needs a separate license) | `false` |
| `limix-2m` | StableAI | Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 + attribution: "Built with StableAI LimiX"; gated by the licence acknowledgement; verified on CPU, CUDA and MLX not run, refused on ROCm) | `true` |

The full SQL surface runs the **real TabFM v1 model** end to end (preprocess →
ONNX Runtime forward → decode); per-row outputs match the PyTorch reference to
~1e-5. The weight-free graphs are compiled into the extension, so after
`tabfm_download` the model works with no companion files. A **model registry**
(`model :=`, `tabfm_list_models()`, `tabfm_register_model`) makes a second model a
data change, not new C++ (FR-5.1).

## Managing weights and devices

```sql
SELECT * FROM tabfm_models();     -- what's cached / loaded
```
```
┌───────────┬────────────────┬──────────┬──────────────┬────────────┬─────────┬───────────────────────────┬─────────┐
│   model   │      task      │ revision │     path     │   bytes    │ loaded  │          license          │ device  │
├───────────┼────────────────┼──────────┼──────────────┼────────────┼─────────┼───────────────────────────┼─────────┤
│ tabfm-v1  │ classification │ main     │ …/model.saf… │ 6557888408 │ true    │ tabfm-non-commercial-v1.0 │ rocm:0  │
└───────────┴────────────────┴──────────┴──────────────┴────────────┴─────────┴───────────────────────────┴─────────┘
```

```sql
CALL tabfm_load('classification');    -- warm the model (else lazy on first predict)
CALL tabfm_unload();                  -- free the weights, memory returns to baseline
CALL tabfm_remove('regression');      -- delete from the cache
SELECT * FROM tabfm_devices();         -- discovered execution devices (CPU / CUDA / ROCm / CoreML)
```

For a gated or private HF repo, add a bearer token — no custom credential store:

```sql
CREATE SECRET hf (TYPE http, BEARER_TOKEN 'hf_…', SCOPE 'https://huggingface.co');
```
