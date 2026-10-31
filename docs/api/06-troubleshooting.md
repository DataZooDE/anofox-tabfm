# Troubleshooting

Every failure is a DuckDB exception that names the function and the fix, e.g.:

```
Invalid Input Error: tabfm: model 'mitra' has no downloaded weights for task 'regression'.
Run: CALL tabfm_download('regression', model := 'mitra');
```

If an error does not tell you what to do, please
[open an issue](https://github.com/DataZooDE/anofox-tabfm/issues) with your setup
(`SELECT * FROM tabfm_devices();`, `SELECT * FROM tabfm_models();`) and the failing call.

- [Installing](#installing)
- [Choosing a model](#choosing-a-model)
- [Downloading weights](#downloading-weights)
- [Predicting](#predicting)
- [GPU](#gpu)

## Installing

* Without `-unsigned`, installing from `get.anofox.com` fails with *"Attempting
  to install an extension file that doesn't have a valid signature"*. The
  community repository needs no flag.
* `custom_extension_repository` applies to **every** install, including ones
  DuckDB triggers for you. If it is set before `httpfs` is present, DuckDB
  autoloads `httpfs` from `get.anofox.com`, which does not serve it, and the
  failure names `httpfs` rather than the setting. Install `httpfs` first.

## Choosing a model

| You see | It means | Do this |
|---|---|---|
| `N models are registered (...) and none is selected` | the call names no model and no default is set; downloading a model does not select it | add `model := '<id>'`, or `SET anofox_tabfm_default_model = '<id>';` |
| `unknown model '<id>'. Registered: ...` | a typo, or a model that was never registered | the message lists the ids; `SELECT * FROM tabfm_list_models();` |
| `model '<id>' does not support task '<task>'` | e.g. `tabfm_regress` on `orion-bix`, which is classify-only | pick a model whose `capabilities` include the task |

## Downloading weights

| You see | It means | Do this |
|---|---|---|
| `has no downloaded weights for task '<task>'` (from a predict), or `tabfm_load: ... is not downloaded` | the weights for that model and task are not in the cache | `CALL tabfm_download('<task>', model := '<id>');` (regression needs its own download) |
| `weights in '<repo>' are licensed '<licence>' ... Run: SET anofox_tabfm_accept_hf_license = true;` | a gated model: its licence needs your acknowledgement | read the licence, then `SET anofox_tabfm_accept_hf_license = true;` and download again |
| `access denied (HTTP 401) for the gated repository ...` | the Hugging Face repository needs a token | `CREATE SECRET hf (TYPE http, BEARER_TOKEN '<hf_token>', SCOPE 'https://huggingface.co');` |
| `access denied (HTTP 403) for the gated repository ...` | the token is fine, but the licence was not accepted on the model page | accept it on the Hugging Face model page while signed in, then download again |
| `size mismatch for ...` | the download was truncated or the source changed | the download is aborted; run it again |

## Predicting

| You see | It means | Do this |
|---|---|---|
| ``the context relation and the `test` relation must expose the same feature columns, but only in ...`` | in the train / test form one relation has a column the other lacks, which would be filled with `NULL` and degrade the prediction silently | drop the column, add it to the other relation, or name the shared columns with `features := [...]` |
| `features := [...] names '<x>', which matched no column` | a misspelled feature name | check it against the relation's columns (matching is case-insensitive) |
| `input exceeds anofox_tabfm_max_rows` | more rows than the guardrail allows in one call or group | send fewer rows, or raise it: `SET anofox_tabfm_max_rows = <n>;` (mind the model's own `max_rows` in `tabfm_list_models()`) |
| `'<model>' supports at most N classes` | the target has more classes than the model's head | reduce the class count or decompose the problem; `SELECT model, max_classes FROM tabfm_list_models();` |
| `task 'regression' requires a numeric target` | `tabfm_regress` on a text or boolean column | cast the target, or use `tabfm_classify` |
| `n_estimators > 1 (ensemble) is not available yet` | only a single estimator is implemented | use `n_estimators = '1'` or leave it out |
| `unknown option '<key>'; valid options are ...` | a typo in `opts` | the message lists the keys ([Options](05-settings.md#options)) |
| an infinite target value is an error naming the column | `+Infinity`, `-Infinity` or a date infinity in the target | `CASE WHEN isfinite(y) THEN y END` turns them into `NULL` ([missing targets](01-functions.md#classification-and-regression)) |
| `... is refused rather than risking an OOM kill` | `anofox_tabfm_max_memory` is set and the process is at or above it | send fewer rows per call, raise the ceiling, or `CALL tabfm_unload();` |

## GPU

| You see | It means | Do this |
|---|---|---|
| `no usable '<device>' device was discovered` | `anofox_tabfm_device` names a device that is not there or not usable | `SELECT * FROM tabfm_devices();` for the driver state, or `SET anofox_tabfm_device = 'auto';` |
| a predict on a named device fails with the reason the model cannot be served there | e.g. `tabicl-v2` on ROCm | `SELECT * FROM tabfm_backends() WHERE NOT supported;`, use `auto` or `cpu` |
| `tabfm_accelerate()` reports `verify failed` | the plugin is missing, unreadable, or its runtime libraries are not installed (MIGraphX on ROCm) | install what the `detail` names, then run `CALL tabfm_accelerate();` again; predictions continue on the CPU |
| a GPU mode is rejected | e.g. `bf16` or `fp16` on CUDA, `tf32` on ROCm | use a mode the backend supports ([Precision](04-gpu.md#precision)) |
| a session on the GPU gives the same answers but is slow | it may be on the CPU | `SELECT model, device FROM tabfm_models() WHERE loaded;` |
