# anofox-tabfm documentation

## Guides

| Guide | Description |
|---|---|
| [Getting Started](guides/01-getting-started.md) | Install, download a model, a first prediction on iris |
| [Choosing a model](guides/02-choosing-a-model.md) | The registry, selecting a model, your own model, how the models compare |
| [Evaluate and compare](guides/03-evaluate-and-compare.md) | A churn walk-through, cross-validation, plotting |
| [GPU acceleration](guides/04-gpu-acceleration.md) | `tabfm_accelerate()`, checking the device, the ROCm first-run compile |
| [Generate and impute](guides/05-generate-and-impute.md) | Synthetic rows and filling NULLs |

## API reference

| Topic | Description |
|---|---|
| [Functions](api/01-functions.md) | `tabfm_classify`, `tabfm_regress`, missing targets, segments, output columns |
| [Models and weights](api/02-models-and-weights.md) | The model table, download / load / unload / remove, the cache, secrets |
| [Generation](api/03-generation.md) | `tabfm_generate`, `tabfm_impute`, options, outputs |
| [GPU](api/04-gpu.md) | Plugins, support matrix, precision, platform limits |
| [Settings](api/05-settings.md) | Every setting and the `opts` keys |
| [Troubleshooting](api/06-troubleshooting.md) | Errors and what to run |
| [Evaluation](api/07-evaluation.md) | Metrics, cross-validation, scoring rules |
| [Plotting](api/08-plotting.md) | The six `_by` macros for anofox-visualization |

## Building

[BUILDING.md](BUILDING.md): building and testing from source.
