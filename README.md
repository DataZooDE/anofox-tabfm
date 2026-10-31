<h1 align="center">Anofox TabFM</h1>
<p align="center"><strong>Zero-shot machine learning for tabular data, inside DuckDB</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://duckdb.org"><img src="https://img.shields.io/badge/DuckDB-1.5.6%20%7C%202.0-green.svg" alt="DuckDB 1.5.6 | 2.0"></a>
  <a href="https://github.com/DataZooDE/anofox-tabfm/actions/workflows/MainDistributionPipeline.yml"><img src="https://github.com/DataZooDE/anofox-tabfm/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=main" alt="Build"></a>
</p>

This extension runs **tabular foundation models**, TabPFN-style in-context learners, so classification
and regression become a single SQL statement. No Python, no training loop, no MLOps: the model reads your
labelled rows as context and predicts the rest.

```sql
SELECT customer_id, yhat AS will_churn, yhat_score
FROM tabfm_classify('history', 'churned', test := 'prospects', model := 'mitra');
```

> **Built with StableAI LimiX.** The `limix-2m` model is StableAI's LimiX-2M, used under the Stable AI
> Technology Co., Ltd. License v1.0 ([licence](https://huggingface.co/stable-ai/LimiX-2M/blob/main/LICENSE.txt)).
> Several models are **non-commercial**: check the `commercial` column in [Models](#models) before you build
> on one.

## Key Features

### Predict
- **Classification and regression** in one call: `tabfm_classify`, `tabfm_regress`
- **Two forms**: an explicit train / test pair, or one table whose missing targets are the rows to score
- **Probabilities**, not just labels: `yhat_score` and a `proba` map per row

### 13 models, plus your own
- **Built in, selectable by name**: Mitra, TabDPT, TabPFN v2 / 2.5 / 2.5-real / 2.6 / 3, TabICL v2, Causilo,
  LimiX-2M, Orion-BiX, Orion-MSP and Google TabFM v1
- **Your own model**, registered entirely in SQL with `tabfm_register_model`
- **One SQL surface**: swap `model :=` to compare models on your own data

### Generate and impute
- **Synthetic rows** that keep the relationships between columns: `tabfm_generate`
- **Fill NULL cells**, conditioned on the rest of each row: `tabfm_impute`

### Evaluate and plot
- **Metrics** as SQL aggregates: accuracy, F1, ROC-AUC, log loss, ECE, RMSE, MAE, R², and more
- **Leakage-safe cross-validation**: `tabfm_cross_validate`
- **Plots** through [anofox-visualization](https://github.com/DataZooDE/anofox-visualization): model comparison,
  confusion matrix, ROC and calibration curves

### Run anywhere
- **CPU** on Linux, macOS (arm64) and Windows
- **GPU** with one call, `CALL tabfm_accelerate()`: NVIDIA CUDA, AMD ROCm, Apple MLX

### Safe by default
- **No weights shipped**: you download each model's weights from Hugging Face yourself
- **Licence gates**: a model whose licence needs an acknowledgement will not download without one
- **Errors that say what to run**

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Documentation](#documentation)
- [Status](#status)
- [Development](#development)
- [License](#license)
- [Telemetry](#telemetry)
- [Feedback](#feedback)

## Installation

```sql
INSTALL anofox_tabfm FROM community;
LOAD anofox_tabfm;
```

**Latest builds:** the community build can trail a release by days. The DataZoo repository has it the
moment it is out, but its builds are unsigned: start DuckDB with `duckdb -unsigned`, and install `httpfs`
**first** ([why](docs/guides/01-getting-started.md#1-install-and-load)):

```sql
INSTALL httpfs; LOAD httpfs;
SET custom_extension_repository = 'https://get.anofox.com';
INSTALL anofox_tabfm;
LOAD anofox_tabfm;
```

Check which build you got:

```sql
SELECT extension_version FROM duckdb_extensions() WHERE extension_name = 'anofox_tabfm';
```

## Quick Start

```sql
-- 1. Weights are fetched over HTTPS, once: mitra is Apache-2.0, ungated, about 303 MB
INSTALL httpfs; LOAD httpfs;
CALL tabfm_download('classification', model := 'mitra');

-- 2. iris, read from Hugging Face, with every fifth species hidden as NULL
CREATE TABLE iris AS
SELECT SepalLengthCm AS sepal_len, SepalWidthCm AS sepal_wid,
       PetalLengthCm AS petal_len, PetalWidthCm AS petal_wid,
       CASE WHEN row_number() OVER () % 5 = 0 THEN NULL ELSE Species END AS species
FROM 'hf://datasets/scikit-learn/iris/**/*.csv';

-- 3. Rows with a species are the context; predict the NULL ones, zero-shot
SELECT sepal_len, petal_len, yhat AS predicted_species, yhat_score
FROM tabfm_classify('iris', 'species', model := 'mitra')
WHERE species IS NULL;
```

Every function takes `model :=`. Downloading does not choose a model for you: with no `model :=` and no
`SET anofox_tabfm_default_model = 'mitra'`, a call fails and says how to pick one. Regression needs its own
weights: `CALL tabfm_download('regression', model := 'mitra');`. The full walk-through with output:
[Getting Started](docs/guides/01-getting-started.md); runnable files in [`examples/`](examples/README.md).

## Models

| Model | By | Tasks | Licence (`tabfm_list_models()`) | Commercial | Gated |
|---|---|---|---|---|---|
| `mitra` | AWS AutoGluon | classify, regress | `apache-2.0` | yes | no |
| `tabdpt` | Layer 6 AI | classify, regress | `apache-2.0` | yes | no |
| `tabicl-v2` | Inria (soda-inria) | classify, regress | `bsd-3-clause` | yes | no |
| `orion-bix` | Lexsi Labs | classify | `mit` | yes | no |
| `orion-msp` | Lexsi Labs | classify | `mit` | yes | no |
| `tabpfn-v2` | Prior Labs | classify, regress | `apache-2.0` (Prior Labs License v1.1: Apache-2.0 + attribution) | yes | yes |
| `limix-2m` | StableAI | classify, regress | `limix-2m-license-v1.0` (Apache-2.0 + attribution: "Built with StableAI LimiX") | yes | yes |
| `tabpfn-v2-5` | Prior Labs | classify, regress | `tabpfn-2.5-license-v1.1` | **no** | yes |
| `tabpfn-v2-5-real` | Prior Labs | classify, regress | `tabpfn-2.5-license-v1.1` | **no** | yes |
| `tabpfn-v2-6` | Prior Labs | classify, regress | `tabpfn-2.6-license-v1.0` | **no** | yes |
| `tabpfn-v3` | Prior Labs | classify, regress | `tabpfn-3-license-v1.0` | **no** | yes |
| `causilo` | Nums AI | classify, regress | `causilo-license-v1.0` (hosted, API or SaaS use needs a separate licence) | **no** | yes |
| `tabfm-v1` | Google | classify, regress | `tabfm-non-commercial-v1.0` | **no** | yes |

**Gated** means the download needs `SET anofox_tabfm_accept_hf_license = true;` first, your acknowledgement
of that licence. Some Hugging Face repositories also need account approval and a token, which is a separate
step: [Getting Started](docs/guides/01-getting-started.md#licence-acknowledgement-and-gated-repositories).
Which model to pick, and how they compared: [Choosing a model](docs/guides/02-choosing-a-model.md). Sizes,
backends and attribution texts: [Models and weights](docs/api/02-models-and-weights.md).

## Documentation

### Guides

| Guide | Description |
|---|---|
| [Getting Started](docs/guides/01-getting-started.md) | Install, download a model, licence gates, a first prediction on iris |
| [Choosing a model](docs/guides/02-choosing-a-model.md) | Selecting a model, the default, your own model, how the models compared |
| [Evaluate and compare](docs/guides/03-evaluate-and-compare.md) | A churn walk-through, cross-validation, plotting the results |
| [GPU acceleration](docs/guides/04-gpu-acceleration.md) | `tabfm_accelerate()`, checking which device served a query, the ROCm first compile |
| [Generate and impute](docs/guides/05-generate-and-impute.md) | Synthetic rows and filling NULL cells |

### API reference

| Topic | Description |
|---|---|
| [Functions](docs/api/01-functions.md) | `tabfm_classify`, `tabfm_regress`, missing targets, segments, `opts`, output columns |
| [Models and weights](docs/api/02-models-and-weights.md) | Every model, download / load / unload / remove, the cache, secrets, `tabfm_register_model` |
| [Generation](docs/api/03-generation.md) | `tabfm_generate`, `tabfm_impute`, options, outputs, cost |
| [GPU acceleration](docs/api/04-gpu.md) | CUDA, ROCm and MLX plugins, the support matrix, precision, platform limits |
| [Settings](docs/api/05-settings.md) | Every setting with its default, and the `opts` keys |
| [Troubleshooting](docs/api/06-troubleshooting.md) | Error messages, what they mean, what to run |
| [Evaluation](docs/api/07-evaluation.md) | Metrics, cross-validation, scoring rules |
| [Plotting](docs/api/08-plotting.md) | The six `_by` macros for anofox-visualization |

### Background

The full index, including design notes and the history of plans and reviews, is
[docs/README.md](docs/README.md).

| Document | What it is |
|---|---|
| [docs/REAL_MODELS.md](docs/REAL_MODELS.md) | Reference: each real model, how it was ported and verified, licences, gated repositories |
| [docs/GENERATE.md](docs/GENERATE.md) | Reference: how generation and imputation work, and what binning costs |
| [docs/BUILDING.md](docs/BUILDING.md) | Reference: building and testing from source |
| [docs/DYNAMIC_BACKENDS.md](docs/DYNAMIC_BACKENDS.md), [docs/KV_CACHE_DESIGN.md](docs/KV_CACHE_DESIGN.md) | Design: GPU backends as runtime plugins; the context cache |
| [docs/ROCM_SINGLE_EVAL_POS.md](docs/ROCM_SINGLE_EVAL_POS.md), [docs/MLX_SPIKE_RESULTS.md](docs/MLX_SPIKE_RESULTS.md) | Report: why some models cannot use ROCm; the MLX spike |
| [docs/README.md](docs/README.md#history-plans-and-reviews) | History: the plans and reviews behind each feature |
| [CHANGELOG.md](CHANGELOG.md) | Every release |

**Moved sections.** The old README's sections now live here (their links still land on this table):

| Old section | Now |
|---|---|
| <a id="quickstart"></a><a id="1-install--load"></a><a id="2-pick-a-model-and-download-its-weights-once"></a><a id="3-predict"></a>Quickstart | [Quick Start](#quick-start), [Getting Started](docs/guides/01-getting-started.md) |
| <a id="we-support-classification-and-regression"></a><a id="output-columns"></a>Classification and regression | [Functions](docs/api/01-functions.md) |
| <a id="generate-synthetic-data-and-fill-gaps"></a><a id="output-columns-1"></a><a id="options"></a>Generate synthetic data and fill gaps | [Generate and impute](docs/guides/05-generate-and-impute.md), [Generation](docs/api/03-generation.md) |
| <a id="multiple-models-the-registry"></a><a id="managing-weights-and-devices"></a>The registry, managing weights | [Choosing a model](docs/guides/02-choosing-a-model.md), [Models and weights](docs/api/02-models-and-weights.md) |
| <a id="a-full-worked-example"></a><a id="evaluation-cross-validation--scoring"></a><a id="plot-with-anofox-visualization"></a>Worked example, evaluation, plotting | [Evaluate and compare](docs/guides/03-evaluate-and-compare.md), [Evaluation](docs/api/07-evaluation.md), [Plotting](docs/api/08-plotting.md) |
| <a id="settings"></a>Settings | [Settings](docs/api/05-settings.md) |
| <a id="flavors-cpu--gpu"></a><a id="status--scope"></a>Flavors (CPU / GPU), Status & scope | [GPU acceleration](docs/guides/04-gpu-acceleration.md), [GPU](docs/api/04-gpu.md) |
| <a id="building"></a>Building | [docs/BUILDING.md](docs/BUILDING.md) |

## Status

The extension is calendar-versioned (`2026.10.26` is the twenty-sixth of October 2026) and is built and
tested on Linux (amd64, arm64), macOS (arm64) and Windows (amd64) against DuckDB 1.5.6, and on DuckDB 2.0.
Every model runs a single estimator (`n_estimators` > 1 is not available yet). What each GPU backend serves:
[the support matrix](docs/api/04-gpu.md#backend-support-matrix).

## Development

```bash
git clone --recurse-submodules https://github.com/DataZooDE/anofox-tabfm.git
cd anofox-tabfm
GEN=ninja make release      # needs vcpkg: see docs/BUILDING.md
make test_release           # sqllogictests + C++ unit tests (fixture models, no real weights)
```

Prerequisites, the GPU plugins and source-build flavors: [docs/BUILDING.md](docs/BUILDING.md).

## License

- **This extension's code:** MIT, see [LICENSE](LICENSE).
- **Model weights:** the extension ships only **weight-free** graphs and never redistributes weights. Each
  model's weights are **your own Hugging Face download**, under that model's licence, listed in
  [Models](#models) and in `tabfm_list_models()` (`license`, `commercial`). The repository and test fixtures
  contain **zero model weight bytes** (fixtures are random-init).
- **Non-commercial models:** `tabfm-v1`, `tabpfn-v2-5`, `tabpfn-v2-5-real`, `tabpfn-v2-6`, `tabpfn-v3` and
  `causilo`. `causilo`'s licence also requires a separate licence for hosted, API or SaaS use.
- **Attribution:** `tabpfn-v2` is under the Prior Labs License v1.1 (Apache-2.0 plus attribution).
  `limix-2m` is under the Stable AI Technology Co., Ltd. License v1.0 (Apache-2.0 plus Section 10): if you
  make available a product or service that incorporates it, link the
  [licence](https://huggingface.co/stable-ai/LimiX-2M/blob/main/LICENSE.txt) and display
  **"Built with StableAI LimiX"**; a model derived from its weights must start its name with "LimiX". Details:
  [docs/REAL_MODELS.md](docs/REAL_MODELS.md#licence).

Licences are as published by each project at the time of writing; this is not legal advice.

## Telemetry

Sends **anonymous** usage telemetry (extension load and per-function call counts: no data, no queries, no
identifiers beyond a hashed machine id) to PostHog EU, matching the other anofox extensions. CI environments
are auto-detected and telemetry is disabled there. Opt out any time with `export DATAZOO_DISABLE_TELEMETRY=1`
or `SET anofox_telemetry_enabled = false;`. What is collected: [TELEMETRY.md](TELEMETRY.md).

## Feedback

If `anofox_tabfm` misbehaves — a model that will not load, a prediction that looks wrong, a device it will
not use — please [open an issue](https://github.com/DataZooDE/anofox-tabfm/issues). Hardware, drivers and
hub availability differ in ways we cannot reproduce here, so a report with your setup is the fastest path to
a fix. Errors from the model functions end with that link.

If it saved you time, a star on the repo helps other people find it.

The first time you load the extension in an interactive terminal each day, a small banner says the same
thing. It never prints when output is piped, in notebooks, or in CI. Silence it with
`SET datazoo_banner = false;` or `DATAZOO_NO_BANNER=1`.
