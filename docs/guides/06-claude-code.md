# Claude Code skills

anofox-tabfm ships a [Claude Code](https://claude.com/claude-code) plugin: five skills that give Claude the
extension's SQL surface as it is in this release (every function, named parameter, `opts` key, output
column, model, setting and error message), so it writes working `tabfm_*` SQL from a plain-English request
instead of guessing signatures.

## Install

In any Claude Code session:

```
/plugin marketplace add DataZooDE/anofox-tabfm
/plugin install anofox-tabfm@anofox-tabfm
```

`/plugin update anofox-tabfm@anofox-tabfm` takes a newer release. The plugin version is the extension
release it describes (`2026.11.01` describes `v2026.11.01`); when the two differ, the skills may name a
function your build does not have yet, or miss a new one.

Working on this repository, load the skills straight from the tree:

```bash
claude --plugin-dir ./plugins/anofox-tabfm
```

## The skills

Claude loads a skill by itself when a request matches it; you do not call them.

| Skill | Covers | Ask, for example |
|---|---|---|
| `anofox-tabfm-predict` | `tabfm_classify`, `tabfm_regress`: both forms, features, `opts`, output columns, segments, `query_batch` | *"predict churn for the prospects table from history"* |
| `anofox-tabfm-models` | the 23 built-in models, licences and commercial use, the licence gate, download / load / unload / remove, `tabfm_register_model`, the context cache | *"which model can I use commercially for a 40-feature table?"* |
| `anofox-tabfm-evaluate` | classification and regression metrics, scoring rules, `tabfm_cross_validate`, the `_by` macros for anofox-visualization | *"5-fold CV of mitra against tabicl-v2, then plot ROC curves"* |
| `anofox-tabfm-generate` | `tabfm_generate`, `tabfm_impute`, options, fidelity checks | *"fill the NULL incomes in raw, then add 500 synthetic rows"* |
| `anofox-tabfm-gpu` | `tabfm_accelerate`, devices and backends, precision, the ROCm precompile, checking what served a query | *"use my AMD GPU and confirm it is actually used"* |

Each skill leads with the mistakes that are easy to make and hard to spot: a `test` relation that still has
the target column, metrics that need `avg :=`, cross-validation that takes the default model only, a GPU that
silently is not used.

## How the skills stay correct

The skills are plain markdown in [`plugins/anofox-tabfm/skills/`](../../plugins/anofox-tabfm/skills/),
checked on every pull request together with this documentation by
[`scripts/check_doc_snippets.py`](../../scripts/check_doc_snippets.py):

- every ```` ```sql ```` example runs against the built extension, with the committed weight-free fixture
  model standing in for each model it names (signatures, parameters, option keys and output columns are
  checked for real, with no weights and no network);
- every `tabfm_*` / `anofox_tabfm_*` name must be a registered function or setting, and every model id a
  built-in;
- the model tables in the README, [Models and weights](../api/02-models-and-weights.md) and the models skill
  must each list every built-in model;
- the plugin version must match every skill and must not be older than the latest release.

Run it locally after `make release`:

```bash
make check-docs          # the full check, against build/release/duckdb
make check-docs-static   # manifests, versions and model tables only, no build needed
```

A change to a function, a setting or the model registry updates the matching skill in the same pull request;
`scripts/bump_plugin_version.sh <YYYY.MM.DD>` sets the plugin version when a release is cut.
