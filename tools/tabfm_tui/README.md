# tabfm-tui

A small [Textual](https://textual.textualize.io/) TUI to try the
`anofox_tabfm` DuckDB extension quickly — pick a dataset, a target, a model, and
see predictions. **Dev/demo tool only:** the extension itself is a native
`.duckdb_extension` with no Python dependency; this app merely *loads* it.

```bash
cd tools/tabfm_tui
uv run tabfm-tui                          # auto-detects build/{release,debug}/…
TABFM_EXT_PATH=/path/to/anofox_tabfm.duckdb_extension uv run tabfm-tui
```

- Left: the model **registry** (`tabfm_list_models()`) + current `anofox_tabfm_*`
  settings.
- Right: a predict form (dataset, target, task, `model :=`, optional `test`
  relation) with quick-pick Hugging Face demos (iris / churn / tips) and a
  results table. Real predictions need the downloaded model weights and network
  for `hf://` datasets.
