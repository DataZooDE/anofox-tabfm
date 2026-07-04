# anofox-tabfm

Zero-shot machine learning for tabular data, inside DuckDB. This extension
embeds Google's [TabFM](https://github.com/google-research/tabfm) foundation
model (TabPFN-style in-context learner) via ONNX Runtime, so "train a model
for this churn/quality/demand column" becomes a single SQL statement — no
Python, no training loop.

```sql
LOAD anofox_tabfm;
SET anofox_tabfm_accept_hf_license = true;
CALL tabfm_download('classification');

-- fit on the labelled history, predict the new rows (like
-- TabFMClassifier.fit(X_train, y_train).predict(X_test))
SELECT * FROM tabfm_classify('history', 'churned', test := 'prospects');

-- or a single table: rows whose target IS NULL are the ones to score
SELECT * FROM tabfm_classify('customers', 'churned');
```

**Status: under construction** — full SQL surface with a placeholder engine;
the real TabFM forward pass is wired next. See `CLAUDE.md` for the module map.

## SQL surface

Two task-specific functions mirroring upstream `TabFMClassifier` /
`TabFMRegressor` — classification and regression:

```sql
tabfm_classify(data, target [, test] [, features] [, opts])
tabfm_regress (data, target [, test] [, features] [, opts])
```

- `data` — the context ("training") relation: a **table/view name or a
  parenthesised subquery** (`'(SELECT … WHERE year = 2025)'`).
- `test` — optional relation of rows to score. When given, only those rows are
  returned (their `target` is ignored). When omitted, rows of `data` whose
  target is `NULL` are scored (all rows come back, flagged `is_training`).
- `features` — optional `VARCHAR[]` of feature columns (default: all others).
- `opts` — trailing `MAP(VARCHAR, VARCHAR)` of options (`seed`, `n_estimators`,
  `output_mode`, `context_rows`, …). The task is fixed by the function.
- **Named parameters** work: `tabfm_classify('history', 'churned', test := 'prospects', opts := MAP{'seed':'42'})`.

Output: every column of the scored rows plus `yhat` (predicted label/value),
`yhat_score` (top-class probability; `NULL` for regression), `proba`
(`MAP(label → probability)`, classification only), and `is_training` in
single-table mode.

Weights and devices:

| Function | Purpose |
|---|---|
| `tabfm_download / tabfm_models / tabfm_load / tabfm_unload / tabfm_remove` | SQL-managed weights lifecycle |
| `tabfm_devices()` | discovered execution devices (CPU/CUDA/ROCm) |

The grouped, composable-aggregate, and windowed surfaces (`tabfm_predict_by` /
`_agg` / `_win`) will be added on the same engine when needed.

## Flavors

One codebase, three builds (`TABFM_FLAVOR`): `cpu` (fallback lane,
community-extension eligible), `cuda` (NVIDIA, bf16), `rocm` (AMD via ORT's
MIGraphX EP, fp16/bf16). GPU flavors are distributed via the anofox extension
repository; vendor runtimes (CUDA/cuDNN, ROCm) resolve from your system.

## License story

- **This extension's code:** MIT.
- **Model weights:** downloaded by *you* directly from
  `google/tabfm-1.0.0-pytorch` under Google's non-commercial license — the
  extension ships only a weight-free computation graph and never
  redistributes or transforms weights on disk. A license gate
  (`SET anofox_tabfm_accept_hf_license = true`) is enforced before any
  download. CI and test fixtures contain zero Google weight bytes.

## Telemetry

This extension sends **anonymous** usage telemetry (extension load + function
usage counts, no data, no queries, no identifiers beyond a hashed machine id)
to PostHog EU, matching the other anofox extensions. *(Deliberate deviation
from spec NFR-S1, decided 2026-07-04.)* Opt out any time:

- environment: `export DATAZOO_DISABLE_TELEMETRY=1`
- SQL: `SET anofox_telemetry_enabled = false;`
- CI environments are auto-detected and telemetry is disabled there.

## Building

```bash
GEN=ninja make release          # cpu flavor
make test_release               # sqllogictests + C++ tests
```

Requires: CMake ≥ 3.19, a C++17 toolchain, and (optionally) vcpkg. ONNX
Runtime is fetched as a prebuilt archive by default; enable the `ort-vcpkg`
manifest feature to build it from source via vcpkg.
