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

-- rows with churned IS NULL get predictions; the others are the in-context "training" set
SELECT * FROM tabfm_predict('customers', 'churned');
```

**Status: under construction** — scaffold + SQL surface stubs. See
`CLAUDE.md` for the module map and the spec repo for the full design.

## SQL surface (target, SQL-API rev 4)

| Function | Purpose |
|---|---|
| `tabfm_predict(tbl, target[, features][, opts])` | predict NULL-target rows of a table |
| `tabfm_predict_by(tbl, grp, target[, features][, opts])` | one in-context model per group |
| `tabfm_predict_agg(row, target[, opts])` | composable aggregate core (CTEs, joins, GROUP BY) |
| `tabfm_predict_win(row, target[, opts]) OVER (...)` | rolling in-context prediction |
| `tabfm_download / tabfm_models / tabfm_load / tabfm_unload / tabfm_remove` | SQL-managed weights lifecycle |
| `tabfm_devices()` | discovered execution devices (CPU/CUDA/ROCm) |

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
