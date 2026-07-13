# anofox_tabfm Telemetry

`anofox_tabfm` collects **anonymous, privacy-preserving usage telemetry** so we
can see which capabilities are used, on which platforms, and where they fail —
and prioritise accordingly. It is **on by default** and **trivial to turn off**.

Telemetry is emitted through the shared
[`DataZooDE/posthog-telemetry`](https://github.com/DataZooDE/posthog-telemetry)
library and follows the cross-product **`telemetry_schema: 2`** envelope
(`posthog-telemetry/TELEMETRY-SCHEMA.md`). Ingestion is the EU PostHog cloud.

## How to turn it off

Any one of these fully short-circuits telemetry — when disabled, **nothing
leaves the machine** (the opt-out is enforced at the transport, not just at the
call sites):

```sql
SET anofox_telemetry_enabled = false;   -- DuckDB setting (per session)
```

```bash
export DATAZOO_DISABLE_TELEMETRY=1       # environment (1|true|yes)
```

Telemetry is also disabled automatically when a CI environment is detected. The
test suite always runs with `DATAZOO_DISABLE_TELEMETRY=1`.

## The guarantee: bounded, enumerated, non-PII

Every property we send is **either** a constant drawn from a small,
code-controlled enumeration **or** a pure number (durations, counts). The library
additionally clamps every outgoing string to 512 bytes as a backstop.

We **never** send: file paths, cache/model directories, table names, column
names, target/feature values, model weights or any row/result data, SQL text,
`WHERE`/`FILTER` clauses, device names, or error messages. Only the bounded
identifiers documented below leave the process.

The instrumentation is centralised: extension load is captured once in
`src/anofox_tabfm_extension.cpp`, and each user-facing function records exactly
one aggregated call at bind/registration time (never on a per-row path).

## What is collected

### Envelope (attached to every event)

`product` (`anofox_tabfm`), `product_version`, `product_edition` (`oss`),
`telemetry_schema` (`2`), `duckdb_version`, `os`, `arch`, `platform`, `is_ci`,
`is_container`, a per-process `$session_id`, and — once associated — the
`deployment` group. `distinct_id` is the SHA-256 of a machine id: a **stable,
pseudonymous** identifier, not tied to any personal data.

### Events

| Event | When | Properties (beyond the envelope) |
|---|---|---|
| `extension_loaded` | the `anofox_tabfm` extension loads | — |
| `function_executed` | a DuckDB function runs — **aggregated** per function per session (not per row) | `function_name`, `call_count`, `duration_ms_p50` |

`function_name` is always one of a fixed, code-controlled set of the extension's
own function names:

- `tabfm_predict` / `tabfm_predict_by` / `tabfm_predict_agg` / `tabfm_predict_win`
  (the prediction surface)
- `tabfm_download`, `tabfm_models`, `tabfm_load`, `tabfm_unload`, `tabfm_remove`,
  `tabfm_gpu_precompile` (weights lifecycle)
- `tabfm_devices` (device discovery)

No arguments, options, column/target values, or model contents are ever attached.

## Function-call aggregation

DuckDB function calls are recorded via `RecordFunctionCall(function_name)`, which
aggregates in-process into a single `function_executed` event per function per
session (carrying `call_count` and `duration_ms_p50`), flushed at session end.
Because recording happens at bind/registration and never on a per-row path, a
million-row prediction produces O(1) telemetry rows, not a firehose.

## Enterprise / account analytics

OSS `anofox_tabfm` associates only the `deployment` group. It has no license key,
so no `account` group is associated.
