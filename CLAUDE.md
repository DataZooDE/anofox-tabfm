# anofox-tabfm — agent guide

DuckDB C++ extension embedding Google's TabFM tabular foundation model
(TabPFN-style in-context learner) via ONNX Runtime. Zero-shot classification
and regression as SQL. Spec (normative):
`/home/jr/Projects/tmp/research/2026-07-04-anofox-tabfm-spec/` — BRD, HLD,
SQL-API rev 4, spikes S01–S06 (all executed; read the RESULTS.md of the spike
backing your module before writing code).

## Build & test

```bash
make debug            # cpu flavor; ORT from prebuilt archive unless the
make release          # vcpkg "ort-vcpkg" manifest feature is enabled
make test_debug       # sqllogictests + C++ tests via build/debug/test/unittest
./build/debug/test/unittest test/sql/settings.test          # single file
TABFM_FLAVOR=rocm make release TABFM_ORT_ROCM_DIR=...       # rocm flavor
```

- DuckDB pinned v1.5.5 (submodule), extension-ci-tools v1.5-variegata.
- Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
- C++ tests: Catch2 TUs listed in `TABFM_CPP_TEST_SOURCES` in CMakeLists.txt,
  compiled into the unittest binary (use `#include "catch.hpp"`).

## Working rules

1. **Red-green TDD.** Write the failing sqllogictest/Catch2 test first, then
   implement. Error-path tests are first-class (SQL-API §5 catalog).
2. **File ownership.** One module = one `src/tabfm_*.cpp` (+ its headers under
   `src/include/`, tests under `test/`). Do not edit shared files
   (CMakeLists.txt source list, anofox_tabfm_extension.cpp,
   tabfm_registration.hpp, tabfm_settings.cpp) without coordinating — they are
   scaffold-owned.
3. **Telemetry.** Every user-facing function calls
   `PostHogTelemetry::Instance().CaptureFunctionExecution("<short_name>")`
   once per execution (bind or global-state init, not per chunk).
4. **Naming.** Full names `anofox_tabfm_*` + short alias `tabfm_*` via
   `anofox_function_alias.hpp` helpers. Settings `anofox_tabfm_*`.
5. **Errors.** Every failure is a DuckDB exception whose message names the
   fixing SET/CALL (SQL-API §5 has the exact texts).
6. **License wall.** No Google weight bytes anywhere in the repo — fixtures
   are random-init (S06); real graphs in `resources/` are weight-free stubs.
   `unnest(res, max_depth := 3)` — never `recursive := true` (S04).

## Layout

- `src/tabfm_safetensors.cpp`, `tabfm_manifest.cpp` — WS-B (readers)
- `src/tabfm_ort_engine.cpp`, `tabfm_devices.cpp`, `tabfm_state.cpp` — WS-C
- `src/tabfm_weights.cpp` — WS-D (download/lifecycle/license gate)
- `src/tabfm_predict_agg.cpp`, `tabfm_macros.cpp` — WS-E (predict surface)
- `src/tabfm_preprocess.cpp`, `tabfm_ensemble.cpp` — WS-F
- `tools/export_onnx`, `tools/make_fixture`, `tools/parity` — WS-A (uv projects)
- `vendor/tabfm` — upstream Apache-2.0 submodule; never edit, patch copies in tools/
- `test/fixtures/` — committed fixture model (graph, safetensors, golden.json,
  manifest.json); CI verifies the pinned sha256, never regenerates

## Flavors and backend plugins (HLD D9, docs/DYNAMIC_BACKENDS.md)

One codebase; `TABFM_FLAVOR=cpu|cuda|rocm|coreml` still picks the ORT build in
`cmake/ort.cmake`, but **a GPU is no longer a flavor**. Both GPU backends are
standalone plugins the extension `dlopen`s at runtime through
`src/include/tabfm_plugin_abi.h`:

- `src/tabfm_migraphx_plugin.cpp` — drives MIGraphX directly, no ORT involved
- `src/tabfm_cuda_plugin.cpp` — links its own shared ORT-GPU distribution

So every build discovers GPUs (`tabfm_devices()` probes NVML and the KFD sysfs
topology in all flavors — neither needs a vendor SDK) and every build accepts
`SET anofox_tabfm_device='cuda'|'rocm'`; `SET anofox_tabfm_ep_path` says where
the plugin lives. CoreML is the exception and stays flavor-gated, being an
in-process ORT EP. GPU code paths must never be *required* for the cpu build —
that constraint is about what is LINKED, which is unchanged.

When touching any of this, two rules earned the hard way:

1. **A device switch must rebuild the session.** `CanReuseSession` in
   `tabfm_state.hpp` checks the device, not just the cache key. Ignoring it
   silently serves the previous device — no error, no GPU.
2. **Never trust "cpu and gpu agree".** That is also what a silent CPU
   fallback prints. Check `SELECT device FROM tabfm_models()`; the scenarios in
   `tools/gpu_test/scenarios/` all print `*_SERVED_BY` for this reason.
