# anofox-tabfm — agent guide

DuckDB C++ extension embedding Google's TabFM tabular foundation model
(TabPFN-style in-context learner). The CPU device runs the bundled ONNX graphs on a vendored **ggml** engine
(`vendor/ggml`; no build links ONNX Runtime, see `docs/BUILDING.md` "Engine" and `docs/GGML_PLAN.md`).
ONNX Runtime survives only as Python dev tooling (the oracle in `tools/ggml_parity`). Zero-shot
classification and regression as SQL. Spec (normative):
`/home/jr/Projects/tmp/research/2026-07-04-anofox-tabfm-spec/` — BRD, HLD,
SQL-API rev 4, spikes S01–S06 (all executed; read the RESULTS.md of the spike
backing your module before writing code).

## Build & test

```bash
make debug            # the ggml engine (vendor/ggml); no ONNX Runtime is built or downloaded
make release
make test_debug       # sqllogictests + C++ tests via build/debug/test/unittest
./build/debug/test/unittest test/sql/settings.test          # single file
cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DTABFM_GGML_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=89  # GPU plugin
cmake -S tools/ggml_plugin -B build/ggml-rocm -G Ninja -DTABFM_GGML_GPU=hip -DAMDGPU_TARGETS=gfx1201       # (docs/BUILDING.md)
```

- DuckDB pinned v1.5.6 (submodule), extension-ci-tools v1.5-variegata.
- Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
- C++ tests: Catch2 TUs listed in `TABFM_CPP_TEST_SOURCES` in CMakeLists.txt,
  compiled into the unittest binary (use `#include "catch.hpp"`).

## Working rules

1. **Red-green TDD.** Write the failing sqllogictest/Catch2 test first, then
   implement. Error-path tests are first-class (SQL-API §5 catalog).
2. **A new ONNX op or attribute form** needs, first, a golden in `tools/ggml_parity/gen_op_goldens.py` (ORT is the oracle; regenerate with
   `uv run tools/ggml_parity/gen_op_goldens.py`), then the lowering in `src/tabfm_ggml_lower.cpp` and the row in `tabfm_ggml_ops.hpp`.
   `GgmlRefusal` names any op the engine lacks. Never compare a golden with a loose tolerance to make it pass: exact ops are bit-exact.
3. **File ownership.** One module = one `src/tabfm_*.cpp` (+ its headers under
   `src/include/`, tests under `test/`). Do not edit shared files
   (CMakeLists.txt source list, anofox_tabfm_extension.cpp,
   tabfm_registration.hpp, tabfm_settings.cpp) without coordinating — they are
   scaffold-owned.
4. **Telemetry.** Every user-facing function calls
   `PostHogTelemetry::Instance().CaptureFunctionExecution("<short_name>")`
   once per execution (bind or global-state init, not per chunk).
5. **Naming.** Full names `anofox_tabfm_*` + short alias `tabfm_*` via
   `anofox_function_alias.hpp` helpers. Settings `anofox_tabfm_*`.
6. **Errors.** Every failure is a DuckDB exception whose message names the
   fixing SET/CALL (SQL-API §5 has the exact texts).
7. **License wall.** No Google weight bytes anywhere in the repo — fixtures
   are random-init (S06); real graphs in `resources/` are weight-free stubs.
   `unnest(res, max_depth := 3)` — never `recursive := true` (S04).
8. **Docs and skills ship with the code.** A change to a function, its parameters or `opts` keys, a
   setting, an error text or the model registry updates, in the same PR, the matching page under
   `docs/api/` and the matching skill in `plugins/anofox-tabfm/skills/`: predict ← `tabfm_macros.cpp`,
   `tabfm_predict_agg.cpp`; models ← `tabfm_registry.cpp`, `tabfm_weights.cpp`, `tabfm_settings.cpp`;
   evaluate ← `tabfm_metrics_*.cpp`, `tabfm_scoring.cpp`, `tabfm_crossval.cpp`, `tabfm_contract.cpp`;
   generate ← `tabfm_generate*.cpp`; gpu ← `tabfm_devices.cpp`, the plugins, `tabfm_backends`. A new model
   gets a row in the README, `docs/api/02-models-and-weights.md` and the models skill. `make check-docs`
   (needs `make release`) must pass; CI runs it on the linux artifact. At a release,
   `scripts/bump_plugin_version.sh <YYYY.MM.DD>`.

## Layout

- `src/tabfm_safetensors.cpp`, `tabfm_manifest.cpp` — WS-B (readers)
- `src/tabfm_session.cpp`, `tabfm_devices.cpp`, `tabfm_state.cpp` — WS-C (sessions; the in-process `TabFMSession` is always ggml on the CPU; GPU devices are served by plugins)
- `src/tabfm_engine.cpp` (dispatch: `TryGgmlGpuBackend`, `TryMlxBackend`), `tabfm_plugin_backend.cpp` (the plugin loader), `src/include/tabfm_plugin_artifacts.hpp` (plugin file names, release tag, sha256 sidecars)
- `src/tabfm_ggml_lower.cpp` (ONNX graph -> ggml graph, `GgmlExecutor`), `tabfm_shape_eval.cpp` (host shape/constant pass, float64 interpreter),
  `tabfm_ggml_session.cpp` (the ggml-backed `TabFMSession`), `tabfm_ggml_engine.cpp`, `tabfm_ggml_ops.hpp` (the op table), `tabfm_rank_reduce.hpp` — the ggml engine
  (CPU, and the same sources in the GPU plugin).
  Every op is checked against ONNX Runtime: `tools/ggml_parity/` (per-op goldens in `test/fixtures/ggml_ops`, whole-graph `graphs_parity.py`)
- `src/tabfm_weights.cpp` — WS-D (download/lifecycle/license gate)
- `src/tabfm_predict_agg.cpp`, `tabfm_macros.cpp` — WS-E (predict surface)
- `src/tabfm_preprocess.cpp`, `tabfm_ensemble.cpp` — WS-F
- `tools/export_onnx`, `tools/make_fixture`, `tools/parity` — WS-A (uv projects)
- `vendor/tabfm` — upstream Apache-2.0 submodule; never edit, patch copies in tools/
- `plugins/anofox-tabfm/` — the Claude Code plugin (5 skills; marketplace in `.claude-plugin/`);
  `scripts/check_doc_snippets.py` + `scripts/doc_snippets/` check it and the user docs (`make check-docs`)
- `test/fixtures/` — committed fixture model (graph, safetensors, golden.json,
  manifest.json); CI verifies the pinned sha256, never regenerates

## Backend plugins (HLD D9)

There are no build flavors: `TABFM_FLAVOR`, `TABFM_ORT_*`, `TABFM_WITH_ORT` and `cmake/ort.cmake` are gone, and
**a GPU is never a build**. Every GPU backend is a standalone plugin the extension `dlopen`s at runtime through
`src/include/tabfm_plugin_abi.h`:

- `src/tabfm_ggml_plugin.cpp` — the **ggml GPU plugin**, the only lane for `cuda` and `rocm`. Built by its own
  CMake project, `tools/ggml_plugin` (`-DTABFM_GGML_GPU=cuda|hip`), from the same engine sources as the CPU;
  outputs `libanofox_tabfm_ggml-cuda_plugin.so` (CUDA runtime + cuBLAS linked statically, needs only
  `libcuda.so.1`; `scripts/check_plugin_deps.sh` enforces it) and `libanofox_tabfm_ggml-rocm_plugin.so` (links
  the system's HIP, hipBLAS, rocBLAS). Reads the plain `ext` graph; `.github/workflows/gpu_plugins.yml` builds
  and publishes both with sha256 sidecars.
- `src/tabfm_mlx_plugin.cpp` — Apple MLX; runs the shipped ONNX graph itself
  via `src/tabfm_mlx_graph.cpp` (+ `tabfm_onnx_reader.cpp`, a minimal
  wire-format parser so the plugin needs no libonnx/libprotobuf). mitra also
  has a hand-ported forward, kept as the fast path and as an independent
  oracle for the interpreter. macOS/arm64 only; see `docs/MLX_PLAN.md`.

Every build discovers GPUs (`tabfm_devices()` probes NVML and the KFD sysfs
topology — neither needs a vendor SDK; `ep` is `ggml`, `ggml-cuda`, `ggml-rocm`
or `mlx`) and every build accepts `SET anofox_tabfm_device='cuda'|'rocm'|'mlx'`;
`SET anofox_tabfm_ep_path` says where the plugin lives, `tabfm_download_runtime`
fetches it. `'coreml'` is accepted but errors (deprecated), `'migraphx'` aliases
`'rocm'`; `anofox_tabfm_mxr_source` and `anofox_tabfm_cpu_prepack` are no-ops.
GPU code paths must never be *required* for the extension build — that
constraint is about what is LINKED. The removed ONNX Runtime / MIGraphX design
notes are in `docs/archive/`.

When touching any of this, two rules earned the hard way:

1. **A device switch must rebuild the session.** `CanReuseSession` in
   `tabfm_state.hpp` checks the device, not just the cache key. Ignoring it
   silently serves the previous device — no error, no GPU.
2. **Never trust "cpu and gpu agree".** That is also what a silent CPU
   fallback prints. Check `SELECT device FROM tabfm_models()`; the scenarios in
   `tools/gpu_test/scenarios/` all print `*_SERVED_BY` for this reason.
