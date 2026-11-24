# anofox-tabfm — agent guide

DuckDB C++ extension embedding Google's TabFM tabular foundation model
(TabPFN-style in-context learner). The CPU device runs the bundled ONNX graphs on a vendored **ggml** engine
(`vendor/anofox-ggml-adapter`, shared with anofox-decide; no build links ONNX Runtime, see `docs/BUILDING.md` "Engine" and `docs/GGML_PLAN.md`).
ONNX Runtime survives only as Python dev tooling (the oracle in `tools/ggml_parity`). Zero-shot
classification and regression as SQL. Spec (normative):
`/home/jr/Projects/tmp/research/2026-07-04-anofox-tabfm-spec/` — BRD, HLD,
SQL-API rev 4, spikes S01–S06 (all executed; read the RESULTS.md of the spike
backing your module before writing code).

## Build & test

```bash
make debug            # the ggml engine (vendor/anofox-ggml-adapter); no ONNX Runtime is built or downloaded
make release
make test_debug       # sqllogictests + C++ tests via build/debug/test/unittest
./build/debug/test/unittest test/sql/settings.test          # single file
# GPU plugins are the adapter's; build them inside vendor/anofox-ggml-adapter (docs/BUILDING.md)
cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DANOFOX_GGML_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=89
cmake -S tools/ggml_plugin -B build/ggml-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DANOFOX_GGML_GPU=hip -DAMDGPU_TARGETS=gfx1201
cmake -S tools/mlx_plugin -B build/mlx -G Ninja -DCMAKE_BUILD_TYPE=Release          # Apple Silicon
```

- DuckDB pinned v1.5.6 (submodule), extension-ci-tools v1.5-variegata.
- Tests always run with `DATAZOO_DISABLE_TELEMETRY=1` (Makefile does this).
- C++ tests: Catch2 TUs listed in `TABFM_CPP_TEST_SOURCES` in CMakeLists.txt,
  compiled into the unittest binary (use `#include "catch.hpp"`).

## Working rules

1. **Red-green TDD.** Write the failing sqllogictest/Catch2 test first, then
   implement. Error-path tests are first-class (SQL-API §5 catalog).
2. **A new ONNX op or attribute form** goes into the shared engine, `vendor/anofox-ggml-adapter` (its own repo, PR there first, then bump
   the submodule): a golden in its `tools/ggml_parity/gen_op_goldens.py` (ORT is the oracle), then the lowering in `src/ggml_lower.cpp`
   and the row in `include/anofox_ggml_adapter/ggml_ops.hpp`.
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
- `src/tabfm_engine.cpp` (dispatch: `TryGgmlGpuBackend`, `TryMlxBackend`), `tabfm_plugin_backend.cpp` (tabfm's glue over the adapter's
  plugin loader: weights by name, the y-prefix rule, the input contract), `src/include/tabfm_plugin_artifacts.hpp` (tabfm's wording over
  the adapter's `plugin_artifacts.hpp`: file names, release tag, sha256 sidecars); `tabfm_devices.cpp` wraps the adapter's `devices.hpp`
- `vendor/anofox-ggml-adapter` — the ggml engine, shared with anofox-decide (ONNX reader, host shape pass, lowering and `GgmlExecutor`,
  the op table, per-op goldens against ONNX Runtime; ggml itself is its submodule). `src/include/tabfm_ggml.hpp` brings its names into
  `duckdb::anofox`; `src/tabfm_ggml_session.cpp` is the ggml-backed `TabFMSession` (the TabFM contract). The adapter also owns
  the GPU plugins (`src/plugin_ggml.cpp`, `src/plugin_mlx.cpp` + `src/mlxgraph/`, plugin ABI v3 in `plugin_abi.h`), built from the same
  sources and released from its own repo (`.github/workflows/plugins.yml`). Whole-graph parity for tabfm's graphs: `tools/ggml_parity/graphs_parity.py`.
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

There are no build flavors: `TABFM_FLAVOR`, `TABFM_ORT_*`, `TABFM_WITH_ORT` and `cmake/ort.cmake` are removed, and
**a GPU is never a build**. Every GPU backend is a standalone plugin the extension `dlopen`s at runtime. The plugins
are **anofox-ggml-adapter's**, shared with anofox-decide: tabfm builds and ships none of them (its own plugin sources,
`tools/ggml_plugin` and the ABI v2 header are gone). They speak the adapter's generic plugin ABI v3
(`include/anofox_ggml_adapter/plugin_abi.h`: named tensors, host-supplied weights by name with shape, outputs by name,
a layout fingerprint); the loader refuses an ABI v2 plugin. `src/tabfm_plugin_backend.cpp` is tabfm's side of it.

- the **ggml GPU plugin** (adapter `src/plugin_ggml.cpp`, `tools/ggml_plugin`, `-DANOFOX_GGML_GPU=cuda|hip`), the only
  lane for `cuda` and `rocm`, from the same engine sources as the CPU: `libanofox_ggml-cuda_plugin.so` (CUDA runtime +
  cuBLAS linked statically, needs only `libcuda.so.1`; the adapter's `scripts/check_plugin_deps.sh` enforces it) and
  `libanofox_ggml-rocm_plugin.so` (links the system's HIP, hipBLAS, rocBLAS). Reads the plain `ext` graph.
- the **MLX plugin** (adapter `src/plugin_mlx.cpp` + `src/mlxgraph/`, `tools/mlx_plugin`) — Apple MLX,
  `libanofox_mlx_plugin.dylib`; runs the shipped ONNX graph itself (with the adapter's `onnx_reader.cpp`, so no
  libonnx/libprotobuf). mitra also has a hand-ported forward, kept as the fast path and as an independent oracle for
  the interpreter. macOS/arm64 only; see `docs/MLX_PLAN.md`. tabfm's CMakeLists builds it (target
  `anofox_tabfm_mlx_plugin`) on an arm64 Mac only so `test/cpp/test_tabfm_mlx_plugin.cpp` can run.

Windows names drop the `lib` prefix and end in `.dll`. The adapter's `.github/workflows/plugins.yml` publishes them with
sha256 sidecars (`ggml-cuda_plugin.sha256`, `ggml-rocm_plugin.sha256`, `mlx_plugin.sha256`) on its release
`ANOFOX_GGML_PLUGIN_RELEASE_TAG` (adapter `plugin_artifacts.hpp`), which `tabfm_download_runtime` / `tabfm_accelerate` fetch.

Every build discovers GPUs (`tabfm_devices()` probes NVML and the KFD sysfs
topology through the adapter's `devices.hpp` — neither needs a vendor SDK; `ep` is `ggml`, `ggml-cuda`, `ggml-rocm`
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
