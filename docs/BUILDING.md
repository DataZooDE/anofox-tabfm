# Building from source

Clone with submodules (`git clone --recurse-submodules`; they include DuckDB, the CI tools, the shared
`posthog-telemetry`, `datazoo-banner` and `duckdb-compat` libraries, and the upstream model code under
`vendor/`), then:

```bash
# vcpkg is required: vcpkg.json depends on openssl unconditionally.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                     # the ggml engine (vendor/ggml submodule): no ONNX Runtime anywhere
GEN=ninja make debug                       # debug build
make test_release                          # sqllogictests + C++ unit tests (telemetry disabled)
./build/release/test/unittest test/sql/tabfm_classify.test   # a single file
make check-docs                            # every SQL example in README, docs/api, docs/guides and the
                                           # Claude Code skills, plus function / setting / model names
```

Without `VCPKG_TOOLCHAIN_PATH` the build stops during configure with a cascade in which only the
first line is the real cause:

```
CMake Error: Could not find toolchain file:
  .../vcpkg_installed//share/vcpkg/scripts/buildsystems/vcpkg.cmake
CMake Error: CMake was unable to find a build program corresponding to "Ninja".
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

CMake aborts before probing for the generator or the compilers, so the last three are reported
missing even when Ninja and the toolchain are installed and on `PATH`.

Requires CMake ≥ 3.10 (≥ 3.19 to also build the bundled C++ unit tests), a C++17 toolchain, and vcpkg (for OpenSSL).
`TABFM_BUILD_ROOT` gives a worktree its own build directory.

## Engine

The CPU device runs on **ggml** (`vendor/ggml`, a pinned git submodule compiled into the extension, CPU backend only, no OpenMP,
AVX2/FMA baseline on x86-64). No build links **ONNX Runtime**: there is no download, no vcpkg ORT port and no build flavor; the
`TABFM_FLAVOR` switch, the `TABFM_ORT_*` make variables, `TABFM_WITH_ORT` and `cmake/ort.cmake` are gone. The bundled ONNX graphs are
lowered op by op to ggml graphs (`docs/GGML_PLAN.md`, `docs/GGML_LOWERING.md`); every op is checked against ONNX Runtime's output
(`tools/ggml_parity/`). Python `onnxruntime` remains only as development tooling: the oracle for the per-op goldens and the parity scripts.

| CMake option | Default | Meaning |
|---|---|---|
| `TABFM_GGML_NATIVE` | `OFF` (`ON` on MSVC arm64, which ggml requires) | compile ggml for this machine's CPU (AVX-512 etc.): faster here, may crash on older CPUs |

ggml is not optional: a checkout without `vendor/ggml` (`git submodule update --init vendor/ggml`) does not configure.
Debugging aids: `TABFM_GGML_PROFILE=1|2` prints phase timings (2 adds per-op-kind compute times).

The test suite runs against random-init fixture models and needs no real weights and no network. A
locally built extension is unsigned: start DuckDB with `duckdb -unsigned` and `LOAD` the file, or use the
`duckdb` binary in `build/release`, which has the extension linked in. Contributors: the module map and
the project's rules are in [`CLAUDE.md`](../CLAUDE.md); end-to-end examples are in
[`examples/`](../examples/README.md).

## GPU plugins from source

GPUs are served by plugins, not by builds of the extension ([GPU](api/04-gpu.md)). The plugins are
published as release assets (`.github/workflows/gpu_plugins.yml`), so you rarely need to build one.

**The ggml GPU plugin** (CUDA and ROCm) is `src/tabfm_ggml_plugin.cpp` plus the engine sources, built by its
own CMake project in `tools/ggml_plugin`. It needs `vendor/ggml` and the vendor's toolkit (the CUDA toolkit,
or ROCm's HIP SDK), not ONNX Runtime:

```bash
# CUDA: libanofox_tabfm_ggml-cuda_plugin.so (the CUDA runtime and cuBLAS are linked statically)
cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=cuda \
      -DCMAKE_CUDA_ARCHITECTURES=89          # your card; the release builds 75-real;80-real;86-real;89-real;90-real;100-real;120
cmake --build build/ggml-cuda

# ROCm: libanofox_tabfm_ggml-rocm_plugin.so (links the ROCm install's HIP, hipBLAS, rocBLAS)
cmake -S tools/ggml_plugin -B build/ggml-rocm -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=hip \
      -DAMDGPU_TARGETS=gfx1201               # your arch; the release builds gfx1030;1100;1101;1102;1150;1151;1200;1201;90a;942
cmake --build build/ggml-rocm
```

Copy the library into the directory `anofox_tabfm_ep_path` names (by default the `runtime` subdirectory of
`anofox_tabfm_cache_dir`) and `SET anofox_tabfm_device = 'cuda'` (or `'rocm'`); `SELECT model, device FROM
tabfm_models()` confirms it served. `TABFM_GGML_CUDA_STATIC` (default `ON`) selects the static CUDA runtime, and
`scripts/check_plugin_deps.sh` fails on a plugin that still needs a shared one. On Windows the CUDA plugin
builds as `anofox_tabfm_ggml-cuda_plugin.dll` with a shared CUDA runtime (there is no static cuBLAS there);
it is not published. `tools/ggml_plugin/abi_check.cpp` loads a built plugin through the plugin ABI and
compares one reference case with ONNX Runtime's output.

**The MLX plugin** (`src/tabfm_mlx_plugin.cpp`, Apple Silicon) needs `brew install mlx mlx-c` and builds
via the `anofox_tabfm_mlx_plugin` CMake target of the main build ([MLX_PLAN.md](MLX_PLAN.md)).

The design notes of the removed ONNX Runtime and MIGraphX backends are in [archive/](archive/README.md).

Releases are calendar-versioned tags (`vYYYY.MM.DD`); see [CHANGELOG.md](../CHANGELOG.md).
