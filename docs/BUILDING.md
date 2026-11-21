# Building from source

Clone with submodules (`git clone --recurse-submodules`; they include DuckDB, the CI tools, the shared
`posthog-telemetry`, `datazoo-banner` and `duckdb-compat` libraries, and the upstream model code under
`vendor/`), then:

```bash
# vcpkg is required for EVERY flavor: vcpkg.json depends on openssl unconditionally.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                     # cpu flavor on the ggml engine (vendor/ggml submodule): no ONNX Runtime anywhere
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

## Engines

The CPU device runs on **ggml** (`vendor/ggml`, a pinned git submodule compiled into the extension, CPU backend only, no OpenMP,
AVX2/FMA baseline on x86-64), and a default `cpu` build links **no ONNX Runtime at all**: no download, no vcpkg ORT port. The
bundled ONNX graphs are lowered op by op to ggml graphs (`docs/GGML_PLAN.md`, `docs/GGML_LOWERING.md`); every op is checked
against ONNX Runtime's output (`tools/ggml_parity/`).

| CMake option | Default | Meaning |
|---|---|---|
| `TABFM_WITH_ORT` | `OFF` for the `cpu` flavor, `ON` for `cuda`/`rocm`/`coreml` | link ONNX Runtime (only those in-process flavors need it) |
| `TABFM_GGML_NATIVE` | `OFF` (`ON` on MSVC arm64, which ggml requires) | compile ggml for this machine's CPU (AVX-512 etc.): faster here, may crash on older CPUs |

ggml is not optional: a checkout without `vendor/ggml` (`git submodule update --init vendor/ggml`) does not configure. The ONNX Runtime CPU engine, its vcpkg
port and the upstream download it used to default to are gone (the upstream source and release archives no longer exist); `TABFM_ORT_URL` (a mirror or a local
archive) is still needed by the `cuda`/`rocm`/`coreml` flavors, which link ONNX Runtime in the extension. Python `onnxruntime` remains the reference for the parity tools.
Debugging aids: `TABFM_GGML_PROFILE=1|2` prints phase timings (2 adds per-op-kind compute times).

The test suite runs against random-init fixture models and needs no real weights and no network. A
locally built extension is unsigned: start DuckDB with `duckdb -unsigned` and `LOAD` the file, or use the
`duckdb` binary in `build/release`, which has the extension linked in. Contributors: the module map and
the project's rules are in [`CLAUDE.md`](../CLAUDE.md); end-to-end examples are in
[`examples/`](../examples/README.md).

## Flavors

`TABFM_FLAVOR` (`cpu` by default; `cuda`, `rocm`) selects which ONNX Runtime gets linked for a source
build; the sources are identical. Released artifacts are always the `cpu` flavor: GPUs are served by
plugins, not by flavors ([GPU](api/04-gpu.md)). `TABFM_FLAVOR` also remains the only mechanism for
`coreml`, which is dropped as a backend, since MLX supersedes it on the only platform it targets.

## GPU plugins from source

The plugins are published as release assets, so you rarely need to build one. To do so: the sources are
`src/tabfm_cuda_plugin.cpp`, `src/tabfm_migraphx_plugin.cpp` and `src/tabfm_mlx_plugin.cpp` (the last
needs `brew install mlx mlx-c` and builds via the `anofox_tabfm_mlx_plugin` CMake target). The ROCm
plugin needs only MIGraphX; building an ONNX Runtime with the MIGraphX EP for the legacy `rocm` flavor is
described in [rocm-build.md](rocm-build.md). How the pieces fit together, as designed:
[DYNAMIC_BACKENDS.md](DYNAMIC_BACKENDS.md).

The **ggml GPU plugin** (experimental; the same engine as the CPU build on a CUDA or HIP device, see
[GGML_PLAN.md](GGML_PLAN.md) "P8 result") is a separate CMake project:
`cmake -S tools/ggml_plugin -B build/ggml-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DTABFM_GGML_GPU=cuda -DCMAKE_CUDA_ARCHITECTURES=89`
(`-DTABFM_GGML_GPU=hip -DAMDGPU_TARGETS=gfx1201` for ROCm), then `cmake --build build/ggml-cuda`. It needs `vendor/ggml` and the vendor's toolkit,
not ONNX Runtime. `tools/ggml_plugin/abi_check.cpp` checks a built plugin against an ORT reference. Only the CUDA plugin is wired to the host today.

Any build can host the CUDA plugin, including a local debug build that loads a shared
`libonnxruntime.so`: the plugin's ONNX Runtime ships under a private SONAME (`libanofoxort_gpu.so`,
applied by `tabfm_download_runtime` and by the plugin build itself), and the plugin loader binds with
`RTLD_DEEPBIND`, so an ONNX Runtime already loaded in the process cannot shadow it.

Releases are calendar-versioned tags (`vYYYY.MM.DD`); see [CHANGELOG.md](../CHANGELOG.md).
