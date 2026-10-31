# Building from source

Clone with submodules (`git clone --recurse-submodules`; they include DuckDB, the CI tools, the shared
`posthog-telemetry`, `datazoo-banner` and `duckdb-compat` libraries, and the upstream model code under
`vendor/`), then:

```bash
# vcpkg is required for EVERY flavor: vcpkg.json depends on openssl unconditionally.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                     # cpu flavor; ONNX Runtime linked statically via vcpkg
GEN=ninja make release TABFM_ORT_VCPKG=0   # faster local build against the prebuilt ONNX Runtime archive
GEN=ninja make debug                       # debug build (prebuilt ONNX Runtime archive, linked dynamically)
make test_release                          # sqllogictests + C++ unit tests (telemetry disabled)
./build/release/test/unittest test/sql/tabfm_classify.test   # a single file
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

Requires CMake ≥ 3.10 (≥ 3.19 to also build the bundled C++ unit tests), a C++17 toolchain, and vcpkg.
Release builds of the cpu flavor link ONNX Runtime statically through the `ort-vcpkg` manifest feature, so
the extension is one self-contained file; the first such build is slow. Local debug builds, and release
builds with `TABFM_ORT_VCPKG=0`, use the prebuilt archive instead. `TABFM_BUILD_ROOT` gives a worktree its
own build directory.

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

Any build can host the CUDA plugin, including a local debug build that loads a shared
`libonnxruntime.so`: the plugin's ONNX Runtime ships under a private SONAME (`libanofoxort_gpu.so`,
applied by `tabfm_download_runtime` and by the plugin build itself), and the plugin loader binds with
`RTLD_DEEPBIND`, so an ONNX Runtime already loaded in the process cannot shadow it.

Releases are calendar-versioned tags (`vYYYY.MM.DD`); see [CHANGELOG.md](../CHANGELOG.md).
