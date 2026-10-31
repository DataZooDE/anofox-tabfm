# Building from source

```bash
# vcpkg is required for EVERY flavor: vcpkg.json depends on openssl unconditionally.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                     # cpu flavor
GEN=ninja TABFM_FLAVOR=cuda make release   # cuda flavor (linux_amd64 / win_amd64)
make test_release                          # sqllogictests + C++ unit tests
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

Requires CMake ≥ 3.10 (≥ 3.19 to also build the bundled C++ unit tests) and a
C++17 toolchain, and vcpkg. ONNX Runtime is fetched as a
prebuilt archive by default; enable the `ort-vcpkg` manifest feature to build it
from source. See [`CLAUDE.md`](../CLAUDE.md) for the module map and
[`examples/`](../examples/README.md) for end-to-end examples.

## Flavors

`TABFM_FLAVOR` still selects which ONNX Runtime gets linked for a source
build, and remains the mechanism for `coreml` alone — which is
[dropped](DYNAMIC_BACKENDS.md#phase-4--coreml--dropped-2026-09-19), since
MLX supersedes it on the only platform it targets.

## GPU plugins from source

You can still build one from source (`src/tabfm_cuda_plugin.cpp`,
`src/tabfm_migraphx_plugin.cpp`, `src/tabfm_mlx_plugin.cpp` — the last needs
`brew install mlx mlx-c` and builds via the `anofox_tabfm_mlx_plugin` CMake
target; see [`docs/rocm-build.md`](rocm-build.md) for the ROCm toolchain
and [`docs/DYNAMIC_BACKENDS.md`](DYNAMIC_BACKENDS.md) for how the pieces
fit together).

**Any build can host the CUDA plugin now.** Earlier, a host that loaded a
*shared* `libonnxruntime.so` (the local `make debug` build) shadowed the
plugin's own ONNX Runtime — same SONAME, first one loaded wins — so CUDA
required the statically-linked release build. Fixed at both ends
(`docs/GPU_HARDENING_PLAN.md` S2): the plugin's core ships under a private
SONAME (`libanofoxort_gpu.so`, applied by `tabfm_download_runtime` and by the
plugin build itself), and the plugin loader binds with `RTLD_DEEPBIND`.
Verified on hardware including the hostile case — a foreign upstream-SONAME
ORT preloaded into the global scope — which the plugin no longer notices.
