# Building ONNX Runtime with the MIGraphX EP (rocm flavor)

The `rocm` flavor of anofox_tabfm links an ONNX Runtime that carries the
MIGraphX execution provider (AMD's inference path; the upstream ROCm EP was
removed in ORT 1.23). Microsoft publishes **no prebuilt** ORT-MIGraphX
archive, so it must be built from source and handed to CMake via
`-DTABFM_ORT_ROCM_DIR=<install tree>` (see `cmake/ort.cmake`).

This document reproduces the reference build exactly.

## Version matrix

| Component     | Version                        | Why                                            |
|---------------|--------------------------------|------------------------------------------------|
| ONNX Runtime  | `v1.23.2`                      | Newest 1.23.x; opset 18 / IR v10 era proven by the model spikes; supports `--use_migraphx` |
| ROCm          | 7.2.4 (`rocm/dev-ubuntu-24.04:7.2.4`) | Matches the host driver stack (Arch, ROCm 7.2.4) |
| MIGraphX      | from the ROCm 7.2.4 apt repo (`migraphx-dev`) | Same release train as the ROCm userspace |
| Base OS       | Ubuntu 24.04 (glibc 2.39, GCC 13, libstdc++ 13) | AMD's reference environment |

Version pairing matters: `libonnxruntime.so` must resolve
`libmigraphx*.so` / ROCm libs from the **same** release train at runtime.
The container is both the build and the runtime environment, which pins the
pairing by construction.

## Prerequisites

- Docker (rootless-usable; the invoking user is in the `video` and `render`
  groups so `--device=/dev/kfd --device=/dev/dri` gives containers the GPU).
- ~30 GB free disk for image + source + build tree.
- Host kernel with amdgpu/KFD (any ROCm-7.x-compatible driver).

## 1. Build the builder/runtime image

All files live in `build/ort-rocm/` (gitignored).

```sh
docker build -t anofox-ort-migraphx:rocm7.2.4 build/ort-rocm/
```

`build/ort-rocm/Dockerfile`:

- `FROM rocm/dev-ubuntu-24.04:7.2.4`
- `apt-get install migraphx-dev miopen-hip-dev rocblas-dev hipblaslt-dev half git cmake ninja-build python3 python3-pip python3-dev`
  (miopen/rocblas/hipblaslt dev packages are required because
  `migraphx-config.cmake` pulls them in via `find_dependency()`)
- `pip3 install --break-system-packages numpy packaging wheel setuptools`

## 2. Fetch the source

```sh
git clone --depth 1 --branch v1.23.2 --recursive \
    https://github.com/microsoft/onnxruntime.git build/ort-rocm/work/onnxruntime
```

(Third-party deps beyond the submodules are fetched by CMake at configure
time, so the build container needs network access.)

## 3. Build

```sh
mkdir -p build/ort-rocm/ort-install
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/build/ort-rocm/work:/work" \
  -v "$PWD/build/ort-rocm/ort-install:/out" \
  -v "$PWD/build/ort-rocm/build-ort.sh:/build-ort.sh:ro" \
  anofox-ort-migraphx:rocm7.2.4 bash /build-ort.sh
```

`build-ort.sh` runs, inside the container:

```sh
cd /work/onnxruntime
./build.sh \
    --config Release \
    --build_shared_lib \
    --parallel \
    --use_migraphx \
    --migraphx_home /opt/rocm \
    --rocm_home /opt/rocm \
    --skip_tests \
    --skip_submodule_sync \
    --allow_running_as_root \
    --cmake_extra_defines \
        CMAKE_INSTALL_PREFIX=/out \
        onnxruntime_BUILD_UNIT_TESTS=OFF
cmake --install build/Linux/Release --prefix /out
```

Roughly 1–2 h on 32 cores. Result: `build/ort-rocm/ort-install/` with

```
include/onnxruntime/...            C/C++ headers (onnxruntime_c_api.h, _cxx_api.h)
lib/libonnxruntime.so.1.23.2       main runtime (+ .so / .so.1 symlinks)
lib/libonnxruntime_providers_migraphx.so   MIGraphX EP (dlopened at session setup)
lib/libonnxruntime_providers_shared.so     provider-bridge helper
```

## 4. Consume from the extension (inside the container)

The produced binaries link Ubuntu 24.04 glibc/libstdc++ and dlopen ROCm
libs, so the extension's rocm-flavor build **and tests run inside the same
image** with the GPU passed through:

```sh
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/repo" -w /repo \
  anofox-ort-migraphx:rocm7.2.4 bash
```

Inside:

```sh
cmake -B build/rocm -DTABFM_FLAVOR=rocm \
      -DTABFM_ORT_ROCM_DIR=/repo/build/ort-rocm/ort-install
cmake --build build/rocm -j
LD_LIBRARY_PATH=/repo/build/ort-rocm/ort-install/lib:/opt/rocm/lib <test runner>
```

## 5. Smoke test

`build/ort-rocm/smoke/` holds a C-API smoke test (`smoke.c`) plus a tiny
opset-18 model generator (`make_model.py`). Inside the GPU-enabled container:

```sh
cd /repo/build/ort-rocm/smoke
pip3 install --break-system-packages onnx
python3 make_model.py tiny.onnx
gcc smoke.c -I../ort-install/include -I../ort-install/include/onnxruntime \
    -L../ort-install/lib -lonnxruntime -o smoke
LD_LIBRARY_PATH=../ort-install/lib:/opt/rocm/lib ./smoke tiny.onnx
```

It verifies: (1) libonnxruntime loads, (2) `MIGraphXExecutionProvider` is
registered, (3) a session with the MIGraphX EP runs a real inference on the
GPU. `rocminfo | grep gfx` and `migraphx-driver` give the hardware verdict.

Or run everything at once from the repo root on the host:

```sh
bash build/ort-rocm/smoke/run-smoke.sh
```

### Reference results (2026-07-04, RX 9070 XT / gfx1201)

Full log: `build/ort-rocm/smoke/smoke-results.txt`.

**gfx1201 verdict: supported.** MIGraphX 2.15.0 compiles and runs on
gfx1201 (RDNA4); `migraphx-driver compile tiny.onnx --gpu` lowered the graph
to a rocMLIR kernel and completed:

```
Running [ MIGraphX Version: 2.15.0.20250912-17-221-gdd11a7555 ]: /opt/rocm/bin/migraphx-driver compile tiny.onnx --gpu
@5 = gpu::code_object[code_object=4832,symbol_name=mlir_dot_add,global=128,local=128,output_arg=3,](x,@2,@1,main:#output_0) -> float_type, {1, 3}, {3, 1}
[ MIGraphX Version: 2.15.0.20250912-17-221-gdd11a7555 ] Complete(0.970501s): /opt/rocm/bin/migraphx-driver compile tiny.onnx --gpu
```

The ORT C API smoke test passes end to end on the GPU:

```
[1] onnxruntime loaded, version 1.23.2 (API 23)
[2] available providers: MIGraphXExecutionProvider CPUExecutionProvider
[2] MIGraphXExecutionProvider registered: yes
[3] MIGraphX EP appended to session options
[3] session created over tiny.onnx with MIGraphX EP
[3] inference OK: y = [7.000000 8.000000 9.000000] (expect [7 8 9])
SMOKE PASS
```

## Known ORT 1.23.x pitfall: legacy MIGraphX options struct

Appending the EP via the **legacy struct API**
(`OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX` with a zeroed
`OrtMIGraphXProviderOptions`) fails session initialization with
`migraphx_save: ... Failure opening file: ""/....mxr`. Cause: the struct is
round-tripped through `MIGraphXExecutionProviderInfo::ToProviderOptions()`,
which stringifies the empty `std::filesystem::path model_cache_dir` as the
quoted literal `""` — non-empty, so the EP treats MXR caching as enabled
with a garbage path (onnxruntime/core/providers/migraphx/
migraphx_execution_provider_info.cc:103, `MakeStringWithClassicLocale`).

**Use the generic string-keyed API instead** (this is what the extension
must do):

```c
const char *keys[]   = {"device_id"};
const char *values[] = {"0"};
api->SessionOptionsAppendExecutionProvider(so, "MIGraphX", keys, values, 1);
```

(Alternative escape hatch if the legacy struct is ever unavoidable: export
`ORT_MIGRAPHX_MODEL_CACHE_PATH=<writable dir>`, which overrides the broken
path.)

## Host-native alternative (not the reference route)

Once `pacman -S migraphx` is possible on the host (Arch `extra/migraphx`,
7.2.3 at the time of writing), the same `build.sh` invocation can run
directly against `/opt/rocm` on the host, yielding a host-runnable install
tree without a container. Note the Arch package (7.2.3) trails the host ROCm
(7.2.4) slightly; the container route avoids that skew and is what was
validated.

## Caveats for linking from the DuckDB extension

- **C++ ABI**: the extension talks to ORT through the **C API surface**
  (`onnxruntime_c_api.h`; the C++ header is an inline wrapper over it), so
  there is no cross-library C++ ABI coupling with ORT — mixing a GCC-13
  (container) ORT with the extension build is safe as long as the extension
  itself is compiled inside the same container for the rocm flavor anyway.
- **libstdc++ floor**: the built `libonnxruntime.so.1.23.2` references
  versioned symbols up to `GLIBCXX_3.4.30` (GCC 12 era) — modest; both the
  container (libstdc++ 13) and the Arch host satisfy it. The ROCm/MIGraphX
  runtime deps are what make the container the supported runtime, not the
  C++ runtime.
- **EP options API**: append the MIGraphX EP with the generic string-keyed
  `SessionOptionsAppendExecutionProvider(so, "MIGraphX", ...)` — the legacy
  `OrtMIGraphXProviderOptions` struct path is broken in 1.23.x (see pitfall
  above).
- **Runtime deps**: `libonnxruntime_providers_migraphx.so` is dlopened when
  the EP is appended; it must sit next to `libonnxruntime.so` (same `lib/`,
  the main lib has `RUNPATH=$ORIGIN`) and links `libmigraphx_c.so.3` and
  `libamdhip64.so.7`, resolved via `LD_LIBRARY_PATH=/opt/rocm/lib` (or an
  rpath).
- **Distribution**: for packaging, ship `libonnxruntime.so*`,
  `libonnxruntime_providers_migraphx.so` and
  `libonnxruntime_providers_shared.so` beside the extension binary
  (`cmake/ort.cmake` records `TABFM_ORT_LIB_DIR` for exactly this).
