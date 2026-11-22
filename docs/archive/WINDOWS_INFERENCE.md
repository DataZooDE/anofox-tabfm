# Windows inference — RESOLVED (2026-07-05)

Windows CPU inference works and is validated in CI: the full engine path
(preprocess → initializer injection → ORT forward → decode) runs on
`windows_amd64` for all three forward-pass suites (`tabfm_classify`,
`tabfm_regress`, `tabfm_cobatch`), same as Linux and macOS. No tests are gated
by platform anymore.

## Root cause

The extension links `onnxruntime.dll` (dynamic). Windows resolves a DLL from the
host executable's directory before `System32` — but **Windows ML ships an old
`C:\Windows\System32\onnxruntime.dll` (v1.17.1)**, and our bundled build is
**1.23.2**. Because our DLL was not staged next to `unittest.exe`, the loader
found the stale System32 copy. `OrtGetApiBase()->GetApi(ORT_API_VERSION=23)` then
returns **null** on the 1.17 runtime ("only API versions [1, 17] are supported"),
and the first ORT call dereferences the null API table → `EXCEPTION_ACCESS_
VIOLATION READ at 0x50`, all frames in `unittest.exe` (the header-only C++ API
wrapper), no `onnxruntime.dll` frame. The extension still *loaded* because the
ancient `OrtGetApiBase` export exists in every ORT version.

This was a pure runtime DLL-resolution problem — never a code/CRT bug. It only
surfaced on Windows because that is the only platform that ships a competing
system `onnxruntime.dll`.

## How it was found

CI-only instrumentation (no local Windows host): heap-free breadcrumbs across the
predict stages plus a vectored exception handler printing the faulting module,
then a crash-safe probe printing the loaded DLL path + version + a
`GetApi(ORT_API_VERSION)` null-check. That probe returned, verbatim:

```
onnxruntime.dll: C:\Windows\SYSTEM32\onnxruntime.dll
runtime version: 1.17.1
GetApi=NULL (version mismatch!)
```

All instrumentation has been removed.

## The fix

1. **Stage the DLL** (`CMakeLists.txt`, `WIN32` block): a `POST_BUILD` copies the
   prebuilt `onnxruntime.dll` next to the test + shell binaries
   (`${CMAKE_BINARY_DIR}/test` and `/`) so the exe-directory search — which
   outranks `System32` — resolves **ours**. Keyed off `TABFM_ORT_LIB_DIR`, set by
   `cmake/ort.cmake` on the prebuilt-archive path (the CI cpu-flavor default).
2. **Defensive guard** (`tabfm_ort_engine.cpp`, `EnsureUsableOrtApi()`): runs
   before the first ORT call; if `GetApi(ORT_API_VERSION)` is null it throws an
   actionable `IOException` (naming the version mismatch and the System32 cause)
   instead of segfaulting. This protects any environment that still shadows our
   DLL.

## Distribution note (open, separate from CI)

The staging above fixes the build tree and any host that runs with our
`onnxruntime.dll` co-located. For a **distributed** loadable extension consumed by
a user's own `duckdb.exe`, the same shadowing can occur if our `onnxruntime.dll`
is not on that process's DLL search path. Options for the packaging story: ship
`onnxruntime.dll` alongside the extension and load it with an altered search path,
or statically link ORT. Until then, `EnsureUsableOrtApi()` guarantees a clear
error rather than a crash.

## GPU on Windows

`tabfm_devices()` used to return the cpu row alone on Windows even with a working
NVIDIA card, because `ProbeCudaDevices` was compiled only on non-Windows. The
CUDA execution provider itself was never guarded — with no device discovered,
`ResolveDevice` simply refused `'cuda'` and there was nothing to run on. NVML is
the same C ABI on both platforms (`nvml.dll`, in System32, from any driver
install), so only the loader differed.

Building and running the `cuda` flavor here, on a self-built extension (there is
no published GPU build — see below):

```
tabfm_devices()
  cuda:0  CUDAExecutionProvider  NVIDIA GeForce RTX 3060  sm_86  12 GB  usable = true
```

Two runtime requirements, neither of which the build enforces:

* **ORT's CUDA provider DLLs must sit next to the host executable**, same as
  `onnxruntime.dll` (§ above). `cmake/ort.cmake` stages
  `onnxruntime_providers_cuda.dll` and `onnxruntime_providers_shared.dll` into
  the build tree alongside it.
* **The CUDA 12 runtime must be on `PATH`.** ORT 1.23.2's GPU build wants
  `cudart64_12` / `cublas64_12` / `cublasLt64_12` / `cufft64_11` / `curand64_10`
  / `cudnn64_9`. A CUDA Toolkit install provides them; failing that, the
  `nvidia-*-cu12` pip wheels ship the same DLLs and their `bin` directories can
  be prepended to `PATH`. Without them the EP fails to load, and — because the
  env is created after the provider is appended — that surfaces as
  `Attempt to use DefaultLogger but none has been registered` rather than
  anything mentioning CUDA (#22 fixes the message).

Measured end to end on this box against the Linux baseline, real `tabicl-v2`
checkpoint, 60-row fixture: **0/60 label mismatches** between Windows CUDA and
the Linux CPU baseline (max softmax delta 3.0e-3, ordinary fp32 CPU/GPU
divergence). Note that `tabicl-v2` additionally needs the ScatterND graph
workaround (#23) to run on CUDA at all, on either platform.
