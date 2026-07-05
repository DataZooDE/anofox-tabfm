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
