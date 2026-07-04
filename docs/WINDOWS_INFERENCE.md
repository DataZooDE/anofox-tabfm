# Windows: known ORT inference crash (tracked)

**Status (2026-07-04):** the extension **builds and loads** on Windows
(`windows_amd64`, `x64-windows-static-release`, ONNX Runtime 1.23.2 prebuilt
`win-x64`). The **first native ORT forward pass SIGSEGVs**. Linux (amd64/arm64)
and macOS (amd64/arm64) run the full engine path — build **and** inference — in
CI. The Windows forward-pass sqllogictests are gated with `require notwindows`
until this is fixed on a Windows host.

Gated tests: `tabfm_classify.test`, `tabfm_regress.test`, `tabfm_cobatch.test`.
Everything else runs on Windows CI: build, load, `settings`, `tabfm_devices`,
`telemetry`, `tabfm_license`, `tabfm_lifecycle`, `tabfm_weights` (7/9 suites).

## What we know

- This is **not a regression** — it is the *first time the inference path ever
  ran on Windows*. Windows failed to **compile** until the `ORTCHAR_T` path fix
  (`src/tabfm_ort_engine.cpp`, `CreateSessionFromPath`, `#ifdef _WIN32` wide
  path) and then failed to **link** until `iphlpapi`/`ws2_32` were added
  (posthog-telemetry `GetAdaptersInfo`). Only after both did the forward pass
  execute — and crash.
- **Not telemetry.** The submodule is at the teardown-SIGSEGV fix
  (`0433745 "deterministically stop telemetry worker before OpenSSL
  teardown"`), and tests run with `DATAZOO_DISABLE_TELEMETRY=1`. The crash is in
  `settings.test`'s (now-removed) fixture predict — the first `unittest` case —
  i.e. the engine, not the telemetry destructor.
- **Reported as:** `test_sqllogictest.cpp(212): FAILED ... SIGSEGV`,
  `assertions: 33 | 32 passed | 1 failed`. Catch2's signal handler gives no
  native stack.
- The C++ inference code (`Run`, `CreateSession*`, `PrepareSessionOptions`,
  `AddExternalInitializers`) is portable and correct on Linux/macOS. The crash
  is inside the native ORT call or at the extension⇄ORT boundary.

## Leading hypothesis

**CRT mismatch.** The extension links the **static** CRT (`/MT`, DuckDB's
`x64-windows-static` triplet); Microsoft's prebuilt `onnxruntime.dll` links the
**dynamic** CRT (`/MD`). The ORT **C** API is designed to be CRT-agnostic, but
`AddExternalInitializers` (which retains references to extension-owned buffers
for the session lifetime) and the injected-initializer path are prime suspects
for a cross-heap access. This does not manifest on Linux/macOS (single CRT).

## To debug on a Windows host

1. Build `make release` on Windows, then run
   `.\build\release\test\unittest.exe "test/sql/tabfm_classify.test"` under a
   debugger (WinDbg / VS) or with `set ORT_LOG_SEVERITY_LEVEL=0`, and capture
   the crashing frame / stack.
2. If it is the CRT mismatch: either build ORT from source with the static CRT,
   or switch the injection path to
   `AddExternalInitializersFromFilesInMemory` / an ORT-owned copy so no
   extension-CRT buffer outlives the boundary.
3. Re-enable the three suites by removing their `require notwindows` lines and
   confirm green on Windows CI.
