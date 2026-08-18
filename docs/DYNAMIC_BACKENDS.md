# One artifact, backends resolved at runtime

**Goal:** a single `anofox_tabfm.duckdb_extension` that runs on CPU everywhere
and lights up CUDA, ROCm or CoreML when the machine can, instead of three
mutually exclusive build flavors chosen at compile time.

**The property that matters:** every backend must produce the *same answers*.
A device switch is an optimisation, not a different model, so the test suite
below is the deliverable — not an afterthought to it.

Status: **phase 0 landed, phases 1–4 pending.** Each phase is independently
shippable and none silently changes results.

## Why this is possible (measured, not assumed)

| backend | what it actually is | evidence |
|---|---|---|
| **CPU** | ORT linked **statically** into the extension | published artifact: no `libonnxruntime.so` in `NEEDED`, zero exported `Ort*` symbols, 69 MB |
| **CUDA** | an ORT execution provider that **ORT itself `dlopen`s** | provider needs *no* ORT symbols (0 of 335 undefined are ORT-namespace); links only libc/libstdc++/CUDA; exports one entry point, `GetProvider` |
| **ROCm** | **not an ORT EP at all** — our own MIGraphX backend over the header-only C++ wrapper on `libmigraphx_c` | `cmake/ort.cmake` `find_library(migraphx_c … REQUIRED)`; ORT's own MIGraphX EP is a dead end (re-inlines initializers, 2 GB protobuf ceiling — `docs/GPU_AND_MEMORY_FINDINGS.md`) |
| **CoreML** | an ORT EP, **not** compiled into the core we ship | published macOS artifact contains zero CoreML internals; the lone `CoreMLExecutionProvider` string is our own probe literal |

So three of the four are already "a library loaded at runtime" in shape. The
blockers are specific and known:

* **CUDA** — ORT 1.23.2's provider is a *classic* provider: it exports
  `GetProvider` but not `CreateEpFactories`, so
  `RegisterExecutionProviderLibrary` (the only API taking an absolute path)
  rejects it. Its one load path is `Env::GetRuntimePath() + filename`, and
  `GetRuntimePath` is `dladdr`-based — with a static core that resolves next to
  the *extension*, not to a cache we control. ORT ≥1.28 can build CUDA as a
  **plugin EP** (`cuda-plugin-ep` in `get_build_info()`), which is registrable
  by path. See `docs/GPU_DISTRIBUTION.md` for the full measurement.
* **ROCm** — nothing external blocks it; the backend is simply compiled into
  the binary today instead of being loaded.
* **CoreML** — needs an ORT built with the EP enabled.

## Phases

### Phase 0 — runtime-aware device resolution ✅

`ResolveDevice` answered "does this *flavor* carry cuda?" from a compile-time
macro. It now answers "is this device's runtime actually usable *here*?", which
is the seam every later phase plugs into. No new capability; better errors, and
the compile-time flags become one input among several rather than the whole
answer.

### Phase 1 — ROCm as a loadable plugin ✅ complete (loader, plugin, and engine wiring)

Move `src/tabfm_migraphx.cpp` into its own shared object that links
`libmigraphx_c` normally, and `dlopen` it from the extension behind a small
versioned C ABI — the same shape ORT uses for providers (`GetProvider` +
a function table). Hand-rolling a symbol table for the header-only C++ wrapper
is the alternative and is worse: the wrapper reaches ~40 C entry points.

Notable: the plugin needs **only** `libmigraphx_c`, not an ORT-with-MIGraphX
build, so this also removes `TABFM_ORT_ROCM_DIR` from the equation.

Tested end to end on `bigfox` (RX 9070 XT / gfx1201). No cloud spend.

- `src/include/tabfm_plugin_abi.h` — the versioned C ABI (`abi_version` first,
  checked before anything else in the table is read).
- `src/tabfm_plugin_backend.cpp` — the loader. Tested against real shared
  libraries (`test/cpp/plugin_fixture/fake_plugin.cpp`, built twice — correct
  and deliberately-wrong ABI — so the ABI-mismatch refusal is exercised by a
  genuinely mismatched library, not a mock). One real bug caught this way: the
  loader read `api->abi_version` / `api->name()` *after* `dlclose`, a
  use-after-unload that segfaulted under ASan; fixed by capturing both before
  unloading.
- `src/tabfm_migraphx_plugin.cpp` — the plugin itself, built as
  `anofox_tabfm_migraphx_plugin` whenever `TABFM_MIGRAPHX_DIR` resolves a
  MIGraphX install (optional, silent skip otherwise — cpu/cuda CI never sees
  a MIGraphX header). Same inference logic as the old compile-time backend,
  ported to plain C++ so it has zero duckdb dependency.
- `test/cpp/test_tabfm_migraphx_plugin.cpp` — CPU (ORT) vs the dlopen'd plugin,
  on the **same real weights**, real gfx1201 hardware. Skips itself when the
  developer's model cache or the plugin `.so` is absent (CI has neither).
  Result below.

**A second real bug, caught only because this ran on actual hardware**:
`migraphx::target("gpu")` has `libmigraphx_c` `dlopen("libmigraphx_gpu.so")`
by bare name from inside itself — that's a runtime `dlopen` call, not a
`DT_NEEDED` entry, so it does **not** inherit `libmigraphx_c`'s own `RUNPATH`
and fails wherever that directory isn't already on `LD_LIBRARY_PATH` or in the
loader cache. Every ROCm layout puts it at `<libmigraphx_c's dir>/migraphx/lib/`,
so the plugin resolves `libmigraphx_c`'s own on-disk location via `dladdr` at
create time and preloads the GPU library from a path relative to *that* —
correct wherever the dependency was actually found, not just at the prefix the
plugin happened to be built against.

**Also learned**: this workload cannot run under the ASan-instrumented debug
build — HIP's GPU-mapped host allocations don't survive ASan's `munmap`
interceptor (`AddressSanitizer: CHECK failed ... unable to unmmap`), a known
class of ASan/GPU-driver conflict, unrelated to plugin correctness. Real GPU
runs need `DISABLE_SANITIZER=1 make debug` (or `make release`).

**Engine wiring, closing out the phase**: `tabfm_engine.cpp`'s ROCm dispatch
now calls `LoadPluginBackend` against `SET anofox_tabfm_ep_path` instead of
the compile-time `MakeMIGraphXBackend` — `src/tabfm_migraphx.cpp` is deleted,
and nothing in the main extension binary links `libmigraphx_c` on any flavor
anymore. A resolved `rocm` device with no `ep_path` configured now throws
rather than silently falling through to CPU (the tier-4 contract). The rocm
flavor's ORT build is still required, but now only so device discovery's
`OrtProviderAvailable` probe has an answer — not to run inference.

### Phase 2 — ORT ≥ 1.28 (in progress)

**The archive naming changed**, so this is not a version bump:

```
1.23   onnxruntime-linux-x64-gpu-1.23.2.tgz
1.28   onnxruntime-linux-x64-gpu_cuda12-1.28.0.tgz   (404 MB)
       onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz   (229 MB)
```

The GPU build is now split by CUDA major version, so `cmake/ort.cmake` gained
`TABFM_ORT_CUDA_MAJOR` (12 | 13) and keeps the old stem for < 1.28. The CPU
archive name is unchanged.


Prerequisite for phase 3 and independently worth doing. Note that the
"ORT 1.26 will not initialise" result from #21 was a *masked EP-load failure*,
fixed by #22 — so the upgrade should be less fraught than that report implied.
Re-runs the ScatterND pin checks against a newer runtime, which is worth
knowing regardless (onnxruntime#32083 is still open).

### Phase 3 — CUDA as a plugin EP + `tabfm_download_runtime('cuda')` ✅ registration + download implemented (GPU-run unverified)

**Premise confirmed on 1.28.** The provider's exported plugin-ABI symbols
(`CreateEpFactories` / `ReleaseEpFactory`):

| ORT | symbols | consequence |
|---|---|---|
| 1.23.2 (shipped) | **0** | classic ABI — `RegisterExecutionProviderLibrary` rejects it |
| 1.28.0 | **2** | registrable by absolute path, i.e. from our cache |

So phase 3 is unblocked by phase 2 and by nothing else.

**The registration sequence is not a drop-in for the old
`AppendExecutionProvider_CUDA(OrtCUDAProviderOptions)` call.** It's
`Env::RegisterExecutionProviderLibrary(name, path)` → `Env::GetEpDevices()` →
filter to the devices the plugin actually contributed → `AppendExecutionProvider_V2`.
There is no `device_id` shortcut analogous to the classic API's — device
selection goes through filtering the `ConstEpDevice` list, done in
`src/tabfm_ort_engine.cpp`'s `RegisterCudaProvider` by indexing into the
matching devices with `config.device_ordinal`. Compiles unconditionally (no
`TABFM_EP_CUDA` guard) — these are core ORT ≥ 1.22 APIs, present in the
CPU-flavor build too, so a cpu-flavor binary can drive CUDA once the provider
is registered at runtime.

**`tabfm_download_runtime('cuda')`** (`src/tabfm_weights.cpp`) fetches the
provider from the **onnxruntime-gpu PyPI wheel** (a ZIP), not the GitHub
release tarball (a `.tar.gz`) — the vendored miniz already gives a ZIP reader
for free, whereas a `.tar.gz` would need writing a TAR parser too. The
vendored miniz is built `MINIZ_NO_STDIO` (no file-path zip API), so extraction
reads the whole wheel into a heap buffer and uses
`mz_zip_reader_extract_to_heap`, not `mz_zip_reader_extract_to_file`. Verified
locally (no GPU needed for this part): downloads and extracts real, valid ELF
shared objects (`libonnxruntime_providers_cuda.so`, 280 MB;
`libonnxruntime_providers_shared.so`, 14 KB) into `SET anofox_tabfm_ep_path`,
idempotent on re-run (reports `cached`), and the unlisted-backend path
(`'rocm'`) errors naming the real fix rather than pretending to support it.

**Corrected size**: the provider is **~280 MB** (wheel, stripped), not
351 MB (that estimate was from the 1.23.2 GitHub archive) or the ~621 MB the
1.28.0 GitHub archive's unstripped `.so` measures at — three different
numbers for three different distribution shapes of the same code; the wheel
is the smallest and the one this implementation actually uses.

**RunPod-verified — registration works, kernel execution crashes.** Ran on a
real NVIDIA GPU (RunPod, CUDA 12.8.1 image, RTX 3070). Findings, each only
found by actually running it, not by reading the headers:

1. The plain PyPI `onnxruntime-gpu` package for 1.28.0 targets **CUDA 13**
   (`nvidia-cuda-nvrtc~=13.0`), not CUDA 12 — its provider failed to `dlopen`
   with `libcublasLt.so.13: cannot open shared object file` on a CUDA-12.4
   image. Fixed by switching `tabfm_download_runtime` to Microsoft's
   `onnxruntime-cuda-12` Azure Artifacts feed instead (linked from
   onnxruntime.ai's own install docs), which does publish a CUDA-12 build.
2. Even the CUDA-12 provider then failed with `undefined symbol:
   cudaLibraryGetKernel, version libcudart.so.12` on the CUDA-12.4.1 image —
   that symbol needs a newer CUDA 12 minor version than the image shipped.
   Resolved by using a CUDA-12.8.1 image instead; not a code issue, a
   deployment-environment minimum-CUDA-version constraint worth documenting
   for anyone using `tabfm_download_runtime('cuda')`.
3. With both of those fixed, `RegisterExecutionProviderLibrary` →
   `GetEpDevices` → `AppendExecutionProvider_V2` all **succeed** — the
   provider loads, the env reports a real CUDA `ConstEpDevice`, and the
   session is created without error. The crash is one step further in:
   **`Ort::Session::Run()` segfaults inside ORT's own CUDA kernel dispatch**
   (`libonnxruntime.so`, 10 frames deep, no symbols — a release build). A
   `gdb -batch -ex run -ex "thread apply all bt full"` capture puts the crash
   in `SessionImpl::Run` → internal ORT frames, not in this codebase's
   registration code, and not in the provider's own `.so`. Whether this is an
   ORT 1.28 plugin-EP-CUDA bug (a newer, less-exercised code path than the
   classic API) or an `ep_options` gap this codebase needs to fill (passed
   empty; `CUDAExecutionProviderInfo::FromProviderOptions` should default
   sensibly on an empty map per ORT's own source, but that's unconfirmed) is
   not yet root-caused.

So: **registration is real and works**; **inference through it does not,
yet**. Phase 3's download/registration code is correct and tested end to end
up to the point of running a model.

**Control test run, isolating the cause.** Same GPU family (RunPod RTX 3070),
same CUDA 12.8.1 environment, same `onnxruntime 1.28.0` build (the CUDA-12
Azure feed wheel) — but through Python's classic `providers=["CUDAExecutionProvider"]`
registration (`tools/gpu_test/ort_ep_check.py resources/graph_tabicl_classification.onnx
--provider cuda`) instead of the plugin-EP sequence. Result: **`failures=0
mismatches=0`, exit code 0.** The graph runs cleanly.

That settles it: CUDA hardware, driver, and this exact ORT 1.28.0 build all
work fine on this GPU. The crash is specific to the **plugin-EP registration
path** (`RegisterExecutionProviderLibrary` → `GetEpDevices` →
`AppendExecutionProvider_V2`) — not something wrong in `tabfm_ort_engine.cpp`'s
`RegisterCudaProvider`, not the fixture graph, not this environment.

**[onnxruntime#28329](https://github.com/microsoft/onnxruntime/issues/28329)
was our first guess, corrected on closer reading.** Its symptom description
matches beat for beat ("session creation succeeds but the first `session.Run()`
crashes... built-in CUDA EP works correctly with the same model and inputs")
but its claimed root cause — a legacy `OrtMemTypeCPU(-1)` value misread in
`cuda_ep_factory.cc` — does not exist in v1.28.0's actual source. That file's
`CreateMemoryInfoForDevices` already uses the newer `CreateMemoryInfo_V2` API
with explicit `OrtDeviceMemoryType_DEFAULT`/`HOST_ACCESSIBLE` enums, no raw
`-1` anywhere. ORT's plugin-EP memory-info handling was reworked between when
that issue was filed (v1.25.1) and 1.28.0, so #28329 is very likely a
different bug that happens to look the same from the outside.

**Root-caused by direct A/B testing on real hardware — the static hypothesis
below turned out to be wrong, kept for the record.** Built ORT v1.28.0 from
source (RelWithDebInfo, debug symbols, `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`,
single-arch `86-real` for build speed) on a RunPod RTX A5000, then ran a
minimal standalone repro (register → `GetEpDevices` → `AppendExecutionProvider_V2`
→ `Run()`, same fixture graph) against every combination of core + CUDA
provider:

| core `libonnxruntime.so` | CUDA provider `.so` | result |
|---|---|---|
| built from source here | built from source here | **runs clean**, logits byte-identical to CPU |
| built from source here | official CUDA-12 Azure-feed wheel | **runs clean**, logits byte-identical to CPU |
| **official GitHub-release prebuilt archive** (`onnxruntime-linux-x64-1.28.0.tgz`) | official CUDA-12 Azure-feed wheel | **SIGSEGV**, same crash signature as the original |

Same v1.28.0 source, same GPU, same driver, same CUDA toolkit — the *only*
variable that flips the outcome is which build of the **core** library is
loaded. That rules out the shared-allocator hypothesis below outright: it's
in `environment.cc`, compiled identically into both cores from the same
source tag, so if it were the real mechanism both would crash equally. It
doesn't rule out an ABI/build-flag mismatch specific to how Microsoft's
official prebuilt archive was compiled (different compiler version, different
optimization/ABI flags, or the archive silently carrying a patch not in the
public `v1.28.0` git tag) — undeterminable further without Microsoft's own
build logs, and not necessary to determine further given what this means
practically (next paragraph).

**This is good news for anofox-tabfm specifically.** The community-extension
*release* build already compiles ORT from source via the `vcpkg_ports/onnxruntime`
overlay port (`TABFM_ORT_VCPKG=1`, this repo's release default) — the same
build shape that was proven clean above, not the prebuilt-archive shape that
crashes. Only the local prebuilt-archive **dev** build (`make debug`, fast
iteration) is suspected to hit this. Verification of the actual release
build (vcpkg-sourced ORT, statically linked, cpu flavor) against the CUDA
plugin-EP path is in progress as of this writing — see the phase-3 status
line at the top of this section once that lands.

<details>
<summary>Original static-trace hypothesis (superseded by the A/B result above — kept for the record, not the diagnosis)</summary>

A full source read of
`onnxruntime/core/providers/cuda/cuda_provider_factory.cc` and
`onnxruntime/core/session/environment.cc` (v1.28.0) turned up something
specific and structural:

- CUDA is not a *true* plugin EP in 1.28.0 — `CudaEpFactory::CreateEpImpl`
  unconditionally returns `ORT_INVALID_ARGUMENT`. It is a "provider bridge":
  `RegisterExecutionProviderLibrary` loads the CUDA `.so`, sees it exports the
  classic `GetProvider` symbol, and wraps it as an `EpLibraryProviderBridge`.
  A session created via `AppendExecutionProvider_V2` therefore ends up
  constructing the exact same in-tree `CUDAExecutionProvider` C++ class the
  classic path uses — confirmed the `ep::adapter::*` bridge headers (where a
  real, already-fixed bug lived — PR #29658, a null-allocator PrePack
  crash) are provably unreachable for CUDA, since `CreateEpImpl` never runs.
- But `Environment::RegisterExecutionProviderLibrary` (environment.cc:582-598)
  does something the classic path never does: for every `OrtEpDevice` the
  library reports, it unconditionally pre-creates and registers a
  process-wide **shared allocator** (`CreateSharedAllocatorImpl`, a plain
  non-arena `CudaOrtAllocator` wrapping bare `CUDAAllocator`/`CUDAPinnedAllocator`)
  keyed by that device's `OrtMemoryInfo`, *before any session exists*. The
  session's own `CUDAExecutionProvider` instance then separately constructs
  **its own** arena allocator and CUDA stream, same as always. Two allocators
  now exist for the same device/memory-info: one process-wide and streamless
  (created at registration time), one session-owned with its own stream
  (created at session-creation time) — and nothing in the traced code
  guarantees which one backs a given tensor. If session/graph memory planning
  ever resolves an allocator by `OrtMemoryInfo` and prefers (or accidentally
  picks up) the registration-time shared one instead of the EP's own, that
  tensor's lifetime and stream-ordering have no relationship to the stream the
  `CUDAExecutionProvider` actually launches kernels on — a textbook
  segfault/use-after-free shape, and one that is structurally *only possible*
  via the `RegisterExecutionProviderLibrary` + `AppendExecutionProvider_V2`
  path (the classic `AppendExecutionProvider_CUDA` API never populates
  `Environment::shared_allocators_`).
- `CreateSyncStreamForDeviceImpl` (cuda_provider_factory.cc:876-883) has an
  explicit comment from ORT's own engineers: *"we're using the 'real' CUDA
  IExecutionProvider implementation for the EP... For use within an inference
  session in a completely plugin EP we'd need the session's CPU allocator to
  be available"* — i.e. this path is known-unwired for in-session CUDA use and
  passes a **null allocator** if anything does reach it. Lower-probability
  than the shared-allocator theory but a second concrete candidate, and cheap
  to rule out with a breakpoint.

This read like a strong, falsifiable hypothesis — and turned out to be wrong,
or at least not the differentiator, per the A/B result above. Left here
because the reasoning about the provider-bridge architecture and the
shared-allocator mechanism is still accurate as *description of the code*;
it just isn't *why one build crashes and the other doesn't*, since both
builds run the identical version of this code.

</details>

Practical upshot: **`SET anofox_tabfm_device = 'cuda'` cannot work while ORT
is statically linked.** Verified on real NVIDIA hardware (RunPod, RTX A5000,
driver 580.159.04, CUDA 12.8.1). The release/vcpkg-sourced build does not
sidestep the problem; it fails in two stages, and the second one is fatal to
the current design. The fix is to ship ORT as a shared `libonnxruntime.so`
for GPU-capable builds, replicating the layout of Microsoft's own wheel
(which is confirmed working with the identical provider `.so` on the same
box).

Three issues were found running the real release build (ORT 1.28.0 built
from source via the vcpkg overlay port, statically linked into the
extension/duckdb binary — the shape the community-extension ships):

1. **Fixed**: `RegisterExecutionProviderLibrary` failed with `undefined
   symbol: Provider_GetHost` when loading `libonnxruntime_providers_cuda.so`.
   That classic-ABI provider `.so` imports `Provider_GetHost`, exported by
   the small sibling library `libonnxruntime_providers_shared.so`. When ORT
   itself is a shared library, its own provider-bridge code dlopens the
   shared-providers library (`RTLD_GLOBAL`) as a side effect of loading any
   provider, which is what resolves that symbol against the caller's global
   scope. Statically linking ORT into the host binary removes that side
   effect, so the symbol stays unresolved. Fixed the same way the MIGraphX
   plugin already had to be fixed for the identical problem with
   `libmigraphx_gpu.so`/`migraphx_target_create`: `RegisterCudaProvider` now
   does its own one-time `dlopen(ep_path + "/libonnxruntime_providers_shared.so",
   RTLD_NOW | RTLD_GLOBAL)` before calling `RegisterExecutionProviderLibrary`
   (`PreloadCudaProvidersSharedLibrary` in `src/tabfm_ort_engine.cpp`).

2. **Open, root-caused, NOT specific to this extension's build shape**: with
   (1) fixed, registration proceeds further but then SIGSEGVs. Initial gdb
   backtraces (release build, `dl_open_worker`/`_dl_open` frames) suggested a
   static-linking/glibc-loader interaction, and a first `codex exec` second
   opinion pushed back on the leading static-TLS-exhaustion theory for that
   shape (see the ruled-out details below). Isolating further with a minimal
   repro proved decisive:

   ```c
   // 20 lines, no ORT headers, no anofox-tabfm code at all:
   dlopen("libonnxruntime_providers_shared.so", RTLD_NOW | RTLD_GLOBAL);
   dlopen("libonnxruntime_providers_cuda.so", RTLD_NOW | RTLD_LOCAL); // <-- crashes here
   ```

   This tiny, ~15KB dynamically-linked binary (no static linking anywhere,
   no anofox-tabfm/DuckDB code in the process at all) reproduces the exact
   same SIGSEGV — which rules out static linking, TLS surplus, and this
   extension's build shape entirely as the cause. Preloading a full ORT core
   first (to test whether an already-loaded ORT core changes anything) made
   no difference — tried both a mismatched ORT 1.20 CPU-wheel
   `libonnxruntime.so` and, as a cleaner ABI-matched control (per a second
   `codex exec` review round), the *exact* ORT 1.28.0 CPU-wheel core +
   matching `libonnxruntime_providers_shared.so` from the same
   `onnxruntime==1.28.0` PyPI wheel — still crashes identically on the same
   `dlopen()` of `libonnxruntime_providers_cuda.so`.

   gdb pinpointed the actual fault:
   ```
   Program received signal SIGSEGV.
   0x... in ?? () from /workspace/ep/libonnxruntime_providers_cuda.so
   #0  0x... in ?? () from .../libonnxruntime_providers_cuda.so
   #1  call_init (...) at ./elf/dl-init.c:70
   #2  call_init (...) at ./elf/dl-init.c:33
   rdi = 0x0
   => mov    (%rdi),%rax        # load vtable from a NULL `this`
      call   *0x1948(%rax)      # virtual call through it -> SIGSEGV
   ```

   Symbolising that frame identified it exactly. The faulting instruction is
   at offset `0x1c0118`, inside `.init_array` entry #2 (function at
   `0x1bff10`), and the call immediately before it resolves through the PLT
   to `Provider_GetHost@Base`:

   ```asm
   call   Provider_GetHost@plt   ; returns NULL
   mov    %rax,%rdi
   mov    %rdi,(%rax_global)     ; cache it  -> g_host
   mov    (%rdi),%rax            ; load vtable from NULL  <-- SIGSEGV
   call   *0x1948(%rax)          ; virtual call -> GetProviderHostCPU()
   ```

   That is a 1:1 match for this ORT source, compiled into every
   provider-bridge library:

   ```cpp
   // onnxruntime/core/providers/shared_library/provider_bridge_provider.cc:91-92
   ProviderHost* g_host = Provider_GetHost();
   ProviderHostCPU& g_host_cpu = g_host->GetProviderHostCPU();  // no null check
   ```

   `g_host` is only non-null once the ORT **core** has called
   `Provider_SetHost`, which happens inside
   `ProviderSharedLibrary::Initialize()` — and that resolves
   `libonnxruntime_providers_shared.so` relative to
   `Env::Default().GetRuntimePath()`, i.e. *the directory of the binary that
   contains ORT*. So the crash is a **load-ordering / library-layout
   problem**, not an unconditional defect in the provider library.

   Two things confirm that reading:

   - **Microsoft's own wheel works on the same hardware with the same
     621MB `.so`.** `pip install onnxruntime-gpu==1.28.0` (cuda-12 feed) runs
     CUDA inference fine, *and* `ort.register_execution_provider_library()`
     — the same plugin-EP entry point this extension uses — registers the
     CUDA EP without crashing. In the wheel, `libonnxruntime.so.1.28.0` and
     `libonnxruntime_providers_shared.so` sit in the same directory as the
     provider, so `GetRuntimePath()` finds it and the host gets set.
   - **Placing `libonnxruntime_providers_shared.so` next to our executable
     makes the SIGSEGV disappear**, and execution then reaches `Run()`.

   ORT's own `LoadPluginOrProviderBridge` (`core/session/utils.cc`) makes
   this a hard crash rather than a diagnosable error, because it discards the
   status and then `dlopen`s the library anyway:

   ```cpp
   bool is_provider_bridge = provider_library->Load() == Status::OK();  // error dropped
   ...
   ORT_RETURN_IF_ERROR(ep_library_plugin->Load());  // dlopen -> ctor deref of NULL g_host
   ```

   (ORT's Python entry point sidesteps this by calling
   `InitProvidersSharedLibrary()` explicitly up front; the C API path does
   not.) That is a legitimate upstream robustness bug worth reporting, but it
   only converts this crash into a clear error message — it is **not** what
   blocks this extension.

3. **The actual blocker: statically-linked ORT is incompatible with the
   prebuilt provider libraries.** With the load order fixed (issue 2), the
   CUDA path gets all the way into `Run()` and then aborts:

   ```
   free(): invalid pointer
   #9  onnxruntime::ConstantOfShape::~ConstantOfShape()   <- statically-linked core
   #10 ?? from libonnxruntime_providers_cuda.so           <- the provider's own copy
   #13 onnxruntime::ExecuteKernel(...)
   ```

   `ConstantOfShape` (and many peers) are compiled into **both** the ORT core
   and the provider `.so`. When the core is statically linked into the host
   executable, the executable's symbols take precedence in the global
   resolution scope, so the provider's internal calls bind to *our* copies —
   an ODR/symbol-interposition mismatch that corrupts the heap. This is
   inherent to static linking, not an ORT bug, and cannot be fixed upstream.

   **Conclusion**: GPU backends require ORT to be shipped as a **shared
   `libonnxruntime.so`**, with `libonnxruntime_providers_shared.so` and the
   downloaded provider libraries living in that same directory — i.e.
   replicating the layout of Microsoft's own wheel, which is verified working
   on real hardware. The current `anofox_tabfm_ep_path` design (provider libs
   in an arbitrary user-chosen directory, ORT statically linked) cannot work
   and needs to be reworked accordingly.

### Phase 3, resolved: CUDA is a plugin, exactly like ROCm

The three constraints above (shared core, providers beside it, core and
providers from one distribution) are all satisfied by *not* putting GPU
inference in the extension binary at all — which is the shape phase 1 already
gave ROCm. CUDA now works the same way:

| | ROCm | CUDA |
|---|---|---|
| plugin | `src/tabfm_migraphx_plugin.cpp` | `src/tabfm_cuda_plugin.cpp` |
| links | its own `libmigraphx_c` | its own **shared ORT-GPU** `libonnxruntime.so` |
| dispatched by | `TryMIGraphXBackend` | `TryCudaBackend` |
| reached via | `tabfm_plugin_abi.h`, `dlopen` | same |

The extension binary keeps its static CPU ORT and its community-extension
eligibility; nothing GPU-shaped is linked into it. Because the ORT-GPU
distribution has the CUDA EP compiled in, the plugin reaches CUDA through the
ordinary `AppendExecutionProvider_CUDA` call — there is no runtime provider
registration anywhere in the design any more, and
`RegisterExecutionProviderLibrary` is gone from `tabfm_ort_engine.cpp`
(the CUDA branch there now throws a message naming the plugin, since reaching
it means dispatch failed rather than that a fallback is wanted).

`tabfm_download_runtime('cuda')` fetches the ORT core alongside the two
provider libraries — from the same wheel, so they match by construction — and
lands the core under its SONAME (`libonnxruntime.so.1`) so the plugin's
`DT_NEEDED` resolves. The plugin is linked with `INSTALL_RPATH=$ORIGIN`, so
it finds all of it wherever `anofox_tabfm_ep_path` points.

**Design validated on real hardware before it was built** (RTX A5000, driver
580.159.04, CUDA 12.8.1, ORT 1.28.0): a C++ host linking the ORT-GPU archive
as a shared library ran the committed fixture graph on CUDA and agreed with
CPU to **7.9e-07** — the spike is `tools/gpu_test/` territory and the
equivalence test that locks it in is `test/cpp/test_tabfm_cuda_plugin.cpp`
(skips itself without a GPU and a real model cache).

   <details>
   <summary>Ruled-out theory: static linking / glibc TLS surplus (kept for the record)</summary>

   Before the minimal repro above, the leading theory was that ORT being
   **statically linked** into a large host binary (`test/unittest` release
   build: ~97MB text segment, only `libstdc++`/`libm`/`libgcc_s`/`libc`
   linked dynamically) that then `dlopen()`s a huge (621MB) provider library
   at runtime, long after startup, was blowing past glibc's static-TLS
   surplus reservation. A first `codex exec` second opinion pushed back:
   the textbook symptom of TLS surplus exhaustion is a clean, catchable
   `cannot allocate memory in static TLS block` error from `dlopen`, not a
   segfault inside the linker's own code —
   `GLIBC_TUNABLES=glibc.rtld.optional_static_tls=4194304` (a much larger
   surplus) made no difference either, weakening the theory further before
   it was conclusively ruled out by the tiny dynamically-linked repro above
   producing the identical crash. `LD_DEBUG=statistics`,
   `/proc/sys/vm/max_map_count` (65530), and `ulimit` were all unremarkable
   and added nothing; no kernel dmesg/journal access in the RunPod
   container for a kernel-level fault address.

   </details>

**vcpkg overlay port bumped to 1.28.0 too** — this was a real, separate
blocker: the release/community-extension build compiles ORT from
`vcpkg_ports/onnxruntime`, still pinned to 1.23.2, so `RegisterExecutionProviderLibrary`
et al wouldn't have existed there even with the code above merged. Two of the
four 1.23.2-era patches were already fixed upstream and dropped; the other two
lost one hunk each to the same effect and kept the rest, rebased. The actual
blocker was one level down: ORT 1.28.0 needs onnx ≥ 1.22.0
(`TensorProto_DataType_INT2`/`UINT2`, `OpSchema::SetNodeDeterminism`), but
vcpkg's registry onnx port tops out at 1.19.0 — fixed with a second overlay
port, `vcpkg_ports/onnx`, pinning 1.22.0. Verified for real: `make release`
builds both from source, the resulting extension is confirmed statically
linked (no `libonnxruntime.so` dependency), and the full suite passes against
it (601 sqllogictest + 71,989 Catch2 assertions).

### Phase 4 — CoreML

**Not a missing capability — a build-source choice.** The official ORT 1.28
`onnxruntime-osx-arm64` dylib *does* carry CoreML (8 CoreML internals and the
`OrtSessionOptionsAppendExecutionProvider_CoreML` entry point). What we publish
for macOS is the **vcpkg static** ORT, which does not. So phase 4 is: either
build the vcpkg port with CoreML enabled, or take macOS from the prebuilt
archive — not "rebuild ORT ourselves".

Still lowest priority: no Apple hardware in the loop to verify against, so the
equivalence suite could not confirm the answers match.

## The equivalence suite

The point of the whole exercise is that a device switch does not change the
answer. That is testable *now*, before any of it exists, and every phase adds
rows to the same matrix rather than its own bespoke check.

**Tier 1 — graph level, any provider, no weights** (`tools/gpu_test/ort_ep_check.py`)
Runs a committed weight-free graph on a provider with synthesized
initializers and compares output digests across providers. Already used to
verify the ScatterND pins on CUDA. Runs on CPU in CI; GPU providers when a
device is present.

**Tier 2 — real checkpoints, CPU reference**
Inject the actual cached weights and compare each backend against the CPU
result. `logits` must agree; the tolerance is per-backend and stated, not
assumed:

| comparison | tolerance | measured | rationale |
|---|---|---|---|
| CPU vs CPU, same ORT (pin/rewrite changes) | **bit-identical** | **0** across 5 graphs × 3 shapes, real checkpoints | same kernels, same order |
| CPU, ORT 1.24.1 vs 1.28.0 | relative 1e-4 | **3.26e-05**, argmax agreement **1.0** (real TabICL) | versions change kernel and fusion choices |
| CPU vs CUDA fp32 | relative ~1e-4 | not yet measured | different kernels and reduction order |
| CPU vs ROCm (MIGraphX plugin) bf16 | class agreement + loose abs bound | **max abs diff 0.52**, argmax agreement **5/5** (real `google/tabfm`, gfx1201) | bf16 has ~3 significant digits; the classification decision is the contract, not the raw logit |
| bf16/fp16 GPU paths (other backends) | class agreement + ~1e-2 | not yet measured | precision is the point of the mode |

The second row is the one worth internalising: an **ORT upgrade is not
bit-identical** on a real model, only within tolerance. The fixture golden test
reports delta 0 because the fixture is small; the 27 M-parameter checkpoint
drifts at 3e-5. Predictions are unchanged, which is the property that matters —
but "the suite is green" and "the numbers are identical" are different claims,
and only the first one holds across versions.

**Tier 3 — SQL surface, end to end**
The same query on each device must produce the same `yhat` for every row, and
`proba` within tolerance. Runs against the fixture in CI, and against real
models where the hardware exists.

**Tier 4 — the negative space**
A device that is *requested but unavailable* must produce an actionable error
and never a silent fallback to CPU. This is the failure mode the harness itself
had (three runs silently measured CPU, see `tools/gpu_test/README.md`), so the
suite asserts the refusal, not just the success.

CI runs tiers 1–4 on CPU. GPU tiers are opt-in — they need hardware CI does not
have — and are run by hand via `tools/gpu_test/`, which is why that harness
refuses to report CPU results as GPU.

## Risks

* **Silent divergence** is the one that matters. Mitigated by making the
  equivalence matrix the deliverable and by tier 4's refusal test.
* **Cache size**: ~280 MB (CUDA provider, measured) and ~6.6 GB per `.mxr` shape bucket
  (ROCm, ~20 min first compile). Both belong in the weights cache with the
  messaging that already exists there.
* **Binary size and CI matrix** grow by one plugin per GPU backend.
* **ABI drift** between the extension and our own ROCm plugin — versioned
  explicitly, refused on mismatch.
