# Upstream ONNX Runtime issue drafts (P8) — ready to file, not yet filed

Status per S5 of `docs/GPU_HARDENING_PLAN.md`: the crash below **does not
reproduce on the 1.29.0 wheel** (exit 0 on an A40), so both drafts are scoped
as "affects ≤ 1.28.x, apparently resolved by 1.29" — worth filing only if
maintainers still take 1.28.x fixes, or as a request to confirm the fix was
intentional rather than incidental. Filing is an outward-facing action;
these stay drafts until explicitly approved.

---

## Draft 1 — `RegisterExecutionProviderLibrary` turns a path misconfiguration into a SIGSEGV in a global constructor

**Affects:** 1.28.0 (verified on RTX A5000 / A40, CUDA 12.8, Ubuntu 22.04).
**Apparently resolved:** the repro below exits cleanly against the 1.29.0
`onnxruntime-gpu` wheel's libraries.

Two connected defects:

1. `LoadPluginOrProviderBridge` (`core/session/utils.cc`) discards the status
   of the provider-bridge attempt and proceeds to `dlopen` the library anyway:

   ```cpp
   bool is_provider_bridge = provider_library->Load() == Status::OK();  // error dropped
   ...
   ORT_RETURN_IF_ERROR(ep_library_plugin->Load());  // dlopen -> crash below
   ```

2. `provider_bridge_provider.cc:91` runs at that `dlopen`'s `call_init`:

   ```cpp
   ProviderHost* g_host = Provider_GetHost();
   ProviderHostCPU& g_host_cpu = g_host->GetProviderHostCPU();  // no null check
   ```

   `g_host` is only set once the core has called `Provider_SetHost`, which
   `ProviderSharedLibrary::Initialize()` does after resolving
   `libonnxruntime_providers_shared.so` relative to `Env::GetRuntimePath()`.
   When that resolution fails (the provider libraries live in a directory that
   is not the core's own), step 1 swallows the error and step 2 dereferences
   null inside the loader — the user sees a SIGSEGV in `dl_open_worker` with
   no actionable message.

**Repro** (15 lines, no ORT headers; crashes on 1.28.0, clean on 1.29.0):

```c
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
    /* argv[1] = libonnxruntime_providers_shared.so, argv[2] =
     * libonnxruntime_providers_cuda.so, both in a directory that is NOT the
     * core's GetRuntimePath. */
    if (!dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL)) { printf("%s\n", dlerror()); return 2; }
    void *cuda = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    printf(cuda ? "loaded\n" : "clean error: %s\n", dlerror());
    return 0;
}
```

**Ask:** propagate the swallowed status with a message naming the expected
directory, and null-guard the global constructor — a misconfigured path should
be a diagnosable error, not a loader crash. If 1.29 fixed this deliberately,
confirming which change did so would help downstreams pin correctly.

---

## Draft 2 — prebuilt CPU-flavor core + CUDA provider library: heap corruption inside `Run()`

**Affects:** 1.28.0 official `onnxruntime-linux-x64-1.28.0.tgz` core paired
with the CUDA provider from the `onnxruntime-cuda-12` wheel; verified on
RTX A5000. A from-source build of the *identical* version does not crash.
**Not re-verified on 1.29** (the pairing is unusual; see below).

With the provider libraries placed in the core's `GetRuntimePath` (so Draft
1's crash does not trigger), sessions create successfully and `Run()` fails
with `free(): invalid pointer` under `onnxruntime::ConstantOfShape::~ConstantOfShape`
— classes that exist in both the core and the provider resolve across the
boundary inconsistently. A/B isolation: official-prebuilt CPU core + any CUDA
provider crashes; from-source core (same tag) + the same providers does not.

**Ask:** primarily a documentation request — if pairing the CPU-flavor
prebuilt core with the GPU providers is unsupported, stating so (or refusing
at registration) would save downstreams the A/B hunt. We found no
source-level difference to point at; the delta is something in the official
archive's build configuration.
