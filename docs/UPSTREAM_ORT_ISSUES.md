# Upstream ONNX Runtime issue drafts (P8) — ready to file, not yet filed

Status per S5 of `docs/GPU_HARDENING_PLAN.md`, updated 2026-08-22: the crash
below **does not reproduce on the 1.29.0 wheel** (exit 0 on an A40), and this
project now pins 1.29.0 everywhere (phase 2 complete) — so we are no longer
affected. Both drafts stay scoped "affects ≤ 1.28.x, apparently resolved by
1.29"; filing would only serve other downstreams on 1.28.x or confirm the fix
was intentional. Filing is an outward-facing action; these stay drafts until
explicitly approved.

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

---

## Draft 3 — CUDA EP: a CPU-side shape tensor's buffer is reused while still live (Expand gets a wrong shape)

Status 2026-11-02: **filed** as [microsoft/onnxruntime#33807](https://github.com/microsoft/onnxruntime/issues/33807). Affects us today; worked around graph-side
(`tools/pin_cuda_shape_operands.py`), so filing is for the fix upstream, not for us.

**Title:** `[CUDA EP] Expand receives a wrong shape: CPU buffer of a live int64 shape tensor is reused (1.29.0, 1.31.1)`

**Versions.** `onnxruntime-gpu` 1.29.0, CUDA-12 build (`onnxruntime-cuda-12` index; what we ship),
and 1.31.1 from PyPI (CUDA-13 build): both fail. CPU EP of the same versions: correct.
RTX 4090 and RTX 3090 (driver 595.91.07, `runpod/pytorch:2.8.0-py3.11-cuda12.8.1` image, cuDNN 9.8 /
cuBLAS 12.8 from the image's `nvidia-*-cu12` wheels), Python 3.11, Ubuntu 22.04. Default
`SessionOptions` (`ORT_ENABLE_ALL`). An earlier run also saw it at every graph optimisation level and
with `enable_mem_pattern = False`; the `ORT_DISABLE_ALL` result was not repeated for this draft.

**Body.**

A dynamo-exported transformer (TabPFN-3.5, opset 18, inputs `x: [1, T, H]`, `y: [1, S]`) computes
an Expand target shape on the host:

```
Shape(x)[1:2] -> Squeeze -> T          Shape(y)[1:2] -> Squeeze -> S
Add(T, Min(T, S)) -> Reshape([-1]) -> a          Shape(x)[2:3] -> Squeeze -> Reshape([-1]) -> H
Concat(a, [1], H) -> Expand(unsqueeze, shape)     # expected {T+min(T,S), 1, H}
```

ORT's CUDA EP places this whole chain on the CPU (`GetCpuPreferredNodes`), and the Expand runs on
CUDA. In the optimized model (`ORT_ENABLE_ALL`) the two `Reshape([-1])` nodes have been merged by common
subexpression elimination and are each read by 11-18 Concats. At run time the Concat reads the **row** expression where `H` should be:

```
node_expand: left operand cannot broadcast on dim 2 LeftShape: {1,1,5}, RightShape: {170,1,170}
                                                       expected RightShape: {170,1,5}
```

With `enable_cpu_mem_arena = False` the same slot holds freed memory
(`RightShape: {170,1,298762}`, `{170,1,1136623992}`). So the CPU buffer behind `H` (or one of the
`Squeeze`/`Reshape` aliases of `Shape(x)`'s output) is released and handed to `Add`'s output
while the alias is still to be read. The classification graph fails the same way one step later,
at a Where (`Shape mismatch attempting to re-use buffer. {1,1,100} != {1,5,100}`).

**Naming the shape tensors as graph outputs makes it go away** (they are then excluded from
reuse), with the CUDA result matching the CPU (max abs 4e-9 on the regression output with random
weights; 0/160 label differences with the real weights). That is the workaround we ship. It is
the same lever that worked for #32083 (a Slice bound read from a recycled CPU buffer).

**Repro** (no weights, no licence needed: the graph is public and weight-free, initializers are
synthesized): [`tools/gpu_test/ort_repro_tabpfn35.py`](../tools/gpu_test/ort_repro_tabpfn35.py)

```bash
pip install onnx numpy
pip install --no-deps onnxruntime-gpu==1.29.0 \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
python ort_repro_tabpfn35.py             # CPU OK, CUDA FAIL at node_expand
python ort_repro_tabpfn35.py --no-arena  # CUDA FAIL with a garbage dimension
python ort_repro_tabpfn35.py --pinned    # shape tensors as graph outputs: both OK
```

Output on the RTX 3090 (ORT's own error log lines omitted):

```
onnxruntime 1.29.0  graph v2026.11.01 (6711 nodes, 1 outputs)  cpu_mem_arena=True
  x=[1,100,5] y=[1,70]
    cpu  OK   logits (1, 100, 1)
    cuda FAIL [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Expand node. Name:'node_expand' Status Message: node_expand: left operand cannot broadcast on dim 2 LeftShape: {1,1,5}, RightShape: {170,1,170}
  x=[1,128,8] y=[1,100]
    cpu  OK   logits (1, 128, 1)
    cuda FAIL [...] LeftShape: {1,1,8}, RightShape: {228,1,228}

--no-arena:   cuda FAIL [...] LeftShape: {1,1,5}, RightShape: {170,1,32140...}   (garbage, varies per run)

--pinned:     onnxruntime 1.29.0  graph v2026.11.01 (6711 nodes, 1123 outputs)  [shape tensors pinned]
  x=[1,100,5] y=[1,70]
    cpu  OK   logits (1, 100, 1)
    cuda OK   logits (1, 100, 1)  max|cuda-cpu|=1.9e-08
  x=[1,128,8] y=[1,100]
    cuda OK   logits (1, 128, 1)  max|cuda-cpu|=3.7e-09
```

**Smaller cuts.** `onnx.utils.Extractor` cuts from `x, y` to `node_expand`'s output (81 nodes) and to
later outputs (95, 82, 555 and 768 nodes) all run correctly on CUDA. A 2146-node cut (to `node_view_19`, 25
weight initializers, synthesized) fails exactly like the full graph. So the bug depends on how much of the
graph is planned around the shape chain, not on the chain alone. We can attach the 2146-node cut (230 KB,
weight-free) if that helps.

**Ask:** the allocation planner should not release or reuse a CPU tensor (or the buffer it
aliases through `Squeeze`/`Reshape`) consumed by a CPU node whose output feeds a CUDA kernel's
host-memory input, while that consumer has not run. Pointers to the planner code involved (the
multi-stream release plan for CPU values with CUDA consumers?) would let us test a patch.
