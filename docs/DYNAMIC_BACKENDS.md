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

### Phase 3 — CUDA as a plugin EP + `tabfm_download_runtime('cuda')`

**Premise confirmed on 1.28.** The provider's exported plugin-ABI symbols
(`CreateEpFactories` / `ReleaseEpFactory`):

| ORT | symbols | consequence |
|---|---|---|
| 1.23.2 (shipped) | **0** | classic ABI — `RegisterExecutionProviderLibrary` rejects it |
| 1.28.0 | **2** | registrable by absolute path, i.e. from our cache |

So phase 3 is unblocked by phase 2 and by nothing else.


Register the provider from the cache by absolute path. Reuses the existing
download machinery — chunked reads, atomic `.part` publish, sha256 validation,
licence gating — the same path `tabfm_download` uses for weights. The 351 MB
provider is cached like a model. CUDA 12 / cuDNN 9 remain the user's to
install: they are not ours to redistribute.

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
* **Cache size**: 351 MB (CUDA provider) and ~6.6 GB per `.mxr` shape bucket
  (ROCm, ~20 min first compile). Both belong in the weights cache with the
  messaging that already exists there.
* **Binary size and CI matrix** grow by one plugin per GPU backend.
* **ABI drift** between the extension and our own ROCm plugin — versioned
  explicitly, refused on mismatch.
