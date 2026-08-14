# Shipping a GPU flavor

Why `SET custom_extension_repository` cannot currently hand anyone a working
CUDA build, and what it would take. Written after issue #25, where the install
route the extension printed turned out to be unreachable — and correcting the
hostname was found to be the smaller half of the problem.

## The constraint

A DuckDB extension is **one file**. There is no sidecar mechanism: whatever
`LOAD anofox_tabfm` needs at that moment must already be inside the
`.duckdb_extension` or already on the system.

The cpu flavor satisfies that by linking ONNX Runtime statically. Verified
against the published artifact:

```
$ readelf -d anofox_tabfm.duckdb_extension | grep NEEDED
  librt.so.1  libdl.so.2  libpthread.so.0  libstdc++.so.6
  libm.so.6   libgcc_s.so.1  libc.so.6  ld-linux-x86-64.so.2
```

No `libonnxruntime.so`, and no exported `Ort*` symbols — 69 MB with ORT 1.23.2
inside it. That is why the community build works with nothing else installed.

The cuda flavor cannot be built that way. `cmake/ort.cmake` fetches the
prebuilt **GPU** archive and links `libonnxruntime.so` dynamically, and ORT
`dlopen`s the execution-provider library at session creation. From the official
`onnxruntime-linux-x64-gpu-1.23.2.tgz`:

| file | size | how it is needed |
|---|---:|---|
| `libonnxruntime_providers_cuda.so` | **351.4 MB** | `dlopen`ed when the CUDA EP is appended |
| `libonnxruntime.so.1.23.2` | 22.8 MB | linked dynamically by the extension |
| `libonnxruntime_providers_shared.so` | small | provider bridge |

Plus CUDA 12 and cuDNN 9 from the user's system, which ORT also `dlopen`s.

So a published `anofox_tabfm.duckdb_extension` built with `TABFM_FLAVOR=cuda`
would fail to load for anyone who has not separately installed a
**version-matched** ORT GPU runtime. `LOAD` would fail on the missing
`libonnxruntime.so`; get past that and appending the CUDA EP fails on the
provider library. (The second failure is at least legible now — before #22 it
was reported as `Attempt to use DefaultLogger but none has been registered`.)

Statically linking the GPU runtime is not a way out: 351 MB of device code in a
loadable extension is impractical, and the CUDA runtime stays dynamic anyway.

## Option A — do not publish (current behaviour)

The flavor-missing error names the from-source route, which is the one that
works:

```
anofox_tabfm: this build is the 'cpu' flavor and does not carry 'cuda'; the GPU
flavors are not published yet, so build one from source with TABFM_FLAVOR=cuda
(see docs/rocm-build.md for the rocm toolchain), or SET anofox_tabfm_device='cpu'.
Released cpu builds: SET custom_extension_repository = 'https://get.anofox.com'
```

Costs nothing and misleads nobody. Its weakness is that "build from source"
means building ONNX Runtime's dependencies too, which is a real afternoon.

## Option B — a runtime downloader

The extension already downloads large artifacts on demand: `tabfm_download`
pulls multi-GB weights into `<cache_dir>/<repo>@<revision>/…`, over DuckDB's
VFS, chunked, atomically published via a `.part` rename, sha256-validated, and
gated on a recorded licence acceptance. A GPU runtime is the same shape of
problem — a large versioned blob that only some users need — so it can reuse
that machinery rather than inventing a second one.

Sketch:

```sql
CALL tabfm_download_runtime('cuda');   -- fetches the ORT GPU libs into the cache
SET anofox_tabfm_device = 'cuda';
```

Open questions, in the order they need answering:

1. **Does the loader find them?** The extension links `libonnxruntime.so`
   dynamically, and by the time SQL runs, `LOAD` has already resolved (or
   failed to resolve) that. So the download cannot rescue the *link* — it can
   only supply the provider libraries that ORT `dlopen`s later. Which implies
   the shipped cuda flavor must link ORT **statically** (CPU core) and treat
   CUDA purely as a downloaded provider. Whether ORT supports that split — a
   static core that still loads `libonnxruntime_providers_cuda.so` — is the
   first thing to establish, and it decides whether this option exists at all.
2. **Version pinning.** The provider library must match the ORT core exactly.
   The download manifest therefore pins an ORT version alongside a sha256, the
   way model weights already are.
3. **Which platforms.** `cmake/ort.cmake` restricts cuda to `linux_amd64` and
   `windows_amd64`; the download would follow the same restriction.
4. **Licence.** ORT is MIT, CUDA and cuDNN are NVIDIA-licensed and not
   redistributable by us — so the download must come from NVIDIA/ORT release
   URLs, not from `get.anofox.com`, and the CUDA/cuDNN prerequisite stays the
   user's.

Question 1 is the gate. If a statically linked ORT core cannot load a
dynamically shipped CUDA provider, Option B collapses into Option C.

## Option C — publish and document prerequisites

Publish the dynamically linked cuda build and document exactly what to install
first. Cheapest to ship, worst to support: an ORT version mismatch surfaces as
a `dlopen` failure at session creation, and every such report costs a
round-trip to establish which of the three libraries is wrong.

## Recommendation

Stay on Option A until someone asks for a published GPU build, then answer
question 1 before designing anything further. The only known GPU user today
builds from source and has done so successfully on an A40 and an RTX 3060.

## Testing a GPU build without a GPU

`tools/gpu_test/` rents one by the minute; `docs/rocm-build.md` covers the ROCm
toolchain, which is testable locally on this hardware.
