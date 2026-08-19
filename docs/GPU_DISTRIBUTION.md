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

> That wording is historical. Device resolution now accepts `cuda` and
> `rocm` on every build, because a plugin can serve them; the message you
> get today names the model or the missing plugin instead. See
> `docs/DYNAMIC_BACKENDS.md`.

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

> **This shipped, though not by the route sketched below.** The user-facing
> shape above is exactly what landed; what changed is the mechanism. Loading
> ORT's CUDA provider into the extension's own ORT turned out to be impossible
> in the build that ships (a statically linked ORT interposes the provider's
> symbols and corrupts the heap), so CUDA runs in a standalone backend plugin
> carrying its own shared ORT-GPU runtime instead. The open questions below are
> kept as the record of how that was established — see
> `docs/DYNAMIC_BACKENDS.md`, "Phase 3, resolved: CUDA is a plugin, exactly
> like ROCm", for the design that exists.

Open questions, in the order they need answering:

1. ~~**Does the loader find them?**~~ **Answered — no, not on the ORT we ship.**
   Measured on a rented GPU with ORT 1.23.2: registering
   `libonnxruntime_providers_cuda.so` from an arbitrary directory via
   `RegisterExecutionProviderLibrary` fails with

   ```
   Failed to get symbol CreateEpFactories with error:
   .../libonnxruntime_providers_cuda.so: undefined symbol: CreateEpFactories
   ```

   That API requires the **plugin-EP ABI**. The 1.23.2 CUDA provider is a
   *classic* provider: it exports `GetProvider` and nothing else (verified —
   zero `CreateEpFactories`/`ReleaseEpFactory` symbols), and the only path that
   loads it is `Env::GetRuntimePath() + filename` in
   `provider_bridge_ort.cc`. `GetRuntimePath` is `dladdr`-based, so with ORT
   linked statically it resolves next to **our `.duckdb_extension`** — DuckDB's
   extension directory, not a cache we control. There is no bare-filename
   fallback on that branch, so `LD_LIBRARY_PATH` does not help either.

   The rest of the architecture is fine, which is why this was worth testing:
   the provider needs **no ORT symbols at all** (0 of 335 undefined symbols are
   ORT-namespace; it links only libc, libstdc++ and the CUDA libraries), and
   the statically linked core we already publish contains the whole bridge —
   `CudaProviderFactoryCreator::Create`, `TryGetProviderInfo_CUDA`,
   `LoadDynamicLibraryFromProvider`, `RegisterExecutionProviderLibrary`. Only
   the provider's ABI is wrong for the API that takes a path.

   **This changes on newer ORT.** 1.28 can build CUDA *as a plugin EP*
   (`cuda-plugin-ep` appears in `get_build_info()`, and the wheel ships
   `_get_cuda_plugin_ep_library_path`), which is the plugin ABI and therefore
   registrable by absolute path. So the option is not dead — it is gated on
   moving to an ORT that offers the CUDA plugin EP, and on sourcing a
   plugin-ABI build of the provider.

2. **Version pinning.** The provider must match the ORT core exactly. The
   download manifest therefore pins an ORT version alongside a sha256, the way
   model weights already are.
3. **Which platforms.** `cmake/ort.cmake` restricts cuda to `linux_amd64` and
   `windows_amd64`; the download would follow the same restriction.
4. **Licence.** ORT is MIT, CUDA and cuDNN are NVIDIA-licensed and not
   redistributable by us — so the download must come from NVIDIA/ORT release
   URLs, not from `get.anofox.com`, and the CUDA/cuDNN prerequisite stays the
   user's.

## Option C — publish and document prerequisites

Publish the dynamically linked cuda build and document exactly what to install
first. Cheapest to ship, worst to support: an ORT version mismatch surfaces as
a `dlopen` failure at session creation, and every such report costs a
round-trip to establish which of the three libraries is wrong.

## Recommendation

Stay on Option A. Option B is blocked on the ORT version rather than on
anything about this project, so the trigger for revisiting it is an ORT upgrade
to a release carrying the CUDA plugin EP — at which point the download
machinery already exists and the design is a day's work rather than a
question. The only known GPU user today
builds from source and has done so successfully on an A40 and an RTX 3060.

## Testing a GPU build without a GPU

`tools/gpu_test/` rents one by the minute; `docs/rocm-build.md` covers the ROCm
toolchain, which is testable locally on this hardware.
