# GPU (ROCm) and weight-loading memory — findings (2026-07-04)

Investigation into (a) running the real 6.56 GB TabFM model on the AMD GPU via
ROCm/MIGraphX, (b) whether ONNX external-data unblocks it, and (c) reducing the
weight-loading memory footprint. Everything below is measured on this box
(RX 9070 XT / gfx1201, ROCm 7.2.4, release build).

## TL;DR

| Path | GPU (MIGraphX) | CPU peak RSS | Output |
|---|---|---|---|
| inject + mmap (current default) | ❌ 2 GB proto → CPU fallback | 18.6 GB | correct |
| external-data (graph refs safetensors) | ❌ **still** 2 GB proto → CPU fallback | **7.3 GB** | correct |

- **GPU is blocked upstream.** ORT 1.23.2's MIGraphX EP re-inlines initializers
  when serializing a subgraph for MIGraphX, so the ~6.55 GB always lands back in
  the `onnx.ModelProto` and blows protobuf's hard 2 GB limit — regardless of how
  the source graph stores weights. Not fixable from the extension.
- **External-data is a 2.6× memory win** (18.6 → 7.3 GB, ~model size + overhead)
  and is the real answer to "do we need to load the whole model / can we mmap".

## ROCm setup (no vcpkg, no sudo)

vcpkg cannot provide MIGraphX (no port; it needs the ROCm toolchain and must
version-match the installed ROCm). But MIGraphX is just shared libs, and all its
heavy deps (rocBLAS, MIOpen, HIP, protobuf 35, msgpack, sqlite) are already on
the system. So:

```bash
URL=$(pacman -Sp migraphx)                       # 7.2.3-4, matches ROCm 7.2.4; no root
curl -sSL "$URL" -o migraphx.pkg.tar.zst
tar --zstd -xf migraphx.pkg.tar.zst -C <prefix>  # extract, don't install
export LD_LIBRARY_PATH=<prefix>/opt/rocm/lib:$PWD/build/ort-rocm/ort-install/lib:/opt/rocm/lib
```

Build the flavor (Makefile now forwards the ORT dir, commit 3be4dcb):

```bash
TABFM_FLAVOR=rocm TABFM_ORT_ROCM_DIR=$PWD/build/ort-rocm/ort-install make release
```

`tabfm_devices()` then shows `rocm:0 / MIGraphXExecutionProvider / gfx1201 /
usable=true` — **MIGraphX does claim RDNA4** (the open question: answered yes).

## The GPU blocker (2 GB protobuf limit)

A real predict with `device='rocm'` prints, twice, and silently runs on CPU
(GPU 0% util/mem via rocm-smi, CPU-equal timing, correct output):

```
[libprotobuf ERROR ...] onnx.ModelProto exceeded maximum protobuf size of 2GB: 6558367202
```

ORT's MIGraphX EP serializes its subgraph (`SerializeToString`) to hand to
`migraphx::parse_onnx_buffer`; that materializes the initializers inline. The
external-data spike (below) confirmed it re-inlines even when the source graph
keeps weights external: the error persists at ~6.55 GB.

### Direct MIGraphX backend — IMPLEMENTED (commit ebd4f36)

The ROCm GPU path is now a real engine backend (`src/tabfm_migraphx.cpp`,
`MIGraphXBackend`), selected by a `TabFMBackend` interface: `rocm` -> direct
MIGraphX, `cuda` -> ORT CUDA EP, `cpu` -> ORT CPU EP. It parses the bundled
migraphx-ready graph (`resources/graph_migraphx_<task>.onnx`: external-data +
Shape-rewrite), compiles per shape-bucket with weights read from disk, caches
the compiled program to a `.mxr`, pads inputs to the bucket (train_size/d mask
the padding), runs, returns the real rows.

**Verified end-to-end (commit pending):** `SET anofox_tabfm_device='rocm'` through
the integrated backend gives **identical output to CPU** on the real model
(age 27 -> false 0.9716; age 61 -> true 0.9937 — exact parity). The compiled
program cached as `graph_migraphx_classification_gfx1201_T128_H16.mxr`.

Performance profile:
- first predict per shape-bucket: ~20 min (compile + save `.mxr`),
- cold start with a cached `.mxr`: ~27 s (load the 6.6 GB program),
- warm session (program cached in memory): fast.

Costs to know: the `.mxr` **embeds the weights (~6.6 GB per bucket per arch)** — a
large, weight-containing, user-local cache artifact (never shipped, like the
safetensors). CPU/CUDA are unaffected (they use ORT).

### Background: the walls that had to fall (kept for reference)
1. **Direct MIGraphX backend (viable — validated parse).** ORT's MIGraphX EP is
   a dead end (it re-inlines). But migraphx's *own* file-based parser handles the
   >2 GB external-data model **fully** after two graph transforms
   (`tools/experimental/make_migraphx_graph.py`): (a) external-data ref to
   `model.safetensors`, and (b) rewrite degenerate `Shape(start==end)` ops into
   `Shape`+`Slice` (migraphx's parser rejects the degenerate form). It then
   parses clean (`@return`, output `{1,16,10}`) and **compiles AND runs on
   gfx1201** — confirmed end-to-end via `migraphx-driver perf --gpu` (static
   shape rows=16, features=8): real GPU kernels execute
   (`gpu::code_object::reduce_max_sub_exp_reduce_sum_div` = softmax, attention
   `dot`s, layernorm, `hip::hip_allocate_memory`), at **~99 ms/inference
   (10 inf/s)**, ~55 ms of it GPU instruction time. One-time compile ~22 min for
   a new arch (cache as `.mxr` in production). So GPU execution of TabFM on RDNA4
   **works** — just not through ORT's EP. **Productionizing = a direct-migraphx
   execution path in the engine** (link `libmigraphx`, `parse_onnx` -> `compile`
   (cache `.mxr`) -> `run`, with shape buckets for the dynamic rows/features
   dims), NOT via ORT. Bigger, but no upstream blocker remains.
2. **int8 quantization** — fp16 is still 3.3 GB (> 2 GB); only int8 (~1.6 GB)
   fits the ORT-EP proto. Accuracy/calibration cost.
3. Newer/patched ORT whose MIGraphX EP keeps subgraphs external (uncertain it
   exists for MIGraphX).

## "All flavours in one build" — not feasible as one binary

Each EP is compiled *into* its ORT core (`nm` shows
`OrtSessionOptionsAppendExecutionProvider_MIGraphX` as a direct export of the
rocm core, not a hot-swappable plugin). No obtainable ORT has both CUDA and
MIGraphX; a combined core needs `--use_cuda --use_migraphx`, i.e. both the CUDA
and ROCm SDKs at build time (this box has no CUDA SDK). **But cpu↔rocm already
switch at runtime** in the rocm build via `SET anofox_tabfm_device` (CPU EP vs
MIGraphX EP, same binary). CUDA stays a per-target artifact.

## External-data memory win — PRODUCTIONIZED ✅

Shipped as the default CPU loader: peak RSS **18.6 → 6.7 GB (2.76×)** on the real
model, identical output. `tools/make_external_graph.py` bakes each initializer's
offset into `resources/graph_ext_<task>.onnx` (weight-free); the engine
(`TryExternalDataSession`) validates the downloaded safetensors' header SHA-256
against a baked constant, stages the graph beside the weights, and
`CreateSessionFromPath` (ORT reads weights from disk — no injection). Any header
mismatch or non-local path falls back to the inject path; `TABFM_DISABLE_EXTERNAL_DATA`
forces inject. Fixtures fall back automatically (SHA mismatch).

### Original spike notes (kept for reference)

The safetensors data section is raw, contiguous, row-major tensor bytes, so an
ONNX external-data offset is simply `8 + header_len + st_data_offset`. Repointing
every initializer at the cached `model.safetensors` and loading the graph
from-path (ORT reads weights itself, no in-memory injection) gives:

- **peak RSS 18.58 → 7.26 GB** (−11.3 GB, 2.6×), correct output.
- simpler loader: no read, no safetensors parse, no F32 arena, no injection.

### Reproduce
`scratchpad/make_external.py` rewrites `resources/graph_classification.onnx` into
`<cache>/classification/graph_ext.onnx` (913/914 initializers externalized; the
1 leftover is the tiny inline constant S01 keeps). The engine spike branch
(env-gated, reverted from main — kept here) in `LoadOrGetSession`:

```cpp
if (std::getenv("TABFM_EXTERNAL_DATA")) {
    // graph references weights as ONNX external-data on disk; ORT loads them,
    // no injection, so the ModelProto stays tiny.
    TabFMSessionConfig cfg; cfg.intra_op_threads = MaxValue<int64_t>(1, ctx.threads);
    auto dev = ResolveDevice(ctx.device, DiscoverDevices());
    cfg.device_id = dev.device_id; cfg.device_ordinal = dev.device_ordinal;
    cfg.model_tag = TabFMTaskName(resolved.manifest.task);
    auto session = CreateSessionFromPath(resolved.graph_path, {}, cfg);
    // hold session only; register in TabFMState
}
```

Run: manifest → `graph_ext.onnx`, `TABFM_EXTERNAL_DATA=1`, `device='cpu'`.

### Productionization plan (default CPU loader)
1. **Export-time**: bake external-data offsets (from the deterministic Google
   checkpoint layout) into the shipped `resources/graph_*.onnx`, location =
   `model.safetensors`. Regenerate the bundled resources.
2. **Load-time**: write the bundled graph beside the cached safetensors, then
   `CreateSessionFromPath`. **Validate** the safetensors header (len/hash)
   matches what the offsets assume; on mismatch **fall back** to the current
   inject path (robust against HF re-serialization).
3. Drop the mmap/read/parse/arena/injection path for the built-in models once
   external-data is the default (keep inject as the fallback + for custom
   manifests without external-data).

Expected result: default CPU RSS ~7.3 GB instead of ~18.6 GB.

## CUDA: `tabicl-v2` and the untrimmed `Slice` (2026-08-13)

Separate blocker, separate vendor, same shape of problem: a graph that runs
correctly on the CPU EP fails inside inference on the GPU one.

Both TabICL graphs died at a `ScatterND`:

```
updates shape: {9,7,1,128}, indices shape: {12,1}, data shape: {12,7,1,128}
```

with 9 training rows and 12 total. **The export is correct.** `indices` is

```
Unsqueeze( Slice( Range(0, T, 1), starts=[0], ends=[S] ), -1 )
```

which is length `S` by construction, matching `updates`. ONNX cannot evaluate
`min(T,S)` symbolically, so the inferred shapes merely *look* mismatched — that
misreading cost a day, and is worth remembering the next time a shape-inference
result disagrees with a graph you can read.

Reading the tensor off the device (feed `updates` with `T` rows so the shape
check passes on CUDA, then expose `indices`) shows the CUDA `Slice` returning its
input **unmodified** — `[0,1,…,11]`, the whole `Range`, not garbage.

Which buffer, by exposing one intermediate at a time as a graph output (graph
outputs are exempt from buffer reuse):

| exposed | CUDA |
|---|---|
| *(none)* | FAIL |
| the **T** scalar | FAIL |
| **the `S` scalar** | **OK** |
| **`ends`** | **OK** |
| the `Range` / `Slice` / `indices` / `data` tensors (CUDA-side) | FAIL |

Only the two tensors carrying `S` fix it, and `enable_cpu_mem_arena=False` fixes
a reduced 68-node cut (though *not* the full graph). So: the CPU-side buffer
holding `S` is recycled before the CUDA `Slice` reads it, the `Slice` sees a
bound ≥ T, and trims nothing.

### The workaround

`tools/pin_dynamic_slice_bounds.py` names those bounds as graph outputs. It finds
them structurally — `ScatterND.indices` → `Slice` → `ends` — because the
classification and regression graphs number their nodes differently and a
re-export renumbers them again. `--check` re-verifies and exits non-zero, so a
re-export cannot silently drop it.

Measured (RTX 3060 sm_86, ORT 1.26.0, random f32 weights, requesting only
`logits` exactly as the engine does):

| graph | CUDA before | CUDA after | CPU vs CUDA after | CPU before vs after |
|---|---|---|---|---|
| classification | FAIL | OK | rel 3.5e-04 | **bit-identical** |
| regression | FAIL | OK | rel 1.5e-06 | **bit-identical** |

The CPU path is unchanged byte for byte, which is what makes this safe to ship.

**Caveats.** Random weights, not the real checkpoint — the agreement shows CPU
and CUDA compute the same function, not that the real checkpoint behaves
identically; the golden fixture is the check worth doing. One GPU. Shapes
T ∈ {12,20}, S ∈ {9,12}, F=3. And this masks an ONNX Runtime bug rather than
fixing it — still present in 1.28.0, so an ORT bump does not help. Upstream
detail, including a 68-node reproduction that needs no weights and no licence,
is in issue #21.
