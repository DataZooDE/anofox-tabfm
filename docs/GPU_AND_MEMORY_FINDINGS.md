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

### Real GPU paths (all separate, larger efforts)
1. **Offline migraphx-native compile** — migraphx's *file-based* `parse_onnx`
   *does* handle external data. Compile the external-data graph to a `.mxr`
   offline (`migraphx-driver compile`), load via the EP's compiled-model option,
   bypassing ORT's proto serialization. Most promising.
2. **int8 quantization** — fp16 is still 3.3 GB (> 2 GB); only int8 (~1.6 GB)
   fits the proto. Accuracy/calibration cost.
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

## External-data memory win (validated, not yet productionized)

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
