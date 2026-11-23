# GPU acceleration

A GPU backend is a plugin the extension `dlopen`s at runtime, not a separate build of the extension: fetch
it, then select the device. Nothing changes until you ask; without a plugin every model runs on the CPU.
The reference (support matrix, precision, platform limits) is [GPU](../api/04-gpu.md).

## 1. Set it up

```sql
CALL tabfm_accelerate();   -- finds the card, fetches its plugin, verifies it loads
```

It reports what it did as `(step, status, detail)` rows, including what is
still ahead of you (the ROCm runtime libraries the AMD plugin uses, MLX's `brew
install mlx mlx-c`). Re-running it re-verifies without re-downloading.

That is all: `anofox_tabfm_device` defaults to `auto`, which routes each model to the best device whose
plugin is present and which can serve that model, else the CPU.

## 2. Check what serves each model

To see the routing — and the reason for every model that stays on the CPU:

```sql
SELECT * FROM tabfm_devices();                       -- cpu, plus any CUDA / ROCm / MLX device found, and whether it is usable
SELECT * FROM tabfm_backends() WHERE NOT supported;  -- per model and device: why not
```

Then confirm what actually served a query, rather than trusting the answers:

```sql
SELECT * FROM tabfm_classify('iris', 'species', model := 'mitra') WHERE species IS NULL;
SELECT model, task, device FROM tabfm_models() WHERE loaded;
```

`device` is which backend is serving the model right now — `cpu`, `cuda:0`,
`rocm:0`, `mlx:0` — and is empty until a predict warms a session. It is the way to check
that a GPU is actually being used: a device that silently fell back to the CPU
produces the same answers, only slower, so agreement alone will not tell you.

## 3. Or place things yourself

```sql
CALL tabfm_download_runtime('cuda');    -- the ggml CUDA plugin, into the cache dir
SET anofox_tabfm_device  = 'cuda';      -- or 'rocm', or 'mlx' on Apple Silicon
-- anofox_tabfm_ep_path is only needed to override where the plugin lives
```

Naming a device explicitly is a hard request: if it cannot be served it
errors, rather than quietly running on the CPU.

## What each plugin needs

- **CUDA** (Linux x64): only the NVIDIA driver. The plugin links the CUDA runtime and cuBLAS statically, so
  no CUDA toolkit is needed; the card must be compute capability 7.5 (Turing) or newer.
- **ROCm** (Linux x64): a ROCm 6.4 or newer install, whose HIP, hipBLAS and rocBLAS libraries the plugin
  loads. A card, iGPUs included, is usable when the published plugin carries kernels for its arch: gfx1030,
  gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201, gfx90a and gfx942. Any other card is listed
  with `usable = false` and `'auto'` passes over it.
- **MLX** (macOS arm64): `brew install mlx mlx-c`.

A GPU session is ready as soon as its weights are on the device; the first call of a new shape builds its
graph and is a little slower than later ones. A model too
large for the card's free memory fails at load, naming `CALL tabfm_unload()`, `anofox_tabfm_max_sessions`
and `SET anofox_tabfm_device='cpu'` as the ways out ([GPU memory](../api/04-gpu.md#gpu-memory)).
