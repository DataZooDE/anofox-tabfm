# GPU acceleration

A GPU backend is a plugin the extension `dlopen`s at runtime, not a separate build of the extension: fetch
it, then select the device. Nothing changes until you ask; without a plugin every model runs on the CPU.
The reference (support matrix, precision, platform limits) is [GPU](../api/04-gpu.md).

## 1. Set it up

```sql
CALL tabfm_accelerate();   -- finds the card, fetches its plugin, verifies it loads
```

It reports what it did as `(step, status, detail)` rows, including what is
still ahead of you (MIGraphX's first-shape compile, MLX's `brew install mlx
mlx-c`). Re-running it re-verifies without re-downloading.

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
CALL tabfm_download_runtime('cuda');    -- plugin + ORT GPU runtime, into the cache dir
SET anofox_tabfm_device  = 'cuda';      -- or 'rocm', or 'mlx' on Apple Silicon
-- anofox_tabfm_ep_path is only needed to override where the plugin lives
```

Naming a device explicitly is a hard request: if it cannot be served it
errors, rather than quietly running on the CPU.

## ROCm: the first-run compile

MIGraphX compiles a program per (GPU arch, precision, shape bucket). The first predict in a bucket pays
that compile, which takes minutes ([per model](../api/04-gpu.md#rocm-compile-time)); every *later* predict
in that bucket is sub-second. Two tools keep it off the query path:

```sql
-- warm a bucket now, on purpose: task, then optionally rows and features
CALL tabfm_gpu_precompile('classification', 1000, 50, model := 'mitra');
SET anofox_tabfm_mxr_source = '/shared/mxr';   -- or stage programs compiled elsewhere
```

Precompile once per shape bucket you will use (the compiled `.mxr` persists in
the weight cache across sessions); `mxr_source` lets a fleet share one
machine's compiles. CUDA has no analogous cost (~30 s cold start, no long
compile).
