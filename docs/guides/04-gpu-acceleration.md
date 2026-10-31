# GPU acceleration

**Using a GPU.** A GPU backend is a plugin the extension `dlopen`s at runtime,
not a separate build of the extension: fetch it, then select the device.

```sql
CALL tabfm_accelerate();   -- finds the card, fetches its plugin, verifies it loads
-- that is it: anofox_tabfm_device defaults to 'auto', which now routes each
-- model to the best device IT can be served on, in this session.
```

It reports what it did as `(step, status, detail)` rows, including what is
still ahead of you (MIGraphX's first-shape compile, MLX's `brew install mlx
mlx-c`). Re-running it re-verifies without re-downloading.

To see the routing — and the reason for every model that stays on the CPU:

```sql
SELECT * FROM tabfm_backends() WHERE NOT supported;
-- and confirm what actually served a query, rather than trusting the answers:
SELECT model, device FROM tabfm_models() WHERE loaded;
```

`device` is which backend is serving the model right now — `cpu`, `cuda:0`,
`rocm:0`, `mlx:0` — and is empty until a predict warms a session. It is the way to check
that a GPU is actually being used: a device that silently fell back to the CPU
produces the same answers, only slower, so agreement alone will not tell you.

The manual equivalent, if you want to place things yourself:

```sql
CALL tabfm_download_runtime('cuda');    -- plugin + ORT GPU runtime, into the cache dir
SET anofox_tabfm_device  = 'cuda';      -- or 'rocm', or 'mlx' on Apple Silicon
-- anofox_tabfm_ep_path is only needed to override where the plugin lives
```

Naming a device explicitly is a hard request: if it cannot be served it
errors, rather than quietly running on the CPU.

## ROCm: the first-run compile

**ROCm first-run cost — plan for it.** MIGraphX compiles a program per
(GPU arch, precision, shape bucket), and on the full-size model that compile is
**~25 minutes** (measured, gfx1201, fp32 or bf16) — while every *later* predict
in that bucket is sub-second. Two tools manage this:

```sql
CALL tabfm_gpu_precompile('classification');   -- warm the bucket now, on purpose
SET anofox_tabfm_mxr_source = '/shared/mxr';   -- or stage programs compiled elsewhere
```

Precompile once per shape bucket you will use (the compiled `.mxr` persists in
the weight cache across sessions); `mxr_source` lets a fleet share one
machine's compiles. CUDA has no analogous cost (~30 s cold start, no long
compile).
