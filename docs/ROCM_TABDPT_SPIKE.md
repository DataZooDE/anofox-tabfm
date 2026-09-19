# Converting tabdpt to run on ROCm — scoped, costed, not executed

`docs/ROCM_SINGLE_EVAL_POS.md` establishes *why* nine of eleven models cannot be
served on MIGraphX, and recommends `tabdpt` as the one worth converting if any.
This is the follow-up it asks for: what the conversion actually is, what it
really costs, and what it would buy.

**Status: scoped against the real export code, not executed.** The blocker is
stated at the end and is not effort.

---

## The change, concretely

The positional split lives in
`tools/export_tabdpt/src/export_tabdpt/tabdpt_patches.py`, in the patched
attention. Every line below is already patched for export; `eval_pos` is
`y.shape[0]`, a symbolic dim:

```python
k     = self.k_proj(h[:, :eval_pos])          # context rows only
h_ctx = h[:, :eval_pos]
v_in  = torch.cat([h_ctx, y], dim=-1)
v     = self.v_proj(v_in)

k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)

beta = self.get_scale_param(eval_pos, device=q.device, dtype=q.dtype)
attn = F.scaled_dot_product_attention(q * beta, k, v, scale=default_scale)
```

`eval_pos` is a **shape** here — it sizes `k` and `v`. That is precisely what a
fixed-shape MIGraphX compile cannot bucket: a different train size is a
different program, so the `(T, H)` bucket table in `tabfm_shape_bucket.hpp`
would need a third axis whose values are unbounded.

The conversion makes it a **value** instead:

```python
# contract becomes x[1,T,H], y_full[1,T], train_size[1], d[1]  -- the
# tabfm-v1 / mitra shape, which buckets today
k = self.k_proj(h)                       # ALL T rows: shape is now T, not eval_pos
v = self.v_proj(torch.cat([h, y_full], dim=-1))
k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

is_context = (torch.arange(T, device=h.device) < train_size)      # [T] bool
bias = torch.where(is_context, 0.0, float("-inf"))                # [T] additive
attn = F.scaled_dot_product_attention(q * beta, k, v,
                                      attn_mask=bias.view(1, 1, 1, T),
                                      scale=default_scale)
```

Every tensor's shape is now a function of `(T, H)` alone. `beta` already avoids
specializing (`_patched_get_scale_param` builds it with
`torch.ones(eval_pos).sum()`); it would take `train_size` as a real input
instead, which is *simpler* than what is there now.

## What it costs at run time

Not free, and the direction is against us:

- **K/V projection grows from `S` rows to `T`.** At the documented 80 % context
  split that is 1.25× the projection work; for a small labelled context scoring
  many rows it is much worse — a 100-row context scoring 4000 rows goes from
  100 to 4100, a 41× increase in K/V work per layer.
- **The attention matrix grows from `L×S` to `T×T`.** Padding to a bucket makes
  this worse again: a 600-row call padded to the 1024 bucket computes 1024²
  scores where the unpadded positional version computed 600×480.
- **`-inf` rows are computed and then discarded.** That is inherent to masking
  and is the trade the train_size-scalar family already makes.

So the converted graph is *slower per call than the same model unconverted on
the CPU would suggest* — it buys GPU execution, and must beat the CPU by more
than this overhead to be worth anything. `docs/DYNAMIC_BACKENDS.md` measures
20–80× at 2500 rows for the models that do run on ROCm, so the headroom is
probably there; "probably" is the whole reason this is a spike.

## What artifacts actually change — two corrections

The pre-merge review of the plan asserted this is a catalog-wide artifact bump:
that regenerating a graph changes its `ExpectedWeightsHeaderShaFor` entry,
which gates the CUDA and CPU bundled-graph paths too, and invalidates every
user's `.mxr` cache. **Both halves are wrong, and I repeated them in commit
`3d5e672` before checking.** What the code actually does:

- **`ExpectedWeightsHeaderShaFor` is not affected.** It returns the SHA-256 of
  the *safetensors file's JSON header* (`ReadWeightsHeaderBytes` reads the
  leading length-prefixed header out of the weights file, `tabfm_engine.cpp`).
  It is a property of the **downloaded weights**, not of the exported graph —
  it exists so a graph's baked external-data offsets are known to index the
  right weight bytes. Masking reuses every existing parameter and adds none, so
  the weights file is untouched and the sha stays valid. Nothing on the CUDA or
  CPU path is disturbed.
- **The `.mxr` cache is not invalidated, only bypassed.** `MxrCacheStem`
  (`tabfm_mxr_cache_key.hpp`) embeds an FNV-1a hash of the graph's *content*.
  A new graph therefore gets a *new* cache entry; old entries are never loaded
  for it. No user gets a wrong answer — the failure mode that motivated content
  hashing in the first place is exactly what prevents this. They pay one
  recompile, and the stale files are inert until deleted.

The real change set is correspondingly smaller:

| artifact | changes? |
|---|---|
| `tabdpt_patches.py` attention + contract | yes — the work |
| `graph_migraphx_tabdpt_*.onnx` | new (does not exist today) |
| `graph_ext_tabdpt_*.onnx` | **yes** — contract change is model-wide, so the CUDA graph must be re-exported to match |
| `model.safetensors` / header sha | no |
| tensor map | no — no new initializers |
| engine C++ | no — the target contract is the one `tabfm-v1`/`mitra` already use |
| user `.mxr` caches | not invalidated; new entries, old ones inert |

The one genuinely catalog-shaped cost is that `ext` and `migraphx` graphs must
agree on the contract, so converting tabdpt means re-exporting **both** of its
graphs and re-establishing CUDA and CPU parity, not just adding a ROCm one.
That is the item to budget, and it is a per-model cost of roughly "one model's
onboarding", not the cross-catalog bump the review feared.

## Why this is not executed here

Two reasons, neither of them effort:

1. **MIGraphX is not installed on the development machine.** ROCm is
   (`/opt/rocm`, gfx1201, `/dev/kfd` with two nodes), but `libmigraphx_c.so` is
   absent — confirmed by `tabfm_accelerate()`, which downloads the published
   plugin successfully and then reports
   `libmigraphx_c.so.3: cannot open shared object file`. A converted graph
   could be exported here but not compiled, not run, and not compared.
2. **An unverified conversion is worse than none.** The change alters attention
   numerics (masked softmax over padded rows rather than a slice). Parity
   against the CPU reference is the only thing that distinguishes "converted"
   from "converted wrong", and a wrong graph that produces plausible scores is
   this repo's documented worst failure — it is what the content-hashed `.mxr`
   key and every `*_SERVED_BY` assertion exist to prevent.

## Recommendation

**Do it, on a box with MIGraphX, as one model.** The economics are better than
the review assumed: no header-sha churn, no cache invalidation, no engine
change — the target contract is one the engine has served since day one. The
cost is re-exporting tabdpt's two graphs and re-establishing parity on CPU,
CUDA and ROCm.

**Do not commit to the other eight on this analysis.** The run-time cost above
is real and grows with the query/context ratio; tabdpt should be measured with
`tools/bench/bench.py` against its own CPU baseline before the pattern is
repeated. If the converted ROCm path does not beat CPU by a wide margin on a
realistic shape, the correct outcome is to keep `docs/ROCM_SINGLE_EVAL_POS.md`'s
original recommendation — document the limitation, refuse loudly — and spend
the time on MLX and CUDA, which serve the whole catalog today.

Until then `tabfm_backends()` answers the question honestly for every model,
which was the actual user-facing problem.
