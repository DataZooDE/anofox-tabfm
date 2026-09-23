# Converting tabdpt to run on ROCm — done, measured, 5.6x

`docs/ROCM_SINGLE_EVAL_POS.md` establishes *why* nine of eleven models cannot be
served on MIGraphX, and recommends `tabdpt` as the one worth converting if any.
This is the follow-up it asks for: what the conversion actually is, what it
really costs, and what it would buy.

**Status: executed and verified on hardware.** TabDPT serves on ROCm with
identical predictions to the CPU. What follows is the analysis as it was written
before the attempt, then what actually happened — kept in that order because the
gap between them is the useful part.

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

## Measured baseline: what ROCm is actually worth here

Run on the dev box (RX 9070 XT, gfx1201, ROCm 7.2.4 + MIGraphX 7.2.3) with
`tools/bench/bench.py`, mitra, 3 features, 80% context, warm (shape buckets
precompiled), every cell confirmed by `served_by`:

| rows | CPU | ROCm | speedup |
|---:|---:|---:|---:|
| 100 | 366 ms | *unmeasurable* | — |
| 1000 | 4035 ms | 443 ms | **9.1x** |
| 2500 | 14914 ms | 2455 ms | **6.1x** |

Two things to read carefully here.

**The 100-row cell is null, not fast.** The harness reports null when the
difference is not positive — at that size the inference is lost in process
startup. It is not evidence that ROCm is slow at 100 rows; it is evidence that
this method cannot measure it, which is the honest output.

**These are lower than the 20-80x in `docs/DYNAMIC_BACKENDS.md`.** Different
measurement: that table is tabfm-v1 (a 6.5 GB model) timed around the forward
pass; this is mitra (300 MB) end-to-end through SQL, including preprocessing
and decode. Neither is wrong; they answer different questions, and this one is
the question a user experiences.

### Why this baseline is the right estimate for converted tabdpt

The obvious objection to the conversion is that masking does strictly more
work: K/V projected over all `T` rows instead of `S`, and attention over `T x T`
instead of `L x S`, with bucket padding making both worse. At 2500 rows padded
to the 4096 bucket that is roughly an order of magnitude more attention
arithmetic. If the GPU only buys 6-9x, the overhead could eat the entire win.

**It does not, and mitra is the proof.** Mitra is already in the
train_size-scalar family: it declares `train_size` and `d`, pads to the same
buckets, and masks exactly as converted tabdpt would. The contract this spike
would give tabdpt is the contract mitra already has. So the 6.1-9.1x above is
not a speedup that the conversion overhead must still be subtracted from — it
is a speedup measured *with that overhead already paid*.

That does not make the conversion certain to pay off for tabdpt specifically:
its architecture differs (retrieval-style attention, a different depth/width
ratio), and only converting it settles that. But it removes the reason to
expect it cannot.

## EXECUTED — and the analysis above was incomplete

Done on the dev box (RX 9070 XT gfx1201, ROCm 7.2.4, MIGraphX 7.2.3). **TabDPT
runs on ROCm.** Classification: `TEST_SERVED_BY=rocm:0`, **zero** query-row
label disagreements against the CPU, and **103.2 ms → 18.59 ms at 100 rows
(5.6x)**. Regression: `rocm:0`, max query-row difference 3.4e-05, which is fp32
GPU noise on continuous outputs.

The conversion is what this document predicted. The *obstacle count* was wrong:
there were **three** blockers, and the documented one turned out to be the
least interesting.

### Blocker 1 — the positional split (the one this document is about)

Solved as described: `arange(T) < train_size` instead of a slice. Cheaper than
the pre-merge review feared, and measurably so — the converted graph has **647
initializers, the same set as the shipped one**, so the tensor map is unchanged
and `ExpectedWeightsHeaderShaFor` is untouched. ROCm-only, exactly as the
correction above claimed.

It hides in FOUR places, not one. Missing any silently changes the answer:
normalisation/clipping statistics (context rows only), attention K/V, the
attention scale parameter, and the head slice — which also drops the
`n_thinking_rows` prefix, so converting only the `eval_pos` half returns
`T + n_think` rows.

### Blocker 2 — `SplitToSequence`, already in the shipped graph

`u, v = self.up(x).chunk(2, dim=-1)` in SwiGLU exports as `SplitToSequence` +
`SequenceAt`, and MIGraphX's ONNX parser implements neither. There are 32 and
64 of them respectively — one chunk per layer — **in the graph TabDPT ships
today**, with nothing to do with the train/test split.

So this document's recommendation ("convert tabdpt, it is the cheapest") rested
on an incomplete count. Converting the split was necessary and never
sufficient. Nothing short of compiling on the hardware could have shown it:
ONNX Runtime implements sequence ops, so the CPU and CUDA paths have always
been happy.

Rewritten as two slices; bit-exact for both bias modes.

### Blocker 3 — a MIGraphX code-generation bug

With the parser satisfied, the compiler dies in its own kernel templates:

```
invalid operands to binary expression
('reducer<...>::inner_storage<float, 1, integral_constant<unsigned,1>>' and 'float')
```

four times, then an assertion in `optional<tuning_config>::operator->` and a
core dump.

Isolated by ablation rather than guesswork: **either masked-statistics block
compiles alone; two chained ones do not.** TabDPT's preprocessing chains three
(clip → normalize → clip). No `MIGRAPHX_DISABLE_*` env var avoids it, and four
hand-built minimal graphs reproducing the pattern all compiled fine — the bug
only appears in the chained context, so the smallest repro is the fixture-sized
model, not a toy.

Worked around by expressing the masked sums as `ones[1, T] @ x[T, rest]`: the
same arithmetic, never entering the broken template, and a GEMM is a shape the
GPU prefers anyway. **This workaround is reusable by any model that hits the
same wall**, which is the part most likely to matter beyond TabDPT.

### A misread worth recording

The first end-to-end run showed 53 of 100 rows disagreeing, which reads as a
broken conversion. It is not: **every query row agrees exactly.** All 53 sit in
context rows, where the CPU path returns a single constant — `argmax` of the
zero pad the positional wrapper writes for rows it never computes — and the
masked graph returns real values.

Chasing that found a pre-existing defect unrelated to ROCm: over 80 context
rows, `tabdpt` and `tabpfn-v2-6` return ONE distinct fitted value while
`tabicl-v2` and `mitra` return three. The README calls these "in-context fitted
values, handy for a sanity check". For those models they are not fitted values
at all.

### What this means for the other eight

The recipe is proven but not yet shown to be general: blockers 1 and 3 look
family-wide (every `single_eval_pos` model splits positionally, and any masked
rewrite chains the same statistics), while blocker 2 is TabDPT's own
architecture. Converting a second model is what settles that, and it is worth
doing before anyone commits to the remaining catalog.

## Why the original write-up stopped short

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

## Backend verification status of the re-exported graphs

The fitted-values fix re-exported tabdpt's graphs, so every backend that reads
them has to be re-checked. `GpuGraphKindFor` routes **CUDA and MLX to the same
`ext_graph`** and ROCm to the new `migraphx_graph`, which makes the exposure
wider than "the ROCm spike":

| Backend | Reads | Status |
|---|---|---|
| CPU | `graph_tabdpt_*.onnx` | verified — `test/sql/tabfm_real_models.test`, 36 assertions incl. the fitted-value and regression oracles |
| ROCm | `graph_migraphx_tabdpt_*.onnx` | verified on gfx1201 — `rocm:0`, `ALL_ROW_DISAGREE=0`, `AUTO_OK=true` |
| CUDA | `graph_ext_tabdpt_*.onnx` | verified on an RTX A5000 — `QUERY_DISAGREE=0`, `ALL_ROW_DISAGREE=0`, `FITTED_DISTINCT=3`, regression `MAXDIFF=5.3e-05` |
| MLX | `graph_ext_tabdpt_*.onnx` | **NOT VERIFIED** |

MLX is open, and the CUDA result does not close it. MLX does not execute the
graph with ONNX Runtime; it walks the same file through its own interpreter
(`src/tabfm_mlx_graph.cpp`), so "the ext graph is correct" and "MLX evaluates
the ext graph correctly" are separate claims.

Static analysis narrows the risk but does not settle it. The new graph's op set
is a strict subset of the old one plus a single `Cast` (the removed
`Concat`/`Expand`/`Constant` nodes were the head's zero-padding), `Cast` is in
the interpreter's table, and the output shape became *simpler* —
`[1, rows, 16]` in place of the old
`[1, train + Max(0, rows - Min(rows + 64, train + 64) + 64), 16]`. Nothing
there predicts a numerical difference. That is an argument, not a measurement,
and this document's own history is the reason it is not being recorded as one.

To close it, on the Mac:

```bash
CALL tabfm_download('classification', model := 'tabdpt');
CALL tabfm_download('regression',     model := 'tabdpt');
duckdb -unsigned -c ".read tools/gpu_test/scenarios/mlx_all_models.sql"
```

`TABDPT_FITTED` is the row that matters, and it is new. Every other comparison
in that scenario looks at query rows only — which is precisely the half of the
output the fitted-values fix did **not** change. A query-row check would agree
perfectly while every fitted row disagreed.

## Reproducing the committed artifacts

The MIGraphX graphs in `resources/` were produced by exactly these commands.
Recorded because a committed binary artifact nobody can regenerate is one
nobody can audit.

```bash
cd tools/export_tabdpt

# 1. the masked graph (weight-free). --contract mask is the whole difference:
#    y becomes full-length and the split arrives as a train_size VALUE.
uv run python -m export_tabdpt.cli --task classification --config real \
    --contract mask --out /tmp/mask_out
uv run python -m export_tabdpt.cli --task regression --config real \
    --contract mask --out /tmp/mask_out

# 2. point it at the cached safetensors and make it MIGraphX-parseable.
#    --dynamic is required: the plugin pins shapes per (T, H) bucket at compile
#    time, so a statically pinned graph would serve exactly one bucket.
#    --inline-ok is needed for regression only: bin_centres is a code-generated
#    constant (bar-distribution midpoints), not a checkpoint weight.
cd ../..
W=~/.cache/anofox-tabfm/Layer6__TabDPT@main/model.safetensors
python tools/make_migraphx_graph.py classification \
    --weights $W --graph /tmp/mask_out/graph_mask_tabdpt_classification.onnx \
    --tensor-map resources/tensor_map_tabdpt_classification.json \
    --dynamic --out resources/graph_migraphx_tabdpt_classification.onnx
python tools/make_migraphx_graph.py regression \
    --weights $W --graph /tmp/mask_out/graph_mask_tabdpt_regression.onnx \
    --tensor-map resources/tensor_map_tabdpt_regression.json \
    --dynamic --inline-ok bin_centres \
    --out resources/graph_migraphx_tabdpt_regression.onnx
```

Before trusting any of it, run the gate — it proves it can fail before it
reports a pass:

```bash
cd tools/export_tabdpt && uv run python -m export_tabdpt.mask_parity
```

The plain and `ext` graphs are regenerated by the same CLI without
`--contract mask`, then `tools/make_external_graph.py`.
