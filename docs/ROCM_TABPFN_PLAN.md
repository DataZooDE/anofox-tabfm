# TabPFN on ROCm — what the masked graph has to reproduce

Status: **v2, v2.5, v2.6 converted, gated, proven on ROCm in both tasks (identical to the CPU to float noise), and BUNDLED as built-in ROCm graphs. v3 not started. Not yet released: the plugin needs the `n_rows` binding, so the next release must bump `TABFM_PLUGIN_RELEASE_TAG`.** The attention layer was
proven exact in the spike (docs/ROCM_TABDPT_SPIKE.md, tools/export_tabpfn
`tabpfn_mask_patches.py`, ~2x cost). This is the inventory of everything else,
made from reading tabpfn 2.x/3 source rather than from the spike.

## The contract problem, and why the sentinel cannot solve it

The compiled MIGraphX graph receives `x[T,H]`, `y[T]`, `train_size`, `d`, with
`T`/`H` padded up to a shape bucket. TabDPT needed nothing more: all of its
statistics run over train rows. TabPFN's do not.

The obvious idea -- recover the real row count from the `-100` label sentinel --
does not work. The engine already fills every query row's `y` with that sentinel
(`tabfm_preprocess.cpp`: `batch.y.assign(T, kTargetPadSentinel)`, then only train
rows overwritten), and the plugin pads further rows with the same value. Real
query rows and padded rows are indistinguishable, so `y != -100` yields
`train_size` and nothing else. Distinguishing them needs a second sentinel, which
lives in the plugin -- a plugin release either way, plus two magic numbers that
must agree across C++ and Python.

So: an explicit `n_rows` input. `PluginRun` binds it only if the compiled graph
declares it, so tabfm-v1, mitra and tabdpt graphs are untouched.

## Sites that consume the split or the row count (v2.6; others mirror it)

| site | today | masked form needs |
|---|---|---|
| `_impute_nan_and_inf_with_mean` | `nanmean(x[:num_train])` | masked mean over `row < train_size` |
| `standard_scaler.fit` | `x[:num_train]` | masked mean AND std (`torch_nanstd`) |
| `_remove_constant_features` | `(x[1:] == x[0]).all(0)` over ALL rows | compare only rows `< n_rows`; padded zeros must not un-constant a column |
| `_normalize_feature_groups` | same all-rows comparison, `Ri - 1` count | `n_rows`, not the padded `Ri` |
| `_prepare_targets` / `_impute_target_*` | NaN-pads y to `Ri`, mean over `[:num_train]` | y is already full length; mask instead of pad |
| `AddThinkingRows` | `single_eval_pos += num_thinking_rows` | offset the mask, not an int |
| `AlongColumnAttention` | proven in the spike | -- |
| **`AlongRowAttention` (over feature groups)** | attends across all `G` groups | **mask padded groups with `d`** -- NOT in the spike |

The last row is the one the spike did not see. `G` is fixed by the padded width,
so a bucket wider than the real feature count adds all-zero groups that the
column attention would treat as real tokens. The target column is the last
index, so the padded groups sit *between* the real ones and the target and the
mask is not a simple prefix.

## Per-architecture

- **v2 / v2.5 / v2.6**: same shape; v2 lacks thinking rows. v2 and v2.5 already
  take the duplicate-rows route for fitted values (see #51), which doubles the
  sequence length again -- measure before assuming these are worth converting.
- **v3**: inducing points, RoPE, a feature-aggregation stack and 4 slice sites
  instead of 2. Architecturally different enough that it should not be assumed to
  fall out of the same patch.

## Gate

Reuse the tabdpt discipline: full-model parity against upstream in PyTorch,
with negative controls that must be caught before the result is believed
(neutralised row mask, neutralised group mask, `n_rows` replaced by the padded
`T`, statistics over all rows). Upstream zero-initialises several projections,
so a random-init model is blind to attention changes -- wake them, as
`mask_parity.py` does.

## Open cost question

The spike measured ~2x for the attention alone. Whether the converted graph beats
the CPU at all is unmeasured. That number, not the parity result, decides whether
the remaining architectures are worth converting.


## Results so far (v2.6 classification, gfx1201)

**Correctness.** `mask_parity_full` compares upstream's ExportWrapper on unpadded
data against the masked wrapper on the same data padded to a larger bucket, on
every real row. v2, v2.5, v2.6, both tasks, five cases (a real constant column, a
column constant only after imputation, a single query row): worst 3.6e-6. Eleven
negative controls -- every one run on every architecture it applies to -- are
caught before any pass counts. ORT on the exported graph matches the PyTorch
wrapper at shapes other than the export example: worst 3.1e-6.

**On the GPU.** Released v2026.09.26 CPU vs the compiled MIGraphX graph, through
the engine, 100 rows x 8 features (padded to the 128x16 bucket): served by
`rocm:0`, **0/30 query rows and 0/70 context rows disagree**. On separable data,
fitted and query accuracy are both 1.0 on CPU and GPU.

**Speed**, 128 rows x 16 features, warm, same data:

| | wall | CPU time |
|---|---|---|
| CPU | ~350 ms | 5.6 s |
| ROCm | ~72 ms | 0.14 s |

About 4.9x and ~40x less CPU. First call on a bucket compiles for **568 s**.
One shape only; the ratio should improve with size but that is unmeasured.

## All six proven on the GPU (gfx1201, released v2026.09.26 CPU as reference)

Same data, warm calls, 128 rows x 16 features, run sequentially so nothing else
competed for the CPU. Agreement is over EVERY row, context and query.
Classification: label disagreements; regression: max |diff| against the CPU.

| model | task | agreement | CPU warm | ROCm warm | speedup | first-call compile |
|---|---|---|---|---|---|---|
| v2   | cls | 0/30 query, 0/70 context | ~0.45 s | ~0.063 s | ~7x   | 154 s |
| v2   | reg | max 9e-6                 | ~0.41 s | ~0.078 s | ~5x   | 162 s |
| v2.5 | cls | 0/30 query, 0/70 context | ~0.39 s | ~0.11 s  | ~3.5x | 553 s |
| v2.5 | reg | max 2e-6                 | ~0.33 s | ~0.063 s | ~5x   | 329 s |
| v2.6 | cls | 0/30 query, 0/70 context | ~0.39 s | ~0.068 s | ~5.7x | 542 s |
| v2.6 | reg | max 3e-6                 | ~0.48 s | ~0.10 s  | ~4.8x | 534 s |

CPU time drops ~40x (0.14 s vs 5.6 s of CPU per call on v2.6). The first call
on each (rows, features) bucket pays the compile, so a user pays it once per
bucket they touch; `tabfm_gpu_precompile` moves it off the query path.

An earlier table in this file showed v2 regression at ~1.5 s CPU / ~28x. That
was measured while a concurrent experiment was loading the CPU and was wrong.

## The numeric gap, and what it was

Before the fix below, v2.5 and v2.6 drifted on the GPU: class probabilities by up
to 0.05, regression by up to 0.034, raw logits by 1-3% relative (logit scale ~75),
while v2 was exact. Labels agreed everywhere, which is why it took a look at the
logits to see it.

**Cause: one integer floor-division, miscomputed by MIGraphX's GPU target.** The
count of real feature groups, `arange(G) < (n_sel + fpg - 1) // fpg`. For n=8,
fpg=3 the GPU marks FOUR groups real instead of three, so a padded all-zero group
was attended to as a real token in every row-attention layer. ORT and MIGraphX's
own `ref` target both get it right, and materialising the quotient as a separate
output also gives the right value, which is what made it hard to see. Reproduced
in ten lines (`j < (n+2)//3` wrong; `j*3 < n` and a float compare both right;
rank-0 vs rank-1 operand makes no difference).

Fix: the equivalent division-free form, `arange(G) * fpg < n_sel`
(j < ceil(n/f) <=> j*f < n for integers). Logit error against ORT, same input:

| | context max | query max |
|---|---|---|
| before | 0.954 | 2.51 |
| after | 5.7e-5 | 2.0e-3 |
| MIGraphX `ref` target (the floor) | 3.8e-5 | 1.8e-3 |

The PyTorch parity gate could not see this, because it never runs on a GPU, so
there is now a structural guard (`tests/test_mask_export_isolated.py`): no integer
`Div` that depends on runtime data may appear in a masked graph. Shape arithmetic
is exempt -- MIGraphX folds it once the bucket is pinned.

How it was found, in the order things were ruled out, because each cheaper
explanation was wrong: not padding (zero-padded data still drifted); not
MIGraphX `fast_math` (a plugin built with it off gave identical probabilities to
six digits); not RMSNorm alone (v2.5 has LayerNorm and drifted); not MLIR, the
GEMM provider, or fast softmax (each disabled, identical). Then a three-way run
(ORT / MIGraphX `ref` / MIGraphX GPU) showed the graph survived MIGraphX intact
and the fault was GPU code generation; exposing the 72 RMSNorm outputs showed the
drift at the FIRST one; exposing every float tensor before it found the first
numeric divergence was the group mask at relative error 1.0; a subgraph extract
compiled in 6 seconds and reproduced it. Worth reporting upstream.

## What bundling surfaced

Bundling is where the "proven on a GPU" claim met the real code paths, and it
found two more defects, both silent.

**1. `tabpfn-v2-5-real` got `tabpfn-v2-5`'s answers on ROCm.** The compiled-program
cache was keyed by the graph's bytes. A graph is weight-free and the compiled
program bakes the weights in, so two models with one byte-identical graph and
different checkpoints shared a program. Measured: CPU real vs CPU regular 0.1148,
GPU real vs GPU regular 0.0, GPU real vs CPU real 0.1148. It was noticed as ONE
flipped label in 100 with a 0.115 probability gap, against ~1e-3 for every other
model; the graph-level check could not see it (GPU vs ORT on the real weights
agreed to 3.5e-5, because that compared a graph to itself). The key now carries a
weights fingerprint (size + 64 sampled 4 KiB blocks); after the fix GPU real vs CPU
real is 5e-6. Cost: one recompile per bucket after upgrading.

**2. `tabfm_backends()` judged servability against the wrong weights file.** The
engine prefers the converted sibling `model.safetensors` over a declared `.ckpt`;
`tabfm_backends()` preferred the declared file. With both on disk it compared the
bundled graph's header against a pickle ("does not match"); with only the
converted file it said "not downloaded". No checkpoint-based model had a GPU graph
to be misjudged until this.

Also: an OLD plugin refusing a new graph now says to update (it used to surface
MIGraphX's bare `Parameter not found: n_rows`), and CI now runs
`tools/check_migraphx_int_div.py` over every committed MIGraphX graph.

## Things found along the way that will bite the next person

* **An old plugin refuses a new graph, loudly.** With the published plugin (no
  `n_rows` binding) the compiled graph fails with `Parameter not found: n_rows`.
  That is a hard error, not a silent fallback -- but it means graph and plugin
  must ship together, and the release that adds the graph must bump
  `TABFM_PLUGIN_RELEASE_TAG`.
* **A registered model reads the staged file in the cache directory, not the
  bundled graph.** A stale `graph_ext_*.onnx` staged by an older binary made the
  CPU side report chance-level fitted accuracy and look like a GPU mismatch. The
  GPU was right; the reference was stale. Compare against a fresh copy.
* **Verify the binary's version before trusting a baseline.** `build/release/duckdb`
  in a shared checkout was built from a branch that predated the fix it was
  being used to check (`extension_version` = c98e5fe). Ask it.
* MIGraphX rejects a Reshape to empty dims (`reshape(())`) as zero elements;
  index scalars with `[0]` instead.

## Not done

v3 (a substantially larger conversion: inducing points, ICL blocks, many-class
decoder, row-count-dependent softmax scaling); regression and v2/v2.5 end to
end on ROCm (their fitted-values route doubles the sequence again -- unmeasured);
bundling the graphs as resources with `BundledGpuGraphId`/servability; more shape
buckets and larger shapes; the plugin release.
