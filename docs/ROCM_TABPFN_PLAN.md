# TabPFN on ROCm — what the masked graph has to reproduce

Status: **v2, v2.5, v2.6 converted, gated, and proven end to end on ROCm in both tasks; v3 not started; nothing bundled or released.** The attention layer was
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

Same data, warm calls, 128 rows x 16 features. Agreement is over EVERY row,
context and query. Classification: label disagreements; regression: max |diff|.

| model | task | label / value agreement | CPU warm | ROCm warm | speedup | first-call compile |
|---|---|---|---|---|---|---|
| v2   | cls | 0/30 query, 0/70 context | ~0.50 s | ~0.052 s | ~10x | 177 s |
| v2   | reg | max 9e-6, corr 1.0       | ~1.5 s  | ~0.053 s | ~28x | 213 s |
| v2.5 | cls | 0/30 query, 0/70 context | ~0.41 s | ~0.10 s  | ~4x  | 585 s |
| v2.5 | reg | max 3.7e-3, corr 0.999999| ~0.34 s | ~0.066 s | ~5x  | 329 s |
| v2.6 | cls | 0/30 query, 0/70 context | ~0.35 s | ~0.072 s | ~4.9x | 568 s |
| v2.6 | reg | max 3.4e-2, corr 0.99999 | ~0.48 s | ~0.10 s  | ~4.8x | 533 s |

**The numeric deviation is real and not yet explained.** Labels agree everywhere,
but class PROBABILITIES on v2.6 differ from the CPU by up to 0.05 (mean 0.001),
and regression by up to 0.034 on a target range of 4.8. ORT running the same
masked graph matches PyTorch to 3e-6, so the gap is the GPU. It tracks the
architecture, not the conversion: v2 is essentially exact (9e-6), v2.5 is 4e-3,
v2.6 is 3e-2.

Ruled out by measurement, not by argument:
* **Padding.** Data that exactly fills its bucket (128x64, zero padding) still
  shows max 0.007 / mean 0.0009 on v2.6.
* **MIGraphX fast_math** (which defaults ON and rewrites erf-GELU). A plugin built
  with `set_fast_math(false)`, compiled fresh into its own cache, gives the SAME
  probabilities to six digits.
* **RMSNorm alone.** v2.5 uses LayerNorm like v2 but still deviates by 4e-3.

Not ruled out: depth/accumulation (v2.5 and v2.6 are deeper), the thinking rows,
a MIGraphX fusion. `migraphx-driver verify` against MIGraphX's own reference
implementation would separate kernel numerics from anything model-side.

Two practical notes. The compile cost is per (rows, features) bucket and is large
for the deeper models (330-585 s; v2 is 177-213 s), so users pay it once per
bucket they touch -- `tabfm_gpu_precompile` exists for that. And the regression
fitted values agree with the CPU's to the same tolerance as the query rows.

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
