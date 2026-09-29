# TabPFN on ROCm — what the masked graph has to reproduce

Status: **plugin binding done; conversion not started.** The attention layer was
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
