# Context cache, next steps: design and tabicl-v2 spike (#37), 2026-11-01

**Status: design + spike.** The C++ change in this round is small and covered by a test: the split path
uses a prepare graph's `context_logits` output when the graph has one. Everything else here is a plan.
Background: [KV_CACHE_DESIGN.md](KV_CACHE_DESIGN.md), which records what shipped in #38 and #40.

## Summary

- **The cost is real.** On `tabicl-v2`, re-encoding the labelled context is most of a `tabfm_classify`
  call: 71-100% in the reporter's measurements on #37, and 83-98% of the CPU time at a 375- or 1000-row
  context in the spike below.
- **A split already exists, but leaves most of the saving on the table.** `export_tabicl --split-context`
  (#38) and `SET anofox_tabfm_context_cache = true` (#40) split the model in two. The query half still
  re-runs the 12-layer ICL transformer over all S support rows on every call. No built-in model ships
  the pair, so in practice nobody can use it yet.
- **The spike measures a better contract** (the "kv" contract):
  - prepare also emits each ICL layer's support keys/values, plus the context rows' own logits;
  - query runs the query rows only, and takes no `y`;
  - TabICL's own `cached_kv` path in `MultiheadAttentionBlock` does the work, with no patches.
- **The kv contract is exact on the real `tabicl-classifier-v2` weights:**
  - zero label differences, max |Δlogit| 1.2e-05 over 7 tables;
  - the context-row fitted values match as well, where today's pair moves them by up to 10 logits;
  - a warm call is **5.7-51x cheaper in CPU time** than the combined graph, against 4.5-8x for today's pair;
  - a cold call costs the same as the combined graph (0.91-1.04x), where today's pair costs 1.4-1.8x.
- **Recommendation: keep the cache transparent. Make it exact, bound it by bytes, and turn it on by
  default** for any model whose pair is exact. Do not add a context-handle API. Details in
  [Recommendation](#recommendation).
- **Every other built-in model splits exactly** at a model-specific cut. Three of them rely on an engine
  invariant to do so. None needs to change what it computes. See [Per model](#per-model).

## What exists today

| piece | where | what it does |
|---|---|---|
| split export | `tools/export_tabicl/src/export_tabicl/split.py`, `--split-context` (#38) | `prepare(x_S, y) -> hidden_0..2, support_reps`; `query(x_Q, y, state) -> logits` |
| engine | `SplitOrtBackend` in `src/tabfm_engine.cpp` (#40) | Finds `graph_{prepare,query}_<stem>_<task>.onnx` next to the combined graph. Caches **one** context, compared verbatim (`memcmp` of x and y), never hashed. On a miss: prepare, then query over S for the fitted values, then query over Q. On a hit: query over Q. CPU only (ORT); MIGraphX, MLX and CUDA plugin devices refuse with the setting on |
| setting | `anofox_tabfm_context_cache`, default `false` | Opt-in, because the old query half has no label path, so a context row's fitted value moves |
| shipped graphs | none | Only the CI fixture (`test/fixtures/tabicl`) carries a pair. `resources/` has none for `tabicl-v2` |

Three gaps in what exists:

1. **The query graph is not O(Q).** `QueryWrapper` concatenates `support_reps` with the query rows and
   runs `icl_predictor` over S+Q rows on every call (`split.py`). The ICL is 12 layers at width 512, so
   at S=1000 it dominates the warm call. `tools/export_tabicl/README.md` claimed the support rows never
   pass through the ICL stage again; that was wrong, and this PR corrects it.
2. **Fitted values change, and a cold call costs extra.** The engine gets context-row values from a
   query pass over the S rows. That pass costs as much as prepare itself, and it scores each row
   without its own label. On random labels at S=1000 this flips 999 of 1000 fitted labels (spike table
   below). That is why the setting is opt-in.
3. **The regression pair does not match the shipped regression graph.** The shipped graph z-scores `y`
   in-graph and returns the mean of the 999 quantiles, inverse-scaled (`ExportWrapper`). The split
   wrappers feed raw `y` and return raw quantiles. The pair's parity check compares against `model()`,
   not against the wrapper, so it passes anyway. No regression pair ships (the fixture's is
   classification only), so nothing is wrong in production, but the export flag is wrong for
   `--task regression`.

## The spike

`uv run spike_context_cache --graphs DIR --out DIR` lives in
`tools/export_tabicl/src/export_tabicl/context_spike.py`. It is a new module, and the shipped graphs
and `--split-context` are unchanged.

**What it compares.** Three contracts, each against the **shipped** combined graph
(`resources/graph_tabicl_classification.onnx`):

| contract | prepare outputs | query inputs | the context rows' fitted values |
|---|---|---|---|
| `pair` (today) | `hidden_0..2`, `support_reps` | `x`, `y`, state | a query pass over S, without labels |
| `fitted` | as `pair`, plus `context_logits` | as `pair` | from prepare, with labels, as in the combined graph |
| **`kv`** | `hidden_0..2`, `icl_k_0..11`, `icl_v_0..11`, `context_logits` | `x`, state (**no `y`**) | from prepare |

**How it runs.**
- All graphs are weight-free.
- The weights are **injected** from the safetensors file through each graph's tensor map
  (`add_external_initializers`), as the extension does.
- Runs use ONNX Runtime 1.27 on the CPU with 4 intra-op threads and spinning off.
- Weights: `~/.cache/anofox-tabfm/jingang__TabICL@main/classification/model.safetensors`
  (`tabicl-classifier-v2-20260212`).

### Parity

Each cell is max |Δlogit| over the real classes, then the number of argmax (label) differences.
"Today" is the shipped pair; the reference is the combined graph.

| table | S | Q | H | C | acc | test rows, today | test rows, kv | kv re-batched | context rows, today | context rows, fitted | context rows, kv | control |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CI fixture (random init) | 12 | 8 | 5 | 3 | - | 0 / 0 | 0 / 0 | 0 | 0.0028 / 0 of 12 | 0 / 0 | 0 / 0 | 2.2e-08 / 0 |
| breast_cancer | 64 | 16 | 30 | 2 | 1.000 | 0 / 0 | 5.2e-06 / 0 | 0 | 4.6 / 0 of 64 | 0 / 0 | 4.8e-06 / 0 | 6.4 / 5 |
| breast_cancer | 375 | 128 | 30 | 2 | 0.977 | 0 / 0 | 6.7e-06 / 0 | 0 | 4.9 / 3 of 375 | 0 / 0 | 6.7e-06 / 0 | 9.3 / 45 |
| digits | 64 | 16 | 64 | 10 | 0.938 | 0 / 0 | 5.1e-06 / 0 | 0 | 8.7 / 8 of 64 | 0 / 0 | 9.1e-06 / 0 | 7.7 / 12 |
| digits | 375 | 128 | 64 | 10 | 0.969 | 0 / 0 | 1.1e-05 / 0 | 0 | 10 / 8 of 375 | 0 / 0 | 1.2e-05 / 0 | 12 / 119 |
| synthetic, random labels | 64 | 16 | 100 | 4 | - | 0 / 0 | 5.7e-06 / 0 | 0 | 6.3 / 0 of 64 | 0 / 0 | 4.8e-06 / 0 | 2.3 / 12 |
| synthetic, random labels | 1000 | 128 | 100 | 4 | - | 0 / 0 | 3.3e-06 / 0 | 0 | 2.1 / 999 of 1000 | 0 / 0 | 4.8e-06 / 0 | 0.63 / 97 |

How to read the table:

- **The control is what makes the identity results mean something.** It prepares a state from
  permuted labels and scores the same query rows against it. On real weights this moves the answers by
  0.6-12 logits. On the random-init fixture it moves them by 2e-08, which is why the fixture proves only
  that the path ran, never parity. The reporter found the same thing on #37.
- **"Today" test rows come out exactly equal (0.0).** Both the combined graph and the pair slice the
  ICL keys to the support rows (`q[..., :train_size, :]`) rather than masking them, so the arithmetic
  is identical.
- **kv differs by about 1e-05.** It takes the keys from a `need_kv` projection of the support rows, and
  the reduction order differs. That is fp32 noise: no labels change.
- **"kv re-batched" splits the query rows in two calls** against one state. The answers are
  bit-identical, so a query row does not depend on which other query rows share its call.

### Timings (graph level)

**The measurement.** S support rows by Q query rows, at H = 100 features and 4 classes. Each value is
the median of 9 runs; the variants are interleaved round by round. The box is shared (load average
30-60 on 32 cores during the run). **CPU seconds** (process CPU, all ORT threads, no spinning) are the
stable measure. Wall seconds follow in a second table for completeness.

| S | Q | combined | pair cold (engine today) | pair warm | kv cold | kv warm | warm: today | warm: kv | cold: today | cold: kv | kv state |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 16 | 1.87 | 2.99 | 0.44 | 1.94 | 0.32 | 4.3x | **5.8x** | 1.6x | 1.03x | 24 MB |
| 64 | 128 | 2.62 | 3.31 | 1.31 | 2.54 | 1.18 | 2.0x | **2.2x** | 1.3x | 0.97x | 24 MB |
| 375 | 16 | 5.10 | 9.03 | 0.65 | 5.09 | 0.25 | 7.8x | **20x** | 1.8x | 1.00x | 39 MB |
| 375 | 128 | 8.08 | 11.39 | 1.80 | 7.37 | 1.41 | 4.5x | **5.7x** | 1.4x | 0.91x | 39 MB |
| 1000 | 16 | 13.85 | 25.24 | 1.73 | 13.58 | 0.27 | 8.0x | **51x** | 1.8x | 0.98x | 70 MB |
| 1000 | 128 | 16.45 | 28.86 | 2.88 | 16.31 | 1.36 | 5.7x | **12x** | 1.8x | 0.99x | 70 MB |

How to read the columns:

- *cold* is the first call against a new context.
  - Today's engine runs prepare, then a query pass over S, then a query pass over Q.
  - kv runs prepare (which also emits `context_logits`), then a query pass over Q.
- *warm* is every later call against the same context: only the query graph runs.
- The *warm* and *cold* ratios compare against the combined graph. For *cold*, above 1x means the
  first call is slower than the combined graph.
- *kv state* is the resident cost of one cached context. Today's pair holds 21-23 MB: it is dominated
  by the column-ISAB `hidden_*`, which scale with H, not S. kv adds 2 × 12 layers × S × 512 floats of
  ICL keys/values.

Wall seconds, same runs (noisier):

| S | Q | combined | pair cold | pair warm | kv cold | kv warm |
|---|---|---|---|---|---|---|
| 64 | 16 | 0.78 | 1.10 | 0.21 | 0.62 | 0.12 |
| 64 | 128 | 0.81 | 1.04 | 0.41 | 0.77 | 0.36 |
| 375 | 16 | 1.49 | 2.63 | 0.21 | 1.47 | 0.08 |
| 375 | 128 | 5.00 | 7.82 | 0.66 | 6.35 | 1.17 |
| 1000 | 16 | 4.65 | 7.26 | 0.51 | 3.89 | 0.10 |
| 1000 | 128 | 4.93 | 8.31 | 0.84 | 4.89 | 0.43 |

**Fixed versus marginal cost** (CPU seconds, kv, read off the table above):

- A call against a cached context costs about **0.1 s fixed + 10 ms per query row** at S=375-1000:
  the fixed term barely moves with S.
- The combined graph pays a fixed cost that grows with S on **every** call: about 1.8 s at S=64,
  4.7 s at S=375 and 13.5 s at S=1000. Its per-row cost also grows with S, from 7 ms to 23-27 ms.
- Today's pair removes most of that fixed cost but keeps an S-proportional ICL term: its warm call at
  Q=16 is 0.44 s at S=64 and 1.7 s at S=1000.

### Through the extension

The same graphs were run through `tabfm_classify`. The kv pair needs no engine change to run: the
released engine already serves it, and gives the same test-row results. This branch's build adds
`context_logits`, and the table below is from that build.

**Setup.**
- A model registered with `tabfm_register_model`. Its `base_dir` holds the shipped combined graph and
  map, plus a pair: either today's pair, or the kv pair renamed to `graph_{prepare,query}_tabicl_*`.
  The engine finds both by name, and the kv query graph takes no `y`, which the engine already allows.
- `SET threads = 4; SET anofox_tabfm_threads = 4`, H = 100, 4 classes, single-relation form, so each
  call returns S + Q rows. Medians of 5 runs.
- *off* is the combined graph.
- *cold* changes one feature of one support row before each call, which forces a re-encode.
- *warm* repeats the call unchanged.

| S | Q | contract | off (combined) | cold | warm | warm vs off, CPU | cold vs off, CPU | context-row labels changed |
|---|---|---|---|---|---|---|---|---|
| 64 | 16 | today's pair | 0.40 s / 1.6 | 0.63 s / 2.7 | 0.11 s / 0.45 | **3.6x** | 1.64x | 0 of 64 |
| 64 | 16 | kv + context_logits | 1.87 s / 6.0 | 2.10 s / 6.2 | 0.47 s / 1.07 | **5.6x** | 1.03x | 0 of 64 |
| 64 | 128 | kv + context_logits | 3.06 s / 9.1 | 3.42 s / 9.5 | 1.85 s / 4.53 | **2.0x** | 1.04x | 0 of 64 |
| 375 | 16 | today's pair | 1.45 s / 7.1 | 2.43 s / 12.1 | 0.21 s / 0.81 | **8.8x** | 1.69x | 75 of 375 |
| 375 | 16 | kv + context_logits | 5.92 s / 17.2 | 8.03 s / 19.8 | 0.65 s / 1.48 | **11.6x** | 1.15x | 0 of 375 |
| 375 | 128 | kv + context_logits | 4.04 s / 15.3 | 1.86 s / 9.2 | 0.36 s / 1.53 | **10.0x** | 0.60x | 0 of 375 |
| 1000 | 16 | today's pair | 3.96 s / 18.9 | 7.52 s / 36.3 | 0.52 s / 2.43 | **7.8x** | 1.92x | 183 of 1000 |
| 1000 | 16 | kv + context_logits | 3.78 s / 18.1 | 4.03 s / 15.6 | 0.17 s / 0.69 | **26.4x** | 0.86x | 0 of 1000 |
| 1000 | 128 | kv + context_logits | 4.34 s / 21.1 | 4.07 s / 19.5 | 0.41 s / 1.81 | **11.7x** | 0.92x | 0 of 1000 |

Each cell is wall seconds / CPU seconds (user + sys) of the whole `tabfm_classify` statement. The box was under heavy
shared load, and the extension's ORT threads spin, so these are noisier than the graph-level table: read the ratios, not
the absolute times. All kv rows ran first, then today's pair, on the same binary. Single values swing
by up to 4x between runs: `off` at S=64, Q=16 was 0.40 s in one run and 1.87 s in the other. The cold
ratio of 0.60x at S=375, Q=128 is such a swing, not a real saving. The graph-level table is the clean
measurement; this one shows that the extension path delivers the same order of saving, and the same
exactness.

**Checks through the extension.** Test rows: 0 label differences against the combined graph for both
pairs (|Δscore| 0 for today's pair, at most 3.7e-06 for kv). Context rows: with this branch's engine the
kv pair's `context_logits` makes them match too (|Δscore| at most 3.1e-06, 0 labels changed), while today's
pair changes 75 of 375 and 183 of 1000 fitted labels on this synthetic table. A custom registration of the
shipped graph also matches the built-in `tabicl-v2` exactly, so the setup measures the real model.

**What the extension adds on top of the graph.** A warm call through `tabfm_classify` costs more than the
query graph alone: at S=1000, Q=16 about 0.2-0.6 s wall against 0.1 s. The rest is O(S) work that does not
go away with the cache: reading and preprocessing the support rows, comparing them with the cached key,
and emitting S fitted rows with their `proba` maps in the single-relation form. Once the model cost is
cached, that is the next thing to measure (Phase 2).

## Recommendation

**Transparent caching, made exact, bounded by bytes, and on by default for models whose pair is exact.
No new SQL function.**

The options considered:

| option | surface | for | against |
|---|---|---|---|
| A. transparent cache (today's design, extended) | none: `tabfm_classify` as it is, plus settings | Every existing query benefits, including the reporter's harness, chunked scoring, cross-validation folds and the per-segment `UNION ALL` pattern. The key is the exact preprocessed tensors, so a hit cannot be wrong. Nothing for the user to keep alive or invalidate | Every call still reads and preprocesses the support rows (milliseconds against seconds). Memory is spent without being asked for, unless it is bounded |
| B. prepared-context handle: `CALL tabfm_fit('ctx', 'history', 'churned', model := ...)`, then `tabfm_classify(test := 'prospects', context := 'ctx')` | 2 new functions, 1 new argument, plus `tabfm_contexts()` / `tabfm_drop_context()` | Explicit lifetime and memory. Skips re-reading the support table | The user has to manage a handle. A snapshot of a table that has since changed returns stale answers, silently, which is the failure the verbatim key exists to rule out. It does not compose with `tabfm_cross_validate` or the `_by` macros without a parallel API. The engine still needs everything in A |
| C. transparent cache keyed by a hash of rows + model + preprocessing | as A | Cheaper lookup with many entries | A collision answers from someone else's training data (KV_CACHE_DESIGN.md). Acceptable only as an index in front of a verbatim compare, which makes it A |

**Why A.**
- B's one real advantage, skipping the read and preprocessing of the support rows, is worth
  milliseconds. Its cost is a second, stateful API with a stale-snapshot failure mode.
- A is exact by construction, because the key is the very input the graph would have seen.
- With the kv contract, A has no downside on the first call: cold costs the same as the combined graph.

**Concretely:**

1. **Exact pairs only.** A pair is *exact* when its prepare graph emits `context_logits`, so every
   output column equals the combined graph's. For exact pairs, `anofox_tabfm_context_cache` becomes
   default `true` and remains as a kill switch. A pair without `context_logits` keeps today's opt-in
   behaviour.
2. **Bound by bytes, not by entries.** `anofox_tabfm_context_cache_memory` sets the bound, with a
   default such as `'1GB'`. Entries are kept in LRU order inside each backend, where they share the
   backend's lifetime and per-device lock.
   - Each entry costs its state plus the verbatim key.
   - A context larger than the whole budget is not cached; the call is still exact.
   - `tabfm_unload()` and session eviction drop the entries with the session, as they do today.
   - `tabfm_models()` gains a `context_bytes` column.
   - `anofox_tabfm_max_memory` already refuses a call above the process ceiling. Cached contexts count
     toward resident memory, so the refusal message should name `anofox_tabfm_context_cache_memory` as
     a remedy.
3. **Lookup.** Use a 64-bit hash of `(H, S, x, y)` as an index, then `memcmp` before accepting a hit,
   so the cache stays verbatim.
   - Lookup is scoped per backend, so per model, task, revision, device and precision, as the session
     already is.
   - The preprocessing is fitted on the support rows only (`tabfm_preprocess.cpp`: the imputation
     means, the train-constant filter and the scaler loop over `n_train`). So the same support set
     gives the same tensors whatever query rows come with it. That property is what makes a transparent
     key hit across query batches, and it needs a regression test (Tests, item 5).
4. **Query chunking inside one call.** Add `opts := MAP{'query_batch': 'N'}`: one call scores its test
   rows N at a time against the context it encoded once.
   - This is the reporter's second point on #37. `--test-chunk` is the only way to bound peak memory,
     and today it costs one full support re-encode per chunk.
   - With a cached context, chunking becomes nearly free, and it no longer needs one SQL call per chunk.
5. **Row order.** DuckDB may hand the support rows over in a different order between calls. A
   different order is a different key, so that call misses: safe, but slower.
   - Sorting the support rows into a canonical order before encoding would make those calls hit. Do it
     only if a model's answer is provably order-invariant, which holds for TabICL (attention over the
     rows, no positional term).
   - Class codes are assigned in order of first appearance, so the sort has to come before the label
     encoding.
   - Phase 3, measured first.

## Graph contract

Two graphs, found by name next to the combined graph. Discovery is unchanged:
`graph_prepare_<stem>_<task>.onnx` and `graph_query_<stem>_<task>.onnx`, each with its
`tensor_map_*.json`.

```
prepare(x [1,S,H] f32, y [1,S] f32)
    -> state tensors, f32, named        # any names except x, y, logits, context_logits
    -> context_logits [1,S,C] f32       # optional; present => the pair is exact
query(x [1,Q,H] f32, <state tensors by name> [, y [1,S] f32])
    -> logits [1,Q,C] f32
```

- **Tensors travel by name.** The engine asks the query graph which state it reads (its non-`x`/`y`
  inputs) and requests exactly those from prepare. Extra outputs are ignored, such as the `val_*` left
  by the ScatterND pinning. The kv query graph takes no `y`. The engine already handles that.
- **State shapes may depend on S and H only, never on Q.** The batch axis is 1.
- **The query graph must be row-independent:** `query(x_a ++ x_b) == query(x_a) ++ query(x_b)`. The
  spike's "re-batched" column checks this.
- **Regression pairs** carry the target scaler: prepare emits `y_mean` and `y_std` as state, and query
  returns the in-graph point estimate (the mean of the quantiles, inverse-scaled), so its output matches
  the combined graph's `[1,Q,1]`.
- **Precision.** The state is cached as f32 in host memory. A GPU backend may keep a device copy; see
  [GPU plugins](#gpu-plugins).

The kv pair for TabICL is about 30 lines of wrapper (`PrepareKVWrapper` and `QueryKVWrapper` in
`context_spike.py`) on top of upstream modules, with no new monkeypatch. TabICL's own `cached_kv` path
handles SSMax with the right key count: `src_len` comes from the cached K.

## Per model

**The engine invariant.** The engine drops every column that is constant on the **train** rows, for
every profile (`tabfm_preprocess.cpp`, UniqueFeatureFilter). So any in-graph "is this column constant?"
test over all rows gives the same answer on support rows alone. The models below that rely on this are
marked †.

**What "state" scales with.** S is support rows, L is layers, d is width, G is feature groups or
columns. O(S·L·d) is a single-axis ICL. O(S·G·L·d) is per-column 2D attention, which is the largest.

| model | where the cut goes | state crossing the boundary | exact? | notes |
|---|---|---|---|---|
| **tabicl-v2** | column-ISAB `hidden` per block; then per-ICL-layer K/V (kv contract) | O(H) ISAB state + O(12·S·512·2) | **yes, measured here** | Ship kv. Regression needs the scaler in the pair. Upstream has `kv_cache.py` |
| causilo | upstream `ModelRunner.build_cache` / `predict_cached` | column-latent K/V (O(1) in S) + prediction K/V O(L·S·d) + normalisation stats | yes | In-graph stats are train-prefix only. Upstream already has the API: second candidate |
| limix-2m | per smf layer, the support K/V of the sample-attention module, per feature group | O(L·S·G·d) | yes † | ValidFeatureEncoder tests constancy over all rows. No upstream cache API |
| nori-30m | as LimiX (fork): per layer, support K/V per group | O(28·S·G·d) | yes | Upstream supports a frozen valid-feature count |
| orion-bix | as TabICL v1: ISAB `hidden` + ICL K/V | as tabicl | yes | Same skip-mask shape trap as `split.py` |
| orion-msp | as orion-bix (the block-sparse mask is within a row; Perceiver memory is off) | as tabicl | yes | |
| tabpfn-v2 / v2.5 / v2.5-real / v2.6 | per layer, first-head K/V of train (+64 thinking) rows | O(L·(S+64)·G·d_head) + feature statistics | yes † | Constant-feature and feature-group masks are fitted over all rows. Upstream: `TabPFNV2Cache` / `TabPFNV2p5Cache` |
| tabpfn-v3 | ICL K/V + per-column inducing hidden + scaler + train embeddings for the retrieval decoder | O(24·S·512) + O(H·128·E) | yes | Upstream: `TabPFNV3Cache` |
| tabpfn-v3.5 / v3.5-fast | as v3, plus the ECDF ruler `(3, H, min(8192,S))` | O(L·S·1024) | yes | Upstream: `TabPFNV3p5Cache`. Classification fitted values need prepare to emit the train embeddings' logits |
| tabdpt / tabdpt-v1.3 | per layer K/V over context (+ thinking) rows, decoder-like | O(32·S·512·2) + normalisation stats | yes | The attention temperature depends on S, which the query half knows from K |
| mitra / mitra-v2 | per layer, support K/V per column + in-graph quantile/moment tables | O(L·S·(F+1)·d) + O(999·F) | yes | v2 regression bucketizes `y` in-graph: that goes in prepare |
| tabfm-v1 / v1.1 | two column ISABs (`hidden`) + per-ICL-layer K/V | O(L·S·2048) (v1), O(L·S·1024) (v1.1) | yes | The largest single-axis state: budget for it |
| kumo-tabular-* | upstream `sdm.cache.Cache` record/replay: column-block K/V, ICL K/V (GQA), impute means, class columns | O(C·S·D) per column block + O(L·S·kv·hd) | yes | Upstream already has the API |
| exaone-tabular | per CAST layer: feature summary + single-head K/V of [summary + support] per column | O(L·G·(S+32)·hw) | yes † | `retained = (src[1:] != src[0]).any(0)` runs over all rows |

**No model needs to change what it computes.** No exported graph has test-to-test attention, and none
has a statistic over train and test rows that is not neutralised by the train-constant filter. The cut
is model-specific every time, and upstream already ships one for causilo, kumo, every TabPFN line and
TabICL's ICL. The order below follows payoff and how much of the cut upstream already provides: tabicl
first, then causilo, tabpfn-v3/v3.5, kumo, tabdpt, mitra, orion, tabfm, and last the † models (whose
prepare graph should compute the all-row tests on support only, or assert the invariant).

The table comes from reading each export tool and its upstream package. The tabicl row is measured; the
others are not, so each model's pair needs its own parity check (Tests, item 1) before it ships.

## GPU plugins

CPU first. Today the setting refuses every non-ORT device (`tabfm_engine.cpp`: "device '%s' cannot
serve this model while anofox_tabfm_context_cache is ..."). Under the recommendation it would fall back
to the combined graph instead, because the cache becomes default-on and must not break GPU calls.

- **CUDA (ORT CUDA EP).** The same pair runs as is. Phase 1 copies the cached state host to device per
  call, about 70 MB at S=1000, which is a few ms over PCIe against a forward pass of 0.1 s or more.
  Phase 2 keeps the state device-resident with `Ort::IoBinding`, under the per-device lock that already
  guards the backend.
- **ROCm (MIGraphX plugin).** It compiles per fixed shape (`.mxr` per bucket). The query graph's shape
  depends on S (the K/V length) as well as on Q and H, so it needs the masked contract that tabdpt and
  the TabPFN family already use (pad S to a bucket and pass a key mask; ROCM_SINGLE_EVAL_POS.md). That
  multiplies the buckets by the S-buckets. Later phase, measure compile cost first.
- **MLX.** The plugin interprets the ONNX graph op by op (`tabfm_mlx_graph.cpp`). The kv graphs use the
  same ops as the combined graph, so they should run once the engine passes named state through the
  plugin ABI. That ABI change is the work.

## Tests

1. **Parity per model and pair**, in the export tool. Against the **shipped combined graph**, on **real
   weights**:
   - test-row labels identical, |Δlogit| ≤ 1e-4;
   - context-row `context_logits` the same;
   - a permuted-label control must move the answers, or the check is void.

   `context_spike.py` is the template. The CI fixture can prove only that the path ran: it is
   near-constant.
2. **Re-batching**: `query(a ++ b) == query(a) ++ query(b)` bit-for-bit.
3. **Engine (sqllogictest)**:
   - every row equals the combined graph's when the pair is exact (`test/sql/tabfm_context_cache_kv.test`,
     added here);
   - a hit gives the identical answer;
   - a changed row, width or label forces a miss;
   - unload drops the entries;
   - the setting off restores the combined graph exactly (`test/sql/tabfm_context_cache.test` already
     covers this for the old pair).
4. **Memory bound**: entries beyond `anofox_tabfm_context_cache_memory` are evicted oldest first; a
   context larger than the bound is not cached and the call is still correct; `tabfm_models()` reports
   the bytes.
5. **Key stability**: one support set with two different query sets hits the cache, so the
   preprocessing is query-independent (explicit train/test form and single-relation form).
6. **Concurrency**: two connections, one on device `auto` and one on `cpu`, against one backend
   (the #42 lock-keying hazard noted in `SplitOrtBackend`).
7. **Real-weights test** (`test/sql/real_*`, skipped without the weights): `tabicl-v2` with the cache
   on and off, every row equal within 1e-5.

## Phases

| phase | scope | done when |
|---|---|---|
| 0 (this PR) | This document. The spike module. The engine reads `context_logits` from prepare when present (prototype, behind the existing off-by-default setting), with a kv-contract CI fixture and its test | spike numbers in this doc; test green |
| 1 | `export_tabicl --split-context` emits the **kv** pair (classification), and the regression pair with the target scaler. Bundle both for `tabicl-v2` in `resources/`. Real-weights parity in CI's real-model job | `tabicl-v2` ships an exact pair |
| 2 | `anofox_tabfm_context_cache_memory`, LRU over several contexts, `context_bytes` in `tabfm_models()`. Default the setting to `true` for exact pairs. Non-ORT devices fall back to the combined graph instead of erroring | the reporter's harness runs unchanged and gets the warm-call numbers |
| 3 | `opts.query_batch`: query chunking inside one call. Canonical row order, if measured worthwhile | chunked scoring costs one encode per call, not one per chunk |
| 4 | Further models, in the order in [Per model](#per-model), each with its own parity row | per model |
| 5 | CUDA device-resident state (IoBinding); ROCm masked S-buckets; MLX named-state ABI | per backend |

## Reproduce

```bash
cd tools/export_tabicl && uv sync
# today's pair, at the real architecture (random weights, weight-free graphs)
uv run export_tabicl --task classification --config real --out /tmp/ctx/graphs --split-context
cp ../../resources/graph_tabicl_classification.onnx ../../resources/tensor_map_tabicl_classification.json /tmp/ctx/graphs/
# fitted + kv pairs (real and fixture scale), parity, timings
uv run spike_context_cache --graphs /tmp/ctx/graphs --out /tmp/ctx/spike --threads 4 --reps 9
```

The extension half used a `tabfm_register_model` registration whose `base_dir` holds the shipped
combined graph plus a pair, and timed `tabfm_classify` with `.timer on`.
