# Context-cache design (perf plan #7) — 2026-07-04

The highest-ceiling optimization: each predict currently re-encodes the whole
train context. For repeated scoring against the **same** context (interactive
exploration, streaming new rows), caching the encoded context and running only
the query rows would drop latency from `O(context+query)` to `O(query)` — a
big-context predict from ~0.14 s to well under 0.02 s. This is a dedicated
effort (export + backend + cache), scoped here so it can run as its own goal.

## Feasibility — YES, and the cache is small (architecture-grounded)

TabFM (`vendor/tabfm/tabfm/src/pytorch/model.py`) is a **Set Transformer** whose
blocks are **Induced Self-Attention Blocks** (ISAB):
`mab1(inducing_points, src, src)` then `mab2(src, hidden, hidden)`, with a
`train_size` mask (`tm = arange(t) < train_size`). It is NOT a causal decoder,
so this is *not* a token KV-cache — it's a **context-state cache**.

Two facts make it work:
1. **Query rows are mutually independent** — proven empirically in
   `test/sql/tabfm_cobatch.test` (a row scored alone == co-batched, same label +
   score). So a query row's output does not depend on which other query rows are
   present.
2. That independence implies the **inducing-point / context representations are
   computed from the context (train) rows only** (masked by `train_size`), i.e.
   independent of the query set. Therefore the per-layer context state can be
   computed once and reused for any query batch.

The cached state is the set of **inducing-point activations per block**
(`num_inds × d_model × num_layers`) — a fixed small tensor, NOT per-row K/V.
That's cheap to store and reuse.

> **Gate before coding:** confirm in `model.py` that `mab1`'s key/value `src`
> (and mab2) are masked to context rows via `attn_mask`/`train_size` so the
> inducing points never attend to query rows. The co-batch proof is strong
> evidence; verify directly before investing in the export.

## Implementation plan

1. **Export (WS-A / tools/export_onnx).** Split the forward into two graphs (or
   one graph with `context_state` as extra I/O):
   - `encode(x_ctx, y_ctx, cat_mask, d) -> context_state[num_layers]`
     (inducing-point activations),
   - `decode(x_q, cat_mask, d, context_state) -> logits_q`.
   Validate each against the monolithic graph (bit-parity on the same inputs).
   Keep the existing single graph as the default/fallback.
2. **Backends.** Both `OrtBackend` and `MIGraphXBackend` run two programs; the
   MIGraphX path compiles/caches an `encode` and a `decode` `.mxr` per bucket
   (decode is the hot one). Reuse the precision/`.mxr`/precompile machinery.
3. **Cache + lifecycle.** Key the context state by a hash of the context rows
   (the preprocessed `x_ctx`/`y_ctx` + cat_mask + d), stored in `TabFMState`
   alongside the loaded model. On predict: hash the context; hit → skip `encode`,
   run `decode`; miss → `encode`, store, then `decode`. Bound the cache (LRU) and
   invalidate on any context change. **Correctness risk lives here** (a stale/
   wrong-context hit returns silently wrong predictions) — the hash must cover
   everything that feeds `encode`, and a unit test must assert cached==fresh.
4. **Surface.** Transparent (automatic) for repeated same-context predicts; no
   API change. Optionally a `anofox_tabfm_context_cache` on/off + size setting.

## Validation
- Bit-identical logits: score query `Q` fresh vs. with a warm context cache.
- Wall-time: second predict on the same context ≫ faster (skips `encode`).
- Cache-miss safety: changing one context row forces a re-encode (assert output
  changes appropriately; assert no stale hit).

## Effort / risk
Multi-day: export-graph surgery (torch + re-export + parity), two-program
backends, cache lifecycle with a correctness-critical hash. Highest ceiling of
any item, but rushing it risks silent wrong predictions — hence its own goal
with the export gate verified first.

## Gate verification — EXECUTED 2026-07-04 (result: PASS, with a correction)

Read `vendor/tabfm/tabfm/src/pytorch/model.py` directly. The gate ("keys masked
to context so query rows are neither attended-to nor attend to each other") is
**confirmed**, but the cache-structure claim above needs a correction.

**Confirmed — caching is valid.** `ICLearning.forward` builds
`tm = arange(t) < train_size` and passes `mask = tm[:,None,None,:]` (key axis) to
`self.tf_icl`. Every row attends **only to context rows**; query rows are never
keys, so nothing attends to them. Context rows' per-layer representations are
therefore query-independent → cacheable. This is the mechanism behind the
empirical `tabfm_cobatch.test` proof.

**Correction — the ICL stage is a plain SAB `Encoder`, NOT an ISAB.**
`self.tf_icl = Encoder(...)` (line ~396) — each layer is self-attention
`mab(q=r, k=r, v=r, mask)`. So for the *in-context learner* there are **no
inducing points**; the cacheable state is the **per-layer context key/value
projections** (or equivalently the context rows' per-layer input reps), size
`∝ num_context × d_model × num_blocks` — it **scales with context size**, it is
not a fixed small tensor. Inducing points (ISAB) live only in the upstream
column/row `SetTransformer` embedders (`tf_col`/`tf_row`), whose train_size-masked
outputs are *also* context-derived and cacheable (and there the per-column
inducing state is small). So "encode" must cache: (a) the context-derived column
state from the embedders, and (b) the per-ICL-layer context K/V. Update step 1's
`context_state` shape accordingly and budget VRAM for an O(num_context) cache,
not an O(1) one — this changes the LRU sizing and the large-context trade-off.

**Net:** #7 is feasible and the independence premise holds; the export must expose
context K/V per ICL layer (+ column state), and the cache is context-sized. No
blocker found — ready for its own implementation goal.

## Implemented for TabICL — `anofox_tabfm_context_cache`

The export half landed in #38 (`export_tabicl --split-context`) and the runtime
half wires it into the engine. Against the plan above:

| plan | built | note |
|---|---|---|
| two graphs, single graph kept as fallback | yes | discovered next to the combined graph; absent pair ⇒ combined path, unchanged |
| cache in `TabFMState` alongside the model | in the **backend**, which lives in `TabFMState` | same lifetime and the same per-device lock, without a second thing to invalidate on unload |
| key by a **hash** of the context | keeps the context rows and compares them | see below |
| bound the cache (LRU) | one entry | see below |
| transparent, no API change | opt-in setting | see below |
| both backends | ORT only | MIGraphX (ROCm) runs bundled graphs, which have no split form |

**Not a hash.** The plan's own risk note — "a stale/wrong-context hit returns
silently wrong predictions" — argues against one. A hash mismatch is safe, but a
hash *collision* does not fail: it answers a query against someone else's
training data, silently, which is the one failure mode nobody would catch. The
support rows are a few hundred kB and a memcmp of them costs nothing next to the
forward pass it guards, so the cache compares what it claims to compare.

**One entry, not an LRU.** The cache is context-sized, as the gate correction
established: at real scale the prepare outputs are hundreds of MB. An LRU of
those is a resident-memory decision the user did not make. One entry serves the
case the whole feature is for — the same context scored repeatedly — and a
second context evicts the first rather than adding to it.

**Opt-in rather than transparent**, which is the one place this departs from the
plan, and it is a property of the TabICL split rather than a preference. The
query half has no label path at all — that is what makes it cheap — so a CONTEXT
row scored through it is no longer scored knowing its own label, and its fitted
value moves. Test-row predictions are unchanged (that is what #38's parity
asserts); the labelled rows' fitted values are not. Silently changing them for
anyone whose model ships a pair is not a thing to do by default. The plan's
transparency holds for a split that cuts a graph without dropping an input; this
one drops an input.

The other cost worth stating: a single call against a fresh context is SLOWER
than the combined graph, because it pays for a context it will not reuse. The
setting earns its place on the second call and after.

Validation, against the list above: cached-vs-fresh identity, cache-miss on a
changed context, and a changed context width are asserted in
`test/sql/tabfm_context_cache.test`; chained-vs-single-pass parity in
`tools/export_tabicl/tests/test_split.py`; wall-time in the trained-weight
prototype in `tools/export_tabicl/README.md` (11.2x at a 375-row context, 22
query rows).
