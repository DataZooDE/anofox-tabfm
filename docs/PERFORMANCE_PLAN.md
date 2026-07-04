# anofox-tabfm — performance plan (2026-07-04)

Consolidated from three independent analyses (Claude + `codex` + `gemini`/Antigravity)
plus first-party benchmarks on this box: **Ryzen 3950X (16C/32T, Zen2, AVX2, no
bf16) + RX 9070 XT (RDNA4/gfx1201, 64 CUs, 16 GB VRAM) + ROCm 7.2.4 / MIGraphX 7.2.3.**

## The one insight that reframes everything: batch size decides the regime

Measured warm forward (real 6.6 GB fp32 model, single forward, model already loaded):

| Rows in the forward (T) | CPU warm | CPU per-row | Regime |
|---|---|---|---|
| 10   | 0.42 s | 41.7 ms | bandwidth-bound (streaming 6.6 GB of weights, barely reused) |
| 200  | 2.06 s | 10.3 ms | transitioning |
| 1000 | 9.87 s | 9.9 ms  | **compute-bound (~10 ms/row)** |

CPU ≈ `0.35 s (weight streaming) + 0.010 s × T`. So **10k rows ≈ 100 s on CPU.**
The GPU is fixed-overhead-dominated at small T (~0.14 s for anything up to a 128-row
bucket) and compute-scales far better, so the **GPU advantage grows from ~3× at
T=10 to an estimated 30–100× at T≥1000** (its TFLOPs vs the CPU's ~10 ms/row).

Consequences that drive the plan:
1. **The GPU is not a nice-to-have; it's essential for bulk scoring.** A single
   `tabfm_classify(train, y, test:=big_table)` is already ONE forward (all query
   rows batched into T) — good. The lever is keeping bulk in one big forward and
   making that forward as fast as possible.
2. **At small batch we're bandwidth-bound** (streaming weights). **bf16 halves the
   bytes AND doubles GPU math** → wins in both regimes. This is why all three
   analyses rank it #1.
3. Per-predict *latency* (0.14 s) matters less than bulk *throughput*; optimize
   throughput first.

---

## Tier 1 — do now (high impact / low–moderate effort)

### 1. bf16 (not fp16) on the GPU — the single biggest win  ⭐ all 3 rank #1
- **What:** compile the MIGraphX program in bf16 (`migraphx::quantize_bf16` / the
  driver's `--bf16`). Prefer **bf16 over fp16**: RDNA4 runs both at ~2× fp32, but
  bf16 keeps fp32's exponent range (matches how the model was trained) → far lower
  accuracy risk than fp16 (whose narrow range can destabilize softmax/layernorm).
- **Impact:** ~1.8–2× faster GPU warm (0.14 s → ~0.07 s); `.mxr` and VRAM weights
  6.6 → 3.3 GB; GPU cold-load ~25 s → ~12 s; frees VRAM for bigger buckets.
- **Effort:** low–moderate. Add a precision to the `.mxr` cache key
  (`..._bf16_T128_H16.mxr`) and a setting `anofox_tabfm_gpu_precision =
  'bf16'|'fp16'|'fp32'|'auto'` (default `bf16`).
- **Risk:** accuracy — must validate. bf16: expected negligible. Keep softmax /
  layernorm / the logit output in fp32 if MIGraphX allows mixed precision.
- **Validate:** run `scenarios/` (churn F1, tips MSE) + a logit-divergence check
  vs the fp32 oracle; gate on ≥99.5% class agreement and no material F1/MSE move.
- **Not for CPU:** Zen2 has no native bf16 and no AVX-512-fp16 → reduced precision
  would add upcast overhead and likely *slow* the CPU. Keep CPU fp32.

### 2. Re-enable ORT prepacking on CPU (opt-in setting)  ⭐ codex+gemini+us
- **What:** `session.disable_prepacking` is currently `1` (chosen when RSS was
  18.6 GB). External-data got RSS to 6.7 GB, so the +16% is now affordable
  (~7.8 GB). Add `anofox_tabfm_cpu_prepack` (default on).
- **Impact:** ~10–16% faster CPU forward (0.45 s → ~0.38–0.40 s at T=10; larger
  absolute gains at big T where matmuls dominate).
- **Effort:** trivial (one config entry + a setting). **Risk:** none (just RSS).
- **Validate:** warm CPU latency + RSS with the setting on/off.

### 3. Keep bulk in one big forward + raise the row cap when VRAM allows
- **What:** a single predict already batches all query rows into T (good). Ensure
  the shape-bucket ladder has large buckets (up to `max_rows`) so big test sets
  don't get chunked into multiple weight-re-streaming forwards. With bf16's freed
  VRAM, consider raising `anofox_tabfm_max_rows` (default 10000) for the GPU path.
- **Impact:** avoids re-streaming 6.6/3.3 GB of weights per chunk on large scores.
- **Effort:** low (bucket-ladder + guardrail tuning). **Risk:** VRAM pressure —
  bound by available VRAM.
- **Caveat to verify (codex):** confirm query rows are mutually independent under
  the graph (attend to context, not each other) so co-batching can't change a
  row's prediction. Our single-table mode already co-batches, so the existing
  golden/scenario parity is evidence; add an explicit "row alone vs co-batched"
  test.

---

## Tier 2 — high value, more engineering

### 4. Pre-compile `.mxr` offline / at download, not on the query path  ⭐ codex+gemini
- **What:** the ~20-min first-predict compile per bucket is the worst UX wart.
  Compile a small set of common buckets (bf16) offline (CI or a `tabfm_download`
  step) and fetch/stage the `.mxr` like the weights. Use `--exhaustive-tune`
  offline (slower compile, faster runtime — free since it's one-time).
- **Impact:** first predict for a known bucket drops from ~20 min to a ~3.3 GB
  download + ~12 s VRAM load.
- **Effort:** medium (build/host artifacts + a download path; `.mxr` is arch- and
  ROCm-version-specific, so key by `gfx1201`+ROCm version, with on-device compile
  as fallback). **Risk:** low (driver/version skew → fall back to compiling).

### 5. Persistent device buffers instead of `offload_copy` (+ graph capture later)  ⭐ codex+gemini
- **What:** allocate per-bucket device buffers once, copy only the small real input
  spans, drop `offload_copy`. Foundation for HIP graph capture to cut kernel-launch
  overhead (which dominates at small T — MIGraphX perf showed ~45–64% "overhead").
- **Impact:** ~10–30% on the warm forward (bigger at small T where launch overhead
  dominates). **Effort:** medium (buffer lifecycle). **Risk:** none to accuracy.
- **Validate:** rocprof (fewer H2D/D2H, less host overhead) + identical logits.

### 6. Narrow the per-device mutex (overlap CPU prep with GPU)  ⭐ codex
- **What:** `DeviceMutex` currently wraps preprocess + tensor materialization +
  Run. Move preprocessing/materialization out; lock only session-load + `Run`.
- **Impact:** low for a single predict; medium for concurrent SQL groups (CPU
  preprocessing overlaps GPU inference). **Effort:** low. **Risk:** none.

---

## Tier 3 — architectural / research (highest ceiling, highest effort)

### 7. Context KV-cache — reuse the encoded train set across predicts  ⭐ gemini "Very High", codex/us "high effort"
- **What:** every predict re-encodes the whole context. For repeated scoring
  against the **same train set** (interactive exploration; streaming new rows),
  cache the per-block attention K/V for the context and run only the query rows
  against it. Latency `O(T_context+T_query)` → `O(T_query)`.
- **Impact:** potentially the largest win for interactive/repeated-context use —
  a big-context predict could drop from 0.14 s to <0.02 s.
- **Effort:** high — export the graph to expose/accept `past_key_values` per block,
  and add C++ cache lifecycle + invalidation keyed on the context. **Risk:**
  correctness of cache management (staleness); numerically identical if correct.
- **Validate:** logits for a query bit-identical cached vs fresh; huge wall-time drop.

### 8. Concurrent GPU streams / sessions for multi-query throughput  ⭐ codex+gemini
- **What:** allow a small pool of concurrent forwards per device instead of strict
  serialization, for many independent predicts. **Impact:** medium (multi-user /
  many-group throughput). **Effort:** medium-high; watch VRAM. Lower priority since
  bulk is already one forward.

---

## Low ROI here (deprioritize)

- **Smaller shape buckets for tiny data (codex #6):** at T≤128 the GPU is
  launch/overhead-bound, not compute-bound, so (16,8) vs (128,16) barely differ
  (~0.1 vs 0.14 s). Marginal; not worth the extra buckets/compiles.
- **threads=32 vs 16:** SMT rarely helps FLOP-bound matmuls on Zen2; quick sweep to
  confirm, but expect neutral. Default 16 (physical) is right.
- **CPU fp16 weights:** halves the small-batch bandwidth cost but Zen2 upcasts for
  compute and ORT CPU fp16 is limited — marginal, likely not worth it.

---

## Recommended sequence

1. **bf16 GPU** (#1) — validate accuracy on `scenarios/`, ship behind
   `anofox_tabfm_gpu_precision`. Biggest, well-scoped win; also unblocks #3/#4.
2. **CPU prepack setting** (#2) — trivial, immediate CPU gain.
3. **Bulk-forward/bucket tuning** (#3) — cheap, and the batch data shows it's where
   real throughput lives.
4. **Pre-compile/exhaustive-tune `.mxr`** (#4) — kills the 20-min wart; pairs with bf16.
5. **Persistent buffers + narrowed mutex** (#5, #6) — steady latency/throughput gains.
6. **Context KV-cache** (#7) — the big architectural bet for interactive use.

**Confidence note:** #1 (bf16), #2 (prepack), #4 (precompile), #5 (persistent
buffers) were independently proposed by all/most of Claude, codex, and gemini —
high confidence. #7 (KV-cache) has the highest ceiling but the most risk/effort.
