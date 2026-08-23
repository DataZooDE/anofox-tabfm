# Apple MLX backend — spike results

Verdicts for the spikes in `docs/MLX_PLAN.md`. The plan required each to end in
a written verdict, and warned that anything not marked "known" was a hypothesis
awaiting hardware. Three of its hypotheses changed on contact; they are marked
**CORRECTION** below.

All measurements: Apple M3, 10 GPU cores, 16 GB unified memory, macOS
(Darwin 25.6.0, arm64), MLX 0.32.1, ONNX Runtime 1.29.0, torch 2.13.0.
Reproduce with `tools/mlx_spike` (`uv sync`, then the commands named per spike).
Every input is seeded from a literal, so a re-run reproduces these numbers rather
than merely numbers of the same magnitude.

**CORRECTION (scope).** The plan states "no Apple Silicon is available to this
environment; every spike and verification runs on your macOS machine", and
front-loads Python spikes to keep the human's time on the Mac short. That is no
longer true — the work now runs directly on an M3. The scripted,
marker-printing style is kept anyway: it is what makes these results
re-checkable by someone else on different hardware.

| spike | verdict | headline |
|---|---|---|
| S-M1 | **route 1 dead, environment green** | no ONNX→MLX importer exists, anywhere |
| S-M2 | **PASS** | MLX matches the shipped graph — and is *closer to the reference implementation than ORT is* |
| S-M4 | **PASS** | 4.3×–11.8× faster than ORT CPU; per-shape compile is 217 ms, not minutes |
| S-M3 | **PASS** | plugin builds, loads, and matches through the real C ABI; mlx-c covers the whole forward |

Plus one finding about the existing suite that the MLX work turned up:
**`equivalence.py`'s default (synthesized-weights) mode cannot pass for any
backend** — see the last section.

---

## S-M1 — environment + `mlx-onnx` feasibility

`uv run sm1_env`

**CORRECTION: route 1 does not exist.** The plan proposed trying
`mlx-onnx`, "ml-explore's experimental ONNX-to-MLX converter", as a one-day
spike vehicle that might also yield a second golden reference. It cannot,
because there is no such converter:

- `github.com/ml-explore/mlx-onnx` is an empty placeholder — one branch, three
  blobs (`.gitignore`, `LICENSE`, a 67-byte README that is just the repo
  title), no code, untouched since 2024-02-21.
- The `mlx-onnx` name **on PyPI is a different project** (`skryl/mlx-onnx`,
  MIT, 12 stars): "Standalone IR/ONNX export library for MLX" — it exports MLX
  graphs *to* ONNX. The opposite direction. Installing the plan's suggested
  `pip install mlx-onnx` would have silently produced this package.
- A GitHub search for any ONNX→MLX importer returns nothing.

So route 2 (hand-port) is not merely the realistic product path, it is the only
path; route 3 (an ONNX interpreter over MLX ops) remains the fallback it always
was. Decision point 3 in the plan — "whether route 3 is worth pursuing if S-M2's
hand-port is painful" — is answered below: the hand-port was not painful.

**Environment, all green.** Metal available; a 512×512 fp32 GEMM on the GPU
agrees with numpy to 9.2e-05. GPU arch `applegpu_g15g`; **max recommended
working set 11.8 GB**, max single buffer 8.9 GB. That working-set figure, not
`hw.memsize`, is the number M2 (tabfm-v1, 6.6 GB of weights) has to fit inside.

**Weights.** `autogluon/mitra-classifier` (Apache-2.0, 303 MB) downloads to the
registry's cache slug and its safetensors JSON header hashes to
`cb1a261b…3b96a` — byte-identical to `ExpectedWeightsHeaderShaFor("mitra",
"classification")` in `src/include/tabfm_model_spec.hpp`. The bundled ext
graph's baked offsets therefore index this file correctly, which is what lets
ORT read the weights straight off disk. 392 tensors, all F32, 12 layers,
dim 512 — as the plan said, no new export is needed.

**Golden reference established** for three shapes (40×8, 50×16 padded, 200×20),
saved beside the weights. Each is asserted to be *discriminating* — a reference
whose logits were undifferentiated would be matched by an equally broken
candidate, so parity against it would be vacuous.

---

## S-M2 — mitra forward hand-ported to MLX

`uv run sm2_parity` · `uv run python -m mlx_spike.sm2_triangulate`

**PASS.** `tools/mlx_spike/src/mlx_spike/mitra_mlx.py` is a line-for-line port
of `mitra_model_patched.py` and its `ExportWrapper`. It took hours, not the
1–2 days budgeted: the model is vanilla pre-norm transformer blocks, and every
non-obvious piece (the sort-based quantile embedding, the additive `-1e9`
attention masks, the "whole table is both support and query" wrapper) was
already spelled out in the patched torch source. Weights load via `mx.load`
straight from the released safetensors — no conversion, no tensor renaming.

### CORRECTION: the plan's parity tolerance cannot be met by the reference itself

The plan set "logits match ORT CPU (rtol 1e-4)", inheriting
`tools/gpu_test/equivalence.py`'s metric: max per-element
`|a-b| / max(|a|, 1e-12)`. The MLX port scored 7.5e-04 and "failed".

It is the metric that is broken, not the port. **Logits cross zero**, so that
denominator floor of 1e-12 turns a negligible absolute difference next to a
near-zero logit into an unbounded relative error. To show this is a property of
the metric rather than of MLX, `sm2_triangulate` brings in a third party:
`mitra_model_patched.py` is the *definition* of the forward — the ONNX graph
was exported from it — so torch eager sits upstream of both ORT and MLX.

| pair | logit max_abs | **strict rel** | prob max_abs | argmax |
|---|---|---|---|---|
| torch ↔ ORT (**the shipped CPU path**) | 8.5e-05 | **1.4e-03** | 4.6e-06 | 1.0000 |
| torch ↔ MLX | 5.9e-05 | 1.2e-03 | 4.3e-06 | 1.0000 |
| ORT ↔ MLX | 6.6e-05 | 1.9e-03 | 3.1e-06 | 1.0000 |

Two things fall out, and the spike asserts both rather than stating them:

1. **On these inputs the shipped CPU graph scores 1.4e-03 against its own
   authoring code** — 14× outside the 1e-4 bar, on a CPU-vs-CPU pair where no
   accelerator is involved at all.
   (`CLAIM_STRICT_METRIC_FAILS_ON_REFERENCE=True`)
2. **MLX is *closer* to the definition than ORT is** — 0.93×, 0.86×, 0.70× the
   error, in every case. (`CLAIM_MLX_NO_WORSE_THAN_ORT=True`)

The scope of (1) matters and is narrowed by the investigation below: the strict
metric's verdict turns on how close the reference logits come to zero, which is
a property of the *inputs and weights*, not of the backend. It is not the case
that every comparison fails — see "Does the 1e-4 bar do any work?".

So parity is judged on what the extension actually returns to SQL: the
post-softmax probabilities, plus the predicted class.

- `prob_max_abs ≤ 1e-4` — bounded in [0, 1], four orders below anything a user
  could observe, and passed by torch↔ORT at 4.6e-06
  (`CLAIM_PROB_METRIC_PASSES_ON_REFERENCE=True`).
- `argmax_agreement == 1.0` — the property the whole exercise exists to hold.
- a coarse `logit_max_abs ≤ 1e-3` guard, to catch a real bug while ignoring
  fp32 dust.

The strict metric is still computed and reported, marked informational, so the
claim stays checkable instead of asserted.

**Results under that bar:** every shape passes, worst `prob_max_abs` 5.8e-06,
argmax agreement 1.0000 throughout — on the real 302 MB weights, not a
synthesized fixture.

`mx.fast.scaled_dot_product_attention` and an explicit `softmax(QK^T)V` differ
at the 1e-05 level on logits, neither consistently closer. Both pass
comfortably; fast SDPA is kept for the speed, and the difference is noted here
so it is not rediscovered as a mystery later.

> **This finding is not MLX-specific.** `equivalence.py` applies the same metric
> to the CUDA and ROCm backends, so it was worth knowing whether those
> comparisons mean what they claim. That investigation is the next section, and
> its answer is *"yes for the real-weights path, no for the default one"*.

---

## S-M4 — performance reality check

`uv run sm4_bench`

**Run before S-M3, deliberately.** S-M3 costs a day of mlx-c work whose only
justification is that MLX is meaningfully faster here. Measuring first spends
that day on evidence rather than hope — the plan's own "measure before
promising", earned by ROCm's 25-minute compile.

**PASS.** mitra classification, 20 features, half the rows as context:

| rows | MLX warm | ORT CPU | speedup | MLX peak mem | parity (prob max_abs / argmax) |
|---|---|---|---|---|---|
| 100 | 421 ms | 1.8 s | 4.3× | 1.97 GB | 4.1e-06 / 1.0000 |
| 250 | 1.06 s | 4.9 s | 4.6× | 1.35 GB | 7.2e-06 / 1.0000 |
| 500 | 2.19 s | 11.6 s | 5.3× | 1.44 GB | 7.8e-06 / 1.0000 |
| 1000 | 4.68 s | 27.6 s | 5.9× | 1.72 GB | 2.2e-05 / 1.0000 |
| 2500 | 15.5 s | **182 s** | **11.8×** | 3.19 GB | 8.7e-06 / 1.0000 |

Parity is re-checked at every size — a speed number from a backend that had
started returning different answers would be worthless.

The advantage widens with rows because both sides are O(T²) in the row
attention but Metal absorbs it far better. At 2500 rows ORT CPU takes over
three minutes, which is the real argument for this backend: on an
Apple-Silicon Mac today, mitra at any interesting table size is effectively
unusable, and MLX makes it merely slow.

**CORRECTION: per-shape compile cost is a non-issue.** The plan carried ROCm's
lesson forward and warned that "MLX recompiles per shape like MIGraphX; if
compile cost is non-trivial, the existing shape-bucket + precompile machinery
(`tabfm_gpu_precompile`, `mxr_source` analogue) generalizes." Measured on a
shape nothing else had touched (333 rows × 17 features): first call 2866 ms,
warm 2649 ms — **217 ms of first-shape overhead, 8% of one call.** Nothing here
needs shape buckets, precompilation, or an `mxr_source` analogue. That machinery
should *not* be generalized to MLX; doing so would add cache-invalidation
surface to buy 217 ms.

**The memory ceiling is the row attention, not the weights.** Peak is 3.19 GB at
2500 rows against 303 MB of weights. The row attention materialises
`(H+1) × heads × T × T` scores — 2.1 GB of the 3.19 GB at T=2500 — and grows
quadratically, so mitra's registry `max_rows` of 10 000 would need ~34 GB and
does not fit in the 11.8 GB working set. The quantile embedding's
`T × 999 × features` intermediate is the second term. Neither is MLX-specific
(ORT CPU has the same shape, and would take hours at that size), but M1 should
know where the wall is rather than discover it.

---

## S-M3 — mlx-c plugin skeleton

`src/tabfm_mlx_plugin.cpp` · `tools/gpu_test/plugin_load_check.c` ·
`uv run python -m mlx_spike.sm3_abi <plugin.dylib>`

**PASS — and the plan's biggest risk did not materialise.** The plan named
**mlx-c maturity** as the top product risk, with "if it blocks, the plugin links
the C++ `libmlx` directly" as the fallback. The fallback was not needed:
`mlx-c` 0.6.0 covers the entire mitra forward, including the two ops most likely
to have been missing, `mlx_fast_layer_norm` and
`mlx_fast_scaled_dot_product_attention`. This was checked against the headers
*before* any code was written, since that is what the risk deserved.

`ml-explore/mlx-c` is actively maintained (231 stars, last push 2026-08-10);
`brew install mlx mlx-c` gives 0.32.1 / 0.6.0 — MLX at the exact version the
Python spikes ran against, so the C++ port is comparable to `mitra_mlx.py` with
the runtime held constant.

**The artifact.** A 76 KB `.dylib` exporting exactly one symbol,
`_TabFMGetPluginApi`, linking only `libmlxc`, `libmlx` and `libc++`. The
project's own `plugin_load_check.c` compiled on macOS **unchanged**, as the plan
predicted, and passes:

```
LOAD_CHECK dlopen ok
LOAD_CHECK_OK name=mlx abi=1
CREATE_WITHOUT_GPU graceful error: … the mlx backend implements the 'mitra'
architecture only, not 'load-check'. Use SET anofox_tabfm_model='mitra', …
```

**Correctness through the ABI, not just loading.** A plugin that loads and
returns wrong logits passes a load check, so `sm3_abi.py` drives
create/run/free_output/destroy through `tabfm_plugin_abi.h` via ctypes — the
struct layouts declared independently of the C header, so a field-order or size
mismatch surfaces as garbage instead of being papered over by including the
header that defined it. Against the same ORT CPU golden:

| case | logit max_abs | prob max_abs | argmax |
|---|---|---|---|
| small (40×8) | 3.052e-05 | 4.3e-06 | 1.0000 |
| padded (50×16) | 1.979e-05 | 5.8e-06 | 1.0000 |
| wide (200×20) | 8.535e-05 | 2.7e-06 | 1.0000 |

These are **the same numbers to every digit** as the Python port's fast-SDPA
run in S-M2 — strong evidence the transcription is faithful rather than
accidentally close. Model load is 114 ms for 303 MB. The wrong-architecture
refusal is asserted, not assumed: a plugin that failed open there would serve
mitra's math over another model's weights.

Two things the plugin does differently from its siblings, both deliberate:

- **It ignores `graph_path`.** There is no ONNX importer, so the forward is
  transcribed from `mitra_model_patched.py`; the parameter is used only to
  locate weights. The consequence — this file is a *second implementation of a
  model we already ship*, and must be kept in step with the torch source — is
  stated at the top of the file, because it is the maintenance cost of the whole
  approach.
- **`precompile` is a no-op that succeeds**, per S-M4's 217 ms measurement.

**Weights must be read on a CPU stream.** MLX's `Load` op has no GPU
implementation; scheduling the safetensors read on the GPU stream produces
arrays that fail at *first forward* with `[Load::eval_gpu] Not implemented` —
far from the cause. The plugin loads on a CPU stream and forces the read to
completion at create time. Unified memory means the materialized tensors are
then usable by GPU ops with no copy.

**Build wiring.** `CMakeLists.txt` gains an optional `anofox_tabfm_mlx_plugin`
target on the MIGraphX pattern — silent when MLX is absent, and additionally
guarded on `APPLE AND arm64`, so the plugin is never required for the cpu build.
Verified to configure and build.

---

## Blocker for M1, found while wiring the build (not MLX's fault)

**The extension's prebuilt-ORT path does not configure on macOS.** That is
`make debug` and `make test_debug` — the development and test-suite path.
`make release` (cpu flavor) is unaffected: `Makefile` defaults
`TABFM_ORT_VCPKG=1`, which builds ONNX Runtime from the vcpkg port and links it
statically, so it never fetches an archive. An earlier draft of this section
said "does not configure on macOS" without that qualification, which was
broader than the evidence.

`cmake/ort.cmake` pinned `osx-universal2` prebuilt archives, and Microsoft no
longer publishes them:

| ORT release | macOS assets |
|---|---|
| v1.29.0 (`TABFM_ORT_VERSION` default) | `onnxruntime-osx-arm64-1.29.0.tgz` only |
| v1.28.0 / v1.27.0 / v1.26.0 | `osx-arm64` only |

so configure died on a 404. This predates the MLX work and blocks nothing in the
spikes above — the plugin builds standalone and was verified standalone — but it
does block the practical M1 workflow, since `make test_debug` is how the engine
integration would be exercised.

**Fixed**: macOS now fetches `osx-arm64`, and `osx_amd64` is dropped from the
build matrix. Two things learned the hard way while verifying it, both worth
keeping:

- `OSX_BUILD_ARCH` is **set but empty** for a normal host build and carries a
  value only when the DuckDB matrix cross-compiles. A guard written as
  `if(DEFINED OSX_BUILD_ARCH AND ...)` therefore rejects every native macOS
  build. Test it for truthiness.
- The guard belongs inside `_tabfm_fetch_prebuilt_ort`, not in the platform
  mapping: only that path needs a macOS archive to exist, and the cpu release
  build (vcpkg ORT) legitimately never reaches it.

It was not a one-line swap. The retired comment explained the universal2 choice:
"The DuckDB extension matrix cross-builds osx_amd64 (`OSX_BUILD_ARCH=x86_64`) on
an arm64 runner, so we cannot key off the host." That is still true of the
matrix; what changed is that the archive carrying both slices stopped existing.
Moving to `osx-arm64` leaves the osx_amd64 target with no ORT at all, so this
was a decision about whether x86_64 macOS remains a supported platform — taken
deliberately rather than as a side effect of unblocking MLX. Pinning macOS back
to 1.23.2 was the alternative, and it collides with this PR's phase 2, which
needs ORT >= 1.28 for CUDA.

*Unrelated trap in the same area*: a stale `build/release/CMakeCache.txt` keeps
whatever `TABFM_ORT_VERSION` it was first configured with, because
`set(... CACHE ...)` does not overwrite an existing entry. A cache from an older
checkout will silently build against a different ORT than the source says.
`rm -rf build/release` when the version matters.

---

## Does the 1e-4 bar do any work for CUDA/ROCm today?

`uv run python -m mlx_spike.tolerance_probe`

S-M2 found the strict metric failing on a CPU-vs-CPU pair, and that metric is
what the CUDA and ROCm comparisons are judged by — while those are reported as
passing. Both could not straightforwardly be true, so this measured which it
was, **importing `equivalence.py` itself** and using its synthesis, its default
`--shapes`, and its `compare()`. Reimplementing them would have tested a
lookalike rather than the thing that ships.

The answer is split, and the split is the useful part.

### With real weights: the bar holds, and it is doing real work

torch (the definition) vs ORT (the reference), mitra, `equivalence.py`'s own
default shapes:

| shape | strict rel | verdict | logit max_abs | prob max_abs | min \|logit\| |
|---|---|---|---|---|---|
| 70×3×60 | 1.301e-05 | **PASS** | 3.052e-05 | 7.9e-06 | 1.255 |
| 128×8×100 | 1.578e-05 | **PASS** | 1.621e-05 | 2.6e-06 | 0.447 |

Comfortably inside 1e-4, with an order of magnitude to spare. **So a
`--weights`-backed CUDA or ROCm comparison is legitimately judged**, and the
S-M2 finding does not invalidate the GPU verification already done on this PR.
Trained mitra produces confident logits that stay well away from zero, so the
denominator never gets small and the metric behaves like the relative error it
is meant to be.

### With synthesized weights — the default mode — it cannot pass at all

Omit `--weights` and `equivalence.py` synthesizes initializers as
`standard_normal * 0.02`. That is an untrained, near-degenerate network: its
logits collapse into `[-0.036, 0.040]`, and the smallest one is 8.0e-05 from
zero. The same torch-vs-ORT pair:

| shape | strict rel | verdict | **logit max_abs** | prob max_abs | min \|logit\| |
|---|---|---|---|---|---|
| 70×3×60 | 1.387e-04 | **FAIL** | **2.794e-08** | 3.3e-09 | 8.0e-05 |
| 128×8×100 | 1.065e-04 | **FAIL** | **2.980e-08** | 2.9e-09 | 1.0e-04 |

Read the absolute column: the two implementations agree to **28 nanounits**,
and the probability delta is 3e-09 — essentially machine precision on a CPU/CPU
pair. The metric calls it a failure.

The mechanism is that both quantities shrink with the weight scale, but not at
the same rate: going from real to synthesized weights divided the absolute
error by ~1000, and divided `min |logit|` by ~15000. The ratio therefore gets
*worse* as the network gets more degenerate. (This also corrects an intermediate
version of this probe, which tried to predict the verdict from
`eps / min|logit|` at a fixed `eps`. That assumes error magnitude is independent
of value scale, which is false — hence the measurement above rather than an
extrapolation.)

### What this means

- **`equivalence.py` run the way its own README documents first** —
  `equivalence.py resources/graph_tabicl_classification.onnx --providers cpu` —
  is on the synthesized path. For `--providers cpu` alone this is invisible,
  because `compare()` short-circuits on `np.array_equal` and CPU-vs-CPU is
  bit-identical. **Add any second backend and it fails**, no matter how correct
  that backend is, because non-bit-identical is all it takes.
- So the suite is sound exactly where it has been exercised with real weights,
  and a trap everywhere else. The design intent — "runs with synthesized
  initializers by default (no licensed weights), or against a real cached
  checkpoint with `--weights`, which is the stronger statement" — has the
  weaker mode being not merely weaker but unusable for its stated purpose.
- The fix is the S-M2 metric: judge post-softmax probabilities plus argmax.
  Under it, the synthesized pair passes at 3e-09 and the real pair at 8e-06,
  both by wide margins, and a genuinely diverging backend still fails.
  Deliberately **not** applied to `equivalence.py` in this PR — changing the
  pass/fail definition for CUDA and ROCm should be verified against that
  hardware, which is not available here.

Recommended follow-up, in order: raise the synthesized weight scale so the
default mode produces non-degenerate logits (a one-line change that makes the
existing metric usable), then adopt the probability metric once it can be
re-run on GPU hardware.

*Note: the probe prints a `recursive_mutex lock failed` line at interpreter
teardown, from having both MLX and torch loaded in one process. It occurs after
the result and does not affect the exit code, which is 0.*

---

## S-M5 — route 3: one interpreter, every model

`uv run python -m mlx_spike.sm5_all_models`

The scope changed to *all* registered models (see the plan's "Scope change"),
which turned the hand-port from the product path into the wrong shape of
answer: seven architectures transcribed by hand is seven second
implementations, each able to drift from the graph the other backends run.
`tools/mlx_spike/src/mlx_spike/onnx_mlx.py` executes the shipped graph instead.

**Op coverage: 13/13 graphs, complete.** Measured, not estimated:

| | |
|---|---|
| graphs | 13 (`resources/graph_ext_*.onnx`) |
| nodes | 4,526 – 12,424 |
| distinct ops per graph | 33 – 48 |
| union of op kinds | 64 |

The first pass — elementwise, shape, reduction and nn ops — already covered
**4 graphs outright** (tabfm-v1 and mitra, both tasks). The gap to the other
nine was exactly seven ops, and exactly the ones this plan predicted would be
the hard part: `Pad` (9 graphs), `ScatterND` (5), `ScatterElements` (4),
`Einsum` (3), `SequenceAt` + `SplitToSequence` (3), `GatherND` (2).

**Correctness, so far: mitra classification AND regression, against real
weights.** Interpreter vs ORT CPU on the same staged initializers:

| graph | weights | logit max_abs | prob max_abs | argmax |
|---|---|---|---|---|
| mitra classification 70×3×60 | real | 2.193e-05 | 6.4e-06 | 1.0000 |
| mitra classification 128×8×100 | real | 1.097e-05 | 2.3e-06 | 1.0000 |
| mitra regression 70×3×60 | real | 2.384e-06 | — | — |
| mitra regression 128×8×100 | real | 2.563e-06 | — | — |

On the golden shapes the interpreter is *closer to ORT than the hand-port is*
(5.3e-05 vs 8.5e-05 on `wide`), which is what running the same graph should
produce.

**Stated plainly: op coverage is not correctness.** All 64 ops are implemented,
but only mitra is parity-verified end to end. The remaining eleven graphs are
covered-but-unverified until `sm5_all_models` has run them, and a `Pad` that
packs its widths wrongly raises nothing — it returns wrong numbers. The harness
uses synthesized initializers by default precisely so every model is testable
without seven checkpoints and seven licences: both sides get the same numbers,
which is all that is needed to test op semantics.

**Still to do before this is the shipping path**: the interpreter is Python. The
plugin currently runs the hand-port. Porting the interpreter to C++ over mlx-c
is the remaining work, and the hand-port stays afterwards as an independent
oracle on the one model where two implementations exist.

---

## M1 production verification — `tools/gpu_test/scenarios/mlx_stress.sql`

4000-row table, 2500-row context, 1500 scored, on the M3. Every stage prints
`*_SERVED_BY`, because "cpu and mlx agree" is also what a silent CPU fallback
prints.

| check | result |
|---|---|
| classification, cpu vs mlx | **0 disagreements / 1500** |
| wall clock | cpu 63.2 s → **mlx 12.9 s (4.9×)** |
| device alternation mid-session | `ALT_STABLE drift=0`, both sessions resident |
| regression, cpu vs mlx | max abs diff **6.95e-05**, correlation **0.99999999999** |
| bf16 label flips | **2 / 1500** |
| fp16 label flips | **1 / 1500** |
| sessions cached | **4** — (cpu) + (mlx × fp32/bf16/fp16), so a precision switch is a new entry, not an eviction |
| single-class context | 1 distinct prediction over 60 rows, no error |
| constant feature (zero variance) | 60/60 non-null — the `where(var==0, 0, x)` guard survives the port |
| NULL-bearing feature | 60/60 non-null |

**This scenario earned its keep by finding a real bug before it shipped.** The
device-alternation stage failed with

    MLX error: There is no Stream(gpu, 0) in current thread

MLX's default streams are thread-local; the backend held one created on the
thread that ran `create()`, and the first forward on any other thread died —
at predict time, far from the cause. Alternating devices mid-session is the
only thing in the suite that gets a second DuckDB thread involved. The backend
now holds no stream and each call acquires the calling thread's own.

Two honest limits of this run:

- **`CLS_ACCURACY` was 0.422 against a 0.333 baseline** on the first version,
  because a positional split of a sin/cos series puts context and query in
  different phase regions. The cpu/mlx agreement was still exact — which is the
  point: agreement is insensitive to whether the task is learnable, so the
  accuracy guard has to be independently meaningful or it is decoration. Now
  split by `hash(row_id)` with the majority-class rate printed beside it.
- **The unsupported-model stage did not reach its assertion**: `tabpfn-v2` has
  no downloaded weights on this machine, so it failed at the download gate
  before the MLX arch check. The refusal itself is covered by
  `test_tabfm_mlx_plugin.cpp` and by `MlxSupportsModel`, but the scenario stage
  is currently proving the wrong error.

## Where all-model coverage actually stands

`uv run python -m mlx_spike.sm5_all_models --shapes 128x8x100`

| verdict | graphs |
|---|---|
| **PASS** (5) | mitra classification + regression (**real weights**), tabicl classification + regression, orion-bix classification |
| **INCONCLUSIVE** (3) | tabpfn-v2, tabpfn-v2-5, tabpfn-v3 classification — the ORT *reference* returns a constant, so any comparison is vacuous |
| **ERROR** (3) | tabpfn-v2, tabpfn-v2-5, tabpfn-v3 regression — **ORT itself** fails a MatMul on the harness's feed |
| **SKIPPED** (2) | tabfm-v1 classification + regression — 6.1 GB of synthesized initializers |

None of the six non-passes is a known interpreter defect; all are limits of the
*harness*. The three INCONCLUSIVE need real checkpoints (synthesized weights
cannot drive those graphs off a constant, even rescaled 10×), the three ERROR
need a feed those graphs accept, and the two SKIPPED need the interpreter to
stream weights rather than materialize them twice in Python. Stated as
inconclusive rather than passed, because an exact match against a constant
reference is exactly what a completely broken interpreter also produces.

---

## All-model verification — 12/13 graphs, against real weights

`uv run python -m mlx_spike.sm5_all_models --shapes 70x3x60`

| model | classification | regression |
|---|---|---|
| tabfm-v1 | not tested (no cached weights) | **PASS** — real 6.6 GB, in situ |
| mitra | **PASS** real | **PASS** real |
| tabpfn-v2 | **PASS** real | **PASS** real |
| tabpfn-v2-5 | **PASS** real | **PASS** real |
| tabpfn-v3 | **PASS** real | **PASS** (synth: see below) |
| tabicl-v2 | **PASS** real | **PASS** real |
| orion-bix | **PASS** real | n/a |

Getting here required fixing three things, and each was a bug in the *harness*
that had been masquerading as backend behaviour:

1. **The single_eval_pos family takes a SHORTER y.** TabPFN, TabICL and Orion
   declare `y` with its own dim (`y [1, s94]` against `x [1, s27, s53]`) and
   infer the train/test split from that length; only tabfm-v1 and mitra take
   `train_size` as a scalar with a full-length y. Feeding the first family a
   full-length y padded with the −100 sentinel made them read every row as
   context and return a **constant** — which the degeneracy guard caught as
   INCONCLUSIVE rather than reporting the exact-match-on-a-flat-reference as a
   pass. Without that guard this would have read as 3 more green rows.
2. **Regression outputs are unbounded.** `tabicl_regression` "failed" at
   1.869e-04 absolute — on outputs spanning 15.4, i.e. 1.2e-05 relative. The
   1e-4 bound is meaningful for post-softmax probabilities and meaningless for
   a regression head; it is now scaled by the reference's own range. This is
   the same mistake as the S-M2 tolerance finding, made a second time in a new
   place.
3. **tabfm-v1 could not be staged at all.** Its 6.6 GB does not survive being
   copied to a temp dir on top of the copies ORT and MLX each make. The engine
   already lays the cache out as graph-beside-weights, so that layout is now
   used in place, and the largest model became testable — 1.317e-05 on real
   weights, which is also the unified-memory answer the plan wanted.

Two gaps stated rather than hidden:

- **tabfm-v1 classification** is untested: only the regression checkpoint is
  cached locally. It is the same graph family as the regression one that passes.
- **tabpfn-v3 regression** runs on synthesized weights because its released
  checkpoint cannot be converted — it carries no `FullSupportBarDistribution
  criterion.borders`, so `convert_weights.py` cannot build the point-estimate
  head. That is upstream of this work.

### A pre-existing bug found on the way

`tools/export_tabpfn/convert_weights.py` accepted `--arch=v3`, fell through its
if/else into the **v2** branch, wrote v2 weights to the v2 path, and printed
`tensor-map keys: 129 | present: 129 | missing: 0` — indistinguishable from
success. Anyone following `docs/REAL_MODELS.md` (which lists tabpfn-v3 as
"ckpt→safetensors convert required") would have been running v2 believing it
was v3. Fixed, and an unrecognised `--arch` now refuses instead of falling
through.

Also worth recording for anyone testing locally: **these checkpoints need a
one-time conversion before they work on ANY backend, cpu included.** The MLX
work did not create that requirement, it just ran into it first.

---

## Production hardening of the interpreter

Two failures found by running the shipped build at realistic sizes rather than
by reasoning about it. Both were mine, and neither was visible on small inputs.

### 1. mitra silently lost its hand-port, and OOM-killed

Once the engine started supplying a graph for every model, the plugin's
"hand-port only when no graph" rule quietly routed mitra through the
interpreter. The stress scenario died with **SIGKILL (137)** partway through
classification at 2500 context / 1500 query rows.

The interpreter keeps every intermediate alive as long as the graph might
reference it; the hand-port reuses buffers. mitra now takes the hand-port
whenever it is asked for. It remains the interpreter's independent oracle.

### 2. The interpreter never freed anything

The same root cause, unfixed, capped every *other* model. Measured on
tabpfn-v3, same metric (peak memory footprint) before and after adding
last-use eviction:

| rows | before | after |
|---|---|---|
| 4 000 | completes, **27.7 s** | completes, **1.8 s** |
| 8 000 | **40.7 GB → killed** | **7.3 GB, exit 0, 3.1 s** |

The fix is the standard one: compute each value's last consumer at load time,
and erase it from the environment once that node has run. Initializers are no
longer copied into the environment at all — a lookup miss falls through to the
mapped weights — so a run holds intermediates only. tabicl-v2 at 8 000 rows
lands at 7.7 GB / 2.1 s.

The 15× speedup at 4 000 rows is not a separate optimisation: it is what
happens when a process stops thrashing.

### Verified after both fixes

- Every model still exact: 1.0 label agreement on all six classification
  models; regression max diff ≤ 6.5e-05, correlation ≥ 0.99999999999.
- Stress scenario, genuine exit 0: 0 disagreements / 1484, cpu 71.4 s vs
  mlx 12.9 s, bf16 1 flip and fp16 0 flips / 1484, four sessions resident,
  device alternation stable, constant-feature and single-class shapes clean.
- Suite: 685 sqllogictest assertions, 72 125 Catch2 assertions.

> **A process note, since it cost real time.** Three times I wrote
> `make ... ; echo "EXIT=$?"`, which reports `echo`'s status, and twice I
> reported a passing build that had actually failed — including the OOM above.
> The exit code of a pipeline is not the exit code of the thing you care about.
