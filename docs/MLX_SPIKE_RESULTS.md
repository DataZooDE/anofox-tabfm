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

**The extension itself does not configure on macOS.** `cmake/ort.cmake` pins
`osx-universal2` prebuilt archives, and Microsoft no longer publishes them:

| ORT release | macOS assets |
|---|---|
| v1.29.0 (`TABFM_ORT_VERSION` default) | `onnxruntime-osx-arm64-1.29.0.tgz` only |
| v1.28.0 / v1.27.0 / v1.26.0 | `osx-arm64` only |

so configure dies on a 404. This predates the MLX work and blocks nothing in the
spikes above — the plugin builds standalone and was verified standalone — but
**M1 cannot be integration-tested until it is fixed**, since that needs a
working macOS build of the extension.

It is not a one-line swap. The comment at `cmake/ort.cmake:45` explains the
universal2 choice: "The DuckDB extension matrix cross-builds osx_amd64
(`OSX_BUILD_ARCH=x86_64`) on an arm64 runner, so we cannot key off the host."
Moving to `osx-arm64` gets arm64 building but leaves the osx_amd64 target with
no ORT, so it is a decision about whether x86_64 macOS is still a supported
platform — worth making deliberately rather than as a side effect of unblocking
MLX.

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
