# Apple MLX backend — spike results

Verdicts for the spikes in `docs/MLX_PLAN.md`. The plan required each to end in
a written verdict, and warned that anything not marked "known" was a hypothesis
awaiting hardware. Three of its hypotheses changed on contact; they are marked
**CORRECTION** below.

All measurements: Apple M3, 10 GPU cores, 16 GB unified memory, macOS
(Darwin 25.6.0, arm64), MLX 0.32.1, ONNX Runtime 1.29.0, torch 2.13.0.
Reproduce with `tools/mlx_spike` (`uv sync`, then the commands named per spike).

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
| S-M3 | pending | `mlx-c` is alive and brew-installable; the feared blocker did not materialise |

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
| torch ↔ ORT (**the shipped CPU path**) | 8.5e-05 | **9.3e-04** | 1.0e-05 | 1.0000 |
| torch ↔ MLX | 6.1e-05 | 7.8e-04 | 6.1e-06 | 1.0000 |
| ORT ↔ MLX | 2.9e-05 | 6.6e-04 | 6.9e-06 | 1.0000 |

Two things fall out, and the spike asserts both rather than stating them:

1. **The shipped CPU graph scores 9.3e-04 against its own authoring code** —
   9× outside the 1e-4 bar. Any backend judged by this metric fails, including
   the reference. (`CLAIM_STRICT_METRIC_FAILS_ON_REFERENCE=True`)
2. **MLX is *closer* to the definition than ORT is** — 0.91×, 0.87×, 0.72× the
   error, in every case. (`CLAIM_MLX_NO_WORSE_THAN_ORT=True`)

So parity is judged on what the extension actually returns to SQL: the
post-softmax probabilities, plus the predicted class.

- `prob_max_abs ≤ 1e-4` — bounded in [0, 1], four orders below anything a user
  could observe, and passed by torch↔ORT at 1.0e-05
  (`CLAIM_PROB_METRIC_PASSES_ON_REFERENCE=True`).
- `argmax_agreement == 1.0` — the property the whole exercise exists to hold.
- a coarse `logit_max_abs ≤ 1e-3` guard, to catch a real bug while ignoring
  fp32 dust.

The strict metric is still computed and reported, marked informational, so the
claim stays checkable instead of asserted.

**Results under that bar:** every shape passes, worst `prob_max_abs` 1.5e-05,
argmax agreement 1.0000 throughout — on the real 302 MB weights, not a
synthesized fixture.

`mx.fast.scaled_dot_product_attention` roughly doubles the logit error versus an
explicit `softmax(QK^T)V` (5.8e-05 vs 2.9e-05). Both pass comfortably; fast SDPA
is kept for the speed and the difference is noted here so it is not rediscovered
as a mystery later.

> **This finding is not MLX-specific and should outlive this plan.**
> `equivalence.py` applies the same metric to the CUDA and ROCm backends. Either
> those comparisons are passing for a reason other than the stated tolerance, or
> the tolerance is doing no work there either. Worth checking before the next
> backend is judged by it.

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

Pending. The plan named **mlx-c maturity as the biggest product risk**, with
"if it blocks, the plugin links the C++ `libmlx` directly" as the fallback.
Early evidence says the risk is smaller than feared, and both paths are open on
this machine:

- `ml-explore/mlx-c` is actively maintained (231 stars, last push 2026-08-10)
  and packaged: `brew install mlx-c` gives **0.6.0**.
- `brew install mlx` gives **0.32.1** — the exact version the Python spikes ran
  against, so a C++ port can be compared against `mitra_mlx.py` with the
  runtime held constant.
- The `mlx` pip wheel *also* ships `libmlx.dylib`, headers, and a working
  `MLXConfig.cmake`, so the fallback needs no source build either.

Both are already installed on the spike machine.
