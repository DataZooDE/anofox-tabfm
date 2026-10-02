# LimiX-2 export spike: blockers found by READING (before trying)

Upstream `limix-ldm-ai/LimiX` @ 516bf39 (2026-09-18), `model/v2_0/*`; checkpoint `stable-ai/LimiX-2`
(406,232,101 params, 1,496 fp32 tensors, 24 layers, width 256, 8 heads, 1.62 GB).

Checked against the code, and where the task brief said "reported as guarded ... verify, do not trust":

| # | blocker | where | verdict |
|---|---|---|---|
| R1 | `import nvtx` at module top level | transformer.py:1, layer.py:5, decoupled...py:14 | hard dependency (extra pip package) |
| R2 | `triton` imported UNGUARDED | layer.py:24 -> operators/rmsnorm/__init__.py -> triton_rmsnorm.py `import triton` | **the "guarded" claim is only half true**: flash_attn IS guarded (layer.py:28, decoupled:35); triton is not, so the model cannot be imported without the triton package, and triton has no macOS/Windows wheel |
| R3 | checkpoint config says `rmsnorm_impl: triton` | config | needs overriding to 'torch' before building |
| R4 | runtime `torch.randn((G, E/4))` positional embedding, `generator=None` | transformer.py:768, `feature_positional_embedding_type: subspace` | the model is STOCHASTIC by design; a fixed table is a choice, not a transcription |
| R5 | in-place `y["data"][:, eval_pos:] = nan` | transformer.py:399 | known class (v1 patch 3) |
| R6 | `padding_xy`: Python `%` branch on a symbolic width + `zeros(T - S)` pad | transformer.py:615 | the SAME two baked-shape defects found in LimiX-2M |
| R7 | NaN guards that raise | transformer.py:443, 723 | known class |
| R8 | `make_feature_padding_mask` builds `torch.tensor(list_of_python_ints)` | transformer.py:14 | symbolic H into a Python list |
| R9 | dict output, 4 target tokens (`num_cls_tokens: 4`), `flatten_y_tokens` | transformer.py:495-610 | needs a wrapper that picks one head |
| R10 | `task_type=="reg"` is a comparison, not an assignment | transformer.py:517 | upstream bug: only the literal "Regression" works |
| R11 | regression head is 5,000 bucket logits (`reg_y_decoder ... linear-5000`), decoded to a value outside the model | predictor.py:3321 | graph must reproduce a bar-distribution mean; `_reg_borders` IS in the checkpoint |
| R12 | `@autobatch` on the main attention methods | layer.py:94,385,468 etc | pass-through unless CUDA OOM; `AutobatchConfig.ENABLE_AUTOBATCH=False` removes it |
| R13 | per-layer `x[:, :eval_pos]` slices (24 layers) | layer.py:1156 | the depth blow-up class: needs `torch._check(S <= T)` |
| R14 | `MulticlassTargetEncoder` uses `torch.unique` + per-row loop | encoders.py:487 | config uses `emby_embedding` for cls y: to be CONFIRMED off-path |
| R15 | `mask_process_4_x` has `randn_like`/`randperm` | transformer.py:732 | only on mask codes 4 / categorical mask: to be CONFIRMED off-path |
| R16 | `assert eval_pos < x.shape[1]` and `eval_pos <= y.shape[1]` | transformer.py:515 | symbolic asserts |
| R17 | `torch.get_autocast_dtype` / autocast branches | transformer.py:493 | fine eager; to be checked traced |
| R18 | python>=3.12, torch>=2.9.1 | README | venv here is py3.12 / torch 2.12.1: satisfied |

---
# Outcome (2026-10-02): export WORKS end to end. Not shipping.

## Found by reading vs found by trying
Read-first list above held up, with these corrections/additions found only by trying:
- R14 was NOT "off-path": `MulticlassTargetEncoder` (label -> dense rank via `torch.unique`) is on the classification path and
  exported as an ONNX `Unique` op (data-dependent shape). Replaced by an exactly equivalent pairwise formulation (checked on gappy
  labels); graph is now Unique-free.
- R15 `mask_process_4_x` evaluates `randn_like` unconditionally (torch.where) => a RandomNormalLike op in the graph. Removed.
- NEW (trying): `nvtx.annotate` is untraceable by Dynamo (cython) -> stubbed (also drops the dependency).
- NEW (trying): `torch.any(feature_padding_mask)` data-dependent branch in DStI.forward -> removed (all-False mask is identical).
- NEW (trying): the 2nd raising NaN guard (`embedded_all`), and the Python `torch.tensor(real_feature_nums)` mask.
- NEW (reading deeper): `SoftmaxScalingMLP` makes attention temperature a function of log(#feature groups + 4) via
  `torch.tensor(n/1.0)`: a baked-length candidate. Kept symbolic via scalar_tensor; parity at 7 shapes with H in {1,5,6,7,9,11,30} passes.
- NEW: truncating layers for small-dims stage needs `feature_emb_layer/reg_y_emb_layer/cls_y_emb_layer` re-pointed (config validation).
- R4: positional embeddings are an UNSEEDED randn per call. A fixed seeded table (randn prefix property verified) is a choice:
  two unseeded upstream runs differ by 0.09 on logits of scale 23 (argmax identical).
- R2: triton is imported unguarded (flash_attn is guarded); the install needs `triton`; CPU path falls back to F.rms_norm itself.
- R10: upstream `task_type=="reg"` is a comparison, not an assignment (bug; only "Regression" works).
- NOT a blocker (read, then confirmed): R5 in-place y write, R7 raising guards, y_type uniform per task, `@autobatch` (disabled by flag),
  flash_attn (guarded), dict outputs, y_token_k=4 flatten, 24 layers' `x[:, :eval_pos]` slices (flat export time with `torch._check(S<=T)`).

## Export products (real 24-layer, 406,232,101 params, fp32)
| graph | ONNX | weights (.data) | export time | nodes / distinct ops |
|---|---|---|---|---|
| classification | 47.9 MB | 1.52 GB | 88-109 s | 23,831 / 45 |
| regression | 47.8 MB | ~1.5 GB | 83 s | 23,780 / 43 |
Two graphs => ~3 GB shipped (one checkpoint is 1.63 GB; weights are duplicated across the two task graphs).

## Parity (real weights, unmodified upstream as oracle, ALL rows, 7 shapes none equal to the trace shape except case 0)
- patched wrapper (eager) vs unmodified upstream: classification 0.0 (bit-exact); regression <= 5.5e-6 relative.
- ORT vs upstream: classification queries <= 7.2e-6, context(fitted) rows <= 1.7e-4 relative to a logit scale of ~40;
  regression <= 8.1e-6 relative. (Context-row gap is larger; cause not isolated.)

## Latency / memory, ORT CPU, S = 0.8 T, classification, steady state (taskset on a 32-core Ryzen-class box)
| shape | 1 thr | 4 thr | 8 thr | peak RSS |
|---|---|---|---|---|
| 150 x 4  | 2.6 s | 0.87 s | 0.85 s | 2.3 GB |
| 400 x 12 | 10.6 s | 4.2 s | 4.4 s | 2.8 GB |
| 1000 x 50 | 86 s | 39 s | 30 s | 11.6 GB |
Session load 5-13 s. Plainly: small tables are usable on a laptop (2-3 GB); 1000x50 needs ~12 GB and 30-90 s PER QUERY SET with
no ensembling. The duplicate route processes T+S rows, which is part of the cost.

## MLX interpreter op set (static)
Not in `src/tabfm_mlx_graph_ops.inc` (68 entries): `Relu` (3 nodes; trivial). `Unique` removed. Everything else covered. Numerics NOT run.

## Quality of ONE forward (raw features, no preprocessing; this is what the engine can claim)
test acc: iris 1.000, wine 1.000, breast_cancer 0.965, synth-5cls 0.973 (LimiX-2M raw single: 0.71 on its synth-5cls);
test R2: diabetes 0.347, synth-lin-20f 0.999, synth-nonlin 0.991.
In-context (fitted) values are COPIES of the labels (diabetes in-context R2 1.000 on a noisy target): the duplicate route lets a
query attend to its identical context row. They carry no in-sample-fit information.

## Preprocessing audit
- `config/cls_default_noretrieval_v2.json` = 32 pipelines (reg: 8), each with its own transform: quantile_norm/uniform (all/5/10),
  kdi_uni, power, robust, SVD features, polynomial interactions, fingerprint features, ordinal/onehot/numeric categorical
  encodings, feature shuffle/rotate; predictions averaged. README: "LimiX-2 under the default configuration attains an Elo of 1935".
  => the leaderboard figure is the 32-member ensemble, which the engine cannot run (n_estimators>1 is rejected) and which would
  cost 32 x the latencies above.
- A single forward needs: nothing beyond raw numeric x (model standardises x on the train rows and NaN-masks); regression y z-scored
  (ddof=1) and bucket-mean inverted; dense class ids; categoricals ordinal-encoded by the caller.
- Crude 5-member ensemble (raw, quantile-normal, power, 2 feature shuffles) vs single, 3 splits: wine 0.994->1.000, breast_cancer
  0.973->0.967, synth-5cls-noisy 0.678->0.669, synth-3cls-30f 0.756->0.756, diabetes R2 0.411->0.415, synth-nonlin 0.904->0.909.
  No measurable gain on small tables; says nothing about TabArena-scale/categorical data or the SVD/polynomial members.

## NOT verified
CUDA / MLX / ROCm-MIGraphX numerics and compile (MIGraphX compile time for a 24-layer, 24k-node graph unknown, TabPFN-v3 took ~16 min
per bucket); fp16; categoricals and NULL features through the engine; n > 1000 rows or H > 50; the cause of the 1.7e-4 context-row
gap; whether the fixed positional-embedding seed is a good draw (spread across seeds not measured beyond 2 runs); any TabArena number.
