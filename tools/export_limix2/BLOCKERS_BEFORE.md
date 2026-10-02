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
