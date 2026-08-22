# Apple MLX backend — implementation plan

Goal: run tabfm inference on Apple-Silicon Macs through
[MLX](https://github.com/ml-explore/mlx) (Metal-backed, unified memory), as a
**backend plugin** behind the existing `tabfm_plugin_abi.h` — the same shape
that made ROCm and CUDA work without touching the shipped binary. This is NOT
phase 4 (CoreML, still excluded): CoreML is an in-process ORT EP needing an
ORT rebuild; MLX is a standalone runtime we drive directly, exactly like
MIGraphX.

Written 2026-08-22, before any code. Everything below that is not marked
"known" is a hypothesis a spike must confirm on real Apple hardware — the
GPU-hardening work earned that rule (three of its plan's claims died on
contact with hardware).

## What is already in place (known, no work)

- **The plugin ABI is platform-neutral C** (`tabfm_plugin_abi.h`, abi_version
  1) and the loader's non-Windows path is plain `dlopen` — a macOS `.dylib`
  loads today. `RTLD_DEEPBIND` is already `#ifdef`-guarded (macOS two-level
  namespaces make it unnecessary).
- **Model artifacts**: every model's weights are single-file F32 safetensors
  in the cache; the tensor maps are transform-free; the ext graphs bake
  byte offsets into them. An MLX backend can mmap the same safetensors and
  slice tensors by the same offsets — no new export needed for weights.
- **The engine dispatch** is one `TryMlxBackend` away: `SelectGpuGraph`-style
  gating, `SET anofox_tabfm_ep_path`, per-(device, precision) session cache,
  and the explicit-device-errors contract all generalize.
- **The verification doctrine**: served-by proof, CPU-parity comparison,
  scenario suite — all reusable as-is on a Mac (`tools/gpu_test/scenarios/`
  plus `equivalence.py`'s plugin route).

## The one big open question: how does the graph get executed?

MLX has **no ONNX importer** in core. Three candidate routes, in the order
they should be tried:

1. **`mlx-onnx`** (ml-explore's experimental ONNX-to-MLX converter, Python).
   If it can load our ext graphs (opset 18, dynamo-exported, external data),
   the Python route proves feasibility in a day — but the plugin must be
   C/C++, so this is a *spike vehicle*, not the product path.
2. **Hand-ported forward in MLX C++** (`mlx` is a C++ library; `mlx-c` wraps
   it in C). The models are standard transformer encoders; mitra is 392
   tensors of vanilla attention/MLP. A hand-port per model *family* (not per
   model) is the realistic product path: one forward for the
   train_size-scalar family (tabfm-v1, mitra), one for the single_eval_pos
   family (TabPFN/TabICL/Orion) later. Weights come straight from the
   safetensors via the committed tensor maps.
3. **A minimal ONNX interpreter over MLX ops** — walk the graph, dispatch
   ~40 op kinds to MLX. More general than 2, more work than it looks
   (ScatterND, the Shape/Slice patterns, dynamic dims); only worth it if the
   hand-port turns out to fight model drift.

The plan assumes 2 wins, scoped to **mitra first** (smallest, fully
permissive license, already the multi-backend pathfinder), tabfm-v1 second.

## Spikes (each on the target Mac, each with a written verdict)

- **S-M1 — environment + mlx-onnx feasibility** (half day). On the Mac:
  `pip install mlx mlx-onnx`; try loading `graph_ext_mitra_classification.onnx`
  (external data materialized from the real safetensors) and run one forward;
  compare logits to ORT CPU at rtol 1e-4. Verdict decides whether route 1
  gives us a golden reference beyond ORT, and whether MLX's op coverage has
  gaps for these graphs.
- **S-M2 — mitra forward hand-port in Python MLX** (1–2 days). Port the
  forward from `tools/export_mitra/src/export_mitra/mitra_model_patched.py`
  to MLX Python, loading weights via the tensor map. Success = logits match
  ORT CPU (rtol 1e-4) on the fixture-sized workload AND on the real weights.
  This de-risks the math before any C++ exists.
- **S-M3 — mlx-c plugin skeleton** (1 day). A `.dylib` exporting
  `TabFMGetPluginApi` that links `mlx-c`, loads the safetensors, and runs the
  S-M2 forward in C. Load it through the extension's real loader
  (`plugin_load_check.c` compiles on macOS unchanged); create/run/destroy
  through the ABI. Verdict: mlx-c maturity (it lags the C++ API; if it
  blocks, the plugin links the C++ `libmlx` directly — it is a plugin, C++
  ABI exposure is contained).
- **S-M4 — performance reality check** (half day). Warm per-predict at
  T=100/T=2500 vs the same Mac's CPU ORT. MLX is lazy/graph-compiled;
  measure compile-on-first-shape cost too (ROCm's 25-minute lesson: measure
  before promising). Unified memory should make the 6.5 GB tabfm-v1 viable
  on 16 GB+ Macs — verify, don't assume.

## Product phases (after spikes green)

1. **M1 — mitra classification+regression on MLX** behind the ABI:
   `src/tabfm_mlx_plugin.cpp` (or `.mm` if Metal setup needs it), device
   `'mlx'` in ResolveDevice (discovery: Apple Silicon + Metal available —
   compile-time `__APPLE__` + runtime check, no SDK probing needed),
   `TryMlxBackend` in the engine with the same explicit-device error
   contract, precision `fp32` default with `fp16`/`bf16` as opt-ins
   (MLX natively supports both; run the flip-rate measurement like C4).
2. **M2 — tabfm-v1** (same family, bigger): the unified-memory large-model
   case; measure, then decide whether the single_eval_pos family (M3) is
   worth a third forward or waits for route 3.
3. **CI + distribution**: a `mlx-plugin` job in `gpu_plugins.yml` on a
   `macos-14`+ runner (builds + dlopen load-check without inference, exactly
   like the CUDA/MIGraphX jobs); the `.dylib` joins the release assets and
   `tabfm_download_runtime('mlx')` gets an entry. The macOS smoke test
   gains the load-check.
4. **Verification on your Mac**: the scenario suite + equivalence runs are
   manual-on-hardware like ROCm — served-by proof, cpu-vs-mlx parity per
   model, concurrency scenario, and the examples run with `device='mlx'`.

## Constraints and honest unknowns

- **Hardware in the loop**: no Apple Silicon is available to this
  environment; every spike and verification runs on your macOS machine.
  The plan front-loads Python spikes so your time on the Mac is short,
  scripted, and marker-verified (the RunPod pattern, minus the billing).
- **mlx-c maturity** is the biggest product risk (S-M3 probes it early).
- **Dynamic shapes**: MLX recompiles per shape like MIGraphX; if compile
  cost is non-trivial, the existing shape-bucket + precompile machinery
  (`tabfm_gpu_precompile`, `mxr_source` analogue) generalizes.
- **The macOS artifact today is CPU-only by design** — nothing in this plan
  changes the shipped binary until M1 lands behind the same "never required
  for the cpu build" rule as the other plugins.

## Decision points for you

1. Approve the spike order (S-M1..S-M4 on your Mac; I prepare each as a
   single scripted, marker-printing run you execute and paste back).
2. mitra-first scoping (vs starting with tabfm-v1).
3. Whether route 3 (ONNX interpreter) is worth pursuing if S-M2's hand-port
   is painful — decide after S-M2's verdict, not before.
