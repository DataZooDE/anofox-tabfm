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

> **Status — see `docs/MLX_SPIKE_RESULTS.md` for the verdicts.** S-M1, S-M2 and
> S-M4 are executed on an M3; S-M3 is next. Three of this plan's hypotheses
> died on contact, exactly as the paragraph above anticipated, and the sections
> below are left as written so the corrections stay legible:
>
> 1. **Route 1 does not exist.** There is no ONNX→MLX importer —
>    `ml-explore/mlx-onnx` is an empty placeholder repo, and the `mlx-onnx` name
>    on PyPI is an unrelated project that exports in the opposite direction.
>    The hand-port (route 2) is the only path; it took hours, not 1–2 days.
> 2. **The parity tolerance below is unmeetable, by anything.** "rtol 1e-4" on
>    logits is failed by the torch reference implementation against its own ONNX
>    export (9.3e-04). Parity is now judged on post-softmax probabilities plus
>    argmax. MLX passes, and is closer to the reference than ORT is.
> 3. **Per-shape compile cost is a non-issue** (217 ms, not ROCm's minutes), so
>    the shape-bucket / precompile machinery must *not* be generalized to MLX.
>
> Also obsolete: "no Apple Silicon is available to this environment". The work
> now runs directly on an M3, so the spikes are executed rather than prepared.

## Scope change (2026-08-22): every supported model, not just mitra

The requirement is now that **all registered models run on MLX**, matching what
cpu already does: `tabfm-v1`, `mitra`, `tabpfn-v2`, `tabpfn-v2-5`, `tabpfn-v3`,
`tabicl-v2`, `orion-bix` — classification and regression.

That inverts this plan's route decision, and the deciding number was measured
rather than argued. Across the 13 shipped `graph_ext_*` graphs:

| | |
|---|---|
| models to cover | 7 (13 model×task graphs) |
| nodes per graph | 4,526 – 12,424 |
| distinct ops per graph | 33 – 48 |
| **union of op kinds, all graphs** | **64** |

- **Route 2 (hand-port), scaled to the new scope**, is 7 architectures ported by
  hand. Each is a *second implementation of a model we already ship* — the cost
  this plan already flags for mitra alone — so the drift surface multiplies by
  seven, and each needs its own parity harness against its own weights. The
  plan's "one forward per model *family*" hoped to amortise this, but the
  families differ inside too (tabpfn-v2-5 and v3 are 12k-node graphs, roughly
  double mitra).
- **Route 3 (an ONNX interpreter over MLX ops)** is **one** implementation
  covering all 13 graphs and any model added later, with *no drift by
  construction*: it executes the same graph every other backend executes. The
  cost is 64 op kinds, most of them elementwise and 1:1 with MLX.

So route 3 becomes the plan of record for coverage, and the mitra hand-port
(already shipped and verified) stays as the fast path and as the **oracle**: it
is an independent implementation to check the interpreter against on the one
model where both exist. This plan predicted the condition exactly — "only worth
it if the hand-port turns out to fight model drift" — and seven hand-ports is
that condition.

Sequencing, each gated on parity against the ORT CPU golden for that model:

1. Interpreter core + mitra's 33 ops → must reproduce the hand-port's logits.
2. `tabfm-v1` (41/38 ops), the largest weights (6.6 GB) — also the unified-memory test.
3. `orion-bix` (34), `tabicl-v2` (42/46).
4. `tabpfn-v2` (44/46), `tabpfn-v2-5` (46/47), `tabpfn-v3` (48/47).

The hard ops are known up front and are the ones this plan named: `ScatterND`,
`GatherND`, `ScatterElements`, `SplitToSequence`/`SequenceAt`, `Einsum`, `Pad`,
`CumSum`, `TopK`, plus the `Shape`/`Slice` dynamic-shape patterns.

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
