# Apple Silicon acceleration via the CoreML EP — plan

**Status:** not started. This scopes adding GPU/Neural-Engine acceleration on
Apple Silicon through ONNX Runtime's **CoreML execution provider** (the only
general ORT path to Apple hardware — there is no Metal/MPS EP for arbitrary
models). It follows the existing `cuda` / `rocm` flavor pattern.

## Current state (grounded)

- macOS ships the **`cpu` flavor**: `cmake/ort.cmake` fetches the
  `osx-universal2` prebuilt (CPU EP only). Apple Silicon inference runs on the
  CPU EP — multi-threaded + NEON-vectorized, but never the GPU or ANE.
- Only two GPU providers are wired in `AppendExecutionProviders`
  (`tabfm_ort_engine.cpp`): `AppendExecutionProvider_CUDA` and
  `_MIGraphX`. `anofox_tabfm_device` accepts only `auto|cpu|cuda|rocm|migraphx`
  (`tabfm_settings.cpp`).
- **Reusable infrastructure already in place** (built for MIGraphX, directly
  applicable to CoreML's static-shape preference):
  - shape buckets — `MIGraphXShapeBucket` + `T_BUCKETS/H_BUCKETS`
    (`tabfm_devices.cpp:431`),
  - per-device precision — `engine_profiles` in the manifest, keyed by device,
    default `cpu:f32` (`tabfm_manifest.cpp:152`; `ParseEngineProfiles`),
  - f32→f16/bf16 injection casts — `CastF32ToF16` / `CastF32ToBF16`
    (`tabfm_ort_engine.cpp`).

## The blocker to settle FIRST — op coverage (measured)

The classification graph (`resources/graph_classification.onnx`) is **8 760
nodes / 41 distinct op types**, and it is **dynamic-shape-heavy** — the CoreML
EP's worst case:

| category | ops (counts) |
|---|---|
| dense core (CoreML-friendly) | MatMul 385, Add 787, Mul 1021, Reshape 1050, Transpose 515, Concat 346, Softmax 42, ReduceMean 341, Sqrt 425, Pow 342 |
| **dynamic-shape / control (risky)** | **Shape 93, Slice 152, Where 121, Range 22, Expand 42, GatherElements 6, Mod 6, Unsqueeze 164, CastLike 223** |
| **likely-unsupported** | **IsNaN 37, IsInf 2**, And 3, Greater 43, Less 9, Equal 1 |

The CoreML EP **partitions** the graph: supported subgraphs run on ANE/GPU,
everything else falls back to the CPU EP, with a **data copy at every partition
boundary**. With ~93 `Shape` + `Range` + dynamic `Slice`/`Where`/`IsNaN`
scattered throughout the attention/masking logic, this graph will likely shatter
into many small partitions. Fragmentation + copy overhead frequently makes
CoreML **slower** than the pure CPU EP for graphs like this.

> **This is the make-or-break.** The upstream op-support lists are already in the
> tree — `tools/export_onnx/.venv/.../onnxruntime/tools/mobile_helpers/
> coreml_supported_mlprogram_ops.md` (and `_neuralnetwork_ops.md`). A first pass
> should diff our 41 op types against them, but the honest signal only comes from
> running the EP on a Mac and reading the partition report (below).

## Gates — cheap, before any flavor work

- **Gate 0 — does the prebuilt carry CoreML?** Confirm the `osx-universal2`
  release archive exports `OrtSessionOptionsAppendExecutionProvider_CoreML`
  (`nm`/`otool` on `libonnxruntime.dylib`). If not, CoreML needs an ORT built
  with `--use_coreml` (or the objc package) — a much bigger acquisition story.
  The provider header confirms the API exists
  (`.../providers/coreml/coreml_provider_factory.h`).
- **Gate 1 — partitioning + wall-time (on a real Mac, ~1 day).** Load
  `graph_classification.onnx` under the CoreML EP with
  `ORT_LOGGING_LEVEL_VERBOSE` and read how many nodes CoreML claims vs. how many
  fall back, and the partition count. Then time a real predict (iris/churn from
  `examples/`) CoreML vs. CPU. **Proceed only if CoreML both claims the bulk of
  the compute AND beats the CPU EP wall-clock.** If it fragments or loses, stop
  here and record the negative result (like the ROCm `usable=false` honesty).
- **Gate 2 — precision + size.** Confirm fp16 parity vs. the f32 CPU reference on
  the parity datasets, and that CoreML's session-init model conversion handles
  the ~3.3 GB fp16 weights (analogous to the >2 GB issue that pushed AMD off
  ORT's MIGraphX EP into a direct backend — verify it does *not* recur here).

## Implementation — the four seams (only if the gates pass)

Mirrors `cuda`/`rocm`; each is a small, well-bounded change:

1. **Flavor / acquisition** (`cmake/ort.cmake`): add `TABFM_FLAVOR=coreml`. Same
   `osx-universal2` archive as the cpu flavor, but define `TABFM_EP_COREML=1` and
   link `-framework CoreML -framework Foundation`. Keeps the default macOS **cpu**
   flavor pure CPU (community-extension eligibility, HLD D9). (Alternative: fold
   CoreML into the macOS cpu build behind a runtime opt-in — rejected to preserve
   a pure-CPU community lane.)
2. **EP append** (`AppendExecutionProviders`, `tabfm_ort_engine.cpp`): a
   `device == "coreml"` branch under `#ifdef TABFM_EP_COREML` calling
   `AppendExecutionProvider("CoreML", opts)` (MLProgram backend, fp16 policy,
   `MLComputeUnits` = All). Static-shape inputs via the existing shape buckets.
3. **Device discovery** (`tabfm_devices.cpp`, under `TABFM_EP_COREML`): emit a
   `coreml:0` row — `ep = "CoreMLExecutionProvider"`, `arch` = the Apple SoC
   (`machdep.cpu.brand_string`, already read via `<sys/sysctl.h>`), `usable` =
   EP registered AND Gate-1 held. Honest `usable=false` if not.
4. **Setting + resolution** (`tabfm_settings.cpp`): add `"coreml"` to the
   `anofox_tabfm_device` validator and to `'auto'` resolution (prefer coreml on
   Apple Silicon when usable, else cpu). Add a `coreml` **engine profile**
   (fp16) to the real manifests.

Plus: reuse `MIGraphXShapeBucket` (rename to a neutral `TabFMShapeBucket` helper
if shared), and the f16 injection cast for the coreml profile.

## Effort & verdict

- Gates 0–2: ~1–2 days on a Mac (the decision point).
- Full four-seam integration + parity + a `tabfm_devices()` test: ~3–5 days
  **if** the gates pass.

**Verdict:** cleanly wireable (the pattern and infra exist), but the payoff is
genuinely uncertain — the graph's dynamic-shape profile is CoreML's worst case,
so heavy CPU fallback and fragmentation are the likely outcome. **Gate 1 is the
whole ballgame; do not build the flavor before measuring it on real hardware.**
A larger, higher-ceiling alternative (out of scope here) is a *direct*
CoreML/MPSGraph backend — the Apple analog of the direct-MIGraphX backend — which
sidesteps ORT's partitioner but is a multi-week effort.
