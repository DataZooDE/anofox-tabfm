# Replace ORT with candle + move anofox-tabfm to Rust — feasibility

**Question (user, 2026-07-13):** Can we replace ONNX Runtime with **candle** and
switch the extension to Rust (like `anofox-forecast` / `anofox-statistics`)?
Driver: no Python anywhere; native, statically-compilable ecosystem.

## Findings

**1. Rust DuckDB extension — PROVEN in-house, low risk.** `anofox-forecast` and
`anofox-statistics` are exactly this: a **C++ binding shell** (`src/*.cpp`:
scalar/table/aggregate function registration, chunk I/O) + a **Rust core** in a
Cargo workspace (`crates/*-core` + `crates/*-ffi`, `crate-type=["staticlib","rlib"]`,
`#[no_mangle]` C-ABI), linked by **Corrosion** (CMake↔Cargo). Same
extension-ci-tools / vcpkg / posthog-telemetry as tabfm. So the *binding + build*
side of a Rust move is a solved, house-standard pattern — not a leap.

**2. candle replacing ORT — technically yes for CPU/CUDA/Metal, with real cost.**
- candle loads **safetensors natively** (no ONNX, no Python, no export step) —
  this is the big win and directly kills the whole PyTorch→ONNX problem and the
  ORT dependency (the 271 MB source build, the vcpkg saga, the Windows System32
  DLL issue, the >2 GB MIGraphX-EP protobuf limit).
- **But candle has no tabular-FM models.** `candle-transformers` ships LLMs/ViT,
  not TabFM/Mitra/TabPFN. Each model's forward pass must be **reimplemented in
  Rust** and **parity-tested** — i.e. re-doing the S01 export spike as a Rust
  module, per family. For the current TabFM (a ~1.6 B Set Transformer) that is
  substantial; for small clean models (Mitra, 72 M) it is very tractable.
- Preprocessing (today's C++ port of the sklearn stack) must be re-ported to Rust.

**3. The decisive risk — AMD ROCm GPU on YOUR hardware.** candle's device
backends: **CPU ✓, CUDA ✓, Metal ✓ (Apple GPU — likely BETTER than the CoreML EP
that just failed Gate-1b), ROCm/HIP ✗/experimental.** The current extension has a
*working, validated* direct-MIGraphX ROCm path on the RX 9070 XT (gfx1201,
bf16 ~1.33×). Moving to candle likely means **CPU-only on this AMD GPU** until
candle's ROCm story matures. That is a concrete acceleration regression on the
exact machine this runs on. **Must be verified, not assumed.**

## Trade-off

| | Keep C++/ORT | Move to Rust/candle |
|---|---|---|
| Python | none at runtime; **Python for model export** (S01/Mitra) | **zero Python anywhere** ✓ |
| New-model onboarding | ONNX export per model (Python) | Rust forward-pass module + safetensors ✓ |
| Dependency weight | ORT (heavy; vcpkg-static saga) | candle (light, pure-Rust + optional CUDA/Metal) ✓ |
| Apple GPU | CoreML EP (failed Gate-1b) | candle Metal (likely works) ✓ |
| **AMD ROCm GPU** | **direct-MIGraphX, working** ✓ | **CPU-only (candle ROCm weak)** ✗ |
| TabFM parity | already shipped + parity-tested | **re-implement + re-parity in Rust** ✗ |
| House consistency | C++ (tabular is C++) | matches forecast/statistics ✓ |
| Effort | registry is days | **rewrite is weeks**; sunk ORT/CoreML/Win work |

## Recommendation — spike before you commit weeks

Do **not** big-bang a weeks-long rewrite on the unverified assumption that candle
serves these models with acceptable GPU. Run a **candle feasibility spike first**
(days, decision-enabling — the same gate-first discipline as the Windows crash and
CoreML Gate-1):

1. A standalone Rust crate that loads a **real** safetensors model and runs a
   forward pass in candle on **CPU**. Use **Mitra** (72 M, Apache-2.0, clean
   transformer — the model we wanted to onboard anyway): reimplement its forward
   in candle, load `autogluon/mitra-classifier` weights, predict.
2. **Parity** vs a reference on a couple of datasets (proves candle math + our
   reimplementation are correct — the real integration cost).
3. **Device probe:** does candle see/use the RX 9070 XT (ROCm/HIP)? Metal on
   Apple? Record what GPU acceleration survives.

Green spike (CPU parity + a viable GPU story) → commit to the Rust/candle
migration and the multi-model registry lands in Rust natively. Red spike (no AMD
GPU, or parity/arch pain) → we learned it in days, and the pure-C++ registry
(P1–P5, zero Python) still delivers M4 on the existing ORT engine.

## Options
- **A. candle spike first (recommended)** — Mitra-in-candle CPU parity + GPU
  probe, then decide the full migration.
- **B. Commit to the Rust/candle rewrite now** — accept the ROCm-GPU risk and the
  reimplement-TabFM cost up front.
- **C. Stay C++/ORT; ship the native registry (P1–P5)** — no Python in the
  extension; keep ONNX export (Python) as the only offline model-onboarding tool.

## Decision (2026-07-13): **C chosen.**
Keep C++/ORT (preserves the working AMD ROCm GPU path); build the pure-C++
multi-model registry per `docs/MULTI_MODEL_PLAN.md`. No Python in the extension.
Real non-TabFM model onboarding = an offline, ships-nothing ONNX export tool
(deferred). Rust/candle revisited later if the AMD-GPU story or export burden
changes.
</content>
