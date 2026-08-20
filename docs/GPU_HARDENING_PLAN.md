# GPU hardening — root causes, spikes, and the path to closing the gaps

Written 2026-08-20, after the realistic-testing round that found and fixed five
GPU bugs (see the tier-4 table in `docs/DYNAMIC_BACKENDS.md`). Those five are
fixed. This plan is about what is still wrong or missing, why — root cause per
problem, not symptom — and the cheapest experiments that decide each fix before
it is built. Platforms considered throughout: CPU, ROCm, CUDA. (CoreML is
phase 4 and out of scope here.)

Method: every fix direction below was pressure-tested two ways before being
written down — against primary sources (glibc/ORT docs and releases) and an
independent `codex exec` review round. Where a check changed the plan, the
section says so.

## 0. The problem inventory

| # | problem | severity | root cause (short) |
|---|---|---|---|
| P1 | The default GPU precision **changes answers** (ROCm bf16: 2/100, 42/2500 label flips) — violating this repo's own contract: "every backend must produce the same answers" | contract violation | per-backend perf defaults chosen before the contract was articulated |
| P2 | `anofox_tabfm_gpu_precision` is silently **ignored by the CUDA plugin** (always fp32) | correctness of configuration | plugin written minimal; no CUDA mapping for the setting was designed |
| P3 | **Only `tabfm-v1` classify/regress can ever reach a GPU.** tabicl, tabpfn, mitra, orion-bix, all SQL-registered models, and `tabfm_impute`/`tabfm_generate` cannot | functional gap | GPU dispatch requires a graph **bundled into the binary** (`graph_ext_<task>`/`graph_migraphx_<task>`) whose weights header matches — a design that cannot scale past one model |
| P4 | Debug builds cannot host the CUDA plugin (SONAME shadowing) | dev iteration | host's shared `libonnxruntime.so.1` is loaded first; glibc reuses an already-loaded SONAME without even stat'ing the plugin's copy |
| P5 | Device-switch fix causes **session-rebuild thrash**: alternating cpu/gpu queries (or two connections pinned to different devices) rebuild a multi-GB session per switch | performance cliff | the model cache has one slot per model key; the fix (70a6800) correctly rebuilds but under the same slot |
| P6 | GPU iteration is slow: ~40–90 min per CUDA pod (full build), ~27 min per new MIGraphX shape bucket, 6.5 GB weights download per pod because the fixture can't pass the weights-header gate | meta — throttles every other fix | no build caching on pods; fixture excluded from GPU paths by the same root cause as P3 |
| P7 | Neither GPU plugin is published — `tabfm_download_runtime('rocm')` errors "build it yourself", CUDA plugin likewise | distribution | no CI job builds/publishes plugin artifacts (neither needs a GPU to *build*) |
| P8 | Upstream ORT bugs unfiled (swallowed error in `LoadPluginOrProviderBridge`; unguarded `g_host` deref; prebuilt-CPU-core + CUDA-provider `Run()` crash) | ecosystem debt | never filed; all have minimal repros already written |

Two documentation corrections fold into this plan:

- `DYNAMIC_BACKENDS.md` currently says `RTLD_DEEPBIND` "would lift" the P4
  restriction. **That is wrong.** glibc dedups loaded libraries by SONAME
  before symbol scope ever matters — the loader reuses the already-loaded
  `libonnxruntime.so.1` without examining the plugin's copy, so DEEPBIND
  changes nothing here. Corrected below and in that file.
- The CUDA row's exact agreement is fp32-with-TF32: ORT's CUDA EP runs TF32
  matmuls **by default** on Ampere+ (`use_tf32` exists since ORT 1.18). Our
  measured 7.9e-07 is fp32/TF32 vs CPU fp32 — fine, but it means "fp32" on
  CUDA today is not bit-strict either, which P2's fix should make explicit.

## 1. Root causes in detail

### P1+P2 — precision is a per-backend accident, not a policy

The contract sentence at the top of `DYNAMIC_BACKENDS.md` is unambiguous: a
device switch is an optimisation, not a different model. Today:

| backend | what "default" actually runs | agreement with CPU |
|---|---|---|
| CPU | fp32 | reference |
| ROCm | **bf16** (MIGraphX quantize, `anofox_tabfm_gpu_precision` default) | 98–98.3% labels |
| CUDA | fp32 + TF32 matmuls (setting ignored) | 100% observed |

Root cause: `bf16` was chosen as the ROCm default for RDNA4 speed/VRAM before
the equivalence contract was the deliverable, and the CUDA plugin simply never
implemented the setting. Neither is a kernel bug — fp32 agrees exactly on both
backends — it is a *policy* inconsistency.

Fix direction (gated on S1/S3; the CUDA mapping below was corrected by the
codex review round — the draft mapped bf16→TF32 as an "analogue", which is
wrong: TF32 keeps fp32 storage and APIs, reduces no memory, and only rounds
matmul inputs. It is its own mode, not a bf16 substitute):
- Default `anofox_tabfm_gpu_precision` to **fp32**, meaning *strict* — on CUDA
  that includes `use_tf32=0`. The contract holds by default; faster modes are
  explicit opt-ins.
- Per-backend meaning of the setting, stated rather than implied:
  `fp32` = strict everywhere; `bf16`/`fp16` = MIGraphX quantize on ROCm and an
  **explicit "unsupported on cuda" error** (until a real fp16 graph conversion
  exists — never a silent fp32 run); a new accepted value `tf32` = CUDA's
  tensor-core fp32 rounding, rejected on ROCm.
- The measured cost of each mode goes into the docs so the user chooses with
  numbers, not vibes.

### P3 — "bundled graph or nothing" cannot scale

`TryCudaBackend`/`TryMIGraphXBackend` engage only when
`GetBundledResource("graph_ext_"+task)` exists **and**
`WeightsHeaderMatches()` against tabfm-v1's exact safetensors header. That
gate is why the random-init fixture can never reach a GPU (which in turn
forces 6.5 GB weight downloads for every GPU path test — P6), and why no other
model can either.

Root cause: the GPU graph is treated as a *property of the binary* when it is
a *property of the model*. `tabfm_register_model` already lets a model declare
graphs per task; it has no notion of a GPU-format (external-data /
migraphx-rewritten) graph.

Fix direction (gated on S4): extend the model spec — registered and built-in —
with optional `ext_graph` / `migraphx_graph` per task. Dispatch prefers a
model-provided GPU graph, falls back to the bundled one for tabfm-v1, declines
otherwise (same honest error as today). CUDA generalizes first (ORT runs any
standard graph; external-data materialization needed only for embedded-
initializer graphs); MIGraphX generalization is a separate later track (its
graphs need the Shape rewrite, so each model needs an exported variant).

This one change also unlocks: fixture-driven GPU tests (P6), other bundled
models on CUDA, and eventually `tabfm_impute`/`tabfm_generate` on GPU (they
run through the same engine once their models can reach it).

### P4 — SONAME shadowing (and the DEEPBIND correction)

Confirmed in both directions on hardware: debug (shared ORT host) fails,
release (static ORT host) works. glibc's loader dedups by SONAME per
namespace, so once *any* `libonnxruntime.so.1` is loaded, every later
dependency resolution reuses it — rpath and `RTLD_DEEPBIND` never enter into
it. Real options:

| option | mechanism | risk |
|---|---|---|
| (a) rename the plugin's ORT SONAME | equal-length in-place `.dynstr` patch of the wheel-extracted core (`libonnxruntime.so.1` → equal-length private name) + matching `DT_NEEDED` in the plugin at build time | ELF hash tables index symbol names, not SONAME strings, so equal-length patching is safe (codex-confirmed) — but **every** DSO edge naming the SONAME must be patched, including any provider library that lists it, or one missed `DT_NEEDED` re-binds to the host (S2 enumerates the edges with `readelf -d` across all shipped files) |
| (b) statically link ORT-GPU **into** the plugin, `-fvisibility=hidden` + version script exporting only `TabFMGetPluginApi` | no shared core at all; ORT's own `GetRuntimePath()` (dladdr-based) then resolves next to the *plugin*, which is exactly where the downloader puts `libonnxruntime_providers_cuda.so` | a vcpkg static ORT-GPU build is heavy; binary size; needs the dladdr claim verified (S2) |
| (c) `dlmopen` namespaces | true isolation | fragile with TLS and GPU drivers; libcuda in two namespaces is asking for trouble — dismissed unless (a) and (b) both fail |

Either (a) or (b) also removes the "CUDA needs the release build" user-facing
constraint. Until then the constraint stands and is documented.

### P5 — one cache slot per model vs per-device sessions

`TabFMState::Register` replaces same-key entries (old session dies when its
last snapshot releases). Correct, but now that reuse is device-aware, a
cpu→gpu→cpu→gpu access pattern rebuilds a multi-GB session every time, and two
connections pinned to different devices degenerate into a rebuild war.

Fix direction (gated on S6's measurement; codex concurred with per-device
entries and added the budget): per-`(model, device)` cache entries — device,
not the raw setting, and precision joins the key if Track A makes it
session-shaping — with `tabfm_unload('<model>')` evicting *all* entries of
that model so the one-name-reaches-everything invariant survives,
`tabfm_models()` showing one row per (model, device), and a conservative
`anofox_tabfm_max_loaded_sessions`-style cap so multi-device caching cannot
quietly hold N× multi-GB sessions. "Accept and document" survives only if S6
shows rebuilds are cheap (plausible for mmap'd injection, implausible for the
6.6 GB real model).

### P6 — the iteration loop is the real bottleneck

Every fix above needs GPU verification, and today each CUDA data point costs
a pod-hour. Three independent causes, three independent fixes:

1. **Pod builds from scratch.** Fix: a persistent RunPod network volume (or a
   prebaked docker image) carrying vcpkg-installed, ccache, and the DuckDB
   build tree. Target: <10 min per iteration after the first.
2. **The fixture can't reach GPU paths** (P3's gate). Fix: S4 — once a
   registered model can carry its own ext graph, the committed fixture gets
   one, and *every* GPU path test runs weightless in minutes. This also gives
   a future GPU CI something it could actually run.
3. **MIGraphX ~27 min per new shape bucket** (measured twice). Inherent, but
   `anofox_tabfm_mxr_source` staging already exists — keep a shared `.mxr`
   cache directory per arch, pre-populate the standard buckets once, and key
   the cache by model hash + bucket + arch + precision + MIGraphX version
   (today's key omits the MIGraphX version — a ROCm upgrade would silently
   load stale programs).
4. **Plugin changes rebuild DuckDB** (codex's structural addition): both
   plugins are standalone `.so`s with no DuckDB dependency, yet today they
   build inside the extension superbuild. A tiny standalone target
   (`make plugins`) that compiles just `tabfm_cuda_plugin.cpp` /
   `tabfm_migraphx_plugin.cpp` against their runtimes turns a plugin
   iteration into seconds, locally and on pods.

### P7 — plugins are buildable in CI, today

Neither plugin needs a GPU to *build*: the CUDA plugin needs only the ORT-GPU
archive (a download), the MIGraphX plugin needs only `migraphx-dev` from the
ROCm apt repo (a docker image). A release workflow can build both, attach them
to GitHub releases with sha256s, and `tabfm_download_runtime('cuda'|'rocm')`
learns to fetch the plugin alongside the runtime. This turns "build it
yourself" into one SQL call on both GPU platforms.

### P8 — upstream debts

Three ORT findings with repros already in hand: the swallowed
`provider_library->Load()` error in `LoadPluginOrProviderBridge` (turns a
diagnosable misconfiguration into a SIGSEGV in a global constructor), the
unguarded `g_host` deref in `provider_bridge_provider.cc:91`, and the
prebuilt-CPU-core + CUDA-provider `Run()` crash. Filing is cheap; the repros
are ~20 lines each. Separately: ORT now publishes an official **CUDA Plugin EP
v0.1.0** package and 1.28.1/1.29.0 exist — if the true plugin-EP ABI works
against a statically linked core (the provider *bridge* provably does not),
it could eventually replace our custom CUDA plugin. Worth one spike (S5), not
a bet.

## 2. Spikes — small, falsifiable, each gating a decision

| id | hypothesis | method | cost | gates |
|---|---|---|---|---|
| S1 | `use_tf32=0` gives strict CPU parity on CUDA at acceptable speed; the TF32 speedup is worth exposing as its own mode | fixture graph + real weights on one pod session: run TF32-off vs TF32-on, record max-rel + wall time | 1 pod session (fast once P6.1 exists) | P1/P2 defaults + whether `tf32` earns its keep |
| S2 | ✅ **done, with a design correction the experiment forced.** A two-ORT process on this box ran the full (SONAME) × (DEEPBIND) matrix: same+plain → host's ORT (dedup); same+deepbind → host's (dedup precedes everything); renamed+plain → **still the host's** (symbol interposition — the renamed core loads but the plugin binds to the global scope); renamed+deepbind → the plugin's own copy. So dedup and interposition are independent failure layers and the fix is **rename AND `RTLD_DEEPBIND` together** — the plan's earlier "DEEPBIND changes nothing" was right about dedup and wrong as a verdict. Also measured: the string occurs exactly once in the core (its own SONAME), `providers_shared` and the CUDA provider carry no ORT `DT_NEEDED`, so the only patch edges are the core's SONAME and the plugin's link line. DEEPBIND is safe for this ABI (no allocation crosses the boundary). The loader half ships now; the rename ships with the runtime in P7 (extractor byte-patch + plugin linked against the renamed copy, name `libanofoxort_gpu.so`, equal length). Static-link loses on build cost and is dropped. Known boundary, documented not solved: a debug host that ALSO loads classic ORT providers itself could still share `providers_shared` state with the plugin's runtime | two-ORT matrix, local, no GPU needed | done | P4 design settled; P7 unblocked with an exact recipe |
| S3 | bf16 label flips are near-ties only | dump logits for the 42 flipped rows of the 2500-row run on ROCm; measure margin distribution | <1 h local | whether bf16 stays offered as-is, or gets a margin warning |
| S4 | ✅ **done** — a registered model carrying its own GPU graph runs on the plugin unmodified. Proven on gfx1201: `REGISTERED_SERVED_BY=rocm:0`, predictions identical to the bundled path (`PATHS_DISAGREE=0`, same weights). Two sub-findings: **U1** — MIGraphX cannot run a plain ext-format graph (fixture fails at eval), so `ext_graph` and `migraphx_graph` are separate spec fields with no cross-fallback; and a sixth realistic-testing bug — `tabfm_models()` was blind to registered models whose weights live outside the cache, so the `device` proof column could not see the very model class this enables (fixed). CUDA/fixture-weightless confirmation still pending a pod run | implemented as Track C v1 (`SelectGpuGraph` + spec fields + dispatch), tests in `test_tabfm_model_spec.cpp` / `tabfm_gpu_graphs.test`, scenario `tools/gpu_test/scenarios/registered_model_gpu.sql` | done | P3 design settled; P6.2 unlocked pending a migraphx-compatible fixture export |
| S5 | ORT ≥1.28.1 / official CUDA Plugin EP v0.1.0 works against a statically linked core (true plugin ABI, not provider bridge); our filed bugs may already be fixed | standalone C host, one pod session; re-run the 20-line dlopen repros against 1.29 | 1 pod session | long-term CUDA strategy; P8 filing text |
| S6 | ✅ **done** — measured 18–27 s of pure rebuild per device alternation (cpu 26.8 s / rocm 24.0 s / cpu 18.1 s on bigfox, real weights), killing "accept and document". P5 implemented on those numbers: sessions cache per (model, device, precision); after the fix the same alternations cost 1.47 s / 0.25 s / 1.46 s (inference, not rebuild) and `tabfm_models()` shows one model loaded on cpu and rocm:0 at once. `tabfm_unload` frees every configuration by one name (two-level map, by construction), and `anofox_tabfm_max_sessions` (default 4, oldest-evicted) bounds total resident sessions — codex's budget requirement | before/after via `s6_rebuild_cost` alternation, state-level tests pin coexistence, replacement, all-config unload and cap eviction | done | P5 shipped |
| S7 | a RunPod network volume + ccache gets CUDA iteration under 10 min | set up once, measure second iteration | half a day + ~$5 | P6.1; every later pod spike inherits it |

Order (revised after the user's "hardest, most critical first" call — S4 was
pulled to the front and is done): **S7, S6, S3 next** (cheap; S7 makes every
later pod spike faster), then S2, then S1 and S5 on the fast pod loop.

## 3. Implementation tracks (gated on spikes)

- **Track A — precision policy**: ✅ **implemented, ROCm-verified**. Default is
  fp32; the validator accepts `tf32`; the MIGraphX plugin rejects `tf32`
  ("CUDA-only mode") and the CUDA plugin rejects `bf16`/`fp16` ("MIGraphX
  modes on ROCm") — a mode either happens or errors, never a silent fp32 run.
  CUDA maps fp32→`use_tf32=0`, tf32→`use_tf32=1` via the V2 provider options.
  Implementing it surfaced that precision had become session-shaping, i.e. the
  device-switch bug one setting over: `CanReuseSession` now keys on precision
  too ("" for CPU sessions, which the setting does not shape), pinned by the
  same exhaustive predicate tests. Verified on gfx1201 at DEFAULT settings:
  `GPU_SERVED_BY=rocm:0`, `DEFAULT_DISAGREEMENTS=0`, and `tf32` on ROCm errors
  naming the platform. **CUDA half now machine-verified** (RTX-class pod,
  plugin-only harness, no DuckDB build): fp32 (use_tf32=0) and tf32 both
  create, run, and agree with CPU within the harness's 1e-3 assertion, and
  bf16 is rejected at create with the exact "MIGraphX modes on ROCm" message.
  Caveat kept honest: the per-mode agreement deltas were truncated out of the
  captured log (a tail cut), so the record is behavioural (exit codes + the
  aggregate PASSED gate), not numeric; S1's cost/parity numbers remain worth
  taking when a pod is next warm.
- **Track B — iteration infrastructure** (after S7, S4): persistent pod
  volume; fixture-as-GPU-model; `make gpu_check` running the scenario matrix
  on whatever hardware is present and printing the tier table.
- **Track C — any-model GPU** (after S4): model-spec GPU-graph fields;
  dispatch prefers model-provided graphs; CUDA first; per-model MIGraphX
  exports as a follow-on; impute/generate inherit automatically. Two risks
  codex added, both now requirements: (1) "runs on the CUDA EP" must mean
  *fully or explicitly* — set ORT's `session.disable_cpu_ep_fallback` (the
  config key already exists in 1.28) or surface partial placement, so an
  unsupported-op graph cannot silently run mostly on CPU while
  `device=cuda:0` implies otherwise; (2) SQL-registered graph/weights paths
  must be validated against the model's base_dir — registration must not
  become an arbitrary-filesystem-read primitive.
- **Track D — distribution + dev builds** (after S2): CI builds and publishes
  both plugins; `tabfm_download_runtime` fetches them; SONAME fix lands so
  debug builds work too.
- **Track E — cache policy**: ✅ **shipped with S6's numbers** (see the S6 row;
  18–27 s alternation rebuilds became 0.25–1.5 s cache hits, verified on
  gfx1201 with real weights).
- **Track F — upstream** (after S5): file the three ORT issues with repros;
  re-evaluate the official CUDA Plugin EP as a replacement backend.

Each track ends the same way, learned the hard way this week: the change is
proven by a scenario that prints `*_SERVED_BY`, on real hardware, on every
platform it touches — CPU always, ROCm locally, CUDA on the fast pod loop.

## 4. The iteration loop, end state

| loop | hardware | latency | runs |
|---|---|---|---|
| unit + sqllogic | CPU (CI + local) | seconds–minutes | every commit; includes `CanReuseSession` and the loader/ABI tests |
| ROCm scenarios | bigfox gfx1201 | minutes (cached buckets) | every GPU-touching change, locally |
| CUDA scenarios | RunPod + persistent volume | target <10 min warm | per GPU-touching change; weightless via fixture once S4 lands |
| full real-weights matrix | bigfox + one pod | ~1 h | before merge of a GPU-touching PR; produces the tier table in `DYNAMIC_BACKENDS.md` |

## 5. Sequencing

1. S7 (pod volume) + S3 (margins) + S6 (rebuild cost) — all cheap, two local.
2. S2 (SONAME/static) and S4 (model-provided graphs) in parallel — these are
   the two structural decisions.
3. Track A (precision) — small diff, big contract value, verify on both GPUs.
4. Tracks C, D on the now-fast loop; E per S6; F whenever a pod session is
   already warm.
5. Exit criteria for calling this done: default settings give **identical
   answers on all three platforms** (asserted by scenarios, not claimed); any
   registered model with an ext graph runs on CUDA; both plugins are one SQL
   call away; debug builds can host the CUDA plugin; the tier table has no
   "not measured" cells for CPU/ROCm/CUDA.

## Sources

- glibc dlopen/dlmopen semantics (SONAME dedup, DEEPBIND scope):
  [man7 dlopen(3)](https://man7.org/linux/man-pages/man3/dlopen.3.html),
  [Postgres-list discussion of DEEPBIND conflicts](https://postgrespro.com/list/thread-id/1857027)
- ORT CUDA EP `use_tf32` (default-on TF32 since Ampere, flag since 1.18):
  [CUDA Execution Provider docs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
  [discussion #20193](https://github.com/microsoft/onnxruntime/discussions/20193)
- ORT releases incl. official CUDA Plugin EP package:
  [microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
