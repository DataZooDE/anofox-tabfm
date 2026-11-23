# Apple Silicon (CoreML EP) — local implementation plan (red-green TDD)

Localizes `docs/COREML_PLAN.md` to **this machine** (Apple M3, macOS 26.5,
arm64) and sequences it as red-green TDD cycles. Read `COREML_PLAN.md` first —
this file assumes its "four seams" framing and only adds: what's already
verified here, how to measure the decision gate locally, and the exact
failing-test-first steps.

## Status ledger (verified 2026-07-05 on this M3)

| Item | State | Evidence |
|---|---|---|
| `make debug` builds, extension loads | ✅ | 30 `tabfm_*`/`anofox_tabfm_*` functions register |
| **Gate 0** — prebuilt carries CoreML | ✅ **PASS** | `nm -gU libonnxruntime.1.23.2.dylib` → `_OrtSessionOptionsAppendExecutionProvider_CoreML`; `coreml_provider_factory.h` present in the fetched archive |
| Custom ORT build needed | ❌ no | Gate 0 passed — same `osx-universal2` archive as the cpu flavor |
| Real graph topology available weight-free | ✅ | `resources/graph_classification.onnx` (8 760 nodes, 1.1 MB stub) — partition report needs topology only, no weights |
| Fixture runs ORT forward pass on macOS | ✅ | `test/fixtures/` random-init model; ORT forward-pass tests run on non-Windows |

## Implementation status (all four seams landed, red-green — 2026-07-05)

| Cycle | Seam | Status | Evidence |
|---|---|---|---|
| A | `ResolveDevice` coreml lane | ✅ green | `[devices]` 50 assertions; cpu build |
| E | `TABFM_FLAVOR=coreml` in `cmake/ort.cmake` | ✅ green | `TABFM_FLAVOR=coreml make debug` builds + links `-framework CoreML/Foundation` |
| B | EP append branch | ✅ green | `[.coreml]` fixture parity matches CPU golden (rtol 1e-4) on the coreml build |
| C | `coreml:0` device discovery | ✅ green | `tabfm_devices()` → `coreml:0 / CoreMLExecutionProvider / Apple M3 / usable=true` |
| D | setting validator + description | ✅ green | `settings.test`: `SET anofox_tabfm_device='coreml'` ok |

**Gate 1a (partial, fixture):** under the CoreML EP the fixture MLP runs correctly
but CoreML **fails to compile at least one subgraph** (`ios16.reduce_sum`,
"Invalid tensor rank 0") and ORT **falls back to the CPU EP per-partition** —
output still matches golden. This confirms the partition+fallback path works and
previews the fragmentation the parent plan predicts for the real 8 760-node graph.

## Gate 1b — MEASURED on this M3 (2026-07-06): ❌ CoreML cannot run the real model

Ran `examples/regression_tips.sql` (real 6.1 GB regression `model.safetensors`
downloaded via the built-in HF manifest, license accepted) on a **release**
coreml-flavor build — the coreml binary carries both EPs, so `SET
anofox_tabfm_device` picks the lane. tips dataset, 56 scored rows:

| Device | Predict wall-time | Outcome | RMSE | MAE |
|---|---|---|---|---|
| `cpu` (ORT CPU EP) | **27.6 s** | ✅ ran (beats mean baseline: MSE 0.97 vs 1.68) | 0.986 | 0.730 |
| `coreml` (ORT CoreML EP) | 18.1 s → **hard error** | ❌ **session init fails** | — | — |

CoreML **throws during session initialization** on the real graph — it does NOT
fall back gracefully:

```
[onnxruntime] Exception during initialization: common.h:32 HandleNegativeAxis
  axis 2 is not in valid range [-2,1]
Invalid Input Error: ONNX Runtime failed to create the session for 'regression'
  (ORT error code 6)
```

The **identical graph runs fine on the CPU EP** (27.6 s, correct output), so this
is specifically a CoreML-EP limitation: its partitioner/compiler chokes on an
axis/rank mismatch in one of the dynamic-shape ops. The small random-init fixture
runs under CoreML (test suite `[.coreml]` case) only because it lacks those ops.

**Verdict: on Apple Silicon (M3), ORT's CoreML EP is a dead end for the real
TabFM graph** — analogous to how ORT's MIGraphX EP was a dead end for ROCm (which
needed the direct MIGraphX backend, `GPU_AND_MEMORY_FINDINGS.md`). Do **not**
enable/recommend the coreml flavor for production on the strength of the EP.

**Follow-ups this dictates:**
- **Robustness (recommended next):** wrap the coreml session-init; on failure,
  fall back to the CPU EP so `SET anofox_tabfm_device='coreml'` **degrades to a
  warning + CPU run** instead of hard-erroring (`AppendExecutionProviders` /
  `LoadOrGetSession` in `tabfm_ort_engine.cpp`).
- **Honesty on `usable`:** `coreml:0 / usable=true` today means only "EP registered
  on Apple Silicon" — this Gate-1b run shows it's optimistic for the real model.
  Gate `usable` on a successful trial session-init, or mark `coreml` experimental.
  Correspondingly, drop `coreml` from the `auto` branch of `ResolveDevice` (keep
  it opt-in only) until an EP path actually runs the model.
- **Real acceleration** (if still wanted) = a **direct CoreML/MPSGraph backend**,
  the Apple analog of `tabfm_migraphx.cpp` — sidesteps ORT's partitioner, but a
  multi-week effort. This is the only path that could make TabFM use the ANE/GPU.
- **coreml fp16 engine profile** — moot unless a working EP/backend exists.

Known caveat surfaced: fixed a **pre-existing** ASan stack-use-after-scope in the
`tabfm_state` re-register test (unrelated to CoreML; `state` outlived the
stack `freed_*` flags its deleter writes) so the debug suite is green on this Mac.

Gate 0 passing means the acquisition story is trivial. **The remaining risk is
entirely Gate 1** (does CoreML actually beat the CPU EP on this graph, or does
it shatter into CPU-fallback partitions). Per the parent plan: *do not build the
flavor before Gate 1 holds.* This plan therefore front-loads a measurement spike
and makes every seam conditional on it.

---

## Phase 0 — Decision gate (measurement spike, NOT shipped code)

Not TDD (it's investigation). Build a throwaway harness under
`tools/coreml_probe/` (or a `[.coreml]`-tagged, hidden-by-default Catch2 case) —
it must not enter the default `make test_debug` run.

### Gate 1a — partition report on the real topology (possible NOW, no weights)
- Load `resources/graph_classification.onnx` with the CoreML EP appended
  (`SessionOptionsAppendExecutionProvider("CoreML", {{"MLComputeUnits","ALL"}})`)
  under `ORT_LOGGING_LEVEL_VERBOSE`.
- Capture: nodes CoreML claims vs. nodes falling back to CPU, and the partition
  count. The parent plan predicts fragmentation from ~93 `Shape` + `Range` +
  dynamic `Slice`/`Where` + `IsNaN`/`IsInf`.
- **Record the numbers in `docs/COREML_PLAN.md`** (like the ROCm honesty note),
  regardless of outcome.

### Gate 1b — wall-time (needs real weights → license gate, user-driven)
- `SELECT tabfm_download('classification')` (accepts the Google license — user
  must run it; no weights in-repo), then time `examples/classification_iris.sql`
  and `classification_churn.sql`: CoreML vs. CPU EP, warm session.
- **Proceed to Phase 1 only if CoreML claims the bulk of compute AND beats CPU
  wall-clock.** Otherwise stop, record `usable=false`, ship nothing.

### Gate 2 — fp16 parity
- Fixture parity (random weights, runnable now) proves EP correctness.
- Real-model fp16 parity vs. the f32 CPU reference on the parity datasets;
  confirm CoreML's model conversion handles ~3.3 GB fp16 (the >2 GB class of
  problem that pushed ROCm off ORT's EP — verify it does *not* recur).

**Exit criterion:** a one-paragraph verdict appended to `COREML_PLAN.md`. Green
→ Phase 1. Red → done (negative result is a valid outcome).

---

## Phase 1 — Implementation (four seams, each a red-green cycle)

Only if Phase 0 is green. Order chosen so each cycle compiles and tests on its
own. `TABFM_EP_COREML` is the compile gate; the default macOS **cpu** flavor
stays pure-CPU (community-extension eligibility, HLD D9).

### ⚠ File-ownership coordination (CLAUDE.md rule 2)
Three of the touch points are **scaffold-owned** — coordinate before editing:
- `src/tabfm_settings.cpp` (device validator — Seam 4)
- `CMakeLists.txt` source list / test-source list (framework link + new test TU)

Module-owned, edit freely: `cmake/ort.cmake`, `src/tabfm_ort_engine.cpp`,
`src/tabfm_devices.cpp`, `src/include/tabfm_ort_engine.hpp`, `test/**`.

### Cycle A — `ResolveDevice` learns the `coreml` lane (pure unit, no hardware)
Start here: it's a pure function with an existing exhaustive test
(`test/cpp/test_tabfm_ort_engine.cpp:331`, "ResolveDevice semantics").

- **RED**: add a `SECTION("coreml flavor")` mirroring the rocm section —
  a `coreml:0` device (`ep="CoreMLExecutionProvider"`, `arch="Apple M3"`,
  `usable=true`); assert `ResolveDevice("coreml", devices, …coreml=true)` →
  `coreml:0`, `auto` → `coreml:0`, `cpu` → `cpu`, and that an un-carried
  `coreml` request throws the `does not carry 'coreml'` / `install the 'coreml'
  flavor` / `SET anofox_tabfm_device='cpu'` message.
- **GREEN**: extend `ResolveDevice` signature with `bool flavor_has_coreml`
  (`tabfm_ort_engine.hpp` default `= TabFMFlavorHasCoreML()`; add that inline
  helper next to `TabFMFlavorHasMIGraphX`, `#ifdef TABFM_EP_COREML`). Add the
  `coreml` branch to the carried-lane and `auto` logic in
  `tabfm_devices.cpp:358`. Existing 2-flag call sites keep working via the new
  default arg.

### Cycle B — EP append branch (`tabfm_ort_engine.cpp`)
- **RED**: a `[.coreml]`-tagged Catch2 case building `SessionOptions` with
  `config.device_id="coreml"` and asserting no throw + (on a hardware runner)
  that the fixture session is created and its forward pass matches
  `golden.json` within tolerance. Under the cpu flavor the branch must throw the
  actionable `ThrowFlavorMissingDeviceLocal("coreml")` (assertable everywhere).
- **GREEN**: add to `AppendExecutionProviders` (`tabfm_ort_engine.cpp:252`),
  after the migraphx branch:
  ```cpp
  if (StringUtil::StartsWith(device, "coreml")) {
  #ifdef TABFM_EP_COREML
      std::unordered_map<std::string,std::string> opts{
          {"ModelFormat","MLProgram"}, {"MLComputeUnits","ALL"},
          {"AllowStaticInputShapes","1"}}; // static shapes via existing buckets
      options.AppendExecutionProvider("CoreML", opts);
      return;
  #else
      ThrowFlavorMissingDeviceLocal("coreml");
  #endif
  }
  ```
  Feed static-shaped inputs via the existing shape buckets
  (`MIGraphXShapeBucket`; rename to `TabFMShapeBucket*` if it now serves two EPs).

### Cycle C — device discovery emits `coreml:0` (`tabfm_devices.cpp`)
- **RED**: a `[.coreml]` C++ case asserting that under `TABFM_EP_COREML`,
  `DiscoverDevices()` contains a `coreml:0` row with
  `ep=="CoreMLExecutionProvider"`, `arch` = the SoC brand (reuse `CpuModelName()`
  /`machdep.cpu.brand_string` at `tabfm_devices.cpp:65`), and `usable` =
  (EP registered AND Gate-1 held). SQL side: extend
  `test/sql/tabfm_devices.test` guarded so it only asserts the coreml row when
  the coreml flavor is loaded (the default cpu run still shows cpu-only).
- **GREEN**: add `ProbeCoreMLDevices(devices)` under `#ifdef TABFM_EP_COREML`,
  called from `DiscoverDevices()` (`:346`). `usable` gated like the rocm probe's
  `MIGraphXClaimsArch` honesty — `true` only on Apple Silicon with the EP
  registered.

### Cycle D — setting validator + auto + engine profile (`tabfm_settings.cpp`)
- **RED**: extend `test/sql/settings.test` — `SET anofox_tabfm_device='coreml'`
  succeeds; the error text for a bad value now lists `coreml`.
- **GREEN**: add `"coreml"` to the validator allow-list
  (`tabfm_settings.cpp:18`) and its description string (`:114`); add a `coreml`
  fp16 **engine profile** to the real manifests (`ParseEngineProfiles`,
  `tabfm_manifest.cpp:152`), reusing the f32→f16 injection cast
  (`CastF32ToF16`). ⚠ scaffold-owned — coordinate.

### Cycle E — flavor / acquisition (`cmake/ort.cmake`) + build wiring
- **RED**: `TABFM_FLAVOR=coreml make debug` fails today (unknown flavor →
  `FATAL_ERROR` at `ort.cmake:109`).
- **GREEN**: add a `coreml` branch: same `osx-universal2` fetch as `cpu`, plus
  `list(APPEND TABFM_ORT_PROVIDERS "TABFM_EP_COREML=1")` and link
  `-framework CoreML -framework Foundation`. Guard it to Darwin/arm64.
  Register the new test TU in `TABFM_CPP_TEST_SOURCES` (CMakeLists.txt:118) —
  ⚠ scaffold-owned.

---

## Sequencing & effort (this machine)

1. **Phase 0 Gate 1a** — partition report on the weight-free graph. ~½ day, no
   blockers, do first: it's the cheapest way to learn whether the rest is worth
   building.
2. **Phase 0 Gate 1b/2** — needs a license-accepted `tabfm_download`. User
   decision.
3. **Phase 1** — Cycles A→E, ~3–5 days *if* the gate is green. A and C–E are
   small; B is the substantive one.

The single highest-value next action is running **Gate 1a now** — it needs
nothing we don't already have on this M3, and its result decides everything
downstream.

## Out of scope
A *direct* CoreML/MPSGraph backend (the Apple analog of the direct-MIGraphX
backend) — sidesteps ORT's partitioner but is a multi-week effort. Only revisit
if Gate 1 fails but Apple acceleration is still wanted.
