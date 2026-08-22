# Completing the dynamic-backends phases (all except CoreML)

Plan of record, 2026-08-22. PR #35's phase list, current truth:

| phase | state | what actually remains |
|---|---|---|
| 0 — runtime-aware device resolution | ✅ shipped | — |
| 1 — ROCm as a loadable plugin | ✅ shipped, hardware-verified | — |
| 2 — ORT ≥ 1.28 | ✅ **complete at 1.29.0**, hardware-verified (A3, 2026-08-22) | — |
| 3 — CUDA as a backend plugin + `tabfm_download_runtime('cuda')` | ✅ code shipped, hardware-verified | the **distribution tail**: no plugin-carrying release exists, so `tabfm_download_runtime` cannot actually hand users a plugin yet (Track B) |
| 4 — CoreML | **out of scope by decision** | — |

Tracks C are not phases from the PR but the gaps the phase work itself
measured (2026-08-21 evaluation: examples + benchmark matrix on CUDA/ROCm/CPU);
they are what stands between "the phases work" and "users get their value".

Ordering: **A → B → C.** A before B so the first pinned, user-downloadable
runtime is already 1.29 (pinning 1.28 first would force every early adopter
through a re-download the week after). C parallelizes freely after B.

---

## Track A — finish phase 2: ORT 1.28.0 → 1.29.0

Why 1.29 and not stay at 1.28: S5 (GPU_HARDENING_PLAN) measured that the
1.28.0 provider-bridge global-ctor crash — the bug behind both unfiled
drafts in `docs/UPSTREAM_ORT_ISSUES.md` — **does not reproduce on the 1.29.0
wheel**. Upgrading buys a crash-class fix in the exact code path
`tabfm_download_runtime` users exercise.

- **A1 — CPU flavor bump.** `TABFM_ORT_VERSION` 1.29.0 in `cmake/ort.cmake`
  (archive stems unchanged since 1.28; verify by download), full local suite.
  The vcpkg port (`vcpkg_ports/onnxruntime`, "ort-vcpkg" manifest feature)
  is pinned separately at 1.28.0 — bump in the same commit or explicitly
  defer with a note; do not let the two pins drift silently.
- **A2 — CUDA runtime chain.** Three synchronized pins:
  `gpu_plugins.yml` `ORT_GPU_VERSION` (plugin links the 1.29 core),
  `src/tabfm_weights.cpp` wheel URL + `wheel_bytes` (aiinfra feed, 1.29,
  cp312) + zip member `onnxruntime/capi/libonnxruntime.so.1.29.*`.
  The SONAME patch is untouched: the SONAME stays `libonnxruntime.so.1`
  (equal-length rename still applies); only the *member name* changes.
  The exactly-one-occurrence gate will catch any archive-shape drift.
- **A3 — hardware re-verification.** One pod round on the warm volume
  (~30 min now that weights + CLI are cached): the 11-example run plus the
  fp32/tf32 bench legs, `SERVED_BY=cuda:0` asserted. ROCm needs nothing —
  the MIGraphX plugin never touches ORT. Local CPU suite covers the rest.
- **A4 — docs closure.** `docs/UPSTREAM_ORT_ISSUES.md` drafts re-scoped to
  "affects ≤ 1.28.x, confirmed resolved in the runtime we now pin" (filing
  remains a separate, user-approved decision); README/DYNAMIC_BACKENDS
  version references.

Exit: suite green on 1.29; pod run green with served-by proof; all three
pins name the same version. Effort: ~half a day + ~1 pod-hour.

Risk: the 1.29 wheel layout or feed URL shape differs → the byte-size and
member-count gates fail loudly; fall back to staying on 1.28 for Track B and
decouple (B does not *require* A, it only prefers it).

## Track B — finish phase 3: a release users can actually download

Today `TABFM_PLUGIN_RELEASE_TAG = ""` — `tabfm_download_runtime('cuda')`
fetches the wheel but errors honestly on the plugin, and `'rocm'` errors
entirely. The code path is done and tested; what is missing is one release.

- **B1 — cut a plugin-carrying tag** (needs your call on version/timing —
  release = outward-facing). `gpu_plugins.yml` already attaches both `.so`s
  + sha256s to the release on any `v*` tag; that workflow has been green on
  every push including its dlopen load-checks.
- **B2 — pin the tag.** Set `TABFM_PLUGIN_RELEASE_TAG` in
  `src/tabfm_weights.cpp`; flip the `'rocm'` branch from "not pinned yet" to
  the plugin-only download; update `test/sql/tabfm_download_runtime.test`
  (the not-pinned error assertions become fetch-shape assertions).
- **B3 — end-to-end from nothing.** Fresh pod, stock DuckDB CLI, published
  extension artifact: `CALL tabfm_download_runtime('cuda')` → `SET ep_path`
  → classify → `SERVED_BY=cuda:0` with **zero manual staging** — the first
  time the full user path runs with no harness hands involved. Locally the
  same for `'rocm'` (plugin download + gfx1201 run).
- **B4 — document the floor.** The evaluation established the CUDA path
  needs a ≥ 12.5 CUDA userspace (`cudaLibraryGetKernel`); state it in
  README's GPU section and in the `tabfm_download_runtime` error text if
  detectable (dlerror on that symbol is a recognizable signature worth a
  targeted remediation message).

Exit: a user on a GPU machine reaches `cuda:0`/`rocm:0` with two SQL
statements and no files copied by hand. Effort: ~half a day once the tag
decision lands.

## Track C — close the gaps the evaluation measured

- **C1 — GPU graphs for mitra** ✅ complete, hardware-verified 2026-08-22 (see below). The 2026-08-21 run:
  every tabfm-v1 example passes on `cuda:0`; all six mitra-involved examples
  refuse (correctly, now with the honest error) because mitra ships no
  `ext_graph`/`migraphx_graph`. The P3 machinery (manifest keys,
  register_model args, `SelectGpuGraph`) is already generic — what is missing
  is *content*: `tools/export_onnx` emitting external-data (and MIGraphX)
  graph variants for mitra, manifest entries, and a pod + gfx1201 verify.
  This is what unlocks GPU generation/imputation. Largest C item (~1–2 days,
  export-tooling work; U1 taught that MIGraphX cannot run plain ext graphs,
  so the migraphx variant is a distinct export, not a copy).
  **Done:** four bundled weight-free graphs (ext+migraphx x clf+reg),
  model-aware engine seams, CUDA plugin binds by graph inputs. Verified:
  ROCm fp32 0/90 vs CPU (bf16 9/90, the quantize opt-in); CPU ext-data path
  byte-identical; **all 11 examples pass on cuda:0** including generation
  (generate_breast_cancer: 25 min CPU -> 19 s GPU) and imputation, both
  models served-by-proven. Bonus catch: the .mxr cache collided across
  models (filename-stem key) and silently served the wrong model's compiled
  program -- fixed with a path-hashed key (tabfm_mxr_cache_key.hpp).
- **C2 — ROCm cold-start UX.** Benchmark: warm ROCm is 20–60× CPU, but the
  first predict per (arch, precision, shape bucket) pays ~25 min of MIGraphX
  compile — worst exactly at the fp32 default. Smallest useful step:
  `tabfm_gpu_precompile` + `anofox_tabfm_mxr_source` get a prominent
  README/examples section. Optional second step: publish precompiled `.mxr`
  for gfx1201 as release assets and teach `tabfm_download_runtime('rocm')`
  to stage them (design exists via `mxr_source`; needs a size/licensing
  look — `.mxr` embeds the weights, so **license-gated models cannot ship
  compiled programs**; mitra/Apache models can).
- **C3 — MIGraphX-compatible weight-free fixture.** The ROCm plugin has no
  CI-runnable inference test (U1: the committed fixture graph is not
  MIGraphX-parseable). A fixture export MIGraphX accepts turns the local
  gfx1201 run into a pre-push check and future-proofs for ROCm runners.
- **C4 — fp16 vs bf16 flip comparison** (open plan item): one local
  gfx1201 equivalence run at both modes, margins analysis like S3, table
  row + settings-description sentence. Hours, not days.

## What this plan deliberately does not include

CoreML (phase 4) — excluded by decision. Windows/macOS *GPU plugins* — the
ABI would port, but no phase promised them and nothing is measured there;
raise separately if wanted. Filing the upstream ORT issues — drafts stay
drafts until explicitly approved.

## Decision points for you

1. **Track B tag**: version name and timing of the first plugin-carrying
   release (everything else in B is mechanical once cut).
2. **C2 second step**: whether shipping precompiled `.mxr` assets (gfx1201)
   is worth the release-asset weight for the non-gated models.
3. **C priority order** if not all of C is wanted: C1 unlocks user-visible
   capability, C2 fixes the worst first-run experience, C3 pays down test
   debt, C4 completes the precision documentation.
