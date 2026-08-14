# Changelog

All notable changes to `anofox_tabfm` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses SemVer.

## [Unreleased]

### Added
- **Synthetic data generation and imputation** (WS-G): `tabfm_generate(data, n, …)`
  samples new rows from a table's joint distribution, and `tabfm_impute(data, …)`
  fills NULL cells with the conditional best estimate. Both factorize the table
  column by column (the chain rule) over the existing predict engine, so
  correlations between columns are preserved and **every model in the registry
  works**, including classify-only ones — generation needs no regression weights
  at all. Generation returns the input columns plus `synthetic_id`; imputation
  returns exactly the input columns and never touches a non-NULL cell.
  Continuous columns are sampled through quantile bins (the ONNX exporters
  reduce the regression bar distribution to its mean inside the graph, so there
  is no continuous density to sample); imputation skips binning and keeps full
  precision. See `docs/GENERATE.md` and `examples/generate_*.sql`.

  Validated by reproducing the Prior Labs cookbook benchmark end to end
  (`examples/generate_breast_cancer.sql`, breast-cancer Wisconsin, 30 features,
  762 synthetic rows): a classifier given **only synthetic** in-context examples
  scores **97.78%** on held-out real rows against **98.33%** for the real
  training data, with correlation-of-correlations **0.970** across all 435
  feature pairs and zero synthetic rows identical to a real one. Correlations
  are attenuated by the binning (strongest pair 0.998 -> 0.954), which exact
  bar-distribution sampling would fix.

- **TabPFN-3** (`tabpfn-v3`, Prior Labs, released 2026-05-12) onboarded:
  weight-free graphs + tensor maps for both tasks, export parity 2.4e-07
  (classification) / 2.4e-06 (regression). Architecturally distinct from 2.5 —
  a distribution-embedding stack with inducing points, a feature-aggregation
  stack, and RoPE instead of a pre-generated column-embedding table — but it
  needed **no engine change and no new export patches**: the only branch
  blocking `torch.export` is the multiclass target-range guard 2.5 already has.
  Licensed `tabpfn-3-license-v1.0` (non-commercial, gated), so `commercial:false`
  like 2.5. Offline fixture: `test/sql/tabfm_tabpfn3.test`.

### Fixed
- **Checkpoint-based models could not load their converted weights.** Every
  `tools/export_*/convert_weights.py` writes a `model.safetensors` into the cache
  slug, but the manifests declare the downloadable `model.ckpt`, so nothing
  pointed at the converter's output. `model := 'tabpfn-v2'` therefore failed with
  `Failed to find existing initializer ...` even with correctly converted weights
  sitting next to the checkpoint — the raw ckpt carries upstream module names
  (`transformer_encoder.layers.*`) while the committed tensor map is keyed to the
  post-conversion namespace (`blocks.*`). The engine now prefers a sibling
  `model.safetensors` when the declared weights file is a `.ckpt`, and a
  checkpoint that matches zero tensor-map entries now fails with a message
  naming `convert_weights.py` instead of "corrupted — re-download it".

### Changed
- The ensemble layer now draws from `duckdb::RandomEngine` (pcg32) via the new
  `TabFMRandom` helper instead of a hand-ported CPython MT19937 (`PyRandom`,
  removed). Member configs no longer match upstream TabFM draw-for-draw — the
  ensemble is a variance-reduction device, so any valid set of permutations is
  statistically equivalent — and the tests now assert the structural invariants
  (a permutation is a permutation, `cat_mask` is composed through it, class
  shifts are in range) rather than literal draws. Draws remain deterministic per
  seed within a build.

- Extension scaffold on the DuckDB extension template (DuckDB v1.5.4),
  flavor-aware ONNX Runtime build (`TABFM_FLAVOR=cpu|cuda|rocm`, `cmake/ort.cmake`).
- SQL surface (SQL-API rev 4): `tabfm_predict`, `tabfm_predict_by`,
  `tabfm_predict_agg`, `tabfm_predict_win` (custom window callback),
  `tabfm_download`/`tabfm_models`/`tabfm_load`/`tabfm_unload`/`tabfm_remove`,
  `tabfm_devices`. Full names `anofox_tabfm_*` with short `tabfm_*` aliases.
- Settings `anofox_tabfm_*` (license gate, cache dir, threads, guardrails,
  device/ep_path, trace level) and house-style PostHog telemetry (opt-out).
- Safetensors reader (F32/BF16/I64), model manifest (built-in tabfm-v1),
  ORT engine (external-initializer injection), device discovery
  (CUDA via NVML, ROCm/MIGraphX via KFD topology).
- Weights lifecycle over DuckDB's VFS: license gate, 8 MiB chunked atomic
  download, cache scan, air-gapped pre-seeded caches.
- CI fixture model (weight-free, random-init) + real weight-free ONNX graphs
  in `resources/`; model tooling under `tools/` (uv).
- ONNX Runtime built with the MIGraphX EP for the ROCm flavor
  (see `docs/rocm-build.md`); verified on gfx1201 (RX 9070 XT).

- Real predict engine (`tabfm_engine.cpp`): the classify/regress surface runs
  the full TabFM pipeline — preprocess (WS-F) → manifest/graph + safetensors
  injection (WS-B) → ORT session cached in `TabFMState` (WS-C) → forward pass →
  softmax/inverse-transform decode — validated end-to-end against the fixture
  model. Errors with the SQL-API §5 remediation text when weights are missing.

### Notes
- The user-facing surface is `tabfm_classify` / `tabfm_regress` (upstream
  `TabFMClassifier` / `TabFMRegressor` shape). The grouped / composable-
  aggregate / windowed surfaces are held behind an internal aggregate and will
  be re-exposed when needed.
- Real 6.6 GB weights, the numeric regression fixture, and NFR-Q1 parity are
  the next milestone; today's e2e coverage uses the classification CI fixture.
- Telemetry is a deliberate deviation from spec NFR-S1 — see the README.

## [v2026.08.15] - 2026-08-15

Two reports from [@maxdemarzi](https://github.com/maxdemarzi) and the follow-up
work they prompted.

### Fixed
- **The flavor-install route pointed at a host that does not exist**
  ([#25](https://github.com/DataZooDE/anofox-tabfm/issues/25)). `ext.anofox.com`
  has no DNS record, so the route the extension printed — and the README
  documented — could not work for anyone. It was wrong in both host and path:
  the repository is `https://get.anofox.com`, serving
  `/<duckdb_version>/<platform>/anofox_tabfm.duckdb_extension.gz`. Correcting
  the host alone would still have dead-ended a GPU user, because that
  repository serves the cpu flavor only, so the message now names the route
  that works — building the flavor from source — and cites the repository for
  released cpu builds. See `docs/GPU_DISTRIBUTION.md` for why a published GPU
  artifact is not a pipeline change.
- **The class ceiling was hardcoded and blamed the user's graph**
  ([#26](https://github.com/DataZooDE/anofox-tabfm/issues/26)). The 10-class
  limit was asserted in the engine and reported as "TabFM v1 supports at most
  10" whatever model was selected, while `size_regime.max_classes` sat in the
  registry read only for display and for generate's bin sizing. A model with a
  narrower head than 10 got no early check at all: it ran preprocessing, weight
  resolution and a full forward pass before the graph-mismatch backstop fired,
  sending people to audit a registration that was not wrong. The ceiling now
  comes from the selected model's registry entry and the message names the
  model, the limit, the column and its class count. A spec that declares
  nothing keeps the historical 10.

### Changed
- **Every ScatterND slice bound is pinned, and CI enforces it.** The pins that
  keep `tabicl-v2` running on the CUDA EP (v2026.08.14) now cover `tabpfn-v3`
  and `orion-bix` too. Those two do *not* currently fail on CUDA — measured,
  unpinned, on an RTX 3070 — so this is insurance: the trigger is a recycled
  CPU-side buffer, an allocation pattern rather than a property of the
  construct, and nothing characterises when it bites. Each newly pinned graph
  was verified against its **real cached checkpoint**: `logits` bit-identical on
  CPU at three shapes, and agreeing with the unpinned graph on CUDA. With every
  affected graph pinned, `.github/workflows/graph_invariants.yml` checks them
  with a plain glob, so a new graph carrying the construct fails until someone
  pins it or excludes it deliberately. The three exporters that can emit the
  construct also apply the pass themselves, so a re-export cannot regress it.

### Added
- `tools/gpu_test/` — a harness for exercising GPU paths on a machine with no
  GPU: rent the cheapest one by the minute, ship files to it, run, destroy. Not
  wired into CI (it costs money per run). Includes the standalone reproducer
  filed as [onnxruntime#32083](https://github.com/microsoft/onnxruntime/issues/32083)
  for the CUDA `Slice` bug the pins work around.
- `docs/GPU_DISTRIBUTION.md` — why `SET custom_extension_repository` cannot hand
  anyone a working CUDA build today, with the measurements, and the one
  unanswered question that gates the only route which would.

## [v2026.08.14] - 2026-08-14

Everything in this release is CUDA, and all of it comes from
[@maxdemarzi](https://github.com/maxdemarzi).

### Fixed
- **`tabicl-v2` could not run on CUDA at all.** Both TabICL graphs failed inside
  inference at a `ScatterND` node — `updates {S,…}` against `indices {T,1}` —
  while running correctly on the CPU EP. The graphs are not at fault: `indices`
  is `Slice(Range(0,T,1), 0, S)` and is length `S` by construction. On the CUDA
  EP that `Slice` returns its input **untrimmed**, `[0..T-1]`, because the
  CPU-side buffer holding `S` is recycled before the kernel reads it — an ONNX
  Runtime bug, still present in 1.28.0 (#21). Naming the bound tensors as graph
  outputs excludes them from buffer reuse, and both graphs now run on CUDA,
  agreeing with the CPU EP to 3.5e-4 (classification) and 1.5e-6 (regression)
  relative. CPU output is bit-identical to before, so nothing on the CPU path
  changes. `tools/pin_dynamic_slice_bounds.py` applies the pins structurally,
  and `--check` re-verifies them so a re-export cannot silently drop the
  workaround ([#23](https://github.com/DataZooDE/anofox-tabfm/pull/23)).
  The recycled buffer was isolated by exposing graph intermediates one at a
  time — only the two tensors carrying `S` change the outcome.
- **A failed execution-provider load reported the wrong error entirely**
  ([#22](https://github.com/DataZooDE/anofox-tabfm/pull/22)). ORT logs EP load
  failures through its *default* logger, which only exists once an `Ort::Env`
  does. Appending a GPU EP before creating the env therefore replaced ORT's real
  diagnosis — which library, which version it wanted — with the unrelated
  `Attempt to use DefaultLogger but none has been registered`. This is what made
  a newer ONNX Runtime look incompatible in #21 when the actual fault was a
  provider library that would not load. The env is now created before any EP is
  appended and before device discovery.

### Added
- **CUDA device discovery on Windows** ([#24](https://github.com/DataZooDE/anofox-tabfm/pull/24)).
  The NVML probe behind `tabfm_devices()` was Linux-only, so on Windows no CUDA
  device was ever discovered and `SET anofox_tabfm_device='cuda'` was refused —
  the CUDA execution provider itself was never the problem. NVML has the same C
  ABI on both platforms, so only the loader differs (`nvml.dll` via
  `LoadLibrary` against `libnvidia-ml.so.1` via `dlopen`); the probe is
  single-sourced. See `docs/WINDOWS_INFERENCE.md`.

## [v2026.08.13] - 2026-08-13

### Changed
- **`anofox_tabfm_threads` now defaults to the container's CPU allocation, not
  the host's core count** ([#19](https://github.com/DataZooDE/anofox-tabfm/pull/19),
  contributed by [@maxdemarzi](https://github.com/maxdemarzi)).
  `std::thread::hardware_concurrency()` reports what the kernel can see, which
  inside a container is the whole machine: on a 64-CPU allocation in a 256-core
  host the default resolved to 128 intra-op threads *per ONNX session*, and
  DuckDB builds one session per concurrent task (measured: 132 threads and a
  load average of 143 against 64 usable cores, for a query with `SET threads =
  4`). The default is now sized by the smaller of the two limits a container can
  impose — the **cpuset** (`sched_getaffinity`) and the **CFS bandwidth quota**
  (cgroup v2 `cpu.max` or v1 `cpu.cfs_quota_us`), taken from the process's own
  cgroup and up the hierarchy. Kubernetes only pins a cpuset under the static
  CPU Manager policy, while the default enforcement for `limits.cpu` is quota,
  so honouring one and not the other leaves the other deployment
  oversubscribed. Unchanged off Linux, on an unconstrained host, and for anyone
  already setting `anofox_tabfm_threads`.

### Fixed
- **Missing weights were reported as a raw ONNX Runtime error on Windows.** ORT
  phrases an unresolvable external-data file per platform — POSIX says `cannot
  get file size`, Windows says `file_size: The system cannot find the file
  specified` (from `FormatMessage`, not `strerror`) — and only the POSIX shapes
  were matched. A Windows user who had not run `tabfm_load` got that raw text
  instead of the remediation naming it. The predicate is now
  `IsMissingWeightsMessage()` with every phrasing pinned by unit tests, since an
  end-to-end test can only assert the shape of the platform it runs on.
- **CI never executed the C++ test suite.** The test targets filtered on
  `"test/*"`, which matches how the sqllogictests register (by path) but not how
  the Catch2 TUs do (by tag), so 106 cases and ~72,000 assertions were compiled
  into every build and then skipped — locally and in CI. Each flavor now runs
  the binary twice. Turning it on immediately surfaced the Windows error-mapping
  bug above. Spotted by [@maxdemarzi](https://github.com/maxdemarzi) while
  contributing #19.
- **A nested feature column crashed with an INTERNAL error instead of a binder
  error** ([#17](https://github.com/DataZooDE/anofox-tabfm/issues/17)). A
  `LIST`/`ARRAY`/`STRUCT`/`MAP`/`UNION` column fell through the preprocessing
  classifier's "unknown → categorical" fallback (right for scalar unknowns like
  `BLOB`, wrong for nested ones): it produced no usable encoding, so a relation
  whose only feature was nested reached the engine with an empty feature matrix
  and failed as `INTERNAL Error: … Run() called with null input buffers`, while a
  nested column *next to* usable scalars was silently encoded as garbage and
  returned predictions with no warning at all. `tabfm_classify`/`tabfm_regress`
  now reject nested feature (and target) columns at bind time, naming the column,
  its type, and both remedies — projecting into scalar columns or excluding it
  with `features := [...]`. `tabfm_generate`/`tabfm_impute` got the same guard:
  there every column is a chain-rule target, so a nested one was not merely an
  unusable feature but was *sampled*, silently emitting meaningless values.
  Independently, a batch that loses every feature column (a relation with none
  besides the target, or all of them constant across the training rows) now
  raises an actionable `Invalid Input Error` rather than the same assertion, and
  that check runs before the model's weights are resolved — a relation that can
  never be scored no longer reports "model not downloaded" first.
