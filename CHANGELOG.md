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
