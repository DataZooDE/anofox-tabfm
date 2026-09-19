# Codebase Concerns

**Analysis Date:** 2026-09-19

## Tech Debt

**Static ORT::Env lifetime & dlclose safety:**
- Issue: `GetOrtEnv()` in `src/tabfm_ort_engine.cpp:132` allocates a static `Ort::Env` pointer with `new` but never deletes it. The comment acknowledges this is intentional — "tearing it down during dlclose / static destruction is unsafe" — but it is a permanent allocation that cannot be cleaned up.
- Files: `src/tabfm_ort_engine.cpp:132–134`
- Impact: Single-instance memory leak per process; negligible (ORT env ~KB), but violates RAII principle and may trigger allocation-count assertions in strict test harnesses.
- Fix approach: Document the deliberate leak clearly in the comment and consider marking the allocation with a comment like `/* intentional: ORT env must outlive extensions */` for clarity.

**TabFMSession member destruction order vulnerability:**
- Issue: `TabFMSession` (in `src/tabfm_ort_engine.cpp:282`) declares `Ort::Session session` as the first member, but ONNX Runtime stores references to `injected_names` and `injected_values` vectors. When the destructor runs, `session` is destroyed **last** (C++ member destruction is reverse declaration order), leaving dangling references during teardown.
- Files: `src/tabfm_ort_engine.cpp:280–285`, `src/tabfm_ort_engine.hpp:169`
- Impact: Potential use-after-free on ORT's internal cleanup (low risk if ORT doesn't deref during teardown, but precarious). Code review flagged this as P0.
- Fix approach: Declare `injected_names` and `injected_values` **before** `Ort::Session`, or explicitly call `session.release()` at the top of `~TabFMSession()`.

**Opaque session handle via `shared_ptr<void>` with unsafe cast:**
- Issue: `LoadedModel::session` is stored as `shared_ptr<void>` to decouple the state layer from ORT headers, then recovered via `reinterpret_cast<TabFMBackend *>()` in `src/tabfm_engine.cpp:798`. If the actual stored type ever changes, the cast becomes silently incorrect.
- Files: `src/include/tabfm_state.hpp:43`, `src/tabfm_engine.cpp:798`, `src/tabfm_ort_engine.cpp`
- Impact: Type-safety regression; any refactoring of session storage must manually verify all cast sites.
- Fix approach: Replace with a typed pimpl (`shared_ptr<LoadedTabFMSession>`) — a lightweight wrapper holding both the ORT session and the backend interface.

**Large monolithic engine file:**
- Issue: `src/tabfm_engine.cpp` (803 lines) combines path resolution, manifest JSON parsing, file I/O, session loading, batch assembly, and prediction decoding. Responsibilities are tangled, making it hard to reason about data flow and test parts independently.
- Files: `src/tabfm_engine.cpp`
- Impact: High cognitive load; difficult to add tests for sub-stages (e.g., just the manifest resolver); refactoring any stage risks breaking others.
- Fix approach: Split into `ModelResolver`, `SessionLoader`, `RowBatchBuilder`, and `PredictionDecoder` classes, each with its own file or module. Deferred as P3 (design track).

**Inconsistent `FileSystem` usage:**
- Issue: Weights lifecycle uses `GetFileSystem(context)` (which respects httpfs auth and proxies), while the predict engine uses `CreateLocal()` for fallback I/O. This inconsistency makes it unclear which path gets VFS features.
- Files: `src/tabfm_weights.cpp:104`, `src/tabfm_engine.cpp:790`
- Impact: Risk of httpfs-aware code paths being bypassed in edge cases (e.g., custom manifests with remote graphs).
- Fix approach: Choose one deliberately (likely `GetFileSystem(context)` everywhere) and document the rationale.

**Multiple yyjson cleanup patterns:**
- Issue: JSON cleanup is handled four different ways: RAII wrapper (`YyjsonDoc`), manual `yyjson_doc_free()`, relying on scopes, and inline destructors.
- Files: `src/tabfm_manifest.cpp`, `src/tabfm_safetensors.cpp`, `src/tabfm_engine.cpp`, `src/tabfm_weights.cpp`
- Impact: Easy to miss a `yyjson_doc_free()` call and leak memory; no single pattern to follow.
- Fix approach: Create a centralized `TabFMJsonDoc` RAII wrapper and use it everywhere.

## Known Bugs

**Model output validation missing (P0 — code review item A):**
- Symptoms: Predictions may use out-of-bounds logits if the ONNX model returns unexpected shapes or empty outputs. A transposed output, missing classes, or zero-size tensor causes incorrect predictions or crashes.
- Files: `src/tabfm_ort_engine.cpp:425`, `src/tabfm_engine.cpp:429–469`
- Trigger: Any custom model or ONNX graph with incorrect output tensor rank, shape mismatch, or missing output names.
- Workaround: Currently relies on the bundled fixture and built-in models being correct. Use with custom models at your own risk.
- Fix: Add `ValidateOutput(out, batch.T, task, n_classes)` before decode to check `shape.size()==3 && shape[0]==1 && shape[1]==T`, `C>=n_classes` (classification) or `C>=1` (regression), and reject with actionable `InvalidInputException` on mismatch.

**NaN features corrupt context statistics (P1 — code review item C):**
- Symptoms: Non-NULL `NaN` in a `FLOAT`/`DOUBLE` feature column propagates through `mean()` and `std()` calculation, making the whole feature NaN. Predictions then become NaN or unstable with no clear error message.
- Files: `src/tabfm_preprocess.cpp:286,301`
- Trigger: Any table with NaN values in numeric feature columns (e.g., from a division by zero or external data source).
- Workaround: Clean NaN to NULL before calling predict.
- Fix: Treat `std::isnan()` and non-finite values the same as NULL in mean-fitting and imputation; decide on policy for NaN in regression targets (reject or treat as query row).

**Datetime feature precision loss and overflow (P1 — code review item D):**
- Symptoms: `TIMESTAMP_NS` values lose sub-microsecond precision; extreme dates (e.g., year 292,000,000) cause signed integer overflow UB in `day*kNsPerDay` and `micros*1000` math.
- Files: `src/tabfm_preprocess.cpp:85–96`
- Trigger: Datetime columns with nanosecond precision or extreme dates.
- Workaround: Use `TIMESTAMP_US` or newer and avoid extreme dates.
- Fix: Branch by logical type and extract native units; use checked arithmetic or `double`/`long double` with range checks; throw on overflow.

**Safetensors shape-product overflow accepts corrupt files (P2 — code review item G):**
- Symptoms: A crafted safetensors header with large shape dimensions may wrap `idx_t` overflow during `ElementCount()`, passing validation but then handing ORT a huge shape over a small buffer. Potential heap corruption.
- Files: `src/tabfm_safetensors.hpp:54`
- Trigger: Manually crafted malicious safetensors file (unlikely in normal use).
- Workaround: Only use trusted safetensors files from official sources.
- Fix: Implement one checked-multiply helper and use it in all shape-product calculations (parser, arena sizing, ORT initializer prep); reject products that don't fit `idx_t`.

**Window aggregate state leak on exception (P2 — code review item H):**
- Symptoms: If an exception is thrown during window-aggregate processing before finalize, `state.reader` allocated via `new WindowRowReader()` is never freed. The leak is latent because the window surface isn't wired yet.
- Files: `src/tabfm_predict_agg.cpp` (window path, around line 380–420)
- Trigger: Exception mid-window-aggregate; only affects `tabfm_predict_win` when it is registered.
- Workaround: Currently not exposed; no user impact yet.
- Fix: Wrap state.reader in `unique_ptr` or ensure it is covered by `StateVectorCleanup` guard when the window path is wired.

**Table-macro `target` parameter interpolation (P1 — code review item F):**
- Symptoms: The macro in `src/tabfm_macros.cpp:91` builds `'... NULL AS ' || target || ...` without quoting. A target column name with spaces, special characters, or SQL injection payload breaks the query.
- Files: `src/tabfm_macros.cpp:91`
- Trigger: `TABFM_PREDICT_MACRO` with a target name like `"churned '; DROP TABLE ...;"` or `"has spaces"`.
- Workaround: Avoid special characters in column names when using the macro.
- Fix: Quote the identifier with `quote_ident()` or validate it; document that `data` and `test` are intentionally unquoted relation strings.

## Security Considerations

**PostHog telemetry enabled by default:**
- Risk: Telemetry is on by default (unless `DATAZOO_DISABLE_TELEMETRY=1` env var is set or the user explicitly runs `SET anofox_telemetry_enabled = false`). Usage data (function names, feature flags) flows to PostHog servers.
- Files: `src/anofox_tabfm_extension.cpp:85–107`, `src/include/telemetry.hpp`
- Current mitigation: Env var gate + explicit SQL setting; documented in README as intentional deviation from spec (NFR-S1 says no telemetry).
- Recommendations: (1) Consider opt-in instead of opt-out for community-extension submission; (2) Document the API key and endpoints clearly for transparency; (3) Ensure no query/data leaks into telemetry (currently only function names and aggregates are sent).

**Weight integrity validation relies on HTTP headers:**
- Risk: Download validation checks remote file size and SHA-256 from the manifest + preflight `HEAD` request. A MITM or CDN misconfiguration could serve incorrect weights without detection if the safetensors header itself isn't verified.
- Files: `src/tabfm_weights.cpp:250–350` (download + size validation)
- Current mitigation: (1) httpfs supplies TLS/proxy auth; (2) byte count is checked against manifest + remote size; (3) safetensors header SHA-256 is validated when the session is loaded (external-data mode).
- Recommendations: Always validate the downloaded safetensors header hash before use, even in inject mode. Consider pinning CDN certificates for the Hugging Face mirror.

**License gate is advisory only:**
- Risk: `anofox_tabfm_accept_hf_license = true` is a user-set flag; there is no enforcement that the user actually accepted the license terms. The gate just prevents downloads, not usage.
- Files: `src/tabfm_weights.cpp:95–101`, `src/tabfm_engine.cpp:240`
- Current mitigation: Error message and docs clearly state the license (non-commercial, no redistribution); the gate is a prompt, not a legal contract.
- Recommendations: Document that this is a courtesy gate; users are responsible for complying with the actual TabFM license. Consider linking to the full license text in error messages.

**Bundled PostHog API key in source:**
- Risk: `src/anofox_tabfm_extension.cpp:64` contains a hardcoded PostHog key (`phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J...`). If this is a public key (likely — PostHog keys are front-end), it is not a secret, but exposure is unnecessary.
- Files: `src/anofox_tabfm_extension.cpp:64`
- Current mitigation: PostHog keys are public by design (client-side analytics); the key itself is not a credential.
- Recommendations: No action needed unless this is a private workspace key, in which case move it to an env var.

## Performance Bottlenecks

**GPU compute blocked by protobuf 2 GB serialization limit (ROCm/MIGraphX):**
- Problem: ORT's MIGraphX EP re-inlines initializers when serializing a subgraph, so the ~6.55 GB TabFM model exceeds protobuf's hard 2 GB limit. Inference silently falls back to CPU.
- Files: `src/tabfm_migraphx.cpp`, `src/tabfm_ort_engine.cpp`
- Cause: Upstream ORT/protobuf limitation; the external-data spike confirmed the reinlining is unavoidable via ORT's API.
- Improvement path: Direct MIGraphX backend (`src/tabfm_migraphx.cpp` — **already implemented as of commit ebd4f36**) parses the graph directly and compiles per shape-bucket. Performance: first predict ~20 min (compile), cached runs ~27 s (load .mxr), warm session is fast. The `.mxr` cache is weight-containing and user-local.

**Weight injection memory peak for CPU (18.6 GB on real model):**
- Problem: Inject mode loads the entire safetensors file into an F32 arena before passing it to ORT. On the 6.6 GB real model, peak RSS is 18.6 GB (2.76× the model size).
- Files: `src/tabfm_safetensors.cpp`, `src/tabfm_engine.cpp`
- Cause: Safetensors is read into memory, converted to F32, then ORT copies it again during session init.
- Improvement path: External-data mode (default as of recent commits) reduces to ~7.3 GB by letting ORT read weights from disk. Fallback to inject mode for custom models or on hash mismatch.

**MIGraphX model compilation per unique shape-bucket:**
- Problem: ROCm paths compile a new MIGraphX program for each (T, H) shape-bucket seen at runtime. First compile is ~20 min.
- Files: `src/tabfm_migraphx.cpp:150–250`
- Cause: MIGraphX requires static shapes; dynamic rows/features are handled via shape-bucket padding.
- Improvement path: Pre-compile common buckets (128, 512, 1024, 2048) offline and ship `.mxr` files with the extension. Requires ~6.6 GB per architecture per bucket.

**Single-device serial ORT inference (HLD §6):**
- Problem: Finalize-time forward passes are serialized per device via `DeviceMutex()`. Parallel groups accumulate concurrently, but the expensive `ORT Run` is one-at-a-time per device.
- Files: `src/include/tabfm_state.hpp:94–96`, `src/tabfm_engine.cpp:796`
- Cause: ONNX Runtime sessions are not thread-safe; serialization is correct but limits throughput.
- Improvement path: Consider pooling multiple sessions per device (increases memory but allows parallel inference) once the aggregate surface is production-ready.

## Fragile Areas

**Custom model manifests without validation:**
- Files: `src/tabfm_engine.cpp:150–200`, `src/tabfm_manifest.cpp:200–250`
- Why fragile: The engine accepts any manifest path from `SET anofox_tabfm_model_manifest`. If the manifest points to a broken graph or mismatched weights, errors are deferred until the first predict. No early validation.
- Safe modification: (1) Add a preflight check in `tabfm_load` to validate the graph is loadable; (2) Ensure error messages name the fixing `SET` statement; (3) Test with deliberately broken manifests.
- Test coverage: `test/sql/tabfm_weights.test` has some negative cases, but no test for malformed custom graphs.

**Date/datetime feature handling:**
- Files: `src/tabfm_preprocess.cpp:70–110`
- Why fragile: Multiple timestamp logicaltypes (TIMESTAMP_NS, TIMESTAMP_US, TIMESTAMP_S, TIMESTAMP_MS, DATE, TIME) are cast to `TIMESTAMP_US` with unchecked arithmetic and precision loss. Edge cases (extreme years, leap seconds) are not tested.
- Safe modification: (1) Branch by logical type explicitly; (2) Use checked arithmetic; (3) Add test cases for extreme dates and nanosecond precision.
- Test coverage: Minimal; only basic classification/regression tests exist.

**Safetensors shape validation:**
- Files: `src/tabfm_safetensors.cpp:50–150`
- Why fragile: Shape dimensions are multiplied without overflow checks; a crafted header with max-int values could cause silent corruption.
- Safe modification: (1) Implement `SafeElementCount()` with explicit overflow detection; (2) Reject on overflow early; (3) Test with pathological shapes.
- Test coverage: No adversarial safetensors tests.

**ONNX external-data fallback path:**
- Files: `src/tabfm_engine.cpp:340–380`, `src/tabfm_ort_engine.cpp:190–210`
- Why fragile: The fallback from external-data to inject mode is automatic on any header mismatch. If the validation logic has a bug, users may silently load wrong weights.
- Safe modification: (1) Add a dry-run check that validates the safetensors path can be read and the header matches; (2) Emit a warning (not silent fallback) if fallback is triggered; (3) Add telemetry/logging.
- Test coverage: Covered by `test/sql/tabfm_weights.test` but only for built-in models.

**Ensemble (weighted averaging) code:**
- Files: `src/tabfm_ensemble.cpp`
- Why fragile: Ensemble logic (voting for classification, averaging for regression) is complex and easy to regress. The passive-set algorithm for load balancing has hand-rolled bit math.
- Safe modification: (1) Add unit tests for ensemble outputs (e.g., 3-class voting, weighted averaging with NaN handling); (2) Verify against a reference implementation.
- Test coverage: No unit tests; only full e2e tests via `tabfm_cobatch`.

## Scaling Limits

**Weight cache unbounded directory growth:**
- Current capacity: No limit on `~/.cache/anofox-tabfm`; each model version downloaded is cached forever.
- Limit: Disk space for the cache directory (typical single model = 6.6 GB).
- Scaling path: Implement cache eviction (LRU or explicit `tabfm_remove`). Currently `tabfm_remove` exists but is not wired to `TabFMState`.

**Session memory per model (7–19 GB per CPU/CUDA instance):**
- Current capacity: One model in memory uses 7–19 GB (inject vs external-data mode). Multiple models compound linearly.
- Limit: Host RAM; many-model scenarios (e.g., one model per shard) will OOM.
- Scaling path: Implement explicit model unloading and periodic eviction (documented in HLD but not yet built). `tabfm_unload` is available but not yet wired to the live `TabFMState`.

**MIGraphX `.mxr` cache per architecture:**
- Current capacity: Each shape-bucket (128×16, 512×64, ...) produces a ~6.6 GB `.mxr` file. Five shape buckets = 33 GB cache per GPU architecture.
- Limit: Disk space; multiple GPU targets multiply this.
- Scaling path: Pre-compile and ship only the most common buckets; allow users to disable MIGraphX compilation and use ORT CPU EP fallback.

**Aggregate window state memory (unbounded per window):**
- Current capacity: `tabfm_predict_agg` and `tabfm_predict_win` buffer all rows for a window partition in RAM (`WindowRowReader`).
- Limit: Host RAM; large partitions (1M+ rows) will OOM.
- Scaling path: Stream-based processing; requires major refactoring of the window callback interface (P3 design track).

## Dependencies at Risk

**ONNX Runtime version pinned to 1.23.2:**
- Risk: ORT 1.23.2 is pinned for the MIGraphX/ROCm flavor (HLD D9). If critical bugs or security patches are released, the ROCm flavor can only upgrade by rebuilding ORT from source.
- Impact: Security lag on ROCm; CPU/CUDA use prebuilt archives and can upgrade more flexibly.
- Migration plan: Monitor ORT releases; the ROCm build is user-driven (not prebuilt), so upgrading is on the user. Document the process in `docs/rocm-build.md`.

**DuckDB v1.5.4 submodule locked in:**
- Risk: The DuckDB submodule is pinned to v1.5.4 (as noted in `CLAUDE.md`). Future DuckDB releases may introduce breaking changes to the extension API.
- Impact: Each DuckDB release requires testing and potential code updates.
- Migration plan: Community-extension submission (PR #2181) expects this; stay in sync with extension-ci-tools versioning.

**Google TabFM weights tied to Google's license:**
- Risk: Built-in models (`tabfm-v1`, `tabfm-1.0.0-pytorch`) require user acceptance of Google's non-commercial license. Any change to Google's license terms or model availability breaks compatibility.
- Impact: Users cannot use the extension if they cannot accept the license or if the HF mirror is unavailable.
- Migration plan: Support custom models via `anofox_tabfm_model_manifest` (already implemented). Users can train their own models or use alternative sources.

**Hugging Face mirror dependency:**
- Risk: Default weight downloads come from `huggingface.co`. Any outage or changes to HF's API break `tabfm_download`.
- Impact: New users cannot initialize their caches; existing cached weights are unaffected.
- Migration plan: `TABFM_ORT_URL` supports mirror configuration for ORT; similar mirror support for weights is not yet implemented but can be added.

## Missing Critical Features

**tabfm_load / tabfm_unload / tabfm_models not connected to live state:**
- Problem: `tabfm_models()` hardcodes `row.loaded = false` and `tabfm_load/unload` do not mutate the live `TabFMState` that predictions actually use. The SQL surface reports lifecycle transitions that don't happen.
- Blocks: Reliable model lifecycle tracking; users cannot tell which models are actually loaded without checking logs.
- Fix: Implement a shared `ModelService` to unify the SQL lifecycle functions with the engine's `TabFMState` (P1 — code review item E).

**Window aggregate (`tabfm_predict_win`) not wired:**
- Problem: The aggregate function exists but is not registered. Only `tabfm_predict_agg` is public.
- Blocks: Composable window-based predictions; users must use the full-table aggregate.
- Fix: Wire the window callback once the interface is stable (P3 design track).

**Grouped aggregate (`tabfm_predict`) not finalized:**
- Problem: The README documents `tabfm_classify` and `tabfm_regress` as the user surface; grouped variants are held behind the internal aggregate.
- Blocks: Grouped predictions (e.g., per-customer partition); users must manually partition and call the full-table function.
- Fix: Re-expose grouped variants once the aggregate design is finalized (P3 design track).

**Ensemble (multi-model voting) not production-ready:**
- Problem: `tabfm_ensemble.cpp` is implemented but `tabfm_cobatch` (the entry point) has limited test coverage and no performance tuning.
- Blocks: High-confidence ensemble predictions; users are advised to use single-model path for now.
- Fix: Add unit tests for ensemble logic, verify against reference implementations, and optimize the voting algorithm.

**Real weight data validation against fixture parity:**
- Problem: The real 6.6 GB weights (6.55 GB exactly per docs) exist but end-to-end testing uses the CI fixture (weight-free random-init).
- Blocks: Full confidence in real-model predictions; NFR-Q1 parity verification is noted as "next milestone".
- Fix: Download real weights to CI, run parity tests against a reference implementation, and add to the test suite.

## Test Coverage Gaps

**Negative validation tests:**
- What's not tested: Malformed ONNX graphs, invalid safetensors headers, corrupted manifests, out-of-range dates, NaN features in classification.
- Files: `src/tabfm_engine.cpp`, `src/tabfm_safetensors.cpp`, `src/tabfm_preprocess.cpp`
- Risk: Edge cases and error paths are not exercised; bugs in error handling hide until production.
- Priority: High — add deliberate negative test fixtures (broken graphs, truncated safetensors, extreme dates) to `test/fixtures/` and test the rejection paths.

**Platform-specific code:**
- What's not tested: CUDA device discovery on Windows (code path exists but untested on that platform), MIGraphX compilation on non-RDNA architectures, mmap fallback on non-POSIX systems.
- Files: `src/tabfm_devices.cpp`, `src/tabfm_engine.cpp:236–310`, `src/tabfm_migraphx.cpp`
- Risk: Platform-specific bugs hide until users hit them; current CI may not cover all flavors.
- Priority: Medium — ensure CI covers cpu/rocm on Linux, cpu/cuda on Windows, and cpu on macOS (all tested; others as available).

**Concurrent prediction with model unload:**
- What's not tested: A predict running in one thread while `tabfm_unload` is called in another. The eviction flag and snapshot mechanism are correct in theory but untested.
- Files: `src/include/tabfm_state.hpp:25–57`, `src/tabfm_state.cpp`
- Risk: Race conditions or use-after-free if the snapshotting mechanism is broken.
- Priority: Medium — add a stress test that spawns concurrent predicts and unloads; verify no crashes or incorrect output.

**Date/time feature transformations:**
- What's not tested: `TIMESTAMP_NS` precision, extreme dates (year 292M), mixed-timezone timestamps, leap-second edge cases.
- Files: `src/tabfm_preprocess.cpp:70–110`
- Risk: Silent precision loss or overflow UB on unusual inputs.
- Priority: High — add test cases with pathological dates and verify correct transformations (or explicit rejection).

**Safetensors parser with adversarial headers:**
- What's not tested: Shapes that overflow `idx_t`, negative shape values, header sizes misaligned to data, checksums that don't match.
- Files: `src/tabfm_safetensors.cpp`
- Risk: Heap corruption or silent data corruption on crafted inputs.
- Priority: High — add fuzzer or hand-crafted adversarial safetensors files; verify the parser rejects them cleanly.

---

*Concerns audit: 2026-09-19*
