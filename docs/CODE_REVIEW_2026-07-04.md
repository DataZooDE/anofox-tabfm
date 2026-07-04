# anofox-tabfm — consolidated code review & improvement plan (2026-07-04)

> **Resolution (2026-07-04):** All bug findings **A–H fixed, red-green TDD**, full
> suite green (1319 C++ assertions + 8 SQL files) and verified end-to-end against
> the real 6.6 GB model (correct predictions; `tabfm_models().loaded`
> true→false across predict/unload). New tests: `ValidateTabFMOutput` contract,
> NaN-imputation, safetensors overflow rejection, quoted-target SQL. `tabfm_load`
> stays a lightweight readiness check by design (session warms lazily on first
> predict); the P3 design-track refactors below are intentionally deferred.

Three independent Codex reviews (read-only) plus first-party verification of every
item below:

- **R1 — memory safety / technical correctness** (6 findings)
- **R2 — design quality / SOLID / DuckDB idioms** (13 findings)
- **R3 — ML / numerical correctness** (5 findings; ML core confirmed sound)

Findings are **deduplicated** across reviews and ranked by risk. Each carries the
reviewer(s) that raised it and whether it was confirmed by direct code read.

> **Reassuring baseline (R3):** context/query assembly, `train_size`, `-100`
> sentinel masking, context-only feature standardization, softmax max-subtraction
> + positive-temperature, regression inverse-transform (same mean/scale), label
> decode fidelity, and `row_source_index` scatter-back were all **verified
> correct**. The bugs below are edge cases and missing guards, not core-algorithm
> errors.

---

## P0 — correctness & safety (do first; small diffs, high value)

### A. Model output is never validated before indexing → OOB / silently-wrong predictions
**R1#2, R1#3, R3#3, R3#4, R3#5 — one root cause. CONFIRMED.**
`Run` (`tabfm_ort_engine.cpp:425`) uses `output_names[0]` without checking it's
non-empty; `Decode` (`tabfm_engine.cpp:429-469`) takes `C = out.shape.back()` and
indexes `out.logits[t*C+c]` with no rank/size/stride check. Consequences:
- short `logits` vector → out-of-bounds read (R1#2)
- zero output tensor → OOB on `output_names[0]` / empty logits (R1#3)
- `C < n_classes` → missing class logits default to `0.0`, a phantom class can win argmax and return as a confident label (R3#3)
- regression `C==0` → every `yhat` collapses to `target_mean`, looks plausible (R3#4)
- transposed `[1,C,T]` output → `shape.back()` is `T`, decodes in-bounds but wrong (R3#5)

**Fix (single guard closes all five):**
- In `Run`: reject `output_names.empty()` at construction or run start.
- Add `ValidateOutput(out, batch.T, task, n_classes)` before decode:
  `shape.size()==3 && shape[0]==1 && shape[1]==T`, `C>=n_classes` (clf) / `C>=1`
  (reg), `logits.size()==shape[0]*shape[1]*shape[2]`. Throw an actionable
  `InvalidInputException` on mismatch.
- **Tests:** commit a deliberately-malformed fixture graph (wrong C, transposed,
  no outputs) and assert each throws.

### B. `TabFMSession` destroys injected `Ort::Value`s before the `Ort::Session`
**R1#1, R2#1. CONFIRMED (member order).**
`tabfm_ort_engine.cpp:282` declares `Ort::Session session` first, so it is
destroyed **last** — after `injected_names`/`injected_values`, which ORT's
`AddExternalInitializers` references for the session's lifetime. `SessionHolder`
order is correct; the risk is inside `TabFMSession`. Severity is precautionary
(ORT teardown may not deref the wrappers) but the fix is trivial and removes a
whole risk class given the lazy-read behaviour that already cost us days.

**Fix:** declare `injected_names`/`injected_values` **before** `Ort::Session`
(so session destructs first), or explicitly release the session at the top of
`~TabFMSession`. Also correct the now-stale comments in
`tabfm_ort_engine.hpp:11,169` and `tabfm_state.hpp:34` that claim buffers "may be
freed after `CreateSession`" — they must outlive the session.

---

## P1 — silent-wrong / functional gaps

### C. `NaN` numeric features poison context statistics
**R3#1. CONFIRMED.**
`tabfm_preprocess.cpp:286,301` check only `Value::IsNull()`. A non-NULL `NaN` in a
`FLOAT`/`DOUBLE` column enters the mean → std → the whole feature column becomes
`NaN` → unstable/NaN logits with no clear error.
**Fix:** treat `std::isnan`/non-finite as missing everywhere NULL is handled
(mean fit + imputation). Decide policy for `NaN` regression targets (reject, or
treat as query row) and enforce it.

### D. Datetime features: `TIMESTAMP_NS` precision loss + signed-overflow UB
**R3#2, R1#5. CONFIRMED.**
`DatetimeDayIndex` (`tabfm_preprocess.cpp:85-96`) casts every timestamp flavor to
microsecond `TIMESTAMP` (sub-µs `TIMESTAMP_NS` values collapse together) and does
unchecked `int64` multiplies (`day*kNsPerDay`, `micros*1000`) that overflow on
extreme dates (UB).
**Fix:** branch by logical type and extract native units; use checked or
`double`/`long double` math with range checks; throw on out-of-range.

### E. `tabfm_models()` / `tabfm_load` / `tabfm_unload` are not wired to `TabFMState`
**R2#6. CONFIRMED.**
`tabfm_weights.cpp:519` hardcodes `row.loaded = false`; load/unload report status
without touching the live `TabFMState` map that prediction actually uses. The SQL
surface reports lifecycle transitions that never happened.
**Fix:** give these functions the `DatabaseInstance`/`TabFMState`, and make
`models/load/unload` read and mutate real state (a small shared `ModelService`).

### F. Table-macro interpolates `target` as unquoted SQL
**R2#11. CONFIRMED.**
`tabfm_macros.cpp:91` builds `'... NULL AS ' || target || ...`. A target with
spaces/special characters (or an injection payload) breaks or alters the query.
**Fix:** quote the identifier (`quote_ident`) or validate it; `data`/`test` are
intentionally relation strings (subquery support) and stay as-is, but document it.

---

## P2 — hardening

### G. Safetensors shape-product overflow accepts corrupt tensors as valid
**R1#4. CONFIRMED (unchecked math).**
`ElementCount()` (`tabfm_safetensors.hpp:54`) and the byte/arena/ORT-init math
multiply dims into `idx_t` with no overflow check. A crafted header could wrap to
a small size, pass validation, then hand ORT a huge shape over a small buffer.
Threat is lower (weights are the user's own HF download) but this is cheap
defence-in-depth.
**Fix:** one checked-multiply helper used by parser, arena sizing, and ORT
initializer prep; reject products that don't fit.

### H. Window-aggregate state can leak on exceptions
**R1#6. CONFIRMED; latent.**
`tabfm_predict_agg.cpp` allocates `state.reader = new WindowRowReader()` in the
window path without the `StateVectorCleanup` guard the update/combine/finalize
paths have; a throw before finalize leaks it. The window surface isn't currently
registered/public, so this is latent — fix it when `tabfm_predict_win` is wired.

---

## P3 — design & maintainability (refactor track, R2)

Not bugs, but they raise the cost of every future change:

- **R2#2** `LoadedModel::session` is `shared_ptr<void>` recovered via
  `reinterpret_cast` → replace with a typed pimpl (`shared_ptr<LoadedTabFMSession>`).
- **R2#3** `tabfm_engine.cpp` is a god-TU (path IO, JSON, resolution, session
  loading, row batching, decode) → split into `ModelResolver` / `SessionLoader` /
  `RowBatchBuilder` / `PredictionDecoder`.
- **R2#4** `WeightsManifest` duplicates `ModelManifest` parsing/built-ins → unify
  on `ModelManifest` (download/models/predict can currently disagree).
- **R2#5** error-remediation strings inconsistent (several failures don't name the
  fixing `SET`/`CALL`) → centralize hint helpers.
- **R2#7** `FileSystem` source inconsistent (`CreateLocal()` in predict vs
  `GetFileSystem(context)` in lifecycle) → choose deliberately.
- **R2#8** yyjson RAII handled four different ways → one `tabfm_json` helper.
- **R2#9** headers expose copy-oriented/implementation-heavy APIs → view/span
  descriptors; split trace data from production output.
- **R2#10** aggregate state uses raw owning pointers + scattered manual cleanup →
  one owning payload type.
- **R2#12** dead/future APIs widen the surface (`MIGraphXShapeBucket`, cast hooks,
  ensemble APIs, unused `InferTask`) → internalize until wired.
- **R2#13** const-correctness pass (`const auto &` in immutable loops).

---

## Suggested sequencing

1. **Sprint 1 — bugs (P0+P1): A, B, C, D, E, F.** All have bounded diffs and clear
   tests; this is the correctness payload. Land each red→green.
2. **Sprint 2 — hardening: G, H.**
3. **Sprint 3 — design track (P3):** start with **R2#2** (typed session) and
   **R2#4** (manifest unification), then the `tabfm_engine.cpp` split and error/JSON
   helper centralization.

Highest bang-for-buck single change: **Finding A** — one output-validation guard
turns five silent-failure modes into clear errors and is the difference between a
custom/mismatched graph corrupting results vs. failing loudly.
