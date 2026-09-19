<!-- refreshed: 2026-09-19 -->
# Architecture

**Analysis Date:** 2026-09-19

## System Overview

The anofox-tabfm extension embeds Google's TabFM tabular foundation model (TabPFN-style in-context learner) into DuckDB as a SQL extension. It enables zero-shot classification and regression on tabular data without Python or training loops.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    DuckDB SQL Frontend                                    │
│  tabfm_classify / tabfm_regress (Macros) → tabfm_* Functions             │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────────┐
│                    Function Registration Layer                           │
│  anofox_tabfm_predict_agg, predict_win, devices, download, load, etc.   │
│  `src/anofox_tabfm_extension.cpp`, `tabfm_registration.hpp`            │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬────────────┐
        │                  │                  │            │
        ▼                  ▼                  ▼            ▼
┌─────────────┐  ┌──────────────────┐  ┌──────────┐  ┌─────────┐
│  Predict    │  │  Weight Lifecycle│  │  Device  │  │ State & │
│  Core       │  │  (WS-D)          │  │ Discovery│  │ Settings│
│  (WS-E/F)   │  │                  │  │ (WS-C)   │  │(WS-C)  │
└──────┬──────┘  └──────────┬───────┘  └────┬─────┘  └────┬────┘
       │                    │               │             │
       │                    │               │      ┌──────▼──────┐
       │                    │               │      │ TabFMState  │
       │                    │               │      │ (object     │
       │                    │               │      │  cache)     │
       │                    │               │      └─────────────┘
       │                    │               │
       │        ┌───────────▼────────┐     │
       │        │ HF Model Download  │     │
       │        │ & Cache Mgmt       │     │
       │        │ `tabfm_weights.cpp`│     │
       │        └────────────────────┘     │
       │                                   │
┌──────▼───────────────────────────────────▼──────────────────────────────┐
│                    Model Loading & Manifest (WS-B)                      │
│  manifest (JSON) + safetensors reader + bundled graphs                 │
│  `tabfm_manifest.cpp`, `tabfm_safetensors.cpp`                         │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐            ┌─────────────────────┐
│ SafetensorsView  │            │ ONNX Graph Loading  │
│ (weight buffer)  │            │ & Tensor Injection  │
│ memory arena     │            │                     │
└────────┬─────────┘            └──────────┬──────────┘
         │                                 │
         │        ┌────────────────────────┘
         │        │
         ▼        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Preprocessing Pipeline (WS-F)                         │
│  encode → filter → scale → outlier-clip                                 │
│  `tabfm_preprocess.cpp` (C++ port of Python TabFM minimal preprocessing)│
└──────────────┬──────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   ONNX Runtime Engine (WS-C)                             │
│  ORT::Env (global), ORT::Session (per-model), forward pass               │
│  CPU/CUDA/ROCm/CoreML via flavor (TABFM_FLAVOR)                         │
│  `tabfm_ort_engine.cpp`, `tabfm_migraphx.cpp`                           │
└──────────────┬──────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Output Decoding                                       │
│  Classification: argmax + softmax(temperature) → label/proba             │
│  Regression: inverse-transform scaled output                             │
│  `tabfm_engine.cpp` (decode step)                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| **Extension Entry** | DuckDB extension loader hook, telemetry init | `src/anofox_tabfm_extension.cpp` |
| **Registration** | Function/macro registration entry points | `src/include/tabfm_registration.hpp` |
| **Settings** | `anofox_tabfm_*` configuration options | `src/tabfm_settings.cpp` |
| **State** | Per-database model cache (ObjectCache) | `src/tabfm_state.cpp`, `src/include/tabfm_state.hpp` |
| **Manifest** | Model manifest parsing (JSON → struct) | `src/tabfm_manifest.cpp`, `src/include/tabfm_manifest.hpp` |
| **Safetensors** | Weight file reader, buffer arena | `src/tabfm_safetensors.cpp`, `src/include/tabfm_safetensors.hpp` |
| **Weights Lifecycle** | Download, cache, models(), load, unload, remove | `src/tabfm_weights.cpp` |
| **Device Discovery** | GPU detection, device selection, tabfm_devices() | `src/tabfm_devices.cpp`, `src/include/tabfm_ort_engine.hpp` |
| **ORT Engine** | ONNX Runtime session management, tensor injection | `src/tabfm_ort_engine.cpp`, `src/include/tabfm_ort_engine.hpp` |
| **MIGraphX** | Direct libMIGraphX backend for ROCm (bypasses ORT EP) | `src/tabfm_migraphx.cpp`, `src/include/tabfm_migraphx.hpp` |
| **Preprocessing** | Encode/filter/scale/clip (C++ port of Python logic) | `src/tabfm_preprocess.cpp`, `src/include/tabfm_preprocess.hpp` |
| **Ensemble** | Multi-estimator support, ensemble tensor prep | `src/tabfm_ensemble.cpp`, `src/include/tabfm_ensemble.hpp` |
| **Predict Core** | `__anofox_tabfm_predict_agg` aggregate function | `src/tabfm_predict_agg.cpp`, `src/include/tabfm_predict.hpp` |
| **Predict Macros** | `tabfm_classify` / `tabfm_regress` table macros | `src/tabfm_macros.cpp` |
| **Predict Engine** | Integration layer: preprocess → ORT → decode | `src/tabfm_engine.cpp` |

## Pattern Overview

**Overall:** Layered composition with strict module ownership.

**Key Characteristics:**
- **Module-per-file:** One module = one `src/tabfm_*.cpp` + headers in `src/include/`. Multiple workstreams never touch the same file (CLAUDE.md rule #2).
- **Seam-based testing:** Each module exports a public interface (seams in headers) that downstream modules depend on; tests inject via those seams.
- **Single-threaded aggregate:** DuckDB aggregate functions execute the Update/Combine/Finalize triplet; concurrency is per-device via mutexes during finalize (ORT Run is serialized per device, groups accumulate in parallel).
- **Flavor-aware compilation:** One codebase, four builds (cpu/cuda/rocm/coreml) selected by `TABFM_FLAVOR`. GPU code paths compile out of cpu flavor (community-extension eligible).
- **Opaque state wrapping:** `TabFMState` holds `shared_ptr<void>` sessions (actual `SessionHolder` erased) so state layer never depends on ORT headers.

## Layers

**Layer 1 — SQL/DuckDB Binding:**
- Purpose: Accept SQL syntax, bind arguments, type inference, macro expansion
- Location: `src/tabfm_macros.cpp` (user-facing), `src/tabfm_predict_agg.cpp` (aggregate bind/update/combine/finalize)
- Contains: Macro body SQL, bind callbacks (argument checking, output schema computation), aggregate function triplet
- Depends on: DuckDB API, telemetry
- Used by: DuckDB query executor

**Layer 2 — Option Parsing & Validation:**
- Purpose: Parse `opts` MAP, task inference from target column type, feature selection
- Location: `src/include/tabfm_predict.hpp` (TabFMPredictOptions), `src/tabfm_predict_agg.cpp` (bind logic)
- Contains: Options struct, validation rules, task type enum
- Depends on: DuckDB types
- Used by: Predict aggregate bind

**Layer 3 — Configuration & State:**
- Purpose: Settings registry, per-database model cache, device mutex tracking
- Location: `src/tabfm_settings.cpp`, `src/tabfm_state.cpp`
- Contains: `anofox_tabfm_*` options, `TabFMState` object cache entry, loaded model snapshots
- Depends on: DuckDB config/catalog/object cache
- Used by: Predict finalize (model snapshot), devices discovery, weight lifecycle

**Layer 4 — Manifest & Weight Discovery:**
- Purpose: Parse model metadata (JSON), resolve file URLs, layout cache directories
- Location: `src/tabfm_manifest.cpp`, `src/tabfm_weights.cpp`
- Contains: `ModelManifest` struct (task/repo/revision/files/graph/license), cache layout, download manifest
- Depends on: yyjson (JSON parser), filesystem
- Used by: Weight download, model loading

**Layer 5 — Weight I/O:**
- Purpose: Download files from HF, verify checksums, manage cache; read safetensors headers and allocate weight buffers
- Location: `src/tabfm_weights.cpp` (download/cache), `src/tabfm_safetensors.cpp` (parse/allocate)
- Contains: `SafetensorsView` (memory arena with tensor accessors), download with resume/restart, checksum validation
- Depends on: DuckDB FileSystem (httpfs integration), filesystem ops
- Used by: Model load (Layer 6)

**Layer 6 — Model Loading & Graph Preparation:**
- Purpose: Assemble manifest + weights + graph + tensor injection; create ORT session
- Location: `src/tabfm_engine.cpp` (SessionHolder, integration), `src/tabfm_ort_engine.cpp` (ORT calls)
- Contains: Manifest file resolution, graph loading (bundled or custom), weight buffer injection by tensor name, device resolution
- Depends on: Manifest (Layer 4), Safetensors (Layer 5), ORT engine (Layer 7)
- Used by: Predict finalize (cached in TabFMState)

**Layer 7 — ORT Engine & Device-Specific Execution:**
- Purpose: ONNX Runtime environment, session management, device probing, dtype casting
- Location: `src/tabfm_ort_engine.cpp` (ORT API wrapping), `src/tabfm_devices.cpp` (discovery), `src/tabfm_migraphx.cpp` (direct libMIGraphX)
- Contains: `Ort::Env` singleton, session creation, execution provider selection, device info structs, GPU dtype conversion
- Depends on: ONNX Runtime library (flavor-specific), filesystem (device file probing)
- Used by: Model loading (Layer 6), predict finalize (forward pass)

**Layer 8 — Preprocessing Pipeline:**
- Purpose: C++ port of TabFM's minimal encoding/filtering/scaling logic (bit-for-bit match to Python reference)
- Location: `src/tabfm_preprocess.cpp`
- Contains: `PreprocessedBatch` (x/y/cat_mask matrices, label decoder, target scaling), column encoding (categorical/numeric/datetime), unique-feature filter, outlier removal
- Depends on: DuckDB ColumnDataCollection (input rows), manifest (preprocessing profile id validation)
- Used by: Predict finalize (feature matrix assembly)

**Layer 9 — Ensemble (Phase 2 prep):**
- Purpose: Multi-estimator support, ensemble tensor layout
- Location: `src/tabfm_ensemble.cpp`
- Contains: Ensemble-specific preprocessing (currently n_estimators > 1 is rejected)
- Depends on: Preprocessing (Layer 8)
- Used by: Predict finalize (future)

**Layer 10 — Output Decoding:**
- Purpose: Classification softmax/argmax, regression inverse-transform, scatter predictions back to input row order
- Location: `src/tabfm_engine.cpp` (decode step)
- Contains: Label lookup, softmax temperature application, regression scaling inverse, output struct assembly
- Depends on: Preprocessing (layer decoder info), ORT output tensors
- Used by: Predict finalize (last step before returning results)

## Data Flow

### Primary Request Path (Single-Relation Predict)

1. **User invokes SQL macro** (`src/tabfm_macros.cpp` lines 85+)
   - Input: `SELECT * FROM tabfm_classify('customers', 'churned')`
   - The macro expands to: `query(...)` subquery that merges data + NULL test rows

2. **Aggregate bind** (`src/tabfm_predict_agg.cpp` Bind())
   - DuckDB collects row type, parses target column name
   - Task inferred from target type (numeric → regression, else classification)
   - Options validated, output schema computed (includes yhat, yhat_score, proba if detail mode)
   - Returns `PredictBindData` (captured at bind time: settings, max_rows, options)

3. **Aggregate update** (`src/tabfm_predict_agg.cpp` Update())
   - Each data chunk appended to `ColumnDataCollection` held in state
   - Row limit enforced incrementally (anofox_tabfm_max_rows)

4. **Aggregate combine** (`src/tabfm_predict_agg.cpp` Combine())
   - Merges accumulated data from parallel chunks into one `ColumnDataCollection`

5. **Aggregate finalize** (`src/tabfm_predict_agg.cpp` Finalize()`)
   - Calls `PredictEngine::Predict()` in `src/tabfm_engine.cpp` with the accumulated batch
   - The engine orchestrates the full pipeline (steps 6-10 below)

6. **Preprocessing** (`src/tabfm_preprocess.cpp` PreprocessBatch())
   - Input: `ColumnDataCollection` + column specs (which are targets/features)
   - Encode: categorical ordinal (appearance order, min_freq=2), numeric mean-impute, datetime → 5 fields
   - Filter: unique-feature filter (drop constant/duplicate columns)
   - Scale: standard scaler (std ddof=0) over TRAIN rows, clipped to [-100, 100]
   - Outlier: 2-stage 4-sigma bounds over TRAIN rows
   - Output: `PreprocessedBatch` (x[T,H], y[T], cat_mask[H], train_size, encoders, label_decoder)

7. **Model loading** (`src/tabfm_engine.cpp` SessionHolder construction)
   - Snapshot from `TabFMState` by cache key (model:task@revision)
   - If not loaded: resolve manifest → download weights if needed → inject into ORT session → cache
   - Returns opaque session `shared_ptr<void>` (holds `SessionHolder` internally)

8. **Forward pass** (`src/tabfm_ort_engine.cpp` ORT::Session::Run())
   - Cast x to GPU dtype (bf16/f16) if needed
   - Feed into ORT: x (feature matrix), y (train labels padded with -100 for test rows), cat_mask, d
   - Output: logits [1, T, C] for classification or [1, T, 1] for regression
   - Device mutex serializes ORT Run per device (HLD §6)

9. **Decoding** (`src/tabfm_engine.cpp` Decode())
   - Classification: argmax to class index → label lookup, softmax(temperature) → proba
   - Regression: linear inverse-transform (pred * target_scale + target_mean)
   - Scatter back to input row order (PreprocessedBatch.row_source_index maps)
   - Mark is_training flag (rows with non-NULL train label)

10. **Output assembly** (`src/tabfm_predict_agg.cpp` finalize)
    - Builds result LIST(STRUCT(original_cols, yhat, yhat_score, [proba], is_training))
    - Unnested by macro layer into individual rows

### Train/Test Explicit Form

Same pipeline, except:
- Macro splits data/test into two subqueries (data rows + test rows with NULL label)
- Finalize returns only test rows (no is_training filter needed; training rows never in the frame)

### State Management

- **Model snapshots:** Lazy-loaded on first predict, cached in `TabFMState` object cache (never evicted by LRU — user-controlled lifecycle via SQL)
- **Weight buffers:** Held in `SafetensorsView` arena, kept alive by `SessionHolder` for the session lifetime (ORT reads lazily during inference)
- **Device mutex:** Per-device serialization during finalize (ORT::Session::Run); groups accumulate concurrently before finalize

## Key Abstractions

**TabFMPredictOptions:**
- Purpose: Parsed and validated engine options (task, n_estimators, seed, output_mode, context_rows, softmax_temperature, model)
- Examples: `src/include/tabfm_predict.hpp` (struct def), `src/tabfm_predict_agg.cpp` (parsing)
- Pattern: All values arrive as VARCHAR (SQL-API Level 2 spec), validated to enum/int at bind time

**ModelManifest:**
- Purpose: Structured metadata for one (model, task) pair
- Examples: `src/include/tabfm_manifest.hpp`, built-in manifests in `src/tabfm_manifest.cpp`
- Pattern: JSON → parsed and strictly validated; unknown fields ignored (forward compat)

**PreprocessedBatch:**
- Purpose: Fully preprocessed feature matrix, ready for ORT feeding
- Examples: `src/include/tabfm_preprocess.hpp`, output of `PreprocessBatch()` function
- Pattern: Row-major [T, H] matrices; train rows first, test rows last; encoders kept for introspection

**TabFMState / LoadedModel:**
- Purpose: DB-instance-level model cache, opaque session wrapping
- Examples: `src/include/tabfm_state.hpp`, object cache entry
- Pattern: `LoadedModel.session` is `shared_ptr<void>` (actually `SessionHolder` erased) to avoid ORT header dependency in state layer

**SafetensorsView:**
- Purpose: Mutable memory arena holding all weights, indexable by tensor name
- Examples: `src/include/tabfm_safetensors.hpp`
- Pattern: Allocates one contiguous arena, parses safetensors header to build name → offset map

## Entry Points

**Extension Load:**
- Location: `src/anofox_tabfm_extension.cpp` AnofoxTabfmExtension::Load()
- Triggers: DuckDB loads the extension (dynamic or static)
- Responsibilities: Register telemetry options, settings, function/macro registrations

**SQL Functions:**
- `tabfm_classify(data, target [, test] [, features] [, opts])` → Table macro
- `tabfm_regress(data, target [, test] [, features] [, opts])` → Table macro
- `__anofox_tabfm_predict_agg(row STRUCT, target VARCHAR [, opts MAP])` → Aggregate function (internal)
- `tabfm_download(task)` → Procedure
- `tabfm_models()` → Table function
- `tabfm_load(task)` → Procedure
- `tabfm_unload()` → Procedure
- `tabfm_remove(task)` → Procedure
- `tabfm_devices()` → Table function

All registered in `src/anofox_tabfm_extension.cpp` LoadInternal() via calls to `RegisterXxxFunctions()` (defined in respective modules, declared in `tabfm_registration.hpp`).

## Architectural Constraints

- **Threading:** Single-threaded event loop (DuckDB aggregate triplet). ORT Run serialized per device via `DeviceMutex()` during finalize; groups accumulate concurrently before finalize.
- **Global state:** One process-wide `Ort::Env` (ONNX Runtime), one per-database `TabFMState` in ObjectCache. PreCompiled GPU programs cached on disk (`.mxr` for ROCm).
- **Circular imports:** Avoided via seam pattern: each module's public interface lives in a header, implementation in a `.cpp`; includes flow one direction (leaf → root).
- **Model eviction:** Never by LRU; only by explicit SQL `CALL tabfm_unload()`. If already loaded, new load replaces (old one marked evicted, freed when last snapshot releases).
- **Weight buffer lifetime:** Must outlive ORT::Session (ORT reads lazily during Run). Caller keeps `SafetensorsView` arena alive via `SessionHolder`.
- **Flavor isolation:** GPU code paths guarded by `#ifdef TABFM_EP_CUDA / TABFM_EP_MIGRAPHX / TABFM_EP_COREML`. CPU flavor has zero GPU code (community-extension requirement).

## Anti-Patterns

### Circular Manifest Dependency

**What happens:** A model specifies a custom manifest via `anofox_tabfm_model_manifest`, which itself references another custom manifest (or is malformed).

**Why it's wrong:** Infinite loop or load failure; the user intended model A but got stuck loading B.

**Do this instead:** Manifest resolution is single-pass (file → parse → validate). If a manifest points to another, the integration layer (WS-A spike tools) pre-resolves it. The extension consumes a single manifest per load, validated upfront in `LoadModelManifestFile()` or `ParseModelManifest()`.

### Skipping Preprocessing

**What happens:** A developer wires the ORT forward pass directly without running through the preprocessing pipeline.

**Why it's wrong:** Feature encoding (categorical ordinal, datetime expansion, scaling, outlier clipping) is baked into the v1 model. Raw input crashes or produces garbage.

**Do this instead:** Every predict path must call `PreprocessBatch()` before feeding x/y/cat_mask to ORT. The preprocessing module is the single source of truth for the encode-filter-scale-clip chain.

### Mutating Weight Buffers

**What happens:** Decoding modifies the raw ORT output tensor in-place instead of copying.

**Why it's wrong:** ORT may reuse tensor memory on subsequent Runs; corruption spreads silently.

**Do this instead:** All postprocessing (softmax, inverse-transform, scatter) works on copied outputs in `Decode()`. ORT tensors are read-only.

## Error Handling

**Strategy:** Every user-facing failure throws a DuckDB exception (InvalidInputException, IOException, etc.) with a message that names the fixing SET/CALL.

**Patterns:**
- Model not downloaded: `"tabfm: model 'classification' is not downloaded. Run: CALL tabfm_download('classification');"`
- License not accepted: `"tabfm: non-commercial license must be explicitly accepted. Run: SET anofox_tabfm_accept_hf_license = true;"`
- Device not found: `"tabfm: device 'rocm:1' is not available. Run: SELECT * FROM tabfm_devices();"`
- Max rows exceeded: `"tabfm: max_rows guardrail exceeded (10000 rows seen). Run: SET anofox_tabfm_max_rows = ...;"`

All validation happens at bind time when possible (manifest, manifest files, options) or during update/finalize for row limits.

## Cross-Cutting Concerns

**Logging:** DuckDB's D_INFO/D_WARN macros; trace level controlled by `anofox_tabfm_trace_level` setting.

**Validation:** Strict at bind (manifest validation, option parsing), incremental at update (row count checks), final at finalize (device availability, model loaded).

**Authentication:** DuckDB SECRET system for HF tokens; `tabfm_download()` reads from the active SECRET (TYPE http, SCOPE https://huggingface.co).

**Telemetry:** PostHog via `PostHogTelemetry::Instance()` singleton. Every function calls `CaptureFunctionExecution()` once per execution (at bind or global-state init, not per chunk). Disabled by `DATAZOO_DISABLE_TELEMETRY` env var or CI detection.

---

*Architecture analysis: 2026-09-19*
