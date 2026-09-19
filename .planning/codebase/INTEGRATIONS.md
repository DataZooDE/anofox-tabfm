# External Integrations

**Analysis Date:** 2026-09-19

## APIs & External Services

**PostHog Analytics:**
- Service: PostHog (eu.posthog.com) - Anonymous usage telemetry
  - SDK/Client: Custom C++ telemetry library (`posthog-telemetry/` submodule)
  - Auth: API key `phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t` (hardcoded in `src/anofox_tabfm_extension.cpp`)
  - Endpoint: `https://eu.posthog.com/batch/` (HTTPS POST with JSON payload)
  - Opt-out: `DATAZOO_DISABLE_TELEMETRY=1` environment variable or `SET anofox_telemetry_enabled = false`
  - Auto-disable: CI environments (GitHub Actions, GitLab CI, Travis, CircleCI, Jenkins, generic CI var)
  - Implementation: `PostHogTelemetry::Instance()` singleton; background worker thread with task queue; best-effort timeout (3s read/write/connection)

**ONNX Runtime Archives:**
- Source: GitHub Releases (microsoft/onnxruntime)
- Versions: v1.23.2 (CPU/CUDA) pinned; ROCm flavor via custom build path
- Downloads: `https://github.com/microsoft/onnxruntime/releases/download/v${TABFM_ORT_VERSION}/${archive_stem}-${TABFM_ORT_VERSION}.${ext}`
- Mirror support: Override via `TABFM_ORT_URL` CMake variable
- Platforms: Linux (x64, aarch64), macOS (universal2), Windows (x64)
- Fetch method: CMake FetchContent; automatic extraction to build tree

**Hugging Face Model Hub:**
- Service: Hugging Face Hub (huggingface.co) - Public weight distribution
  - Default repo: Derived from manifest `repo` field (e.g., `google-research/tabfm`)
  - Revision: Manifest `revision` field (e.g., `main`, git SHAs)
  - Download URL: `https://huggingface.co/{repo}/resolve/{revision}/{file.path}`
  - Override: Per-file explicit `url` in manifest (static GitHub URLs for weight-free graphs)
  - Auth: Optional HF_TOKEN for private models (not currently used; fixture is weight-free)
  - File format: safetensors (binary with JSON header)

## Data Storage

**Databases:**
- DuckDB - Host extension target (SQL queries, function binding)
  - Connection: In-process; part of DuckDB instance
  - Client: DuckDB C++ API via extension scaffold

**File Storage:**
- Local filesystem only (no cloud storage)
  - Weight cache: `~/.anofox/` or custom path via `anofox_tabfm_weights_path` setting
  - Model manifest: JSON file (co-located with graph/weights or bundled)
  - Fixture: Test resources in `test/fixtures/` (committed to repo)
  - ONNX model: `.onnx` files (external-data or bundled; safetensors tensors mapped at runtime)

**Caching:**
- Weight file caching: WS-D module (`src/tabfm_weights.cpp`) manages downloaded safetensors in local cache
- Memory-mapped weights: `mmap(2)` for readonly access to cached safetensors files
- Bundled resources: Weight-free graphs + tensor maps embedded in binary via `cmake/embed_resources.cmake` (no external files)

## Authentication & Identity

**Auth Provider:**
- None (public model distribution)

**Machine Identity (Telemetry):**
- Platform-specific machine ID detection for anonymous usage tracking:
  - **Linux**: MAC address from `/sys/class/net/*/address`
  - **macOS**: IOKit registry (IOPlatformExpertDevice)
  - **Windows**: IP Helper API (GetAdaptersInfo → physical MAC address)
  - Implementation: `posthog-telemetry/src/telemetry.cpp`

## Monitoring & Observability

**Error Tracking:**
- None (no external error service)

**Logs:**
- Console output via DuckDB's logging system
- Telemetry events (function execution, device info, model state) → PostHog API
- Internal: C++ exception-based error handling with user-friendly SQL error messages

## CI/CD & Deployment

**Hosting:**
- GitHub - Source repository + submodule coordination
- Community Extensions - DuckDB official extension registry (PR #2181)

**CI Pipeline:**
- GitHub Actions (inferred from CI environment detection)
  - Telemetry auto-disable when `GITHUB_ACTIONS=true`
  - Extension-ci-tools v1.5-variegata provides CI matrix support

## Environment Configuration

**Required env vars:**
- None (all build/runtime options have defaults or are CMake-configurable)

**Optional env vars:**
- `TABFM_FLAVOR=cpu|cuda|rocm` - Runtime execution provider (build-time selection)
- `TABFM_ORT_ROCM_DIR=/path/to/ort` - ROCm flavor ORT install tree
- `TABFM_ORT_URL=https://mirror.example.com/ort/archive` - Archive mirror override
- `DATAZOO_DISABLE_TELEMETRY=1|true|yes` - Telemetry opt-out

**Secrets location:**
- PostHog API key: Embedded in binary (`phc_t3wwRLtpyEmLHYaZCSszG0MqVr74J6wnCrj9D41zk2t`) — not a secret, hardcoded for public analytics
- Hugging Face token: Not currently used (fixture is public/weight-free)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- PostHog telemetry endpoint: HTTPS POST to `https://eu.posthog.com/batch/` with event JSON payload
  - Event name, distinct ID (machine ID hash), timestamp, properties (model_id, task, device, etc.)
  - Fire-and-forget: No retry or callback expected; best-effort with timeout

## Model & Weight Distribution

**Weight Acquisition:**
- Primary flow: `CALL tabfm_download('model_id')` triggers safetensors fetch from manifest URL
  - Manifest location: Built-in (bundled) or loaded from user-provided JSON
  - Safetensors file: Downloaded to local cache, SHA-256 validated
- Fixture flow (testing): Weight-free ONNX graph + test fixture safetensors committed in `test/fixtures/`
- Air-gapped deployment: Use bundled graphs; omit `tabfm_download()` call

**Version Pinning:**
- Manifest `manifest_version`: 1 (format contract)
- ONNX opset: 18 (graph compatibility)
- TabFM upstream: Google Research Apache-2.0 (no weight bytes; code only in `vendor/tabfm/`)

---

*Integration audit: 2026-09-19*
