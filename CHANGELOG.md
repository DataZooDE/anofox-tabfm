# Changelog

All notable changes to `anofox_tabfm` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project uses SemVer.

## [Unreleased]

### Added
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

### Notes
- The predict engine is currently a deterministic placeholder (majority
  class / mean) behind the `PredictEngine` seam; the real TabFM forward pass
  is wired in a later milestone.
- Telemetry is a deliberate deviation from spec NFR-S1 — see the README.
