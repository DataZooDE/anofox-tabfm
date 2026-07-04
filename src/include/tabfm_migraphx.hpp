//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_migraphx.hpp — factory for the direct AMD MIGraphX inference backend.
//
// ORT's MIGraphX execution provider cannot run this model: it serializes each
// subgraph (weights inlined) into an onnx.ModelProto and blows protobuf's 2 GB
// limit (see docs/GPU_AND_MEMORY_FINDINGS.md). MIGraphX's own parser handles the
// >2 GB external-data model, so the ROCm path bypasses ORT and drives MIGraphX
// directly: parse the migraphx-ready graph (external-data + Shape-rewrite),
// compile per shape-bucket (cached to .mxr on disk), run.
//
// Only the rocm flavor carries an implementation; other flavors get a stub that
// throws an actionable error.
//===----------------------------------------------------------------------===//

#pragma once

#include "tabfm_ort_engine.hpp" // TabFMBackend

namespace duckdb {
namespace anofox {

//! Build a direct-MIGraphX backend.
//!  - graph_path: migraphx-ready ONNX (external-data -> model.safetensors, with
//!    degenerate Shape ops rewritten). Parsed per shape-bucket.
//!  - weights_dir: directory holding model.safetensors (external-data root).
//!  - cache_dir: where compiled .mxr programs are cached (per arch + precision + bucket).
//!  - arch: device arch string ("gfx1201") — part of the .mxr cache key.
//!  - device_ordinal: HIP device index.
//!  - precision: "bf16" | "fp16" | "fp32" — MIGraphX compile precision. bf16/fp16
//!    run ~2x faster than fp32 on RDNA4 and halve VRAM/.mxr; bf16 keeps fp32's
//!    exponent range (safest). Part of the .mxr cache key.
//! Throws InvalidInputException on non-rocm builds.
unique_ptr<TabFMBackend> MakeMIGraphXBackend(const string &graph_path, const string &weights_dir,
                                             const string &cache_dir, const string &arch, int device_ordinal,
                                             const string &precision);

} // namespace anofox
} // namespace duckdb
