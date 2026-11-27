#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ggml.hpp — the ggml engine (vendor/anofox-ggml-adapter, shared with anofox-decide) as tabfm's code sees it.
//
// The engine lives in ::anofox::ggml_adapter (the ONNX reader in ::anofox::onnxread). This header brings its names into
// duckdb::anofox, so tabfm's glue (tabfm_ggml_session.cpp, the GPU plugin, the MLX graph interpreter) uses them unqualified.
//===----------------------------------------------------------------------===//

#include "anofox_ggml_adapter/cpu_support.hpp"
#include "anofox_ggml_adapter/ggml_engine.hpp"
#include "anofox_ggml_adapter/ggml_lower.hpp"
#include "anofox_ggml_adapter/ggml_ops.hpp"
#include "anofox_ggml_adapter/onnx_reader.hpp"
#include "anofox_ggml_adapter/rank_reduce.hpp"
#include "anofox_ggml_adapter/shape_eval.hpp"

namespace duckdb {
namespace anofox {
using namespace ::anofox::ggml_adapter; // NOLINT: the engine's API, by design
} // namespace anofox
} // namespace duckdb
