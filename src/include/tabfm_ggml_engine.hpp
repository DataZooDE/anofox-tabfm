#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ggml_engine.hpp — the ggml CPU engine (docs/GGML_PLAN.md).
//
// P0 is the skeleton: the refusal contract. The backend itself (GgmlBackend : TabFMBackend) arrives with P5.
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

//! Empty when every op in `op_types` is implemented; otherwise the error text: it names the missing ops and
//! the build option that selects the ONNX Runtime engine instead, because this build has no runtime switch.
std::string GgmlRefusal(const std::vector<std::string> &op_types);

} // namespace anofox
} // namespace duckdb
