#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ckpt.hpp — native PyTorch checkpoint (.ckpt/.pt) reader.
//
// PyTorch's default save format is a ZIP archive (STORED, no compression)
// containing `<root>/data.pkl` (a pickle-protocol-2 object graph of the
// state_dict) and `<root>/data/<key>` raw tensor storage blobs. This reads it
// natively — no Python — so tabfm_download can fetch a HuggingFace .ckpt and the
// engine injects it directly, the same as a safetensors file.
//
// Only the subset of the pickle VM that torch.save emits for a plain state_dict
// (OrderedDict of str → Tensor via _rebuild_tensor_v2) is implemented.
//===----------------------------------------------------------------------===//

#include "duckdb/common/common.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

#include <cstdint>

namespace duckdb {
namespace anofox {

//! One tensor recovered from a checkpoint. `data`/`nbytes` point into the caller's
//! file buffer (which must outlive this) — the raw little-endian storage bytes.
struct CkptTensor {
	string dtype; // "f32" | "f16" | "bf16" | "f64" | "i64" | "i32" | "i16" | "i8" | "u8" | "bool"
	vector<int64_t> shape;
	const uint8_t *data = nullptr;
	idx_t nbytes = 0;
};

//! True if `buf` looks like a PyTorch zip checkpoint (PK\x03\x04 local header).
bool IsTorchCkpt(const uint8_t *buf, idx_t size);

//! Parse a PyTorch .ckpt/.pt buffer into its state_dict (param name → tensor).
//! Throws InvalidInputException on a malformed archive or an unsupported pickle
//! opcode. Tensor `data` pointers alias `buf`.
unordered_map<string, CkptTensor> ReadTorchCkpt(const uint8_t *buf, idx_t size);

} // namespace anofox
} // namespace duckdb
