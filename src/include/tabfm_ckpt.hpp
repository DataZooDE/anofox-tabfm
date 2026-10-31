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
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include <cstdint>
#include <functional>

namespace duckdb {
namespace anofox {

//! One tensor recovered from a checkpoint, in index order. `data`/`nbytes` usually point into the
//! caller's file buffer (which must outlive this) -- the raw little-endian storage bytes. A tensor
//! saved as a VIEW (a transpose, a slice) keeps its parent's storage and strides, so its storage bytes
//! are not its elements in order; those are gathered into `owned` and `data` points there.
struct CkptTensor {
	string dtype; // "f32" | "f16" | "bf16" | "f64" | "i64" | "i32" | "i16" | "i8" | "u8" | "bool"
	vector<int64_t> shape;
	//! Element strides; empty once the elements are known to be in index order (ReadTorchCkpt
	//! guarantees that for everything it returns).
	vector<int64_t> stride;
	const uint8_t *data = nullptr;
	idx_t nbytes = 0;
	//! The gathered elements of a strided view; null when `data` aliases the file buffer.
	shared_ptr<vector<uint8_t>> owned;

	//! Row-major contiguous: `data` holds the elements in index order.
	bool IsContiguous() const;
};

//! True if `buf` looks like a PyTorch zip checkpoint (PK\x03\x04 local header).
bool IsTorchCkpt(const uint8_t *buf, idx_t size);

//! Parse a PyTorch .ckpt/.pt buffer into its state_dict (param name → tensor), every tensor in index
//! order. Throws InvalidInputException on a malformed archive or an unsupported pickle opcode. Tensor
//! `data` pointers alias `buf`, except a strided view's, which is gathered into its `owned` copy.
unordered_map<string, CkptTensor> ReadTorchCkpt(const uint8_t *buf, idx_t size);

//! The safetensors a checkpoint becomes for the GPU graphs: every tensor, keys sorted, contiguous
//! float32 (other dtypes are cast), no __metadata__ -- byte for byte what tools/export_*/
//! convert_*_weights write with safetensors.torch.save_file, so the header sha the bundled graphs pin
//! (ExpectedWeightsHeaderShaFor) is a property of the checkpoint alone.
//!
//! CkptSafetensorsHeader returns the JSON header including its space padding to 8 bytes: the bytes
//! WeightsHeaderMatches hashes, computable without writing anything. Throws on a non-contiguous tensor.
string CkptSafetensorsHeader(const unordered_map<string, CkptTensor> &state_dict);
//! Stream the whole file (length prefix, header, data) to `write`.
void WriteCkptAsSafetensors(const unordered_map<string, CkptTensor> &state_dict,
                            const std::function<void(const void *, idx_t)> &write);

} // namespace anofox
} // namespace duckdb
