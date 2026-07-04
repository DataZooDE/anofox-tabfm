//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_safetensors.hpp — safetensors parser + f32 materialization (HLD §4.3)
//
// Format: [u64 LE header_len][header_len bytes of JSON][tensor data section]
// Header JSON: { "<name>": {"dtype": "F32"|"BF16"|"I64", "shape": [..],
//                           "data_offsets": [begin, end]}, ...,
//               "__metadata__": {"k": "v", ...} }
// data_offsets are relative to the start of the data section.
//
// Only the dtypes TabFM needs are supported: F32, BF16, I64 (HLD §4.3).
// Parsing is buffer-level (caller owns the bytes: mmap or plain read); the
// returned view points into the caller's buffer and never copies tensor data.
// MaterializeF32Arena builds the injection-ready set for the ORT engine:
// BF16 tensors are upcast to F32 into an owned arena (S02: exact 16-bit shift
// into the high half); F32 and I64 tensors pass through pointing at the
// source buffer. Both the source buffer and the arena may be freed right
// after Ort::Session creation (S02: ORT copies injected initializers).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

#include <cstring>

namespace duckdb {
namespace anofox {

enum class SafetensorsDtype : uint8_t { F32 = 0, BF16 = 1, I64 = 2 };

//! Size in bytes of one element of the given dtype.
idx_t SafetensorsDtypeSize(SafetensorsDtype dtype);
//! Canonical safetensors name of the dtype ("F32", "BF16", "I64").
const char *SafetensorsDtypeName(SafetensorsDtype dtype);

//! One tensor as declared by the safetensors header; `data` points into the
//! buffer that was passed to ParseSafetensors (not owned).
struct SafetensorsTensorInfo {
	SafetensorsDtype dtype = SafetensorsDtype::F32;
	vector<int64_t> shape;
	//! Offset of the first byte relative to the start of the data section.
	idx_t data_offset = 0;
	//! Pointer to the first byte inside the caller's buffer.
	const_data_ptr_t data = nullptr;
	//! Declared byte length (== product(shape) * dtype size, validated).
	idx_t nbytes = 0;

	idx_t ElementCount() const {
		idx_t count = 1;
		for (auto dim : shape) {
			count *= static_cast<idx_t>(dim);
		}
		return count;
	}
};

//! Parsed safetensors buffer: tensor directory + optional __metadata__ map.
//! Value-semantic; validity is bounded by the lifetime of the parsed buffer.
struct SafetensorsView {
	unordered_map<string, SafetensorsTensorInfo> tensors;
	//! The single optional "__metadata__" string->string map.
	unordered_map<string, string> metadata;
	//! Data section bounds inside the parsed buffer.
	const_data_ptr_t data_section = nullptr;
	idx_t data_section_size = 0;
	//! Name the buffer was parsed under (file path or description), for errors.
	string source;

	bool Has(const string &name) const {
		return tensors.find(name) != tensors.end();
	}
	//! Lookup by name; throws InvalidInputException naming the tensor.
	const SafetensorsTensorInfo &Get(const string &name) const;
};

//! Parse a complete safetensors buffer. `source` is used in error messages
//! (pass the file path when reading from disk). Throws InvalidInputException
//! on any structural problem: truncated header, header length beyond the
//! buffer, malformed JSON, non-object header, unsupported dtype, negative
//! shape, out-of-bounds or overlapping data_offsets, and byte sizes that do
//! not match product(shape) * dtype size.
SafetensorsView ParseSafetensors(const_data_ptr_t buffer, idx_t buffer_size,
                                 const string &source = "safetensors buffer");

//! BF16 -> F32 upcast: the 16 bf16 bits become the high half of the f32 bits.
inline float Bf16ToF32(uint16_t bits) {
	uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
	float result;
	std::memcpy(&result, &f32_bits, sizeof(result));
	return result;
}

//! One tensor after materialization for ORT injection.
struct MaterializedTensor {
	//! Dtype as stored in the safetensors file.
	SafetensorsDtype source_dtype = SafetensorsDtype::F32;
	//! Dtype of the bytes at `data`: F32 (from F32 or BF16) or I64.
	SafetensorsDtype dtype = SafetensorsDtype::F32;
	vector<int64_t> shape;
	//! Points into the arena for upcast BF16 tensors, into the source buffer
	//! for F32/I64 passthrough.
	const_data_ptr_t data = nullptr;
	idx_t nbytes = 0;
};

//! Movable owning buffer holding the BF16->F32 upcast copies; F32/I64
//! tensors reference the source buffer, so the source must stay alive as
//! long as this arena is used. Free both right after session creation.
class F32Arena {
public:
	F32Arena() = default;
	F32Arena(F32Arena &&) noexcept = default;
	F32Arena &operator=(F32Arena &&) noexcept = default;
	F32Arena(const F32Arena &) = delete;
	F32Arena &operator=(const F32Arena &) = delete;

	unordered_map<string, MaterializedTensor> tensors;

	//! Lookup by name; throws InvalidInputException naming the tensor.
	const MaterializedTensor &Get(const string &name) const;
	//! Bytes owned by the arena (upcast copies only).
	idx_t ArenaBytes() const {
		return arena_bytes;
	}

private:
	friend F32Arena MaterializeF32Arena(const SafetensorsView &view);
	unsafe_unique_array<data_t> arena;
	idx_t arena_bytes = 0;
};

//! Materialize the injection-ready tensor set: BF16 upcast into the arena,
//! F32/I64 passthrough (zero copy).
F32Arena MaterializeF32Arena(const SafetensorsView &view);

} // namespace anofox
} // namespace duckdb
