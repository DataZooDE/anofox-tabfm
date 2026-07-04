// tabfm_safetensors.cpp — safetensors parser + f32 materialization (WS-B).
//
// See tabfm_safetensors.hpp for the format contract. JSON parsing uses
// DuckDB's bundled yyjson (namespace duckdb_yyjson) — HLD §4.3.
//
// Spike S02 lessons baked in:
// - the JSON document is bound to a named RAII guard before iteration
//   (friction log #3: iterating a temporary is UB with nlohmann; with yyjson
//   the document must simply outlive all yyjson_val pointers);
// - BF16 -> F32 upcast is a plain 16-bit shift into the high half;
// - both the source buffer and the arena are freeable right after
//   Ort::Session creation (ORT copies injected initializers).

#include "tabfm_safetensors.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"

#include "yyjson.hpp"

#include <string>

namespace duckdb {
namespace anofox {

using namespace duckdb_yyjson; // NOLINT

idx_t SafetensorsDtypeSize(SafetensorsDtype dtype) {
	switch (dtype) {
	case SafetensorsDtype::F32:
		return 4;
	case SafetensorsDtype::BF16:
		return 2;
	case SafetensorsDtype::I64:
		return 8;
	default:
		throw InternalException("unknown SafetensorsDtype");
	}
}

const char *SafetensorsDtypeName(SafetensorsDtype dtype) {
	switch (dtype) {
	case SafetensorsDtype::F32:
		return "F32";
	case SafetensorsDtype::BF16:
		return "BF16";
	case SafetensorsDtype::I64:
		return "I64";
	default:
		throw InternalException("unknown SafetensorsDtype");
	}
}

const SafetensorsTensorInfo &SafetensorsView::Get(const string &name) const {
	auto entry = tensors.find(name);
	if (entry == tensors.end()) {
		throw InvalidInputException("Safetensors '%s' has no tensor named '%s'", source, name);
	}
	return entry->second;
}

namespace {

//! RAII owner for a yyjson document (S02 friction log #3: never iterate a
//! JSON document whose owner has already been destroyed).
struct YyjsonDoc {
	explicit YyjsonDoc(yyjson_doc *doc_p) : doc(doc_p) {
	}
	~YyjsonDoc() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	YyjsonDoc(const YyjsonDoc &) = delete;
	YyjsonDoc &operator=(const YyjsonDoc &) = delete;

	yyjson_doc *doc;
};

SafetensorsDtype ParseDtype(const string &dtype, const string &source, const string &tensor_name) {
	if (dtype == "F32") {
		return SafetensorsDtype::F32;
	}
	if (dtype == "BF16") {
		return SafetensorsDtype::BF16;
	}
	if (dtype == "I64") {
		return SafetensorsDtype::I64;
	}
	throw InvalidInputException("Safetensors '%s': unsupported dtype '%s' for tensor '%s'; "
	                            "supported dtypes: F32, BF16, I64",
	                            source, dtype, tensor_name);
}

//! Read one non-negative integer out of a JSON array element.
idx_t GetNonNegativeInt(yyjson_val *val, const string &source, const string &tensor_name, const char *field) {
	if (!val || !yyjson_is_int(val)) {
		throw InvalidInputException("Safetensors '%s': tensor '%s' field \"%s\" must contain integers", source,
		                            tensor_name, field);
	}
	if (yyjson_is_sint(val) && yyjson_get_sint(val) < 0) {
		throw InvalidInputException("Safetensors '%s': tensor '%s' field \"%s\" contains negative value %lld", source,
		                            tensor_name, field, yyjson_get_sint(val));
	}
	return UnsafeNumericCast<idx_t>(yyjson_get_uint(val));
}

string ShapeToString(const vector<int64_t> &shape) {
	string result = "[";
	for (idx_t i = 0; i < shape.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += std::to_string(shape[i]);
	}
	return result + "]";
}

} // anonymous namespace

SafetensorsView ParseSafetensors(const_data_ptr_t buffer, idx_t buffer_size, const string &source) {
	if (buffer_size < sizeof(uint64_t)) {
		throw InvalidInputException("Safetensors '%s' is truncated: %llu bytes cannot hold the 8-byte "
		                            "little-endian header length",
		                            source, static_cast<unsigned long long>(buffer_size));
	}
	uint64_t header_len;
	memcpy(&header_len, buffer, sizeof(header_len));
	if (header_len > buffer_size - sizeof(uint64_t)) {
		throw InvalidInputException("Safetensors '%s': header length %llu extends beyond the buffer "
		                            "(%llu bytes after the length prefix); the file is truncated or corrupt",
		                            source, static_cast<unsigned long long>(header_len),
		                            static_cast<unsigned long long>(buffer_size - sizeof(uint64_t)));
	}

	SafetensorsView view;
	view.source = source;
	view.data_section = buffer + sizeof(uint64_t) + header_len;
	view.data_section_size = buffer_size - sizeof(uint64_t) - header_len;

	yyjson_read_err error;
	YyjsonDoc guard(yyjson_read_opts(reinterpret_cast<char *>(const_cast<data_ptr_t>(buffer)) + sizeof(uint64_t),
	                                 header_len, YYJSON_READ_NOFLAG, nullptr, &error));
	if (!guard.doc) {
		throw InvalidInputException("Safetensors '%s': malformed JSON header: %s (at byte %llu of the header)",
		                            source, error.msg, static_cast<unsigned long long>(error.pos));
	}
	auto root = yyjson_doc_get_root(guard.doc);
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("Safetensors '%s': the JSON header must be an object mapping tensor names to "
		                            "{dtype, shape, data_offsets}",
		                            source);
	}

	// name -> [begin, end) ranges for the overlap check
	vector<std::pair<std::pair<idx_t, idx_t>, string>> ranges;

	yyjson_obj_iter iter;
	yyjson_obj_iter_init(root, &iter);
	yyjson_val *key;
	while ((key = yyjson_obj_iter_next(&iter))) {
		auto val = yyjson_obj_iter_get_val(key);
		string name(yyjson_get_str(key), yyjson_get_len(key));

		if (name == "__metadata__") {
			if (!yyjson_is_obj(val)) {
				throw InvalidInputException("Safetensors '%s': __metadata__ must be an object of strings", source);
			}
			yyjson_obj_iter meta_iter;
			yyjson_obj_iter_init(val, &meta_iter);
			yyjson_val *meta_key;
			while ((meta_key = yyjson_obj_iter_next(&meta_iter))) {
				auto meta_val = yyjson_obj_iter_get_val(meta_key);
				if (!yyjson_is_str(meta_val)) {
					throw InvalidInputException("Safetensors '%s': __metadata__ values must be strings", source);
				}
				view.metadata.emplace(string(yyjson_get_str(meta_key), yyjson_get_len(meta_key)),
				                      string(yyjson_get_str(meta_val), yyjson_get_len(meta_val)));
			}
			continue;
		}

		if (view.tensors.find(name) != view.tensors.end()) {
			throw InvalidInputException("Safetensors '%s': duplicate tensor name '%s' in the header", source, name);
		}
		if (!yyjson_is_obj(val)) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' entry must be an object", source, name);
		}

		SafetensorsTensorInfo info;

		auto dtype_val = yyjson_obj_get(val, "dtype");
		if (!dtype_val || !yyjson_is_str(dtype_val)) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' is missing the \"dtype\" string", source,
			                            name);
		}
		info.dtype = ParseDtype(yyjson_get_str(dtype_val), source, name);

		auto shape_val = yyjson_obj_get(val, "shape");
		if (!shape_val || !yyjson_is_arr(shape_val)) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' is missing the \"shape\" array", source, name);
		}
		yyjson_arr_iter shape_iter;
		yyjson_arr_iter_init(shape_val, &shape_iter);
		yyjson_val *dim;
		while ((dim = yyjson_arr_iter_next(&shape_iter))) {
			info.shape.push_back(
			    UnsafeNumericCast<int64_t>(GetNonNegativeInt(dim, source, name, "shape")));
		}

		auto offsets_val = yyjson_obj_get(val, "data_offsets");
		if (!offsets_val || !yyjson_is_arr(offsets_val) || yyjson_arr_size(offsets_val) != 2) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' is missing the \"data_offsets\" "
			                            "[begin, end] array",
			                            source, name);
		}
		auto begin = GetNonNegativeInt(yyjson_arr_get(offsets_val, 0), source, name, "data_offsets");
		auto end = GetNonNegativeInt(yyjson_arr_get(offsets_val, 1), source, name, "data_offsets");
		if (begin > end) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' has data_offsets begin %llu > end %llu",
			                            source, name, static_cast<unsigned long long>(begin),
			                            static_cast<unsigned long long>(end));
		}
		if (end > view.data_section_size) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' data_offsets [%llu, %llu) out of bounds: "
			                            "the data section holds only %llu bytes",
			                            source, name, static_cast<unsigned long long>(begin),
			                            static_cast<unsigned long long>(end),
			                            static_cast<unsigned long long>(view.data_section_size));
		}

		info.data_offset = begin;
		info.data = view.data_section + begin;
		info.nbytes = end - begin;

		// Overflow-checked shape product: a crafted header could otherwise declare
		// dimensions whose product wraps idx_t to a small value that matches a tiny
		// data_offsets range, smuggling a huge shape past validation and over a
		// too-small backing buffer (OOB reads downstream in ORT).
		idx_t element_count = 1;
		for (auto dim : info.shape) {
			auto d = static_cast<idx_t>(dim); // dim already validated non-negative
			if (d != 0 && element_count > NumericLimits<idx_t>::Maximum() / d) {
				throw InvalidInputException("Safetensors '%s': tensor '%s' shape %s product overflows", source, name,
				                            ShapeToString(info.shape));
			}
			element_count *= d;
		}
		const idx_t dtype_size = SafetensorsDtypeSize(info.dtype);
		if (element_count != 0 && element_count > NumericLimits<idx_t>::Maximum() / dtype_size) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' byte size overflows", source, name);
		}
		auto expected_bytes = element_count * dtype_size;
		if (info.nbytes != expected_bytes) {
			throw InvalidInputException("Safetensors '%s': tensor '%s' declares %llu bytes but shape %s with "
			                            "dtype %s requires %llu bytes",
			                            source, name, static_cast<unsigned long long>(info.nbytes),
			                            ShapeToString(info.shape), SafetensorsDtypeName(info.dtype),
			                            static_cast<unsigned long long>(expected_bytes));
		}

		if (info.nbytes > 0) {
			ranges.emplace_back(std::make_pair(begin, end), name);
		}
		view.tensors.emplace(std::move(name), std::move(info));
	}

	std::sort(ranges.begin(), ranges.end());
	for (idx_t i = 1; i < ranges.size(); i++) {
		if (ranges[i].first.first < ranges[i - 1].first.second) {
			throw InvalidInputException("Safetensors '%s': tensors '%s' and '%s' have overlapping data ranges",
			                            source, ranges[i - 1].second, ranges[i].second);
		}
	}

	return view;
}

const MaterializedTensor &F32Arena::Get(const string &name) const {
	auto entry = tensors.find(name);
	if (entry == tensors.end()) {
		throw InvalidInputException("f32 arena has no tensor named '%s'", name);
	}
	return entry->second;
}

F32Arena MaterializeF32Arena(const SafetensorsView &view) {
	F32Arena result;
	for (auto &entry : view.tensors) {
		if (entry.second.dtype == SafetensorsDtype::BF16) {
			result.arena_bytes += entry.second.ElementCount() * sizeof(float);
		}
	}
	if (result.arena_bytes > 0) {
		result.arena = make_unsafe_uniq_array<data_t>(result.arena_bytes);
	}

	idx_t cursor = 0;
	for (auto &entry : view.tensors) {
		auto &info = entry.second;
		MaterializedTensor tensor;
		tensor.source_dtype = info.dtype;
		tensor.shape = info.shape;
		if (info.dtype == SafetensorsDtype::BF16) {
			// upcast into the arena: 16-bit shift into the high half (S02)
			auto count = info.ElementCount();
			auto src = reinterpret_cast<const uint16_t *>(info.data);
			auto dst = reinterpret_cast<float *>(result.arena.get() + cursor);
			for (idx_t i = 0; i < count; i++) {
				dst[i] = Bf16ToF32(src[i]);
			}
			tensor.dtype = SafetensorsDtype::F32;
			tensor.data = result.arena.get() + cursor;
			tensor.nbytes = count * sizeof(float);
			cursor += tensor.nbytes;
		} else {
			// F32 / I64 passthrough: zero copy, points at the source buffer
			tensor.dtype = info.dtype;
			tensor.data = info.data;
			tensor.nbytes = info.nbytes;
		}
		result.tensors.emplace(entry.first, std::move(tensor));
	}
	return result;
}

} // namespace anofox
} // namespace duckdb
