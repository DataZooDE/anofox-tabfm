//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_shape_bucket.hpp — MIGraphX shape-bucket padding (HLD §9).
//
// Pure C++, no duckdb types: this is included both by the extension
// (tabfm_devices.cpp, which wraps it in a duckdb exception) and by the
// standalone MIGraphX plugin (tabfm_migraphx_plugin.cpp), which cannot link
// duckdb at all. One table, so the two can never pad a shape differently and
// silently miss each other's .mxr cache.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace duckdb {
namespace anofox {

struct ShapeBucket {
	int64_t padded_t;
	int64_t padded_h;
};

//! Pad (T, H) up to the nearest compiled-session bucket:
//! T in {128, 512, 1024, 2048, 4096, 10000}, H in {16, 64, 128, 256, 512}.
//! Padding is semantically inert: the model's `train_size` and `d` masks
//! ignore padded rows/columns (S01 validated `d < H` parity at 5e-8).
//! Values above the largest bucket are returned unchanged (callers guard via
//! anofox_tabfm_max_rows / anofox_tabfm_max_features before reaching here).
//! Throws std::invalid_argument for non-positive dimensions.
inline ShapeBucket PadToShapeBucket(int64_t rows_t, int64_t features_h) {
	if (rows_t <= 0 || features_h <= 0) {
		throw std::invalid_argument("shape bucket needs positive dimensions, got T=" + std::to_string(rows_t) +
		                            " H=" + std::to_string(features_h));
	}
	static constexpr int64_t T_BUCKETS[] = {128, 512, 1024, 2048, 4096, 10000};
	static constexpr int64_t H_BUCKETS[] = {16, 64, 128, 256, 512};

	auto pad_up = [](int64_t value, const int64_t *buckets, size_t count) {
		for (size_t i = 0; i < count; i++) {
			if (value <= buckets[i]) {
				return buckets[i];
			}
		}
		// Above the largest bucket: return unchanged; callers guard via the
		// anofox_tabfm_max_rows / anofox_tabfm_max_features settings.
		return value;
	};

	ShapeBucket bucket;
	bucket.padded_t = pad_up(rows_t, T_BUCKETS, sizeof(T_BUCKETS) / sizeof(T_BUCKETS[0]));
	bucket.padded_h = pad_up(features_h, H_BUCKETS, sizeof(H_BUCKETS) / sizeof(H_BUCKETS[0]));
	return bucket;
}

} // namespace anofox
} // namespace duckdb
