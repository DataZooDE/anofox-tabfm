/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_soname_patch.hpp — rename the ORT core's SONAME, in place.
 *
 * Half of the SONAME-shadowing fix (docs/GPU_HARDENING_PLAN.md S2; the other
 * half is the loader's RTLD_DEEPBIND). The CUDA plugin ships its own ORT-GPU
 * core; if that core keeps upstream's SONAME (`libonnxruntime.so.1`) and the
 * host process already loaded a shared ORT under the same name, glibc reuses
 * the host's copy without ever examining the plugin's — measured, all four
 * quadrants of a (SONAME × DEEPBIND) matrix. So the shipped core is renamed to
 * `libanofoxort_gpu.so`, deliberately the same length, which makes the rename
 * a byte-for-byte `.dynstr` replacement: no section offsets move, and ELF hash
 * tables are unaffected (they index symbol names, not the SONAME string).
 *
 * The match includes the terminating NUL, so the SONAME entry cannot be
 * confused with a longer string it prefixes (e.g. "libonnxruntime.so.1.28.0"
 * as a file name in some note). Exactly one occurrence is required: zero means
 * the wheel changed shape, more than one means the assumption that only the
 * `.dynstr` entry carries the string no longer holds — both must fail loudly
 * at download time, never surface later as a plugin that half-loads.
 *
 * Dependency-free on purpose (the tabfm_shape_bucket.hpp precedent): the
 * extractor, the CMake plugin build and the tests all include just this.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <cstddef>
#include <cstring>

namespace duckdb {
namespace anofox {

//! Upstream's SONAME and our replacement. Equal length is load-bearing — the
//! patch is in-place — and guarded by static_assert below.
constexpr const char *ORT_UPSTREAM_SONAME = "libonnxruntime.so.1";
constexpr const char *ORT_RENAMED_SONAME = "libanofoxort_gpu.so";

namespace soname_detail {
constexpr size_t ConstLen(const char *s) {
	size_t n = 0;
	while (s[n] != '\0') {
		n++;
	}
	return n;
}
} // namespace soname_detail

static_assert(soname_detail::ConstLen(ORT_UPSTREAM_SONAME) == soname_detail::ConstLen(ORT_RENAMED_SONAME),
              "the SONAME patch is in-place: replacement must be exactly as long as the original");

//! Replace every NUL-terminated occurrence of ORT_UPSTREAM_SONAME in `data`
//! with ORT_RENAMED_SONAME and return how many were replaced. The caller
//! decides what count is acceptable (the extractor requires exactly 1).
inline size_t PatchOrtSonameInPlace(char *data, size_t size) {
	const size_t name_len = soname_detail::ConstLen(ORT_UPSTREAM_SONAME);
	const size_t needle_len = name_len + 1; // include the terminating NUL
	if (!data || size < needle_len) {
		return 0;
	}
	size_t replaced = 0;
	for (size_t i = 0; i + needle_len <= size; i++) {
		if (data[i] == ORT_UPSTREAM_SONAME[0] && std::memcmp(data + i, ORT_UPSTREAM_SONAME, name_len) == 0 &&
		    data[i + name_len] == '\0') {
			std::memcpy(data + i, ORT_RENAMED_SONAME, name_len);
			replaced++;
			i += name_len; // no overlapping matches
		}
	}
	return replaced;
}

} // namespace anofox
} // namespace duckdb
