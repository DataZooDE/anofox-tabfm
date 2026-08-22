/*
 * The .mxr program-cache stem for a compiled MIGraphX graph.
 *
 * The cache key used to be the graph's filename stem alone — and every model's
 * bundled graph is staged as graph_migraphx_<task>.onnx beside its own
 * weights, so two different models produced the SAME stem and silently loaded
 * each other's compiled programs (found 2026-08-22: mitra on ROCm returned
 * tabfm-v1's answers for 24/30 query rows; scores looked plausible because the
 * program was a real model — just the wrong one).
 *
 * The stem therefore embeds a hash of the graph's full path: same-named graphs
 * in different directories (different models, or different registered models
 * sharing a filename) get distinct cache entries, while the human-readable
 * stem stays in front for debuggability. FNV-1a because the plugin links no
 * crypto library and this is a partition key, not a security boundary.
 *
 * Header-only and dependency-free: compiled into the standalone plugins AND
 * into the unittest binary, so the collision contract is enforced by tests
 * that do not need a GPU.
 */
#ifndef ANOFOX_TABFM_MXR_CACHE_KEY_HPP
#define ANOFOX_TABFM_MXR_CACHE_KEY_HPP

#include <cstdint>
#include <string>

namespace anofox_tabfm_mxr {

inline uint64_t Fnv1a64(const std::string &s) {
	uint64_t h = 1469598103934665603ULL;
	for (unsigned char c : s) {
		h ^= c;
		h *= 1099511628211ULL;
	}
	return h;
}

//! "<filename stem>_<8 hex chars of full-path hash>". The path — not the
//! bytes — because the path identifies which model's weights sit beside the
//! graph, which is what the compiled program actually depends on.
inline std::string MxrCacheStem(const std::string &graph_path) {
	auto slash = graph_path.find_last_of("/\\");
	auto start = slash == std::string::npos ? 0 : slash + 1;
	auto dot = graph_path.rfind('.');
	if (dot == std::string::npos || dot < start) {
		dot = graph_path.size();
	}
	std::string stem = graph_path.substr(start, dot - start);
	static const char *hex = "0123456789abcdef";
	uint64_t h = Fnv1a64(graph_path);
	std::string suffix;
	for (int i = 7; i >= 0; i--) {
		suffix.push_back(hex[(h >> (i * 4)) & 0xf]);
	}
	return stem + "_" + suffix;
}

} // namespace anofox_tabfm_mxr

#endif
