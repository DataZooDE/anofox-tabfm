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
 * The stem therefore embeds a hash of the graph's CONTENT: same-named graphs
 * of different models get distinct cache entries (their bytes differ), while
 * the same graph staged under any path on any machine hashes identically —
 * which is what keeps anofox_tabfm_mxr_source sharing working across
 * machines (a path hash, the first fix, silently broke it). The
 * human-readable stem stays in front for debuggability. FNV-1a because the
 * plugin links no crypto library and this is a partition key, not a
 * security boundary.
 *
 * SECOND HALF (found 2026-09-29, when the first built-in pair of models sharing
 * one graph was bundled): a graph is weight-FREE, so hashing its bytes separates
 * models whose graphs differ and says nothing about two models that share a
 * byte-identical graph and differ only in weights. tabpfn-v2-5 and
 * tabpfn-v2-5-real are that pair. The compiled program bakes the weight VALUES
 * in, so on ROCm the second model asked for silently received the first one's
 * program: GPU real vs GPU regular differed by 0.0, CPU real vs CPU regular by
 * 0.115, no error. The stem therefore also carries a fingerprint of the weights
 * the graph will read -- size plus 64 sampled 4 KiB blocks, so a 6.6 GB
 * checkpoint costs ~256 KB of reads, and identical on every machine holding the
 * same file, so mxr_source sharing still works. Sampling is safe because it is a
 * partition key: two different checkpoints of one architecture differ in
 * essentially every block, and a fine-tune that differed ONLY in unsampled
 * blocks would need to be byte-identical elsewhere.
 *
 * Header-only and dependency-free: compiled into the standalone plugins AND
 * into the unittest binary, so the collision contract is enforced by tests
 * that do not need a GPU.
 */
#ifndef ANOFOX_TABFM_MXR_CACHE_KEY_HPP
#define ANOFOX_TABFM_MXR_CACHE_KEY_HPP

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace anofox_tabfm_mxr {

inline uint64_t Fnv1a64(const std::string &s) {
	uint64_t h = 1469598103934665603ULL;
	for (unsigned char c : s) {
		h ^= c;
		h *= 1099511628211ULL;
	}
	return h;
}

//! A fingerprint of ONE weights file: FNV-1a over its size and 64 sampled 4 KiB
//! blocks (the first, the last and 62 evenly between; the whole file when it is
//! smaller than that). 0 for a missing, unreadable or empty file, so callers get
//! a defined value rather than an exception on a path that is legitimately absent.
inline uint64_t WeightsFingerprint(const std::string &path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		return 0;
	}
	const uint64_t size = static_cast<uint64_t>(f.tellg());
	if (size == 0) {
		return 0;
	}
	constexpr uint64_t kBlock = 4096;
	constexpr uint64_t kSamples = 64;
	uint64_t h = 1469598103934665603ULL;
	auto mix = [&h](const char *data, size_t n) {
		for (size_t i = 0; i < n; i++) {
			h ^= static_cast<unsigned char>(data[i]);
			h *= 1099511628211ULL;
		}
	};
	// The size goes in first: a truncated or extended copy must not collide even
	// if every sampled block happened to match.
	mix(reinterpret_cast<const char *>(&size), sizeof(size));
	std::vector<char> buf(kBlock);
	if (size <= kBlock * kSamples) {
		f.seekg(0);
		for (uint64_t off = 0; off < size; off += kBlock) {
			const auto n = static_cast<std::streamsize>(std::min<uint64_t>(kBlock, size - off));
			f.read(buf.data(), n);
			mix(buf.data(), static_cast<size_t>(f.gcount()));
		}
		return h ? h : 1;
	}
	for (uint64_t i = 0; i < kSamples; i++) {
		// Multiply BEFORE dividing: `i * ((size - kBlock) / (kSamples - 1))` floors, so
		// the "last" sample stops up to 62 bytes short of the end and a change to the
		// file's final bytes goes unseen. (The test for this caught it.)
		const uint64_t off = (i * (size - kBlock)) / (kSamples - 1);
		f.clear();
		f.seekg(static_cast<std::streamoff>(off));
		f.read(buf.data(), static_cast<std::streamsize>(kBlock));
		mix(buf.data(), static_cast<size_t>(f.gcount()));
	}
	return h ? h : 1;
}

//! Combined fingerprint of every *.safetensors in a weights directory, in name
//! order, names included. A graph reads its weights by relative name from that
//! directory; fingerprinting all of them over-invalidates slightly (a fixture
//! holding both tasks' weights re-keys if either changes) and can never
//! under-invalidate, which is the direction that matters. 0 when there are none.
inline uint64_t DirWeightsFingerprint(const std::string &dir) {
	namespace fs = std::filesystem;
	std::error_code ec;
	std::vector<fs::path> files;
	for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
		if (it->path().extension() == ".safetensors") {
			files.push_back(it->path());
		}
	}
	if (files.empty()) {
		return 0;
	}
	std::sort(files.begin(), files.end());
	uint64_t h = 1469598103934665603ULL;
	for (auto &p : files) {
		const auto name = p.filename().string();
		const auto fp = WeightsFingerprint(p.string());
		for (unsigned char c : name) {
			h ^= c;
			h *= 1099511628211ULL;
		}
		for (int i = 0; i < 8; i++) {
			h ^= (fp >> (i * 8)) & 0xff;
			h *= 1099511628211ULL;
		}
	}
	return h ? h : 1;
}

//! "<filename stem>_<8 hex chars of content hash>[_w<8 hex chars of weights
//! fingerprint>]". content_hash is Fnv1a64 over the graph file's bytes (weight-
//! free graphs are ~1 MB; the plugin reads them for parsing anyway). The `_w`
//! part appears only for a nonzero fingerprint, so a graph with no external
//! weights keeps the key it always had.
inline std::string MxrCacheStem(const std::string &graph_path, uint64_t content_hash,
                                uint64_t weights_fingerprint = 0) {
	auto slash = graph_path.find_last_of("/\\");
	auto start = slash == std::string::npos ? 0 : slash + 1;
	auto dot = graph_path.rfind('.');
	if (dot == std::string::npos || dot < start) {
		dot = graph_path.size();
	}
	std::string stem = graph_path.substr(start, dot - start);
	static const char *hex = "0123456789abcdef";
	uint64_t h = content_hash;
	std::string suffix;
	for (int i = 7; i >= 0; i--) {
		suffix.push_back(hex[(h >> (i * 4)) & 0xf]);
	}
	std::string key = stem + "_" + suffix;
	if (weights_fingerprint != 0) {
		key += "_w";
		for (int i = 7; i >= 0; i--) {
			key.push_back(hex[(weights_fingerprint >> (i * 4)) & 0xf]);
		}
	}
	return key;
}

} // namespace anofox_tabfm_mxr

#endif
