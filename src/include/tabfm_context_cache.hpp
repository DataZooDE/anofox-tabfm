#pragma once

//===----------------------------------------------------------------------===//
// tabfm_context_cache.hpp — the context cache's admission window (docs/CONTEXT_CACHE_PLAN.md, "Phase 1b").
// SplitOrtBackend in tabfm_engine.cpp uses it; it is declared here so the Catch2 tests can pin it without
// an ORT session.
//===----------------------------------------------------------------------===//

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

#include <list>

namespace duckdb {
namespace anofox {

//! How many recently seen context keys the admission rule remembers, per loaded model. A key costs about
//! 64 bytes here (hash, list node, map node), so this is about 256 KB at most.
static constexpr idx_t CONTEXT_SIGHTING_WINDOW = 4096;

//! A bounded, least-recently-used set of recently seen context keys (hashes only, never the rows). The
//! admission rule keeps a context the second time its key is seen within the window. A hash collision
//! only admits a context one sighting early; it can never answer for the wrong rows, because a hit still
//! needs the full compare.
class ContextSightings {
public:
	explicit ContextSightings(idx_t capacity = CONTEXT_SIGHTING_WINDOW) : capacity(capacity) {
	}
	//! Record a sighting of `key`. Returns whether the key was already in the window (seen before), and makes
	//! it the most recently seen; the least recently seen key drops out beyond the capacity.
	bool Sight(hash_t key);
	idx_t Size() const {
		return order.size();
	}
	idx_t Capacity() const {
		return capacity;
	}

private:
	idx_t capacity;
	std::list<hash_t> order; // most recently seen first
	unordered_map<hash_t, std::list<hash_t>::iterator> index;
};

} // namespace anofox
} // namespace duckdb
