#pragma once

//===----------------------------------------------------------------------===//
// tabfm_owned_registry.hpp — heap objects owned by a query, freed by an idempotent handle or with the query.
//
// The predict aggregate's per-group row buffers (tabfm_predict_agg.cpp) are owned by the bind data of the query that created them, not only
// by the aggregate states that point at them: DuckDB does not destroy every state on every path (when one branch of a UNION ALL fails during
// finalize, the hash-aggregate states of another branch were dropped without their destructor callback), and a lost buffer leaked in every
// build and tripped the debug allocator's allocation_count == 0 check at exit.
//
// Handles, not pointers: a freed buffer's address can be reused, so a stale free by pointer could hit a new buffer; a stale handle is simply
// not found. test/cpp/test_tabfm_owned_registry.cpp pins the rules.
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {
namespace anofox {

struct OwnedRegistry {
	std::mutex lock;
	uint64_t next_id = 1;
	uint64_t next_owner = 1;
	std::unordered_map<uint64_t, std::pair<uint64_t, std::shared_ptr<void>>> objects;
};

//! Never destroyed (no static-destruction order against DuckDB's default allocator); empty whenever no query is running.
inline OwnedRegistry &Owned() {
	static auto *registry = new OwnedRegistry();
	return *registry;
}

inline uint64_t NewOwnerId() {
	auto &registry = Owned();
	std::lock_guard<std::mutex> guard(registry.lock);
	return registry.next_owner++;
}

//! Free everything still registered under `owner` (what no state freed).
inline void ReleaseOwner(uint64_t owner) {
	std::vector<std::shared_ptr<void>> orphans; // destroyed outside the lock
	{
		auto &registry = Owned();
		std::lock_guard<std::mutex> guard(registry.lock);
		for (auto it = registry.objects.begin(); it != registry.objects.end();) {
			if (it->second.first == owner) {
				orphans.push_back(std::move(it->second.second));
				it = registry.objects.erase(it);
			} else {
				++it;
			}
		}
	}
}

//! The owner a bind-data object carries. EVERY COPY SHARES IT, and what is registered under it goes when the last copy does: DuckDB 2.0
//! runs an aggregate through copies of its bind data and destroys a copy while the states built under it are still being combined and
//! finalized, so a per-copy owner freed buffers in use (SIGSEGV in GlobalUngroupedAggregateState::Finalize).
struct RowBufferOwner {
	struct Token {
		uint64_t id = NewOwnerId();
		~Token() {
			ReleaseOwner(id);
		}
	};
	std::shared_ptr<Token> token = std::make_shared<Token>();
	uint64_t Id() const {
		return token->id;
	}
};

//! Register `object` under `owner`; returns its handle.
inline uint64_t RegisterOwned(const RowBufferOwner &owner, std::shared_ptr<void> object) {
	auto &registry = Owned();
	std::lock_guard<std::mutex> guard(registry.lock);
	const auto id = registry.next_id++;
	registry.objects.emplace(id, std::make_pair(owner.Id(), std::move(object)));
	return id;
}

//! Free the object behind `id`. Idempotent: false when it is already gone (freed, or swept with its owner).
inline bool ReleaseOwned(uint64_t id) {
	std::shared_ptr<void> doomed;
	{
		auto &registry = Owned();
		std::lock_guard<std::mutex> guard(registry.lock);
		auto it = registry.objects.find(id);
		if (it == registry.objects.end()) {
			return false;
		}
		doomed = std::move(it->second.second);
		registry.objects.erase(it);
	}
	return true;
}

inline size_t LiveOwnedCount() {
	auto &registry = Owned();
	std::lock_guard<std::mutex> guard(registry.lock);
	return registry.objects.size();
}

} // namespace anofox
} // namespace duckdb
