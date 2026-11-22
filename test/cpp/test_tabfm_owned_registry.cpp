// Ownership of the predict aggregate's per-group row buffers (tabfm_owned_registry.hpp).
//
// DuckDB does not destroy every aggregate state on every path (a failing branch of a UNION ALL dropped the others' states without their
// destructor), so the buffers are owned by the query's bind data and freed with it. DuckDB 2.0 runs an aggregate through COPIES of its
// bind data and destroys a copy while the states built under it are still being combined and finalized: freeing on the copy's
// destruction read freed memory in finalize (SIGSEGV in GlobalUngroupedAggregateState::Finalize). The rule pinned here: every copy
// shares one owner, and the buffers go only when the last copy does.

#include "catch.hpp"

#include "tabfm_owned_registry.hpp"

#include <memory>

using namespace duckdb::anofox;

TEST_CASE("owned registry: a copy of the owner does not free what was registered under it while another copy lives", "[tabfm][owned_registry]") {
	const auto before = LiveOwnedCount();
	std::weak_ptr<int> watch;
	{
		RowBufferOwner original;
		uint64_t id = 0;
		{
			RowBufferOwner copy = original; // DuckDB 2.0 executes the aggregate through a copy like this one
			auto object = std::make_shared<int>(42);
			watch = object;
			id = RegisterOwned(copy, object);
		} // the copy is destroyed mid-query
		REQUIRE(!watch.expired()); // still alive: the original (the plan's bind data) shares the owner
		REQUIRE(LiveOwnedCount() == before + 1);
		REQUIRE(ReleaseOwned(id));  // the state's destructor frees it
		REQUIRE(!ReleaseOwned(id)); // idempotent: a second free finds nothing
		REQUIRE(watch.expired());
	}
	REQUIRE(LiveOwnedCount() == before);
}

TEST_CASE("owned registry: what no state freed goes when the last copy of the owner does", "[tabfm][owned_registry]") {
	const auto before = LiveOwnedCount();
	std::weak_ptr<int> lost;
	uint64_t id = 0;
	{
		RowBufferOwner original;
		RowBufferOwner copy(original);
		auto object = std::make_shared<int>(7);
		lost = object;
		id = RegisterOwned(original, object); // a state DuckDB never destroys
		RowBufferOwner assigned;
		assigned = copy; // assignment shares the owner too
	}
	REQUIRE(lost.expired()); // swept with the owner
	REQUIRE(!ReleaseOwned(id)); // a late destructor callback after the sweep is a no-op, never a double free
	REQUIRE(LiveOwnedCount() == before);

	// two unrelated queries keep their own buffers
	RowBufferOwner a, b;
	auto kept = std::make_shared<int>(1);
	std::weak_ptr<int> kept_watch = kept;
	const auto kept_id = RegisterOwned(a, kept);
	kept.reset();
	{
		RowBufferOwner other = b;
		RegisterOwned(other, std::make_shared<int>(2));
	}
	b = RowBufferOwner(); // the last holder of b's owner lets go: only b's buffer goes
	REQUIRE(!kept_watch.expired());
	REQUIRE(ReleaseOwned(kept_id));
}
