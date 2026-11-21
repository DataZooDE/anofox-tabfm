//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P2a (docs/GGML_PLAN.md): rank reduction. Written red first against a stub that returned
// the shapes unchanged.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_rank_reduce.hpp"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

using namespace duckdb::anofox;
using S = RankShape;

namespace {

int64_t Numel(const S &s) {
	int64_t n = 1;
	for (auto d : s) {
		n *= d;
	}
	return n;
}

//! Reference: element-wise a+b with numpy broadcasting, written the slow obvious way on the ORIGINAL shapes.
std::vector<double> NaiveBroadcastAdd(const std::vector<double> &a, const S &sa, const std::vector<double> &b,
                                      const S &sb, const S &out) {
	const size_t rank = out.size();
	std::vector<double> result(static_cast<size_t>(Numel(out)));
	std::vector<int64_t> idx(rank, 0);
	auto offset = [&](const S &shape) {
		int64_t off = 0, stride = 1;
		for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--) {
			const int oi = i + static_cast<int>(rank) - static_cast<int>(shape.size());
			off += (shape[i] == 1 ? 0 : idx[oi]) * stride;
			stride *= shape[i];
		}
		return off;
	};
	for (size_t n = 0; n < result.size(); n++) {
		result[n] = a[offset(sa)] + b[offset(sb)];
		for (int d = static_cast<int>(rank) - 1; d >= 0; d--) {
			if (++idx[d] < out[d]) {
				break;
			}
			idx[d] = 0;
		}
	}
	return result;
}

S BroadcastOf(const S &a, const S &b) {
	const size_t rank = std::max(a.size(), b.size());
	S out(rank);
	for (size_t i = 0; i < rank; i++) {
		const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
		const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
		out[i] = da == 1 ? db : da;
	}
	return out;
}

} // namespace

TEST_CASE("rank_reduce: identical shapes collapse to one dimension", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}});
	REQUIRE(r.out == S {720});
	REQUIRE(r.operands[0] == S {720});
	REQUIRE(r.operands[1] == S {720});
}

TEST_CASE("rank_reduce: size-1 dimensions shared by every operand are dropped", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{1, 10, 1, 64}, {1, 10, 1, 64}});
	REQUIRE(r.out == S {640});
	r = CoalesceBroadcast({{1, 5, 1}, {1, 5, 1}});
	REQUIRE(r.out == S {5});
	r = CoalesceBroadcast({{1, 1, 1}, {1}});
	REQUIRE(r.out == S {1}); // a scalar stays a rank-1 tensor of one element
}

TEST_CASE("rank_reduce: a broadcast axis stays its own dimension", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{8, 1, 32}, {8, 5, 32}});
	REQUIRE(r.out == S {8, 5, 32});
	REQUIRE(r.operands[0] == S {8, 1, 32});
	REQUIRE(r.operands[1] == S {8, 5, 32});
}

TEST_CASE("rank_reduce: neighbouring dimensions every operand treats alike are merged", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{2, 3, 1, 4, 5}, {2, 3, 7, 4, 5}});
	REQUIRE(r.out == S {6, 7, 20});
	REQUIRE(r.operands[0] == S {6, 1, 20});
	REQUIRE(r.operands[1] == S {6, 7, 20});
}

TEST_CASE("rank_reduce: operands of lower rank are aligned from the right", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{4, 6, 8}, {8}});
	REQUIRE(r.out == S {24, 8});
	REQUIRE(r.operands[0] == S {24, 8});
	REQUIRE(r.operands[1] == S {1, 8});
}

TEST_CASE("rank_reduce: a Nori-style rank-6 attention shape fits ggml's four dimensions", "[tabfm][rank_reduce]") {
	// [1, rows, groups, heads, 1, d] against a per-head scale [1, 1, 1, heads, 1, 1]
	auto r = CoalesceBroadcast({{1, 40, 3, 8, 1, 32}, {1, 1, 1, 8, 1, 1}});
	REQUIRE(r.out.size() <= 4);
	REQUIRE(r.out == S {120, 8, 32});
}

TEST_CASE("rank_reduce: zero-size tensors reduce to an empty one-dimensional result and keep every operand's count", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{0, 3}, {1, 3}});
	REQUIRE(r.out == S {0});
	REQUIRE(r.operands[0] == S {0});
	REQUIRE(r.operands[1] == S {3}); // three elements stay three elements, whatever the caller does with them
	r = CoalesceBroadcast({{2, 0}, {2, 1}, {1, 1}});
	REQUIRE(r.out == S {0});
	REQUIRE(r.operands == std::vector<S> {S {0}, S {2}, S {1}});
}

TEST_CASE("rank_reduce: three operands and a rank-0 operand", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{4, 1, 6}, {4, 5, 6}, {}});
	REQUIRE(r.out == S {4, 5, 6});
	REQUIRE(r.operands[0] == S {4, 1, 6});
	REQUIRE(r.operands[1] == S {4, 5, 6});
	REQUIRE(r.operands[2] == S {1, 1, 1});
	r = CoalesceBroadcast({{}, {}});
	REQUIRE(r.out == S {1});
	r = CoalesceBroadcast({{2, 3}, {3}, {2, 1}});
	REQUIRE(r.out == S {2, 3});
	REQUIRE(r.operands[1] == S {1, 3});
	REQUIRE(r.operands[2] == S {2, 1});
}

TEST_CASE("rank_reduce: shapes that do not broadcast are refused with both shapes named", "[tabfm][rank_reduce]") {
	try {
		CoalesceBroadcast({{2, 3}, {4, 3}});
		FAIL("expected an exception");
	} catch (const std::runtime_error &e) {
		const std::string message = e.what();
		REQUIRE(message.find("2,3") != std::string::npos);
		REQUIRE(message.find("4,3") != std::string::npos);
	}
}

// The property that matters: the reduced computation gives the same numbers as the naive one on the original
// shapes. Random ranks up to 6, random broadcast patterns, random data.
TEST_CASE("rank_reduce: the reduced element-wise add equals the naive broadcast add", "[tabfm][rank_reduce]") {
	std::mt19937 rng(7);
	for (int trial = 0; trial < 600; trial++) {
		const int rank = 1 + static_cast<int>(rng() % 6);
		S sa(rank), sb(rank);
		for (int i = 0; i < rank; i++) {
			const int64_t d = static_cast<int64_t>(rng() % 4); // 0..3: zero-size tensors included
			const int kind = static_cast<int>(rng() % 4);       // both, a broadcasts, b broadcasts, both 1
			sa[i] = kind == 1 || kind == 3 ? 1 : d;
			sb[i] = kind == 2 || kind == 3 ? 1 : d;
		}
		// unequal ranks: drop leading size-1 dimensions from one operand
		while (sb.size() > 1 && sb[0] == 1 && rng() % 2) {
			sb.erase(sb.begin());
		}
		std::vector<double> a(static_cast<size_t>(Numel(sa))), b(static_cast<size_t>(Numel(sb)));
		for (auto &v : a) {
			v = static_cast<double>(rng() % 1000);
		}
		for (auto &v : b) {
			v = static_cast<double>(rng() % 1000);
		}
		const S out = BroadcastOf(sa, sb);
		const auto want = NaiveBroadcastAdd(a, sa, b, sb, out);

		const auto r = CoalesceBroadcast({sa, sb});
		INFO("trial " << trial << " rank " << rank << " a " << RankShapeToString(sa) << " b " << RankShapeToString(sb));
		REQUIRE(Numel(r.out) == Numel(out));
		REQUIRE(Numel(r.operands[0]) == Numel(sa));
		REQUIRE(Numel(r.operands[1]) == Numel(sb));
		REQUIRE(r.operands[0].size() == r.out.size());
		REQUIRE(r.operands[1].size() == r.out.size());
		if (Numel(out) > 0) {
			const auto got = NaiveBroadcastAdd(a, r.operands[0], b, r.operands[1], r.out);
			REQUIRE(got == want);
			// minimal: no dimension of size 1 everywhere, and no two neighbours with the same broadcast pattern
			for (size_t i = 0; i < r.out.size(); i++) {
				if (r.out.size() > 1) {
					REQUIRE(r.out[i] != 1);
				}
				if (i > 0) {
					const bool same_a = (r.operands[0][i] == 1 && r.out[i] != 1) == (r.operands[0][i - 1] == 1 && r.out[i - 1] != 1);
					const bool same_b = (r.operands[1][i] == 1 && r.out[i] != 1) == (r.operands[1][i - 1] == 1 && r.out[i - 1] != 1);
					REQUIRE(!(same_a && same_b));
				}
			}
		}
		REQUIRE(r.out.size() <= std::max(sa.size(), sb.size()));
	}
}

// Rank is information when operands broadcast along alternating axes: a bcast, b bcast, a bcast, ... cannot merge.
// Those are the residual rank-5 tensors the plan sends to custom ops (docs/GGML_PLAN.md).
TEST_CASE("rank_reduce: alternating broadcast patterns keep their rank", "[tabfm][rank_reduce]") {
	auto r = CoalesceBroadcast({{3, 1, 3, 1, 3}, {1, 4, 1, 4, 1}});
	REQUIRE(r.out == S {3, 4, 3, 4, 3});
	REQUIRE(r.out.size() == 5);
}
