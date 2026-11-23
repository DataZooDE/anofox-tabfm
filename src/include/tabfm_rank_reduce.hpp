#pragma once

//===----------------------------------------------------------------------===//
// tabfm_rank_reduce.hpp — shrink tensor ranks for ggml, which is 4-D at most (docs/GGML_PLAN.md, P2).
//
// 16 of the 38 bundled graphs use rank-5 and rank-6 tensors. Most of that rank is not information: size-1
// dimensions, and neighbouring dimensions that every operand of an op treats the same way (all broadcast, or all
// not). Dropping the former and merging the latter leaves the same element-wise result in a lower rank, the way a
// TensorIterator coalesces dimensions. Ops that give an axis meaning (reductions, softmax, concat, gather,
// transpose) are handled by their own lowering in P3; this file is the broadcast case.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

using RankShape = std::vector<int64_t>;

struct CoalescedBroadcast {
	//! One reduced shape per operand, all of the same rank, aligned with `out`.
	std::vector<RankShape> operands;
	RankShape out;
};

inline std::string RankShapeToString(const RankShape &shape) {
	std::string text = "[";
	for (size_t i = 0; i < shape.size(); i++) {
		text += (i ? "," : "") + std::to_string(shape[i]);
	}
	return text + "]";
}

//! Numpy-style broadcast of `operands` (right-aligned), coalesced: dimensions every operand has as 1 are dropped,
//! and neighbouring dimensions that each operand either broadcasts across both of or not across either are merged.
//! The element-wise result on the reduced shapes holds the same numbers as on the originals, in the same order.
//! Throws std::runtime_error naming the shapes when they do not broadcast.
inline CoalescedBroadcast CoalesceBroadcast(const std::vector<RankShape> &operands) {
	size_t rank = 0;
	for (const auto &shape : operands) {
		rank = std::max(rank, shape.size());
	}
	std::vector<RankShape> padded;
	for (const auto &shape : operands) {
		RankShape p(rank - shape.size(), 1);
		p.insert(p.end(), shape.begin(), shape.end());
		padded.push_back(std::move(p));
	}
	RankShape out(rank, 1);
	for (size_t i = 0; i < rank; i++) {
		for (size_t k = 0; k < padded.size(); k++) {
			const int64_t d = padded[k][i];
			if (d == 1 || d == out[i]) {
				continue;
			}
			if (out[i] != 1) {
				std::string all;
				for (const auto &shape : operands) {
					all += (all.empty() ? "" : " and ") + RankShapeToString(shape);
				}
				throw std::runtime_error("shapes do not broadcast: " + all);
			}
			out[i] = d;
		}
	}

	CoalescedBroadcast result;
	int64_t numel = 1;
	for (auto d : out) {
		numel *= d;
	}
	if (numel == 0) {
		result.out = {0};
		for (const auto &p : padded) {
			int64_t n = 1;
			for (auto d : p) {
				n *= d;
			}
			result.operands.push_back({n}); // keep the operand's element count even though nothing is computed
		}
		return result;
	}

	result.operands.assign(padded.size(), RankShape());
	std::vector<bool> previous;
	for (size_t i = 0; i < rank; i++) {
		if (out[i] == 1) {
			continue; // size 1 for every operand
		}
		std::vector<bool> pattern;
		for (const auto &p : padded) {
			pattern.push_back(p[i] == 1); // this operand broadcasts across the dimension
		}
		if (!result.out.empty() && pattern == previous) {
			result.out.back() *= out[i];
			for (size_t k = 0; k < padded.size(); k++) {
				if (!pattern[k]) {
					result.operands[k].back() *= padded[k][i];
				}
			}
		} else {
			result.out.push_back(out[i]);
			for (size_t k = 0; k < padded.size(); k++) {
				result.operands[k].push_back(pattern[k] ? 1 : padded[k][i]);
			}
		}
		previous = pattern;
	}
	if (result.out.empty()) {
		result.out = {1};
		for (auto &o : result.operands) {
			o = {1};
		}
	}
	return result;
}

} // namespace anofox
} // namespace duckdb
