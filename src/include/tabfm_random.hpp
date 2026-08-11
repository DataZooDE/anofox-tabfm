//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_random.hpp — the single source of randomness for the extension.
//
// Every draw in anofox-tabfm goes through TabFMRandom, a thin wrapper over
// duckdb::RandomEngine (a vendored pcg32). Two consumers:
//
//   * tabfm_ensemble.cpp — feature permutations, class-shift offsets and the
//     config shuffle for the ensemble-diversity layer.
//   * tabfm_generate.cpp — the chain-rule column permutation, marginal draws,
//     weighted categorical draws and uniform-within-bin expansion.
//
// WHY pcg32 AND NOT A CPython MT19937 PORT
// ----------------------------------------
// An earlier revision shipped `PyRandom`, a bit-exact CPython `random.Random`
// port, so ensemble member configs matched upstream TabFM draw-for-draw. That
// bought a cross-language debugging oracle at the cost of ~140 lines of
// hand-ported Mersenne Twister plus a `SampleRange` branch that threw rather
// than diverge. The ensemble is a variance-reduction device: any set of valid
// permutations and class shifts is statistically equivalent, so bit-exact
// agreement was never required for correctness. It is gone; the tests now
// assert the structural invariants directly (a permutation really is a
// permutation, cat_mask really is composed through it), which is what the
// literal expectations were standing in for anyway.
//
// DETERMINISM CONTRACT
// --------------------
// Same seed -> same draws, on every platform: pcg32 is pure specified integer
// arithmetic (no std::*_distribution, whose output is explicitly not portable
// across standard libraries) and NextRandom() is ldexp(uint64, -64), IEEE754.
// The guarantee is scoped to a single build: duckdb/common/random_engine.hpp is
// a DuckDB *internal* header, so a submodule bump could in principle remap
// seed -> output. That is the documented promise in README (reproducible within
// a build). If cross-version stability is ever needed, vendor pcg32 here.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/random_engine.hpp"
#include "duckdb/common/vector.hpp"

#include <utility>

namespace duckdb {
namespace anofox {

//! Deterministic, seedable RNG. Not thread-safe by design — each consumer holds
//! its own instance (the predict/generate aggregates run one finalize per
//! group), so there is no shared state and no lock on the hot path.
class TabFMRandom {
public:
	explicit TabFMRandom(int64_t seed) : engine_(seed) {
	}

	//! Uniform double in [0, 1).
	double NextDouble() {
		return engine_.NextRandom();
	}

	//! Uniform double in [lo, hi). Returns `lo` when the interval is empty or
	//! inverted (degenerate bins are common: a quantile bin whose edges collapse
	//! because the column has ties).
	double NextDouble(double lo, double hi) {
		if (!(hi > lo)) {
			return lo;
		}
		return engine_.NextRandom(lo, hi);
	}

	//! Uniform integer in [0, n) — the half-open convention, matching
	//! RandomEngine::NextRandomInteger. Returns 0 for n == 0.
	//!
	//! The multiply-by-double mapping can round up to exactly `n` at the top of
	//! the range (NextRandom() < 1 holds, but u * n may round to n once n is
	//! large), so the result is clamped. Bias is O(n / 2^64) — irrelevant at the
	//! scales here (columns, bins, ensemble members).
	idx_t NextBelow(idx_t n) {
		if (n == 0) {
			return 0;
		}
		auto v = static_cast<idx_t>(NextDouble() * static_cast<double>(n));
		return v >= n ? n - 1 : v;
	}

	//! Fisher-Yates, in place.
	template <class T>
	void Shuffle(vector<T> &values) {
		if (values.size() <= 1) {
			return;
		}
		for (idx_t i = values.size() - 1; i >= 1; i--) {
			auto j = NextBelow(i + 1);
			std::swap(values[i], values[j]);
		}
	}

	//! `k` distinct values drawn from [0, n) without replacement, in draw order
	//! (a partial Fisher-Yates over a pool). k is clamped to n.
	vector<int64_t> SampleRange(int64_t n, int64_t k) {
		vector<int64_t> result;
		if (n <= 0 || k <= 0) {
			return result;
		}
		if (k > n) {
			k = n;
		}
		vector<int64_t> pool(static_cast<idx_t>(n));
		for (int64_t i = 0; i < n; i++) {
			pool[static_cast<idx_t>(i)] = i;
		}
		result.resize(static_cast<idx_t>(k));
		for (int64_t i = 0; i < k; i++) {
			auto j = NextBelow(static_cast<idx_t>(n - i));
			result[static_cast<idx_t>(i)] = pool[j];
			pool[j] = pool[static_cast<idx_t>(n - i - 1)];
		}
		return result;
	}

	//! A permutation of [0, n) — SampleRange(n, n) by another name, spelled out
	//! at the call sites that mean "shuffle the columns".
	vector<int64_t> Permutation(int64_t n) {
		return SampleRange(n, n);
	}

	//! Index sampled proportional to `weights`. Weights need not be normalized
	//! and negatives are treated as 0. Returns the last positive-weight index if
	//! floating-point drift leaves the cumulative scan short, and 0 when every
	//! weight is 0 (a uniform fallback would be a silent lie about the model's
	//! output; index 0 is at least deterministic and the callers guarantee a
	//! non-degenerate distribution).
	idx_t WeightedChoice(const vector<double> &weights) {
		if (weights.empty()) {
			return 0;
		}
		double total = 0;
		for (auto w : weights) {
			if (w > 0) {
				total += w;
			}
		}
		if (!(total > 0)) {
			return 0;
		}
		const double target = NextDouble() * total;
		double cumulative = 0;
		idx_t last_positive = 0;
		for (idx_t i = 0; i < weights.size(); i++) {
			if (!(weights[i] > 0)) {
				continue;
			}
			last_positive = i;
			cumulative += weights[i];
			if (target < cumulative) {
				return i;
			}
		}
		return last_positive;
	}

private:
	RandomEngine engine_;
};

} // namespace anofox
} // namespace duckdb
