//===----------------------------------------------------------------------===//
// test_tabfm_distribution_decode.cpp — RDIST-02, MGEN-03
//
// Catch2 golden tests for the distribution decode helpers:
//   DistributionMean, DistributionQuantile, ValidateDistributionOutput
//
// Golden values derived from a Python reference implementation of
// FullSupportBarDistribution decode:
//   K=8, non-uniform borders {-5,-3,-1.5,-0.5,0,0.5,1.5,3,5}
//   logits {0.1,0.5,1.2,2.0,1.8,0.9,0.3,-0.2}
//
//   probs = softmax(logits):
//     [0.0458, 0.0683, 0.1375, 0.3061, 0.2506, 0.1019, 0.0559, 0.0339]
//
//   mean (FullSupportBarDistribution with outer-bin half-normal correction):
//     -0.16643975 (z-space)
//
//   quantiles at {0.1,0.5,0.9}:
//     q0.1 = -1.80915018, q0.5 = -0.09423481, q0.9 = 1.40018675
//
//   affine transform (y_std=3.0, y_mean=10.0):
//     raw_mean = 9.50068074
//     raw_q0.1 = 4.57254945, raw_q0.5 = 9.71729558, raw_q0.9 = 14.20056026
//
// Non-uniform borders are essential: the outer bin widths are 2.0 and the inner
// bins are 0.5. A uniform-bin assumption (width=range/K) would mis-place the
// outer-bin corrections and produce wrong quantiles visibly far from these
// golden values.
//===----------------------------------------------------------------------===//

#include "catch.hpp"
#include "tabfm_ort_engine.hpp" // ValidateDistributionOutput
#include "tabfm_predict.hpp"    // DistributionMean, DistributionQuantile

#include <cmath>

using namespace duckdb::anofox;
using namespace duckdb;

namespace {

//! K=8 non-uniform border fixture (widths: 2.0, 1.5, 1.0, 0.5, 0.5, 1.0, 1.5, 2.0)
const vector<double> kBorders = {-5.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0};
//! Raw logits (before softmax) for one test row
const vector<double> kLogits = {0.1, 0.5, 1.2, 2.0, 1.8, 0.9, 0.3, -0.2};

//! Numerically stable softmax (mirrors SoftmaxInPlace in tabfm_engine.cpp)
vector<double> Softmax(const vector<double> &x) {
	double m = x[0];
	for (auto v : x) {
		if (v > m) {
			m = v;
		}
	}
	vector<double> out(x.size());
	double sum = 0.0;
	for (size_t i = 0; i < x.size(); i++) {
		out[i] = std::exp(x[i] - m);
		sum += out[i];
	}
	for (auto &v : out) {
		v /= sum;
	}
	return out;
}

} // anonymous namespace

//===--------------------------------------------------------------------===//
// Test 1: DistributionMean — golden mean within 1e-4 (z-space, non-uniform)
//===--------------------------------------------------------------------===//
TEST_CASE("distribution decode: DistributionMean matches golden (non-uniform borders)", "[tabfm][distribution_decode]") {
	auto probs = Softmax(kLogits);
	double mean = DistributionMean(probs, kBorders);
	// Python golden: -0.16643975 (FullSupportBarDistribution with half-normal outer-bin correction)
	CHECK(std::abs(mean - (-0.16643975)) < 1e-4);
}

//===--------------------------------------------------------------------===//
// Test 2: DistributionQuantile — q=0.1, q=0.5, q=0.9 match golden (z-space)
//
// Asserts that the CDF inverse uses actual bucket widths: a uniform-bin
// assumption would produce wrong results visibly outside the 1e-4 tolerance.
//===--------------------------------------------------------------------===//
TEST_CASE("distribution decode: DistributionQuantile matches golden at q={0.1,0.5,0.9} (non-uniform)", "[tabfm][distribution_decode]") {
	auto probs = Softmax(kLogits);
	double q01 = DistributionQuantile(probs, kBorders, 0.1);
	double q05 = DistributionQuantile(probs, kBorders, 0.5);
	double q09 = DistributionQuantile(probs, kBorders, 0.9);
	// Python goldens
	CHECK(std::abs(q01 - (-1.80915018)) < 1e-4);
	CHECK(std::abs(q05 - (-0.09423481)) < 1e-4);
	CHECK(std::abs(q09 - 1.40018675) < 1e-4);
	// median is within the distribution interior (not equal to 0.1 or 0.9 quantiles)
	CHECK(q01 < q05);
	CHECK(q05 < q09);
}

//===--------------------------------------------------------------------===//
// Test 3: Affine transform → raw-space mean and quantiles
//
// raw = z * y_std + y_mean; confirms the affine transform is applied correctly
// and does not corrupt the non-uniform-border decode (Pitfall 4).
//===--------------------------------------------------------------------===//
TEST_CASE("distribution decode: affine transform (z-space → raw-space) matches golden", "[tabfm][distribution_decode]") {
	const double y_std = 3.0;
	const double y_mean = 10.0;

	// Transform z-space borders to raw space
	vector<double> raw_borders(kBorders.size());
	for (size_t i = 0; i < kBorders.size(); i++) {
		raw_borders[i] = kBorders[i] * y_std + y_mean;
	}

	auto probs = Softmax(kLogits);
	double raw_mean = DistributionMean(probs, raw_borders);
	double raw_q01 = DistributionQuantile(probs, raw_borders, 0.1);
	double raw_q05 = DistributionQuantile(probs, raw_borders, 0.5);
	double raw_q09 = DistributionQuantile(probs, raw_borders, 0.9);

	// Python goldens (raw-space = z-space * 3.0 + 10.0)
	CHECK(std::abs(raw_mean - 9.50068074) < 1e-4);
	CHECK(std::abs(raw_q01 - 4.57254945) < 1e-4);
	CHECK(std::abs(raw_q05 - 9.71729558) < 1e-4);
	CHECK(std::abs(raw_q09 - 14.20056026) < 1e-4);
}

//===--------------------------------------------------------------------===//
// Test 4: ValidateDistributionOutput — rejects contract violations (MGEN-03)
//===--------------------------------------------------------------------===//
TEST_CASE("distribution decode: ValidateDistributionOutput rejects bad contracts (MGEN-03)", "[tabfm][distribution_decode]") {
	const idx_t N = 4; // n_test rows

	// Good output: logits [N, K=8], borders [K+1=9]
	{
		TabFMRunOutput good;
		good.shape = {static_cast<int64_t>(N), 8};
		good.logits.assign(N * 8, 0.0f);
		good.borders.assign(9, 0.0f);
		// Must not throw
		REQUIRE_NOTHROW(ValidateDistributionOutput(good, N));
	}

	// Bad: wrong borders length (K+1 mismatch)
	{
		TabFMRunOutput bad;
		bad.shape = {static_cast<int64_t>(N), 8};
		bad.logits.assign(N * 8, 0.0f);
		bad.borders.assign(7, 0.0f); // should be 9 = K+1
		REQUIRE_THROWS_AS(ValidateDistributionOutput(bad, N), InvalidInputException);
	}

	// Bad: wrong rank (rank-3 instead of rank-2)
	{
		TabFMRunOutput bad;
		bad.shape = {1, static_cast<int64_t>(N), 8};
		bad.logits.assign(N * 8, 0.0f);
		bad.borders.assign(9, 0.0f);
		REQUIRE_THROWS_AS(ValidateDistributionOutput(bad, N), InvalidInputException);
	}

	// Bad: wrong n_test in first dimension
	{
		TabFMRunOutput bad;
		bad.shape = {static_cast<int64_t>(N + 1), 8}; // wrong n_test
		bad.logits.assign((N + 1) * 8, 0.0f);
		bad.borders.assign(9, 0.0f);
		REQUIRE_THROWS_AS(ValidateDistributionOutput(bad, N), InvalidInputException);
	}

	// Bad: empty borders (distribution_output=true but no borders tensor read)
	{
		TabFMRunOutput bad;
		bad.shape = {static_cast<int64_t>(N), 8};
		bad.logits.assign(N * 8, 0.0f);
		// borders empty
		REQUIRE_THROWS_AS(ValidateDistributionOutput(bad, N), InvalidInputException);
	}
}
