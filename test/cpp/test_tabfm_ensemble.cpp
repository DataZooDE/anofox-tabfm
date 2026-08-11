// Catch2 tests for tabfm_ensemble — WS-F.
//
//  * TabFMRandom (duckdb::RandomEngine / pcg32) satisfies the properties the
//    ensemble relies on: seeded determinism, half-open ranges, real
//    permutations, weighted draws that track their weights.
//  * member configs for n_estimators {1,4} satisfy their STRUCTURAL invariants
//    — a permutation really is a permutation, cat_mask is composed through it,
//    class shifts are in range, norm methods group in emission order.
//
//    These deliberately do NOT pin literal permutations. An earlier revision
//    asserted the exact draws of a CPython MT19937 port so member configs
//    matched upstream TabFM bit-for-bit; the port is gone (see
//    tabfm_random.hpp). The ensemble is a variance-reduction device — any set
//    of valid permutations is statistically equivalent — so the invariants
//    below are what the literals were standing in for, asserted directly and
//    without a hand-ported Mersenne Twister to maintain.
//
//  * temperature softmax matches blend.softmax_temperature.
//  * NNLS blend reproduces BOTH golden toy cases (classification + regression):
//    raw weights, blended weights, and the blended probabilities/predictions.
//    These goldens STAY: NNLS and softmax have unique mathematically-defined
//    answers, so checking them against scipy is checking arithmetic, not
//    mimicking another implementation's arbitrary choices.
//  * apply / inverse (feature permutation, class shift, logit un-shift).

#include "catch.hpp"

#include "tabfm_ensemble.hpp"
#include "tabfm_random.hpp"

#include <algorithm>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

void CheckVec(const vector<double> &actual, const std::vector<double> &expected,
              double eps = 1e-6, double margin = 1e-9) {
	REQUIRE(actual.size() == expected.size());
	for (size_t i = 0; i < expected.size(); i++) {
		INFO("index " << i << " actual=" << actual[i] << " expected=" << expected[i]);
		REQUIRE(actual[i] == Approx(expected[i]).epsilon(eps).margin(margin));
	}
}

void CheckPerm(const vector<int64_t> &actual, const std::vector<int64_t> &expected) {
	REQUIRE(actual.size() == expected.size());
	for (size_t i = 0; i < expected.size(); i++) {
		INFO("perm index " << i);
		REQUIRE(actual[i] == expected[i]);
	}
}

void CheckMask(const vector<bool> &actual, const std::vector<bool> &expected) {
	REQUIRE(actual.size() == expected.size());
	for (size_t i = 0; i < expected.size(); i++) {
		INFO("mask index " << i);
		REQUIRE(actual[i] == expected[i]);
	}
}

//! A permutation of [0, n): right length, every value in range, no repeats.
void RequirePermutationOf(const vector<int64_t> &perm, int64_t n) {
	REQUIRE(perm.size() == (size_t)n);
	std::vector<bool> seen((size_t)n, false);
	for (auto v : perm) {
		INFO("value " << v << " of " << n);
		REQUIRE(v >= 0);
		REQUIRE(v < n);
		REQUIRE_FALSE(seen[(size_t)v]);
		seen[(size_t)v] = true;
	}
}

} // namespace

TEST_CASE("random: seeded determinism and divergence", "[tabfm][random]") {
	TabFMRandom a(42), b(42), c(43);
	std::vector<double> da, db, dc;
	for (int i = 0; i < 64; i++) {
		da.push_back(a.NextDouble());
		db.push_back(b.NextDouble());
		dc.push_back(c.NextDouble());
	}
	// Same seed -> identical stream.
	REQUIRE(da == db);
	// Different seed -> a different stream (not merely a shifted one).
	REQUIRE(da != dc);
}

TEST_CASE("random: NextDouble stays in [0,1) and spans the range", "[tabfm][random]") {
	TabFMRandom r(7);
	double lo = 1.0, hi = 0.0, sum = 0.0;
	const int kDraws = 20000;
	for (int i = 0; i < kDraws; i++) {
		double v = r.NextDouble();
		REQUIRE(v >= 0.0);
		REQUIRE(v < 1.0);
		lo = std::min(lo, v);
		hi = std::max(hi, v);
		sum += v;
	}
	// "Close enough": a uniform generator's mean is 1/2, and 20k draws should
	// reach into both tails. Loose bounds — this catches a broken generator
	// (constant, clumped, out of range), not a subtly biased one.
	REQUIRE(sum / kDraws == Approx(0.5).margin(0.02));
	REQUIRE(lo < 0.01);
	REQUIRE(hi > 0.99);
}

TEST_CASE("random: NextBelow is half-open and covers every value", "[tabfm][random]") {
	TabFMRandom r(11);
	REQUIRE(r.NextBelow(0) == 0);
	REQUIRE(r.NextBelow(1) == 0);
	std::vector<int> counts(5, 0);
	for (int i = 0; i < 5000; i++) {
		auto v = r.NextBelow(5);
		// The boundary that is easy to get wrong: n itself must never appear.
		REQUIRE(v < 5);
		counts[(size_t)v]++;
	}
	for (auto c : counts) {
		REQUIRE(c > 700); // uniform would be 1000 each
	}
}

TEST_CASE("random: NextDouble(lo,hi) respects bounds and degenerate spans", "[tabfm][random]") {
	TabFMRandom r(3);
	for (int i = 0; i < 1000; i++) {
		double v = r.NextDouble(-2.5, 4.0);
		REQUIRE(v >= -2.5);
		REQUIRE(v < 4.0);
	}
	// A collapsed quantile bin (ties in the column) must not produce garbage.
	REQUIRE(r.NextDouble(1.5, 1.5) == 1.5);
	REQUIRE(r.NextDouble(2.0, 1.0) == 2.0);
}

TEST_CASE("random: SampleRange draws without replacement", "[tabfm][random]") {
	TabFMRandom r(42);
	auto full = r.SampleRange(9, 9);
	RequirePermutationOf(full, 9);

	auto partial = r.SampleRange(20, 5);
	REQUIRE(partial.size() == 5);
	std::vector<int64_t> sorted(partial.begin(), partial.end());
	std::sort(sorted.begin(), sorted.end());
	REQUIRE(std::unique(sorted.begin(), sorted.end()) == sorted.end());
	for (auto v : partial) {
		REQUIRE(v >= 0);
		REQUIRE(v < 20);
	}

	// k > n is clamped, not an error (callers pass min(n_estimators, perms)).
	REQUIRE(r.SampleRange(3, 10).size() == 3);
	REQUIRE(r.SampleRange(0, 5).empty());
	REQUIRE(r.SampleRange(5, 0).empty());
}

TEST_CASE("random: WeightedChoice tracks its weights", "[tabfm][random]") {
	TabFMRandom r(5);
	// Degenerate distribution: the certain label must always win.
	for (int i = 0; i < 100; i++) {
		REQUIRE(r.WeightedChoice({0.0, 1.0, 0.0}) == 1);
	}
	// Zero-weight entries are never drawn, and the empirical shares track the
	// weights. This is the property tabfm_generate depends on when it samples a
	// class from the model's proba MAP.
	vector<double> weights = {0.6, 0.0, 0.3, 0.1};
	std::vector<int> counts(4, 0);
	const int kDraws = 20000;
	for (int i = 0; i < kDraws; i++) {
		counts[r.WeightedChoice(weights)]++;
	}
	REQUIRE(counts[1] == 0);
	REQUIRE((double)counts[0] / kDraws == Approx(0.6).margin(0.02));
	REQUIRE((double)counts[2] / kDraws == Approx(0.3).margin(0.02));
	REQUIRE((double)counts[3] / kDraws == Approx(0.1).margin(0.02));
	// Un-normalized weights behave identically.
	REQUIRE(r.WeightedChoice({0.0, 0.0}) == 0);
	REQUIRE(r.WeightedChoice({}) == 0);
}

TEST_CASE("ensemble: member configs n_estimators=1", "[tabfm][ensemble]") {
	EnsembleSpec spec;
	spec.n_estimators = 1;
	spec.seed = 42;
	spec.n_features = 9;
	spec.cat_feature_indices = {0, 1};
	spec.n_classes = 3;
	spec.classification = true;

	auto members = GenerateEnsemble(spec);
	REQUIRE(members.size() == 1);
	REQUIRE(members[0].norm_method == "none");
	CheckPerm(members[0].feature_permutation, {0, 1, 2, 3, 4, 5, 6, 7, 8});
	REQUIRE(members[0].class_shift_offset == 0);
	REQUIRE(members[0].d == 9);
	CheckMask(members[0].cat_mask,
	          {true, true, false, false, false, false, false, false, false});
}

TEST_CASE("ensemble: member configs n_estimators=4 invariants", "[tabfm][ensemble]") {
	EnsembleSpec spec;
	spec.n_estimators = 4;
	spec.seed = 42;
	spec.n_features = 9;
	spec.cat_feature_indices = {0, 1};
	spec.n_classes = 3;
	spec.classification = true;

	auto members = GenerateEnsemble(spec);
	REQUIRE(members.size() == 4);

	// prepare_ensemble_tensors emits all members of norm_methods[0] before any
	// of norm_methods[1]; with E=4 and {none,power} cycled that is 2 and 2.
	REQUIRE(members[0].norm_method == "none");
	REQUIRE(members[1].norm_method == "none");
	REQUIRE(members[2].norm_method == "power");
	REQUIRE(members[3].norm_method == "power");

	for (idx_t i = 0; i < members.size(); i++) {
		INFO("member " << i);
		auto &m = members[i];
		RequirePermutationOf(m.feature_permutation, spec.n_features);
		REQUIRE(m.d == spec.n_features);

		// The composition that is genuinely easy to invert: cat_mask is indexed
		// by POST-permutation position, and is true exactly where the permutation
		// points at an original categorical column ({0,1} here).
		REQUIRE(m.cat_mask.size() == (size_t)spec.n_features);
		for (idx_t j = 0; j < m.cat_mask.size(); j++) {
			auto source = m.feature_permutation[j];
			bool source_is_cat = (source == 0 || source == 1);
			INFO("position " << j << " <- source column " << source);
			REQUIRE(m.cat_mask[j] == source_is_cat);
		}
		// Exactly as many categorical flags as there are categorical columns.
		REQUIRE(std::count(m.cat_mask.begin(), m.cat_mask.end(), true) == 2);

		REQUIRE(m.class_shift_offset >= 0);
		REQUIRE(m.class_shift_offset < spec.n_classes);
	}
}

TEST_CASE("ensemble: member configs are deterministic per seed", "[tabfm][ensemble]") {
	EnsembleSpec spec;
	spec.n_estimators = 4;
	spec.seed = 42;
	spec.n_features = 9;
	spec.cat_feature_indices = {0, 1};
	spec.n_classes = 3;
	spec.classification = true;

	auto a = GenerateEnsemble(spec);
	auto b = GenerateEnsemble(spec);
	REQUIRE(a.size() == b.size());
	for (idx_t i = 0; i < a.size(); i++) {
		INFO("member " << i);
		CheckPerm(a[i].feature_permutation, std::vector<int64_t>(b[i].feature_permutation.begin(),
		                                                         b[i].feature_permutation.end()));
		REQUIRE(a[i].class_shift_offset == b[i].class_shift_offset);
		REQUIRE(a[i].norm_method == b[i].norm_method);
	}

	// A different seed must actually move the draws, or the ensemble has no
	// diversity to contribute.
	spec.seed = 43;
	auto c = GenerateEnsemble(spec);
	bool any_differs = false;
	for (idx_t i = 0; i < a.size() && !any_differs; i++) {
		any_differs = a[i].feature_permutation != c[i].feature_permutation;
	}
	REQUIRE(any_differs);
}

TEST_CASE("ensemble: members are distinct when the permutation space is small",
          "[tabfm][ensemble]") {
	// <= 5 features enumerates permutations and samples without replacement, so
	// four members must be four DIFFERENT views (drawing independently would
	// collide often at 4! = 24 possibilities).
	EnsembleSpec spec;
	spec.n_estimators = 4;
	spec.seed = 42;
	spec.n_features = 4;
	spec.cat_feature_indices = {0};
	spec.n_classes = 3;
	spec.classification = true;

	auto members = GenerateEnsemble(spec);
	REQUIRE(members.size() == 4);
	for (auto &m : members) {
		RequirePermutationOf(m.feature_permutation, spec.n_features);
	}
	for (idx_t i = 0; i < members.size(); i++) {
		for (idx_t j = i + 1; j < members.size(); j++) {
			INFO("members " << i << " and " << j);
			REQUIRE(members[i].feature_permutation != members[j].feature_permutation);
		}
	}
}

TEST_CASE("ensemble: apply / inverse transforms", "[tabfm][ensemble]") {
	// 2 rows x 3 cols, permute to [2,0,1].
	vector<double> x = {10, 11, 12, 20, 21, 22};
	auto px = ApplyFeaturePermutation(x, 2, 3, {2, 0, 1});
	CheckVec(px, {12, 10, 11, 22, 20, 21});

	// class shift (golden member1: shift 2, 3 classes).
	auto shifted = ShiftClassContext({1, 2, 0, 1, 2, 1, 0, 2}, 2, 3);
	REQUIRE(shifted == vector<int64_t>({0, 1, 2, 0, 1, 0, 2, 1}));

	// logit un-shift is the inverse of the class shift rotation.
	vector<double> logits = {0.1, 0.2, 0.7};
	auto un = UnshiftLogits(logits, 1, 3, 1);
	CheckVec(un, {0.2, 0.7, 0.1});
}

TEST_CASE("ensemble: temperature softmax (golden)", "[tabfm][ensemble]") {
	vector<double> logits = {2.0, 1.0, 0.5, -1.0, 0.0, 1.0, 0.3, 0.3, 0.3};
	auto probs = SoftmaxTemperature(logits, 3, 3, 0.9);
	CheckVec(probs,
	         {0.6587317636120843, 0.216849877427433, 0.1244183589604827,
	          0.07538325148667331, 0.22899409853365954, 0.6956226499796672,
	          0.3333333333333333, 0.3333333333333333, 0.3333333333333333});
}

TEST_CASE("ensemble: NNLS classification blend (golden)", "[tabfm][ensemble]") {
	const idx_t E = 4, N = 6, K = 3;
	// oof logits [E,N,K] flat.
	std::vector<double> logits = {
	    0.304717, -1.039984, 0.750451, 0.940565, -1.951035, -1.30218,
	    0.12784, -0.316243, -0.016801, -0.853044, 0.879398, 0.777792,
	    0.066031, 1.127241, 0.467509, -0.859292, 0.368751, -0.958883,
	    0.87845, -0.049926, -0.184862, -0.68093, 1.222541, -0.154529,
	    -0.428328, -0.352134, 0.532309, 0.365444, 0.412733, 0.430821,
	    2.141648, -0.406415, -0.512243, -0.813773, 0.615979, 1.128972,
	    -0.113947, -0.840156, -0.824481, 0.650593, 0.743254, 0.543154,
	    -0.66551, 0.232161, 0.116686, 0.218689, 0.871429, 0.223596,
	    0.678914, 0.067579, 0.289119, 0.631288, -1.457156, -0.319671,
	    -0.470373, -0.638878, -0.275142, 1.494941, -0.865831, 0.968278,
	    -1.68287, -0.334885, 0.162753, 0.586222, 0.711227, 0.793347,
	    -0.348725, -0.462352, 0.857976, -0.191304, -1.275686, -1.133287};
	vector<double> lin(logits.begin(), logits.end());

	// softmax(t=0.9) over the E*N rows.
	auto probs = SoftmaxTemperature(lin, E * N, K, 0.9);
	std::vector<double> exp_probs = {
	    0.3489947520131356, 0.07833109594434347, 0.5726741520425209,
	    0.890480224667062, 0.03583309854318312, 0.07368667678975488,
	    0.40616254375314415, 0.24797498595608397, 0.345862470290772,
	    0.07154287459960627, 0.49040543644180923, 0.4380516889585845,
	    0.17200733120617043, 0.5592850778465773, 0.2687075909472523,
	    0.1721472230230901, 0.6737386531394437, 0.1541141238374662,
	    0.6012171010878891, 0.21431078192476988, 0.18447211698734117,
	    0.09021922379895582, 0.7478558954670521, 0.16192488073399222,
	    0.20015684230886724, 0.21784009201425458, 0.5820030656768782,
	    0.319561023304296, 0.3368007755848981, 0.34363820111080595,
	    0.8998086192237411, 0.05303759407617554, 0.04715378670008332,
	    0.0687001525225742, 0.3364214712756469, 0.594878376201779,
	    0.5262272701029431, 0.23482353166520645, 0.2389491982318504,
	    0.3337876370419346, 0.3699846950992644, 0.2962276678588009,
	    0.16404102300240916, 0.44475730840517624, 0.3912016685924147,
	    0.24565504511272734, 0.5073468859734476, 0.2469980689138251,
	    0.46393303391740265, 0.23521008823156517, 0.3008568778510321,
	    0.6916329842702713, 0.06793573005793951, 0.24043128567178917,
	    0.32557343289478136, 0.26998334143825387, 0.40444322566696483,
	    0.6136535655696407, 0.04453834677831483, 0.34180808765204446,
	    0.07550049444517501, 0.3376130159346365, 0.5868864896201885,
	    0.29344637659653167, 0.33717071128562165, 0.36938291211784663,
	    0.17533364894740722, 0.15453776527821564, 0.6701285857743772,
	    0.6057507254134445, 0.18156244939975918, 0.21268682518679619};
	CheckVec(probs, exp_probs);

	// Build A = oof_flat.T with oof_flat = probs reshaped (E, N*K); b = one-hot.
	const idx_t M = N * K; // 18
	vector<double> A(M * E);
	for (idx_t e = 0; e < E; e++) {
		for (idx_t nk = 0; nk < M; nk++) {
			A[nk * E + e] = probs[e * M + nk];
		}
	}
	vector<double> b = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0};

	double rnorm = 0;
	auto w_raw = SolveNNLS(A, M, E, b, &rnorm);
	CheckVec(w_raw, {0.29821718990717355, 0.6967279237429832, 0.0, 0.0}, 1e-5);

	auto w_final = NnlsBlendWeights(w_raw, 0.75);
	CheckVec(w_final,
	         {0.2872992269742677, 0.5877007730257323, 0.0625, 0.0625}, 1e-5);

	auto blended = BlendClassification(probs, E, N, K, w_final);
	std::vector<double> exp_blend = {
	    0.5068392214744502, 0.18000550508675975, 0.3131552734387902,
	    0.3680712629143705, 0.47571799950673577, 0.15621073757889387,
	    0.24929386063796538, 0.24816595751815018, 0.5025401818438845,
	    0.24205431184964135, 0.39161352878931277, 0.36633215936104585,
	    0.6181899620614997, 0.21621164640591944, 0.16559839153258085,
	    0.17091937870003546, 0.4068733891769045, 0.4222072321230601};
	CheckVec(blended, exp_blend, 1e-5);
}

TEST_CASE("ensemble: NNLS regression blend (golden)", "[tabfm][ensemble]") {
	const idx_t E = 3, N = 6;
	std::vector<double> members = {
	    0.724164, 2.149148, 3.042728, 4.207146, 4.871824, 6.047562,
	    1.3, 2.4000000000000004, 3.5000000000000004, 4.6000000000000005, 5.7, 6.800000000000001,
	    3.5, 3.5, 3.5, 3.5, 3.5, 3.5};
	vector<double> mem(members.begin(), members.end());
	vector<double> y_orig = {1, 2, 3, 4, 5, 6};

	// A = y_oof.T (N x E); b = y_orig.
	vector<double> A(N * E);
	for (idx_t e = 0; e < E; e++) {
		for (idx_t n = 0; n < N; n++) {
			A[n * E + e] = mem[e * N + n];
		}
	}
	auto w_raw = SolveNNLS(A, N, E, y_orig);
	CheckVec(w_raw, {0.23194062177122302, 0.6685131341162237, 0.0}, 1e-5);

	auto w_final = NnlsBlendWeights(w_raw, 0.75);
	CheckVec(w_final,
	         {0.27651978537603106, 0.6401468812906357, 0.08333333333333333}, 1e-5);

	auto blended = BlendRegression(mem, E, N, w_final);
	CheckVec(blended,
	         {1.3241032862015412, 2.422301125465519, 3.373555244701532,
	          4.399701429569219, 5.287659616893087, 6.316936005731232},
	         1e-5);
}
