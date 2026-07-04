// Catch2 tests for tabfm_ensemble — WS-F.
//
// Red-green parity against test/fixtures/golden_preprocess.json (seed 42):
//  * PyRandom (MT19937) reproduces the exact member configs EnsembleGenerator
//    draws for n_estimators {1,4} (feature permutation, class-shift offset,
//    per-member cat_mask, norm-method grouping) — FULL RNG PARITY.
//  * temperature softmax matches blend.softmax_temperature.
//  * NNLS blend reproduces BOTH golden toy cases (classification + regression):
//    raw weights, blended weights, and the blended probabilities/predictions.
//  * apply / inverse (feature permutation, class shift, logit un-shift).

#include "catch.hpp"

#include "tabfm_ensemble.hpp"

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

} // namespace

TEST_CASE("ensemble: PyRandom MT19937 matches CPython random", "[tabfm][ensemble]") {
	// Reference values from CPython 3: random.Random(42).getrandbits(32).
	PyRandom r(42);
	REQUIRE(r.GetRandBits(32) == 2746317213u);
	REQUIRE(r.GetRandBits(32) == 478163327u);
	// random.Random(42).sample(range(9), 9) reference sequence.
	PyRandom r2(42);
	auto s = r2.SampleRange(9, 9);
	CheckPerm(s, {1, 0, 5, 2, 8, 4, 7, 6, 3});
}

TEST_CASE("ensemble: member configs n_estimators=1 (golden)", "[tabfm][ensemble]") {
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

TEST_CASE("ensemble: member configs n_estimators=4 (golden)", "[tabfm][ensemble]") {
	EnsembleSpec spec;
	spec.n_estimators = 4;
	spec.seed = 42;
	spec.n_features = 9;
	spec.cat_feature_indices = {0, 1};
	spec.n_classes = 3;
	spec.classification = true;

	auto members = GenerateEnsemble(spec);
	REQUIRE(members.size() == 4);

	// flat 0: norm none
	REQUIRE(members[0].norm_method == "none");
	CheckPerm(members[0].feature_permutation, {6, 0, 7, 8, 1, 4, 2, 5, 3});
	REQUIRE(members[0].class_shift_offset == 0);
	CheckMask(members[0].cat_mask,
	          {false, true, false, false, true, false, false, false, false});
	// flat 1: norm none
	REQUIRE(members[1].norm_method == "none");
	CheckPerm(members[1].feature_permutation, {1, 0, 5, 2, 8, 4, 7, 6, 3});
	REQUIRE(members[1].class_shift_offset == 2);
	CheckMask(members[1].cat_mask,
	          {true, true, false, false, false, false, false, false, false});
	// flat 2: norm power
	REQUIRE(members[2].norm_method == "power");
	CheckPerm(members[2].feature_permutation, {5, 4, 1, 6, 2, 0, 3, 8, 7});
	REQUIRE(members[2].class_shift_offset == 2);
	CheckMask(members[2].cat_mask,
	          {false, false, true, false, false, true, false, false, false});
	// flat 3: norm power
	REQUIRE(members[3].norm_method == "power");
	CheckPerm(members[3].feature_permutation, {8, 6, 1, 3, 4, 2, 0, 5, 7});
	REQUIRE(members[3].class_shift_offset == 1);
	CheckMask(members[3].cat_mask,
	          {false, false, true, false, false, false, true, false, false});
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
