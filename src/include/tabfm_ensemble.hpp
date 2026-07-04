//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_ensemble.hpp — C++ port of the TabFM ensemble-diversity layer
// (HLD §4.5 "v2", BRD FR-3.x). Reproduces, against test/fixtures/
// golden_preprocess.json (seed 42), the per-member data views that
// EnsembleGenerator draws, the class-shift apply/inverse, the temperature
// softmax, and the NNLS blend-weight solve.
//
// Upstream reference (never edited):
//   vendor/tabfm/tabfm/src/classifier_and_regressor.py
//
// RNG-PARITY STATUS: FULL PARITY.
//   The upstream draws come from Python's `random.Random(seed)` (Mersenne
//   Twister MT19937). This module ships a faithful MT19937 + CPython
//   `getrandbits` / `_randbelow_with_getrandbits` / `random.sample` /
//   `random.shuffle` port (PyRandom), so the generated member configs
//   (feature permutation, class-shift offset, per-member cat_mask, norm method
//   grouping) match the golden member_configs BIT-FOR-BIT. The tests assert
//   this equality. Scope of the port: the ensemble presets exercised by the
//   goldens — norm_methods {'none','power'}, feat_shuffle='random',
//   class_shift=on, and NO feature-crosses / SVD / row-subsample / cat-permute
//   (all disabled in the shipped presets). Those disabled paths draw extra RNG
//   values upstream; if they are ever enabled, this port must be extended (it
//   throws rather than silently diverging).
//
// C++ symbol → upstream Python symbol map
// ---------------------------------------
//   PyRandom                         → Python stdlib `random.Random` (MT19937)
//   PyRandom::SampleRange            → random.sample(range(n), k) (pool alg.)
//   PyRandom::Shuffle                → random.shuffle (Fisher–Yates)
//   FeatureShuffle (internal)        → FeatureShuffler.shuffle
//   GenerateEnsemble / EnsembleMember→ EnsembleGenerator._generate_ensemble +
//                                       prepare_ensemble_tensors (flattened,
//                                       grouped by norm method)
//   ApplyFeaturePermutation          → X_variant[:, shuffle_pattern]
//   ShiftClassContext                → (y + shift_offset) % n_classes
//   UnshiftLogits                    → concat(out[...,off:], out[...,:off])
//                                       in TabFM*._predict_proba_internal
//   SoftmaxTemperature               → TabFMClassifier.softmax (temperature)
//   SolveNNLS                        → scipy.optimize.nnls (Lawson–Hanson)
//   BlendClassification              → _process_logits NNLS branch
//                                       (tensordot(weights, probs))
//   BlendRegression                  → _combine_predictions NNLS branch
//   NnlsBlendWeights                 → nnls_beta * w_norm + (1-beta)/E
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/constants.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/string.hpp"

#include <cstdint>
#include <utility>

namespace duckdb {
namespace anofox {

//! Default ensemble seed (upstream _DEFAULT_RANDOM_STATE).
static constexpr uint32_t kDefaultEnsembleSeed = 42;
//! Default temperature applied before the classification softmax.
static constexpr double kDefaultSoftmaxTemperature = 0.9;
//! Default NNLS blend mixing weight against the uniform vector.
static constexpr double kDefaultNnlsBeta = 0.75;

//===----------------------------------------------------------------------===//
// PyRandom — CPython `random.Random` (MT19937) faithful port
//===----------------------------------------------------------------------===//

//! Reproduces the exact draw sequence of Python's stdlib Mersenne Twister so
//! ensemble member configs match the reference bit-for-bit. Only the surface
//! the ensemble needs is implemented (getrandbits/randbelow/sample/shuffle).
class PyRandom {
public:
	explicit PyRandom(uint64_t seed) {
		SeedInt(seed);
	}

	//! Seed exactly as `random.seed(int)` does (init_by_array of the int words).
	void SeedInt(uint64_t seed);
	//! CPython getrandbits(k) for 1 <= k <= 64.
	uint64_t GetRandBits(int k);
	//! CPython Random._randbelow_with_getrandbits(n).
	uint64_t RandBelow(uint64_t n);
	//! random.sample(range(n), k) — returns the drawn population values.
	vector<int64_t> SampleRange(int64_t n, int64_t k);
	//! random.shuffle(x) in place (Fisher–Yates, CPython order).
	template <class T>
	void Shuffle(vector<T> &x) {
		if (x.size() <= 1) {
			return;
		}
		for (idx_t i = x.size() - 1; i >= 1; i--) {
			uint64_t j = RandBelow((uint64_t)i + 1);
			std::swap(x[i], x[(idx_t)j]);
		}
	}

private:
	uint32_t GenrandUint32();
	static constexpr int N = 624;
	uint32_t mt_[624];
	int mti_ = N + 1;

	void InitGenrand(uint32_t s);
	void InitByArray(const uint32_t *key, int key_length);
};

//===----------------------------------------------------------------------===//
// Ensemble member generation
//===----------------------------------------------------------------------===//

//! One ensemble member's data-view configuration, in the flattened order
//! prepare_ensemble_tensors produces (all 'none'-norm members first, then the
//! remaining norm methods in order).
struct EnsembleMember {
	//! Normalization method for this member ("none" or "power").
	string norm_method;
	//! Feature permutation into the unique-filtered encoded columns.
	vector<int64_t> feature_permutation;
	//! Class-shift offset applied to the context labels (classification).
	int64_t class_shift_offset = 0;
	//! Categorical mask AFTER applying the permutation (length == d).
	vector<bool> cat_mask;
	//! Active feature count (== feature_permutation.size()).
	int64_t d = 0;
};

//! Inputs describing the (post-unique-filter) feature schema and task.
struct EnsembleSpec {
	int64_t n_estimators = 1;
	uint32_t seed = kDefaultEnsembleSeed;
	//! Number of features after the unique filter (EnsembleGenerator n_features).
	int64_t n_features = 0;
	//! Categorical feature indices among the filtered features.
	vector<int64_t> cat_feature_indices;
	//! Number of classes (classification); 0 for regression.
	int64_t n_classes = 0;
	//! True for classification (enables class shift), false for regression.
	bool classification = true;
	//! Normalization methods; defaults to {"none","power"} when empty.
	vector<string> norm_methods;
};

//! Generate the flattened ensemble member configs for `spec`, matching
//! EnsembleGenerator._generate_ensemble + prepare_ensemble_tensors for the
//! shipped presets (see RNG-PARITY note above). Throws if an unsupported
//! (extra-RNG-drawing) option would be required.
vector<EnsembleMember> GenerateEnsemble(const EnsembleSpec &spec);

//===----------------------------------------------------------------------===//
// Apply / inverse transforms
//===----------------------------------------------------------------------===//

//! Permute the columns of a row-major [rows, H] matrix by `permutation`
//! (out[:, i] = in[:, permutation[i]]). Returns a [rows, permutation.size()]
//! matrix (row-major).
vector<double> ApplyFeaturePermutation(const vector<double> &x, idx_t rows,
                                       idx_t cols,
                                       const vector<int64_t> &permutation);

//! Shift class context labels: out[i] = (y[i] + offset) % n_classes.
vector<int64_t> ShiftClassContext(const vector<int64_t> &y, int64_t offset,
                                  int64_t n_classes);

//! Un-shift a member's per-row logits/probs back to canonical class order:
//! out[..., j] = in[..., (j + offset) % K]. `logits` is row-major [rows, K].
vector<double> UnshiftLogits(const vector<double> &logits, idx_t rows, idx_t k,
                             int64_t offset);

//! Temperature softmax over the last axis of a row-major [rows, K] matrix,
//! with max-subtraction for stability (upstream TabFM*.softmax).
vector<double> SoftmaxTemperature(const vector<double> &logits, idx_t rows,
                                  idx_t k, double temperature);

//===----------------------------------------------------------------------===//
// NNLS blend
//===----------------------------------------------------------------------===//

//! Solve min_{x>=0} ||A x - b||_2 via the Lawson–Hanson active-set algorithm.
//! `a` is row-major [m, n]; `b` has length m; result has length n. `out_rnorm`
//! (if non-null) receives the residual 2-norm. Mirrors scipy.optimize.nnls.
vector<double> SolveNNLS(const vector<double> &a, idx_t m, idx_t n,
                         const vector<double> &b, double *out_rnorm = nullptr);

//! Combine raw NNLS weights with the uniform vector: normalize the raw weights
//! (uniform if they sum to 0) then blend beta*w_norm + (1-beta)/E.
vector<double> NnlsBlendWeights(const vector<double> &weights_raw,
                                double nnls_beta);

//! Classification blend: probs_all is [E, N, K] flattened (member-major,
//! row-major within), weights length E. Returns [N, K] row-major
//! sum_e weights[e] * probs_all[e].
vector<double> BlendClassification(const vector<double> &probs_all, idx_t e,
                                   idx_t n, idx_t k,
                                   const vector<double> &weights);

//! Regression blend: member_predictions is [E, N] flattened, weights length E.
//! Returns length-N vector sum_e weights[e] * member_predictions[e].
vector<double> BlendRegression(const vector<double> &member_predictions,
                               idx_t e, idx_t n, const vector<double> &weights);

} // namespace anofox
} // namespace duckdb
