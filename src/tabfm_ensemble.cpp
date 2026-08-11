//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_ensemble.cpp — implementation of the ensemble-diversity layer. See
// tabfm_ensemble.hpp for the C++ -> upstream Python symbol map and the RNG
// note (draws come from TabFMRandom / pcg32, not a CPython MT19937 port —
// structurally equivalent, not draw-for-draw identical). Reference (never
// edited):
//   vendor/tabfm/tabfm/src/classifier_and_regressor.py
//===----------------------------------------------------------------------===//

#include "tabfm_ensemble.hpp"

#include "tabfm_random.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// Ensemble member generation (EnsembleGenerator._generate_ensemble subset)
//===----------------------------------------------------------------------===//

namespace {

//! A pre-grouping ensemble config tuple (feature perm, class shift).
struct RawConfig {
	vector<int64_t> feature_permutation;
	int64_t class_shift_offset = 0;
};

//! FeatureShuffler.shuffle: n_estimators permutations of range(n_features).
vector<vector<int64_t>> FeatureShuffle(int64_t n_features, int64_t n_estimators,
                                       uint32_t seed) {
	vector<vector<int64_t>> patterns;
	if (n_estimators == 1) {
		vector<int64_t> ident((idx_t)n_features);
		for (int64_t i = 0; i < n_features; i++) {
			ident[(idx_t)i] = i;
		}
		patterns.push_back(std::move(ident));
		return patterns;
	}
	TabFMRandom rng(seed);
	if (n_features <= 5) {
		// Small feature space: enumerate every permutation and sample WITHOUT
		// replacement, so members are guaranteed distinct. Drawing independently
		// (the branch below) would repeat permutations often enough at <= 5
		// features to collapse the ensemble's diversity.
		vector<vector<int64_t>> all_perms;
		vector<int64_t> base((idx_t)n_features);
		for (int64_t i = 0; i < n_features; i++) {
			base[(idx_t)i] = i;
		}
		vector<int64_t> perm = base;
		std::sort(perm.begin(), perm.end());
		do {
			all_perms.push_back(perm);
		} while (std::next_permutation(perm.begin(), perm.end()));
		// itertools.permutations order == lexicographic for sorted input; ok.
		int64_t take = std::min<int64_t>(n_estimators, (int64_t)all_perms.size());
		vector<int64_t> idx = rng.SampleRange((int64_t)all_perms.size(), take);
		for (auto id : idx) {
			patterns.push_back(all_perms[(idx_t)id]);
		}
	} else {
		for (int64_t e = 0; e < n_estimators; e++) {
			patterns.push_back(rng.Permutation(n_features));
		}
	}
	return patterns;
}

} // namespace

vector<EnsembleMember> GenerateEnsemble(const EnsembleSpec &spec) {
	vector<string> norm_methods = spec.norm_methods;
	if (norm_methods.empty()) {
		norm_methods = {"none", "power"};
	}
	const int64_t E = spec.n_estimators;

	// 1. Feature shuffle patterns (FeatureShuffler owns its own Random(seed)).
	vector<vector<int64_t>> shuffle_patterns =
	    FeatureShuffle(spec.n_features, E, spec.seed);
	// If fewer patterns than estimators, cycle (FeatureShuffler tail logic).
	if ((int64_t)shuffle_patterns.size() < E) {
		vector<vector<int64_t>> cycled;
		for (int64_t i = 0; i < E; i++) {
			cycled.push_back(shuffle_patterns[(idx_t)(i % shuffle_patterns.size())]);
		}
		shuffle_patterns = std::move(cycled);
	}

	// EnsembleGenerator.rng_. A stream independent of FeatureShuffle's, which
	// owns its own; in the shipped presets nothing is drawn before the
	// class-shift draw (no crosses / SVD / subsample).
	TabFMRandom rng(spec.seed);

	// 2. Class shift offsets.
	vector<int64_t> shift_offsets((idx_t)E, 0);
	if (spec.classification && E > 1 && spec.n_classes > 1) {
		vector<int64_t> base = rng.SampleRange(spec.n_classes, spec.n_classes);
		for (int64_t i = 0; i < E; i++) {
			shift_offsets[(idx_t)i] = base[(idx_t)(i % base.size())];
		}
	}

	// 3./4. cat permutations and row subsample: disabled -> no RNG draws.

	// 5. Combine and shuffle the config tuples.
	vector<RawConfig> configs((idx_t)E);
	for (int64_t i = 0; i < E; i++) {
		configs[(idx_t)i].feature_permutation = shuffle_patterns[(idx_t)i];
		configs[(idx_t)i].class_shift_offset = shift_offsets[(idx_t)i];
	}
	rng.Shuffle(configs);

	// 6. Assign normalization methods (cycled), then group by method preserving
	// order — prepare_ensemble_tensors emits all members of norm_methods[0]
	// first, then norm_methods[1], etc.
	vector<string> norm_for_est((idx_t)E);
	for (int64_t i = 0; i < E; i++) {
		norm_for_est[(idx_t)i] = norm_methods[(idx_t)(i % norm_methods.size())];
	}

	// Categorical membership test over the filtered feature indices.
	auto is_cat = [&](int64_t feat) {
		for (auto c : spec.cat_feature_indices) {
			if (c == feat) {
				return true;
			}
		}
		return false;
	};

	vector<EnsembleMember> members;
	for (auto &nm : norm_methods) {
		for (int64_t i = 0; i < E; i++) {
			if (norm_for_est[(idx_t)i] != nm) {
				continue;
			}
			EnsembleMember m;
			m.norm_method = nm;
			m.feature_permutation = configs[(idx_t)i].feature_permutation;
			m.class_shift_offset = configs[(idx_t)i].class_shift_offset;
			m.d = (int64_t)m.feature_permutation.size();
			m.cat_mask.assign(m.feature_permutation.size(), false);
			for (idx_t j = 0; j < m.feature_permutation.size(); j++) {
				m.cat_mask[j] = is_cat(m.feature_permutation[j]);
			}
			members.push_back(std::move(m));
		}
	}
	return members;
}

//===----------------------------------------------------------------------===//
// Apply / inverse transforms
//===----------------------------------------------------------------------===//

vector<double> ApplyFeaturePermutation(const vector<double> &x, idx_t rows,
                                       idx_t cols,
                                       const vector<int64_t> &permutation) {
	const idx_t out_cols = permutation.size();
	vector<double> out(rows * out_cols);
	for (idx_t r = 0; r < rows; r++) {
		for (idx_t j = 0; j < out_cols; j++) {
			out[r * out_cols + j] = x[r * cols + (idx_t)permutation[j]];
		}
	}
	return out;
}

vector<int64_t> ShiftClassContext(const vector<int64_t> &y, int64_t offset,
                                  int64_t n_classes) {
	vector<int64_t> out(y.size());
	for (idx_t i = 0; i < y.size(); i++) {
		out[i] = (y[i] + offset) % n_classes;
	}
	return out;
}

vector<double> UnshiftLogits(const vector<double> &logits, idx_t rows, idx_t k,
                             int64_t offset) {
	vector<double> out(rows * k);
	int64_t off = ((offset % (int64_t)k) + (int64_t)k) % (int64_t)k;
	for (idx_t r = 0; r < rows; r++) {
		for (idx_t j = 0; j < k; j++) {
			out[r * k + j] = logits[r * k + (idx_t)(((int64_t)j + off) % (int64_t)k)];
		}
	}
	return out;
}

vector<double> SoftmaxTemperature(const vector<double> &logits, idx_t rows,
                                  idx_t k, double temperature) {
	vector<double> out(rows * k);
	for (idx_t r = 0; r < rows; r++) {
		double maxv = -std::numeric_limits<double>::infinity();
		for (idx_t j = 0; j < k; j++) {
			double v = logits[r * k + j] / temperature;
			if (v > maxv) {
				maxv = v;
			}
		}
		double sum = 0.0;
		for (idx_t j = 0; j < k; j++) {
			double e = std::exp(logits[r * k + j] / temperature - maxv);
			out[r * k + j] = e;
			sum += e;
		}
		for (idx_t j = 0; j < k; j++) {
			out[r * k + j] /= sum;
		}
	}
	return out;
}

//===----------------------------------------------------------------------===//
// NNLS (Lawson–Hanson active-set)
//===----------------------------------------------------------------------===//

vector<double> SolveNNLS(const vector<double> &a, idx_t m, idx_t n,
                         const vector<double> &b, double *out_rnorm) {
	// Solve min_{x>=0} ||A x - b||. Classic Lawson & Hanson algorithm operating
	// on the normal equations via A^T A / A^T b, matching scipy.optimize.nnls
	// (which is a wrapper around the same method).
	vector<double> x(n, 0.0);
	vector<int> P(n, 0); // 1 == in passive (free) set
	// Precompute A^T A (n x n) and A^T b (n).
	vector<double> AtA(n * n, 0.0);
	vector<double> Atb(n, 0.0);
	for (idx_t i = 0; i < n; i++) {
		for (idx_t j = 0; j < n; j++) {
			double s = 0.0;
			for (idx_t r = 0; r < m; r++) {
				s += a[r * n + i] * a[r * n + j];
			}
			AtA[i * n + j] = s;
		}
		double sb = 0.0;
		for (idx_t r = 0; r < m; r++) {
			sb += a[r * n + i] * b[r];
		}
		Atb[i] = sb;
	}

	const double tol = 1e-10;
	const idx_t max_iter = 3 * n + 10;
	idx_t outer_iter = 0;

	auto gradient = [&](idx_t i) {
		// w = A^T b - A^T A x
		double g = Atb[i];
		for (idx_t j = 0; j < n; j++) {
			g -= AtA[i * n + j] * x[j];
		}
		return g;
	};

	while (true) {
		if (outer_iter++ > max_iter) {
			break;
		}
		// Find index in the active set with the largest positive gradient.
		double wmax = tol;
		idx_t t = n;
		for (idx_t i = 0; i < n; i++) {
			if (P[i]) {
				continue;
			}
			double g = gradient(i);
			if (g > wmax) {
				wmax = g;
				t = i;
			}
		}
		if (t == n) {
			break; // optimality reached
		}
		P[t] = 1;

		// Inner loop: solve the least-squares problem on the passive set.
		while (true) {
			// Gather passive indices.
			vector<idx_t> idxs;
			for (idx_t i = 0; i < n; i++) {
				if (P[i]) {
					idxs.push_back(i);
				}
			}
			const idx_t p = idxs.size();
			// Solve (AtA_PP) s_P = Atb_P via Gaussian elimination.
			vector<double> M(p * p, 0.0);
			vector<double> rhs(p, 0.0);
			for (idx_t a1 = 0; a1 < p; a1++) {
				for (idx_t a2 = 0; a2 < p; a2++) {
					M[a1 * p + a2] = AtA[idxs[a1] * n + idxs[a2]];
				}
				rhs[a1] = Atb[idxs[a1]];
			}
			vector<double> s_p(p, 0.0);
			// Gaussian elimination with partial pivoting.
			bool singular = false;
			for (idx_t col = 0; col < p; col++) {
				idx_t piv = col;
				double best = std::fabs(M[col * p + col]);
				for (idx_t r = col + 1; r < p; r++) {
					double val = std::fabs(M[r * p + col]);
					if (val > best) {
						best = val;
						piv = r;
					}
				}
				if (best < 1e-300) {
					singular = true;
					break;
				}
				if (piv != col) {
					for (idx_t c = 0; c < p; c++) {
						std::swap(M[col * p + c], M[piv * p + c]);
					}
					std::swap(rhs[col], rhs[piv]);
				}
				double diag = M[col * p + col];
				for (idx_t r = 0; r < p; r++) {
					if (r == col) {
						continue;
					}
					double f = M[r * p + col] / diag;
					if (f == 0.0) {
						continue;
					}
					for (idx_t c = col; c < p; c++) {
						M[r * p + c] -= f * M[col * p + c];
					}
					rhs[r] -= f * rhs[col];
				}
			}
			if (singular) {
				// Degenerate; drop the just-added coordinate and stop inner loop.
				for (idx_t i = 0; i < p; i++) {
					s_p[i] = 0.0;
				}
			} else {
				for (idx_t i = 0; i < p; i++) {
					s_p[i] = rhs[i] / M[i * p + i];
				}
			}

			// If all passive coordinates positive, accept.
			bool all_pos = true;
			for (idx_t i = 0; i < p; i++) {
				if (s_p[i] <= tol) {
					all_pos = false;
					break;
				}
			}
			if (all_pos) {
				for (idx_t i = 0; i < n; i++) {
					x[i] = 0.0;
				}
				for (idx_t i = 0; i < p; i++) {
					x[idxs[i]] = s_p[i];
				}
				break;
			}

			// Otherwise move x toward s by the maximal feasible step alpha.
			double alpha = std::numeric_limits<double>::infinity();
			for (idx_t i = 0; i < p; i++) {
				if (s_p[i] <= tol) {
					double denom = x[idxs[i]] - s_p[i];
					if (denom > 0) {
						double ratio = x[idxs[i]] / denom;
						if (ratio < alpha) {
							alpha = ratio;
						}
					}
				}
			}
			if (!std::isfinite(alpha)) {
				alpha = 0.0;
			}
			for (idx_t i = 0; i < p; i++) {
				x[idxs[i]] += alpha * (s_p[i] - x[idxs[i]]);
			}
			// Remove coordinates that hit zero from the passive set.
			for (idx_t i = 0; i < n; i++) {
				if (P[i] && x[i] <= tol) {
					x[i] = 0.0;
					P[i] = 0;
				}
			}
		}
	}

	if (out_rnorm) {
		double rn = 0.0;
		for (idx_t r = 0; r < m; r++) {
			double ax = 0.0;
			for (idx_t j = 0; j < n; j++) {
				ax += a[r * n + j] * x[j];
			}
			double d = ax - b[r];
			rn += d * d;
		}
		*out_rnorm = std::sqrt(rn);
	}
	return x;
}

vector<double> NnlsBlendWeights(const vector<double> &weights_raw,
                                double nnls_beta) {
	const idx_t E = weights_raw.size();
	double sum = 0.0;
	for (auto w : weights_raw) {
		sum += w;
	}
	vector<double> w_norm(E);
	if (sum > 0.0) {
		for (idx_t i = 0; i < E; i++) {
			w_norm[i] = weights_raw[i] / sum;
		}
	} else {
		for (idx_t i = 0; i < E; i++) {
			w_norm[i] = 1.0 / (double)E;
		}
	}
	vector<double> final_w(E);
	double uniform = 1.0 / (double)E;
	for (idx_t i = 0; i < E; i++) {
		final_w[i] = nnls_beta * w_norm[i] + (1.0 - nnls_beta) * uniform;
	}
	return final_w;
}

vector<double> BlendClassification(const vector<double> &probs_all, idx_t e,
                                   idx_t n, idx_t k,
                                   const vector<double> &weights) {
	vector<double> out(n * k, 0.0);
	for (idx_t m = 0; m < e; m++) {
		double w = weights[m];
		const double *pm = probs_all.data() + m * n * k;
		for (idx_t i = 0; i < n * k; i++) {
			out[i] += w * pm[i];
		}
	}
	return out;
}

vector<double> BlendRegression(const vector<double> &member_predictions,
                               idx_t e, idx_t n, const vector<double> &weights) {
	vector<double> out(n, 0.0);
	for (idx_t m = 0; m < e; m++) {
		double w = weights[m];
		const double *pm = member_predictions.data() + m * n;
		for (idx_t i = 0; i < n; i++) {
			out[i] += w * pm[i];
		}
	}
	return out;
}

} // namespace anofox
} // namespace duckdb
