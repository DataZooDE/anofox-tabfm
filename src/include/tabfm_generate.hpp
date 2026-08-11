//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_generate.hpp — WS-G: synthetic data generation and imputation.
//
// Both surfaces sit ABOVE the PredictEngine seam (tabfm_predict.hpp): they call
// Predict() repeatedly with a narrowed row schema and never touch ONNX, tensor
// contracts or preprocessing. That is why they work with every model in the
// registry, not just one family.
//
//   tabfm_generate(data, n, …)  — sample n new rows from the joint distribution
//   tabfm_impute (data, cols, …) — fill NULL cells with the conditional best
//                                  estimate
//
// THE CHAIN RULE
// --------------
// A table is a joint density, factorized column-wise:
//
//   p(c_1..c_H) = p(c_π1) · p(c_π2 | c_π1) · … · p(c_πH | c_π1..π(H-1))
//
// For a column permutation π, step i fits on the REAL rows using the already
// generated columns c_π(1..i-1) as features and c_π(i) as target, predicts on
// the partially filled synthetic rows, and SAMPLES from the predictive
// distribution. Step 1 has no features and comes from the empirical marginal.
// This is the technique in docs.priorlabs.ai/cookbook/generate_synthetic_data,
// generalized off TabPFN onto the whole registry.
//
// CONTINUOUS COLUMNS: WHY BINNING
// -------------------------------
// Sampling needs a predictive DISTRIBUTION. Our classification path has one
// (the proba MAP), but the regression path does not: the ONNX exporters reduce
// the bar distribution to its mean inside the graph
// (resources/export_report_tabpfn_regression.json — "logits: f32[1,T,1] …
// bar-distribution mean, de-standardized"), and tabfm_engine.cpp reads that one
// scalar. So a continuous column is binned into K <= max_classes quantile bins,
// the step runs as CLASSIFICATION over bin labels, a bin is sampled from proba
// at temperature, and a value is drawn from that bin's EMPIRICAL distribution
// (see ExpandBin — uniform across the bin's span is subtly but badly wrong).
//
// Consequences, stated plainly because they are user-visible:
//   + a genuine multi-modal conditional density, which a point estimate cannot
//     give at all;
//   + works on classify-only models (Orion-BiX), so EVERY registry model can
//     generate;
//   + a draw is interpolated between two observations, so it emits novel values
//     rather than copying real ones;
//   - resolution is capped at K bins, and nothing is ever drawn outside the
//     observed [min, max] (no tail extrapolation).
// Exact bar-distribution sampling needs a graph re-export exposing bucket
// logits + borders — a separate, per-model-family piece of work.
//
// IMPUTATION NEEDS NONE OF THAT. It wants the conditional mode/mean, not a
// sample, so it uses the classification argmax and the regression point
// estimate directly, at full precision. It is both simpler and higher fidelity
// than generation; that is the whole semantic split between the two functions.
//===----------------------------------------------------------------------===//

#pragma once

#include "tabfm_predict.hpp"
#include "tabfm_random.hpp"

#include "duckdb/common/types/value.hpp"

namespace duckdb {
namespace anofox {

//! Engine-facing limit: the predict path refuses a target with more than this
//! many distinct labels (tabfm_engine.cpp), and every shipped model declares
//! size_regime.max_classes = 10.
static constexpr idx_t kMaxGenerateBins = 10;
//! Default quantile bins per continuous column.
static constexpr idx_t kDefaultGenerateBins = 10;

//===----------------------------------------------------------------------===//
// Column planning
//===----------------------------------------------------------------------===//

enum class GenColumnKind : uint8_t {
	//! Modelled as classification over the observed labels.
	CATEGORICAL,
	//! Modelled as classification over quantile bins, expanded back to a value.
	CONTINUOUS,
	//! One observed value (or none at all): emitted as-is, no model call.
	CONSTANT,
	//! Cannot be a generation target (temporal, or too many distinct labels).
	UNSUPPORTED
};

//! What one input column is, and everything needed to sample it.
struct ColumnPlan {
	string name;
	LogicalType type;
	GenColumnKind kind = GenColumnKind::CONSTANT;

	//! CATEGORICAL: the observed label domain, in first-appearance order.
	vector<Value> labels;
	//! CATEGORICAL: observed occurrences per label (drives the marginal draw).
	vector<idx_t> label_counts;

	//! CONTINUOUS: quantile bin edges, size == bin_count + 1, strictly
	//! increasing. edges.front()/back() are the observed min/max.
	vector<double> edges;
	//! CONTINUOUS: observed rows per bin (drives the marginal draw).
	vector<idx_t> bin_counts;
	//! CONTINUOUS: every observed value, ascending — the empirical distribution
	//! a sampled bin is expanded through.
	vector<double> sorted_values;
	//! CONTINUOUS: where each bin starts in `sorted_values`; size bin_count + 1,
	//! so bin b owns [bin_offsets[b], bin_offsets[b + 1]).
	vector<idx_t> bin_offsets;

	//! CONSTANT: the value every row gets.
	Value constant_value;

	//! UNSUPPORTED: why, for the error message.
	string unsupported_reason;

	idx_t null_count = 0;
	idx_t observed_count = 0;

	idx_t BinCount() const {
		return edges.empty() ? 0 : edges.size() - 1;
	}
	//! How many labels the engine will see for this column as a target.
	idx_t ClassCount() const {
		return kind == GenColumnKind::CONTINUOUS ? BinCount() : labels.size();
	}
};

//! Classify `values` and build the sampling plan. `bins` is the requested
//! quantile-bin count for continuous columns (clamped to the number of distinct
//! values); `max_labels` is how many classes the selected model's output head
//! can actually represent. Columns whose type cannot be a target come back
//! UNSUPPORTED with a populated `unsupported_reason` rather than throwing — the
//! caller decides whether that column is actually needed.
ColumnPlan PlanColumn(const string &name, const LogicalType &type, const vector<Value> &values, idx_t bins,
                      idx_t max_labels = kMaxGenerateBins);

//===----------------------------------------------------------------------===//
// Sampling primitives (pure; unit-tested without an engine)
//===----------------------------------------------------------------------===//

//! Quantile bin edges over `values` (need not be sorted). Returns at most
//! bins+1 strictly increasing edges; ties collapse, so a column with 3 distinct
//! values yields 3 bins however many were requested. Empty when `values` has
//! fewer than 2 distinct entries (that column is CONSTANT).
vector<double> QuantileEdges(vector<double> values, idx_t bins);

//! Index of the bin containing `value` (edges half-open except the last, which
//! is closed so the maximum lands in the top bin). Clamped into range.
idx_t BinOf(double value, const vector<double> &edges);

//! Turn a sampled bin index into a column value, and coerce it to the column's
//! type (integers round, decimals cast). The result never leaves the bin, hence
//! never leaves the observed range of the column.
//!
//! Inside the bin the draw follows the EMPIRICAL distribution: pick a uniform
//! position among the bin's observed values and interpolate between neighbouring
//! order statistics. Sampling uniformly across the bin's span instead — the
//! obvious implementation — is badly wrong in the outer bins of a skewed column,
//! where the span is wide but the real mass hugs one end: on the breast-cancer
//! benchmark that inflated `area_se`'s mean by 67% and visibly bent the top 15%
//! of its CDF. Interpolating between observed neighbours still produces novel
//! values (it lands strictly between two real ones), so nothing is copied
//! verbatim unless the neighbours are tied.
Value ExpandBin(const ColumnPlan &plan, idx_t bin, TabFMRandom &rng);

//! Draw one value from the column's empirical marginal — used for the first
//! column of the chain, which has no features to condition on.
Value DrawMarginal(const ColumnPlan &plan, TabFMRandom &rng);

//! The target Value handed to the engine for one GENERATION step. Always
//! VARCHAR, so the step is unambiguously a classification problem whatever the
//! column's real type: the label's string form for CATEGORICAL, the bin index
//! for CONTINUOUS, and a VARCHAR NULL for a NULL input (the engine's "score me"
//! marker). Sampling needs the proba MAP, whose keys are exactly these strings.
//!
//! Imputation does NOT use this — it passes raw values through and lets the
//! task follow the column kind, so continuous fills keep full precision.
Value EncodeTarget(const ColumnPlan &plan, const Value &row_value);

//! The key `EncodeTarget` produces for label/bin index `i`.
string TargetKey(const ColumnPlan &plan, idx_t index);

//! Turn one engine result back into a column value.
//! `proba` is the engine's MAP(VARCHAR, DOUBLE) for the row; when it is NULL or
//! empty the caller's `fallback` (the argmax yhat) is used instead of sampling.
Value SampleFromProba(const ColumnPlan &plan, const Value &proba, const Value &fallback, TabFMRandom &rng);

//! Decode an engine `yhat` for `plan` WITHOUT sampling — the imputation path:
//! CATEGORICAL yields the predicted label, CONTINUOUS the point estimate
//! coerced to the column type and clamped to the observed range.
Value DecodePointEstimate(const ColumnPlan &plan, const Value &yhat);

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

enum class GenColumnOrder : uint8_t { RANDOM, NATURAL, MISSINGNESS };

struct TabFMGenerateOptions {
	int64_t seed = 42;
	double temperature = 1.0;
	idx_t bins = kDefaultGenerateBins;
	GenColumnOrder column_order = GenColumnOrder::RANDOM;
	//! Imputation only: MICE-style refinement sweeps.
	idx_t rounds = 1;
	string model;
	//! How many classes the selected model's output head can represent, from its
	//! `size_regime.max_classes` (kMaxGenerateBins when the model declares none).
	//! Every step of generation is a classification problem, so this caps BOTH
	//! the quantile-bin count and the categorical cardinality we can model —
	//! feeding more labels than the head is wide fails inside the engine with a
	//! shape error that says nothing about `bins`.
	idx_t max_classes = kMaxGenerateBins;
};

//! NOTE: no `context_rows` here, unlike TabFMPredictOptions. That key is parsed
//! and validated by the predict surface but never read by the engine
//! (tabfm_preprocess.cpp / tabfm_engine.cpp ignore it), so it is a no-op today.
//! Mirroring it would ship a documented knob that does nothing.

//===----------------------------------------------------------------------===//
// Drivers (call the PredictEngine seam)
//===----------------------------------------------------------------------===//

//! Sample `n` rows from the joint distribution of `real_rows` by the chain rule.
//! Returns n rows in `fields` order. Makes (modellable columns - 1) engine
//! calls, strictly sequentially: step i conditions on what step i-1 produced.
vector<vector<Value>> RunChainRule(const vector<vector<Value>> &real_rows, const child_list_t<LogicalType> &fields,
                                   const vector<ColumnPlan> &plans, idx_t n, const TabFMGenerateOptions &opts,
                                   const PredictContext &ctx);

//! Fill the NULL cells of the `targets` columns with the conditional best
//! estimate, leaving every non-NULL cell untouched. One engine call per target
//! column per round; `opts.rounds > 1` re-runs the sweep so later columns'
//! fills inform earlier ones (MICE-style).
vector<vector<Value>> RunImpute(const vector<vector<Value>> &input_rows, const child_list_t<LogicalType> &fields,
                                const vector<ColumnPlan> &plans, const vector<idx_t> &targets,
                                const TabFMGenerateOptions &opts, const PredictContext &ctx);

} // namespace anofox
} // namespace duckdb
