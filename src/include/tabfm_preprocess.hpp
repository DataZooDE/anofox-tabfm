//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_preprocess.hpp — C++ port of the TabFM "minimal" preprocessing
// pipeline (HLD §4.5, BRD FR-3.3/3.5/3.6), driven by DuckDB types.
//
// Upstream reference (never edited):
//   vendor/tabfm/tabfm/src/classifier_and_regressor.py
//
// This module reproduces, bit-for-bit against test/fixtures/golden_preprocess.json
// (rtol 1e-6), the encode → filter → scale → outlier-clip chain that the
// Python TabFMClassifier / TabFMRegressor run before the model forward pass,
// for the single-member (n_estimators=1, norm='none', identity permutation)
// configuration. The ensemble-diversity layer on top of this lives in
// tabfm_ensemble.hpp.
//
// C++ symbol → upstream Python symbol map
// ---------------------------------------
//   FeatureKind / column classification  → TransformToNumerical.fit column
//                                           typing (categorical / continuous /
//                                           datetime ColumnTransformer)
//   ColumnEncoder (CATEGORICAL)          → CategoricalOrdinalEncoder
//                                           (mode="appearance", min_frequency=2,
//                                            unknown_value=-1)
//   ColumnEncoder (NUMERIC)              → sklearn SimpleImputer (strategy=mean)
//   ColumnEncoder (DATETIME)             → DatetimeTransformer (epoch_ns +
//                                           year/month/day/dayofweek, NaT filled
//                                           with the TRAIN mean timestamp)
//   PreprocessStages.unique_filter_keep  → UniqueFeatureFilter
//   PreprocessStages.scaler_*            → CustomStandardScaler (std ddof=0 +1e-6,
//                                           clip [-100,100])
//   PreprocessStages.outlier_*           → OutlierRemover (2-stage 4-sigma,
//                                           log1p clipping)
//   PreprocessedBatch (x/y/cat_mask/d/   → EnsembleGenerator(n_estimators=1,
//     train_size)                          norm_methods=['none']) member 0 +
//                                           prepare_ensemble_tensors, y padded
//                                           to T with -100.0
//   label_decoder                        → CategoricalOrdinalEncoder
//                                           (mode="alphabetical") inverse, i.e.
//                                           sklearn LabelEncoder classes_
//   target_mean/target_scale             → sklearn StandardScaler on the target
//                                           (regression), inverse_transform
//
// Pinned upstream semantics (see .cpp for details):
//  * NaN / padding sentinel for the y target of non-train (to-score) rows is
//    -100.0 (upstream `_batch_forward` np.pad constant_values=-100.0).
//  * First-appearance scope: categorical categories are learned over TRAIN rows
//    only (rows whose target is non-NULL); rare (<min_frequency), NULL and
//    unseen values all encode to -1.
//  * Datetime expansion: one input column -> 5 encoded features
//    [epoch_ns(as float64), year, month, day, dayofweek(Mon=0..Sun=6)].
//  * Encoded column order: ALL categorical features first (input order), then
//    numeric (input order), then per datetime column its 5 fields — matching
//    sklearn ColumnTransformer output order.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/string.hpp"

namespace duckdb {
namespace anofox {

//! Preprocessing profile id chosen by the model manifest (HLD D7 / §4.5). This
//! module implements exactly one profile.
static constexpr const char *kPreprocessProfileId = "tabfm_v1_minimal";

//! Sentinel written into the y tensor for rows without a train label (padding
//! and to-score rows). Upstream `_batch_forward` pads y with this constant.
static constexpr double kTargetPadSentinel = -100.0;

//! min_frequency for CategoricalOrdinalEncoder (TransformToNumerical default).
static constexpr int kMinCatFrequency = 2;

//! Prediction task; upstream picks this from the target field type (FR-3.2).
enum class PreprocessTask : uint8_t { CLASSIFICATION, REGRESSION };

//! How an input feature column is routed, mirroring TransformToNumerical.fit.
enum class FeatureKind : uint8_t { NUMERIC, CATEGORICAL, DATETIME };

//! Caller-supplied metadata for one column of the input ColumnDataCollection.
//! Columns appear in the same order as the collection's columns.
struct PreprocessColumnSpec {
	string name;
	LogicalType type = LogicalType::SQLNULL;
	bool is_target = false;
	bool is_feature = false;
};

//! Per-input-column fitted encoder, kept for decoding / introspection.
struct ColumnEncoder {
	string name;
	FeatureKind kind = FeatureKind::NUMERIC;
	//! NUMERIC: train mean used for NULL imputation.
	double impute_mean = 0.0;
	//! CATEGORICAL: kept categories in first-appearance order (code == index).
	vector<string> categories;
	//! DATETIME: TRAIN mean epoch (nanoseconds, as float64) used to fill NaT.
	double fill_epoch_ns = 0.0;
	//! Number of encoded output columns this input produces (1, or 5 datetime).
	idx_t encoded_width = 1;
};

//! Every intermediate stage of the pipeline, exposed for golden-vector parity.
struct PreprocessStages {
	idx_t n_train = 0;
	idx_t n_test = 0;
	//! Encoded (pre-unique-filter) column names, in ColumnTransformer order.
	vector<string> encoded_column_names;
	idx_t n_encoded = 0;
	//! Encoded matrices (row-major), before the unique-feature filter.
	vector<double> encoded_train; // n_train x n_encoded
	vector<double> encoded_test;  // n_test  x n_encoded
	//! Unique-feature-filter keep mask over the encoded columns.
	vector<bool> unique_filter_keep;
	//! Categorical column indices AFTER the unique filter (remapped).
	vector<int64_t> cat_feature_indices_kept;
	//! CustomStandardScaler statistics over the filtered TRAIN rows (width H).
	vector<double> scaler_mean;
	vector<double> scaler_scale; // std(ddof=0) + epsilon
	//! OutlierRemover second-stage statistics / bounds (width H).
	vector<double> outlier_means;
	vector<double> outlier_stds;
	vector<double> outlier_lower;
	vector<double> outlier_upper;
};

//! Fully preprocessed batch ready for the ORT engine (up to the float32 cast,
//! which happens at the model feed — tensors here are float64 pipeline outputs).
struct PreprocessedBatch {
	PreprocessTask task = PreprocessTask::CLASSIFICATION;
	idx_t T = 0;          //!< total rows (train first, then to-score rows)
	idx_t H = 0;          //!< feature width after the unique filter
	idx_t d = 0;          //!< active (non-padded) feature count (== H here)
	idx_t train_size = 0; //!< in-context train row count

	//! Row-major [T, H] feature matrix (train rows first, then to-score rows).
	vector<double> x;
	//! Length-T target vector: train labels/scaled targets then kTargetPadSentinel.
	vector<double> y;
	//! Length-train_size integer class codes (classification only).
	vector<int64_t> y_train;
	//! Length-H categorical mask (true == categorical feature).
	vector<bool> cat_mask;

	//! CLASSIFICATION: class code -> original label Value (alphabetical order).
	vector<Value> label_decoder;
	//! REGRESSION: target inverse transform is pred * target_scale + target_mean.
	double target_mean = 0.0;
	double target_scale = 1.0;

	//! Per-input-column encoders (feature columns only), in input order.
	vector<ColumnEncoder> encoders;
	//! Maps each output row (train then test) back to its source collection row.
	vector<idx_t> row_source_index;

	PreprocessStages stages;
};

//! Infer the task from the target column type: numeric -> regression, else
//! classification (upstream FR-3.2 semantics).
PreprocessTask InferTask(const LogicalType &target_type);

//! Run the minimal preprocessing pipeline over `data`, whose columns are
//! described by `columns` (same order). Exactly one column must have
//! is_target=true; feature columns are those with is_feature=true. Rows whose
//! target is NULL are the to-score (test) rows; the rest are the in-context
//! train rows. Throws InvalidInputException on malformed input.
//!
//! `standardize` (default true) runs the CustomStandardScaler (z-score, fit on
//! train) + OutlierRemover. Set false for models that normalize features INSIDE
//! their graph (TabPFN/TabICL, a "*_raw" preprocessing profile): features are
//! then ordinal-encoded + NULL/NaN-imputed but passed through un-standardized,
//! and the scaler/outlier stages are identity.
PreprocessedBatch PreprocessBatch(const ColumnDataCollection &data,
                                  const vector<PreprocessColumnSpec> &columns,
                                  PreprocessTask task, bool standardize = true);

} // namespace anofox
} // namespace duckdb
