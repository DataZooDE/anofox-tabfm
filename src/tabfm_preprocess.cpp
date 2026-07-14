//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_preprocess.cpp — implementation of the minimal TabFM preprocessing
// pipeline. See tabfm_preprocess.hpp for the C++ -> upstream Python symbol map
// and the pinned upstream semantics. Reference (never edited):
//   vendor/tabfm/tabfm/src/classifier_and_regressor.py
//
// The chain, per fixture golden_preprocess.json/_docs:
//   TransformToNumerical (categorical ordinal / mean-impute / datetime expand)
//     -> UniqueFeatureFilter (drop <=1 unique on train)
//     -> CustomStandardScaler (z-score, std ddof=0 + 1e-6, clip [-100,100])
//     -> OutlierRemover (two-stage 4-sigma, log1p clipping)
// All statistics are fit on TRAIN rows (rows with a non-NULL target); the
// transform is then applied to [train; test] and the target padded to T with
// -100.0. This equals EnsembleGenerator(n_estimators=1, norm=['none']) member 0
// with the identity feature permutation.
//===----------------------------------------------------------------------===//

#include "tabfm_preprocess.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/timestamp.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace duckdb {
namespace anofox {

static constexpr int64_t kNsPerDay = 86400LL * 1000000000LL;

//===----------------------------------------------------------------------===//
// Column-type classification (TransformToNumerical.fit)
//===----------------------------------------------------------------------===//

static FeatureKind ClassifyColumn(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
		return FeatureKind::DATETIME;
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::ENUM:
	case LogicalTypeId::BOOLEAN:
		return FeatureKind::CATEGORICAL;
	default:
		if (type.IsNumeric()) {
			return FeatureKind::NUMERIC;
		}
		// Anything else (e.g. BLOB) is treated as categorical, matching the
		// upstream "fallback to categorical if unknown" branch.
		return FeatureKind::CATEGORICAL;
	}
}

PreprocessTask InferTask(const LogicalType &target_type) {
	// Upstream: a numeric target -> regression, otherwise classification.
	switch (target_type.id()) {
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::ENUM:
	case LogicalTypeId::BOOLEAN:
		return PreprocessTask::CLASSIFICATION;
	default:
		return target_type.IsNumeric() ? PreprocessTask::REGRESSION
		                               : PreprocessTask::CLASSIFICATION;
	}
}

//===----------------------------------------------------------------------===//
// Datetime helpers
//===----------------------------------------------------------------------===//

//! Extract the epoch-nanosecond value (as float64) and the UTC day index of a
//! non-NULL datetime Value.
static int64_t DatetimeDayIndex(const Value &v, const LogicalType &type,
                                double &epoch_ns_out) {
	if (type.id() == LogicalTypeId::DATE) {
		date_t d = DateValue::Get(v);
		int64_t day = d.days;
		// double, not int64: day * kNsPerDay overflows int64 for far-future dates
		// (signed overflow is UB). The day index itself needs no scaling.
		epoch_ns_out = static_cast<double>(day) * static_cast<double>(kNsPerDay);
		return day;
	}
	// Read the timestamp in its NATIVE unit (the physical int64) instead of
	// downcasting every flavour to microseconds — that downcast silently dropped
	// sub-microsecond resolution for TIMESTAMP_NS. The day index is an exact
	// integer floor-division in the native unit (no overflow); the epoch value is
	// scaled to nanoseconds in double (int64 ns overflows past ~year 2262).
	int64_t raw = v.GetValueUnsafe<int64_t>();
	int64_t units_per_day;
	double ns_per_unit;
	switch (type.id()) {
	case LogicalTypeId::TIMESTAMP_SEC:
		units_per_day = 86400LL;
		ns_per_unit = 1e9;
		break;
	case LogicalTypeId::TIMESTAMP_MS:
		units_per_day = 86400LL * 1000LL;
		ns_per_unit = 1e6;
		break;
	case LogicalTypeId::TIMESTAMP_NS:
		units_per_day = 86400LL * 1000000000LL;
		ns_per_unit = 1.0;
		break;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
	default:
		units_per_day = 86400LL * 1000000LL; // micros
		ns_per_unit = 1e3;
		break;
	}
	epoch_ns_out = static_cast<double>(raw) * ns_per_unit;
	// Floor division for the UTC calendar day.
	int64_t day = raw / units_per_day;
	if (raw < 0 && raw % units_per_day != 0) {
		day -= 1;
	}
	return day;
}

//! Fill the 5 derived datetime features [epoch_ns, year, month, day, dow] for a
//! given day index and epoch value into `out` (dow uses pandas Monday=0..Sun=6).
static void DatetimeFeatures(int64_t day_index, double epoch_ns, double *out) {
	date_t d(static_cast<int32_t>(day_index));
	int32_t year, month, day;
	Date::Convert(d, year, month, day);
	int32_t iso_dow = Date::ExtractISODayOfTheWeek(d); // 1=Mon..7=Sun
	out[0] = epoch_ns;
	out[1] = static_cast<double>(year);
	out[2] = static_cast<double>(month);
	out[3] = static_cast<double>(day);
	out[4] = static_cast<double>(iso_dow - 1);
}

//===----------------------------------------------------------------------===//
// PreprocessBatch
//===----------------------------------------------------------------------===//

namespace {

struct FeatureCol {
	idx_t source_col;
	string name;
	LogicalType type;
	FeatureKind kind;
};

} // namespace

PreprocessedBatch PreprocessBatch(const ColumnDataCollection &data,
                                  const vector<PreprocessColumnSpec> &columns,
                                  PreprocessTask task, bool standardize) {
	if (data.ColumnCount() != columns.size()) {
		throw InvalidInputException(
		    "tabfm preprocess: column metadata count (%llu) does not match the "
		    "collection column count (%llu)",
		    (unsigned long long)columns.size(),
		    (unsigned long long)data.ColumnCount());
	}

	// Locate the (single) target column and the feature columns, preserving
	// input order.
	idx_t target_col = DConstants::INVALID_INDEX;
	vector<FeatureCol> cat_cols, num_cols, date_cols;
	for (idx_t c = 0; c < columns.size(); c++) {
		if (columns[c].is_target) {
			if (target_col != DConstants::INVALID_INDEX) {
				throw InvalidInputException(
				    "tabfm preprocess: more than one target column specified");
			}
			target_col = c;
		}
		if (columns[c].is_feature) {
			FeatureCol fc{c, columns[c].name, columns[c].type,
			              ClassifyColumn(columns[c].type)};
			switch (fc.kind) {
			case FeatureKind::CATEGORICAL:
				cat_cols.push_back(fc);
				break;
			case FeatureKind::NUMERIC:
				num_cols.push_back(fc);
				break;
			case FeatureKind::DATETIME:
				date_cols.push_back(fc);
				break;
			}
		}
	}
	if (target_col == DConstants::INVALID_INDEX) {
		throw InvalidInputException("tabfm preprocess: no target column specified");
	}

	const idx_t n_rows = data.Count();
	// ColumnDataCollection exposes random-access GetValue only through a
	// materialized row view (GetValue is a member of ColumnDataRowCollection).
	ColumnDataRowCollection rows = data.GetRows();

	// Train rows = non-NULL target (in order), then to-score rows = NULL target.
	vector<idx_t> train_rows, test_rows;
	for (idx_t r = 0; r < n_rows; r++) {
		if (rows.GetValue(target_col, r).IsNull()) {
			test_rows.push_back(r);
		} else {
			train_rows.push_back(r);
		}
	}
	const idx_t n_train = train_rows.size();
	const idx_t n_test = test_rows.size();
	if (n_train == 0) {
		throw InvalidInputException(
		    "tabfm preprocess: no in-context train rows (all target values NULL)");
	}

	vector<idx_t> src(n_train + n_test);
	for (idx_t i = 0; i < n_train; i++) {
		src[i] = train_rows[i];
	}
	for (idx_t i = 0; i < n_test; i++) {
		src[n_train + i] = test_rows[i];
	}

	// Encoded column layout: categorical, then numeric, then datetime(x5).
	const idx_t n_encoded =
	    cat_cols.size() + num_cols.size() + date_cols.size() * 5;

	PreprocessedBatch batch;
	batch.task = task;
	batch.row_source_index = src;

	PreprocessStages &stages = batch.stages;
	stages.n_train = n_train;
	stages.n_test = n_test;
	stages.n_encoded = n_encoded;
	stages.encoded_train.assign(n_train * n_encoded, 0.0);
	stages.encoded_test.assign(n_test * n_encoded, 0.0);

	// Column indices in the encoded matrix that are categorical.
	vector<bool> encoded_is_cat(n_encoded, false);

	idx_t out_col = 0; // running encoded-column cursor

	// --- Categorical columns (CategoricalOrdinalEncoder) ---
	for (auto &fc : cat_cols) {
		// Fit: first-appearance order + counts over TRAIN rows.
		vector<string> appearance;
		std::set<string> seen;
		std::map<string, idx_t> counts;
		for (idx_t i = 0; i < n_train; i++) {
			Value v = rows.GetValue(fc.source_col, train_rows[i]);
			if (v.IsNull()) {
				continue;
			}
			string key = v.ToString();
			if (key == "nan") {
				continue; // upstream drops the literal string "nan"
			}
			counts[key]++;
			if (seen.insert(key).second) {
				appearance.push_back(key);
			}
		}
		ColumnEncoder enc;
		enc.name = fc.name;
		enc.kind = FeatureKind::CATEGORICAL;
		enc.encoded_width = 1;
		std::map<string, int64_t> code_of;
		for (auto &key : appearance) {
			if (counts[key] >= (idx_t)kMinCatFrequency) {
				code_of[key] = (int64_t)enc.categories.size();
				enc.categories.push_back(key);
			}
		}
		batch.encoders.push_back(enc);
		stages.encoded_column_names.push_back(fc.name);
		encoded_is_cat[out_col] = true;

		// Transform.
		auto encode = [&](idx_t source_row) -> double {
			Value v = rows.GetValue(fc.source_col, source_row);
			if (v.IsNull()) {
				return -1.0;
			}
			string key = v.ToString();
			auto it = code_of.find(key);
			return it == code_of.end() ? -1.0 : (double)it->second;
		};
		for (idx_t i = 0; i < n_train; i++) {
			stages.encoded_train[i * n_encoded + out_col] = encode(train_rows[i]);
		}
		for (idx_t i = 0; i < n_test; i++) {
			stages.encoded_test[i * n_encoded + out_col] = encode(test_rows[i]);
		}
		out_col++;
	}

	// --- Numeric columns (SimpleImputer, strategy=mean) ---
	for (auto &fc : num_cols) {
		double sum = 0.0;
		idx_t cnt = 0;
		for (idx_t i = 0; i < n_train; i++) {
			Value v = rows.GetValue(fc.source_col, train_rows[i]);
			// A non-finite value (NaN/Inf) is non-NULL but would poison the mean
			// and, downstream, the z-score statistics for the whole column — treat
			// it as missing, exactly like NULL (SimpleImputer skips it).
			if (!v.IsNull()) {
				double dv = v.GetValue<double>();
				if (std::isfinite(dv)) {
					sum += dv;
					cnt++;
				}
			}
		}
		double mean = cnt > 0 ? sum / (double)cnt : 0.0;
		ColumnEncoder enc;
		enc.name = fc.name;
		enc.kind = FeatureKind::NUMERIC;
		enc.impute_mean = mean;
		enc.encoded_width = 1;
		batch.encoders.push_back(enc);
		stages.encoded_column_names.push_back(fc.name);

		auto encode = [&](idx_t source_row) -> double {
			Value v = rows.GetValue(fc.source_col, source_row);
			if (v.IsNull()) {
				return mean;
			}
			double dv = v.GetValue<double>();
			return std::isfinite(dv) ? dv : mean; // impute NaN/Inf like NULL
		};
		for (idx_t i = 0; i < n_train; i++) {
			stages.encoded_train[i * n_encoded + out_col] = encode(train_rows[i]);
		}
		for (idx_t i = 0; i < n_test; i++) {
			stages.encoded_test[i * n_encoded + out_col] = encode(test_rows[i]);
		}
		out_col++;
	}

	// --- Datetime columns (DatetimeTransformer) ---
	for (auto &fc : date_cols) {
		// Fit: TRAIN mean epoch (float64), for NaT fill.
		double sum = 0.0;
		idx_t cnt = 0;
		for (idx_t i = 0; i < n_train; i++) {
			Value v = rows.GetValue(fc.source_col, train_rows[i]);
			if (v.IsNull()) {
				continue;
			}
			double epoch_ns;
			DatetimeDayIndex(v, fc.type, epoch_ns);
			sum += epoch_ns;
			cnt++;
		}
		double fill_epoch_ns = cnt > 0 ? sum / (double)cnt : 0.0;
		int64_t fill_day = static_cast<int64_t>(fill_epoch_ns) / kNsPerDay;
		if (fill_epoch_ns < 0 && static_cast<int64_t>(fill_epoch_ns) % kNsPerDay != 0) {
			fill_day -= 1;
		}

		ColumnEncoder enc;
		enc.name = fc.name;
		enc.kind = FeatureKind::DATETIME;
		enc.fill_epoch_ns = fill_epoch_ns;
		enc.encoded_width = 5;
		batch.encoders.push_back(enc);
		stages.encoded_column_names.push_back(fc.name + ".epoch_ns");
		stages.encoded_column_names.push_back(fc.name + ".year");
		stages.encoded_column_names.push_back(fc.name + ".month");
		stages.encoded_column_names.push_back(fc.name + ".day");
		stages.encoded_column_names.push_back(fc.name + ".dayofweek");

		auto encode = [&](idx_t source_row, double *out5) {
			Value v = rows.GetValue(fc.source_col, source_row);
			if (v.IsNull()) {
				DatetimeFeatures(fill_day, fill_epoch_ns, out5);
			} else {
				double epoch_ns;
				int64_t day = DatetimeDayIndex(v, fc.type, epoch_ns);
				DatetimeFeatures(day, epoch_ns, out5);
			}
		};
		for (idx_t i = 0; i < n_train; i++) {
			encode(train_rows[i], &stages.encoded_train[i * n_encoded + out_col]);
		}
		for (idx_t i = 0; i < n_test; i++) {
			encode(test_rows[i], &stages.encoded_test[i * n_encoded + out_col]);
		}
		out_col += 5;
	}

	// --- UniqueFeatureFilter (drop columns with <=1 unique value on train) ---
	stages.unique_filter_keep.assign(n_encoded, true);
	if (n_train > 1) { // threshold safety: <= threshold(1) rows keeps all
		for (idx_t col = 0; col < n_encoded; col++) {
			std::set<double> uniq;
			for (idx_t i = 0; i < n_train; i++) {
				uniq.insert(stages.encoded_train[i * n_encoded + col]);
				if (uniq.size() > 1) {
					break;
				}
			}
			stages.unique_filter_keep[col] = uniq.size() > 1;
		}
	}

	// Kept-column index list + remapped categorical indices.
	vector<idx_t> kept_cols;
	for (idx_t col = 0; col < n_encoded; col++) {
		if (stages.unique_filter_keep[col]) {
			if (encoded_is_cat[col]) {
				stages.cat_feature_indices_kept.push_back((int64_t)kept_cols.size());
			}
			kept_cols.push_back(col);
		}
	}
	const idx_t H = kept_cols.size();

	batch.cat_mask.assign(H, false);
	for (idx_t j = 0; j < H; j++) {
		batch.cat_mask[j] = encoded_is_cat[kept_cols[j]];
	}

	// Filtered train / full ([train; test]) matrices, width H.
	const idx_t T = n_train + n_test;
	vector<double> filt_train(n_train * H);
	vector<double> filt_full(T * H);
	for (idx_t i = 0; i < n_train; i++) {
		for (idx_t j = 0; j < H; j++) {
			double val = stages.encoded_train[i * n_encoded + kept_cols[j]];
			filt_train[i * H + j] = val;
			filt_full[i * H + j] = val;
		}
	}
	for (idx_t i = 0; i < n_test; i++) {
		for (idx_t j = 0; j < H; j++) {
			filt_full[(n_train + i) * H + j] =
			    stages.encoded_test[i * n_encoded + kept_cols[j]];
		}
	}

	// --- CustomStandardScaler (fit on filtered train) ---
	// Identity by default; only fit + transform when the model wants standardized
	// features (skipped for "*_raw" profiles — the model normalizes internally).
	stages.scaler_mean.assign(H, 0.0);
	stages.scaler_scale.assign(H, 1.0);
	if (standardize) {
		for (idx_t j = 0; j < H; j++) {
			double sum = 0.0;
			for (idx_t i = 0; i < n_train; i++) {
				sum += filt_train[i * H + j];
			}
			double mean = sum / (double)n_train;
			double var = 0.0;
			for (idx_t i = 0; i < n_train; i++) {
				double dv = filt_train[i * H + j] - mean;
				var += dv * dv;
			}
			var /= (double)n_train; // ddof=0
			stages.scaler_mean[j] = mean;
			stages.scaler_scale[j] = std::sqrt(var) + 1e-6;
		}
		// transform [train; test], clip [-100,100].
		for (idx_t i = 0; i < T; i++) {
			for (idx_t j = 0; j < H; j++) {
				double s = (filt_full[i * H + j] - stages.scaler_mean[j]) /
				           stages.scaler_scale[j];
				if (s < -100.0) {
					s = -100.0;
				} else if (s > 100.0) {
					s = 100.0;
				}
				filt_full[i * H + j] = s;
			}
		}
	}

	// --- OutlierRemover (two-stage, fit on scaled train) ---
	// Identity by default; only active alongside standardization (a "*_raw"
	// profile passes features straight through to the model's own normalizer).
	stages.outlier_means.assign(H, 0.0);
	stages.outlier_stds.assign(H, 0.0);
	stages.outlier_lower.assign(H, 0.0);
	stages.outlier_upper.assign(H, 0.0);
	if (standardize) {
	const double kThreshold = 4.0;
	const bool use_ddof1 = n_train > 1;
	for (idx_t j = 0; j < H; j++) {
		auto mean_std = [&](const vector<char> &masked) {
			double sum = 0.0;
			idx_t cnt = 0;
			for (idx_t i = 0; i < n_train; i++) {
				if (masked[i]) {
					continue;
				}
				sum += filt_full[i * H + j];
				cnt++;
			}
			double mean = cnt > 0 ? sum / (double)cnt : 0.0;
			double var = 0.0;
			for (idx_t i = 0; i < n_train; i++) {
				if (masked[i]) {
					continue;
				}
				double dv = filt_full[i * H + j] - mean;
				var += dv * dv;
			}
			idx_t denom = use_ddof1 ? (cnt > 1 ? cnt - 1 : 1) : cnt;
			double stdv = std::sqrt(var / (double)(denom == 0 ? 1 : denom));
			return std::make_pair(mean, std::max(stdv, 1e-6));
		};
		// Stage 1.
		vector<char> none_masked(n_train, 0);
		auto st1 = mean_std(none_masked);
		double lo1 = st1.first - kThreshold * st1.second;
		double up1 = st1.first + kThreshold * st1.second;
		// Mark outliers, recompute.
		vector<char> masked(n_train, 0);
		for (idx_t i = 0; i < n_train; i++) {
			double val = filt_full[i * H + j];
			if (val < lo1 || val > up1) {
				masked[i] = 1;
			}
		}
		auto st2 = mean_std(masked);
		stages.outlier_means[j] = st2.first;
		stages.outlier_stds[j] = st2.second;
		stages.outlier_lower[j] = st2.first - kThreshold * st2.second;
		stages.outlier_upper[j] = st2.first + kThreshold * st2.second;
	}
	// transform: X = max(-log1p(|X|)+lower, X); X = min(log1p(|X|)+upper, X).
	for (idx_t i = 0; i < T; i++) {
		for (idx_t j = 0; j < H; j++) {
			double x = filt_full[i * H + j];
			double a = -std::log1p(std::fabs(x)) + stages.outlier_lower[j];
			x = std::max(a, x);
			double b = std::log1p(std::fabs(x)) + stages.outlier_upper[j];
			x = std::min(b, x);
			filt_full[i * H + j] = x;
		}
	}
	} // if (standardize)

	batch.T = T;
	batch.H = H;
	batch.d = H;
	batch.train_size = n_train;
	batch.x = std::move(filt_full);

	// --- Target encoding ---
	batch.y.assign(T, kTargetPadSentinel);
	if (task == PreprocessTask::CLASSIFICATION) {
		// Alphabetical label encode (sklearn LabelEncoder convention).
		std::set<string> label_keys;
		std::map<string, Value> repr;
		for (idx_t i = 0; i < n_train; i++) {
			Value v = rows.GetValue(target_col, train_rows[i]);
			string key = v.ToString();
			label_keys.insert(key);
			if (repr.find(key) == repr.end()) {
				repr[key] = v;
			}
		}
		vector<string> classes(label_keys.begin(), label_keys.end()); // sorted
		std::map<string, int64_t> code_of;
		batch.label_decoder.clear();
		for (idx_t k = 0; k < classes.size(); k++) {
			code_of[classes[k]] = (int64_t)k;
			batch.label_decoder.push_back(repr[classes[k]]);
		}
		batch.y_train.assign(n_train, 0);
		for (idx_t i = 0; i < n_train; i++) {
			int64_t code = code_of[rows.GetValue(target_col, train_rows[i]).ToString()];
			batch.y_train[i] = code;
			batch.y[i] = (double)code;
		}
	} else {
		double sum = 0.0;
		for (idx_t i = 0; i < n_train; i++) {
			sum += rows.GetValue(target_col, train_rows[i]).GetValue<double>();
		}
		double mean = sum / (double)n_train;
		double var = 0.0;
		for (idx_t i = 0; i < n_train; i++) {
			double dv =
			    rows.GetValue(target_col, train_rows[i]).GetValue<double>() - mean;
			var += dv * dv;
		}
		double stdv = std::sqrt(var / (double)n_train); // ddof=0
		if (stdv == 0.0) {
			stdv = 1.0; // sklearn StandardScaler zero-variance guard
		}
		batch.target_mean = mean;
		batch.target_scale = stdv;
		for (idx_t i = 0; i < n_train; i++) {
			double yv = rows.GetValue(target_col, train_rows[i]).GetValue<double>();
			batch.y[i] = (yv - mean) / stdv;
		}
	}

	return batch;
}

} // namespace anofox
} // namespace duckdb
