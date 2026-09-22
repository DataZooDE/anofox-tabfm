//===----------------------------------------------------------------------===//
// tabfm_preprocess_tabpfn_v2.cpp — tabpfn_v2 fixture-scoped preprocessing profile
//
// MODL-01 fixture-scoped; full-fidelity tabpfn_v2 preprocessing deferred
// (real-export work). See RESEARCH §1 for required PreprocessedBatch fields.
//
// This is an intentionally MINIMAL profile that passes features through as-is
// (no z-score / no categorical encoding) — enough to exercise the registry +
// distribution path end-to-end on the K=16 fixture. The real TabFM v1 pipeline
// (encode → filter → scale → outlier-clip) is NOT applied here because:
//   (a) The fixture ONNX graph is hand-built to accept raw float32 tensors
//       directly (no preprocessing pipeline baked in).
//   (b) Full-fidelity preprocessing requires the real TabPFN v2 export, which
//       is deferred (ONNX export blocked by data-dependent preprocessing, see
//       SPIKE-tabpfn-v2-tensor-contract.md §5).
//
// The profile self-registers via the ProfileRegistration RAII pattern and is
// kept linked by ForceTabPFNV2ProfileInit() chained from ForceProfileInit()
// (RESEARCH Pitfall 2).
//===----------------------------------------------------------------------===//

#include "tabfm_profile_registry.hpp"
#include "tabfm_preprocess.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

#include <cmath>

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// tabpfn_v2 fixture-scoped preprocessing function
//===----------------------------------------------------------------------===//

/// Minimal fixture-scoped PreprocessedBatch for the tabpfn_v2 distribution model.
///
/// Populates fields required by the engine's ORT Run + DecodeDistribution path:
///   T, H, d, train_size   — row/feature counts
///   x                     — row-major [T, H] float64 features (train first, test last)
///   y                     — train targets z-scored with target_mean/target_scale,
///                           kTargetPadSentinel for test rows
///   cat_mask              — all false (fixture has no categoricals)
///   target_mean           — mean of training targets (for affine inverse transform)
///   target_scale          — std of training targets (ddof=0)
///   label_decoder         — empty (regression)
///   row_source_index      — maps output row index back to source collection row
///
/// Feature extraction: reads numeric/double values directly (no encoding pipeline).
/// Non-numeric feature values are silently cast to 0.0 (fixture columns are numeric).
///
/// Reference: RESEARCH §1 "tabpfn_v2 Preprocessing Profile (fixture-scoped)",
/// code example at lines 880-896.
PreprocessedBatch TabPFNV2PreprocessBatch(const ColumnDataCollection &data,
                                         const vector<PreprocessColumnSpec> &columns,
                                         PreprocessTask task) {
	// ── Find target column and feature columns ────────────────────────────────
	idx_t target_col = DConstants::INVALID_INDEX;
	vector<idx_t> feature_cols;

	for (idx_t c = 0; c < columns.size(); c++) {
		if (columns[c].is_target) {
			target_col = c;
		} else if (columns[c].is_feature) {
			feature_cols.push_back(c);
		}
	}
	if (target_col == DConstants::INVALID_INDEX) {
		throw InvalidInputException(
		    "tabfm: tabpfn_v2 profile: no target column found in collection");
	}

	const idx_t n_rows = data.Count();
	const idx_t H = feature_cols.size();
	if (H == 0) {
		throw InvalidInputException(
		    "tabfm: tabpfn_v2 profile: no feature columns found in collection");
	}

	ColumnDataRowCollection rows = data.GetRows();

	// ── Separate train rows (non-NULL target) from test rows (NULL target) ────
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
		    "tabfm: tabpfn_v2 profile: no in-context train rows "
		    "(all target values are NULL)");
	}

	const idx_t T = n_train + n_test;

	// ── Build row_source_index: train rows first, test rows last ─────────────
	// This bookkeeping matches the engine's DecodeDistribution row scatter:
	//   batch.row_source_index[t] → original collection row for ORT output row t
	vector<idx_t> src(T);
	for (idx_t i = 0; i < n_train; i++) {
		src[i] = train_rows[i];
	}
	for (idx_t i = 0; i < n_test; i++) {
		src[n_train + i] = test_rows[i];
	}

	// ── Feature matrix x [T, H] ── row-major, train rows first ──────────────
	// Fixture-scoped: pass features through as-is (no z-score / encoding pipeline).
	// The fixture ONNX graph accepts raw float32 tensors directly.
	vector<double> x(T * H, 0.0);
	for (idx_t i = 0; i < n_train; i++) {
		idx_t r = train_rows[i];
		for (idx_t j = 0; j < H; j++) {
			Value v = rows.GetValue(feature_cols[j], r);
			x[i * H + j] = v.IsNull() ? 0.0 : v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		}
	}
	for (idx_t i = 0; i < n_test; i++) {
		idx_t r = test_rows[i];
		for (idx_t j = 0; j < H; j++) {
			Value v = rows.GetValue(feature_cols[j], r);
			x[(n_train + i) * H + j] = v.IsNull() ? 0.0 : v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
		}
	}

	// ── Target vector y [T] ───────────────────────────────────────────────────
	// Train targets: z-scored (y_z = (y - mean) / std) so the engine's affine
	// inverse transform (yhat = z * target_scale + target_mean) recovers raw space.
	// Test rows: kTargetPadSentinel (-100.0) per upstream contract.
	//
	// Compute target_mean and target_scale (ddof=0) over train targets.
	double sum = 0.0;
	for (idx_t i = 0; i < n_train; i++) {
		sum += rows.GetValue(target_col, train_rows[i])
		           .DefaultCastAs(LogicalType::DOUBLE)
		           .GetValue<double>();
	}
	double target_mean = sum / static_cast<double>(n_train);

	double var = 0.0;
	for (idx_t i = 0; i < n_train; i++) {
		double yv = rows.GetValue(target_col, train_rows[i])
		                .DefaultCastAs(LogicalType::DOUBLE)
		                .GetValue<double>();
		double dv = yv - target_mean;
		var += dv * dv;
	}
	double target_scale = std::sqrt(var / static_cast<double>(n_train));
	if (target_scale < 1e-8) {
		target_scale = 1.0; // zero-variance guard (matches sklearn StandardScaler)
	}

	vector<double> y(T, kTargetPadSentinel);
	for (idx_t i = 0; i < n_train; i++) {
		double yv = rows.GetValue(target_col, train_rows[i])
		                .DefaultCastAs(LogicalType::DOUBLE)
		                .GetValue<double>();
		y[i] = (yv - target_mean) / target_scale;
	}
	// test rows stay kTargetPadSentinel

	// ── Assemble PreprocessedBatch ─────────────────────────────────────────────
	PreprocessedBatch batch;
	batch.task           = task;
	batch.T              = T;
	batch.H              = H;
	batch.d              = H;  // all H features are active (no unique-filter)
	batch.train_size     = n_train;
	batch.x              = std::move(x);
	batch.y              = std::move(y);
	batch.cat_mask.assign(H, false);   // fixture has no categorical features
	batch.label_decoder.clear();       // regression: no label decoder
	batch.target_mean    = target_mean;
	batch.target_scale   = target_scale;
	batch.row_source_index = std::move(src);
	// batch.encoders, batch.y_train, batch.stages left at defaults
	// (engine reads only the fields populated above for the distribution path)

	return batch;
}

//===----------------------------------------------------------------------===//
// Self-registration of the tabpfn_v2 profile
//===----------------------------------------------------------------------===//
static const ProfileRegistration kTabPFNV2Reg("tabpfn_v2", TabPFNV2PreprocessBatch);

//===----------------------------------------------------------------------===//
// ForceTabPFNV2ProfileInit — chained from ForceProfileInit() to keep this TU linked
//===----------------------------------------------------------------------===//
void ForceTabPFNV2ProfileInit() {
	// Empty body intentional (see ForceProfileInit comment in
	// tabfm_profile_registry.cpp). Existence prevents linker stripping.
}

} // namespace anofox
} // namespace duckdb
