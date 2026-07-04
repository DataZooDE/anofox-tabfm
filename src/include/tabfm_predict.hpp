#pragma once

#include "duckdb.hpp"
#include "tabfm_manifest.hpp" // TabFMTask

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// Predict surface (WS-E) — shared declarations for tabfm_predict_agg.cpp and
// tabfm_macros.cpp, plus the ENGINE SEAM the Phase-2 TabFM integration (WS-F)
// plugs into.
//
// Design notes (decisions made against DuckDB v1.5.4, see tabfm_predict_agg.cpp
// for the mechanics):
//
// * anofox_tabfm_predict_agg(row STRUCT, target VARCHAR [, opts MAP]) is a
//   plain aggregate returning LIST(STRUCT(cols, yhat, yhat_score, is_training
//   [, proba])) — the return type is computed AT BIND TIME (task inferred from
//   the target field's type per FR-3.2, 'output_mode' decides proba).
//
// * anofox_tabfm_predict_win is a SEPARATE function set (not a window overload
//   of the aggregate) whose bind returns the scalar STRUCT(yhat, yhat_score
//   [, proba]). Rationale: DuckDB reuses the aggregate's bound return type for
//   window usage, and a rolling prediction is one value per row — forcing the
//   aggregate's LIST shape through a window would demand per-row unnest
//   gymnastics. The non-window path of predict_win throws (an OVER clause with
//   a moving frame is required); the real work happens in the custom window
//   callback (SetWindowCallback).
//
// * Window "current row" contract: DuckDB's custom-window API (aggregate_window_t)
//   hands the callback the SubFrames in partition-row space, but `rid` indexes
//   the output chunk, NOT the partition (verified against
//   window_custom_aggregator.cpp / WindowAggregator::EvaluateSubFrames in the
//   pinned v1.5.4 sources — this deviates from earlier spike assumptions).
//   For a frame ending at `1 PRECEDING`, FrameEnd() computes the half-open
//   frame end as exactly the current row's partition index, unclamped, for
//   every row including the first. tabfm_predict_win therefore defines: the
//   scored row is the row immediately AFTER the frame (frames.back().end),
//   which IS the current row under the documented
//   `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` usage. The scored row's target
//   is never part of the context, even when non-NULL.
//===----------------------------------------------------------------------===//

//! Task (TabFMTask) is defined in tabfm_manifest.hpp; it is resolved at bind
//! time (FR-3.2: type-based, overridable via opts).

//! Options parsed (and fully validated) at bind from the constant opts MAP.
//! All values arrive as VARCHAR (forecast-style, SQL-API §2 Level 2).
struct TabFMPredictOptions {
	TabFMTask task = TabFMTask::CLASSIFICATION;
	//! ensemble members; 1..32 parse, >1 currently raises NotImplemented
	//! (the ensemble port arrives with the WS-F engine integration, M3)
	idx_t n_estimators = 1;
	int64_t seed = 42;
	//! output_mode == 'detail' → proba MAP(VARCHAR, DOUBLE) field
	bool detail = false;
	//! subsample the context to at most N rows (0 = use all context rows)
	idx_t context_rows = 0;
	double softmax_temperature = 0.9;
	string model = "tabfm-v1";
};

//! Per-row predictions for EVERY input row, in input order: context rows get
//! in-context fitted values, NULL-target rows get real predictions.
struct TabFMPredictResult {
	//! typed like the target for classification, DOUBLE for regression
	vector<Value> yhat;
	//! top-class probability (DOUBLE), NULL for regression
	vector<Value> yhat_score;
	//! MAP(VARCHAR, DOUBLE) label→probability; only populated when
	//! opts.detail && classification, empty vector otherwise
	vector<Value> proba;
};

//===----------------------------------------------------------------------===//
// ENGINE SEAM
//
// The predict surface is final; only this interface's implementation changes.
// The real engine (tabfm_engine.cpp) runs Preprocessor → ORT forward → decode,
// with the session cached in the DB-instance TabFMState and the forward pass
// serialized per device (HLD §4.6/§6). Inputs:
//
//   * `rows`        — the materialized group (or window context + scored row):
//                     one entry per input row, the row-struct's field values in
//                     struct-field order.
//   * `row_type`    — the row STRUCT type (field names + types).
//   * `target_idx`  — index of the target field inside each row.
//   * `target_type` / `target_name` — for output typing and error texts.
//   * `opts`        — bind-validated options (task already resolved).
//   * `ctx`         — settings + DatabaseInstance captured at bind time
//                     (finalize runs off-thread and cannot read the context).
//
// Contract: returns one prediction per input row, in input order. Rows with a
// non-NULL target are the context ("training") rows; the engine enforces the
// ≤ 10-class limit (SQL-API §5 text). Callers guarantee at least one non-NULL
// target row (the aggregate raises the §5 all-NULL error, the window
// variant soft-NULLs).
//===----------------------------------------------------------------------===//

//! Settings + DB handle captured at bind (finalize has no ClientContext).
//! DatabaseInstance is a complete type via duckdb.hpp above.
struct PredictContext {
	DatabaseInstance *db = nullptr;
	//! SET anofox_tabfm_model_manifest ('' → the built-in TabFM v1 manifest).
	string model_manifest_path;
	//! SET anofox_tabfm_cache_dir (already ~-expanded is not required here).
	string cache_dir;
	//! SET anofox_tabfm_threads (ORT intra-op).
	int64_t threads = 1;
	//! SET anofox_tabfm_device ('auto'|'cpu'|'cuda'|'rocm').
	string device = "auto";
	//! SET anofox_tabfm_gpu_precision ('bf16'|'fp16'|'fp32') — MIGraphX compile
	//! precision on the ROCm GPU (RDNA4 runs bf16/fp16 at ~2x fp32). bf16 keeps
	//! fp32's exponent range → safest. Ignored by the CPU/CUDA ORT paths.
	string gpu_precision = "bf16";
	//! SET anofox_tabfm_cpu_prepack — ORT weight prepacking on the CPU EP: faster
	//! matmuls at ~+16% RSS. On by default now that external-data keeps RSS low.
	bool cpu_prepack = true;
};

struct PredictInput {
	const vector<vector<Value>> &rows;
	const LogicalType &row_type;
	idx_t target_idx;
	const LogicalType &target_type;
	const string &target_name;
	const TabFMPredictOptions &opts;
	const PredictContext &ctx;
};

class PredictEngine {
public:
	virtual ~PredictEngine() = default;
	virtual TabFMPredictResult Predict(const PredictInput &input) = 0;
};

//! The process-wide TabFM engine (tabfm_engine.cpp): resolves the model from
//! the manifest, loads/caches an ORT session, preprocesses, runs one forward
//! pass and decodes. Errors with the SQL-API §5 remediation text when the
//! model's weights are not available.
PredictEngine &GetPredictEngine();

} // namespace anofox
} // namespace duckdb
