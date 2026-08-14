//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_generate.cpp — WS-G. Column planning + sampling primitives (the pure,
// unit-tested half) and the chain-rule / imputation drivers that call the
// PredictEngine seam. The SQL surface lives in tabfm_generate_macros.cpp.
//
// See tabfm_generate.hpp for the chain rule and the binning rationale.
//===----------------------------------------------------------------------===//

#include "tabfm_generate.hpp"
#include "tabfm_preprocess.hpp"
#include "tabfm_registration.hpp"
#include "tabfm_registry.hpp"
#include "tabfm_state.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/to_string.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace duckdb {
namespace anofox {

namespace {

//! Numeric value of `v` as a double, or NaN when it is NULL / not numeric.
double AsDouble(const Value &v) {
	if (v.IsNull()) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	try {
		return DoubleValue::Get(v.DefaultCastAs(LogicalType::DOUBLE));
	} catch (const std::exception &) {
		return std::numeric_limits<double>::quiet_NaN();
	}
}

//! A column we can bin: real numbers, but NOT booleans (two states — that is a
//! categorical) and not temporal (see PlanColumn).
bool IsContinuousType(const LogicalType &type) {
	if (type.id() == LogicalTypeId::BOOLEAN) {
		return false;
	}
	return type.IsNumeric();
}

bool IsTemporalType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::INTERVAL:
		return true;
	default:
		return false;
	}
}

//! Coerce a sampled double into `type`, rounding for integral targets (a plain
//! cast truncates, which would bias every integer column downward) and clamping
//! into [lo, hi] so nothing escapes the observed range.
Value CoerceNumeric(double v, const LogicalType &type, double lo, double hi) {
	if (v < lo) {
		v = lo;
	}
	if (v > hi) {
		v = hi;
	}
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::UHUGEINT: {
		double rounded = std::round(v);
		// Rounding can step outside the bin; pull it back before the cast.
		if (rounded < lo) {
			rounded = std::ceil(lo);
		}
		if (rounded > hi) {
			rounded = std::floor(hi);
		}
		return Value::DOUBLE(rounded).DefaultCastAs(type);
	}
	default:
		return Value::DOUBLE(v).DefaultCastAs(type);
	}
}

} // namespace

//===--------------------------------------------------------------------===//
// Sampling primitives
//===--------------------------------------------------------------------===//

vector<double> QuantileEdges(vector<double> values, idx_t bins) {
	vector<double> edges;
	// Drop NULL-ish entries; a column of all NaN has no distribution.
	values.erase(std::remove_if(values.begin(), values.end(),
	                            [](double v) { return !std::isfinite(v); }),
	             values.end());
	if (values.size() < 2 || bins < 2) {
		return edges;
	}
	std::sort(values.begin(), values.end());
	const idx_t n = values.size();

	// Order-statistic quantiles. More bins than distinct values is pointless —
	// the duplicate edges would collapse below anyway — so cap up front.
	idx_t distinct = 1;
	for (idx_t i = 1; i < n; i++) {
		if (values[i] != values[i - 1]) {
			distinct++;
		}
	}
	if (distinct < 2) {
		return edges;
	}
	const idx_t k = MinValue<idx_t>(bins, distinct);

	edges.push_back(values.front());
	for (idx_t i = 1; i < k; i++) {
		auto pos = static_cast<idx_t>((static_cast<double>(i) * static_cast<double>(n)) / static_cast<double>(k));
		if (pos >= n) {
			pos = n - 1;
		}
		auto candidate = values[pos];
		// Ties collapse: keep the edge list strictly increasing so every bin has
		// a non-empty span and BinOf stays unambiguous.
		if (candidate > edges.back()) {
			edges.push_back(candidate);
		}
	}
	if (values.back() > edges.back()) {
		edges.push_back(values.back());
	}
	if (edges.size() < 2) {
		edges.clear();
	}
	return edges;
}

idx_t BinOf(double value, const vector<double> &edges) {
	if (edges.size() < 2) {
		return 0;
	}
	const idx_t last = edges.size() - 2; // index of the final bin
	if (!std::isfinite(value) || value <= edges.front()) {
		return 0;
	}
	if (value >= edges.back()) {
		return last;
	}
	// upper_bound gives the first edge strictly greater than value; the bin is
	// the one starting at the edge before it.
	auto it = std::upper_bound(edges.begin(), edges.end(), value);
	auto index = static_cast<idx_t>(it - edges.begin());
	return index == 0 ? 0 : MinValue<idx_t>(index - 1, last);
}

Value ExpandBin(const ColumnPlan &plan, idx_t bin, TabFMRandom &rng) {
	const auto &edges = plan.edges;
	if (edges.size() < 2) {
		return Value(plan.type);
	}
	const idx_t bin_count = edges.size() - 1;
	if (bin >= bin_count) {
		bin = bin_count - 1;
	}
	const double lo = edges[bin];
	const double hi = edges[bin + 1];

	// Empirical inverse CDF within the bin, when we know which observations fell
	// in it: pick a uniform position among them and interpolate between
	// neighbours. This keeps the bin's internal SHAPE, which matters most in the
	// outermost bins of a skewed column — those are wide, and a uniform fill
	// would push mass out into the sparse tail.
	if (bin + 1 < plan.bin_offsets.size()) {
		const idx_t start = plan.bin_offsets[bin];
		const idx_t end = plan.bin_offsets[bin + 1];
		if (end > start) {
			const idx_t count = end - start;
			const double position = rng.NextDouble() * static_cast<double>(count);
			auto offset = static_cast<idx_t>(position);
			if (offset >= count) {
				offset = count - 1;
			}
			const double fraction = position - static_cast<double>(offset);
			const double a = plan.sorted_values[start + offset];
			// Interpolate toward the next observation, or toward the bin's upper
			// edge for the last one, so the top of the bin stays reachable.
			const double b = (start + offset + 1 < end) ? plan.sorted_values[start + offset + 1] : hi;
			return CoerceNumeric(a + fraction * (b - a), plan.type, lo, hi);
		}
	}
	// No observations recorded for this bin (degenerate plan): fall back to a
	// uniform draw over its span.
	return CoerceNumeric(rng.NextDouble(lo, hi), plan.type, lo, hi);
}

Value DrawMarginal(const ColumnPlan &plan, TabFMRandom &rng) {
	switch (plan.kind) {
	case GenColumnKind::CATEGORICAL: {
		if (plan.labels.empty()) {
			return Value(plan.type);
		}
		vector<double> weights;
		weights.reserve(plan.label_counts.size());
		for (auto c : plan.label_counts) {
			weights.push_back(static_cast<double>(c));
		}
		return plan.labels[rng.WeightedChoice(weights)];
	}
	case GenColumnKind::CONTINUOUS: {
		vector<double> weights;
		weights.reserve(plan.bin_counts.size());
		for (auto c : plan.bin_counts) {
			weights.push_back(static_cast<double>(c));
		}
		return ExpandBin(plan, rng.WeightedChoice(weights), rng);
	}
	case GenColumnKind::CONSTANT:
		return plan.constant_value;
	default:
		return Value(plan.type);
	}
}

string TargetKey(const ColumnPlan &plan, idx_t index) {
	if (plan.kind == GenColumnKind::CONTINUOUS) {
		return to_string(index);
	}
	if (index >= plan.labels.size()) {
		return string();
	}
	return plan.labels[index].ToString();
}

Value EncodeTarget(const ColumnPlan &plan, const Value &row_value) {
	if (row_value.IsNull()) {
		return Value(LogicalType::VARCHAR); // the engine's "predict me" marker
	}
	if (plan.kind == GenColumnKind::CONTINUOUS) {
		return Value(to_string(BinOf(AsDouble(row_value), plan.edges)));
	}
	return Value(row_value.ToString());
}

Value SampleFromProba(const ColumnPlan &plan, const Value &proba, const Value &fallback, TabFMRandom &rng) {
	const idx_t classes = plan.ClassCount();
	if (classes == 0) {
		return Value(plan.type);
	}
	vector<double> weights(classes, 0.0);
	bool any = false;
	if (!proba.IsNull()) {
		// Index the MAP by the keys EncodeTarget produced, rather than trusting
		// its ordering: the engine builds label_decoder from the values it saw,
		// which need not be in plan order.
		for (auto &kv : MapValue::GetChildren(proba)) {
			auto &entry = StructValue::GetChildren(kv);
			if (entry[0].IsNull() || entry[1].IsNull()) {
				continue;
			}
			const auto key = entry[0].ToString();
			for (idx_t i = 0; i < classes; i++) {
				if (TargetKey(plan, i) == key) {
					weights[i] = DoubleValue::Get(entry[1].DefaultCastAs(LogicalType::DOUBLE));
					any = any || weights[i] > 0;
					break;
				}
			}
		}
	}
	if (!any) {
		// No usable distribution (compact mode, or only labels the plan never
		// saw): fall back to the argmax rather than inventing one.
		//
		// Careful — on a generation step `fallback` is the engine's yhat, which
		// is the winning ENCODED key. For a continuous column that key is a BIN
		// INDEX, so it must be expanded, not read as a value: decoding "3" as
		// the number 3.0 would silently emit garbage in the column's units.
		if (plan.kind == GenColumnKind::CONTINUOUS) {
			idx_t bin = 0;
			if (!fallback.IsNull()) {
				const auto key = fallback.ToString();
				for (idx_t i = 0; i < classes; i++) {
					if (TargetKey(plan, i) == key) {
						bin = i;
						break;
					}
				}
			}
			return ExpandBin(plan, bin, rng);
		}
		return DecodePointEstimate(plan, fallback);
	}
	const idx_t choice = rng.WeightedChoice(weights);
	if (plan.kind == GenColumnKind::CONTINUOUS) {
		return ExpandBin(plan, choice, rng);
	}
	return plan.labels[choice];
}

Value DecodePointEstimate(const ColumnPlan &plan, const Value &yhat) {
	if (yhat.IsNull()) {
		return Value(plan.type);
	}
	if (plan.kind == GenColumnKind::CONTINUOUS) {
		const double lo = plan.edges.empty() ? -std::numeric_limits<double>::infinity() : plan.edges.front();
		const double hi = plan.edges.empty() ? std::numeric_limits<double>::infinity() : plan.edges.back();
		auto raw = AsDouble(yhat);
		if (!std::isfinite(raw)) {
			return Value(plan.type);
		}
		return CoerceNumeric(raw, plan.type, lo, hi);
	}
	if (plan.kind == GenColumnKind::CONSTANT) {
		return plan.constant_value;
	}
	// Classification: the engine hands back a label from the domain we fed it,
	// as VARCHAR when EncodeTarget was used. Map it home by key.
	const auto key = yhat.ToString();
	for (idx_t i = 0; i < plan.labels.size(); i++) {
		if (plan.labels[i].ToString() == key) {
			return plan.labels[i];
		}
	}
	try {
		return yhat.DefaultCastAs(plan.type);
	} catch (const std::exception &) {
		return Value(plan.type);
	}
}

//===--------------------------------------------------------------------===//
// Column planning
//===--------------------------------------------------------------------===//

ColumnPlan PlanColumn(const string &name, const LogicalType &type, const vector<Value> &values, idx_t bins,
                      idx_t max_labels) {
	if (max_labels < 2) {
		max_labels = 2;
	}
	ColumnPlan plan;
	plan.name = name;
	plan.type = type;
	plan.constant_value = Value(type);

	vector<Value> observed;
	observed.reserve(values.size());
	for (auto &v : values) {
		if (v.IsNull()) {
			plan.null_count++;
		} else {
			observed.push_back(v);
		}
	}
	plan.observed_count = observed.size();

	if (IsTemporalType(type)) {
		// Fine as a FEATURE (the preprocessor expands temporal columns into five
		// numeric features); as a TARGET it needs an epoch round-trip that v1
		// does not do.
		plan.kind = GenColumnKind::UNSUPPORTED;
		plan.unsupported_reason =
		    StringUtil::Format("column '%s' is %s; generating temporal values is not supported yet", name,
		                       type.ToString());
		return plan;
	}

	if (observed.empty()) {
		plan.kind = GenColumnKind::CONSTANT; // all-NULL: emit NULL
		return plan;
	}

	if (IsContinuousType(type)) {
		vector<double> numbers;
		numbers.reserve(observed.size());
		for (auto &v : observed) {
			numbers.push_back(AsDouble(v));
		}
		plan.edges = QuantileEdges(numbers, MinValue<idx_t>(bins, max_labels));
		if (plan.edges.size() < 2) {
			plan.kind = GenColumnKind::CONSTANT;
			plan.constant_value = observed.front();
			return plan;
		}
		plan.kind = GenColumnKind::CONTINUOUS;
		plan.bin_counts.assign(plan.BinCount(), 0);
		for (auto number : numbers) {
			if (std::isfinite(number)) {
				plan.bin_counts[BinOf(number, plan.edges)]++;
			}
			if (std::isfinite(number)) {
				plan.sorted_values.push_back(number);
			}
		}
		// The empirical distribution ExpandBin draws through. Bins partition the
		// SORTED values contiguously (BinOf is monotone in value), so the offsets
		// are just the running counts.
		std::sort(plan.sorted_values.begin(), plan.sorted_values.end());
		plan.bin_offsets.assign(plan.BinCount() + 1, 0);
		for (idx_t b = 0; b < plan.BinCount(); b++) {
			plan.bin_offsets[b + 1] = plan.bin_offsets[b] + plan.bin_counts[b];
		}
		return plan;
	}

	// Categorical: distinct values in first-appearance order (stable output for
	// a given input, independent of hash iteration order).
	for (auto &v : observed) {
		const auto key = v.ToString();
		bool seen = false;
		for (idx_t i = 0; i < plan.labels.size(); i++) {
			if (plan.labels[i].ToString() == key) {
				plan.label_counts[i]++;
				seen = true;
				break;
			}
		}
		if (!seen) {
			plan.labels.push_back(v);
			plan.label_counts.push_back(1);
		}
	}
	if (plan.labels.size() == 1) {
		plan.kind = GenColumnKind::CONSTANT;
		plan.constant_value = plan.labels.front();
		return plan;
	}
	if (plan.labels.size() > max_labels) {
		plan.kind = GenColumnKind::UNSUPPORTED;
		plan.unsupported_reason = StringUtil::Format(
		    "column '%s' has %llu distinct values; at most %llu can be modelled", name,
		    static_cast<unsigned long long>(plan.labels.size()), static_cast<unsigned long long>(max_labels));
		return plan;
	}
	plan.kind = GenColumnKind::CATEGORICAL;
	return plan;
}

//===--------------------------------------------------------------------===//
// Chain-rule driver
//===--------------------------------------------------------------------===//

namespace {

//! Predict options for one generation step. Always classification with the
//! proba MAP on: sampling needs a distribution, and the argmax alone would
//! collapse every synthetic row onto the conditional mode.
TabFMPredictOptions StepOptions(const TabFMGenerateOptions &opts, TabFMTask task, bool detail) {
	TabFMPredictOptions popts;
	popts.task = task;
	popts.detail = detail;
	popts.n_estimators = 1;
	popts.seed = opts.seed;
	popts.softmax_temperature = opts.temperature;
	popts.model = opts.model;
	return popts;
}

//! The order columns are visited in — the chain rule holds for any permutation,
//! but which one you pick changes which conditionals the model is asked for.
vector<idx_t> ChainOrder(const vector<ColumnPlan> &plans, const vector<idx_t> &modellable,
                         const TabFMGenerateOptions &opts, TabFMRandom &rng) {
	vector<idx_t> order = modellable;
	switch (opts.column_order) {
	case GenColumnOrder::NATURAL:
		break;
	case GenColumnOrder::MISSINGNESS: {
		std::stable_sort(order.begin(), order.end(),
		                 [&](idx_t a, idx_t b) { return plans[a].null_count < plans[b].null_count; });
		break;
	}
	case GenColumnOrder::RANDOM:
	default: {
		auto permutation = rng.Permutation(NumericCast<int64_t>(order.size()));
		vector<idx_t> shuffled;
		shuffled.reserve(order.size());
		for (auto p : permutation) {
			shuffled.push_back(order[NumericCast<idx_t>(p)]);
		}
		order = std::move(shuffled);
		break;
	}
	}
	return order;
}

} // namespace

vector<vector<Value>> RunChainRule(const vector<vector<Value>> &real_rows, const child_list_t<LogicalType> &fields,
                                   const vector<ColumnPlan> &plans, idx_t n, const TabFMGenerateOptions &opts,
                                   const PredictContext &ctx) {
	const idx_t columns = fields.size();
	vector<vector<Value>> synthetic(n, vector<Value>(columns));

	// Constant columns need no model — and carry no information as features
	// either, so they stay out of the chain entirely.
	vector<idx_t> modellable;
	for (idx_t c = 0; c < columns; c++) {
		if (plans[c].kind == GenColumnKind::CONSTANT) {
			for (idx_t r = 0; r < n; r++) {
				synthetic[r][c] = plans[c].constant_value;
			}
		} else {
			modellable.push_back(c);
		}
	}
	if (modellable.empty()) {
		throw InvalidInputException(
		    "tabfm_generate: every column is constant or NULL — there is no distribution to sample. "
		    "Pass a relation with at least one varying column");
	}

	TabFMRandom rng(opts.seed);
	auto order = ChainOrder(plans, modellable, opts, rng);

	// Step 1: no features to condition on, so the chain rule's first factor is
	// the empirical marginal. (With a single modellable column that marginal is
	// the entire joint distribution and no model call happens at all.)
	const idx_t first = order.front();
	for (idx_t r = 0; r < n; r++) {
		synthetic[r][first] = DrawMarginal(plans[first], rng);
	}

	// Steps 2..H: p(c_i | c_1..c_i-1), one engine call each. Sequential by
	// construction — step i needs the columns step i-1 produced.
	for (idx_t step = 1; step < order.size(); step++) {
		const idx_t target = order[step];
		auto &plan = plans[target];

		child_list_t<LogicalType> sub_fields;
		for (idx_t f = 0; f < step; f++) {
			const idx_t source = order[f];
			sub_fields.emplace_back(fields[source].first, fields[source].second);
		}
		// The target is VARCHAR whatever the column's real type: EncodeTarget
		// makes every step a classification problem, which is the only path with
		// a predictive distribution to sample from.
		sub_fields.emplace_back(fields[target].first, LogicalType::VARCHAR);
		const LogicalType sub_type = LogicalType::STRUCT(sub_fields);

		vector<vector<Value>> rows;
		rows.reserve(real_rows.size() + n);
		for (auto &real : real_rows) {
			vector<Value> row;
			row.reserve(step + 1);
			for (idx_t f = 0; f < step; f++) {
				row.push_back(real[order[f]]);
			}
			row.push_back(EncodeTarget(plan, real[target]));
			rows.push_back(std::move(row));
		}
		for (idx_t r = 0; r < n; r++) {
			vector<Value> row;
			row.reserve(step + 1);
			for (idx_t f = 0; f < step; f++) {
				row.push_back(synthetic[r][order[f]]);
			}
			row.push_back(Value(LogicalType::VARCHAR)); // predict me
			rows.push_back(std::move(row));
		}

		const auto popts = StepOptions(opts, TabFMTask::CLASSIFICATION, /*detail=*/true);
		const LogicalType target_type = LogicalType::VARCHAR;
		const string &target_name = fields[target].first;
		PredictInput input {rows, sub_type, step, target_type, target_name, popts, ctx};
		auto predictions = GetPredictEngine().Predict(input);

		for (idx_t r = 0; r < n; r++) {
			const idx_t row_index = real_rows.size() + r;
			Value proba(LogicalType::VARCHAR); // NULL == "no distribution"
			if (row_index < predictions.proba.size()) {
				proba = predictions.proba[row_index];
			}
			synthetic[r][target] = SampleFromProba(plan, proba, predictions.yhat[row_index], rng);
		}
	}
	return synthetic;
}

//===--------------------------------------------------------------------===//
// Imputation driver
//===--------------------------------------------------------------------===//

namespace {

//! The deterministic no-feature fill: the modal label, or the midpoint of the
//! most populated bin. Only reachable when a column is the ONLY modellable one.
Value BestGuess(const ColumnPlan &plan) {
	switch (plan.kind) {
	case GenColumnKind::CATEGORICAL: {
		idx_t best = 0;
		for (idx_t i = 1; i < plan.label_counts.size(); i++) {
			if (plan.label_counts[i] > plan.label_counts[best]) {
				best = i;
			}
		}
		return plan.labels.empty() ? Value(plan.type) : plan.labels[best];
	}
	case GenColumnKind::CONTINUOUS: {
		idx_t best = 0;
		for (idx_t i = 1; i < plan.bin_counts.size(); i++) {
			if (plan.bin_counts[i] > plan.bin_counts[best]) {
				best = i;
			}
		}
		const double lo = plan.edges[best];
		const double hi = plan.edges[best + 1];
		return DecodePointEstimate(plan, Value::DOUBLE((lo + hi) / 2.0));
	}
	case GenColumnKind::CONSTANT:
		return plan.constant_value;
	default:
		return Value(plan.type);
	}
}

} // namespace

vector<vector<Value>> RunImpute(const vector<vector<Value>> &input_rows, const child_list_t<LogicalType> &fields,
                                const vector<ColumnPlan> &plans, const vector<idx_t> &targets,
                                const TabFMGenerateOptions &opts, const PredictContext &ctx) {
	auto filled = input_rows;
	if (targets.empty() || input_rows.empty()) {
		return filled;
	}
	const idx_t columns = fields.size();

	// Which cells we are allowed to write. Captured up front so later rounds
	// refine the SAME cells instead of drifting over already-known values.
	vector<vector<bool>> missing(input_rows.size(), vector<bool>(columns, false));
	for (idx_t r = 0; r < input_rows.size(); r++) {
		for (auto t : targets) {
			missing[r][t] = input_rows[r][t].IsNull();
		}
	}

	// Ascending missingness: the best-observed columns are filled first, so the
	// sparsest ones get the most context to lean on.
	vector<idx_t> order = targets;
	std::stable_sort(order.begin(), order.end(),
	                 [&](idx_t a, idx_t b) { return plans[a].null_count < plans[b].null_count; });

	for (idx_t round = 0; round < MaxValue<idx_t>(opts.rounds, 1); round++) {
		for (auto target : order) {
			auto &plan = plans[target];
			if (plan.kind == GenColumnKind::CONSTANT || plan.kind == GenColumnKind::UNSUPPORTED) {
				for (idx_t r = 0; r < filled.size(); r++) {
					if (missing[r][target]) {
						filled[r][target] = plan.constant_value;
					}
				}
				continue;
			}
			bool any_missing = false;
			for (idx_t r = 0; r < filled.size() && !any_missing; r++) {
				any_missing = missing[r][target];
			}
			if (!any_missing) {
				continue;
			}

			vector<idx_t> features;
			for (idx_t c = 0; c < columns; c++) {
				if (c == target) {
					continue;
				}
				// UNSUPPORTED columns are kept as FEATURES on purpose. "Cannot be
				// a target" is not "carries no information": the preprocessor
				// expands temporal columns into numeric parts and ordinal-encodes
				// high-cardinality ones, so dropping them would throw away signal
				// for no reason. Only constants are skipped — they have none.
				if (plans[c].kind != GenColumnKind::CONSTANT) {
					features.push_back(c);
				}
			}
			if (features.empty()) {
				// Nothing to condition on: the conditional collapses to the
				// marginal, whose best estimate is the mode / modal bin.
				auto guess = BestGuess(plan);
				for (idx_t r = 0; r < filled.size(); r++) {
					if (missing[r][target]) {
						filled[r][target] = guess;
					}
				}
				continue;
			}

			// Imputation wants the conditional MODE/MEAN, not a sample — so the
			// raw column value is the target and the task follows its kind. That
			// keeps continuous fills at full precision instead of binning them.
			const bool continuous = plan.kind == GenColumnKind::CONTINUOUS;
			child_list_t<LogicalType> sub_fields;
			for (auto f : features) {
				sub_fields.emplace_back(fields[f].first, fields[f].second);
			}
			sub_fields.emplace_back(fields[target].first, fields[target].second);
			const LogicalType sub_type = LogicalType::STRUCT(sub_fields);

			vector<vector<Value>> rows;
			rows.reserve(filled.size());
			for (idx_t r = 0; r < filled.size(); r++) {
				vector<Value> row;
				row.reserve(features.size() + 1);
				for (auto f : features) {
					row.push_back(filled[r][f]);
				}
				// A cell we are filling stays NULL for THIS call even if an
				// earlier round already guessed it — a guessed value must never
				// be presented to the model as ground truth.
				row.push_back(missing[r][target] ? Value(fields[target].second) : filled[r][target]);
				rows.push_back(std::move(row));
			}

			const auto popts = StepOptions(opts, continuous ? TabFMTask::REGRESSION : TabFMTask::CLASSIFICATION,
			                               /*detail=*/false);
			const LogicalType target_type = fields[target].second;
			const string &target_name = fields[target].first;
			PredictInput input {rows, sub_type, features.size(), target_type, target_name, popts, ctx};
			auto predictions = GetPredictEngine().Predict(input);

			for (idx_t r = 0; r < filled.size(); r++) {
				if (missing[r][target]) {
					filled[r][target] = DecodePointEstimate(plan, predictions.yhat[r]);
				}
			}
		}
	}
	return filled;
}

//===--------------------------------------------------------------------===//
// The aggregates
//
// Mirrors tabfm_predict_agg.cpp: materialize the group, call the engine in
// finalize, return LIST(STRUCT(...)) whose type is computed at bind. Registered
// under internal `__anofox_tabfm_*` names; the user-facing surface is the
// tabfm_generate / tabfm_impute table macros (tabfm_generate_macros.cpp).
//===--------------------------------------------------------------------===//

namespace {

struct GenerateBindData : public FunctionData {
	//! "tabfm_generate" or "tabfm_impute" (error-text prefix)
	string function_name;
	bool is_impute = false;
	LogicalType row_type;
	//! generate: rows to synthesize.
	idx_t n = 0;
	//! impute: column indices to fill (empty = every column that has NULLs).
	vector<idx_t> targets;
	bool targets_explicit = false;
	TabFMGenerateOptions options;
	idx_t max_rows = 10000;
	//! What `features := [...]` asked for, unit-separated, as the macro saw it.
	string requested_features;
	PredictContext context;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<GenerateBindData>(*this);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<GenerateBindData>();
		return function_name == other.function_name && row_type == other.row_type && n == other.n &&
		       targets == other.targets && targets_explicit == other.targets_explicit &&
		       options.seed == other.options.seed && options.temperature == other.options.temperature &&
		       options.bins == other.options.bins && options.column_order == other.options.column_order &&
		       options.rounds == other.options.rounds && options.model == other.options.model &&
		       max_rows == other.max_rows;
	}

	//! generate: LIST(STRUCT(cols, synthetic_id)); impute: LIST(STRUCT(cols)).
	LogicalType ResultType() const {
		child_list_t<LogicalType> fields;
		fields.emplace_back("cols", row_type); // 'row' is a reserved word (S04)
		if (!is_impute) {
			fields.emplace_back("synthetic_id", LogicalType::BIGINT);
		}
		return LogicalType::LIST(LogicalType::STRUCT(std::move(fields)));
	}
};

//===--------------------------------------------------------------------===//
// Options parsing (bind time; opts must be a constant MAP, values VARCHAR)
//===--------------------------------------------------------------------===//

int64_t ParseIntegerOption(const string &fname, const string &key, const string &raw, const char *expectation) {
	try {
		size_t pos = 0;
		auto parsed = std::stoll(raw, &pos);
		if (pos != raw.size()) {
			throw std::invalid_argument(raw);
		}
		return parsed;
	} catch (const std::exception &) {
		throw BinderException("%s: %s %s, got '%s'", fname, key, expectation, raw);
	}
}

void ParseOneOption(const string &fname, GenerateBindData &bind, const string &key_raw, const string &val) {
	auto key = StringUtil::Lower(key_raw);
	auto &opts = bind.options;
	if (key == "seed") {
		opts.seed = ParseIntegerOption(fname, "seed", val, "must be an integer");
	} else if (key == "temperature") {
		try {
			size_t pos = 0;
			auto parsed = std::stod(val, &pos);
			if (pos != val.size() || !(parsed > 0)) {
				throw std::invalid_argument(val);
			}
			opts.temperature = parsed;
		} catch (const std::exception &) {
			throw BinderException("%s: temperature must be a positive number, got '%s'", fname, val);
		}
	} else if (key == "bins") {
		auto n = ParseIntegerOption(fname, "bins", val, "must be an integer");
		if (n < 2 || n > (int64_t)kMaxGenerateBins) {
			throw BinderException("%s: bins must be an integer between 2 and %llu, got '%s'", fname,
			                      (unsigned long long)kMaxGenerateBins, val);
		}
		opts.bins = NumericCast<idx_t>(n);
	} else if (key == "column_order") {
		auto lower = StringUtil::Lower(val);
		if (lower == "random") {
			opts.column_order = GenColumnOrder::RANDOM;
		} else if (lower == "natural") {
			opts.column_order = GenColumnOrder::NATURAL;
		} else if (lower == "missingness") {
			opts.column_order = GenColumnOrder::MISSINGNESS;
		} else {
			throw BinderException("%s: column_order must be 'random', 'natural' or 'missingness', got '%s'", fname,
			                      val);
		}
	} else if (key == "rounds") {
		auto n = ParseIntegerOption(fname, "rounds", val, "must be a positive integer");
		if (n < 1 || n > 16) {
			throw BinderException("%s: rounds must be an integer between 1 and 16, got '%s'", fname, val);
		}
		opts.rounds = NumericCast<idx_t>(n);
	} else if (key == "__features") {
		// Internal: what `features := [...]` asked for, so bind can tell a
		// misspelled column from one deliberately excluded.
		bind.requested_features = val;
	} else if (key == "model") {
		opts.model = val;
	} else {
		throw BinderException("%s: unknown option '%s'; valid options are seed, temperature, bins, column_order, "
		                      "rounds, model",
		                      fname, key_raw);
	}
}

void ParseOptionsArgument(ClientContext &context, Expression &expr, GenerateBindData &bind, const string &fname) {
	const auto type_id = expr.return_type.id();
	if (type_id == LogicalTypeId::SQLNULL || type_id == LogicalTypeId::UNKNOWN) {
		return;
	}
	if (type_id != LogicalTypeId::MAP && type_id != LogicalTypeId::STRUCT) {
		throw BinderException("%s: options must be a MAP, e.g. opts := MAP{'seed':'7'}", fname);
	}
	if (!expr.IsFoldable()) {
		throw BinderException("%s: options MAP must be a constant", fname);
	}
	auto opts_value = ExpressionExecutor::EvaluateScalar(context, expr);
	if (opts_value.IsNull()) {
		return;
	}
	if (type_id == LogicalTypeId::MAP) {
		for (auto &kv : MapValue::GetChildren(opts_value)) {
			auto &entry = StructValue::GetChildren(kv);
			ParseOneOption(fname, bind, entry[0].ToString(), entry[1].IsNull() ? string() : entry[1].ToString());
		}
	} else {
		auto &child_types = StructType::GetChildTypes(opts_value.type());
		auto &children = StructValue::GetChildren(opts_value);
		for (idx_t i = 0; i < children.size(); i++) {
			ParseOneOption(fname, bind, child_types[i].first, children[i].IsNull() ? string() : children[i].ToString());
		}
	}
}

idx_t ReadSettingUBigint(ClientContext &context, const char *name, idx_t fallback) {
	Value value;
	if (context.TryGetCurrentSetting(name, value) && !value.IsNull()) {
		return NumericCast<idx_t>(BigIntValue::Get(value.DefaultCastAs(LogicalType::BIGINT)));
	}
	return fallback;
}

string ExpandUserHome(const string &path) {
	if (path.empty() || path[0] != '~') {
		return path;
	}
	const char *home = std::getenv("HOME");
	if (!home) {
		return path;
	}
	return string(home) + path.substr(1);
}

void CaptureContext(ClientContext &context, GenerateBindData &bind) {
	bind.context.db = &DatabaseInstance::GetDatabase(context);
	bind.context.threads = NumericCast<int64_t>(ReadSettingUBigint(context, "anofox_tabfm_threads", 1));
	Value setting;
	if (context.TryGetCurrentSetting("anofox_tabfm_default_model", setting) && !setting.IsNull()) {
		bind.context.default_model = setting.ToString();
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_cache_dir", setting) && !setting.IsNull()) {
		bind.context.cache_dir = ExpandUserHome(setting.ToString());
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_device", setting) && !setting.IsNull()) {
		bind.context.device = setting.ToString();
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_gpu_precision", setting) && !setting.IsNull()) {
		bind.context.gpu_precision = StringUtil::Lower(setting.ToString());
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_cpu_prepack", setting) && !setting.IsNull()) {
		bind.context.cpu_prepack = BooleanValue::Get(setting);
	}
	if (context.TryGetCurrentSetting("anofox_tabfm_mxr_source", setting) && !setting.IsNull()) {
		bind.context.mxr_source = setting.ToString();
	}
}

unique_ptr<FunctionData> GenerateBindInternal(ClientContext &context, AggregateFunction &function,
                                              vector<unique_ptr<Expression>> &arguments, const string &fname,
                                              bool is_impute) {
	auto bind = make_uniq<GenerateBindData>();
	bind->function_name = fname;
	bind->is_impute = is_impute;

	if (arguments[0]->return_type.id() != LogicalTypeId::STRUCT) {
		throw BinderException("%s: first argument must be the whole-row STRUCT — call the table macro, e.g. "
		                      "SELECT * FROM %s('customers', 100)",
		                      fname, fname);
	}
	bind->row_type = arguments[0]->return_type;
	auto &fields = StructType::GetChildTypes(bind->row_type);
	if (fields.empty()) {
		throw BinderException("%s: the input relation has no columns", fname);
	}
	// Every column is both a feature and (in turn) a target of the chain rule, so
	// a nested column is unusable in either role: it would be ordinal-encoded as
	// garbage and — for generate — SAMPLED, silently emitting meaningless values
	// in that column. Reject it here (issue #17).
	for (auto &field : fields) {
		if (!IsUnsupportedNestedType(field.second)) {
			continue;
		}
		throw BinderException("%s: column '%s' has unsupported type %s; columns must be scalar — project it into "
		                      "scalar columns (e.g. %s[1] AS f1, %s[2] AS f2) or exclude it with features := [...]",
		                      fname, field.first, field.second.ToString(), field.first, field.first);
	}

	// arg 1: generate -> n (BIGINT); impute -> columns (VARCHAR[] or NULL)
	if (!arguments[1]->IsFoldable()) {
		throw BinderException("%s: %s must be a constant", fname, is_impute ? "columns" : "n");
	}
	auto arg1 = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	if (is_impute) {
		if (!arg1.IsNull()) {
			auto list_value = arg1.type().id() == LogicalTypeId::LIST
			                      ? arg1
			                      : arg1.DefaultCastAs(LogicalType::LIST(LogicalType::VARCHAR));
			for (auto &entry : ListValue::GetChildren(list_value)) {
				if (entry.IsNull()) {
					continue;
				}
				const auto wanted = entry.ToString();
				bool found = false;
				for (idx_t i = 0; i < fields.size(); i++) {
					if (StringUtil::CIEquals(fields[i].first, wanted)) {
						bind->targets.push_back(i);
						found = true;
						break;
					}
				}
				if (!found) {
					throw BinderException("%s: column '%s' not found in the input relation", fname, wanted);
				}
			}
			bind->targets_explicit = true;
		}
	} else {
		if (arg1.IsNull()) {
			throw BinderException("%s: n must be a positive integer, not NULL", fname);
		}
		auto n = BigIntValue::Get(arg1.DefaultCastAs(LogicalType::BIGINT));
		if (n <= 0) {
			throw BinderException("%s: n must be a positive integer, got %lld", fname, (long long)n);
		}
		bind->n = NumericCast<idx_t>(n);
	}

	if (arguments.size() > 2) {
		ParseOptionsArgument(context, *arguments[2], *bind, fname);
	}

	// How wide the selected model's class head is. Every generation step is a
	// classification problem, so this caps both `bins` and the categorical
	// cardinality we can model. Without it a narrow model fails deep inside the
	// engine with a tensor-shape error that never mentions `bins`.
	//
	// Resolution is best-effort: an unresolvable model (ambiguous default, or an
	// id that does not exist) is NOT diagnosed here — the engine owns that error
	// and phrases it far better. We just keep the default cap.
	try {
		auto registry = ModelRegistry::Build(TabFMState::Get(context)->RegisteredSpecs());
		Value default_model_setting;
		string default_model;
		if (context.TryGetCurrentSetting("anofox_tabfm_default_model", default_model_setting) &&
		    !default_model_setting.IsNull()) {
			default_model = default_model_setting.ToString();
		}
		const auto &spec = registry.Resolve(bind->options.model, default_model);
		if (spec.size_regime.max_classes > 0) {
			bind->options.max_classes =
			    MinValue<idx_t>(kMaxGenerateBins, NumericCast<idx_t>(spec.size_regime.max_classes));
		}
	} catch (const std::exception &) {
		// keep kMaxGenerateBins
	}
	bind->options.bins = MinValue<idx_t>(bind->options.bins, bind->options.max_classes);

	// A name that matches no column is dropped by the macro's COLUMNS(lambda)
	// filter, so without this the call succeeds having generated FEWER columns
	// than the caller listed — silently, and with plausible-looking output.
	if (!bind->requested_features.empty()) {
		size_t start = 0;
		while (start <= bind->requested_features.size()) {
			auto end = bind->requested_features.find('\x1f', start);
			if (end == string::npos) {
				end = bind->requested_features.size();
			}
			auto wanted = bind->requested_features.substr(start, end - start);
			start = end + 1;
			if (wanted.empty()) {
				continue;
			}
			bool found = false;
			for (auto &field : fields) {
				if (StringUtil::CIEquals(field.first, wanted)) {
					found = true;
					break;
				}
			}
			if (!found) {
				// Deliberately no list of "available" columns: by the time this
				// runs the row struct has ALREADY been filtered to the requested
				// names, so the one column a caller most needs to see — the one
				// they misspelled — is precisely the one missing from it.
				throw BinderException(
				    "%s: features := [...] names '%s', which matched no column. Check it against the relation's "
				    "columns (matching is case-insensitive); every other listed name was found.",
				    fname, wanted);
			}
		}
	}

	bind->max_rows = ReadSettingUBigint(context, "anofox_tabfm_max_rows", 10000);
	const auto max_features = ReadSettingUBigint(context, "anofox_tabfm_max_features", 500);
	if (fields.size() > max_features) {
		throw BinderException("%s: %llu columns exceed anofox_tabfm_max_features (%llu). Raise it with "
		                      "SET anofox_tabfm_max_features = %llu; or pass a feature list",
		                      fname, (unsigned long long)fields.size(), (unsigned long long)max_features,
		                      (unsigned long long)fields.size());
	}
	if (!is_impute && bind->n > bind->max_rows) {
		throw BinderException("%s: n (%llu) exceeds anofox_tabfm_max_rows (%llu). Raise it with "
		                      "SET anofox_tabfm_max_rows = <n>; or generate fewer rows",
		                      fname, (unsigned long long)bind->n, (unsigned long long)bind->max_rows);
	}

	function.return_type = bind->ResultType();
	CaptureContext(context, *bind);
	PostHogTelemetry::Instance().RecordFunctionCall(fname);
	return std::move(bind);
}

unique_ptr<FunctionData> GenerateAggBind(ClientContext &context, AggregateFunction &function,
                                         vector<unique_ptr<Expression>> &arguments) {
	return GenerateBindInternal(context, function, arguments, "tabfm_generate", false);
}

unique_ptr<FunctionData> ImputeAggBind(ClientContext &context, AggregateFunction &function,
                                       vector<unique_ptr<Expression>> &arguments) {
	return GenerateBindInternal(context, function, arguments, "tabfm_impute", true);
}

//===--------------------------------------------------------------------===//
// Aggregate state (same shape as the predict aggregate: the state slot stays
// trivially movable, the destructor callback releases the heap buffer)
//===--------------------------------------------------------------------===//

struct GenerateRowBuffer {
	explicit GenerateRowBuffer(const LogicalType &row_type)
	    : collection(Allocator::DefaultAllocator(), vector<LogicalType> {row_type}) {
		append_chunk.Initialize(Allocator::DefaultAllocator(), vector<LogicalType> {row_type}, 1);
	}
	ColumnDataCollection collection;
	DataChunk append_chunk;

	void Append(const Value &row_value) {
		append_chunk.Reset();
		append_chunk.SetValue(0, 0, row_value);
		append_chunk.SetCardinality(1);
		collection.Append(append_chunk);
	}
};

struct GenerateAggState {
	GenerateRowBuffer *rows;
};

void GenerateStateInitialize(const AggregateFunction &, data_ptr_t state_ptr) {
	reinterpret_cast<GenerateAggState *>(state_ptr)->rows = nullptr;
}

void FreeState(GenerateAggState &state) {
	delete state.rows;
	state.rows = nullptr;
}

void GenerateStateDestroy(Vector &state_vector, AggregateInputData &, idx_t count) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<GenerateAggState **>(sdata.data);
	for (idx_t i = 0; i < count; i++) {
		FreeState(*states[sdata.sel->get_index(i)]);
	}
}

//! DuckDB does not guarantee Destroy runs when a query errors mid-aggregation,
//! and our buffers come from a DuckDB Allocator — an un-freed one trips the
//! allocation_count==0 assertion at shutdown.
struct StateVectorCleanup {
	Vector &state_vector;
	idx_t count;
	bool released = false;

	~StateVectorCleanup() {
		if (released) {
			return;
		}
		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);
		auto states = reinterpret_cast<GenerateAggState **>(sdata.data);
		for (idx_t i = 0; i < count; i++) {
			FreeState(*states[sdata.sel->get_index(i)]);
		}
	}
};

idx_t GenerateStateSize(const AggregateFunction &) {
	return sizeof(GenerateAggState);
}

void GenerateAggUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t, Vector &state_vector,
                       idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<GenerateBindData>();
	StateVectorCleanup guard {state_vector, count};

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<GenerateAggState **>(sdata.data);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		if (!state.rows) {
			state.rows = new GenerateRowBuffer(bind.row_type);
		}
		if (state.rows->collection.Count() >= bind.max_rows) {
			throw InvalidInputException("%s: input exceeds anofox_tabfm_max_rows (%llu). Raise it with "
			                            "SET anofox_tabfm_max_rows = <n>; or pass a smaller relation",
			                            bind.function_name, (unsigned long long)bind.max_rows);
		}
		state.rows->Append(inputs[0].GetValue(i));
	}
	guard.released = true;
}

void GenerateAggCombine(Vector &source_vector, Vector &target_vector, AggregateInputData &aggr_input_data,
                        idx_t count) {
	auto &bind = aggr_input_data.bind_data->Cast<GenerateBindData>();
	StateVectorCleanup source_guard {source_vector, count};
	StateVectorCleanup target_guard {target_vector, count};

	UnifiedVectorFormat source_data, target_data;
	source_vector.ToUnifiedFormat(count, source_data);
	target_vector.ToUnifiedFormat(count, target_data);
	auto sources = reinterpret_cast<GenerateAggState **>(source_data.data);
	auto targets = reinterpret_cast<GenerateAggState **>(target_data.data);

	for (idx_t i = 0; i < count; i++) {
		auto &source = *sources[source_data.sel->get_index(i)];
		auto &target = *targets[target_data.sel->get_index(i)];
		if (!source.rows) {
			continue;
		}
		if (!target.rows) {
			target.rows = source.rows;
			source.rows = nullptr;
			continue;
		}
		target.rows->collection.Combine(source.rows->collection);
		delete source.rows;
		source.rows = nullptr;
		if (target.rows->collection.Count() > bind.max_rows) {
			throw InvalidInputException("%s: input exceeds anofox_tabfm_max_rows (%llu). Raise it with "
			                            "SET anofox_tabfm_max_rows = <n>; or pass a smaller relation",
			                            bind.function_name, (unsigned long long)bind.max_rows);
		}
	}
	source_guard.released = true;
	target_guard.released = true;
}

vector<Value> RowChildren(const Value &row_value, const LogicalType &row_type) {
	if (row_value.IsNull()) {
		vector<Value> children;
		for (auto &field : StructType::GetChildTypes(row_type)) {
			children.emplace_back(field.second);
		}
		return children;
	}
	return StructValue::GetChildren(row_value);
}

//! Build one ColumnPlan per input column, and refuse any column that cannot be
//! a generation target with the reason the plan recorded.
vector<ColumnPlan> PlanAll(const vector<vector<Value>> &rows, const child_list_t<LogicalType> &fields,
                           const TabFMGenerateOptions &opts, const string &fname,
                           const vector<idx_t> *restrict_to) {
	vector<ColumnPlan> plans;
	plans.reserve(fields.size());
	for (idx_t c = 0; c < fields.size(); c++) {
		vector<Value> column;
		column.reserve(rows.size());
		for (auto &row : rows) {
			column.push_back(row[c]);
		}
		plans.push_back(PlanColumn(fields[c].first, fields[c].second, column, opts.bins, opts.max_classes));
	}
	for (idx_t c = 0; c < plans.size(); c++) {
		if (plans[c].kind != GenColumnKind::UNSUPPORTED) {
			continue;
		}
		// An unsupported column is only fatal if we actually have to WRITE it.
		// As a feature it is fine — the preprocessor handles it.
		if (restrict_to) {
			bool needed = false;
			for (auto t : *restrict_to) {
				needed = needed || t == c;
			}
			if (!needed) {
				continue;
			}
		}
		throw InvalidInputException("%s: %s. Exclude it with features := [...] and it will not be generated",
		                            fname, plans[c].unsupported_reason);
	}
	return plans;
}

void GenerateAggFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {
	auto &bind = aggr_input_data.bind_data->Cast<GenerateBindData>();
	StateVectorCleanup guard {state_vector, count};

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = reinterpret_cast<GenerateAggState **>(sdata.data);
	auto list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &fields = StructType::GetChildTypes(bind.row_type);

	for (idx_t i = 0; i < count; i++) {
		auto &state = *states[sdata.sel->get_index(i)];
		const idx_t result_idx = i + offset;
		const idx_t list_start = ListVector::GetListSize(result);

		if (!state.rows || state.rows->collection.Count() == 0) {
			throw InvalidInputException("%s: the input relation is empty — there is nothing to learn from",
			                            bind.function_name);
		}

		vector<vector<Value>> rows;
		rows.reserve(state.rows->collection.Count());
		for (auto &chunk : state.rows->collection.Chunks()) {
			for (idx_t r = 0; r < chunk.size(); r++) {
				rows.push_back(RowChildren(chunk.data[0].GetValue(r), bind.row_type));
			}
		}

		if (bind.is_impute) {
			vector<idx_t> targets = bind.targets;
			if (!bind.targets_explicit) {
				// Default: every column that actually has a NULL to fill.
				for (idx_t c = 0; c < fields.size(); c++) {
					bool has_null = false;
					for (idx_t r = 0; r < rows.size() && !has_null; r++) {
						has_null = rows[r][c].IsNull();
					}
					if (has_null) {
						targets.push_back(c);
					}
				}
			}
			auto plans = PlanAll(rows, fields, bind.options, bind.function_name, &targets);
			auto filled = RunImpute(rows, fields, plans, targets, bind.options, bind.context);
			for (auto &row : filled) {
				child_list_t<Value> outer;
				outer.emplace_back("cols", Value::STRUCT(bind.row_type, row));
				ListVector::PushBack(result, Value::STRUCT(std::move(outer)));
			}
			list_entries[result_idx] = list_entry_t(list_start, filled.size());
			continue;
		}

		// Generation: the chain rule runs over real rows + n synthetic rows, so
		// the per-step tensor is that tall.
		if (rows.size() + bind.n > bind.max_rows) {
			throw InvalidInputException(
			    "%s: %llu input rows + %llu generated rows exceed anofox_tabfm_max_rows (%llu). Raise it with "
			    "SET anofox_tabfm_max_rows = <n>; or generate fewer rows",
			    bind.function_name, (unsigned long long)rows.size(), (unsigned long long)bind.n,
			    (unsigned long long)bind.max_rows);
		}
		auto plans = PlanAll(rows, fields, bind.options, bind.function_name, nullptr);
		auto synthetic = RunChainRule(rows, fields, plans, bind.n, bind.options, bind.context);
		for (idx_t r = 0; r < synthetic.size(); r++) {
			child_list_t<Value> outer;
			outer.emplace_back("cols", Value::STRUCT(bind.row_type, synthetic[r]));
			outer.emplace_back("synthetic_id", Value::BIGINT(NumericCast<int64_t>(r + 1)));
			ListVector::PushBack(result, Value::STRUCT(std::move(outer)));
		}
		list_entries[result_idx] = list_entry_t(list_start, synthetic.size());
	}
	guard.released = true;
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//

AggregateFunction MakeGenerateFunction(const string &name, bool with_opts, bool is_impute) {
	vector<LogicalType> arguments {LogicalType::ANY, LogicalType::ANY};
	if (with_opts) {
		arguments.push_back(LogicalType::ANY);
	}
	return AggregateFunction(name, std::move(arguments), LogicalType::ANY, GenerateStateSize, GenerateStateInitialize,
	                         GenerateAggUpdate, GenerateAggCombine, GenerateAggFinalize,
	                         /*simple_update=*/nullptr, is_impute ? ImputeAggBind : GenerateAggBind,
	                         GenerateStateDestroy);
}

void RegisterGenerateSet(ExtensionLoader &loader, const string &full_name, bool is_impute) {
	AggregateFunctionSet set(full_name);
	set.AddFunction(MakeGenerateFunction(full_name, false, is_impute));
	set.AddFunction(MakeGenerateFunction(full_name, true, is_impute));
	CreateAggregateFunctionInfo info(set);
	loader.RegisterFunction(info);
}

} // anonymous namespace

void RegisterGenerateAggFunctions(ExtensionLoader &loader) {
	RegisterGenerateSet(loader, "__anofox_tabfm_generate_agg", /*is_impute=*/false);
	RegisterGenerateSet(loader, "__anofox_tabfm_impute_agg", /*is_impute=*/true);
}

} // namespace anofox
} // namespace duckdb
