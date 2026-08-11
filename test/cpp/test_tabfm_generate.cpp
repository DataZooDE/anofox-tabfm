// Catch2 tests for tabfm_generate — WS-G, milestone G1.
//
// The pure half of synthetic-data generation: column planning and the sampling
// primitives. No ORT, no engine, no model — everything here is a function of
// its inputs, which is exactly why it is unit-tested rather than left to the
// sqllogictests that drive the real engine.
//
// The invariants that matter downstream:
//   * a sampled value is always a LEGAL value for its column (right type, never
//     outside the observed range, categorical values from the observed domain);
//   * bins tile the observed range without gaps or overlap;
//   * the encode -> proba-key -> decode round trip is lossless, since that is
//     how a sampled class finds its way back to a typed column value.

#include "catch.hpp"

#include "tabfm_generate.hpp"
#include "tabfm_random.hpp"

#include "duckdb/common/types/date.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

vector<Value> Doubles(const std::vector<double> &raw) {
	vector<Value> out;
	for (auto v : raw) {
		out.push_back(Value::DOUBLE(v));
	}
	return out;
}

vector<Value> Ints(const std::vector<int32_t> &raw) {
	vector<Value> out;
	for (auto v : raw) {
		out.push_back(Value::INTEGER(v));
	}
	return out;
}

vector<Value> Strings(const std::vector<std::string> &raw) {
	vector<Value> out;
	for (auto &v : raw) {
		out.push_back(Value(v));
	}
	return out;
}

} // namespace

//===--------------------------------------------------------------------===//
// QuantileEdges / BinOf
//===--------------------------------------------------------------------===//

TEST_CASE("generate: quantile edges tile the observed range", "[tabfm][generate]") {
	vector<double> values;
	for (int i = 0; i < 100; i++) {
		values.push_back(i);
	}
	auto edges = QuantileEdges(values, 10);
	REQUIRE(edges.size() == 11);
	// The outer edges ARE the observed extremes — nothing is ever generated
	// outside them.
	REQUIRE(edges.front() == 0.0);
	REQUIRE(edges.back() == 99.0);
	for (idx_t i = 1; i < edges.size(); i++) {
		INFO("edge " << i);
		REQUIRE(edges[i] > edges[i - 1]); // strictly increasing: no empty bins
	}
}

TEST_CASE("generate: ties collapse instead of producing empty bins", "[tabfm][generate]") {
	// Only three distinct values: asking for 10 bins must not fabricate seven
	// zero-width ones.
	vector<double> values;
	for (int i = 0; i < 30; i++) {
		values.push_back(i % 3 == 0 ? 1.0 : (i % 3 == 1 ? 5.0 : 9.0));
	}
	auto edges = QuantileEdges(values, 10);
	REQUIRE(edges.size() >= 2);
	REQUIRE(edges.size() <= 4);
	for (idx_t i = 1; i < edges.size(); i++) {
		REQUIRE(edges[i] > edges[i - 1]);
	}
	REQUIRE(edges.front() == 1.0);
	REQUIRE(edges.back() == 9.0);
}

TEST_CASE("generate: degenerate columns have no edges", "[tabfm][generate]") {
	REQUIRE(QuantileEdges({}, 10).empty());
	REQUIRE(QuantileEdges({5.0}, 10).empty());
	REQUIRE(QuantileEdges({5.0, 5.0, 5.0}, 10).empty()); // constant
	REQUIRE(QuantileEdges({1.0, 2.0}, 1).empty());       // bins < 2
}

TEST_CASE("generate: BinOf covers every value including the extremes", "[tabfm][generate]") {
	vector<double> edges = {0.0, 10.0, 20.0, 30.0};
	REQUIRE(BinOf(0.0, edges) == 0);
	REQUIRE(BinOf(5.0, edges) == 0);
	REQUIRE(BinOf(10.0, edges) == 1);
	REQUIRE(BinOf(19.9, edges) == 1);
	REQUIRE(BinOf(20.0, edges) == 2);
	// The maximum belongs to the LAST bin, not a phantom bin past the end.
	REQUIRE(BinOf(30.0, edges) == 2);
	// Out of range (possible on an imputed feature) clamps rather than escapes.
	REQUIRE(BinOf(-100.0, edges) == 0);
	REQUIRE(BinOf(1e9, edges) == 2);
	REQUIRE(BinOf(0.0, {}) == 0);
}

//===--------------------------------------------------------------------===//
// ExpandBin: values are legal for their column
//===--------------------------------------------------------------------===//

TEST_CASE("generate: ExpandBin stays inside its bin", "[tabfm][generate]") {
	// 0..99 in 10 bins, so bin 1 spans [10, 20).
	vector<Value> values;
	for (int i = 0; i < 100; i++) {
		values.push_back(Value::DOUBLE(i));
	}
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 10);
	TabFMRandom rng(42);
	for (int i = 0; i < 2000; i++) {
		auto v = ExpandBin(plan, 1, rng);
		REQUIRE_FALSE(v.IsNull());
		auto d = DoubleValue::Get(v);
		REQUIRE(d >= plan.edges[1]);
		REQUIRE(d <= plan.edges[2]);
	}
	// An out-of-range bin index clamps to the last bin instead of reading past
	// the edge array.
	auto v = ExpandBin(plan, 99, rng);
	REQUIRE(DoubleValue::Get(v) >= plan.edges[plan.BinCount() - 1]);
}

TEST_CASE("generate: ExpandBin rounds integer columns instead of truncating", "[tabfm][generate]") {
	// A column of 3s and 4s: one bin spanning [3, 4]. Truncation would make every
	// draw a 3; rounding must produce both, and never anything outside.
	vector<Value> values;
	for (int i = 0; i < 40; i++) {
		values.push_back(Value::INTEGER(i % 2 == 0 ? 3 : 4));
	}
	auto plan = PlanColumn("n", LogicalType::INTEGER, values, 10);
	REQUIRE(plan.kind == GenColumnKind::CONTINUOUS);
	TabFMRandom rng(1);
	bool saw_three = false, saw_four = false;
	for (int i = 0; i < 500; i++) {
		auto v = ExpandBin(plan, 0, rng);
		REQUIRE(v.type().id() == LogicalTypeId::INTEGER);
		auto n = IntegerValue::Get(v);
		REQUIRE(n >= 3);
		REQUIRE(n <= 4);
		saw_three = saw_three || n == 3;
		saw_four = saw_four || n == 4;
	}
	REQUIRE(saw_three);
	REQUIRE(saw_four);
}

TEST_CASE("generate: ExpandBin keeps integers within a wide bin", "[tabfm][generate]") {
	vector<Value> values;
	for (int i = -5; i <= 5; i++) {
		values.push_back(Value::BIGINT(i));
	}
	auto plan = PlanColumn("n", LogicalType::BIGINT, values, 2);
	TabFMRandom rng(9);
	for (int i = 0; i < 1000; i++) {
		for (idx_t b = 0; b < plan.BinCount(); b++) {
			auto n = BigIntValue::Get(ExpandBin(plan, b, rng));
			REQUIRE(n >= -5);
			REQUIRE(n <= 5);
		}
	}
}

TEST_CASE("generate: ExpandBin follows the bin's shape, not its span",
          "[tabfm][generate]") {
	// REGRESSION GUARD for a defect the breast-cancer benchmark exposed.
	//
	// A long-tailed column: 99 values packed into [0, 10] and one outlier at
	// 1000. The top quantile bin therefore SPANS a huge range while the real
	// mass sits at its bottom. Drawing uniformly across that span — the obvious
	// implementation — put ~half the draws above 500 and inflated the column
	// mean by 67% on the real benchmark. Sampling through the empirical
	// distribution instead has to keep the draws near the observations.
	vector<Value> values;
	for (int i = 0; i < 99; i++) {
		values.push_back(Value::DOUBLE(i / 10.0));
	}
	values.push_back(Value::DOUBLE(1000.0));
	auto plan = PlanColumn("skewed", LogicalType::DOUBLE, values, 10);
	REQUIRE(plan.kind == GenColumnKind::CONTINUOUS);

	TabFMRandom rng(42);
	double sum = 0;
	const int kDraws = 5000;
	for (int i = 0; i < kDraws; i++) {
		sum += DoubleValue::Get(DrawMarginal(plan, rng));
	}
	// mean = (sum of 0.0..9.8 step 0.1, i.e. 485.1, plus 1000) / 100
	const double real_mean = 14.851;
	const double sampled_mean = sum / kDraws;
	INFO("sampled mean " << sampled_mean << " vs real " << real_mean);

	// Where the threshold comes from. The top bin here holds 9.0..9.8 AND the
	// 1000 outlier, so it spans [9, 1000]:
	//   uniform across the span   -> ~54.5   (3.7x the truth)
	//   empirical + interpolation -> ~20.9   (1.4x)
	// The residual overshoot is inherent to linear interpolation of the empirical
	// CDF: interpolating between two adjacent order statistics bridges the gap
	// between 9.8 and 1000. Returning the observation verbatim instead would be
	// exact, but would copy real values into the output and break the novelty
	// property the memorization check relies on — so the smoothing is a
	// deliberate trade, not an oversight. 2x fails the old behaviour by a wide
	// margin while leaving room for RNG wobble.
	REQUIRE(sampled_mean < real_mean * 2.0);
}

//===--------------------------------------------------------------------===//
// PlanColumn
//===--------------------------------------------------------------------===//

TEST_CASE("generate: numeric columns plan as continuous", "[tabfm][generate]") {
	auto plan = PlanColumn("x", LogicalType::DOUBLE, Doubles({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}), 5);
	REQUIRE(plan.kind == GenColumnKind::CONTINUOUS);
	REQUIRE(plan.BinCount() >= 2);
	REQUIRE(plan.BinCount() <= 5);
	REQUIRE(plan.bin_counts.size() == plan.BinCount());
	// Every observed row is accounted for by exactly one bin.
	idx_t total = 0;
	for (auto c : plan.bin_counts) {
		total += c;
	}
	REQUIRE(total == 10);
	REQUIRE(plan.observed_count == 10);
	REQUIRE(plan.null_count == 0);
}

TEST_CASE("generate: string columns plan as categorical with counts", "[tabfm][generate]") {
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings({"a", "b", "a", "c", "a", "b"}), 10);
	REQUIRE(plan.kind == GenColumnKind::CATEGORICAL);
	REQUIRE(plan.labels.size() == 3);
	// First-appearance order, so the plan is stable for a given input.
	REQUIRE(plan.labels[0].ToString() == "a");
	REQUIRE(plan.labels[1].ToString() == "b");
	REQUIRE(plan.labels[2].ToString() == "c");
	REQUIRE(plan.label_counts[0] == 3);
	REQUIRE(plan.label_counts[1] == 2);
	REQUIRE(plan.label_counts[2] == 1);
}

TEST_CASE("generate: booleans are categorical, not continuous", "[tabfm][generate]") {
	vector<Value> values = {Value::BOOLEAN(true), Value::BOOLEAN(false), Value::BOOLEAN(true)};
	auto plan = PlanColumn("flag", LogicalType::BOOLEAN, values, 10);
	REQUIRE(plan.kind == GenColumnKind::CATEGORICAL);
	REQUIRE(plan.labels.size() == 2);
}

TEST_CASE("generate: NULLs are counted, never modelled", "[tabfm][generate]") {
	vector<Value> values = Strings({"a", "b"});
	values.push_back(Value(LogicalType::VARCHAR));
	values.push_back(Value(LogicalType::VARCHAR));
	auto plan = PlanColumn("g", LogicalType::VARCHAR, values, 10);
	REQUIRE(plan.null_count == 2);
	REQUIRE(plan.observed_count == 2);
	REQUIRE(plan.labels.size() == 2);
}

TEST_CASE("generate: single-valued and all-NULL columns are constant", "[tabfm][generate]") {
	auto single = PlanColumn("c", LogicalType::VARCHAR, Strings({"only", "only"}), 10);
	REQUIRE(single.kind == GenColumnKind::CONSTANT);
	REQUIRE(single.constant_value.ToString() == "only");

	auto constant_number = PlanColumn("n", LogicalType::INTEGER, Ints({7, 7, 7}), 10);
	REQUIRE(constant_number.kind == GenColumnKind::CONSTANT);
	REQUIRE(IntegerValue::Get(constant_number.constant_value) == 7);

	vector<Value> all_null = {Value(LogicalType::INTEGER), Value(LogicalType::INTEGER)};
	auto empty = PlanColumn("e", LogicalType::INTEGER, all_null, 10);
	REQUIRE(empty.kind == GenColumnKind::CONSTANT);
	REQUIRE(empty.constant_value.IsNull());
}

TEST_CASE("generate: high-cardinality categoricals are unsupported, with a reason",
          "[tabfm][generate]") {
	std::vector<std::string> many;
	for (int i = 0; i < 25; i++) {
		many.push_back("label_" + std::to_string(i));
	}
	auto plan = PlanColumn("id", LogicalType::VARCHAR, Strings(many), 10);
	REQUIRE(plan.kind == GenColumnKind::UNSUPPORTED);
	REQUIRE_FALSE(plan.unsupported_reason.empty());
	// The message has to name the column and the limit — it becomes user-facing.
	REQUIRE(plan.unsupported_reason.find("id") != std::string::npos);
	REQUIRE(plan.unsupported_reason.find("25") != std::string::npos);
}

TEST_CASE("generate: temporal columns are unsupported targets", "[tabfm][generate]") {
	vector<Value> values = {Value::DATE(date_t(1)), Value::DATE(date_t(2))};
	auto plan = PlanColumn("ts", LogicalType::DATE, values, 10);
	REQUIRE(plan.kind == GenColumnKind::UNSUPPORTED);
	REQUIRE(plan.unsupported_reason.find("ts") != std::string::npos);
}

TEST_CASE("generate: bins are capped at the engine's class limit", "[tabfm][generate]") {
	vector<double> raw;
	for (int i = 0; i < 500; i++) {
		raw.push_back(i);
	}
	vector<Value> values;
	for (auto v : raw) {
		values.push_back(Value::DOUBLE(v));
	}
	// Asking for 50 bins must not exceed the <= 10 labels the engine accepts.
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 50);
	REQUIRE(plan.kind == GenColumnKind::CONTINUOUS);
	REQUIRE(plan.ClassCount() <= kMaxGenerateBins);
}

//===--------------------------------------------------------------------===//
// Marginal draws
//===--------------------------------------------------------------------===//

TEST_CASE("generate: categorical marginal follows the observed shares", "[tabfm][generate]") {
	// 'a' three times as common as 'b'; the first column of the chain has no
	// features, so this draw IS the model for it.
	std::vector<std::string> raw;
	for (int i = 0; i < 300; i++) {
		raw.push_back(i % 4 == 0 ? "b" : "a");
	}
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings(raw), 10);
	TabFMRandom rng(42);
	int a = 0, b = 0;
	for (int i = 0; i < 4000; i++) {
		auto v = DrawMarginal(plan, rng);
		REQUIRE_FALSE(v.IsNull());
		if (v.ToString() == "a") {
			a++;
		} else if (v.ToString() == "b") {
			b++;
		} else {
			FAIL("drew a label that was never observed: " << v.ToString());
		}
	}
	REQUIRE((double)a / 4000 == Approx(0.75).margin(0.03));
	REQUIRE((double)b / 4000 == Approx(0.25).margin(0.03));
}

TEST_CASE("generate: continuous marginal stays in the observed range", "[tabfm][generate]") {
	vector<Value> values;
	for (int i = 0; i < 200; i++) {
		values.push_back(Value::DOUBLE(-3.0 + 0.05 * i)); // [-3, 6.95]
	}
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 10);
	TabFMRandom rng(7);
	double lo = 1e18, hi = -1e18;
	for (int i = 0; i < 3000; i++) {
		auto d = DoubleValue::Get(DrawMarginal(plan, rng));
		lo = std::min(lo, d);
		hi = std::max(hi, d);
	}
	REQUIRE(lo >= plan.edges.front());
	REQUIRE(hi <= plan.edges.back());
	// It should also actually spread out, not collapse onto one bin.
	REQUIRE(hi - lo > (plan.edges.back() - plan.edges.front()) * 0.5);
}

TEST_CASE("generate: constant columns draw their constant", "[tabfm][generate]") {
	auto plan = PlanColumn("c", LogicalType::INTEGER, Ints({4, 4, 4}), 10);
	TabFMRandom rng(1);
	for (int i = 0; i < 20; i++) {
		REQUIRE(IntegerValue::Get(DrawMarginal(plan, rng)) == 4);
	}
}

//===--------------------------------------------------------------------===//
// Encode / decode round trip
//===--------------------------------------------------------------------===//

TEST_CASE("generate: encoding a target is a lossless round trip", "[tabfm][generate]") {
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings({"x", "y", "z", "x"}), 10);
	// Every label must survive encode -> key -> decode, because that path is how
	// a sampled class becomes a column value.
	for (idx_t i = 0; i < plan.labels.size(); i++) {
		auto encoded = EncodeTarget(plan, plan.labels[i]);
		REQUIRE(encoded.type().id() == LogicalTypeId::VARCHAR);
		REQUIRE(encoded.ToString() == TargetKey(plan, i));
		REQUIRE(DecodePointEstimate(plan, encoded).ToString() == plan.labels[i].ToString());
	}
	// A NULL target is the engine's "predict me" marker and must stay NULL.
	auto null_encoded = EncodeTarget(plan, Value(LogicalType::VARCHAR));
	REQUIRE(null_encoded.IsNull());
}

TEST_CASE("generate: continuous targets encode to their bin index", "[tabfm][generate]") {
	vector<Value> values;
	for (int i = 0; i < 100; i++) {
		values.push_back(Value::DOUBLE(i));
	}
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 10);
	auto low = EncodeTarget(plan, Value::DOUBLE(0.0));
	auto high = EncodeTarget(plan, Value::DOUBLE(99.0));
	REQUIRE(low.ToString() == "0");
	REQUIRE(high.ToString() == to_string(plan.BinCount() - 1));
	// Typed NULL of the ENCODED type, not the column type.
	REQUIRE(EncodeTarget(plan, Value(LogicalType::DOUBLE)).IsNull());
}

TEST_CASE("generate: integer categorical decodes back to INTEGER", "[tabfm][generate]") {
	// The column type must survive the VARCHAR encoding used for the engine.
	auto plan = PlanColumn("k", LogicalType::INTEGER, Ints({10, 20, 30, 10}), 10);
	REQUIRE(plan.kind == GenColumnKind::CONTINUOUS); // numeric -> binned
	auto decoded = DecodePointEstimate(plan, Value::DOUBLE(21.0));
	REQUIRE(decoded.type().id() == LogicalTypeId::INTEGER);
	REQUIRE(IntegerValue::Get(decoded) >= 10);
	REQUIRE(IntegerValue::Get(decoded) <= 30);
}

TEST_CASE("generate: point estimates are clamped to the observed range", "[tabfm][generate]") {
	vector<Value> values;
	for (int i = 0; i < 50; i++) {
		values.push_back(Value::DOUBLE(i));
	}
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 10);
	// A wild regression output must not escape the column's observed support.
	REQUIRE(DoubleValue::Get(DecodePointEstimate(plan, Value::DOUBLE(1e9))) == Approx(49.0));
	REQUIRE(DoubleValue::Get(DecodePointEstimate(plan, Value::DOUBLE(-1e9))) == Approx(0.0));
	REQUIRE(DecodePointEstimate(plan, Value(LogicalType::DOUBLE)).IsNull());
}

//===--------------------------------------------------------------------===//
// Sampling from the engine's proba MAP
//===--------------------------------------------------------------------===//

namespace {

Value ProbaMap(const std::vector<std::pair<std::string, double>> &entries) {
	vector<Value> keys, values;
	for (auto &e : entries) {
		keys.push_back(Value(e.first));
		values.push_back(Value::DOUBLE(e.second));
	}
	return Value::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE, std::move(keys), std::move(values));
}

} // namespace

TEST_CASE("generate: sampling follows the model's probabilities", "[tabfm][generate]") {
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings({"a", "b", "c", "a"}), 10);
	auto proba = ProbaMap({{"a", 0.7}, {"b", 0.2}, {"c", 0.1}});
	TabFMRandom rng(42);
	std::map<std::string, int> counts;
	const int kDraws = 6000;
	for (int i = 0; i < kDraws; i++) {
		auto v = SampleFromProba(plan, proba, Value("a"), rng);
		REQUIRE(v.type().id() == LogicalTypeId::VARCHAR);
		counts[v.ToString()]++;
	}
	REQUIRE((double)counts["a"] / kDraws == Approx(0.7).margin(0.03));
	REQUIRE((double)counts["b"] / kDraws == Approx(0.2).margin(0.03));
	REQUIRE((double)counts["c"] / kDraws == Approx(0.1).margin(0.03));
}

TEST_CASE("generate: sampling is keyed, not positional", "[tabfm][generate]") {
	// The engine builds its label decoder from the values it saw, so the proba
	// MAP need not arrive in plan order. Reversing it must not permute labels.
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings({"a", "b", "c", "a"}), 10);
	auto reversed = ProbaMap({{"c", 1.0}, {"b", 0.0}, {"a", 0.0}});
	TabFMRandom rng(3);
	for (int i = 0; i < 50; i++) {
		REQUIRE(SampleFromProba(plan, reversed, Value("a"), rng).ToString() == "c");
	}
}

TEST_CASE("generate: sampling falls back to the argmax without a distribution",
          "[tabfm][generate]") {
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings({"a", "b", "c", "a"}), 10);
	TabFMRandom rng(3);
	// NULL proba (compact mode) and an all-zero map both fall back rather than
	// inventing a uniform distribution.
	REQUIRE(SampleFromProba(plan, Value(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)),
	                        Value("b"), rng)
	            .ToString() == "b");
	REQUIRE(SampleFromProba(plan, ProbaMap({{"a", 0.0}, {"b", 0.0}}), Value("c"), rng).ToString() == "c");
	// Unknown keys are ignored, not mapped onto some arbitrary label.
	REQUIRE(SampleFromProba(plan, ProbaMap({{"zzz", 1.0}}), Value("a"), rng).ToString() == "a");
}

TEST_CASE("generate: continuous fallback expands the bin, not the index", "[tabfm][generate]") {
	// Regression guard. On a generation step the engine's yhat is the winning
	// ENCODED key, and for a continuous column that key is a BIN INDEX. Reading
	// "3" as the number 3.0 would emit a value in the wrong units entirely —
	// here the column lives in [100, 199], so a leaked index would be obvious.
	vector<Value> values;
	for (int i = 100; i < 200; i++) {
		values.push_back(Value::DOUBLE(i));
	}
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 10);
	TabFMRandom rng(5);
	// Empty distribution forces the fallback path; yhat is the bin key "3".
	for (int i = 0; i < 200; i++) {
		auto v = SampleFromProba(plan, Value(LogicalType::MAP(LogicalType::VARCHAR, LogicalType::DOUBLE)),
		                         Value("3"), rng);
		auto d = DoubleValue::Get(v);
		REQUIRE(d >= plan.edges[3]);
		REQUIRE(d <= plan.edges[4]);
	}
}

TEST_CASE("generate: continuous sampling expands the sampled bin", "[tabfm][generate]") {
	vector<Value> values;
	for (int i = 0; i < 100; i++) {
		values.push_back(Value::DOUBLE(i));
	}
	auto plan = PlanColumn("x", LogicalType::DOUBLE, values, 10);
	// Force all mass onto the top bin; every draw must land inside it.
	const idx_t top = plan.BinCount() - 1;
	auto proba = ProbaMap({{to_string(top), 1.0}});
	TabFMRandom rng(11);
	for (int i = 0; i < 500; i++) {
		auto d = DoubleValue::Get(SampleFromProba(plan, proba, Value::DOUBLE(0.0), rng));
		REQUIRE(d >= plan.edges[top]);
		REQUIRE(d <= plan.edges.back());
	}
}

TEST_CASE("generate: sampling is deterministic per seed", "[tabfm][generate]") {
	auto plan = PlanColumn("g", LogicalType::VARCHAR, Strings({"a", "b", "c", "a"}), 10);
	auto proba = ProbaMap({{"a", 0.5}, {"b", 0.3}, {"c", 0.2}});
	std::vector<std::string> first, second;
	{
		TabFMRandom rng(99);
		for (int i = 0; i < 100; i++) {
			first.push_back(SampleFromProba(plan, proba, Value("a"), rng).ToString());
		}
	}
	{
		TabFMRandom rng(99);
		for (int i = 0; i < 100; i++) {
			second.push_back(SampleFromProba(plan, proba, Value("a"), rng).ToString());
		}
	}
	REQUIRE(first == second);
}
