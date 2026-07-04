// Catch2 tests for tabfm_preprocess — WS-F.
//
// Red-green parity against test/fixtures/golden_preprocess.json (produced by
// UNMODIFIED upstream vendor/tabfm code). Golden values are transcribed inline
// (the C++ test target's include dirs do not carry a JSON parser, and the
// CMake source list is scaffold-owned). Every intermediate stage documented in
// the fixture's "_docs" key is asserted: first-appearance ordinal encoding
// (min_frequency=2, -1 unknowns), mean imputation, datetime expansion + NaT
// train-mean fill, unique-feature filter, z-score scaling, and the two-stage
// outlier bounds, plus the final x / y / cat_mask / d / train_size tensors and
// label/target decoders. Tolerance: rtol 1e-6 (Approx epsilon), abs margin
// 1e-9 near zero.

#include "catch.hpp"

#include "tabfm_preprocess.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/date.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

Value VDouble(double v) {
	return Value::DOUBLE(v);
}
Value VNullDouble() {
	return Value(LogicalType::DOUBLE);
}
Value VBigint(int64_t v) {
	return Value::BIGINT(v);
}
Value VStr(const char *s) {
	return Value(string(s));
}
Value VNullStr() {
	return Value(LogicalType::VARCHAR);
}
Value VBool(bool b) {
	return Value::BOOLEAN(b);
}
Value VNullBool() {
	return Value(LogicalType::BOOLEAN);
}
Value VDate(int32_t y, int32_t m, int32_t d) {
	return Value::DATE(Date::FromDate(y, m, d));
}
Value VNullDate() {
	return Value(LogicalType::DATE);
}

// Build a ColumnDataCollection from a row-major grid of Values.
unique_ptr<ColumnDataCollection> MakeCollection(const vector<LogicalType> &types,
                                                const std::vector<std::vector<Value>> &rows) {
	auto collection =
	    make_uniq<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), types);
	for (idx_t r = 0; r < rows.size(); r++) {
		for (idx_t c = 0; c < types.size(); c++) {
			chunk.SetValue(c, r, rows[r][c]);
		}
	}
	chunk.SetCardinality(rows.size());
	collection->Append(chunk);
	return collection;
}

void CheckVec(const vector<double> &actual, const std::vector<double> &expected,
              double eps = 1e-6, double margin = 1e-9) {
	REQUIRE(actual.size() == expected.size());
	for (size_t i = 0; i < expected.size(); i++) {
		INFO("index " << i << " actual=" << actual[i] << " expected=" << expected[i]);
		REQUIRE(actual[i] == Approx(expected[i]).epsilon(eps).margin(margin));
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

TEST_CASE("preprocess: NaN numeric feature is imputed like NULL (never poisons stats)", "[tabfm][preprocess]") {
	// A single non-NULL NaN in a numeric column would, if summed into the mean,
	// turn the imputation mean and then the z-score statistics NaN, poisoning the
	// whole feature column across context AND query rows.
	vector<LogicalType> types = {LogicalType::DOUBLE, LogicalType::VARCHAR};
	std::vector<std::vector<Value>> rows = {
	    {VDouble(1.0), VStr("a")},
	    {Value::DOUBLE(std::numeric_limits<double>::quiet_NaN()), VStr("b")}, // NaN context value
	    {VDouble(3.0), VStr("a")},
	    {VDouble(5.0), VStr("b")},
	    {VDouble(2.0), VNullStr()}, // query row (NULL target)
	};
	auto data = MakeCollection(types, rows);
	vector<PreprocessColumnSpec> cols = {
	    {"num", LogicalType::DOUBLE, false, true},
	    {"target", LogicalType::VARCHAR, true, false},
	};
	auto batch = PreprocessBatch(*data, cols, PreprocessTask::CLASSIFICATION);
	for (double v : batch.x) {
		INFO("x contains a non-finite value — NaN leaked through preprocessing");
		REQUIRE(std::isfinite(v));
	}
}

TEST_CASE("preprocess: profile id and task inference", "[tabfm][preprocess]") {
	REQUIRE(string(kPreprocessProfileId) == "tabfm_v1_minimal");
	REQUIRE(InferTask(LogicalType::VARCHAR) == PreprocessTask::CLASSIFICATION);
	REQUIRE(InferTask(LogicalType::BOOLEAN) == PreprocessTask::CLASSIFICATION);
	REQUIRE(InferTask(LogicalType::DOUBLE) == PreprocessTask::REGRESSION);
	REQUIRE(InferTask(LogicalType::BIGINT) == PreprocessTask::REGRESSION);
}

TEST_CASE("preprocess: mixed_types golden parity", "[tabfm][preprocess]") {
	vector<LogicalType> types = {LogicalType::DOUBLE, LogicalType::BIGINT,
	                             LogicalType::VARCHAR, LogicalType::BOOLEAN,
	                             LogicalType::DATE, LogicalType::VARCHAR};
	// num_a, num_b, cat_c, bool_d, date_e, target(classification)
	std::vector<std::vector<Value>> rows = {
	    {VDouble(1.5), VBigint(10), VStr("red"), VBool(true), VDate(2023, 1, 15), VStr("cat")},
	    {VNullDouble(), VBigint(12), VStr("blue"), VBool(false), VDate(2023, 6, 30), VStr("dog")},
	    {VDouble(3.25), VBigint(11), VStr("red"), VBool(true), VNullDate(), VStr("bird")},
	    {VDouble(4.0), VBigint(15), VStr("green"), VNullBool(), VDate(2024, 2, 29), VStr("cat")},
	    {VDouble(2.75), VBigint(14), VStr("blue"), VBool(false), VDate(2023, 11, 5), VStr("dog")},
	    {VDouble(5.5), VBigint(13), VStr("red"), VBool(true), VDate(2024, 7, 4), VStr("cat")},
	    {VDouble(1.25), VBigint(10), VNullStr(), VBool(true), VDate(2023, 3, 10), VStr("bird")},
	    {VDouble(3.0), VBigint(12), VStr("violet"), VBool(false), VDate(2024, 12, 1), VStr("dog")},
	    {VDouble(2.0), VBigint(11), VStr("green"), VBool(false), VDate(2024, 3, 15), VNullStr()},
	    {VNullDouble(), VBigint(14), VStr("red"), VBool(true), VNullDate(), VNullStr()},
	    {VDouble(4.5), VBigint(15), VStr("purple"), VNullBool(), VDate(2023, 8, 20), VNullStr()},
	    {VDouble(3.5), VBigint(10), VNullStr(), VBool(false), VDate(2024, 5, 5), VNullStr()},
	};
	auto data = MakeCollection(types, rows);
	vector<PreprocessColumnSpec> cols = {
	    {"num_a", LogicalType::DOUBLE, false, true},
	    {"num_b", LogicalType::BIGINT, false, true},
	    {"cat_c", LogicalType::VARCHAR, false, true},
	    {"bool_d", LogicalType::BOOLEAN, false, true},
	    {"date_e", LogicalType::DATE, false, true},
	    {"target", LogicalType::VARCHAR, true, false},
	};

	auto batch = PreprocessBatch(*data, cols, PreprocessTask::CLASSIFICATION);

	REQUIRE(batch.train_size == 8);
	REQUIRE(batch.T == 12);
	REQUIRE(batch.H == 9);
	REQUIRE(batch.d == 9);

	// encoded column order
	std::vector<string> exp_names = {"cat_c",          "bool_d",       "num_a",
	                                 "num_b",          "date_e.epoch_ns", "date_e.year",
	                                 "date_e.month",   "date_e.day",   "date_e.dayofweek"};
	REQUIRE(batch.stages.encoded_column_names.size() == exp_names.size());
	for (size_t i = 0; i < exp_names.size(); i++) {
		REQUIRE(batch.stages.encoded_column_names[i] == exp_names[i]);
	}

	// encoded train matrix (8x9)
	std::vector<double> exp_train = {
	    0, 0, 1.5, 10, 1.6737408e18, 2023, 1, 15, 6,
	    1, 1, 3.0357142857142856, 12, 1.6880832e18, 2023, 6, 30, 4,
	    0, 0, 3.25, 11, 1.7002285714285714e18, 2023, 11, 17, 4,
	    -1, -1, 4.0, 15, 1.7091648e18, 2024, 2, 29, 3,
	    1, 1, 2.75, 14, 1.6991424e18, 2023, 11, 5, 6,
	    0, 0, 5.5, 13, 1.7200512e18, 2024, 7, 4, 3,
	    -1, 0, 1.25, 10, 1.6784064e18, 2023, 3, 10, 4,
	    -1, 1, 3.0, 12, 1.7330112e18, 2024, 12, 1, 6};
	CheckVec(batch.stages.encoded_train, exp_train);

	std::vector<double> exp_test = {
	    -1, 1, 2.0, 11, 1.7104608e18, 2024, 3, 15, 4,
	    0, 0, 3.0357142857142856, 14, 1.7002285714285714e18, 2023, 11, 17, 4,
	    -1, -1, 4.5, 15, 1.6924896e18, 2023, 8, 20, 6,
	    -1, 1, 3.5, 10, 1.7148672e18, 2024, 5, 5, 6};
	CheckVec(batch.stages.encoded_test, exp_test);

	// unique filter keeps all; cat indices
	CheckMask(batch.stages.unique_filter_keep,
	          {true, true, true, true, true, true, true, true, true});
	REQUIRE(batch.stages.cat_feature_indices_kept ==
	        vector<int64_t>({0, 1}));

	// scaler
	CheckVec(batch.stages.scaler_mean,
	         {-0.125, 0.25, 3.0357142857142856, 12.125, 1.7002285714285714e18,
	          2023.375, 6.625, 13.875, 4.5});
	CheckVec(batch.stages.scaler_scale,
	         {0.7806257497997998, 0.6614388277661477, 1.2588979094296118,
	          1.6909696573085853, 1.8953866812718868e16, 0.4841239182759271,
	          4.090768042988393, 10.349366922606078, 1.2247458713915889});

	// outlier bounds
	CheckVec(batch.stages.outlier_lower,
	         {-4.2761743927115035, -4.276173405632279, -4.276176473834178,
	          -4.2761773417657905, -4.276179870598789, -4.2761710377778765,
	          -4.276178825274337, -4.276179457416042, -4.276176379115398});
	CheckVec(batch.stages.outlier_upper,
	         {4.2761743927115035, 4.276173405632279, 4.276176473834178,
	          4.2761773417657905, 4.276179870598792, 4.2761710377778765,
	          4.276178825274337, 4.276179457416042, 4.276176379115398});

	// final x (12x9)
	std::vector<double> exp_x = {
	    0.16012794867714478, -0.37796390158151966, -1.2198878671663655, -1.2566754174538144, -1.3974864174310304, -0.7745950692447884, -1.3750474094079452, 0.10870230115647628, 1.2247438713924057,
	    1.441151538094303, 1.1338917047445591, 0.0, -0.07392208337963614, -0.6407859435005273, -0.7745950692447884, -0.1527830454897717, 1.55806631657616, -0.4082479571308019,
	    0.16012794867714478, -0.37796390158151966, 0.17021691169763253, -0.6652987504167253, 0.0, -0.7745950692447884, 1.069481318428402, 0.30195083654576743, -0.4082479571308019,
	    -1.1208956407400135, -1.8898195079075983, 0.765976102639346, 1.7002079177316314, 0.47147258444551327, 1.2909917820746473, -1.1305945366243106, 1.4614420488815143, -1.2247438713924057,
	    1.441151538094303, 1.1338917047445591, -0.2269558822635098, 1.1088312506945421, -0.05730605998785028, -0.7745950692447884, 1.069481318428402, -0.8575403757899794, 1.2247438713924057,
	    0.16012794867714478, -0.37796390158151966, 1.9574944845227729, 0.517454583657453, 1.0458355947783047, 1.2909917820746473, 0.09166982729386301, -0.954164643484625, -1.2247438713924057,
	    -1.1208956407400135, -0.37796390158151966, -1.4184742641469368, -1.2566754174538144, -1.1513308415741197, -0.7745950692447884, -0.8861416638406758, -0.3744190373167516, -0.4082479571308019,
	    -1.1208956407400135, 1.1338917047445591, -0.028369485282938638, -0.07392208337963614, 1.729601083269723, 1.2909917820746473, 1.3139341912120366, -1.2440374465685617, 1.2247438713924057,
	    -1.1208956407400135, 1.1338917047445591, -0.8227150732052233, -0.6652987504167253, 0.5398491332946551, 1.2909917820746473, -0.8861416638406758, 0.10870230115647628, -0.4082479571308019,
	    0.16012794867714478, -0.37796390158151966, 0.0, 1.1088312506945421, 0.0, -0.7745950692447884, 1.069481318428402, 0.30195083654576743, -0.4082479571308019,
	    -1.1208956407400135, -1.8898195079075983, 1.1631488966004884, 1.7002079177316314, -0.40830567741344503, -0.7745950692447884, 0.3361227000774977, 0.5918236396297042, 1.2247438713924057,
	    -1.1208956407400135, 1.1338917047445591, 0.36880330867820366, -1.2566754174538144, 0.7723293993817374, 1.2909917820746473, -0.39723591827340643, -0.8575403757899794, 1.2247438713924057};
	CheckVec(batch.x, exp_x);

	// y padded
	CheckVec(batch.y, {1, 2, 0, 1, 2, 1, 0, 2, -100, -100, -100, -100});
	REQUIRE(batch.y_train == vector<int64_t>({1, 2, 0, 1, 2, 1, 0, 2}));
	CheckMask(batch.cat_mask,
	          {true, true, false, false, false, false, false, false, false});

	// label decode
	REQUIRE(batch.label_decoder.size() == 3);
	REQUIRE(batch.label_decoder[0].ToString() == "bird");
	REQUIRE(batch.label_decoder[1].ToString() == "cat");
	REQUIRE(batch.label_decoder[2].ToString() == "dog");
}

TEST_CASE("preprocess: constant_and_rare golden parity", "[tabfm][preprocess]") {
	vector<LogicalType> types = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                             LogicalType::VARCHAR, LogicalType::VARCHAR};
	// num_const, num_x, cat_y, target
	std::vector<std::vector<Value>> rows = {
	    {VDouble(7), VDouble(0.5), VStr("A"), VStr("no")},
	    {VDouble(7), VDouble(1.5), VStr("A"), VStr("yes")},
	    {VDouble(7), VNullDouble(), VStr("B"), VStr("yes")},
	    {VDouble(7), VDouble(2.5), VStr("B"), VStr("no")},
	    {VDouble(7), VDouble(0.75), VStr("A"), VStr("no")},
	    {VDouble(7), VDouble(1.25), VStr("B"), VStr("yes")},
	    {VDouble(7), VDouble(2.0), VStr("A"), VStr("no")},
	    {VDouble(7), VDouble(0.25), VStr("A"), VStr("no")},
	    {VDouble(7), VDouble(1.0), VStr("B"), VNullStr()},
	    {VDouble(7), VNullDouble(), VStr("A"), VNullStr()},
	    {VDouble(7), VDouble(2.25), VStr("C"), VNullStr()},
	    {VDouble(7), VDouble(0.5), VStr("B"), VNullStr()},
	};
	auto data = MakeCollection(types, rows);
	vector<PreprocessColumnSpec> cols = {
	    {"num_const", LogicalType::DOUBLE, false, true},
	    {"num_x", LogicalType::DOUBLE, false, true},
	    {"cat_y", LogicalType::VARCHAR, false, true},
	    {"target", LogicalType::VARCHAR, true, false},
	};
	auto batch = PreprocessBatch(*data, cols, PreprocessTask::CLASSIFICATION);

	REQUIRE(batch.H == 2);
	REQUIRE(batch.d == 2);
	REQUIRE(batch.train_size == 8);

	// encoded order: cat_y, num_const, num_x
	std::vector<double> exp_train = {0, 7, 0.5,  0, 7, 1.5,  1, 7, 1.25, 1, 7, 2.5,
	                                 0, 7, 0.75, 1, 7, 1.25, 0, 7, 2.0,  0, 7, 0.25};
	CheckVec(batch.stages.encoded_train, exp_train);
	std::vector<double> exp_test = {1, 7, 1.0, 0, 7, 1.25, -1, 7, 2.25, 1, 7, 0.5};
	CheckVec(batch.stages.encoded_test, exp_test);

	CheckMask(batch.stages.unique_filter_keep, {true, false, true});
	REQUIRE(batch.stages.cat_feature_indices_kept == vector<int64_t>({0}));

	CheckVec(batch.stages.scaler_mean, {0.375, 1.25});
	CheckVec(batch.stages.scaler_scale,
	         {0.4841239182759271, 0.7071077811865476});
	CheckVec(batch.stages.outlier_lower,
	         {-4.2761710377778765, -4.276173823175774});
	CheckVec(batch.stages.outlier_upper,
	         {4.2761710377778765, 4.276173823175774});

	std::vector<double> exp_x = {
	    -0.7745950692447884, -1.0606586717819424, -0.7745950692447884, 0.3535528905939808,
	    1.2909917820746473,  0.0,                 1.2909917820746473,  1.7677644529699041,
	    -0.7745950692447884, -0.7071057811879616, 1.2909917820746473,  0.0,
	    -0.7745950692447884, 1.0606586717819424,  -0.7745950692447884, -1.4142115623759233,
	    1.2909917820746473,  -0.3535528905939808, -0.7745950692447884, 0.0,
	    -2.840181920564224,  1.4142115623759233,  1.2909917820746473,  -1.0606586717819424};
	CheckVec(batch.x, exp_x);

	CheckVec(batch.y, {0, 1, 1, 0, 0, 1, 0, 0, -100, -100, -100, -100});
	CheckMask(batch.cat_mask, {true, false});
	REQUIRE(batch.label_decoder.size() == 2);
	REQUIRE(batch.label_decoder[0].ToString() == "no");
	REQUIRE(batch.label_decoder[1].ToString() == "yes");
}

TEST_CASE("preprocess: regression_dates golden parity", "[tabfm][preprocess]") {
	vector<LogicalType> types = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                             LogicalType::VARCHAR, LogicalType::DATE,
	                             LogicalType::DOUBLE};
	// num_p, num_q, cat_r, date_s, target(regression)
	std::vector<std::vector<Value>> rows = {
	    {VDouble(10), VDouble(-1.0), VStr("x"), VDate(2024, 1, 5), VDouble(100)},
	    {VDouble(20), VDouble(0.5), VStr("y"), VDate(2024, 2, 10), VDouble(110)},
	    {VNullDouble(), VDouble(1.5), VStr("x"), VNullDate(), VDouble(95)},
	    {VDouble(40), VDouble(-0.5), VStr("z"), VDate(2024, 4, 20), VDouble(130)},
	    {VDouble(50), VDouble(2.0), VStr("y"), VDate(2024, 5, 25), VDouble(140)},
	    {VDouble(60), VDouble(0.0), VStr("x"), VDate(2024, 6, 30), VDouble(150)},
	    {VDouble(70), VDouble(1.0), VStr("z"), VDate(2024, 8, 4), VDouble(160)},
	    {VDouble(80), VDouble(-1.5), VStr("y"), VDate(2024, 9, 9), VDouble(170)},
	    {VDouble(15), VDouble(0.25), VStr("z"), VDate(2024, 10, 14), VNullDouble()},
	    {VDouble(25), VDouble(-0.75), VStr("x"), VNullDate(), VNullDouble()},
	    {VNullDouble(), VDouble(1.75), VStr("y"), VDate(2024, 12, 24), VNullDouble()},
	    {VDouble(55), VDouble(0.5), VStr("w"), VDate(2025, 1, 28), VNullDouble()},
	};
	auto data = MakeCollection(types, rows);
	vector<PreprocessColumnSpec> cols = {
	    {"num_p", LogicalType::DOUBLE, false, true},
	    {"num_q", LogicalType::DOUBLE, false, true},
	    {"cat_r", LogicalType::VARCHAR, false, true},
	    {"date_s", LogicalType::DATE, false, true},
	    {"target", LogicalType::DOUBLE, true, false},
	};
	auto batch = PreprocessBatch(*data, cols, PreprocessTask::REGRESSION);

	REQUIRE(batch.H == 7);
	REQUIRE(batch.d == 7);
	REQUIRE(batch.train_size == 8);

	// encoded order: cat_r, num_p, num_q, date_s.epoch, .year, .month, .day, .dow
	std::vector<double> exp_train = {
	    0, 10, -1, 1.7044128e18, 2024, 1, 5, 4,
	    1, 20, 0.5, 1.7075232e18, 2024, 2, 10, 5,
	    0, 47.142857142857146, 1.5, 1.7157682285714286e18, 2024, 5, 15, 2,
	    2, 40, -0.5, 1.7135712e18, 2024, 4, 20, 5,
	    1, 50, 2, 1.7165952e18, 2024, 5, 25, 5,
	    0, 60, 0, 1.7197056e18, 2024, 6, 30, 6,
	    2, 70, 1, 1.7227296e18, 2024, 8, 4, 6,
	    1, 80, -1.5, 1.72584e18, 2024, 9, 9, 0};
	CheckVec(batch.stages.encoded_train, exp_train);
	std::vector<double> exp_test = {
	    2, 15, 0.25, 1.728864e18, 2024, 10, 14, 0,
	    0, 25, -0.75, 1.7157682285714286e18, 2024, 5, 15, 2,
	    1, 47.142857142857146, 1.75, 1.7349984e18, 2024, 12, 24, 1,
	    -1, 55, 0.5, 1.7380224e18, 2025, 1, 28, 1};
	CheckVec(batch.stages.encoded_test, exp_test);

	CheckMask(batch.stages.unique_filter_keep,
	          {true, true, true, true, false, true, true, true});
	REQUIRE(batch.stages.cat_feature_indices_kept == vector<int64_t>({0}));

	CheckVec(batch.stages.scaler_mean,
	         {0.875, 47.14285714285714, 0.25, 1.7157682285714286e18, 5.0, 14.75,
	          4.125});
	CheckVec(batch.stages.scaler_scale,
	         {0.7806257497997998, 22.20038709702865, 1.1456449237389599,
	          6780866796677673.0, 2.5495107567963924, 8.88467882195843,
	          1.9645302056877139});

	std::vector<double> exp_x = {
	    -1.1208956407400135, -1.6730725000659301, -1.091088498799841, -1.67462787751446, -1.5689284657210982, -1.0973947618571123, -0.0636284439089303,
	    0.16012794867714478, -1.2226299038943336, 0.2182176997599682, -1.2159254588909356, -1.1766963492908238, -0.5346282173150034, 0.4453991073625121,
	    -1.1208956407400135, 3.200587145866483e-16, 1.091088498799841, 0.0, 0.0, 0.028138327227105443, -1.081683546451815,
	    1.441151538094303, -0.32174471155114026, -0.6546530992799046, -0.32400408934519337, -0.39223211643027456, 0.5909048717692144, 0.4453991073625121,
	    0.16012794867714478, 0.12869788462045637, 1.5275238983197774, 0.12195659542767773, 0.0, 1.1536714163113233, 0.4453991073625121,
	    -1.1208956407400135, 0.5791404807920529, -0.2182176997599682, 0.5806590140512022, 0.39223211643027456, 1.7164379608534321, 0.9544266586339545,
	    1.441151538094303, 1.0295830769636496, 0.6546530992799046, 1.0266196988240734, 1.1766963492908238, -1.2099480707655341, 0.9544266586339545,
	    0.16012794867714478, 1.4800256731352461, -1.5275238983197774, 1.485322117447598, 1.5689284657210982, -0.6471815262234252, -2.0997386489946996,
	    1.441151538094303, -1.4478512019801317, 0.0, 1.931282802220469, 1.961160582151373, -0.08441498168131634, -2.0997386489946996,
	    -1.1208956407400135, -0.9974086058085352, -0.8728707990398727, 0.0, 0.0, 0.028138327227105443, -1.081683546451815,
	    0.16012794867714478, 3.200587145866483e-16, 1.3093061985598091, 2.8359459056168648, 2.745624815011922, 1.0411181074029014, -1.5907110977232575,
	    -2.4019192301571715, 0.35391918270625466, 0.2182176997599682, 3.281906590389736, -1.5689284657210982, 1.4913313430365887, -1.5907110977232575};
	CheckVec(batch.x, exp_x);

	CheckVec(batch.y,
	         {-1.2160103078465596, -0.8345168779339135, -1.4067570228028827,
	          -0.07153001810862115, 0.309963411804025, 0.6914568417166711,
	          1.0729502716293173, 1.4544437015419633, -100, -100, -100, -100});
	CheckMask(batch.cat_mask,
	          {true, false, false, false, false, false, false});

	REQUIRE(batch.target_mean == Approx(131.875).epsilon(1e-9));
	REQUIRE(batch.target_scale == Approx(26.21277121938846).epsilon(1e-9));
}
