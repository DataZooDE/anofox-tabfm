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
#include "duckdb/common/types/timestamp.hpp"

#include <cfloat>
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

// ---------------------------------------------------------------------------
// What a TARGET value means (the shared predicate every train/query decision uses).
//
//   NULL, or NaN in FLOAT/DOUBLE      -> MISSING   (the row is to be predicted)
//   +/-Infinity (FLOAT/DOUBLE, DATE, TIMESTAMP[...])  -> INFINITE  (an error for a model target)
//   a finite double beyond FLT_MAX    -> FLOAT32_OVERFLOW (it would become Infinity in the graph input)
//   anything else, INCLUDING the strings 'nan' / 'inf'  -> USABLE
// ---------------------------------------------------------------------------

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
Value TsFromText(const char *text, const LogicalType &type) {
	return Value(string(text)).DefaultCastAs(type);
}
} // namespace

TEST_CASE("target value: NULL and NaN are MISSING, in every type that can be NULL", "[tabfm][preprocess][target]") {
	REQUIRE(ClassifyTargetValue(VNullDouble()) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(VNullStr()) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(VNullBool()) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(VNullDate()) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(Value(LogicalType::BIGINT)) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(kNaN)) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(-kNaN)) == TargetValueKind::MISSING); // sign bit set
	REQUIRE(ClassifyTargetValue(Value::FLOAT(std::numeric_limits<float>::quiet_NaN())) == TargetValueKind::MISSING);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(std::numeric_limits<double>::signaling_NaN())) ==
	        TargetValueKind::MISSING);
}

TEST_CASE("target value: Infinity is INFINITE for floats and for temporal types", "[tabfm][preprocess][target]") {
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(kInf)) == TargetValueKind::INFINITE);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(-kInf)) == TargetValueKind::INFINITE);
	REQUIRE(ClassifyTargetValue(Value::FLOAT(std::numeric_limits<float>::infinity())) == TargetValueKind::INFINITE);
	REQUIRE(ClassifyTargetValue(Value::FLOAT(-std::numeric_limits<float>::infinity())) == TargetValueKind::INFINITE);
	// DuckDB has date/timestamp infinity; ToString() would make it the class label "infinity"
	REQUIRE(ClassifyTargetValue(Value::DATE(date_t::infinity())) == TargetValueKind::INFINITE);
	REQUIRE(ClassifyTargetValue(Value::DATE(date_t::ninfinity())) == TargetValueKind::INFINITE);
	REQUIRE(ClassifyTargetValue(Value::TIMESTAMP(timestamp_t::infinity())) == TargetValueKind::INFINITE);
	REQUIRE(ClassifyTargetValue(Value::TIMESTAMP(timestamp_t::ninfinity())) == TargetValueKind::INFINITE);
	// every temporal flavour, through its own text form (the native units differ)
	for (auto id : {LogicalTypeId::TIMESTAMP, LogicalTypeId::TIMESTAMP_TZ, LogicalTypeId::TIMESTAMP_SEC,
	                LogicalTypeId::TIMESTAMP_MS, LogicalTypeId::TIMESTAMP_NS}) {
		const LogicalType type(id);
		INFO("type " << type.ToString());
		REQUIRE(ClassifyTargetValue(TsFromText("infinity", type)) == TargetValueKind::INFINITE);
		REQUIRE(ClassifyTargetValue(TsFromText("-infinity", type)) == TargetValueKind::INFINITE);
		REQUIRE(ClassifyTargetValue(TsFromText("2023-06-30 12:00:00", type)) == TargetValueKind::USABLE);
	}
	REQUIRE(ClassifyTargetValue(TsFromText("infinity", LogicalType::DATE)) == TargetValueKind::INFINITE);
}

TEST_CASE("target value: the float32 range boundary for a double model target", "[tabfm][preprocess][target]") {
	const double fmax = static_cast<double>(FLT_MAX);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(fmax)) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(-fmax)) == TargetValueKind::USABLE);
	// the very next representable double no longer fits float32 and would become Infinity in the graph
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(std::nextafter(fmax, kInf))) == TargetValueKind::FLOAT32_OVERFLOW);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(std::nextafter(-fmax, -kInf))) == TargetValueKind::FLOAT32_OVERFLOW);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(1e300)) == TargetValueKind::FLOAT32_OVERFLOW);
	REQUIRE(ClassifyTargetValue(Value::DOUBLE(DBL_MAX)) == TargetValueKind::FLOAT32_OVERFLOW);
	// a FLOAT column cannot overflow float32 by construction
	REQUIRE(ClassifyTargetValue(Value::FLOAT(FLT_MAX)) == TargetValueKind::USABLE);
	REQUIRE(IsFloat32Finite(fmax));
	REQUIRE_FALSE(IsFloat32Finite(std::nextafter(fmax, kInf)));
	REQUIRE_FALSE(IsFloat32Finite(kInf));
	REQUIRE_FALSE(IsFloat32Finite(kNaN));
	REQUIRE(IsFloat32Finite(0.0));
	REQUIRE(IsFloat32Finite(-0.0));
}

TEST_CASE("target value: ordinary values, zeros, subnormals, other types and STRINGS are USABLE",
          "[tabfm][preprocess][target]") {
	for (double v : {0.0, -0.0, 1.0, -1.5, DBL_MIN, std::numeric_limits<double>::denorm_min(), 3.4e38}) {
		INFO("value " << v);
		REQUIRE(ClassifyTargetValue(Value::DOUBLE(v)) == TargetValueKind::USABLE);
	}
	REQUIRE(ClassifyTargetValue(Value::BIGINT(42)) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(Value::HUGEINT(hugeint_t(1) << 100)) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(Value::DECIMAL(int32_t(12345), 7, 2)) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(VBool(true)) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(VDate(2023, 1, 15)) == TargetValueKind::USABLE);
	// A string is a LABEL, never a missing marker: 'nan' and 'inf' stay legitimate class names
	for (const char *label : {"nan", "NaN", "inf", "-inf", "infinity", "", "NULL"}) {
		INFO("label '" << label << "'");
		REQUIRE(ClassifyTargetValue(VStr(label)) == TargetValueKind::USABLE);
	}
}

TEST_CASE("target value: which kinds are errors for a model target", "[tabfm][preprocess][target]") {
	for (bool regression : {false, true}) {
		REQUIRE_FALSE(IsInvalidModelTarget(TargetValueKind::MISSING, regression));
		REQUIRE_FALSE(IsInvalidModelTarget(TargetValueKind::USABLE, regression));
		REQUIRE(IsInvalidModelTarget(TargetValueKind::INFINITE, regression));
	}
	// a huge double is only an error where it feeds a float32 tensor: a class label is a string
	REQUIRE(IsInvalidModelTarget(TargetValueKind::FLOAT32_OVERFLOW, true));
	REQUIRE_FALSE(IsInvalidModelTarget(TargetValueKind::FLOAT32_OVERFLOW, false));
}

TEST_CASE("target value: a finite timestamp far outside the microsecond range is USABLE, not an error",
          "[tabfm][preprocess][target]") {
	// 10^13 seconds is year ~318,000: it fits TIMESTAMP_SEC's int64 but not a microsecond timestamp, so a
	// cast to TIMESTAMP would throw. The native-value check must neither throw nor call it infinite.
	REQUIRE(ClassifyTargetValue(Value::TIMESTAMPSEC(timestamp_sec_t(10000000000000LL))) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(Value::TIMESTAMPSEC(timestamp_sec_t(-10000000000000LL))) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(Value::TIMESTAMPMS(timestamp_ms_t(10000000000000000LL))) == TargetValueKind::USABLE);
}

TEST_CASE("target value: UHUGEINT is the one integer type that can exceed float32", "[tabfm][preprocess][target]") {
	REQUIRE(ClassifyTargetValue(Value::UHUGEINT(uhugeint_t(1) << 100)) == TargetValueKind::USABLE);
	REQUIRE(ClassifyTargetValue(Value::UHUGEINT(NumericLimits<uhugeint_t>::Maximum())) ==
	        TargetValueKind::FLOAT32_OVERFLOW);
	REQUIRE(ClassifyTargetValue(Value::HUGEINT(NumericLimits<hugeint_t>::Maximum())) == TargetValueKind::USABLE);
}

// ---------------------------------------------------------------------------
// The preprocessor's train/query split follows the same rule: a NaN target is a QUERY row, exactly
// like NULL; Infinity is an error naming the column, the category, the count and the first row.
// ---------------------------------------------------------------------------

namespace {

vector<PreprocessColumnSpec> TargetCols(const LogicalType &target_type) {
	return {{"f1", LogicalType::DOUBLE, false, true},
	        {"f2", LogicalType::DOUBLE, false, true},
	        {"target", target_type, true, false}};
}

// 8 rows. Rows 0-4 carry the labels 0..2 / values; rows 5-7 take `markers[i]`.
unique_ptr<ColumnDataCollection> DoubleTargetTable(const std::vector<Value> &markers) {
	const std::vector<double> labelled = {0.0, 1.0, 2.0, 1.0, 0.0};
	std::vector<std::vector<Value>> rows;
	for (size_t i = 0; i < 8; i++) {
		Value target = i < 5 ? VDouble(labelled[i]) : markers[i - 5];
		rows.push_back({VDouble(1.0 + 0.5 * i), VDouble(10.0 - i), target});
	}
	return MakeCollection({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, rows);
}

void RequireSameBatch(const PreprocessedBatch &a, const PreprocessedBatch &b) {
	REQUIRE(a.T == b.T);
	REQUIRE(a.train_size == b.train_size);
	CheckVec(a.x, b.x, 1e-12, 1e-12);
	CheckVec(a.y, b.y, 1e-12, 1e-12);
	REQUIRE(a.y_train == b.y_train);
	REQUIRE(a.target_mean == Approx(b.target_mean));
	REQUIRE(a.target_scale == Approx(b.target_scale));
	REQUIRE(a.row_source_index == b.row_source_index);
	REQUIRE(a.label_decoder.size() == b.label_decoder.size());
	for (size_t i = 0; i < a.label_decoder.size(); i++) {
		REQUIRE(a.label_decoder[i].ToString() == b.label_decoder[i].ToString());
	}
}

} // namespace

TEST_CASE("preprocess: a NaN target is a QUERY row, the same batch as NULL, in every branch",
          "[tabfm][preprocess][target]") {
	const Value nan = Value::DOUBLE(kNaN);
	for (auto task : {PreprocessTask::CLASSIFICATION, PreprocessTask::REGRESSION}) {
		for (bool standardize : {true, false}) {
			INFO("task " << (task == PreprocessTask::CLASSIFICATION ? "classification" : "regression")
			             << " standardize " << standardize);
			auto null_table = DoubleTargetTable({VNullDouble(), VNullDouble(), VNullDouble()});
			auto nan_table = DoubleTargetTable({nan, nan, nan});
			auto mixed_table = DoubleTargetTable({VNullDouble(), nan, VNullDouble()});
			auto cols = TargetCols(LogicalType::DOUBLE);
			auto reference = PreprocessBatch(*null_table, cols, task, standardize);
			REQUIRE(reference.train_size == 5);
			REQUIRE(reference.T == 8);
			// Baseline: a NaN target row was a TRAINING row (train_size 8) and poisoned the target mean.
			RequireSameBatch(reference, PreprocessBatch(*nan_table, cols, task, standardize));
			RequireSameBatch(reference, PreprocessBatch(*mixed_table, cols, task, standardize));
			if (task == PreprocessTask::REGRESSION) {
				REQUIRE(std::isfinite(reference.target_mean));
				REQUIRE(std::isfinite(reference.target_scale));
				for (double y : reference.y) {
					REQUIRE(std::isfinite(y));
				}
			}
		}
	}
}

TEST_CASE("preprocess: a table whose targets are ALL missing has no context, whichever marker is used",
          "[tabfm][preprocess][target]") {
	auto cols = TargetCols(LogicalType::DOUBLE);
	for (auto marker : {VNullDouble(), Value::DOUBLE(kNaN)}) {
		std::vector<std::vector<Value>> rows;
		for (size_t i = 0; i < 8; i++) {
			rows.push_back({VDouble(1.0 * i), VDouble(2.0 * i), marker});
		}
		auto all_missing = MakeCollection({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, rows);
		REQUIRE_THROWS_WITH(PreprocessBatch(*all_missing, cols, PreprocessTask::REGRESSION),
		                    Catch::Matchers::Contains("no in-context train rows"));
	}
}

TEST_CASE("preprocess: an Infinity target is an error naming the column, the category, the count and the row",
          "[tabfm][preprocess][target]") {
	for (auto task : {PreprocessTask::CLASSIFICATION, PreprocessTask::REGRESSION}) {
		for (bool standardize : {true, false}) {
			for (double inf : {kInf, -kInf}) {
				INFO("task " << (task == PreprocessTask::CLASSIFICATION ? "classification" : "regression")
				             << " standardize " << standardize << " value " << inf);
				// the Infinity sits in the 7th row; the other query rows are NaN and must stay legal
				auto table = DoubleTargetTable({Value::DOUBLE(kNaN), Value::DOUBLE(inf), Value::DOUBLE(kNaN)});
				REQUIRE_THROWS_WITH(PreprocessBatch(*table, TargetCols(LogicalType::DOUBLE), task, standardize),
				                    Catch::Matchers::Contains("target 'target' has 1 Infinity value") &&
				                        Catch::Matchers::Contains("input row 7") &&
				                        Catch::Matchers::Contains("CASE WHEN isfinite("));
			}
		}
	}
}

TEST_CASE("preprocess: a finite target beyond float32 is an error for REGRESSION only",
          "[tabfm][preprocess][target]") {
	auto table = DoubleTargetTable({VNullDouble(), Value::DOUBLE(1e300), VNullDouble()});
	for (bool standardize : {true, false}) {
		REQUIRE_THROWS_WITH(
		    PreprocessBatch(*table, TargetCols(LogicalType::DOUBLE), PreprocessTask::REGRESSION, standardize),
		    Catch::Matchers::Contains("target 'target'") && Catch::Matchers::Contains("float32") &&
		        Catch::Matchers::Contains("input row 7"));
	}
	// A class label is a string, never a float tensor: a huge double label is just another class.
	auto batch = PreprocessBatch(*table, TargetCols(LogicalType::DOUBLE), PreprocessTask::CLASSIFICATION, false);
	REQUIRE(batch.train_size == 6); // the five labelled rows and the 1e300 row
	// ...and FLT_MAX itself is fine for regression
	auto edge = DoubleTargetTable({VNullDouble(), Value::DOUBLE(static_cast<double>(FLT_MAX)), VNullDouble()});
	REQUIRE_NOTHROW(PreprocessBatch(*edge, TargetCols(LogicalType::DOUBLE), PreprocessTask::REGRESSION, false));
}

TEST_CASE("preprocess: with several invalid rows the error reports the count and the FIRST row, 1-based",
          "[tabfm][preprocess][target]") {
	// rows 6, 7 and 8 (1-based): -Infinity, NaN (legal), +Infinity
	auto table = DoubleTargetTable({Value::DOUBLE(-kInf), Value::DOUBLE(kNaN), Value::DOUBLE(kInf)});
	REQUIRE_THROWS_WITH(PreprocessBatch(*table, TargetCols(LogicalType::DOUBLE), PreprocessTask::REGRESSION),
	                    Catch::Matchers::Contains("has 2 Infinity value(s)") &&
	                        Catch::Matchers::Contains("first at input row 6"));
}

TEST_CASE("preprocess: UHUGEINT maximum as a regression target is an error, as a class label it is fine",
          "[tabfm][preprocess][target]") {
	std::vector<std::vector<Value>> rows;
	for (int i = 0; i < 6; i++) {
		rows.push_back({VDouble(1.0 * i), VDouble(2.0 * i),
		                i == 5 ? Value::UHUGEINT(NumericLimits<uhugeint_t>::Maximum())
		                       : (i == 4 ? Value(LogicalType::UHUGEINT) : Value::UHUGEINT(uhugeint_t(i + 1)))});
	}
	auto table = MakeCollection({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::UHUGEINT}, rows);
	REQUIRE_THROWS_WITH(PreprocessBatch(*table, TargetCols(LogicalType::UHUGEINT), PreprocessTask::REGRESSION),
	                    Catch::Matchers::Contains("beyond the float32 range") &&
	                        Catch::Matchers::Contains("input row 6"));
	REQUIRE_NOTHROW(PreprocessBatch(*table, TargetCols(LogicalType::UHUGEINT), PreprocessTask::CLASSIFICATION));
}

TEST_CASE("preprocess: DATE infinity as a classification target is an error, not the class 'infinity'",
          "[tabfm][preprocess][target]") {
	std::vector<std::vector<Value>> rows;
	for (int i = 0; i < 6; i++) {
		rows.push_back({VDouble(1.0 * i), VDouble(2.0 * i),
		                i == 5 ? Value::DATE(date_t::infinity()) : (i == 4 ? VNullDate() : VDate(2023, 1, 1 + i))});
	}
	auto table = MakeCollection({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DATE}, rows);
	REQUIRE_THROWS_WITH(PreprocessBatch(*table, TargetCols(LogicalType::DATE), PreprocessTask::CLASSIFICATION),
	                    Catch::Matchers::Contains("target 'target' has 1 Infinity value") &&
	                        Catch::Matchers::Contains("input row 6"));
}

TEST_CASE("preprocess: the strings 'nan' and 'inf' remain legitimate labels", "[tabfm][preprocess][target]") {
	std::vector<std::vector<Value>> rows = {
	    {VDouble(1.0), VDouble(2.0), VStr("nan")}, {VDouble(2.0), VDouble(1.0), VStr("inf")},
	    {VDouble(3.0), VDouble(0.0), VStr("nan")}, {VDouble(4.0), VDouble(5.0), VStr("-infinity")},
	    {VDouble(5.0), VDouble(6.0), VNullStr()},
	};
	auto table = MakeCollection({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::VARCHAR}, rows);
	auto batch = PreprocessBatch(*table, TargetCols(LogicalType::VARCHAR), PreprocessTask::CLASSIFICATION);
	REQUIRE(batch.train_size == 4);
	REQUIRE(batch.label_decoder.size() == 3);
}

// ---------------------------------------------------------------------------
// FEATURES follow the rule from the other side: nothing here is an error, everything unusable is MISSING
// and mean-imputed, decided at the SOURCE, before the mean is fitted. A value beyond float32 or a date
// infinity used to reach the fit (shifting the mean and every other row) and then the float32 cast.
// ---------------------------------------------------------------------------

namespace {

unique_ptr<ColumnDataCollection> FeatureTable(const LogicalType &type, const std::vector<Value> &feature) {
	std::vector<std::vector<Value>> rows;
	for (size_t i = 0; i < feature.size(); i++) {
		rows.push_back({feature[i], VDouble(10.0 - 0.5 * i), i < 6 ? VDouble(1.0 * (i % 3)) : VNullDouble()});
	}
	return MakeCollection({type, LogicalType::DOUBLE, LogicalType::DOUBLE}, rows);
}

vector<PreprocessColumnSpec> FeatureCols(const LogicalType &type) {
	return {{"feat", type, false, true}, {"other", LogicalType::DOUBLE, false, true},
	        {"target", LogicalType::DOUBLE, true, false}};
}

} // namespace

TEST_CASE("preprocess feature: a double beyond float32 is mean-imputed, exactly like NULL",
          "[tabfm][preprocess][feature]") {
	auto base = [](Value marker_train, Value marker_test) {
		return std::vector<Value> {VDouble(1.0), VDouble(2.0), marker_train, VDouble(4.0), VDouble(5.0), VDouble(6.0),
		                           VDouble(7.0), marker_test};
	};
	auto cols = FeatureCols(LogicalType::DOUBLE);
	for (bool standardize : {true, false}) {
		INFO("standardize " << standardize);
		auto null_table = FeatureTable(LogicalType::DOUBLE, base(VNullDouble(), VNullDouble()));
		auto reference = PreprocessBatch(*null_table, cols, PreprocessTask::REGRESSION, standardize);
		for (double marker : {1e300, -1e300, static_cast<double>(FLT_MAX) * 1.0000001, kInf, -kInf, kNaN}) {
			INFO("marker " << marker);
			auto table = FeatureTable(LogicalType::DOUBLE, base(Value::DOUBLE(marker), Value::DOUBLE(marker)));
			auto batch = PreprocessBatch(*table, cols, PreprocessTask::REGRESSION, standardize);
			CheckVec(batch.x, reference.x, 1e-12, 1e-12);
			for (double v : batch.x) {
				REQUIRE(std::isfinite(v));
				REQUIRE(std::fabs(v) <= static_cast<double>(FLT_MAX));
			}
		}
		// ...and FLT_MAX itself is an ordinary finite value, NOT imputed
		auto edge = FeatureTable(LogicalType::DOUBLE,
		                         base(Value::DOUBLE(static_cast<double>(FLT_MAX)), VNullDouble()));
		auto edge_batch = PreprocessBatch(*edge, cols, PreprocessTask::REGRESSION, false);
		bool differs = false;
		for (size_t i = 0; i < edge_batch.x.size(); i++) {
			differs = differs || std::fabs(edge_batch.x[i] - reference.x[i]) > 1e-9;
		}
		REQUIRE(differs);
	}
}

TEST_CASE("preprocess feature: UHUGEINT beyond float32 is imputed too", "[tabfm][preprocess][feature]") {
	std::vector<Value> feat;
	for (int i = 0; i < 8; i++) {
		feat.push_back(i == 2 ? Value::UHUGEINT(NumericLimits<uhugeint_t>::Maximum())
		                      : (i == 7 ? Value(LogicalType::UHUGEINT) : Value::UHUGEINT(uhugeint_t(i + 1))));
	}
	std::vector<Value> null_feat = feat;
	null_feat[2] = Value(LogicalType::UHUGEINT);
	auto cols = FeatureCols(LogicalType::UHUGEINT);
	auto batch = PreprocessBatch(*FeatureTable(LogicalType::UHUGEINT, feat), cols, PreprocessTask::REGRESSION, false);
	auto reference =
	    PreprocessBatch(*FeatureTable(LogicalType::UHUGEINT, null_feat), cols, PreprocessTask::REGRESSION, false);
	CheckVec(batch.x, reference.x, 1e-12, 1e-12);
}

TEST_CASE("preprocess feature: a date or timestamp infinity is imputed like NULL, never expanded",
          "[tabfm][preprocess][feature]") {
	// A sentinel passed to the day-index arithmetic yields a nonsense year/month/day, and entering the
	// mean fit it shifts the fill for every NULL row.
	struct Flavour {
		LogicalType type;
		Value finite, infinity, ninfinity, null;
	};
	std::vector<Flavour> flavours = {
	    {LogicalType::DATE, VDate(2023, 6, 1), Value::DATE(date_t::infinity()), Value::DATE(date_t::ninfinity()),
	     VNullDate()},
	    {LogicalType::TIMESTAMP, Value::TIMESTAMP(timestamp_t(1700000000000000LL)),
	     Value::TIMESTAMP(timestamp_t::infinity()), Value::TIMESTAMP(timestamp_t::ninfinity()),
	     Value(LogicalType::TIMESTAMP)},
	    {LogicalType::TIMESTAMP_TZ, Value::TIMESTAMPTZ(timestamp_tz_t(1700000000000000LL)),
	     Value::TIMESTAMPTZ(timestamp_tz_t(timestamp_t::infinity())),
	     Value::TIMESTAMPTZ(timestamp_tz_t(timestamp_t::ninfinity())), Value(LogicalType::TIMESTAMP_TZ)},
	};
	for (auto &f : flavours) {
		INFO(f.type.ToString());
		auto series = [&](const Value &marker) {
			std::vector<Value> v;
			for (int i = 0; i < 8; i++) {
				v.push_back(i == 2 || i == 7 ? marker : f.finite);
			}
			// make the finite ones differ so there is something to average
			return v;
		};
		auto cols = FeatureCols(f.type);
		auto reference = PreprocessBatch(*FeatureTable(f.type, series(f.null)), cols, PreprocessTask::REGRESSION, false);
		for (const Value &marker : {f.infinity, f.ninfinity}) {
			auto batch = PreprocessBatch(*FeatureTable(f.type, series(marker)), cols, PreprocessTask::REGRESSION, false);
			CheckVec(batch.x, reference.x, 1e-9, 1e-9);
			REQUIRE(batch.H == reference.H);
		}
	}
}
