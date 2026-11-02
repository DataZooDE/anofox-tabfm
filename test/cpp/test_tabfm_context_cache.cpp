//===----------------------------------------------------------------------===//
// Catch2 tests for the context cache (docs/CONTEXT_CACHE_PLAN.md, #37):
//   - the bundled tabicl-v2 pair, the path users actually run, is compiled in, byte-equal to resources/,
//     and passes the exactness contract the engine checks before it serves a pair;
//   - the pair contract itself (ContextCachePairProblem);
//   - concurrency: several connections and threads against one shared cache.
//===----------------------------------------------------------------------===//

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#include "tabfm_bundled_resources.hpp"
#include "tabfm_onnx_reader.hpp"
#include "tabfm_predict.hpp"

#include <atomic>
#include <fstream>
#include <iterator>
#include <map>
#include <thread>
#include <utility>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

//! What Connection::Query returns: MaterializedQueryResult before DuckDB 2.0, a different type after.
using QueryResultPtr = decltype(std::declval<Connection &>().Query(string()));

string ThisDir() {
	string file = __FILE__;
	return file.substr(0, file.find_last_of("/\\"));
}

string ReadDiskFile(const string &path) {
	std::ifstream file(path, std::ios::binary);
	return string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

vector<string> ToVec(const std::vector<std::string> &names) {
	return vector<string>(names.begin(), names.end());
}

::anofox::onnxread::Graph ParseBundled(const string &id) {
	auto res = GetBundledResource(id);
	REQUIRE(res.data != nullptr);
	return ::anofox::onnxread::ParseModelBytes(reinterpret_cast<const uint8_t *>(res.data), res.size, id);
}

} // namespace

// The built-in tabicl-v2 pair is found by name next to the bundled combined graph (FindSplitGraphs). If an
// embed entry or an id drifts, the engine silently serves the combined graph and every default test stays
// green -- the cache just never engages. This pins all eight artifacts and the contract the engine checks.
TEST_CASE("context cache: the bundled tabicl-v2 pair is compiled in and exact", "[tabfm][bundled][context_cache]") {
	const string resources = ThisDir() + "/../../resources/";
	for (auto *task : {"classification", "regression"}) {
		INFO("task " << task);
		for (auto *half : {"prepare", "query"}) {
			INFO("half " << half);
			const string graph_file = string("graph_") + half + "_tabicl_" + task + ".onnx";
			const string map_file = string("tensor_map_") + half + "_tabicl_" + task + ".json";
			// the ids FindSplitGraphs asks for: the graph by stem, the map by file name
			auto graph = GetBundledResource(string("graph_") + half + "_tabicl_" + task);
			auto map = GetBundledResource(map_file);
			REQUIRE(graph.data != nullptr);
			REQUIRE(map.data != nullptr);
			REQUIRE(string(graph.data, graph.size) == ReadDiskFile(resources + graph_file));
			REQUIRE(string(map.data, map.size) == ReadDiskFile(resources + map_file));
		}
		auto prepare = ParseBundled(string("graph_prepare_tabicl_") + task);
		auto query = ParseBundled(string("graph_query_tabicl_") + task);
		auto combined = ParseBundled(string("graph_tabicl_") + task);
		REQUIRE(ContextCachePairProblem(ToVec(prepare.inputs), ToVec(prepare.outputs), ToVec(query.inputs),
		                                ToVec(query.outputs), ToVec(combined.inputs)) == "");
		// 3 column-ISAB blocks + 12 ICL layers x (K, V), plus the scaler for regression
		idx_t state = 0;
		for (auto &name : query.inputs) {
			state += name != "x";
		}
		REQUIRE(state == (string(task) == "regression" ? 29 : 27));
	}
}

TEST_CASE("context cache: the pair contract", "[tabfm][context_cache]") {
	const vector<string> xy = {"x", "y"};
	const vector<string> prep_out = {"hidden_0", "icl_k_0", "icl_v_0", "context_logits", "val_3"};
	const vector<string> qry_in = {"x", "hidden_0", "icl_k_0", "icl_v_0"};
	const vector<string> logits = {"logits", "val_4"};
	REQUIRE(ContextCachePairProblem(xy, prep_out, qry_in, logits, xy) == "");
	// a query graph that also reads y is fine
	REQUIRE(ContextCachePairProblem(xy, prep_out, {"x", "y", "hidden_0"}, logits, xy) == "");

	SECTION("no context_logits: the context rows' outputs would differ") {
		REQUIRE(ContextCachePairProblem(xy, {"hidden_0", "icl_k_0", "icl_v_0"}, qry_in, logits, xy) != "");
	}
	SECTION("query reads a tensor prepare does not emit") {
		REQUIRE(ContextCachePairProblem(xy, prep_out, {"x", "hidden_0", "icl_k_1"}, logits, xy) != "");
	}
	SECTION("query reads context_logits") {
		REQUIRE(ContextCachePairProblem(xy, prep_out, {"x", "context_logits"}, logits, xy) != "");
	}
	SECTION("query takes no state") {
		REQUIRE(ContextCachePairProblem(xy, prep_out, {"x", "y"}, logits, xy) != "");
	}
	SECTION("prepare takes more than x and y") {
		REQUIRE(ContextCachePairProblem({"x", "y", "cat_mask"}, prep_out, qry_in, logits, xy) != "");
	}
	SECTION("the combined graph takes inputs the key does not cover (cat_mask, d, train_size)") {
		REQUIRE(ContextCachePairProblem(xy, prep_out, qry_in, logits, {"x", "y", "cat_mask", "d", "train_size"}) !=
		        "");
	}
	SECTION("query emits no logits") {
		REQUIRE(ContextCachePairProblem(xy, prep_out, qry_in, {"val_4"}, xy) != "");
	}
}

namespace {

struct KvFixture {
	DuckDB db;
	Connection setup;
	static constexpr int TABLES = 4;

	KvFixture() : db(nullptr), setup(db) {
		Q("LOAD anofox_tabfm");
		const string dir = ThisDir() + "/../fixtures/tabicl_kv";
		Q("CALL tabfm_register_model(id := 'kvfx', base_dir := '" + dir +
		  "', classification_graph := 'graph_tabicl_classification.onnx', "
		  "classification_weights := '../tabicl/model_classification.safetensors', "
		  "tensor_map := 'tensor_map_tabicl_classification.json', license := 'bsd-3-clause', commercial := true, "
		  "weights_repo := 'local:kvfx', weights_revision := 'kvfx-1', preprocessing_profile := 'tabicl_v2_raw', "
		  "max_rows := 4096, max_features := 64, max_classes := 3)");
		// TABLES support sets of the same shape (so every context costs the same), 30 labelled + 10 to score
		for (int k = 0; k < TABLES; k++) {
			Q("CREATE TABLE s" + std::to_string(k) + " AS SELECT i AS id, (i % 7) * 0.3 + " + std::to_string(k) +
			  " * 0.01 AS f1, (i % 5) * 0.2 AS f2, (i % 3) * 0.25 AS f3, CASE WHEN i < 30 THEN 'c' || (i % 3) END "
			  "AS label FROM range(40) r(i)");
		}
	}

	QueryResultPtr Q(const string &sql) {
		return Q(setup, sql);
	}
	static QueryResultPtr Q(Connection &con, const string &sql) {
		auto result = con.Query(sql);
		if (result->HasError()) {
			FAIL(sql << ": " << result->GetError());
		}
		return result;
	}
	//! The full answer for table k, in id order, as text: the label and the score to 1e-6. Not every digit:
	//! a query batch of a different size can take a different ORT kernel (a one-row batch differs from the
	//! unbatched pass at ~1e-9), so batching is exact to float rounding, not bit for bit.
	static string Answer(Connection &con, int k, const string &query_batch) {
		const string opts = query_batch.empty() ? "" : ", opts := MAP{'query_batch': '" + query_batch + "'}";
		auto result = con.Query("SELECT string_agg(yhat || ':' || printf('%.6f', yhat_score), ',' ORDER BY id) "
		                        "FROM tabfm_classify('s" +
		                        std::to_string(k) + "', 'label', features := ['f1', 'f2', 'f3'], model := 'kvfx'" +
		                        opts + ") JOIN s" + std::to_string(k) + " USING (f1, f2, f3)");
		if (result->HasError()) {
			return "ERROR " + result->GetError();
		}
		return duckdb_compat::ResultValue(*result, 0, 0).ToString();
	}
	//! entries, bytes, hits, misses of the context-cache session (zeros when not loaded)
	vector<int64_t> Counters() {
		auto r = Q("SELECT coalesce(sum(context_entries), 0), coalesce(sum(context_bytes), 0), "
		           "coalesce(sum(context_hits), 0), coalesce(sum(context_misses), 0) FROM tabfm_models() "
		           "WHERE model = 'kvfx' AND loaded AND context_cache");
		return {duckdb_compat::ResultValue(*r, 0, 0).GetValue<int64_t>(), duckdb_compat::ResultValue(*r, 1, 0).GetValue<int64_t>(),
		        duckdb_compat::ResultValue(*r, 2, 0).GetValue<int64_t>(), duckdb_compat::ResultValue(*r, 3, 0).GetValue<int64_t>()};
	}
};

} // namespace

// The cache is one per loaded model, shared by every connection and by parallel finalize threads. Its list
// is guarded by its own lock (and, today, also by the per-device lock the caller holds). Several
// connections -- 'auto' and 'cpu', which resolve to one backend and so one cache -- hammer it with mixed
// contexts, query_batch values and budgets; every answer must equal the sequential one, and the counters
// must stay consistent.
TEST_CASE("context cache: concurrent connections share one cache safely", "[tabfm][context_cache]") {
	KvFixture fx;
	std::map<int, string> reference;
	for (int k = 0; k < KvFixture::TABLES; k++) {
		reference[k] = KvFixture::Answer(fx.setup, k, "");
		REQUIRE(reference[k].find("ERROR") == string::npos);
	}
	// one context's size, as the cache accounts it (every table has the same shape)
	fx.Q("CALL tabfm_unload()");
	REQUIRE(KvFixture::Answer(fx.setup, 0, "") == reference[0]);
	const int64_t one = fx.Counters()[1];
	REQUIRE(one > 0);

	const vector<string> batches = {"", "3", "7", "100"};
	const int THREADS = 4, ITERS = 12;

	SECTION("mixed contexts and query_batch: answers, hits + misses == calls, bytes == entries x one") {
		fx.Q("CALL tabfm_unload()");
		std::atomic<int> mismatches {0}, calls {0};
		vector<string> errors(THREADS);
		vector<std::thread> pool;
		for (int t = 0; t < THREADS; t++) {
			pool.emplace_back([&, t]() {
				Connection con(fx.db);
				con.Query(t % 2 ? "SET anofox_tabfm_device = 'cpu'" : "SET anofox_tabfm_device = 'auto'");
				for (int i = 0; i < ITERS; i++) {
					const int k = (t + i) % KvFixture::TABLES;
					auto got = KvFixture::Answer(con, k, batches[(t + i) % batches.size()]);
					calls++;
					if (got != reference[k]) {
						mismatches++;
						size_t d = 0;
						while (d < got.size() && d < reference[k].size() && got[d] == reference[k][d]) {
							d++;
						}
						errors[t] = "k=" + std::to_string(k) + " batch='" + batches[(t + i) % batches.size()] +
						            "' t=" + std::to_string(t) + " at " + std::to_string(d) + ": got " +
						            got.substr(d > 20 ? d - 20 : 0, 60) + " want " +
						            reference[k].substr(d > 20 ? d - 20 : 0, 60) + "\n";
					}
				}
			});
		}
		for (auto &th : pool) {
			th.join();
		}
		INFO("first error: " << errors[0] << errors[1] << errors[2] << errors[3]);
		REQUIRE(mismatches == 0);
		auto c = fx.Counters();
		REQUIRE(c[2] + c[3] == calls.load()); // one lookup per call
		REQUIRE(c[0] == KvFixture::TABLES);    // every context fits in the default budget
		REQUIRE(c[3] == KvFixture::TABLES);    // each encoded exactly once
		REQUIRE(c[1] == c[0] * one);
	}

	SECTION("budgets changing underneath, including '0', and unloads in flight: answers stay exact") {
		std::atomic<bool> done {false};
		std::atomic<int> mismatches {0};
		vector<std::thread> pool;
		for (int t = 0; t < THREADS; t++) {
			pool.emplace_back([&, t]() {
				Connection con(fx.db);
				con.Query(t % 2 ? "SET anofox_tabfm_device = 'cpu'" : "SET anofox_tabfm_device = 'auto'");
				for (int i = 0; i < ITERS; i++) {
					const int k = (t * 3 + i) % KvFixture::TABLES;
					if (KvFixture::Answer(con, k, batches[i % batches.size()]) != reference[k]) {
						mismatches++;
					}
				}
			});
		}
		// one more connection flips the instance-wide budget and unloads while the others score
		std::thread churn([&]() {
			Connection con(fx.db);
			const vector<string> budgets = {"0", "1KB", "100KB", "0", "1GB"};
			for (int i = 0; !done.load() && i < 200; i++) {
				con.Query("SET anofox_tabfm_context_cache_memory = '" + budgets[i % budgets.size()] + "'");
				if (i % 3 == 2) {
					con.Query("CALL tabfm_unload()");
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
			}
		});
		for (auto &th : pool) {
			th.join();
		}
		done = true;
		churn.join();
		REQUIRE(mismatches == 0);
		// settle: a fixed budget, one more call, and the accounting is consistent again
		fx.Q("SET anofox_tabfm_context_cache_memory = '1GB'");
		REQUIRE(KvFixture::Answer(fx.setup, 1, "3") == reference[1]);
		auto c = fx.Counters();
		REQUIRE(c[0] >= 1);
		REQUIRE(c[1] == c[0] * one);
	}
}
