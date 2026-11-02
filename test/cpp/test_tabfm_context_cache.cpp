//===----------------------------------------------------------------------===//
// Catch2 tests for the context cache (docs/CONTEXT_CACHE_PLAN.md, #37):
//   - the bundled tabicl-v2 pair, the path users actually run, is compiled in, byte-equal to resources/,
//     and passes the exactness contract the engine checks before it serves a pair;
//   - the pair contract itself (ContextCachePairProblem);
//   - concurrency: several connections and threads against one shared cache;
//   - the admission rule: keep a context on its second sighting.
//===----------------------------------------------------------------------===//

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb_compat.hpp"

#include "tabfm_bundled_resources.hpp"
#include "tabfm_context_cache.hpp"
#include "tabfm_state.hpp"
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
	//! entries, bytes, hits, misses, first_seen of the context-cache session (zeros when not loaded)
	vector<int64_t> Counters() {
		auto r = Q("SELECT coalesce(sum(context_entries), 0), coalesce(sum(context_bytes), 0), "
		           "coalesce(sum(context_hits), 0), coalesce(sum(context_misses), 0), "
		           "coalesce(sum(context_first_seen), 0) FROM tabfm_models() "
		           "WHERE model = 'kvfx' AND loaded AND context_cache");
		vector<int64_t> out;
		for (idx_t col = 0; col < 5; col++) {
			out.push_back(duckdb_compat::ResultValue(*r, col, 0).GetValue<int64_t>());
		}
		return out;
	}
	//! Prepare-graph runs of every loaded kvfx session (not reported by tabfm_models()).
	int64_t Encodes() {
		auto state = TabFMState::Get(*db.instance);
		int64_t total = 0;
		for (auto &key : state->LoadedKeys()) {
			if (key.rfind("kvfx:", 0) != 0) {
				continue;
			}
			for (auto &model : state->SnapshotsFor(key)) {
				if (model->context_stats) {
					total += NumericCast<int64_t>(model->context_stats->encodes.load());
				}
			}
		}
		return total;
	}
	//! The answer for an arbitrary relation over s<k>'s rows (e.g. a reordering), in id order, as Answer.
	static string AnswerFor(Connection &con, int k, const string &relation, const string &query_batch = "") {
		const string opts = query_batch.empty() ? "" : ", opts := MAP{'query_batch': '" + query_batch + "'}";
		auto result =
		    con.Query("SELECT string_agg(yhat || ':' || printf('%.6f', yhat_score), ',' ORDER BY id) FROM "
		              "tabfm_classify('" +
		              relation + "', 'label', features := ['f1', 'f2', 'f3'], model := 'kvfx'" + opts + ") JOIN s" +
		              std::to_string(k) + " USING (f1, f2, f3)");
		if (result->HasError()) {
			return "ERROR " + result->GetError();
		}
		return duckdb_compat::ResultValue(*result, 0, 0).ToString();
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
	// one context's size, as the cache accounts it (every table has the same shape); kept on its second
	// sighting
	fx.Q("CALL tabfm_unload()");
	REQUIRE(KvFixture::Answer(fx.setup, 0, "") == reference[0]);
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
		REQUIRE(c[0] == KvFixture::TABLES);     // every context fits in the default budget
		REQUIRE(c[3] == 2 * KvFixture::TABLES); // each encoded twice: first sighting (not kept), then kept
		REQUIRE(c[4] == KvFixture::TABLES);     // one first sighting per table
		REQUIRE(fx.Encodes() == c[3]);          // one encode per miss, whatever query_batch is
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
		// twice: an unload in flight may have dropped the sightings, so the first may not be kept
		REQUIRE(KvFixture::Answer(fx.setup, 1, "3") == reference[1]);
		REQUIRE(KvFixture::Answer(fx.setup, 1, "3") == reference[1]);
		auto c = fx.Counters();
		REQUIRE(c[0] >= 1);
		REQUIRE(c[1] == c[0] * one);
	}
}

//===----------------------------------------------------------------------===//
// The admission rule (docs/CONTEXT_CACHE_PLAN.md, "Phase 1b")
//===----------------------------------------------------------------------===//

TEST_CASE("context cache: sightings window", "[tabfm][context_cache]") {
	ContextSightings window(3);
	REQUIRE(window.Capacity() == 3);
	REQUIRE_FALSE(window.Sight(1)); // first sighting
	REQUIRE(window.Sight(1));       // second
	REQUIRE(window.Sight(1));
	REQUIRE_FALSE(window.Sight(2));
	REQUIRE_FALSE(window.Sight(3));
	REQUIRE(window.Size() == 3);
	// 1 is the least recently seen; a fourth key pushes it out
	REQUIRE(window.Sight(2)); // order now 2, 3, 1
	REQUIRE(window.Sight(1)); // order now 1, 2, 3
	REQUIRE_FALSE(window.Sight(4));
	REQUIRE(window.Size() == 3);
	REQUIRE_FALSE(window.Sight(3)); // evicted: a first sighting again
	REQUIRE(window.Sight(1));

	SECTION("bounded at any volume") {
		ContextSightings big(CONTEXT_SIGHTING_WINDOW);
		for (hash_t k = 0; k < CONTEXT_SIGHTING_WINDOW * 3; k++) {
			REQUIRE_FALSE(big.Sight(k * 7919));
		}
		REQUIRE(big.Size() == CONTEXT_SIGHTING_WINDOW);
		REQUIRE_FALSE(big.Sight(0)); // the oldest are gone
		REQUIRE(big.Sight((CONTEXT_SIGHTING_WINDOW * 3 - 1) * 7919));
	}
	SECTION("capacity 0 remembers nothing") {
		ContextSightings none(0);
		REQUIRE_FALSE(none.Sight(5));
		REQUIRE_FALSE(none.Sight(5));
		REQUIRE(none.Size() == 0);
	}
}

TEST_CASE("context cache: a context is kept on its second sighting", "[tabfm][context_cache]") {
	KvFixture fx;
	const string ref = KvFixture::Answer(fx.setup, 0, "");
	REQUIRE(ref.find("ERROR") == string::npos);
	// the kill switch's answer for every table
	fx.Q("SET anofox_tabfm_context_cache = false");
	REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
	const string ref1 = KvFixture::Answer(fx.setup, 1, "");
	fx.Q("RESET anofox_tabfm_context_cache");
	fx.Q("CALL tabfm_unload()");

	SECTION("first sighting: answered, not kept; second: kept; third: a hit") {
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		REQUIRE(fx.Counters() == vector<int64_t> {0, 0, 0, 1, 1});
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		auto c = fx.Counters();
		REQUIRE(c[0] == 1);
		REQUIRE(c[1] > 0);
		REQUIRE(c[2] == 0);
		REQUIRE(c[3] == 2);
		REQUIRE(c[4] == 1);
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		c = fx.Counters();
		REQUIRE(c[2] == 1);
		REQUIRE(c[3] == 2);
		REQUIRE(fx.Encodes() == 2);
	}
	SECTION("query_batch on a first sighting: one encode for the whole call") {
		REQUIRE(KvFixture::Answer(fx.setup, 1, "3") == ref1); // 10 query rows: four passes
		REQUIRE(fx.Encodes() == 1);
		REQUIRE(fx.Counters() == vector<int64_t> {0, 0, 0, 1, 1});
		REQUIRE(KvFixture::Answer(fx.setup, 1, "3") == ref1);
		REQUIRE(fx.Encodes() == 2);
		REQUIRE(KvFixture::Answer(fx.setup, 1, "3") == ref1);
		REQUIRE(fx.Encodes() == 2);
		REQUIRE(fx.Counters()[2] == 1);
	}
	SECTION("row order is part of the key: the same rows reordered are another context") {
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		REQUIRE(fx.Counters()[0] == 1);
		REQUIRE(KvFixture::AnswerFor(fx.setup, 0, "(SELECT * FROM s0 ORDER BY f1 DESC, f2, f3, id)") == ref);
		auto c = fx.Counters();
		REQUIRE(c[2] == 0); // a miss, and a first sighting: not kept
		REQUIRE(c[3] == 3);
		REQUIRE(c[4] == 2);
		REQUIRE(c[0] == 1);
	}
	SECTION("'first' keeps a context at once") {
		fx.Q("SET anofox_tabfm_context_cache_admit = 'first'");
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		auto c = fx.Counters();
		REQUIRE(c[0] == 1);
		REQUIRE(c[4] == 1); // still a first sighting, counted, but kept
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		REQUIRE(fx.Counters()[2] == 1);
		fx.Q("RESET anofox_tabfm_context_cache_admit");
	}
	SECTION("a context seen once, then pushed out of the window, is a first sighting again") {
		// Scored through the aggregate: one context per group. CONTEXT_SIGHTING_WINDOW + 1 distinct groups
		// after s0's first sighting push it out of the window.
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		fx.Q("CREATE TABLE many AS SELECT g, (i % 4) * 0.5 + g * 1e-3 AS f1, (i % 3) * 0.25 AS f2, "
		     "CASE WHEN i < 4 THEN 'c' || (i % 2) END AS label FROM range(" +
		     std::to_string(CONTEXT_SIGHTING_WINDOW + 1) + ") g(g), range(5) r(i)");
		// count(len(res)), not count(*): an unused aggregate would be optimized away
		auto r = fx.Q("SELECT count(len(res)) FROM (SELECT g, __anofox_tabfm_predict_agg({'f1': f1, 'f2': f2, "
		              "'label': label}, 'label', MAP{'model': 'kvfx', 'task': 'classification'}) AS res FROM many "
		              "GROUP BY g)");
		REQUIRE(duckdb_compat::ResultValue(*r, 0, 0).GetValue<int64_t>() ==
		        static_cast<int64_t>(CONTEXT_SIGHTING_WINDOW + 1));
		auto c = fx.Counters();
		REQUIRE(c[0] == 0); // every group seen once: nothing kept
		REQUIRE(c[4] == static_cast<int64_t>(CONTEXT_SIGHTING_WINDOW + 2));
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		c = fx.Counters();
		REQUIRE(c[0] == 0); // s0 had dropped out of the window: not kept
		REQUIRE(c[4] == static_cast<int64_t>(CONTEXT_SIGHTING_WINDOW + 3));
		REQUIRE(KvFixture::Answer(fx.setup, 0, "") == ref);
		REQUIRE(fx.Counters()[0] == 1);
	}
}
