//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P1 (docs/GGML_PLAN.md): the coverage matrix. Every ONNX op that appears in any bundled
// CPU graph (resources/graph_*.onnx, the weight-free graphs the CPU engine runs, not the ext/migraphx variants)
// has a classification, saying whether it is a native ggml op, a composite, a custom host kernel inside the graph,
// or evaluated on the host in the shape pass. Written red first: the table was empty.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_ggml_ops.hpp"
#include "tabfm_onnx_reader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;
namespace fs = std::filesystem;

namespace {

//! op -> number of bundled CPU graphs using it, over resources/graph_*.onnx minus the ext and migraphx variants.
std::map<std::string, int> BundledCpuGraphOps(int *n_graphs) {
	std::map<std::string, int> uses;
	int graphs = 0;
	for (const auto &entry : fs::directory_iterator("resources")) {
		const std::string name = entry.path().filename().string();
		if (name.rfind("graph_", 0) != 0 || entry.path().extension() != ".onnx" ||
		    name.rfind("graph_ext_", 0) == 0 || name.rfind("graph_migraphx_", 0) == 0) {
			continue;
		}
		auto graph = ::anofox::onnxread::ParseModelFile(entry.path().string());
		std::set<std::string> ops;
		for (const auto &node : graph.nodes) {
			ops.insert(node.op_type);
		}
		for (const auto &op : ops) {
			uses[op]++;
		}
		graphs++;
	}
	*n_graphs = graphs;
	return uses;
}

} // namespace

TEST_CASE("ggml lowering: every op in every bundled CPU graph is classified", "[tabfm][ggml][ggml_lowering]") {
	int n_graphs = 0;
	const auto uses = BundledCpuGraphOps(&n_graphs);
	REQUIRE(n_graphs >= 45); // the scan found the graphs (45 at P1; more models only add)
	REQUIRE(uses.size() >= 70);
	std::vector<std::string> unclassified;
	for (const auto &kv : uses) {
		if (GgmlLoweringTable().count(kv.first) == 0) {
			unclassified.push_back(kv.first);
		}
	}
	INFO("unclassified ops: " << unclassified.size());
	CHECK(unclassified == std::vector<std::string> {});
}

TEST_CASE("ggml lowering: the table has no dead rows and every row says how", "[tabfm][ggml][ggml_lowering]") {
	int n_graphs = 0;
	const auto uses = BundledCpuGraphOps(&n_graphs);
	REQUIRE(!GgmlLoweringTable().empty());
	for (const auto &kv : GgmlLoweringTable()) {
		INFO(kv.first);
		CHECK(uses.count(kv.first) == 1); // an op no bundled graph uses does not belong in the matrix
		CHECK(std::string(kv.second.how).size() >= 5);
	}
}

TEST_CASE("ggml lowering: implemented ops are a subset of the classified ones", "[tabfm][ggml][ggml_lowering]") {
	for (const auto &op : GgmlSupportedOps()) {
		INFO(op);
		CHECK(GgmlLoweringTable().count(op) == 1);
	}
}

// The custom-kernel list is the real size of the P3 work: pin it so adding or dropping a custom op is a
// deliberate edit, not a side effect (docs/GGML_LOWERING.md carries the reasons).
TEST_CASE("ggml lowering: the custom-kernel set is pinned", "[tabfm][ggml][ggml_lowering]") {
	std::set<std::string> custom;
	for (const auto &kv : GgmlLoweringTable()) {
		if (kv.second.kind == GgmlLowerKind::CUSTOM) {
			custom.insert(kv.first);
		}
	}
	const std::set<std::string> expected = {
	    "And",        "Clip",     "Equal",         "Erf",            "GatherElements", "GatherND",  "Greater",
	    "GreaterOrEqual", "IsInf", "IsNaN",         "Less",           "LessOrEqual", "Max",
	    "Min",        "Mod",      "Not",           "Or",             "Pow",       "ReduceMax",
	    "ReduceMin",  "ReduceProd", "Round",       "ScatterElements", "ScatterND", "Where"};
	CHECK(custom == expected);
}

// ggml_clamp is MAX(MIN(x, hi), lo) on plain C macros: a NaN input comes out as the upper bound, while ONNX Clip
// passes NaN through, and the graphs mask with NaN. The same goes for ggml_top_k, whose kernel documents "indices
// in no particular order" and swaps the first two, where ONNX TopK is sorted. A review found both marked as plain
// ggml ops; these two checks keep the reasons honest (docs/GGML_LOWERING.md).
TEST_CASE("ggml lowering: ops whose ggml namesake has different semantics are not marked native", "[tabfm][ggml][ggml_lowering]") {
	const auto &table = GgmlLoweringTable();
	REQUIRE(table.at("Clip").kind == GgmlLowerKind::CUSTOM);
	REQUIRE(std::string(table.at("Clip").how).find("NaN") != std::string::npos);
	REQUIRE(table.at("TopK").kind != GgmlLowerKind::NATIVE);
	REQUIRE(std::string(table.at("TopK").how).find("argsort") != std::string::npos);
	REQUIRE(table.at("Pad").kind != GgmlLowerKind::NATIVE); // ggml_pad_ext zero-fills only
	// the doc is generated from the table; a stale one hides exactly this kind of reclassification
	std::ifstream doc("docs/GGML_LOWERING.md");
	REQUIRE(doc.good());
	std::map<std::string, std::string> doc_kind;
	std::string line;
	while (std::getline(doc, line)) {
		if (line.rfind("| ", 0) != 0) {
			continue;
		}
		std::vector<std::string> cells;
		size_t start = 2;
		for (size_t bar = line.find(" | "); bar != std::string::npos; bar = line.find(" | ", start)) {
			cells.push_back(line.substr(start, bar - start));
			start = bar + 3;
		}
		if (cells.size() >= 2 && table.count(cells[0]) == 1) {
			doc_kind[cells[0]] = cells[1];
		}
	}
	const char *names[] = {"NATIVE", "COMPOSITE", "CUSTOM", "HOST"};
	REQUIRE(doc_kind.size() == table.size());
	for (const auto &kv : table) {
		INFO(kv.first);
		CHECK(doc_kind[kv.first] == names[static_cast<int>(kv.second.kind)]);
	}
}
