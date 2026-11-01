// Which graphs the MLX plugin can execute, known to the extension without loading the plugin.
//
// The plugin's interpreter refuses a graph that uses an op its Handlers() table lacks. The extension
// needs the same answer earlier -- for tabfm_backends() and for 'auto' -- on machines where the plugin
// is not even built, so the op set is mirrored in tabfm_mlx_ops.hpp. The first test keeps the mirror
// equal to the table by reading the plugin source itself; it runs on every platform.

#include "catch.hpp"

#include "tabfm_bundled_resources.hpp"
#include "tabfm_mlx_ops.hpp"
#include "tabfm_onnx_reader.hpp"

#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

TEST_CASE("mlx ops: the extension's op list is the plugin's handler table", "[tabfm][mlx_ops]") {
	std::ifstream f("src/tabfm_mlx_graph_ops.inc");
	REQUIRE(f.good());
	std::stringstream ss;
	ss << f.rdbuf();
	const std::string src = ss.str();
	const auto begin = src.find("Handlers() {");
	REQUIRE(begin != std::string::npos);
	const auto end = src.find("\n}\n", begin);
	const std::string table = src.substr(begin, end - begin);

	std::set<std::string> handled;
	std::regex macro(R"((?:UNARY|BINARY)\("(\w+)\")");
	std::regex entry(R"(\n\s*\{"(\w+)\",)");
	for (auto *re : {&macro, &entry}) {
		for (std::sregex_iterator it(table.begin(), table.end(), *re), stop; it != stop; ++it) {
			handled.insert((*it)[1].str());
		}
	}
	REQUIRE(handled.size() > 50); // the scrape found the table, not an empty match
	// the executor runs these three itself (Graph::MissingOps skips them)
	handled.insert({"Constant", "SplitToSequence", "SequenceAt"});
	REQUIRE(handled == MlxImplementedOps());
}

TEST_CASE("mlx ops: which bundled ext graphs MLX can run", "[tabfm][mlx_ops]") {
	auto missing_for = [](const std::string &id) {
		auto res = GetBundledResource(id);
		REQUIRE(res.data != nullptr);
		auto graph = ::anofox::onnxread::ParseModelBytes(reinterpret_cast<const uint8_t *>(res.data), res.size, id);
		std::vector<std::string> ops;
		for (auto &n : graph.nodes) {
			ops.push_back(n.op_type);
		}
		return MlxMissingOps(ops);
	};
	using V = std::vector<std::string>;
	REQUIRE(missing_for("graph_ext_mitra_classification") == V {});
	REQUIRE(missing_for("graph_ext_tabpfn35_regression") == V {});
	REQUIRE(missing_for("graph_ext_classification") == V {}); // tabfm-v1
	// ConstantOfShape and ReduceProd are implemented, so every bundled ext graph is executable
	REQUIRE(missing_for("graph_ext_tabdpt_classification") == V {});
	REQUIRE(missing_for("graph_ext_tabdpt13_regression") == V {});
	REQUIRE(missing_for("graph_ext_kumo_small_classification") == V {});
	REQUIRE(missing_for("graph_ext_orion_msp_classification") == V {});
}

// Executable is not the same as right, so a graph can be refused by name (MlxKnownWrong). None is today:
// tabdpt and tabdpt-v1-3 classification and orion-msp used to be, until the strided-view reads that made
// them disagree with the CPU were fixed (test_tabfm_mlx_plugin.cpp pins that).
TEST_CASE("mlx ops: no bundled graph is refused as known wrong", "[tabfm][mlx_ops]") {
	for (auto *model : {"tabdpt", "tabdpt-v1-3", "orion-msp", "kumo-tabular-small", "kumo-tabular-medium",
	                    "kumo-tabular-large"}) {
		for (auto *task : {"classification", "regression"}) {
			INFO(model << " " << task);
			REQUIRE(MlxKnownWrong(model, task).empty());
		}
	}
}
