//===----------------------------------------------------------------------===//
// ggml CPU engine, phase P0 (docs/GGML_PLAN.md): the vendored ggml builds, links and computes, and the engine
// refuses a graph it cannot run by NAMING the missing ops (the pattern tabfm_mlx_ops.hpp set for MLX).
//
// Written red first: before vendor/ggml was wired into CMakeLists.txt, this file did not compile (no ggml.h) and
// tabfm_ggml_ops.hpp did not exist.
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "ggml-cpu.h"
#include "ggml.h"
#include "tabfm_ggml_engine.hpp"
#include "tabfm_ggml_ops.hpp"

#include <cmath>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

TEST_CASE("ggml: the vendored CPU backend computes a batched matmul", "[tabfm][ggml]") {
	// ggml_mul_mat(a, b): a ne=[k,n,batch], b ne=[k,m,batch] -> [n,m,batch], i.e. b[m x k] times a^T[k x n] row-major.
	const int64_t k = 3, n = 2, m = 2, batch = 2;
	std::vector<float> a(k * n * batch), b(k * m * batch), out(n * m * batch, -1.0f);
	for (size_t i = 0; i < a.size(); i++) {
		a[i] = static_cast<float>(i % 5) - 1.0f;
	}
	for (size_t i = 0; i < b.size(); i++) {
		b[i] = static_cast<float>((i * 3) % 7) * 0.5f;
	}
	ggml_init_params params {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
	ggml_context *ctx = ggml_init(params);
	REQUIRE(ctx != nullptr);
	ggml_tensor *ta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n, batch);
	ggml_tensor *tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, batch);
	ta->data = a.data();
	tb->data = b.data();
	ggml_tensor *r = ggml_mul_mat(ctx, ta, tb);
	r->data = out.data();
	ggml_cgraph *graph = ggml_new_graph(ctx);
	ggml_build_forward_expand(graph, r);
	ggml_cplan plan = ggml_graph_plan(graph, 2, nullptr);
	std::vector<uint8_t> work(plan.work_size);
	plan.work_data = work.data();
	REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
	ggml_free(ctx);

	for (int64_t bi = 0; bi < batch; bi++) {
		for (int64_t mi = 0; mi < m; mi++) {
			for (int64_t ni = 0; ni < n; ni++) {
				double want = 0;
				for (int64_t ki = 0; ki < k; ki++) {
					want += static_cast<double>(b[(bi * m + mi) * k + ki]) * a[(bi * n + ni) * k + ki];
				}
				REQUIRE(std::fabs(out[(bi * m + mi) * n + ni] - want) < 1e-5);
			}
		}
	}
}

TEST_CASE("ggml_ops: GgmlMissingOps names exactly the ops the engine lacks, sorted and unique", "[tabfm][ggml]") {
	REQUIRE(GgmlMissingOps({}).empty());
	REQUIRE(GgmlMissingOps({"Identity", "Identity"}).empty());
	const auto missing = GgmlMissingOps({"Identity", "Zebra", "Aardvark", "Zebra"});
	REQUIRE(missing == std::vector<std::string> {"Aardvark", "Zebra"});
}

TEST_CASE("ggml_engine: a graph with unsupported ops is refused with the op names and the way out", "[tabfm][ggml]") {
	REQUIRE(GgmlRefusal({"Identity"}).empty());
	const std::string message = GgmlRefusal({"Identity", "Zebra"});
	REQUIRE(message.find("Zebra") != std::string::npos);
	REQUIRE(message.find("Identity") == std::string::npos); // only the missing ones are named
	REQUIRE(message.find("TABFM_CPU_ENGINE=ort") != std::string::npos);
}
