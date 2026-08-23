//===----------------------------------------------------------------------===//
// Catch2 tests for the backend-plugin loader (phase 1 of
// docs/DYNAMIC_BACKENDS.md).
//
// The loader is the trust boundary: on the far side of it a mismatched ABI is
// undefined behaviour rather than a wrong answer, so every refusal is asserted
// here against a REAL shared library rather than a mock. The fixture plugin
// (test/cpp/plugin_fixture/fake_plugin.cpp) is built twice — once correct, once
// with a deliberately wrong ABI version.
//
// The paths come from compile definitions set in CMakeLists.txt; the cases skip
// when those are absent (a generator that cannot build the fixture library).
//===----------------------------------------------------------------------===//

#include "catch.hpp"

#include "tabfm_plugin_backend.hpp"

#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

bool FileExists(const string &path) {
	std::ifstream stream(path);
	return stream.good();
}

TabFMPluginCreateParams FakeParams(const char *arch = "gfx1201") {
	TabFMPluginCreateParams params {};
	params.graph_path = "graph.onnx";
	params.weights_dir = "weights";
	params.cache_dir = "cache";
	params.arch = arch;
	params.precision = "bf16";
	params.mxr_source = "";
	params.device_ordinal = 0;
	return params;
}

} // namespace

#ifdef TABFM_FAKE_PLUGIN_PATH

TEST_CASE("plugin_backend: loads a plugin and round-trips a forward pass", "[tabfm][plugin]") {
	const string path = TABFM_FAKE_PLUGIN_PATH;
	if (!FileExists(path)) {
		SUCCEED("fixture plugin not built, skipping");
		return;
	}

	auto backend = LoadPluginBackend(path, FakeParams());
	REQUIRE(backend != nullptr);

	// Inputs the fixture reads back into its output, so a mis-marshalled
	// pointer or a wrong struct offset shows up as a wrong number rather than
	// as silence.
	const int64_t t = 4, h = 2, train_size = 3;
	vector<float> x {10.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f, 40.0f, 0.0f};
	vector<float> y {0.0f, 1.0f, 0.0f};
	vector<bool> mask_bits {true, false};
	auto cat_mask = make_unsafe_uniq_array<bool>(2);
	cat_mask[0] = true;
	cat_mask[1] = false;

	TabFMRunInput input;
	input.x = x.data();
	input.y = y.data();
	input.cat_mask = cat_mask.get();
	input.t = t;
	input.h = h;
	input.train_size = train_size;
	input.d = h;

	auto output = backend->Run(input);

	REQUIRE(output.shape.size() == 3);
	REQUIRE(output.shape[0] == 1);
	REQUIRE(output.shape[1] == t);
	REQUIRE(output.shape[2] == 3);
	REQUIRE(output.logits.size() == (size_t)(t * 3));

	// logits[row][c] = x[row][0] + 100 (cat_mask[0] set) + c + train_size
	for (int64_t row = 0; row < t; row++) {
		for (int64_t c = 0; c < 3; c++) {
			const float expected = x[(size_t)(row * h)] + 100.0f + (float)c + (float)train_size;
			REQUIRE(output.logits[(size_t)(row * 3 + c)] == Approx(expected));
		}
	}

	// Precompile forwards without throwing.
	backend->Precompile(128, 16);
}

TEST_CASE("plugin_backend: a backend that refuses to initialise reports why", "[tabfm][plugin]") {
	const string path = TABFM_FAKE_PLUGIN_PATH;
	if (!FileExists(path)) {
		SUCCEED("fixture plugin not built, skipping");
		return;
	}
	try {
		LoadPluginBackend(path, FakeParams("refuse"));
		FAIL("expected an exception");
	} catch (std::exception &error) {
		string message = error.what();
		// The plugin's own diagnosis has to survive the boundary — an exception
		// cannot cross dlopen, so it travels as a status plus a buffer.
		REQUIRE(message.find("no device matching arch 'refuse'") != string::npos);
		REQUIRE(message.find("fake") != string::npos);
	}
}

TEST_CASE("plugin_backend: a missing library names the fix", "[tabfm][plugin]") {
	try {
		LoadPluginBackend("/nonexistent/libanofox_tabfm_nothing.so", FakeParams());
		FAIL("expected an exception");
	} catch (std::exception &error) {
		string message = error.what();
		REQUIRE(message.find("cannot load the backend plugin") != string::npos);
		REQUIRE(message.find("tabfm_download_runtime") != string::npos);
	}
}

TEST_CASE("plugin_backend: a library that is not a plugin is refused", "[tabfm][plugin]") {
	// The extension itself is a perfectly good shared library that exports no
	// plugin entry point — exactly the "right name, wrong contents" case.
	const string path = TABFM_FAKE_PLUGIN_PATH;
	if (!FileExists(path)) {
		SUCCEED("fixture plugin not built, skipping");
		return;
	}
	// Point at a real ELF/dylib that is not a plugin: the test binary itself.
#ifdef TABFM_NOT_A_PLUGIN_PATH
	const string other = TABFM_NOT_A_PLUGIN_PATH;
	if (FileExists(other)) {
		try {
			LoadPluginBackend(other, FakeParams());
			FAIL("expected an exception");
		} catch (std::exception &error) {
			string message = error.what();
			REQUIRE(message.find("is not an anofox backend plugin") != string::npos);
			REQUIRE(message.find("TabFMGetPluginApi") != string::npos);
		}
	}
#endif
}

#endif // TABFM_FAKE_PLUGIN_PATH

#ifdef TABFM_FAKE_PLUGIN_BAD_ABI_PATH

TEST_CASE("plugin_backend: an ABI mismatch is refused before anything is read", "[tabfm][plugin]") {
	const string path = TABFM_FAKE_PLUGIN_BAD_ABI_PATH;
	if (!FileExists(path)) {
		SUCCEED("bad-ABI fixture plugin not built, skipping");
		return;
	}
	// This is the case that would otherwise corrupt memory silently: every
	// field after abi_version would sit at the wrong offset. The check must
	// happen before the table is otherwise touched, and must say both versions.
	try {
		LoadPluginBackend(path, FakeParams());
		FAIL("expected an exception");
	} catch (std::exception &error) {
		string message = error.what();
		REQUIRE(message.find("plugin ABI version") != string::npos);
		REQUIRE(message.find(std::to_string(TABFM_PLUGIN_ABI_VERSION)) != string::npos);
	}
}

#endif // TABFM_FAKE_PLUGIN_BAD_ABI_PATH

#include "tabfm_soname_patch.hpp"

TEST_CASE("soname_patch: exactly the NUL-terminated SONAME is rewritten", "[tabfm][plugin]") {
	// Half of the SONAME-shadowing fix (GPU_HARDENING_PLAN S2). The dangerous
	// mistakes are (a) rewriting a longer string the SONAME merely prefixes and
	// (b) missing an occurrence and shipping a half-renamed core — both pinned.
	using namespace duckdb::anofox;

	SECTION("the plain case: one .dynstr-style entry") {
		char buffer[] = "xx\0libonnxruntime.so.1\0yy";
		REQUIRE(PatchOrtSonameInPlace(buffer, sizeof(buffer)) == 1);
		REQUIRE(std::memcmp(buffer + 3, ORT_RENAMED_SONAME, 19) == 0);
		REQUIRE(buffer[3 + 19] == '\0'); // terminator untouched
		REQUIRE(buffer[sizeof(buffer) - 3] == 'y');
	}
	SECTION("a longer string it prefixes is NOT touched") {
		char buffer[] = "libonnxruntime.so.1.28.0\0";
		REQUIRE(PatchOrtSonameInPlace(buffer, sizeof(buffer)) == 0);
		REQUIRE(std::memcmp(buffer, "libonnxruntime.so.1.28.0", 24) == 0);
	}
	SECTION("every occurrence is counted, so the caller can refuse a drifted wheel") {
		char buffer[] = "libonnxruntime.so.1\0mid\0libonnxruntime.so.1\0";
		REQUIRE(PatchOrtSonameInPlace(buffer, sizeof(buffer)) == 2);
	}
	SECTION("absent, empty, and too-small inputs are zero, never a crash") {
		char buffer[] = "nothing to see";
		REQUIRE(PatchOrtSonameInPlace(buffer, sizeof(buffer)) == 0);
		REQUIRE(PatchOrtSonameInPlace(nullptr, 0) == 0);
		char tiny[] = "lib";
		REQUIRE(PatchOrtSonameInPlace(tiny, sizeof(tiny)) == 0);
	}
	SECTION("a match at the very end of the buffer is found") {
		char buffer[] = "pad\0libonnxruntime.so.1\0";
		REQUIRE(PatchOrtSonameInPlace(buffer, sizeof(buffer) - 0) == 1);
	}
}

#include "tabfm_mxr_cache_key.hpp"

TEST_CASE("mxr cache key: different graph content never collides; same content shares", "[tabfm][plugin_backend]") {
	using anofox_tabfm_mxr::Fnv1a64;
	using anofox_tabfm_mxr::MxrCacheStem;
	// The 2026-08-22 bug pair: a stem-only key let mitra load tabfm-v1's
	// compiled program (wrong answers, no error); the first fix hashed the
	// PATH, which silently broke anofox_tabfm_mxr_source sharing across
	// machines. Content hashing satisfies both: different models' graphs
	// differ in bytes, the same graph anywhere hashes identically.
	auto tabfm_bytes = std::string("pretend-tabfm-graph-bytes");
	auto mitra_bytes = std::string("pretend-mitra-graph-bytes");
	auto here = MxrCacheStem("/cache/a/graph_migraphx_classification.onnx", Fnv1a64(tabfm_bytes));
	auto there = MxrCacheStem("/other/machine/graph_migraphx_classification.onnx", Fnv1a64(tabfm_bytes));
	auto mitra = MxrCacheStem("/cache/b/graph_migraphx_classification.onnx", Fnv1a64(mitra_bytes));
	REQUIRE(here == there); // same bytes, any path, any machine
	REQUIRE(here != mitra); // different model, same filename
	REQUIRE(here.rfind("graph_migraphx_classification_", 0) == 0);
	REQUIRE(here.size() == std::string("graph_migraphx_classification_").size() + 8);
}

TEST_CASE("mxr cache key: stem extraction handles plain names and extensionless paths", "[tabfm][plugin_backend]") {
	using anofox_tabfm_mxr::MxrCacheStem;
	REQUIRE(MxrCacheStem("graph.onnx", 1).rfind("graph_", 0) == 0);
	REQUIRE(MxrCacheStem("/a.b/dir/noext", 1).rfind("noext_", 0) == 0);
}
