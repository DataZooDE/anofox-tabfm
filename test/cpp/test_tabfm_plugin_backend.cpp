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
#include "tabfm_plugin_artifacts.hpp"

#include <filesystem>
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

// The SECOND half of the 2026-08-22 bug, which the content hash above did not close.
//
// A graph is weight-FREE: it references `model.safetensors` by offset, and the
// compiled .mxr bakes the weight VALUES in. Hashing the graph bytes therefore
// separates models whose graphs differ -- and says nothing about two models that
// share a byte-identical graph and differ only in weights. tabpfn-v2-5 and
// tabpfn-v2-5-real are exactly that pair: same architecture, same graph, different
// checkpoint. On ROCm the second one asked for silently received the first one's
// compiled program -- GPU real vs GPU regular differed by 0.0, CPU real vs CPU
// regular by 0.115 -- with no error and a plausible answer, which is the failure
// mode this repo has the most scar tissue around.
TEST_CASE("mxr cache key: same graph with different weights never collides", "[tabfm][plugin_backend]") {
	using anofox_tabfm_mxr::Fnv1a64;
	using anofox_tabfm_mxr::MxrCacheStem;
	using anofox_tabfm_mxr::WeightsFingerprint;
	namespace fs = std::filesystem;

	const auto root = fs::temp_directory_path() / "tabfm_weights_fp_test";
	fs::remove_all(root);
	fs::create_directories(root / "a");
	fs::create_directories(root / "b");
	fs::create_directories(root / "c");
	auto write = [](const fs::path &p, const std::string &bytes) {
		std::ofstream(p, std::ios::binary) << bytes;
	};
	// Big enough that the fingerprint SAMPLES rather than reading everything, and
	// large enough that a difference in one block is a real test of the sampling.
	std::string base(4096 * 200, 'x');
	auto with_byte_at = [&](size_t at, char c) {
		auto s2 = base;
		s2[at] = c;
		return s2;
	};

	write(root / "a" / "model.safetensors", base);
	write(root / "b" / "model.safetensors", base);                          // identical content, other path
	write(root / "c" / "model.safetensors", with_byte_at(0, 'y'));         // differs in the first block
	fs::create_directories(root / "d");
	write(root / "d" / "model.safetensors", base + "extra");               // different SIZE, same prefix
	fs::create_directories(root / "e");
	// Differs only in a block the sampler visits (first, last and 62 between).
	write(root / "e" / "model.safetensors", with_byte_at(base.size() - 1, 'z'));

	const auto fa = WeightsFingerprint((root / "a" / "model.safetensors").string());
	REQUIRE(fa != 0);
	// same content anywhere shares (keeps anofox_tabfm_mxr_source working across machines)
	REQUIRE(fa == WeightsFingerprint((root / "b" / "model.safetensors").string()));
	// different content, same size, must not
	REQUIRE(fa != WeightsFingerprint((root / "c" / "model.safetensors").string()));
	REQUIRE(fa != WeightsFingerprint((root / "e" / "model.safetensors").string()));
	// different size must not
	REQUIRE(fa != WeightsFingerprint((root / "d" / "model.safetensors").string()));
	// a missing file is a defined value, not an exception
	REQUIRE(WeightsFingerprint((root / "nope").string()) == 0);

	// ...and it must actually reach the key: same graph bytes, different weights.
	const auto graph = Fnv1a64("one-shared-graph");
	const auto regular = MxrCacheStem("/w/graph_migraphx_tabpfn25_classification.onnx", graph, fa);
	const auto real =
	    MxrCacheStem("/w/graph_migraphx_tabpfn25_classification.onnx", graph,
	                 WeightsFingerprint((root / "c" / "model.safetensors").string()));
	const auto regular_elsewhere = MxrCacheStem("/other/graph_migraphx_tabpfn25_classification.onnx", graph,
	                                            WeightsFingerprint((root / "b" / "model.safetensors").string()));
	REQUIRE(regular != real);              // the bug
	REQUIRE(regular == regular_elsewhere); // sharing across machines still works
	fs::remove_all(root);
}

TEST_CASE("mxr cache key: stem extraction handles plain names and extensionless paths", "[tabfm][plugin_backend]") {
	using anofox_tabfm_mxr::MxrCacheStem;
	REQUIRE(MxrCacheStem("graph.onnx", 1).rfind("graph_", 0) == 0);
	REQUIRE(MxrCacheStem("/a.b/dir/noext", 1).rfind("noext_", 0) == 0);
}

//===----------------------------------------------------------------------===//
// Plugin artifacts: filenames and published-platform refusals
//===----------------------------------------------------------------------===//


// These take the platform explicitly so every combination is testable from any
// host — which is the point, since the combinations that need guarding are the
// ones the developer's machine is not.

TEST_CASE("plugin_artifacts: filenames follow each platform's convention", "[tabfm][plugin_artifacts]") {
	REQUIRE(PluginFileNameFor("cuda", "linux") == "libanofox_tabfm_cuda_plugin.so");
	REQUIRE(PluginFileNameFor("mlx", "osx") == "libanofox_tabfm_mlx_plugin.dylib");
	// Windows DLLs carry no 'lib' prefix — that is what CMake emits there, so
	// guessing otherwise would look for a file the build never produces.
	REQUIRE(PluginFileNameFor("cuda", "windows") == "anofox_tabfm_cuda_plugin.dll");

	// 'rocm' is the device; MIGraphX is the library that drives it, and the
	// artifact has always been named for the library.
	REQUIRE(PluginFileNameFor("rocm", "linux") == "libanofox_tabfm_migraphx_plugin.so");
	REQUIRE(PluginFileNameFor("migraphx", "linux") == PluginFileNameFor("rocm", "linux"));
}

TEST_CASE("plugin_artifacts: the published platform for each backend is the only one accepted",
          "[tabfm][plugin_artifacts]") {
	// What IS published and hardware-verified.
	REQUIRE(UnsupportedPluginPlatform("cuda", "linux", "amd64").empty());
	REQUIRE(UnsupportedPluginPlatform("rocm", "linux", "amd64").empty());
	REQUIRE(UnsupportedPluginPlatform("mlx", "osx", "arm64").empty());
}

TEST_CASE("plugin_artifacts: Windows CUDA is refused before the download, not after",
          "[tabfm][plugin_artifacts]") {
	// Windows discovers an NVIDIA card through NVML like any other platform,
	// so a user can reasonably ask for 'cuda' there. The wheel is manylinux and
	// no Windows plugin is built, so without this the request walked a 475 MB
	// download and ended at a missing library.
	auto refusal = UnsupportedPluginPlatform("cuda", "windows", "amd64");
	REQUIRE(!refusal.empty());
	REQUIRE(refusal.find("windows/amd64") != string::npos);
	REQUIRE(refusal.find("linux/amd64 only") != string::npos);
	// and names something the user can actually do next
	REQUIRE(refusal.find("anofox_tabfm_device='cpu'") != string::npos);
}

TEST_CASE("plugin_artifacts: an unavailable backend points at the one that works there",
          "[tabfm][plugin_artifacts]") {
	// Apple Silicon asking for CUDA is told about MLX, not merely refused:
	// MLX is published for exactly this machine and serves every model.
	auto on_mac = UnsupportedPluginPlatform("cuda", "osx", "arm64");
	REQUIRE(!on_mac.empty());
	REQUIRE(on_mac.find("tabfm_download_runtime('mlx')") != string::npos);

	// and Linux asking for MLX is told about the two backends that do run there
	auto on_linux = UnsupportedPluginPlatform("mlx", "linux", "amd64");
	REQUIRE(!on_linux.empty());
	REQUIRE(on_linux.find("'cuda'") != string::npos);
	REQUIRE(on_linux.find("'rocm'") != string::npos);
}

TEST_CASE("plugin_artifacts: linux arm64 gets no GPU plugin, and is told so", "[tabfm][plugin_artifacts]") {
	// linux_arm64 ships a CPU extension, so the refusal must not read as a bug.
	REQUIRE(!UnsupportedPluginPlatform("cuda", "linux", "arm64").empty());
	REQUIRE(!UnsupportedPluginPlatform("rocm", "linux", "arm64").empty());
}

TEST_CASE("plugin_artifacts: an unknown backend is rejected by name", "[tabfm][plugin_artifacts]") {
	auto refusal = UnsupportedPluginPlatform("tpu", "linux", "amd64");
	REQUIRE(refusal.find("unknown backend 'tpu'") != string::npos);
}

TEST_CASE("plugin_artifacts: release asset urls are built from the single pinned tag",
          "[tabfm][plugin_artifacts]") {
	// One definition of the tag; a stale one is caught by the loader's ABI
	// check rather than by anyone remembering to bump it.
	REQUIRE(string(TABFM_PLUGIN_RELEASE_TAG).find("v20") == 0);
	auto url = PluginReleaseAssetUrl("libanofox_tabfm_cuda_plugin.so", TABFM_PLUGIN_RELEASE_TAG);
	REQUIRE(url.find("https://github.com/DataZooDE/anofox-tabfm/releases/download/") == 0);
	REQUIRE(url.find(TABFM_PLUGIN_RELEASE_TAG) != string::npos);
	REQUIRE(StringUtil::EndsWith(url, "libanofox_tabfm_cuda_plugin.so"));
	// An empty tag yields no url at all rather than a 404-shaped one.
	REQUIRE(PluginReleaseAssetUrl("x.so", "").empty());
}

TEST_CASE("plugin_artifacts: this host resolves to a platform it knows", "[tabfm][plugin_artifacts]") {
	const auto os = HostPluginOs();
	REQUIRE((os == "linux" || os == "osx" || os == "windows"));
	// The filename used by dispatch and by the download must be the same one.
	REQUIRE(PluginFileName("cuda") == PluginFileNameFor("cuda", os));
}

//===----------------------------------------------------------------------===//
// Plugin integrity: the sha256 sidecar
//===----------------------------------------------------------------------===//

TEST_CASE("plugin_artifacts: the sidecar asset is named for the backend's base name",
          "[tabfm][plugin_artifacts]") {
	// CI writes `sha256sum <files> | tee <base>_plugin.sha256`.
	REQUIRE(PluginSha256AssetName("cuda") == "cuda_plugin.sha256");
	REQUIRE(PluginSha256AssetName("rocm") == "migraphx_plugin.sha256");
	REQUIRE(PluginSha256AssetName("mlx") == "mlx_plugin.sha256");
}

TEST_CASE("plugin_artifacts: a digest is matched by FILENAME, not by position",
          "[tabfm][plugin_artifacts]") {
	// The real CUDA sidecar covers two files. Taking the first line would
	// verify the plugin against the ORT core's digest and always fail — or,
	// worse with the operands swapped, pass on the wrong file.
	const string sidecar =
	    "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111  libanofox_tabfm_cuda_plugin.so\n"
	    "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222  libanofoxort_gpu.so\n";
	REQUIRE(Sha256FromSidecar(sidecar, "libanofox_tabfm_cuda_plugin.so") ==
	        "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111");
	REQUIRE(Sha256FromSidecar(sidecar, "libanofoxort_gpu.so") ==
	        "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222");
}

TEST_CASE("plugin_artifacts: a sidecar that does not mention the file vouches for nothing",
          "[tabfm][plugin_artifacts]") {
	// Must be empty, and callers must treat empty as a failure rather than as
	// "no check required" — otherwise an attacker supplies a sidecar naming
	// some other file and the verification silently passes.
	const string sidecar =
	    "cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333  some_other_file.so\n";
	REQUIRE(Sha256FromSidecar(sidecar, "libanofox_tabfm_cuda_plugin.so").empty());
	REQUIRE(Sha256FromSidecar("", "libanofox_tabfm_cuda_plugin.so").empty());
	REQUIRE(Sha256FromSidecar("not a sidecar at all\n", "x.so").empty());
}

TEST_CASE("plugin_artifacts: sidecar parsing tolerates the formats sha256sum emits",
          "[tabfm][plugin_artifacts]") {
	const string hash = "dddd4444dddd4444dddd4444dddd4444dddd4444dddd4444dddd4444dddd4444";
	// binary marker
	REQUIRE(Sha256FromSidecar(hash + " *plug.so\n", "plug.so") == hash);
	// a leading path, as `sha256sum dir/file` writes it
	REQUIRE(Sha256FromSidecar(hash + "  build/plug.so\n", "plug.so") == hash);
	// CRLF, and no trailing newline
	REQUIRE(Sha256FromSidecar(hash + "  plug.so\r\n", "plug.so") == hash);
	REQUIRE(Sha256FromSidecar(hash + "  plug.so", "plug.so") == hash);
	// uppercase digests compare equal to our lowercase hex
	REQUIRE(Sha256FromSidecar(StringUtil::Upper(hash) + "  plug.so\n", "plug.so") == hash);
}

// An OLDER plugin refusing a NEWER graph must say what to do about it.
//
// The bundled TabPFN MIGraphX graphs take a fifth input, `n_rows`, that the
// plugin binds only if the graph declares it. A plugin built before that binding
// existed compiles the graph fine and then dies on the first predict with MIGraphX's
// own "Parameter not found: n_rows" -- accurate, and useless to someone who has
// never heard of n_rows. The likely victim is anyone who updates the extension
// but keeps an old plugin in a custom anofox_tabfm_ep_path.
TEST_CASE("plugin_backend: an old plugin refusing a newer graph says to update it", "[tabfm][plugin]") {
	const string migraphx_msg = "MIGraphX inference failed on gfx1201: "
	                            "/usr/src/debug/migraphx/migraphx/src/program.cpp:491: operator(): "
	                            "Parameter not found: n_rows";

	auto hint = PluginFailureHint("migraphx", migraphx_msg);
	REQUIRE_FALSE(hint.empty());
	// names the fix, both spellings, and the cause
	REQUIRE(hint.find("tabfm_accelerate") != string::npos);
	REQUIRE(hint.find("tabfm_download_runtime") != string::npos);
	REQUIRE(hint.find("n_rows") != string::npos);

	// only the migraphx backend, and only for this failure: an unrelated
	// migraphx error must not be blamed on the plugin version, and another
	// backend that happens to mention n_rows is a different problem.
	REQUIRE(PluginFailureHint("migraphx", "MIGraphX inference failed on gfx1201: out of memory").empty());
	REQUIRE(PluginFailureHint("cuda", migraphx_msg).empty());
	REQUIRE(PluginFailureHint("migraphx", "").empty());
}
