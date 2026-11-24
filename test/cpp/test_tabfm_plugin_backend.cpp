//===----------------------------------------------------------------------===//
// Catch2 tests for the backend-plugin loader (phase 1 of
// docs/archive/DYNAMIC_BACKENDS.md).
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
#include "anofox_ggml_adapter/plugin_abi.h"
#include "duckdb/common/string_util.hpp"
#include "tabfm_weights.hpp"

#include <filesystem>
#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {

bool FileExists(const string &path) {
	std::ifstream stream(path);
	return stream.good();
}

//! The fixture's external-data graph: its 323 weights point into test/fixtures/model.safetensors, which tabfm maps and hands over by name.
TabFMPluginRequest FakeParams(const char *arch = "gfx1201") {
	TabFMPluginRequest request;
	request.graph_path = "test/fixtures/graph_ext_fixture.onnx";
	request.weights_dir = "test/fixtures";
	request.weights_path = "test/fixtures/model.safetensors";
	request.model = "tabfm-v1";
	request.cache_dir = "cache";
	request.arch = arch;
	request.precision = "fp32";
	return request;
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

	// logits[row][c] = x[row][0] + 100 (cat_mask[0] set) + c + train_size + 1000 (the graph's weights were mapped and handed over)
	for (int64_t row = 0; row < t; row++) {
		for (int64_t c = 0; c < 3; c++) {
			const float expected = x[(size_t)(row * h)] + 100.0f + (float)c + (float)train_size + 1000.0f;
			REQUIRE(output.logits[(size_t)(row * 3 + c)] == Approx(expected));
		}
	}
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
			REQUIRE(message.find("AnofoxGgmlGetPluginApi") != string::npos);
		}
	}
#endif
}

// A graph declaring an input tabfm does not feed must be refused by the HOST, before the plugin runs anything: an unfed input fails
// inside the runtime's own run call, where it cannot be reported this clearly (the removed MIGraphX plugin crashed unittest that way).
TEST_CASE("plugin_backend: a graph input tabfm does not feed is refused before create", "[tabfm][plugin]") {
	const string path = TABFM_FAKE_PLUGIN_PATH;
	if (!FileExists(path)) {
		SUCCEED("fixture plugin not built, skipping");
		return;
	}
	// tabfm feeds x, y, cat_mask, train_size and d -- every input of the classic contract
	auto params = FakeParams();
	REQUIRE(LoadPluginBackend(path, params) != nullptr);

	// a graph that also declares an input the plugin does not know: a minimal ModelProto written by hand, ir_version + a graph whose
	// inputs are the classic five plus n_rows
	auto field = [](int number, const string &payload) {
		string out(1, static_cast<char>((number << 3) | 2));
		size_t len = payload.size();
		do {
			unsigned char byte = len & 0x7f;
			len >>= 7;
			out.push_back(static_cast<char>(len ? (byte | 0x80) : byte));
		} while (len);
		return out + payload;
	};
	string graph_proto;
	for (const char *input : {"x", "y", "cat_mask", "train_size", "d", "n_rows"}) {
		graph_proto += field(11, field(1, input)); // GraphProto.input = ValueInfoProto{name}
	}
	const string model_proto = string("\x08\x08", 2) + field(7, graph_proto); // ir_version 8, ModelProto.graph
	const string extra_input_graph = (std::filesystem::temp_directory_path() / "anofox_tabfm_graph_with_n_rows.onnx").string();
	{
		std::ofstream out(extra_input_graph, std::ios::binary);
		out << model_proto;
	}
	params.graph_path = extra_input_graph;
	try {
		LoadPluginBackend(path, params);
		FAIL("expected an exception");
	} catch (std::exception &error) {
		const string message = error.what();
		REQUIRE(message.find("'n_rows'") != string::npos);
		REQUIRE(message.find("UPDATE EXTENSIONS") != string::npos);
		REQUIRE(message.find("anofox_tabfm_graph_with_n_rows.onnx") != string::npos);
	}
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
		REQUIRE(message.find(std::to_string(ANOFOX_GGML_PLUGIN_ABI_VERSION)) != string::npos);
	}
}

#endif // TABFM_FAKE_PLUGIN_BAD_ABI_PATH

// These take the platform explicitly so every combination is testable from any
// host — which is the point, since the combinations that need guarding are the
// ones the developer's machine is not.

TEST_CASE("plugin_artifacts: filenames follow each platform's convention", "[tabfm][plugin_artifacts]") {
	// cuda and rocm are both served by the ggml GPU plugin, one file per vendor
	REQUIRE(PluginFileNameFor("cuda", "linux") == "libanofox_ggml-cuda_plugin.so");
	REQUIRE(PluginFileNameFor("mlx", "osx") == "libanofox_mlx_plugin.dylib");
	// Windows DLLs carry no 'lib' prefix — that is what CMake emits there, so
	// guessing otherwise would look for a file the build never produces.
	REQUIRE(PluginFileNameFor("cuda", "windows") == "anofox_ggml-cuda_plugin.dll");

	REQUIRE(PluginFileNameFor("rocm", "linux") == "libanofox_ggml-rocm_plugin.so");
	// tabfm_download_runtime normalizes its argument once: the deprecated 'migraphx' fetches the ROCm plugin
	REQUIRE(NormalizeRuntimeBackend("migraphx") == "rocm");
	REQUIRE(NormalizeRuntimeBackend("MIGraphX") == "rocm"); // deprecated spelling
	REQUIRE(NormalizeRuntimeBackend("CUDA") == "cuda");
	REQUIRE(NormalizeRuntimeBackend("tpu") == "tpu"); // refused by name later
	// the artifact names are idempotent: asking for the plugin's own name finds the same file
	REQUIRE(PluginFileNameFor("ggml-cuda", "linux") == PluginFileNameFor("cuda", "linux"));
	REQUIRE(PluginFileNameFor("ggml-rocm", "linux") == PluginFileNameFor("rocm", "linux"));
}

TEST_CASE("plugin_artifacts: an explicit cuda or rocm request without the ggml plugin names the file and the way to get it", "[tabfm][plugin_artifacts]") {
	// With the ggml plugin as the ONLY GPU lane there is no other plugin to fall back to: the error must say which file is missing, where it looked, and
	// the two ways out (fetch it, or run on the CPU) — never the removed MIGraphX / ONNX Runtime wording.
	for (const char *vendor : {"cuda", "rocm"}) {
		INFO("vendor " << vendor);
		const auto file = PluginFileName(vendor);
		const auto message = GgmlGpuPluginMissingMessage(vendor, "/opt/tabfm/ep", file);
		REQUIRE(message.find("device '" + string(vendor) + "'") != string::npos);
		REQUIRE(message.find(file) != string::npos);
		REQUIRE(message.find("/opt/tabfm/ep") != string::npos);
		REQUIRE(message.find("tabfm_download_runtime('" + string(vendor) + "')") != string::npos);
		REQUIRE(message.find("anofox_tabfm_device='cpu'") != string::npos);
		REQUIRE(message.find("MIGraphX") == string::npos); // removed
		REQUIRE(message.find("ONNX Runtime") == string::npos);
	}
	// no ep_path configured at all is its own sentence
	const auto unset = GgmlGpuPluginMissingMessage("cuda", "", PluginFileName("cuda"));
	REQUIRE(unset.find("anofox_tabfm_ep_path is not set") != string::npos);
}

TEST_CASE("plugin_artifacts: tabfm_download_runtime('cuda') fetches only the self-contained ggml plugin, no ONNX Runtime wheel", "[tabfm][plugin_artifacts]") {
	if (HostPluginOs() != "linux" || HostPluginArch() != "amd64") {
		return; // the published platform; elsewhere the refusal tests above apply
	}
	for (const char *backend : {"cuda", "rocm"}) {
		INFO("backend " << backend);
		RuntimeArtifact artifact;
		string error;
		REQUIRE(ResolveRuntimeArtifact(backend, artifact, error));
		REQUIRE(error.empty());
		REQUIRE(artifact.plugin_name == PluginFileName(backend));
		REQUIRE(!artifact.plugin_url.empty()); // the plugin alone: no runtime archive comes with it (the struct has no field for one any more)
	}
	// and a backend nobody publishes says so by name
	RuntimeArtifact unknown;
	string unknown_error;
	REQUIRE(!ResolveRuntimeArtifact("vulkan", unknown, unknown_error));
	REQUIRE(!unknown_error.empty());
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
	// no Windows plugin is built (the ggml CUDA plugin is Linux-only so far), so
	// without this the request walked a download and ended at a missing library.
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
	auto url = PluginReleaseAssetUrl("libanofox_ggml-cuda_plugin.so", TABFM_PLUGIN_RELEASE_TAG);
	REQUIRE(url.find("https://github.com/DataZooDE/anofox-ggml-adapter/releases/download/") == 0); // the shared plugins' release
	REQUIRE(url.find(TABFM_PLUGIN_RELEASE_TAG) != string::npos);
	REQUIRE(StringUtil::EndsWith(url, "libanofox_ggml-cuda_plugin.so"));
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
	REQUIRE(PluginSha256AssetName("cuda") == "ggml-cuda_plugin.sha256");
	REQUIRE(PluginSha256AssetName("rocm") == "ggml-rocm_plugin.sha256");
	REQUIRE(PluginSha256AssetName("mlx") == "mlx_plugin.sha256");
}

TEST_CASE("plugin_artifacts: a digest is matched by FILENAME, not by position",
          "[tabfm][plugin_artifacts]") {
	// A sidecar may cover several files. Taking the first line would verify the
	// plugin against another file's digest and always fail — or, worse with the
	// operands swapped, pass on the wrong file.
	const string sidecar =
	    "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111  libanofox_ggml-cuda_plugin.so\n"
	    "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222  libanofox_tabfm_other_plugin.so\n";
	REQUIRE(Sha256FromSidecar(sidecar, "libanofox_ggml-cuda_plugin.so") ==
	        "aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111aaaa1111");
	REQUIRE(Sha256FromSidecar(sidecar, "libanofox_tabfm_other_plugin.so") ==
	        "bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222bbbb2222");
}

TEST_CASE("plugin_artifacts: a sidecar that does not mention the file vouches for nothing",
          "[tabfm][plugin_artifacts]") {
	// Must be empty, and callers must treat empty as a failure rather than as
	// "no check required" — otherwise an attacker supplies a sidecar naming
	// some other file and the verification silently passes.
	const string sidecar =
	    "cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333cccc3333  some_other_file.so\n";
	REQUIRE(Sha256FromSidecar(sidecar, "libanofox_ggml-cuda_plugin.so").empty());
	REQUIRE(Sha256FromSidecar("", "libanofox_ggml-cuda_plugin.so").empty());
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
