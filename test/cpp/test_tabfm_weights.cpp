#include "catch.hpp"

#include "tabfm_weights.hpp"
#include "tabfm_predict.hpp" // ResolveEpPath — shared by download and dispatch
#include "tabfm_ckpt.hpp"
#include "tabfm_model_spec.hpp"

#include "duckdb/common/file_system.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace duckdb;
using namespace duckdb::anofox;

// A gated HuggingFace repo answers an anonymous GET with 401 (no token) or 403
// (token present, license not accepted). Both must become an actionable error
// naming the exact SQL that fixes them (SQL-API §5) — the raw httpfs text names
// neither the secret nor the license page.

TEST_CASE("tabfm_weights: HTTP 401 maps to an actionable remediation", "[tabfm][weights]") {
	const string url = "https://huggingface.co/Prior-Labs/tabpfn_2_5/resolve/main/model.ckpt";
	auto hint = HttpAuthRemediation("HTTP GET error: HTTP 401 (Unauthorized)", url);
	REQUIRE(!hint.empty());
	// names the fix: the secret, the token parameter, and the scope
	REQUIRE(hint.find("CREATE SECRET") != string::npos);
	REQUIRE(hint.find("BEARER_TOKEN") != string::npos);
	REQUIRE(hint.find("TYPE http") != string::npos);
	// and points at the repo whose license must be accepted
	REQUIRE(hint.find("Prior-Labs/tabpfn_2_5") != string::npos);
	// never leaks the resolve/ path or a query string
	REQUIRE(hint.find("resolve/main") == string::npos);
}

TEST_CASE("tabfm_weights: HTTP 403 also maps, and mentions license acceptance", "[tabfm][weights]") {
	const string url = "https://huggingface.co/stable-ai/LimiX-2M/resolve/main/LimiX-2M.ckpt";
	auto hint = HttpAuthRemediation("HTTP GET error on ... (HTTP 403 Forbidden)", url);
	REQUIRE(!hint.empty());
	REQUIRE(hint.find("stable-ai/LimiX-2M") != string::npos);
	REQUIRE(hint.find("license") != string::npos);
}

TEST_CASE("tabfm_weights: non-auth failures are not rewritten", "[tabfm][weights]") {
	const string url = "https://huggingface.co/autogluon/mitra-classifier/resolve/main/model.safetensors";
	// 404, connection refused, and the httpfs-missing case must pass through
	// untouched — rewriting them would hide their own remediation.
	REQUIRE(HttpAuthRemediation("HTTP GET error: HTTP 404 (Not Found)", url).empty());
	REQUIRE(HttpAuthRemediation("Connection refused", url).empty());
	REQUIRE(HttpAuthRemediation("Extension \"httpfs\" not found", url).empty());
	// a bare "403" inside an unrelated word/number must not trigger it
	REQUIRE(HttpAuthRemediation("wrote 4031 of 5000 bytes", url).empty());
}

TEST_CASE("tabfm_weights: a non-HuggingFace host still gets the secret hint", "[tabfm][weights]") {
	auto hint = HttpAuthRemediation("HTTP 401", "https://models.example.com/a/b/weights.safetensors");
	REQUIRE(!hint.empty());
	REQUIRE(hint.find("CREATE SECRET") != string::npos);
	REQUIRE(hint.find("models.example.com") != string::npos);
	// no HuggingFace-specific license line for a generic host
	REQUIRE(hint.find("huggingface.co/") == string::npos);
}

//===----------------------------------------------------------------------===//
// ep_path resolution
//
// tabfm_download_runtime has always written the plugin into
// <cache_dir>/runtime, but the two predict bind paths read
// anofox_tabfm_ep_path raw and dispatch throws on an empty one — so
// download-then-predict failed until the user also issued a SET naming the
// directory the download had just chosen for them. One shared resolver gives
// both sides the same default, and the SET goes back to being an override.
//===----------------------------------------------------------------------===//

TEST_CASE("tabfm_weights: an unset ep_path resolves to the directory downloads land in",
          "[tabfm][weights][ep_path]") {
	// The empty setting is the default (anofox_tabfm_ep_path defaults to ''),
	// so this is the path every user who never issued a SET takes.
	REQUIRE(ResolveEpPath("", "/home/u/.cache/anofox-tabfm") == "/home/u/.cache/anofox-tabfm/runtime");
}

TEST_CASE("tabfm_weights: a set ep_path overrides the default and is ~-expanded",
          "[tabfm][weights][ep_path]") {
	REQUIRE(ResolveEpPath("/opt/plugins", "/home/u/.cache/anofox-tabfm") == "/opt/plugins");
	// ~ must expand here exactly as it does for cache_dir: the setting is a
	// directory a human types.
	const auto expanded = ResolveEpPath("~/plugins", "/home/u/.cache/anofox-tabfm");
	REQUIRE(expanded.find('~') == string::npos);
	REQUIRE(StringUtil::EndsWith(expanded, "/plugins"));
}

TEST_CASE("tabfm_weights: ep_path resolution never yields an empty directory",
          "[tabfm][weights][ep_path]") {
	// Dispatch throws on an empty ep_path naming a SET as the fix. With a
	// default in place that message becomes unreachable through the settings
	// path, so no combination of blank inputs may produce one.
	REQUIRE(!ResolveEpPath("", "/c").empty());
	REQUIRE(!ResolveEpPath("   ", "/c").empty());
}

//===----------------------------------------------------------------------===//
// Stale converted weights beside a checkpoint (2026-11-12, ROCm probe). A cache whose model.safetensors was
// converted by an older release has another header than the bundled GPU graph's, and only tabfm_download ran
// the converter, so every GPU path compared the stale header and refused the model, advising a graph
// registration that could not help. RefreshConvertedWeights runs first on those paths now.
//===----------------------------------------------------------------------===//

namespace {
namespace stdfs = std::filesystem;

std::string SlurpFile(const stdfs::path &p) {
	std::ifstream in(p, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

//! A valid safetensors file with a header no bundled graph was built against.
void WriteStaleSafetensors(const stdfs::path &p) {
	std::string json = R"({"stale.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
	while (json.size() % 8 != 0) {
		json += ' ';
	}
	const uint64_t n = json.size();
	std::ofstream out(p, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char *>(&n), 8);
	out << json;
	const float one = 1.0f;
	out.write(reinterpret_cast<const char *>(&one), 4);
}

//! A fresh directory holding the tiny fixture checkpoint as <dir>/model.ckpt, removed again when the test
//! ends (also on a failed REQUIRE). Process-unique, so concurrent test runs cannot collide.
struct CkptDir {
	stdfs::path dir;
	explicit CkptDir(const char *name) {
#ifdef _WIN32
		const auto pid = std::to_string(_getpid());
#else
		const auto pid = std::to_string(getpid());
#endif
		dir = stdfs::temp_directory_path() / (std::string(name) + "_" + pid);
			stdfs::create_directories(dir);
		stdfs::copy_file("test/fixtures/ckpt/tiny.ckpt", dir / "model.ckpt");
	}
	~CkptDir() {
		std::error_code ec;
		stdfs::permissions(dir, stdfs::perms::owner_all, stdfs::perm_options::add, ec);
		stdfs::remove_all(dir, ec);
	}
	stdfs::path operator/(const char *leaf) const {
		return dir / leaf;
	}
};

bool AnyPartFile(const stdfs::path &dir) {
	for (auto &entry : stdfs::directory_iterator(dir)) {
		if (entry.path().filename().string().find(".part") != std::string::npos) {
			return true;
		}
	}
	return false;
}

//! The header the native converter writes for the tiny checkpoint: what a bundled graph built against a
//! fresh conversion of it would pin.
string TinyConvertedHeaderSha() {
	const auto bytes = SlurpFile("test/fixtures/ckpt/tiny.ckpt");
	auto sd = ReadTorchCkpt(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
	const auto header = CkptSafetensorsHeader(sd);
	return Sha256Hex(const_data_ptr_cast(header.data()), header.size());
}
} // namespace

TEST_CASE("tabfm_weights: a stale model.safetensors beside a .ckpt is reconverted before its header is judged",
          "[tabfm][weights][stale_weights]") {
	const CkptDir dir("tabfm_stale_weights_reconvert");
	WriteStaleSafetensors(dir / "model.safetensors");
	auto fs = FileSystem::CreateLocal();
	const auto expected = TinyConvertedHeaderSha();
	const auto sibling = (dir / "model.safetensors").string();
	REQUIRE(WeightsHeaderSha(*fs, sibling) != expected); // the precondition of the bug

	auto first = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), expected, string());
	REQUIRE(first.state == ConvertedWeightsState::CONVERTED);
	REQUIRE(first.path == sibling);
	REQUIRE(WeightsHeaderSha(*fs, sibling) == expected);
	REQUIRE(first.bytes == NumericCast<int64_t>(stdfs::file_size(dir / "model.safetensors")));
	REQUIRE(!AnyPartFile(dir.dir));

	// Idempotent: a matching file is left alone.
	auto second = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), expected, string());
	REQUIRE(second.state == ConvertedWeightsState::MATCHES);
}

TEST_CASE("tabfm_weights: a missing converted file is written from the checkpoint",
          "[tabfm][weights][stale_weights]") {
	const CkptDir dir("tabfm_stale_weights_missing");
	auto fs = FileSystem::CreateLocal();
	const auto expected = TinyConvertedHeaderSha();
	auto res = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), expected, string());
	REQUIRE(res.state == ConvertedWeightsState::CONVERTED);
	REQUIRE(WeightsHeaderSha(*fs, (dir / "model.safetensors").string()) == expected);
}

TEST_CASE("tabfm_weights: a checkpoint that cannot produce the pinned header leaves the stale file alone",
          "[tabfm][weights][stale_weights]") {
	// TabPFN v2 / 2.5: their bundled graphs want the output of the Python convert step (split and transposed
	// attention weights), which the native converter cannot produce. Overwriting the file with a conversion
	// no graph reads would only make it worse; the caller reports STALE and says how to fix it.
	const CkptDir dir("tabfm_stale_weights_unconvertible");
	WriteStaleSafetensors(dir / "model.safetensors");
	const auto before = SlurpFile(dir / "model.safetensors");
	auto fs = FileSystem::CreateLocal();
	const string other(64, '0');
	auto res = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), other, string());
	REQUIRE(res.state == ConvertedWeightsState::STALE);
	REQUIRE(res.ckpt_present); // the remedy then names the Python step, not a re-download
	REQUIRE(SlurpFile(dir / "model.safetensors") == before);
	// Memoized by file sizes: asking again gives the same answer without reconverting.
	REQUIRE(RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), other, string()).state ==
	        ConvertedWeightsState::STALE);

	// No checkpoint to reconvert from (convert_weights.py deletes its transient copy): STALE, not present.
	stdfs::rename(dir / "model.ckpt", dir / "moved.bin");
	res = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), other, string());
	REQUIRE(res.state == ConvertedWeightsState::STALE);
	REQUIRE(!res.ckpt_present);
	stdfs::rename(dir / "moved.bin", dir / "model.ckpt");

	stdfs::remove(dir / "model.safetensors");
	res = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), other, string());
	REQUIRE(res.state == ConvertedWeightsState::UNCONVERTIBLE);
	REQUIRE(!stdfs::exists(dir / "model.safetensors"));
}

TEST_CASE("tabfm_weights: a declared safetensors or a model without a pinned header is not touched",
          "[tabfm][weights][stale_weights]") {
	const CkptDir dir("tabfm_stale_weights_na");
	WriteStaleSafetensors(dir / "model.safetensors");
	auto fs = FileSystem::CreateLocal();
	REQUIRE(RefreshConvertedWeights(*fs, (dir / "model.safetensors").string(), TinyConvertedHeaderSha(), string())
	            .state == ConvertedWeightsState::NOT_APPLICABLE);
	REQUIRE(RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), string(), string()).state ==
	        ConvertedWeightsState::NOT_APPLICABLE);
	REQUIRE(WeightsHeaderSha(*fs, (dir / "model.safetensors").string()) != TinyConvertedHeaderSha());
}

TEST_CASE("tabfm_weights: concurrent refreshes of one stale file publish one complete conversion",
          "[tabfm][weights][stale_weights]") {
	// Two connections resolving the same stale model both refresh it. They used to share one ".part" that
	// DuckDB opens with O_TRUNC, so one could truncate the file the other was about to rename into place.
	const CkptDir dir("tabfm_stale_weights_threads");
	WriteStaleSafetensors(dir / "model.safetensors");
	const auto expected = TinyConvertedHeaderSha();
	const auto ckpt = (dir / "model.ckpt").string();
	std::atomic<int> converted {0};
	std::atomic<int> bad {0};
	std::vector<std::thread> threads;
	for (int i = 0; i < 8; i++) {
		threads.emplace_back([&]() {
			auto fs = FileSystem::CreateLocal();
			for (int round = 0; round < 4; round++) {
				auto res = RefreshConvertedWeights(*fs, ckpt, expected, string());
				if (res.state == ConvertedWeightsState::CONVERTED) {
					converted++;
				} else if (res.state != ConvertedWeightsState::MATCHES) {
					bad++;
				}
			}
		});
	}
	for (auto &t : threads) {
		t.join();
	}
	REQUIRE(bad == 0);
	REQUIRE(converted == 1); // the others find it current once they hold the lock
	auto fs = FileSystem::CreateLocal();
	REQUIRE(WeightsHeaderSha(*fs, (dir / "model.safetensors").string()) == expected);
	REQUIRE(!AnyPartFile(dir.dir));
	// A complete file: the converter's own output, byte for byte, from a fresh directory.
	const CkptDir fresh("tabfm_stale_weights_threads_ref");
	REQUIRE(RefreshConvertedWeights(*fs, (fresh / "model.ckpt").string(), expected, string()).state ==
	        ConvertedWeightsState::CONVERTED);
	REQUIRE(SlurpFile(dir / "model.safetensors") == SlurpFile(fresh / "model.safetensors"));
}

#ifndef _WIN32
TEST_CASE("tabfm_weights: a cache that will not take the converted file reports the write error, not the Python step",
          "[tabfm][weights][stale_weights]") {
	if (geteuid() == 0) {
		WARN("skipped: root ignores directory permissions");
		return;
	}
	const CkptDir dir("tabfm_stale_weights_readonly");
	WriteStaleSafetensors(dir / "model.safetensors");
	const auto before = SlurpFile(dir / "model.safetensors");
	stdfs::permissions(dir.dir, stdfs::perms::owner_read | stdfs::perms::owner_exec, stdfs::perm_options::replace);
	auto fs = FileSystem::CreateLocal();
	const auto expected = TinyConvertedHeaderSha();
	auto res = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), expected, string());
	REQUIRE(res.state == ConvertedWeightsState::FAILED);
	REQUIRE(res.status.find("could not write") != string::npos);
	REQUIRE(SlurpFile(dir / "model.safetensors") == before);
	REQUIRE(!AnyPartFile(dir.dir));
	// Memoized: the next resolve does not re-read and re-convert the checkpoint, and says the same thing.
	auto again = RefreshConvertedWeights(*fs, (dir / "model.ckpt").string(), expected, string());
	REQUIRE(again.state == ConvertedWeightsState::FAILED);
	REQUIRE(again.status == res.status);

	const auto remedy = ConvertedWeightsRemedy("nori-30m", "classification", res.path, true, true, res.status);
	REQUIRE(remedy.find("writable") != string::npos);
	REQUIRE(remedy.find("could not write") != string::npos);
	REQUIRE(remedy.find("convert_weights.py") == string::npos);
}
#endif
