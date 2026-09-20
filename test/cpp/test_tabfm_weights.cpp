#include "catch.hpp"

#include "tabfm_weights.hpp"
#include "tabfm_predict.hpp" // ResolveEpPath — shared by download and dispatch

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
