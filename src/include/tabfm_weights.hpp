//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_weights.hpp — the weights-lifecycle module's testable surface (WS-D).
// The SQL functions themselves register through tabfm_registration.hpp; this
// header exposes only the pure helpers that carry their own unit tests.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace anofox {

//! Map an HTTP auth failure raised while fetching `url` into an actionable
//! remediation (SQL-API §5: every error names the SET/CALL that fixes it).
//!
//! A gated HuggingFace repo answers an anonymous GET with **401** (no
//! credentials) or **403** (credentials present, license not accepted), and
//! httpfs surfaces that as a bare status line naming neither the secret nor the
//! license page. Returns the replacement message, or **""** when `error` is not
//! an auth failure — 404s, connection errors and the missing-httpfs
//! MissingExtensionException must keep their own remediation.
string HttpAuthRemediation(const string &error, const string &url);

//! What tabfm_download_runtime fetches for a backend: its plugin, from anofox-ggml-adapter's release assets (built by its plugins.yml).
struct RuntimeArtifact {
	string plugin_name;
	string plugin_url;
};

//! Resolve what tabfm_download_runtime(`backend`) must fetch on THIS host. False with `error` set for a backend that is unknown or not published here.
bool ResolveRuntimeArtifact(const string &backend, RuntimeArtifact &out, string &error);

} // namespace anofox
} // namespace duckdb
