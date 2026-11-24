#pragma once

#include "duckdb/common/string.hpp"

#include "anofox_ggml_adapter/plugin_artifacts.hpp"

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// tabfm_plugin_artifacts.hpp — where a backend plugin comes from, and whether one exists for this machine at all.
//
// The plugins are anofox-ggml-adapter's (shared with anofox-decide): their file names, the adapter release that carries them with sha256
// sidecars, and which platforms are published come from its plugin_artifacts.hpp. This header adds tabfm's wording: every refusal names
// tabfm_download_runtime / anofox_tabfm_device.
//===----------------------------------------------------------------------===//

//! The adapter release tabfm_download_runtime fetches the plugins from. A stale tag is safe: the loader refuses a plugin built against
//! another ABI version.
static constexpr const char *TABFM_PLUGIN_RELEASE_TAG = ::anofox::ggml_adapter::ANOFOX_GGML_PLUGIN_RELEASE_TAG;

inline string PluginLibrarySuffix(const string &os) {
	return ::anofox::ggml_adapter::PluginLibrarySuffix(os);
}

//! The plugin's own name for a device: `ggml-cuda`, `ggml-rocm` (the ggml plugin per vendor) or `mlx`.
inline string PluginBaseName(const string &backend) {
	return ::anofox::ggml_adapter::PluginBaseName(backend);
}

//! tabfm_download_runtime's backend argument, normalized once at that entry point: lowercased, and the deprecated name 'migraphx' (the
//! removed MIGraphX plugin) is the ROCm plugin.
inline string NormalizeRuntimeBackend(const string &requested) {
	return ::anofox::ggml_adapter::NormalizeBackend(requested);
}

//! The error for an explicit cuda/rocm request when the ggml GPU plugin is not in anofox_tabfm_ep_path: the file, where it looked, and the
//! ways out.
inline string GgmlGpuPluginMissingMessage(const string &vendor, const string &ep_path, const string &plugin_file) {
	const string where = ep_path.empty() ? string("anofox_tabfm_ep_path is not set")
	                                     : "it is not in anofox_tabfm_ep_path '" + ep_path + "'";
	return "anofox_tabfm: device '" + vendor + "' needs the ggml GPU backend plugin " + plugin_file + ", but " + where + ". CALL tabfm_download_runtime('" +
	       vendor + "') to fetch it (it is saved to anofox_tabfm_ep_path), or SET anofox_tabfm_ep_path to the directory that holds it, or SET "
	       "anofox_tabfm_device='cpu' to run on the CPU.";
}

inline string PluginFileNameFor(const string &backend, const string &os) {
	return ::anofox::ggml_adapter::PluginFileNameFor(backend, os);
}

inline string HostPluginOs() {
	return ::anofox::ggml_adapter::HostPluginOs();
}

inline string HostPluginArch() {
	return ::anofox::ggml_adapter::HostPluginArch();
}

inline string PluginFileName(const string &backend) {
	return ::anofox::ggml_adapter::PluginFileName(backend);
}

//! Empty when a plugin for `backend` is published for (os, arch); otherwise the refusal, naming what to do instead.
inline string UnsupportedPluginPlatform(const string &backend, const string &os, const string &arch) {
	const string reason = ::anofox::ggml_adapter::UnpublishedPluginReason(backend, os, arch);
	if (reason.empty()) {
		return "";
	}
	const auto b = NormalizeRuntimeBackend(backend);
	string remedy;
	if (b == "cuda" && os == "osx") {
		remedy = "On Apple Silicon use the MLX backend instead: CALL tabfm_download_runtime('mlx').";
	} else if (b == "mlx" && os == "linux") {
		remedy = "On Linux use 'cuda' for NVIDIA or 'rocm' for AMD.";
	} else if (b == "cuda" || b == "rocm" || b == "mlx") {
		remedy = "SET anofox_tabfm_device='cpu' to run on the CPU.";
	}
	return "tabfm_download_runtime: " + reason + (remedy.empty() ? string() : ". " + remedy);
}

inline string PluginSha256AssetName(const string &backend) {
	return ::anofox::ggml_adapter::PluginSha256AssetName(backend);
}

//! One file's digest out of a sha256sum sidecar, matched by file name; "" when not listed, which callers treat as a failure.
inline string Sha256FromSidecar(const string &sidecar, const string &file_name) {
	return ::anofox::ggml_adapter::Sha256FromSidecar(sidecar, file_name);
}

//! Download URL for one of the pinned adapter release's assets.
inline string PluginReleaseAssetUrl(const string &asset, const string &release_tag) {
	return ::anofox::ggml_adapter::PluginReleaseAssetUrl(asset, release_tag);
}

} // namespace anofox
} // namespace duckdb
