#pragma once

#include "duckdb/common/string.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// tabfm_plugin_artifacts.hpp — where a backend plugin comes from, and whether
// one exists for this machine at all.
//
// Two facts used to be spelled out separately at every site that needed them:
// the plugin's filename (hardcoded at three dispatch sites in
// tabfm_engine.cpp, three more in tabfm_weights.cpp, and in the settings help
// text) and the release the download fetches from. Spreading them meant a
// Windows user could reach a 475 MB download whose plugin does not exist for
// their platform, and be told so only by a missing-file error afterwards.
//
// Everything here is pure and takes the platform explicitly, so the refusal
// for every (backend, os, arch) combination is testable from any host — which
// matters precisely because the combinations that need guarding are the ones
// this machine is not.
//===----------------------------------------------------------------------===//

//! The release whose assets carry the built plugins. ONE definition, bumped BY
//! HAND before tagging: gpu_plugins.yml refuses to attach plugins to a release
//! whose tag is not the one pinned here ("The pinned plugin release must be the
//! one being published"), so a release always contains code pointing at itself.
//! Forgetting fails the release rather than shipping a version that quietly
//! downloads the previous one's plugins.
//!
//! A stale tag is safe by construction rather than by vigilance: the loader
//! checks the plugin's abi_version against TABFM_PLUGIN_ABI_VERSION and
//! refuses a mismatch (tabfm_plugin_backend.cpp), so an older plugin cannot
//! silently misbehave. It is NOT derived from the extension version, which
//! would 404 on every dev or dirty build.
static constexpr const char *TABFM_PLUGIN_RELEASE_TAG = "v2026.09.26";

//! Shared-library suffix for plugins on the platform being asked about.
inline string PluginLibrarySuffix(const string &os) {
	if (os == "windows") {
		return ".dll";
	}
	if (os == "osx") {
		return ".dylib";
	}
	return ".so";
}

//! The plugin's own name for a device. 'rocm' is driven by MIGraphX and the
//! artifact has always been named for the library, not the device.
inline string PluginBaseName(const string &backend) {
	if (backend == "rocm" || backend == "migraphx") {
		return "migraphx";
	}
	return backend;
}

//! Plugin filename for a backend on a given platform. Windows drops the 'lib'
//! prefix, following what CMake emits there.
inline string PluginFileNameFor(const string &backend, const string &os) {
	const string prefix = os == "windows" ? "" : "lib";
	return prefix + "anofox_tabfm_" + PluginBaseName(backend) + "_plugin" + PluginLibrarySuffix(os);
}

//! Host OS as this header names them: "linux" | "osx" | "windows".
inline string HostPluginOs() {
#if defined(_WIN32)
	return "windows";
#elif defined(__APPLE__)
	return "osx";
#else
	return "linux";
#endif
}

//! Host architecture: "amd64" | "arm64" | "" when neither.
inline string HostPluginArch() {
#if defined(__aarch64__) || defined(_M_ARM64)
	return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
	return "amd64";
#else
	return "";
#endif
}

//! Plugin filename for a backend on THIS machine.
inline string PluginFileName(const string &backend) {
	return PluginFileNameFor(backend, HostPluginOs());
}

//! Empty when a plugin for `backend` is published for (os, arch). Otherwise an
//! actionable refusal naming what IS available there.
//!
//! This is deliberately a statement about what is PUBLISHED and verified, not
//! about what could compile. A user cannot act on the latter, and claiming a
//! platform we have never run on is how a support matrix starts lying.
inline string UnsupportedPluginPlatform(const string &backend, const string &os, const string &arch) {
	const string where = os + "/" + (arch.empty() ? string("unknown") : arch);
	if (backend == "cuda") {
		if (os == "linux" && arch == "amd64") {
			return "";
		}
		return "tabfm_download_runtime: no CUDA plugin or runtime is published for " + where +
		       " — the CUDA backend is built and verified for linux/amd64 only. " +
		       (os == "osx" ? "On Apple Silicon use the MLX backend instead: CALL tabfm_download_runtime('mlx')."
		                    : "SET anofox_tabfm_device='cpu' to run on the CPU.");
	}
	if (backend == "rocm" || backend == "migraphx") {
		if (os == "linux" && arch == "amd64") {
			return "";
		}
		return "tabfm_download_runtime: no ROCm plugin is published for " + where +
		       " — MIGraphX runs on linux/amd64 only, and the AMD compute stack has no build for " + os +
		       ". SET anofox_tabfm_device='cpu' to run on the CPU.";
	}
	if (backend == "mlx") {
		if (os == "osx" && arch == "arm64") {
			return "";
		}
		return "tabfm_download_runtime: no MLX plugin is published for " + where +
		       " — MLX is Apple Silicon only (macOS/arm64). " +
		       (os == "linux" ? "On Linux use 'cuda' for NVIDIA or 'rocm' for AMD."
		                      : "SET anofox_tabfm_device='cpu' to run on the CPU.");
	}
	return "tabfm_download_runtime: unknown backend '" + backend + "' — expected 'cuda', 'rocm' or 'mlx'.";
}

//! The release asset carrying a backend's sha256 sidecar. CI writes these with
//! `sha256sum <files> | tee <base>_plugin.sha256`, so the name is keyed to the
//! plugin's base name, not its full filename.
inline string PluginSha256AssetName(const string &backend) {
	return PluginBaseName(backend) + "_plugin.sha256";
}

//! Pull one file's digest out of a `sha256sum` sidecar. The CUDA sidecar covers
//! two files (the plugin and its private ORT core), so the line must be matched
//! by filename rather than assumed to be the first.
//!
//! Returns "" when the file is not listed — which callers must treat as a
//! failure, not as "nothing to check": a sidecar that does not mention the file
//! it is supposed to vouch for proves nothing about it.
inline string Sha256FromSidecar(const string &sidecar, const string &file_name) {
	size_t pos = 0;
	while (pos < sidecar.size()) {
		auto eol = sidecar.find('\n', pos);
		const auto line = sidecar.substr(pos, eol == string::npos ? string::npos : eol - pos);
		pos = eol == string::npos ? sidecar.size() : eol + 1;
		// "<64 hex>  <name>" — the name may carry a path or a '*' binary marker.
		auto space = line.find(' ');
		if (space == string::npos || space != 64) {
			continue;
		}
		auto named = line.substr(space);
		while (!named.empty() && (named.front() == ' ' || named.front() == '*')) {
			named.erase(named.begin());
		}
		while (!named.empty() && (named.back() == '\r' || named.back() == ' ')) {
			named.pop_back();
		}
		auto slash = named.find_last_of("/\\");
		if (slash != string::npos) {
			named = named.substr(slash + 1);
		}
		if (named == file_name) {
			return StringUtil::Lower(line.substr(0, 64));
		}
	}
	return "";
}

//! Download URL for one of the pinned release's assets.
inline string PluginReleaseAssetUrl(const string &asset, const string &release_tag) {
	if (release_tag.empty()) {
		return "";
	}
	return "https://github.com/DataZooDE/anofox-tabfm/releases/download/" + release_tag + "/" + asset;
}

} // namespace anofox
} // namespace duckdb
