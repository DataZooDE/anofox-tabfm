//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_plugin_backend.cpp — load a backend from a shared library.
//
// Phase 1 of docs/DYNAMIC_BACKENDS.md: a GPU backend becomes a library the
// extension dlopens on demand rather than a build flavor. This is the loader
// side of tabfm_plugin_abi.h — it adapts the C function table back onto the
// TabFMBackend interface the engine already speaks.
//
// The loader is the trust boundary, so every failure it can meet has a message
// naming what to do about it: a missing file, a library that is not a plugin,
// and — the one that would otherwise corrupt memory silently — a plugin built
// against a different ABI version.
//===----------------------------------------------------------------------===//

#include "tabfm_plugin_backend.hpp"

#include "duckdb/common/exception.hpp"

#include <cstring>

// Is this translation unit built with AddressSanitizer? GCC defines the macro
// directly; Clang answers through __has_feature. Used only to drop
// RTLD_DEEPBIND below, which the sanitizer runtime cannot tolerate.
#if defined(__SANITIZE_ADDRESS__)
#define TABFM_SANITIZER_BUILD 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define TABFM_SANITIZER_BUILD 1
#endif
#endif

#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace duckdb {
namespace anofox {

namespace {

#ifndef _WIN32
using LibraryHandle = void *;
LibraryHandle OpenLibrary(const string &path) {
	// RTLD_DEEPBIND: the plugin prefers its OWN dependencies over symbols the
	// host process already exports. Measured necessity (GPU_HARDENING_PLAN S2,
	// all four quadrants of a SONAME x DEEPBIND matrix): when the host has a
	// shared ORT loaded, a plugin whose ORT core merely has a different SONAME
	// still BINDS to the host's — dedup and interposition are independent
	// failure layers, and isolating the plugin's runtime takes both the rename
	// (ships with the runtime, P7) and this flag. Safe for this ABI: no
	// allocation crosses the boundary (outputs are plugin-malloc'd and
	// plugin-freed via free_output), which is the classic DEEPBIND hazard.
	// glibc-only: macOS has no RTLD_DEEPBIND (and needs none — Mach-O
	// two-level namespaces bind each image to the library it linked against).
	//
	// Exception, sanitizer builds only: RTLD_DEEPBIND is incompatible with the
	// ASan/LSan runtime, which aborts the process on the dlopen rather than
	// failing it (google/sanitizers#611). That abort killed the whole `make
	// test_debug` C++ stage at the first plugin test, taking every later case
	// with it — so a debug build could not run the plugin tests at all, which
	// is exactly where plugin changes need testing. Dropping the flag under a
	// sanitizer costs nothing real: DEEPBIND isolates a plugin's own ORT/HIP
	// runtime from the host's, and the sanitizer build loads the weight-free
	// fixture plugin, which has no such runtime to isolate.
	int flags = RTLD_NOW | RTLD_LOCAL;
#if defined(RTLD_DEEPBIND) && !defined(TABFM_SANITIZER_BUILD)
	flags |= RTLD_DEEPBIND;
#endif
	return dlopen(path.c_str(), flags);
}
void *LibrarySymbol(LibraryHandle lib, const char *symbol) {
	return dlsym(lib, symbol);
}
void CloseLibrary(LibraryHandle lib) {
	dlclose(lib);
}
string LibraryError() {
	const char *err = dlerror();
	return err ? string(err) : string("unknown dynamic-loader error");
}
#else
using LibraryHandle = HMODULE;
LibraryHandle OpenLibrary(const string &path) {
	return LoadLibraryA(path.c_str());
}
void *LibrarySymbol(LibraryHandle lib, const char *symbol) {
	return reinterpret_cast<void *>(GetProcAddress(lib, symbol));
}
void CloseLibrary(LibraryHandle lib) {
	FreeLibrary(lib);
}
string LibraryError() {
	return "LoadLibrary/GetProcAddress failed (error " + std::to_string(GetLastError()) + ")";
}
#endif

//! Adapts the C plugin table onto the engine's C++ backend interface.
class PluginBackend : public TabFMBackend {
public:
	PluginBackend(LibraryHandle library, const TabFMPluginApi *api, void *handle, string path)
	    : library(library), api(api), handle(handle), path(std::move(path)) {
	}

	~PluginBackend() override {
		if (api && handle) {
			api->destroy(handle);
		}
		// The library is intentionally NOT closed: ORT's env, MIGraphX's HIP
		// context and any thread-locals the plugin registered outlive this
		// object, and unloading underneath them is how a clean shutdown turns
		// into a segfault in someone else's destructor.
	}

	TabFMRunOutput Run(const TabFMRunInput &input) override {
		TabFMPluginRunInput in {};
		in.x = input.x;
		in.y = input.y;
		// TabFMRunInput carries `const bool *`; the ABI uses uint8_t because
		// _Bool's size is implementation-defined and this crosses a compiler
		// boundary. They are layout-compatible everywhere we build, but the
		// cast is explicit so the assumption is visible.
		in.cat_mask = reinterpret_cast<const uint8_t *>(input.cat_mask);
		in.t = input.t;
		in.h = input.h;
		in.train_size = input.train_size;
		in.d = input.d;

		TabFMPluginRunOutput out {};
		char err[512] = {0};
		if (api->run(handle, &in, &out, err, sizeof(err)) != TABFM_PLUGIN_OK) {
			api->free_output(&out);
			throw InvalidInputException("anofox_tabfm: the '%s' backend failed: %s", api->name(), err);
		}

		TabFMRunOutput result;
		result.logits.assign(out.logits, out.logits + out.logits_len);
		result.shape.assign(out.shape, out.shape + out.shape_len);
		api->free_output(&out);
		return result;
	}

	void Precompile(int64_t rows, int64_t features) override {
		char err[512] = {0};
		if (api->precompile(handle, rows, features, err, sizeof(err)) != TABFM_PLUGIN_OK) {
			throw InvalidInputException("anofox_tabfm: the '%s' backend could not precompile: %s", api->name(), err);
		}
	}

private:
	LibraryHandle library;
	const TabFMPluginApi *api;
	void *handle;
	string path;
};

} // namespace

bool PluginLoadable(const string &library_path, string *error) {
	auto set_error = [&](string message) {
		if (error) {
			*error = std::move(message);
		}
	};
	auto library = OpenLibrary(library_path);
	if (!library) {
		// The loader's own text is the useful part and is usually specific:
		// a plugin present but unloadable almost always names the dependency
		// that is missing (libmigraphx_c.so, a CUDA runtime), which is a
		// different problem from the file not being there and has a different
		// fix. Reporting only "not loadable" would hide it.
		set_error(LibraryError());
		return false;
	}
	auto entry = reinterpret_cast<TabFMGetPluginApiFn>(LibrarySymbol(library, TABFM_PLUGIN_ENTRY_SYMBOL));
	if (!entry) {
		CloseLibrary(library);
		set_error("the library exports no " + string(TABFM_PLUGIN_ENTRY_SYMBOL) +
		          " — a file with the right name but the wrong contents");
		return false;
	}
	const TabFMPluginApi *api = entry();
	if (api && api->abi_version != TABFM_PLUGIN_ABI_VERSION) {
		set_error("built against plugin ABI version " + std::to_string(api->abi_version) + ", but this build speaks " +
		          std::to_string(TABFM_PLUGIN_ABI_VERSION));
	}
	// Same ordering rule as LoadPluginBackend: abi_version is the only field
	// safe to read before it has been checked.
	const bool ok = api && api->abi_version == TABFM_PLUGIN_ABI_VERSION;
	// Deliberately NOT unloaded on success: this library is about to be opened
	// for real by LoadPluginBackend, and a GPU plugin that has been mapped once
	// must not be unmapped underneath the driver context it may already have
	// registered. On failure nothing was initialised, so closing is safe and
	// keeps a probe of a wrong file from pinning it.
	if (!ok) {
		CloseLibrary(library);
	}
	return ok;
}

unique_ptr<TabFMBackend> LoadPluginBackend(const string &library_path, const TabFMPluginCreateParams &params) {
	auto library = OpenLibrary(library_path);
	if (!library) {
		throw IOException("anofox_tabfm: cannot load the backend plugin '%s': %s. Fetch it with CALL "
		                  "tabfm_download_runtime(...), or point anofox_tabfm_ep_path at the directory holding it.",
		                  library_path, LibraryError());
	}

	auto entry = reinterpret_cast<TabFMGetPluginApiFn>(LibrarySymbol(library, TABFM_PLUGIN_ENTRY_SYMBOL));
	if (!entry) {
		CloseLibrary(library);
		throw IOException("anofox_tabfm: '%s' is not an anofox backend plugin — it exports no %s. A file with the "
		                  "right name but the wrong contents is the usual cause.",
		                  library_path, TABFM_PLUGIN_ENTRY_SYMBOL);
	}

	const TabFMPluginApi *api = entry();
	if (!api) {
		CloseLibrary(library);
		throw IOException("anofox_tabfm: the backend plugin '%s' returned no API table", library_path);
	}
	// Before touching anything else in the struct: a plugin built against a
	// different layout would have every later field at the wrong offset, and
	// reading those is undefined behaviour rather than a wrong answer.
	if (api->abi_version != TABFM_PLUGIN_ABI_VERSION) {
		// Read it BEFORE unloading: `api` points into the library's own memory,
		// so dlclose unmaps it and the message would be built from freed pages.
		const int plugin_abi = api->abi_version;
		CloseLibrary(library);
		throw IOException("anofox_tabfm: the backend plugin '%s' was built against plugin ABI version %d, but this "
		                  "build speaks version %d. Update the plugin (or the extension) so the two match.",
		                  library_path, plugin_abi, TABFM_PLUGIN_ABI_VERSION);
	}

	char err[512] = {0};
	void *handle = api->create(&params, err, sizeof(err));
	if (!handle) {
		const string backend_name = api->name();
		// Deliberately NOT CloseLibrary here (review finding): create has
		// already executed plugin code — the CUDA plugin constructs an
		// Ort::Env and probes the driver before it can fail, leaving TLS and
		// atexit registrations pointing into the library. Unmapping it turns
		// a clean "could not be initialised" into a crash at shutdown or on
		// retry. The pre-create failure paths above ran no plugin code and
		// may keep closing; this path leaks the handle exactly like the
		// destructor does, for the same reason.
		throw InvalidInputException("anofox_tabfm: the '%s' backend could not be initialised: %s", backend_name, err);
	}
	return make_uniq<PluginBackend>(library, api, handle, library_path);
}

} // namespace anofox
} // namespace duckdb
