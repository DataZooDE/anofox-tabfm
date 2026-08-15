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
	return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
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
		// Same hazard: api->name() returns a pointer into the library.
		const string backend_name = api->name();
		CloseLibrary(library);
		throw InvalidInputException("anofox_tabfm: the '%s' backend could not be initialised: %s", backend_name, err);
	}
	return make_uniq<PluginBackend>(library, api, handle, library_path);
}

} // namespace anofox
} // namespace duckdb
