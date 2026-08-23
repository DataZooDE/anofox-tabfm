/*
 * plugin_load_check — does a built backend plugin LOAD on a machine with no
 * GPU?
 *
 * CI builds the plugins on plain runners (gpu_plugins.yml); the nm/readelf
 * sanity gates prove the entry symbol exists in the file, but only a real
 * dlopen proves the artifact resolves — a missing DT_NEEDED, a bad rpath, or
 * an SONAME regression all pass nm and fail here, exactly the way they would
 * fail on a user's machine at `SET anofox_tabfm_ep_path`.
 *
 *   PASS requires: dlopen resolves, TabFMGetPluginApi answers, and the ABI
 *   version matches the header this harness was compiled against.
 *
 * It then calls create() with deliberately GPU-less parameters. Either
 * outcome is acceptable — what is NOT acceptable is a crash: a user pointing
 * ep_path at a plugin on a machine without the hardware must get an error
 * message, not a SIGSEGV. The exit code only reflects the load contract; the
 * create result is printed for the log.
 *
 * Usage: plugin_load_check <plugin.so>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "../../src/include/tabfm_plugin_abi.h"

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: %s <plugin.so>\n", argv[0]);
		return 2;
	}

	void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		printf("LOAD_CHECK_FAIL dlopen: %s\n", dlerror());
		return 1;
	}
	printf("LOAD_CHECK dlopen ok\n");

	TabFMGetPluginApiFn get_api = (TabFMGetPluginApiFn)dlsym(lib, TABFM_PLUGIN_ENTRY_SYMBOL);
	if (!get_api) {
		printf("LOAD_CHECK_FAIL dlsym(%s): %s\n", TABFM_PLUGIN_ENTRY_SYMBOL, dlerror());
		return 1;
	}
	const TabFMPluginApi *api = get_api();
	if (!api) {
		printf("LOAD_CHECK_FAIL %s returned NULL\n", TABFM_PLUGIN_ENTRY_SYMBOL);
		return 1;
	}
	if (api->abi_version != TABFM_PLUGIN_ABI_VERSION) {
		printf("LOAD_CHECK_FAIL abi_version %d, harness built against %d\n", api->abi_version,
		       TABFM_PLUGIN_ABI_VERSION);
		return 1;
	}
	printf("LOAD_CHECK_OK name=%s abi=%d\n", api->name ? api->name() : "(null)", api->abi_version);

	/* No hardware, no model: create must refuse cleanly or (a fake) succeed.
	 * Reaching the printf below IS the assertion — a crash never gets there. */
	TabFMPluginCreateParams params;
	memset(&params, 0, sizeof(params));
	params.graph_path = "/nonexistent/graph.onnx";
	params.weights_dir = "/nonexistent";
	params.cache_dir = "/tmp";
	params.arch = "load-check";
	params.precision = "fp32";
	params.device_ordinal = 0;
	char err[2048];
	err[0] = '\0';
	void *handle = api->create(&params, err, sizeof(err));
	if (handle) {
		printf("CREATE_WITHOUT_GPU unexpectedly succeeded (fake plugin?); destroying\n");
		api->destroy(handle);
	} else {
		printf("CREATE_WITHOUT_GPU graceful error: %s\n", err[0] ? err : "(empty message)");
	}
	return 0;
}
