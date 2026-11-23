// abi_check <plugin.so> <dir> <tag> <graph.onnx> [fp32|tf32]: dlopen a tabfm plugin and run one ORT reference case (ref_<tag>_{x,y,logits}.f32 + ref_<tag>.txt, see
// tools/ggml_parity/ggml_driver.cpp) through the plugin ABI, the way the extension's LoadPluginBackend does. Build: g++ -std=c++17 -Isrc/include tools/ggml_plugin/abi_check.cpp -ldl
#include "tabfm_plugin_abi.h"
#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
static std::vector<float> Read(const std::string &p) { std::ifstream f(p, std::ios::binary | std::ios::ate); size_t n = f.tellg(); f.seekg(0); std::vector<float> v(n / 4); f.read((char *)v.data(), n); return v; }
int main(int argc, char **argv) {
	std::string plugin = argv[1], dir = argv[2], tag = argv[3], graph = argv[4], precision = argc > 5 ? argv[5] : "fp32";
	void *h = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!h) { printf("dlopen: %s\n", dlerror()); return 1; }
	auto get = (TabFMGetPluginApiFn)dlsym(h, TABFM_PLUGIN_ENTRY_SYMBOL);
	const TabFMPluginApi *api = get();
	printf("plugin '%s' abi %d\n", api->name(), api->abi_version);
	std::ifstream m(dir + "/ref_" + tag + ".txt"); long T, S, H, K, TS, D; m >> T >> S >> H >> K >> TS >> D;
	auto x = Read(dir + "/ref_" + tag + "_x.f32"), y = Read(dir + "/ref_" + tag + "_y.f32"), want = Read(dir + "/ref_" + tag + "_logits.f32");
	std::string g = dir + "/" + graph;
	TabFMPluginCreateParams p{}; p.graph_path = g.c_str(); p.weights_dir = dir.c_str(); p.cache_dir = "/tmp"; p.arch = ""; p.precision = precision.c_str(); p.mxr_source = ""; p.device_ordinal = 0;
	char err[2048] = {0};
	void *b = api->create(&p, err, sizeof err);
	if (!b) { printf("create failed: %s\n", err); return 1; }
	std::vector<uint8_t> mask(H, 0);
	TabFMPluginRunInput in{x.data(), y.data(), mask.data(), T, H, S, H};
	for (int rep = 0; rep < 3; rep++) {
		TabFMPluginRunOutput out{};
		if (api->run(b, &in, &out, err, sizeof err) != TABFM_PLUGIN_OK) { printf("run failed: %s\n", err); return 1; }
		double md = 0; for (long i = 0; i < out.logits_len && i < (long)want.size(); i++) { double d = std::fabs(out.logits[i] - want[i]); md = std::isnan(d) ? INFINITY : std::fmax(md, d); }
		printf("run %d: %ld logits, max |diff| vs ORT %.3g\n", rep, (long)out.logits_len, md);
		api->free_output(&out);
	}
	api->destroy(b);
	return 0;
}
