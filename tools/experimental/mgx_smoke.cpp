// Standalone smoke test for the direct-MIGraphX backend API cycle:
// parse_onnx (external-data) -> compile(gpu) -> save(.mxr) -> load(.mxr) -> eval.
// Validates the exact C++ calls the engine backend will use, and the compiled-
// program disk cache, without a 20-min recompile once the .mxr exists.
//
// Build (rocm prefix = extracted migraphx-root/opt/rocm):
//   g++ -std=c++17 -I<rocm>/include mgx_smoke.cpp -L<rocm>/lib -lmigraphx -lmigraphx_c -o mgx_smoke
// Run:
//   LD_LIBRARY_PATH=<rocm>/lib:/opt/rocm/lib ./mgx_smoke <graph_static.onnx> <cache.mxr>
#include <migraphx/migraphx.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
	if (argc < 3) {
		std::cerr << "usage: mgx_smoke <graph.onnx> <cache.mxr>\n";
		return 2;
	}
	const std::string graph = argv[1];
	const std::string mxr = argv[2];

	migraphx::program prog;
	std::ifstream probe(mxr, std::ios::binary);
	if (probe.good()) {
		probe.close();
		std::cout << "loading cached compiled program: " << mxr << "\n";
		prog = migraphx::load(mxr.c_str());
	} else {
		std::cout << "parse_onnx (external-data resolved from graph dir) ...\n";
		prog = migraphx::parse_onnx(graph.c_str()); // graph_static: shapes fixed, external->model.safetensors
		std::cout << "compile(target gpu) ... [slow, one-time]\n";
		migraphx::compile_options co;
		co.set_offload_copy(true); // pass host buffers; migraphx copies to/from device
		prog.compile(migraphx::target("gpu"), co);
		std::cout << "save(" << mxr << ") ...\n";
		migraphx::save(prog, mxr.c_str());
	}

	// Build inputs from the compiled program's expected parameter shapes.
	auto pshapes = prog.get_parameter_shapes();
	migraphx::program_parameters params;
	std::vector<std::vector<char>> buffers;
	for (auto *name : pshapes.names()) {
		migraphx::shape s = pshapes[name];
		std::vector<char> buf(s.bytes(), 0);
		const std::string n = name;
		if (n == "train_size" || n == "d") {
			int64_t v = 8;
			std::memcpy(buf.data(), &v, sizeof(v)); // [1] int64
		} else if (n == "y") {
			// context labels 0, query rows -100 sentinel
			auto *f = reinterpret_cast<float *>(buf.data());
			for (size_t i = 0; i < s.elements(); i++) {
				f[i] = (i < 8) ? 0.0f : -100.0f;
			}
		}
		// x (float), cat_mask (bool) left zero
		params.add(name, migraphx::argument(s, buf.data()));
		buffers.push_back(std::move(buf));
	}

	std::cout << "eval on gpu ...\n";
	auto results = prog.eval(params);
	auto out = results[0];
	auto os = out.get_shape();
	std::cout << "output dims: [";
	for (auto d : os.lengths()) {
		std::cout << d << " ";
	}
	std::cout << "] elements=" << os.elements() << "\n";
	auto *logits = reinterpret_cast<float *>(out.data());
	bool finite = true;
	for (size_t i = 0; i < os.elements(); i++) {
		if (!std::isfinite(logits[i])) {
			finite = false;
			break;
		}
	}
	std::cout << "first logits: " << logits[0] << " " << logits[1] << " " << logits[2]
	          << " | all_finite=" << (finite ? "yes" : "no") << "\n";
	std::cout << "OK: direct-migraphx parse/compile/save/load/eval cycle works\n";
	return finite ? 0 : 1;
}
