//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_plugin_backend.cpp — a GPU backend from a plugin (anofox-ggml-adapter, plugin ABI v3).
//
// The adapter's loader (plugin_host) checks the ABI before touching anything else and reports each failure by kind; this file supplies
// what is TabFM-specific: the weights the graph reads, mapped from the checkpoint and passed by name; the input contract (x, y, cat_mask,
// train_size, d; a graph without train_size reads the split from y's length, so it gets only the context rows of y); the logits output;
// and tabfm's wording of every refusal.
//===----------------------------------------------------------------------===//

#include "tabfm_plugin_backend.hpp"
#include "tabfm_ggml.hpp"
#include "tabfm_safetensors.hpp"

#include "anofox_ggml_adapter/onnx_reader.hpp"
#include "anofox_ggml_adapter/plugin_host.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <set>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace duckdb {
namespace anofox {

namespace {

namespace ga = ::anofox::ggml_adapter;

//! A read-only view of one checkpoint file: mmapped (the pages stay the page cache's), or read whole where mmap is unavailable.
class WeightFile {
public:
	explicit WeightFile(const string &path) {
#ifndef _WIN32
		const int fd = ::open(path.c_str(), O_RDONLY);
		if (fd >= 0) {
			struct stat st {};
			if (::fstat(fd, &st) == 0 && st.st_size > 0) {
				void *p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
				if (p != MAP_FAILED) {
					base_ = p;
					size_ = static_cast<size_t>(st.st_size);
				}
			}
			::close(fd);
		}
#endif
		if (!base_) {
			std::ifstream in(path, std::ios::binary);
			if (!in) {
				throw IOException("anofox_tabfm: cannot open the weights file '%s' for the GPU backend. Download the model again: CALL "
				                  "tabfm_download(<model>);",
				                  path);
			}
			copy_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
			size_ = copy_.size();
		}
	}
	~WeightFile() {
#ifndef _WIN32
		if (base_) {
			::munmap(base_, size_);
		}
#endif
	}
	WeightFile(const WeightFile &) = delete;
	WeightFile &operator=(const WeightFile &) = delete;
	const char *data() const {
		return base_ ? static_cast<const char *>(base_) : copy_.data();
	}
	size_t size() const {
		return size_;
	}

private:
	void *base_ = nullptr;
	size_t size_ = 0;
	string copy_;
};

int64_t ElementCount(const std::vector<int64_t> &dims) {
	int64_t n = 1;
	for (auto d : dims) {
		n *= d;
	}
	return n;
}

//! The plugin session plus the checkpoint it reads. `files` is declared FIRST so it is destroyed LAST, after the session (the plugin
//! reads the mapped weights for the session's whole life).
class PluginBackend : public TabFMBackend {
public:
	vector<unique_ptr<WeightFile>> files;
	string output_name;
	bool graph_has_train_size = true;
	std::set<string> inputs; // the inputs the graph declares
	std::unique_ptr<ga::PluginSession> session;

	TabFMRunOutput Run(const TabFMRunInput &input) override {
		// Same contract as the CPU engines: a graph without a train_size input derives the split from y's length, so it gets the prefix.
		const int64_t y_len = graph_has_train_size ? input.t : input.train_size;
		vector<uint8_t> mask;
		vector<ga::PluginTensor> in;
		const int64_t train_size = input.train_size;
		const int64_t d = input.d;
		in.push_back({"x", ANOFOX_GGML_DTYPE_FLOAT, {1, input.t, input.h}, input.x});
		in.push_back({"y", ANOFOX_GGML_DTYPE_FLOAT, {1, y_len}, input.y});
		if (inputs.count("cat_mask")) {
			mask.resize(static_cast<size_t>(input.h));
			for (int64_t i = 0; i < input.h; i++) {
				mask[static_cast<size_t>(i)] = input.cat_mask && input.cat_mask[i] ? 1 : 0;
			}
			in.push_back({"cat_mask", ANOFOX_GGML_DTYPE_BOOL, {1, input.h}, mask.data()});
		}
		if (inputs.count("train_size")) {
			in.push_back({"train_size", ANOFOX_GGML_DTYPE_INT64, {1}, &train_size, true});
		}
		if (inputs.count("d")) {
			in.push_back({"d", ANOFOX_GGML_DTYPE_INT64, {1}, &d, true});
		}
		ga::HostTensors out;
		try {
			out = session->Run(in, {output_name});
		} catch (const ga::PluginError &e) {
			if (e.kind == ga::PluginError::Kind::OUT_OF_MEMORY) {
				throw OutOfMemoryException("anofox_tabfm: the '%s' backend ran out of device memory: %s", e.backend, e.detail);
			}
			throw InvalidInputException("anofox_tabfm: the '%s' backend failed: %s", e.backend, e.detail);
		}
		TabFMRunOutput result;
		auto &logits = out.at(output_name);
		result.logits = std::move(logits.data);
		result.shape = std::move(logits.shape);
		return result;
	}
};

} // namespace

const vector<string> &TabFMPluginInputs() {
	static const vector<string> inputs = {"x", "y", "cat_mask", "train_size", "d"};
	return inputs;
}

bool PluginLoadable(const string &library_path, string *error) {
	return ga::PluginLoadable(library_path, error);
}

unique_ptr<TabFMBackend> LoadPluginBackend(const string &library_path, const TabFMPluginRequest &request) {
	auto backend = make_uniq<PluginBackend>();
	ga::PluginCreate params;
	if (!request.graph_path.empty()) {
		// The graph's external initializers point into a file beside it, by location and byte offset: map it and hand the plugin each one.
		::anofox::onnxread::Graph graph;
		try {
			graph = ::anofox::onnxread::ParseModelFile(request.graph_path);
		} catch (const std::exception &e) {
			throw IOException("anofox_tabfm: cannot read the GPU graph '%s': %s", request.graph_path, e.what());
		}
		std::set<string> initializers;
		std::map<string, WeightFile *> by_location;
		for (const auto &t : graph.initializers) {
			initializers.insert(t.name);
			if (t.location != ::anofox::onnxread::DataLocation::EXTERNAL) {
				continue;
			}
			const auto loc = t.external.find("location");
			const auto off = t.external.find("offset");
			if (loc == t.external.end() || off == t.external.end()) {
				throw IOException("anofox_tabfm: the GPU graph '%s' names no location/offset for the weight '%s'", request.graph_path, t.name);
			}
			auto file = by_location.find(loc->second);
			if (file == by_location.end()) {
				backend->files.push_back(make_uniq<WeightFile>(request.weights_dir + "/" + loc->second));
				file = by_location.emplace(loc->second, backend->files.back().get()).first;
			}
			const size_t offset = static_cast<size_t>(std::stoull(off->second));
			const int64_t count = ElementCount(t.dims);
			// The ONNX element size of each initializer, for the bounds check below; a type with no fixed size (string, complex) is
			// refused by name rather than handed over with a guessed size.
			size_t element = 0;
			switch (t.data_type) {
			case 2:  // uint8
			case 3:  // int8
			case 9:  // bool
				element = 1;
				break;
			case 4:  // uint16
			case 5:  // int16
			case 10: // float16
			case 16: // bfloat16
				element = 2;
				break;
			case 1:  // float
			case 6:  // int32
			case 12: // uint32
				element = 4;
				break;
			case 7:  // int64
			case 11: // double
			case 13: // uint64
				element = 8;
				break;
			default:
				throw IOException("anofox_tabfm: the GPU graph '%s' stores the weight '%s' as ONNX element type %d, which has no fixed "
				                  "size the GPU plugins can take. SET anofox_tabfm_device='cpu'.",
				                  request.graph_path, t.name, t.data_type);
			}
			if (offset + static_cast<size_t>(count) * element > file->second->size()) {
				throw IOException("anofox_tabfm: the weight '%s' does not fit the checkpoint '%s' (is the graph for another checkpoint, or the "
				                  "file truncated?). Download the model again: CALL tabfm_download(<model>);",
				                  t.name, loc->second);
			}
			params.weights.push_back({t.name, t.data_type, file->second->data() + offset, count, t.dims});
		}
		vector<string> unknown;
		for (const auto &name : graph.inputs) {
			if (initializers.count(name)) {
				continue;
			}
			const auto &known = TabFMPluginInputs();
			if (std::find(known.begin(), known.end(), name) == known.end()) {
				unknown.push_back(name);
			}
			backend->inputs.insert(name);
		}
		if (!unknown.empty()) {
			// Refused before the plugin runs anything: an unfed input would fail inside the runtime's own run call.
			throw IOException("anofox_tabfm: the GPU graph '%s' declares the input(s) '%s', which this build of anofox_tabfm does not feed; "
			                  "it is newer than the extension. Update the extension (INSTALL anofox_tabfm FROM community; UPDATE EXTENSIONS;) "
			                  "or SET anofox_tabfm_device='cpu'.",
			                  request.graph_path, StringUtil::Join(unknown, "', '"));
		}
		backend->graph_has_train_size = backend->inputs.count("train_size") > 0;
		backend->output_name = graph.outputs.empty() ? string("logits") : graph.outputs.front();
		for (const auto &o : graph.outputs) {
			if (o == "logits") {
				backend->output_name = o;
			}
		}
		params.graph_path = request.graph_path;
	} else {
		if (!StringUtil::StartsWith(request.model, "mitra")) {
			throw InvalidInputException("anofox_tabfm: the GPU backend was given no graph for '%s', and only mitra has a forward without one. "
			                            "Register the model with its ext graph, or SET anofox_tabfm_device='cpu' to run it on the CPU.",
			                            request.model);
		}
		// No graph (mitra's MLX hand-port): every tensor of the checkpoint, by its state_dict name.
		backend->files.push_back(make_uniq<WeightFile>(request.weights_path));
		const auto &file = *backend->files.back();
		auto view = ParseSafetensors(reinterpret_cast<const_data_ptr_t>(file.data()), file.size(), request.weights_path);
		for (const auto &kv : view.tensors) {
			const auto &t = kv.second;
			if (t.dtype == SafetensorsDtype::I64) {
				continue; // not a weight
			}
			params.weights.push_back({kv.first, t.dtype == SafetensorsDtype::BF16 ? ANOFOX_GGML_DTYPE_BFLOAT16 : ANOFOX_GGML_DTYPE_FLOAT,
			                          t.data, static_cast<int64_t>(t.ElementCount()), t.shape});
		}
		backend->inputs = {"x", "y", "train_size", "d"};
		backend->output_name = "logits";
	}
	params.model = request.model;
	params.arch = request.arch;
	params.precision = request.precision;
	params.cache_dir = request.cache_dir;
	params.device_ordinal = request.device_ordinal;
	params.out_of_memory_remedy = request.out_of_memory_remedy;
	try {
		backend->session = ga::PluginSession::Open(library_path, params);
	} catch (const ga::PluginError &e) {
		switch (e.kind) {
		case ga::PluginError::Kind::NOT_LOADABLE:
			throw IOException("anofox_tabfm: cannot load the backend plugin '%s': %s. Fetch it with CALL tabfm_download_runtime(...), or point "
			                  "anofox_tabfm_ep_path at the directory holding it.",
			                  library_path, e.detail);
		case ga::PluginError::Kind::NOT_A_PLUGIN:
			throw IOException("anofox_tabfm: '%s' is not an anofox backend plugin: %s.", library_path, e.detail);
		case ga::PluginError::Kind::ABI_MISMATCH:
			throw IOException("anofox_tabfm: the backend plugin '%s' was built against plugin ABI version %d, but this build speaks version %d. "
			                  "Update the plugin: CALL tabfm_accelerate() fetches the one built for this release (or point anofox_tabfm_ep_path "
			                  "at a matching build).",
			                  library_path, e.plugin_abi, ANOFOX_GGML_PLUGIN_ABI_VERSION);
		default:
			throw InvalidInputException("anofox_tabfm: the '%s' backend could not be initialised: %s. SET anofox_tabfm_device='cpu' to run on "
			                            "the CPU.",
			                            e.backend, e.detail);
		}
	}
	return std::move(backend);
}

} // namespace anofox
} // namespace duckdb
