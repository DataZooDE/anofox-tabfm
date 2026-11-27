//===----------------------------------------------------------------------===//
// tabfm_session.cpp — the in-process inference session (WS-C, HLD §4.4 / §9)
//
// Sessions are created from weight-free graph bytes with all initializers injected by reference and are served by the ggml CPU engine
// (tabfm_ggml_session.cpp). GPU devices never get an in-process session: their backend plugins serve them.
//===----------------------------------------------------------------------===//

#include "tabfm_session.hpp"
#include "tabfm_ggml_session.hpp"
#include "tabfm_ggml.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"


#include <array>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// TabFMSession
//===----------------------------------------------------------------------===//

class TabFMSession {
public:

	// Member DECLARATION ORDER is the destruction contract: `ggml` points into the caller's weight buffers and is declared last, so it is
	// destroyed first. The constructor init-list mirrors this order.
	TabFMSessionConfig config;
	vector<string> input_names;
	vector<string> output_names;
	//! The ggml CPU engine's state for this session (docs/GGML_PLAN.md); never null.
	std::unique_ptr<GgmlSessionState> ggml;

	//! A session served by the ggml engine.
	TabFMSession(TabFMSessionConfig config_p, std::unique_ptr<GgmlSessionState> ggml_p, vector<string> inputs_p, vector<string> outputs_p)
	    : config(std::move(config_p)), input_names(std::move(inputs_p)), output_names(std::move(outputs_p)), ggml(std::move(ggml_p))
	{
	}
};


// P4: verify the graph's actual input/output names match the manifest-declared
// tensor_contract (when one is declared), so a manifest that lies about its
// weight-free graph fails fast at load with an actionable error.
static void ValidateDeclaredContract(const TabFMSession &session, const TabFMSessionConfig &config) {
	auto has = [](const vector<string> &names, const string &n) {
		for (auto &x : names) {
			if (x == n) {
				return true;
			}
		}
		return false;
	};
	auto check = [&](const vector<string> &declared, const vector<string> &actual, const char *kind) {
		for (auto &d : declared) {
			if (!has(actual, d)) {
				throw InvalidInputException(
				    "anofox_tabfm: the manifest's tensor_contract declares %s '%s', but the model graph has no such "
				    "%s — the manifest does not match its weight-free graph.",
				    kind, d, kind);
			}
		}
	};
	// The engine feeds exactly these graph inputs (see Run): a contract that
	// declares any other input name describes a graph this engine cannot drive.
	// Reject it now with an actionable error rather than let Run() fail with an
	// opaque "unexpected input" deep in inference. (A future contract-driven
	// feeding path would relax this.)
	static const char *kFeedable[] = {"x", "y", "train_size", "cat_mask", "d"};
	for (auto &d : config.contract_inputs) {
		bool feedable = false;
		for (auto *f : kFeedable) {
			if (d == f) {
				feedable = true;
				break;
			}
		}
		if (!feedable) {
			throw InvalidInputException(
			    "anofox_tabfm: the manifest's tensor_contract declares input '%s', which this engine cannot feed. "
			    "TabFM graphs must use the inputs x, y, train_size, cat_mask, d.",
			    d);
		}
	}
	check(config.contract_inputs, session.input_names, "input");
	check(config.contract_outputs, session.output_names, "output");
}

namespace {

//! Only the CPU device has an in-process session (ggml). A GPU device is served by its plugin through TabFMBackend, never through CreateSession: reaching
//! here for one is a dispatch error, reported with the way out.
[[noreturn]] void ThrowNoInProcessSession(const TabFMSessionConfig &config) {
	throw InvalidInputException("anofox_tabfm: device '" + config.device_id +
	                            "' has no in-process session: GPU devices are served by their backend plugin (SET anofox_tabfm_ep_path to the directory "
	                            "holding it, CALL tabfm_download_runtime('cuda'|'rocm'|'mlx') to fetch it), or SET anofox_tabfm_device='cpu'.");
}

TabFMSessionHandle MakeGgmlSession(::anofox::onnxread::Graph graph, const vector<TabFMTensorRef> &initializers, const TabFMSessionConfig &config) {
	auto state = CreateGgmlSessionState(std::move(graph), initializers, config);
	auto inputs = state->input_names;
	auto outputs = state->output_names;
	auto handle = make_shared_ptr<TabFMSession>(config, std::move(state), std::move(inputs), std::move(outputs));
	ValidateDeclaredContract(*handle, config);
	return handle;
}

} // namespace

TabFMSessionHandle CreateSession(const void *graph_bytes, idx_t graph_size,
                                 const vector<TabFMTensorRef> &initializers, const TabFMSessionConfig &config) {
	if (!graph_bytes || graph_size == 0) {
		throw InvalidInputException("anofox_tabfm: cannot create a session from empty model graph bytes");
	}
	if (config.device_id != "cpu") {
		ThrowNoInProcessSession(config);
	}
	return MakeGgmlSession(::anofox::onnxread::ParseModelBytes(static_cast<const uint8_t *>(graph_bytes), graph_size, "model graph"), initializers,
	                       config);
}

TabFMSessionHandle CreateSessionFromPath(const string &graph_path, const vector<TabFMTensorRef> &initializers,
                                         const TabFMSessionConfig &config) {
	if (config.device_id != "cpu") {
		ThrowNoInProcessSession(config);
	}
	return MakeGgmlSession(::anofox::onnxread::ParseModelFile(graph_path), initializers, config);
}

//===----------------------------------------------------------------------===//
// Run
//===----------------------------------------------------------------------===//

TabFMRunOutput Run(TabFMSession &session, const TabFMRunInput &input) {
	if (!input.x || !input.y || !input.cat_mask) {
		throw InternalException("anofox_tabfm: Run() called with null input buffers");
	}
	if (input.t <= 0 || input.h <= 0) {
		throw InvalidInputException("anofox_tabfm: Run() needs positive row count T and feature count H, got T=" +
		                            std::to_string(input.t) + " H=" + std::to_string(input.h));
	}

	return GgmlRun(*session.ggml, input);
}

//===----------------------------------------------------------------------===//
// Split (prepare/query) models
//===----------------------------------------------------------------------===//

vector<string> SplitContextInputs(const TabFMSession &query_session) {
	vector<string> wanted;
	for (auto &name : query_session.input_names) {
		if (name != "x" && name != "y") {
			wanted.push_back(name);
		}
	}
	return wanted;
}

bool SessionHasOutput(const TabFMSession &session, const string &name) {
	for (auto &output : session.output_names) {
		if (output == name) {
			return true;
		}
	}
	return false;
}

TabFMPreparedContext RunPrepare(TabFMSession &session, const float *x, const float *y, int64_t s, int64_t h,
                                const vector<string> &wanted) {
	if (!x || !y) {
		throw InternalException("anofox_tabfm: RunPrepare() called with null input buffers");
	}
	if (s <= 0 || h <= 0) {
		throw InvalidInputException("anofox_tabfm: RunPrepare() needs a positive context size S and feature count H, "
		                            "got S=" +
		                            std::to_string(s) + " H=" + std::to_string(h));
	}
	if (wanted.empty()) {
		throw InvalidInputException(
		    "anofox_tabfm: the query graph takes no context tensors, so it is not the query half of a split pair.");
	}

	return GgmlRunPrepare(*session.ggml, x, y, s, h, wanted);
}

TabFMRunOutput RunQuery(TabFMSession &session, const float *x, int64_t q, int64_t h, const float *y, int64_t s,
                        const TabFMPreparedContext &context) {
	if (!x || !y) {
		throw InternalException("anofox_tabfm: RunQuery() called with null input buffers");
	}
	if (q <= 0 || h <= 0 || s <= 0) {
		throw InvalidInputException("anofox_tabfm: RunQuery() needs positive Q, H and S, got Q=" + std::to_string(q) +
		                            " H=" + std::to_string(h) + " S=" + std::to_string(s));
	}

	return GgmlRunQuery(*session.ggml, x, q, h, y, s, context);
}

TabFMRunOutput TabFMBackend::RunBatched(const TabFMRunInput &input, int64_t query_batch) {
	const int64_t s = input.train_size;
	const int64_t q = input.t - s;
	const int64_t h = input.h;
	if (query_batch <= 0 || q <= query_batch) {
		return Run(input);
	}
	// Each pass sees the full context plus one batch. A query row's answer depends on the context only,
	// never on the other query rows (the graph contract's row independence), so this is the answer one
	// unbatched pass gives; the context rows' outputs come from the first pass. y is [T]: the context's
	// labels, then the query rows' padding.
	TabFMRunOutput out;
	vector<float> batch_x, batch_y;
	for (int64_t offset = 0; offset < q; offset += query_batch) {
		const int64_t n = MinValue<int64_t>(query_batch, q - offset);
		batch_x.assign(input.x, input.x + s * h);
		batch_x.insert(batch_x.end(), input.x + (s + offset) * h, input.x + (s + offset + n) * h);
		batch_y.assign(input.y, input.y + s);
		batch_y.insert(batch_y.end(), input.y + s + offset, input.y + s + offset + n);
		TabFMRunInput part_input = input;
		part_input.x = batch_x.data();
		part_input.y = batch_y.data();
		part_input.t = s + n;
		auto part = Run(part_input);
		if (part.shape.size() != 3 || part.shape[0] != 1 || part.shape[1] != s + n || part.shape[2] < 1 ||
		    part.logits.size() != static_cast<size_t>((s + n) * part.shape[2]) ||
		    (offset > 0 && part.shape[2] != out.shape[2])) {
			return part; // the caller's contract check names the problem
		}
		const int64_t c = part.shape[2];
		if (offset == 0) {
			out.shape = {1, input.t, c};
			out.logits.reserve(static_cast<size_t>(input.t * c));
			out.logits.insert(out.logits.end(), part.logits.begin(), part.logits.end());
		} else {
			out.logits.insert(out.logits.end(), part.logits.begin() + s * c, part.logits.end());
		}
	}
	return out;
}

void ValidateTabFMOutput(const TabFMRunOutput &out, idx_t expected_t, idx_t min_classes, const char *task_name) {
	auto shape_str = [&]() {
		string s = "[";
		for (idx_t i = 0; i < out.shape.size(); i++) {
			s += (i ? ", " : "") + std::to_string(out.shape[i]);
		}
		return s + "]";
	};
	if (out.shape.size() != 3 || out.shape[0] != 1 ||
	    out.shape[1] != NumericCast<int64_t>(expected_t)) {
		throw InvalidInputException(
		    "anofox_tabfm: model produced logits of shape %s, but the engine contract is [1, %llu, C] for %s "
		    "(HLD §4.4). The configured graph does not match the engine — check the registered model's graph.",
		    shape_str(), static_cast<unsigned long long>(expected_t), task_name);
	}
	const auto C = NumericCast<idx_t>(out.shape[2]);
	if (C < 1 || C < min_classes) {
		throw InvalidInputException(
		    "anofox_tabfm: model output has %llu class column(s) but %s needs at least %llu — the graph's "
		    "class head does not match the data. Check the registered model's graph.",
		    static_cast<unsigned long long>(C), task_name, static_cast<unsigned long long>(min_classes));
	}
	// shape[0] == 1, so the element count is expected_t * C (both bounded by the
	// row/feature guardrails, no overflow).
	const idx_t expected = expected_t * C;
	if (out.logits.size() != expected) {
		throw InvalidInputException(
		    "anofox_tabfm: model returned %llu logits but the [1, %llu, %llu] output requires %llu — truncated or "
		    "malformed model output.",
		    static_cast<unsigned long long>(out.logits.size()), static_cast<unsigned long long>(expected_t),
		    static_cast<unsigned long long>(C), static_cast<unsigned long long>(expected));
	}
}

} // namespace anofox
} // namespace duckdb
