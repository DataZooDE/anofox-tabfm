#pragma once

//===----------------------------------------------------------------------===//
// tabfm_session.hpp — the in-process inference session + device layer (WS-C)
//
// Value-oriented API around per-model sessions served by the ggml CPU engine (tabfm_ggml_session.hpp; GPUs are served by their backend
// plugins, tabfm_plugin_backend.hpp). Deliberately takes plain buffers/maps as inputs — no dependency on the safetensors/manifest readers
// (WS-B); the integration layer wires those together.
//
// Lifetime contract: the session keeps REFERENCES to every injected tensor and reads them during inference, so the buffers behind
// TabFMTensorRef (the safetensors mapping or the bf16 upcast arena) must outlive the SESSION, not just CreateSession(). The caller keeps
// the source arena alive alongside it (see SessionHolder in tabfm_engine.cpp).
//===----------------------------------------------------------------------===//

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/string.hpp"

#include <cctype>
#include <cstdint>

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// Dtypes
//===----------------------------------------------------------------------===//

//! The dtype of an injected tensor. bf16 checkpoints are upcast to f32 before injection (tabfm_safetensors.hpp, MaterializeF32Arena).
enum class TabFMTensorDtype : uint8_t { F32, I64, BOOL };

//===----------------------------------------------------------------------===//
// Device discovery + resolution (tabfm_devices(), SET anofox_tabfm_device)
//===----------------------------------------------------------------------===//

struct TabFMDeviceInfo {
	//! Stable id: "cpu", "cuda:<n>", "rocm:<n>", "mlx:0"
	string device_id;
	//! The engine that serves this device: "ggml" (cpu), "ggml-cuda", "ggml-rocm" (the ggml GPU plugin) or "mlx"
	string ep;
	//! Human-readable device name
	string name;
	//! "" (cpu), "sm_XY" (cuda), "gfxNNNN" (rocm)
	string arch;
	//! VRAM in bytes; -1 means unknown / NULL in SQL output
	int64_t vram_total = -1;
	int64_t vram_free = -1;
	//! Driver / runtime version string ("" if unknown)
	string driver;
	//! Whether the published plugin carries code for this device (CUDA compute capability, ROCm gfx arch)
	bool usable = false;
	//! Ordinal within the vendor runtime (the ggml CUDA / HIP device index)
	int device_ordinal = 0;
};

//! Enumerate the devices this machine has. Always contains the 'cpu' row; GPU rows come from NVML (CUDA), the KFD topology (ROCm) and the Apple SoC
//! (MLX), whether or not their plugin is installed.
vector<TabFMDeviceInfo> DiscoverDevices();

//! The ROCm rows of a KFD topology directory (`/sys/class/kfd/kfd/topology/nodes` in production): one row per compute-capable GPU node, in node order,
//! `rocm:N` with N the ordinal the ggml HIP backend uses. A seam so the discovery rules can be tested with a fake topology.
vector<TabFMDeviceInfo> DiscoverRocmDevicesFrom(const string &nodes_path, const string &driver);

//! Whether the ggml CUDA plugin ships kernels for a card of this compute capability (it is built for 7.5 and newer).
bool CudaComputeCapabilityUsable(int major, int minor);

//! The gfx architectures the released ggml ROCm plugin carries code objects for (the adapter's devices.hpp, mirroring AMDGPU_TARGETS in
//! anofox-ggml-adapter's .github/workflows/plugins.yml). HIP code objects do not run on another architecture, so a device outside this list is reported
//! unusable rather than chosen and failing at its first query.
const vector<string> &GgmlRocmPluginArchs();
bool RocmArchUsable(const string &arch);

//! The anofox_tabfm_device value as dispatch uses it: lowercased, and the deprecated alias 'migraphx' is 'rocm'. The SET validator does the
//! same, but DuckDB can store an extension option without calling it, so CaptureGpuDispatchSettings normalizes again where dispatch reads it.
inline string NormalizeDeviceSetting(const string &value) {
	string out = value;
	for (auto &c : out) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	return out == "migraphx" ? "rocm" : out;
}

//! Resolve `SET anofox_tabfm_device` ('auto'|'cpu'|'cuda'|'rocm'|'mlx', already normalized by the setting validator) against the discovered devices.
//! 'auto' answers the CPU here (a GPU is chosen per model by the engine); a GPU setting returns its first usable device or throws naming
//! tabfm_devices(); 'coreml' throws pointing at 'mlx'.
TabFMDeviceInfo ResolveDevice(const string &setting_value, const vector<TabFMDeviceInfo> &devices);


//===----------------------------------------------------------------------===//
// Session creation & inference
//===----------------------------------------------------------------------===//

//! Non-owning view of one initializer to inject into the session.
//! `name` is the ONNX initializer name (graph namespace — the tensor map owns
//! the safetensors-key -> graph-name translation). `data` must stay valid
//! until CreateSession returns; see the lifetime contract above.
struct TabFMTensorRef {
	string name;
	TabFMTensorDtype dtype = TabFMTensorDtype::F32;
	vector<int64_t> shape;
	const void *data = nullptr;
	idx_t size_bytes = 0;
};

struct TabFMSessionConfig {
	//! ggml threads for one forward pass (anofox_tabfm_threads).
	int64_t threads = 1;
	//! The device: an in-process session exists for "cpu" only (GPUs are served by their plugins).
	string device_id = "cpu";
	//! Human-readable model tag for error messages ("classification", ...).
	string model_tag;
	//! Manifest-declared tensor contract (P4). When non-empty, the graph's actual
	//! input / output names are validated against these at session creation, so a
	//! manifest that lies about its graph fails fast with an actionable error.
	//! Empty = no declared contract (skip — the built-in x/y/... contract).
	vector<string> contract_inputs;
	vector<string> contract_outputs;
};

//! Opaque engine session (pimpl over the ggml session state; keeps the engine headers out of every consumer TU).
class TabFMSession;
using TabFMSessionHandle = shared_ptr<TabFMSession>;

//! Build a session from weight-free graph bytes + injected initializers (the CPU device only; others throw, naming their plugin).
//!   - weights the graph needs but were not injected -> "model weights are not loaded" with the CALL tabfm_load/tabfm_download hint;
//!     a tensor of the wrong dims/dtype -> corrupted checkpoint error naming the tensor.
//! The graph bytes may be freed as soon as this returns, but the initializer buffers must OUTLIVE the returned session (it reads them
//! during inference; the caller keeps the underlying arena alive alongside it).
TabFMSessionHandle CreateSession(const void *graph_bytes, idx_t graph_size,
                                 const vector<TabFMTensorRef> &initializers, const TabFMSessionConfig &config);

//! Same, but loads the graph from a file PATH. This is what the extension uses
//! for the shipped weight-free graphs in resources/; the injected initializers
//! satisfy every external-data reference (the .data file need not exist).
TabFMSessionHandle CreateSessionFromPath(const string &graph_path, const vector<TabFMTensorRef> &initializers,
                                         const TabFMSessionConfig &config);

//! One TabFM forward pass. Buffers are non-owning and read-only:
//!   x [1,T,H] f32, y [1,T] f32, train_size [1] i64, cat_mask [1,H] bool,
//!   d [1] i64  ->  logits [1,T,C] f32 (HLD §4.4).
struct TabFMRunInput {
	const float *x = nullptr;
	const float *y = nullptr;
	const bool *cat_mask = nullptr;
	int64_t t = 0;
	int64_t h = 0;
	int64_t train_size = 0;
	int64_t d = 0;
	//! anofox_tabfm_context_cache_memory, in bytes: how much a split backend may
	//! keep in cached contexts. Ignored by every other backend.
	idx_t context_cache_bytes = 0;
	//! anofox_tabfm_context_cache_admit = 'first': keep a context the first time it is seen, rather than
	//! the second. Ignored by every other backend.
	bool context_cache_admit_first = false;
};

struct TabFMRunOutput {
	vector<float> logits;
	//! [1, T, C]
	vector<int64_t> shape;
};

TabFMRunOutput Run(TabFMSession &session, const TabFMRunInput &input);

//===----------------------------------------------------------------------===//
// Split (prepare/query) models — the encoded context, reused across calls
//===----------------------------------------------------------------------===//

//! The support half of a split model: everything derivable from the labelled
//! context, computed once and fed to every query batch that shares it.
//!
//! Tensors are held by NAME, never by position. The prepare graph's outputs and
//! the query graph's inputs are named from one contract by the exporter
//! (tools/export_tabicl/src/export_tabicl/split.py), and both graphs also carry
//! extra outputs from the ScatterND slice-bound pinning of their export (`val_*`,
//! `sym_size_int_*`) that are NOT part of it — so the runtime asks the query
//! graph what it needs and requests exactly that, rather than shovelling every
//! prepare output across.
struct TabFMPreparedContext {
	vector<string> names;
	vector<vector<int64_t>> shapes;
	vector<vector<float>> data;
	//! The context rows' own fitted values, [S, C] row-major: one query pass over
	//! the support rows. They depend only on the context (query rows are
	//! independent of each other given the support set), so they are computed
	//! with the context and reused with it.
	vector<float> context_logits;
	int64_t context_rows = 0;
	int64_t context_classes = 0;
	//! Resident cost of everything above.
	idx_t bytes = 0;
};

//! Run the support half: x [1,S,H] f32, y [1,S] f32 -> the named tensors in
//! `wanted`, which are the query graph's non-(x,y) inputs.
TabFMPreparedContext RunPrepare(TabFMSession &session, const float *x, const float *y, int64_t s, int64_t h,
                                const vector<string> &wanted);

//! Run the query half over `q` rows against a prepared context:
//! x [1,Q,H], y [1,S] and the context tensors -> logits [1,Q,C].
TabFMRunOutput RunQuery(TabFMSession &session, const float *x, int64_t q, int64_t h, const float *y, int64_t s,
                        const TabFMPreparedContext &context);

//! The query graph's inputs other than x/y — i.e. what the prepare graph has to
//! supply. Used to drive RunPrepare and to reject a mismatched pair at load.
vector<string> SplitContextInputs(const TabFMSession &query_session);

//! Whether the session's graph declares an output of this name. The split path
//! asks it of the prepare graph for the optional `context_logits` output.
bool SessionHasOutput(const TabFMSession &session, const string &name);

//! A compiled, ready-to-run inference backend for one loaded (model, device).
//! The engine holds one per LoadedModel and calls Run() per forward pass.
//! Implementations: CpuBackend and SplitContextBackend (the ggml CPU engine, tabfm_engine.cpp) and PluginBackend (the ggml CUDA/ROCm
//! plugin and MLX, tabfm_plugin_backend.cpp).
class TabFMBackend {
public:
	virtual ~TabFMBackend() = default;
	virtual TabFMRunOutput Run(const TabFMRunInput &input) = 0;
	//! One call's forward with its query rows scored at most `query_batch` per pass (opts query_batch).
	//! The default runs Run() once per batch on the support rows plus that batch, which re-encodes the
	//! context per batch; a backend with a context cache overrides it to encode (or look up) the context
	//! once and loop over the batches. The context rows' outputs come from the first pass.
	virtual TabFMRunOutput RunBatched(const TabFMRunInput &input, int64_t query_batch);
	//! Shrink any retained per-backend state (the context cache) to `budget` bytes. Called before the
	//! anofox_tabfm_max_memory check, so lowering the budget can free memory even when inference is refused.
	virtual void TrimContextCache(idx_t budget) {
	}
};

//! Validate a forward-pass output against the engine contract before it is
//! decoded (HLD §4.4): shape must be exactly [1, expected_t, C] with
//! C >= min_classes (>= 1), and logits.size() == expected_t * C. A mismatched
//! or custom graph would otherwise be indexed out of bounds, or silently
//! decoded with the wrong stride / zero-filled classes. Throws
//! InvalidInputException naming the contract on any mismatch. `task_name` is
//! "classification"/"regression" for the message.
void ValidateTabFMOutput(const TabFMRunOutput &out, idx_t expected_t, idx_t min_classes, const char *task_name);

} // namespace anofox
} // namespace duckdb
