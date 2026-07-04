#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ort_engine.hpp — ONNX Runtime engine + device/EP layer (WS-C)
//
// Value-oriented API around one process-wide Ort::Env and per-model
// Ort::Sessions (HLD §4.4, §9). Deliberately takes plain buffers/maps as
// inputs — no dependency on the safetensors/manifest readers (WS-B); the
// integration layer wires those together.
//
// Lifetime contract (license-critical): AddExternalInitializers keeps
// REFERENCES to every injected tensor (and reads them lazily during
// inference), so at real scale the buffers behind TabFMTensorRef must outlive
// the SESSION, not just CreateSession(). The returned TabFMSession owns the
// injected OrtValues and the caller keeps the source arena alive alongside it
// (see SessionHolder in tabfm_engine.cpp). Historically believed to copy — it
// does at fixture scale, which masked the bug; at 913 tensors / 6.6 GB it does
// not, and dropping the buffers yields a session that runs on zeroed weights.
// The buffers behind TabFMTensorRef must stay valid until CreateSession()
// RETURNS at minimum, and the OrtValues until the session is destroyed; the
// caller MUST free/munmap the source arena right after that to avoid paying
// 2x weights of resident memory (HLD §6 as amended by S02 RESULTS).
//===----------------------------------------------------------------------===//

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/string.hpp"

#include <cstdint>

namespace duckdb {
namespace anofox {

//===----------------------------------------------------------------------===//
// Dtypes & cast hooks
//===----------------------------------------------------------------------===//

enum class TabFMTensorDtype : uint8_t {
	F32,  // CPU lane (default)
	F16,  // GPU lane, RDNA3+ (HLD §9 dtype-per-arch)
	BF16, // GPU lane, CDNA / CUDA (upstream-reference dtype)
	I64,
	BOOL
};

//! Byte width of one element of the given dtype.
idx_t TabFMDtypeSize(TabFMTensorDtype dtype);

//! Cast hooks for the GPU dtype profiles (HLD §9): the injection point owns
//! the cast. f32 -> fp16 / bf16 use round-to-nearest-even; bf16 -> f32 is the
//! exact upcast S02 exercised. The returned buffer is a fresh copy — free the
//! source right after CreateSession like any other injection arena.
vector<uint16_t> CastF32ToF16(const float *src, idx_t count);
vector<uint16_t> CastF32ToBF16(const float *src, idx_t count);
vector<float> CastBF16ToF32(const uint16_t *src, idx_t count);

//===----------------------------------------------------------------------===//
// Device discovery + resolution (tabfm_devices(), SET anofox_tabfm_device)
//===----------------------------------------------------------------------===//

struct TabFMDeviceInfo {
	//! Stable id: "cpu", "cuda:<n>", "rocm:<n>"
	string device_id;
	//! ORT execution provider that serves this device
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
	//! Whether the flavor's EP can actually run on this device
	bool usable = false;
	//! Ordinal within the vendor runtime (cudaSetDevice / MIGraphX device_id)
	int device_ordinal = 0;
};

//! Compiled-in flavor capabilities (cmake/ort.cmake sets the macros).
constexpr bool TabFMFlavorHasCuda() {
#ifdef TABFM_EP_CUDA
	return true;
#else
	return false;
#endif
}
constexpr bool TabFMFlavorHasMIGraphX() {
#ifdef TABFM_EP_MIGRAPHX
	return true;
#else
	return false;
#endif
}
inline const char *TabFMFlavorName() {
#ifdef TABFM_FLAVOR_NAME
	return TABFM_FLAVOR_NAME;
#else
	return "cpu";
#endif
}

//! Enumerate devices this build can see. Always contains the 'cpu' row;
//! GPU rows appear only in the cuda/rocm flavors (probes compiled per macro).
vector<TabFMDeviceInfo> DiscoverDevices();

//! Resolve `SET anofox_tabfm_device` ('auto'|'cpu'|'cuda'|'rocm', already
//! normalized by the setting validator) against the discovered devices.
//! Throws InvalidInputException with the flavor-install hint when the
//! requested device is not carried by this flavor, and a tabfm_devices()
//! hint when the flavor carries it but no usable device was discovered.
//! The flag arguments exist for unit tests; production callers use the
//! defaults, which reflect the compiled flavor.
TabFMDeviceInfo ResolveDevice(const string &setting_value, const vector<TabFMDeviceInfo> &devices,
                              bool flavor_has_cuda = TabFMFlavorHasCuda(),
                              bool flavor_has_rocm = TabFMFlavorHasMIGraphX());

//===----------------------------------------------------------------------===//
// MIGraphX shape buckets (HLD §9: static-shape preference)
//===----------------------------------------------------------------------===//

struct TabFMShapeBucket {
	int64_t padded_t;
	int64_t padded_h;
};

//! Pad (T, H) up to the nearest compiled-session bucket:
//! T in {128, 512, 1024, 2048, 4096, 10000}, H in {16, 64, 128, 256, 512}.
//! Padding is semantically inert: the model's `train_size` and `d` masks
//! ignore padded rows/columns (S01 validated `d < H` parity at 5e-8).
//! Values above the largest bucket are returned unchanged (callers guard via
//! anofox_tabfm_max_rows / anofox_tabfm_max_features before reaching here).
TabFMShapeBucket MIGraphXShapeBucket(int64_t rows_t, int64_t features_h);

//===----------------------------------------------------------------------===//
// Session creation & inference
//===----------------------------------------------------------------------===//

//! Non-owning view of one initializer to inject via AddExternalInitializers.
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
	//! ORT intra-op threads (from anofox_tabfm_threads); inter-op is fixed to 1
	//! because inference runs inside a single DuckDB task (HLD §4.4).
	int64_t intra_op_threads = 1;
	//! Resolved execution device (ResolveDevice output).
	string device_id = "cpu";
	int device_ordinal = 0;
	//! Human-readable model tag for error messages ("classification", ...).
	string model_tag;
	//! ORT weight prepacking (anofox_tabfm_cpu_prepack): faster matmuls at ~+16%
	//! RSS. Disabling it was the old default when RSS was ~18 GB.
	bool prepack = true;
};

//! Opaque engine session (pimpl over Ort::Session; keeps ORT headers out of
//! every consumer TU).
class TabFMSession;
using TabFMSessionHandle = shared_ptr<TabFMSession>;

//! Build an Ort::Session from weight-free graph bytes + injected initializers.
//!   - SessionOptions: intra_op from config, inter-op 1, and
//!     "session.disable_prepacking"="1" (S02: -16% RSS, -60% init time).
//!   - EPs appended per config.device_id (CUDA / MIGraphX under their macros).
//!   - Ort error mapping: NO_SUCHFILE (external stub not injected) -> "model
//!     weights are not loaded" with the CALL tabfm_load/tabfm_download hint;
//!     FAIL on initializer replacement (dims/dtype mismatch) -> corrupted
//!     checkpoint error naming the tensor when extractable.
//! The graph bytes may be freed as soon as this returns, but the initializer
//! buffers must OUTLIVE the returned session (ORT keeps references and reads
//! them lazily during inference — the returned TabFMSession owns the wrapping
//! OrtValues; the caller must keep the underlying arena alive alongside it).
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
};

struct TabFMRunOutput {
	vector<float> logits;
	//! [1, T, C]
	vector<int64_t> shape;
};

TabFMRunOutput Run(TabFMSession &session, const TabFMRunInput &input);

//! A compiled, ready-to-run inference backend for one loaded (model, device).
//! The engine holds one per LoadedModel and calls Run() per forward pass.
//! Implementations: OrtBackend (CPU + CUDA execution provider, via ORT) and
//! MIGraphXBackend (direct AMD MIGraphX — ORT's MIGraphX EP cannot handle >2 GB
//! models, so ROCm bypasses ORT entirely).
class TabFMBackend {
public:
	virtual ~TabFMBackend() = default;
	virtual TabFMRunOutput Run(const TabFMRunInput &input) = 0;
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
