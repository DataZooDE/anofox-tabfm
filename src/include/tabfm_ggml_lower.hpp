#pragma once

//===----------------------------------------------------------------------===//
// tabfm_ggml_lower.hpp — lower an ONNX graph to ONE ggml graph for a concrete input shape and run it
// (docs/GGML_PLAN.md, P3). The shape pass (tabfm_shape_eval.hpp) supplies every tensor's shape and the folded
// integer constants; the lowering emits ggml ops for the float tensors only.
//
// P3 grows this op by op against ORT goldens (tools/ggml_parity/gen_op_goldens.py). A graph that needs an op or a
// layout the lowering lacks is refused by name, never run wrongly.
//===----------------------------------------------------------------------===//

#include "tabfm_onnx_reader.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace duckdb {
namespace anofox {

//! A float32 tensor on the host (row-major, ONNX layout).
struct HostTensor {
	std::vector<int64_t> shape;
	std::vector<float> data;
};

using HostTensors = std::map<std::string, HostTensor>;

//! A float32 weight kept outside the ONNX file: `count` is the number of floats at `data`, checked against the initializer's shape.
struct ExternalWeight {
	const float *data = nullptr; // `count` elements of the type below (reinterpreted: for 10 and 16 these are 2-byte values, not floats)
	size_t count = 0;
	int onnx_type = 1;           // ONNX element type of the bytes at `data`: 1 float32, 10 float16, 16 bfloat16 (the latter two only on a device path, upcast on upload)
};

//! Where the data of an initializer the ONNX file keeps outside itself (safetensors) lives: the float32 buffer for `name`, valid for the whole
//! call, or a null `data` when the weight is unknown. The runner uploads it into its own arena (nothing is mapped in place).
using WeightLookup = std::function<ExternalWeight(const std::string &name, const std::vector<int64_t> &dims)>;

//! An integer graph input (train_size, d): its shape and values. The values steer shapes, so the host folds them.
struct IntInput {
	std::vector<int64_t> shape;
	std::vector<int64_t> values;
};

struct GgmlRunOptions {
	int threads = 1;
	WeightLookup weights;                       // external initializers (safetensors)
	std::map<std::string, IntInput> int_inputs; // integer graph inputs
	std::set<std::string> bool_inputs;          // float-held graph inputs that are bool masks (their data is 0/1)
	std::vector<std::string> outputs;           // outputs to return; empty = all graph outputs. Nodes nothing requested depends on are not run.
	std::string device = "cpu";                 // see GgmlExecutorConfig::device (used by the one-shot GgmlRunGraph; an executor has its own)
	bool strict_fp32 = false;                   // see GgmlExecutorConfig::strict_fp32 (the executor fills this in from its own configuration)
};

//! Run `graph` on the float `inputs` (plus the typed inputs in `options`) and return its outputs, through ggml. Inline initializers are read from the
//! graph, external ones through `options.weights`. Throws std::runtime_error naming the node and op when something is not supported.
HostTensors GgmlRunGraph(const ::anofox::onnxread::Graph &graph, const HostTensors &inputs, const GgmlRunOptions &options);

//! A reusable executor for ONE graph. It compiles (shape pass, lowering, ggml graph, activation arena) once per input-shape signature and keeps the
//! last few compiled graphs, so repeated calls with the same shapes (the context cache's query batches, a SQL scan) skip all of that; it owns the
//! CPU thread pool and reads the caller's weight buffers in place. Calls are serialised. `graph` must outlive the executor, and so must the
//! buffers behind `weights` (the same lifetime contract as an ORT session's injected initializers).
struct GgmlExecutorConfig {
	int threads = 1;
	WeightLookup weights;
	size_t cache_entries = 2; // compiled graphs kept (each keeps its activation arena resident)
	//! Where the graph runs. "cpu" (default): the CPU backend straight from the caller's buffers. "gpu" or "gpu:N": the N-th ggml GPU device the build contains
	//! (CUDA, HIP, Metal, Vulkan...): the weights are uploaded once into device memory and every op ggml's backend cannot run (the custom kernels) runs on the CPU
	//! through the backend scheduler. "cpu-sched": the same scheduler path on the CPU backend with executor-owned weight copies (how the GPU code path is tested
	//! without a GPU). A "gpu" request with no GPU device in the build throws, naming what the build contains.
	std::string device = "cpu";
	//! Strict float32 matmuls on a GPU. ggml's CUDA backend multiplies float32 weights with tensor-core TF32 instructions when the activations have 16 or
	//! fewer rows (its small-batch kernel), and the NVIDIA TF32 switch does not reach that kernel. With this set, every weight matmul is padded to 17 activation
	//! rows (zero rows, sliced off afterwards) so ggml takes its cuBLAS path, which honours the switch. Costs a few rows of extra work on tiny matmuls.
	bool strict_fp32 = false;
	//! Free device memory in bytes (a test seam, and for backends ggml cannot ask): when set it replaces the query of the device itself in PreloadWeights.
	std::function<size_t()> device_memory_probe;
};

//! The ggml devices this build can run on, e.g. "CPU", "CUDA0", "ROCm0": a diagnostic for `device` errors and the backend table.
std::vector<std::string> GgmlDeviceNames();

class GgmlExecutor {
public:
	GgmlExecutor(const ::anofox::onnxread::Graph &graph, GgmlExecutorConfig config);
	~GgmlExecutor();
	GgmlExecutor(const GgmlExecutor &) = delete;
	GgmlExecutor &operator=(const GgmlExecutor &) = delete;

	//! Run on `inputs` (the integer/bool/output selections of `options` are part of the compile signature; its threads and weights are ignored:
	//! the executor's own configuration applies).
	HostTensors Run(const HostTensors &inputs, const GgmlRunOptions &options);

	//! Uploads every float32 external weight into device memory now (a device executor does it lazily, the first time a graph needs each one), so device
	//! out-of-memory and the one-time upload cost land at load, not in the first query. A weight whose layout the lowering cannot place is skipped here
	//! and reported by the run that needs it. A no-op for the CPU device. Returns the number of weights uploaded.
	size_t PreloadWeights();
	//! Drop every compiled graph and its activation memory (the next call recompiles).
	void Trim();
	//! A test and diagnostics hook: compiles (or reuses) the graph for these inputs WITHOUT running it and returns, for every matrix multiplication in it, the
	//! number of activation rows the backend will see (src1->ne[1]).
	std::vector<int64_t> MulMatActivationRows(const HostTensors &inputs, const GgmlRunOptions &options);

	//! Number of graphs compiled so far (a test hook: a repeated signature must not recompile).
	size_t CompileCount() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

//! Convenience form for float-only graphs.
HostTensors GgmlRunGraph(const ::anofox::onnxread::Graph &graph, const HostTensors &inputs, int threads = 1, const WeightLookup &weights = nullptr);

} // namespace anofox
} // namespace duckdb
