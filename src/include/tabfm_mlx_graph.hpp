/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_mlx_graph.hpp — execute a shipped ONNX graph on Apple MLX.
 *
 * docs/MLX_PLAN.md route 3, promoted to the plan of record when the goal became
 * "every registered model on MLX". The alternative was seven hand-ported
 * forwards — seven second implementations of models we already ship, each free
 * to drift from the graph every other backend runs. This runs the graph itself,
 * so a new model costs no new math and drift is impossible by construction.
 *
 * Scope was measured before any of it was written (docs/MLX_SPIKE_RESULTS.md):
 * 13 graphs, 4.5k–12.4k nodes, 33–48 distinct ops each, 64 in union. The Python
 * prototype in tools/mlx_spike settled every op's semantics against ORT CPU on
 * real weights first; this is that prototype in C++ over mlx-c.
 *
 * Two things about these graphs shape the whole design:
 *
 *   - SHAPES ARE DYNAMIC. They were produced by torch.export with dynamic rows
 *     and features, so they compute shapes at runtime (Shape/Slice/Concat) and
 *     feed them to Reshape/Expand. Integer tensors must therefore stay integers;
 *     a Shape result quietly promoted to fp32 breaks any dimension past 2^24.
 *   - INITIALIZERS LIVE OUTSIDE THE FILE. The ext graphs name model.safetensors
 *     and an absolute byte offset. They are mapped, never copied — which is what
 *     lets tabfm-v1's 6.6 GB load at all on a 16 GB machine.
 *===----------------------------------------------------------------------===*/

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace anofox {
namespace mlxgraph {

//! One input or output tensor, as the caller sees it. Always float32: the
//! engine's tensor contract has no other input dtype, and int64 scalars
//! (train_size, d) are converted on the way in.
struct Tensor {
	std::vector<int> shape;
	std::vector<float> data;
};

//! A loaded graph. Construction parses the proto, maps the weights and builds
//! the node list; Run() executes it. Not thread-safe by itself — MLX streams
//! are thread-local, so Run() acquires the calling thread's stream each time.
class Graph {
public:
	//! `graph_path` is the ext graph; `weights_dir` holds the model.safetensors
	//! its initializers point into. Throws std::runtime_error naming the fix.
	Graph(const std::string &graph_path, const std::string &weights_dir);
	~Graph();
	Graph(const Graph &) = delete;
	Graph &operator=(const Graph &) = delete;

	//! Named feeds in, the graph's first output back.
	Tensor Run(const std::vector<std::pair<std::string, Tensor>> &feeds) const;

	//! Input names in graph order, so the caller can tell the two model
	//! families apart (train_size-scalar vs single_eval_pos) without guessing.
	const std::vector<std::string> &InputNames() const;

	//! Ops this graph needs that are not implemented. Empty means runnable.
	std::vector<std::string> MissingOps() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

} // namespace mlxgraph
} // namespace anofox
