/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_mlx_plugin.cpp — the Apple MLX backend, behind tabfm_plugin_abi.h.
 *
 * docs/MLX_PLAN.md phase M1, with the spike verdicts in
 * docs/MLX_SPIKE_RESULTS.md. Like the MIGraphX plugin and unlike the CUDA one,
 * this drives a runtime directly rather than going through ONNX Runtime: MLX
 * has no ONNX importer, and S-M1 established that none exists anywhere, so the
 * forward is hand-ported instead of interpreted from the graph.
 *
 * That has a consequence worth stating plainly at the top: THIS FILE IS A
 * SECOND IMPLEMENTATION OF A MODEL WE ALREADY SHIP. The authority on what
 * mitra computes is
 * `tools/export_mitra/src/export_mitra/mitra_model_patched.py` — the ONNX graph
 * every other backend runs was exported from it. This is a transcription of
 * that file (via the MLX Python port in `tools/mlx_spike`, which is where the
 * math was proven against the real weights), and it must be kept in step with
 * it. The `graph_path` parameter is therefore NOT loaded; it is used only to
 * locate the weights and to reject a model this plugin does not implement.
 *
 * Consequently the plugin is model-specific: it serves mitra. Asking it for
 * anything else fails loudly at create time rather than quietly computing the
 * wrong thing with the wrong architecture's weights.
 *
 * mlx-c (0.6+) covers the whole forward — S-M3 checked this before any code was
 * written, since the plan named mlx-c maturity as the biggest product risk.
 * The C API returns a status and writes through an out-param, and every
 * intermediate array must be freed, so `Arr` below is a move-only RAII handle
 * and `Ok()` turns a non-zero status into an exception that never escapes the
 * ABI boundary.
 *===----------------------------------------------------------------------===*/

#include "tabfm_plugin_abi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/c/array.h"
#include "mlx/c/fast.h"
#include "mlx/c/io.h"
#include "mlx/c/map.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"
#include "mlx/c/version.h"

namespace {

// Must match mitra_model_patched.NEG_INF: a *finite* additive sentinel. -inf
// would make a fully-masked attention row produce NaN rather than a uniform
// distribution, and padded feature columns do exist.
constexpr float kNegInf = -1.0e9f;
constexpr float kLayerNormEps = 1e-5f; // torch.nn.LayerNorm default
constexpr int kNQuantiles = 999;       // upstream: arange(1, 1000) / 1000

// ---------------------------------------------------------------------------
// RAII + error plumbing
// ---------------------------------------------------------------------------
struct MlxError : std::runtime_error {
	explicit MlxError(const std::string &what) : std::runtime_error(what) {
	}
};

//! Move-only owner of an mlx_array. mlx-c hands back borrowed-or-owned handles
//! depending on the call, so every construction site here is one that owns.
class Arr {
public:
	Arr() : a_(mlx_array_new()) {
	}
	explicit Arr(mlx_array a) : a_(a) {
	}
	Arr(const Arr &) = delete;
	Arr &operator=(const Arr &) = delete;
	Arr(Arr &&o) noexcept : a_(o.a_) {
		o.a_ = mlx_array_new();
	}
	Arr &operator=(Arr &&o) noexcept {
		if (this != &o) {
			mlx_array_free(a_);
			a_ = o.a_;
			o.a_ = mlx_array_new();
		}
		return *this;
	}
	~Arr() {
		mlx_array_free(a_);
	}

	mlx_array get() const {
		return a_;
	}
	mlx_array *out() {
		return &a_;
	}

	size_t ndim() const {
		return mlx_array_ndim(a_);
	}
	std::vector<int> shape() const {
		const int *s = mlx_array_shape(a_);
		return std::vector<int>(s, s + mlx_array_ndim(a_));
	}
	int dim(size_t i) const {
		return mlx_array_shape(a_)[i];
	}

private:
	mlx_array a_;
};

void Ok(int status, const char *what) {
	if (status != 0) {
		throw MlxError(std::string("mlx call failed: ") + what);
	}
}

//! The GPU stream every op below runs on. One per backend instance; MLX is
//! lazy, so this only orders work, it does not execute it.
struct Stream {
	mlx_stream s;
	explicit Stream(bool gpu) : s(gpu ? mlx_default_gpu_stream_new() : mlx_default_cpu_stream_new()) {
	}
	Stream(const Stream &) = delete;
	Stream &operator=(const Stream &) = delete;
	~Stream() {
		mlx_stream_free(s);
	}
};

// ---------------------------------------------------------------------------
// Op helpers — thin, named after the torch/MLX-python line they replace
// ---------------------------------------------------------------------------
class Ops {
public:
	explicit Ops(mlx_stream s) : s_(s) {
	}

	Arr FromData(const void *data, const std::vector<int> &shape, mlx_dtype dtype) const {
		return Arr(mlx_array_new_data(data, shape.data(), static_cast<int>(shape.size()), dtype));
	}
	Arr Scalar(float v) const {
		return Arr(mlx_array_new_float32(v));
	}
	Arr Full(const std::vector<int> &shape, float v, mlx_dtype dtype = MLX_FLOAT32) const {
		Arr val = Scalar(v), r;
		Ok(mlx_full(r.out(), shape.data(), shape.size(), val.get(), dtype, s_), "full");
		return r;
	}
	Arr Arange(double start, double stop, mlx_dtype dtype) const {
		Arr r;
		Ok(mlx_arange(r.out(), start, stop, 1.0, dtype, s_), "arange");
		return r;
	}

	Arr Add(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_add(r.out(), a.get(), b.get(), s_), "add");
		return r;
	}
	Arr Sub(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_subtract(r.out(), a.get(), b.get(), s_), "subtract");
		return r;
	}
	Arr Mul(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_multiply(r.out(), a.get(), b.get(), s_), "multiply");
		return r;
	}
	Arr Div(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_divide(r.out(), a.get(), b.get(), s_), "divide");
		return r;
	}
	Arr MatMul(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_matmul(r.out(), a.get(), b.get(), s_), "matmul");
		return r;
	}
	Arr Sqrt(const Arr &a) const {
		Arr r;
		Ok(mlx_sqrt(r.out(), a.get(), s_), "sqrt");
		return r;
	}
	Arr Floor(const Arr &a) const {
		Arr r;
		Ok(mlx_floor(r.out(), a.get(), s_), "floor");
		return r;
	}
	Arr Erf(const Arr &a) const {
		Arr r;
		Ok(mlx_erf(r.out(), a.get(), s_), "erf");
		return r;
	}
	Arr Max(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_maximum(r.out(), a.get(), b.get(), s_), "maximum");
		return r;
	}
	Arr Min(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_minimum(r.out(), a.get(), b.get(), s_), "minimum");
		return r;
	}
	Arr Where(const Arr &c, const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_where(r.out(), c.get(), a.get(), b.get(), s_), "where");
		return r;
	}
	Arr Not(const Arr &a) const {
		Arr r;
		Ok(mlx_logical_not(r.out(), a.get(), s_), "logical_not");
		return r;
	}
	Arr GreaterEqual(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_greater_equal(r.out(), a.get(), b.get(), s_), "greater_equal");
		return r;
	}
	Arr Less(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_less(r.out(), a.get(), b.get(), s_), "less");
		return r;
	}
	Arr Equal(const Arr &a, const Arr &b) const {
		Arr r;
		Ok(mlx_equal(r.out(), a.get(), b.get(), s_), "equal");
		return r;
	}
	Arr AsType(const Arr &a, mlx_dtype t) const {
		Arr r;
		Ok(mlx_astype(r.out(), a.get(), t, s_), "astype");
		return r;
	}
	Arr Clip(const Arr &a, float lo, float hi) const {
		Arr l = Scalar(lo), h = Scalar(hi), r;
		Ok(mlx_clip(r.out(), a.get(), l.get(), h.get(), s_), "clip");
		return r;
	}
	Arr Reshape(const Arr &a, const std::vector<int> &shape) const {
		Arr r;
		Ok(mlx_reshape(r.out(), a.get(), shape.data(), shape.size(), s_), "reshape");
		return r;
	}
	Arr Transpose(const Arr &a, const std::vector<int> &axes) const {
		Arr r;
		Ok(mlx_transpose_axes(r.out(), a.get(), axes.data(), axes.size(), s_), "transpose");
		return r;
	}
	Arr BroadcastTo(const Arr &a, const std::vector<int> &shape) const {
		Arr r;
		Ok(mlx_broadcast_to(r.out(), a.get(), shape.data(), shape.size(), s_), "broadcast_to");
		return r;
	}
	Arr SumAxis(const Arr &a, int axis, bool keepdims) const {
		Arr r;
		Ok(mlx_sum_axis(r.out(), a.get(), axis, keepdims, s_), "sum_axis");
		return r;
	}
	Arr SortAxis(const Arr &a, int axis) const {
		Arr r;
		Ok(mlx_sort_axis(r.out(), a.get(), axis, s_), "sort_axis");
		return r;
	}
	Arr TakeAlongAxis(const Arr &a, const Arr &idx, int axis) const {
		Arr r;
		Ok(mlx_take_along_axis(r.out(), a.get(), idx.get(), axis, s_), "take_along_axis");
		return r;
	}
	Arr TakeAxis(const Arr &a, const Arr &idx, int axis) const {
		Arr r;
		Ok(mlx_take_axis(r.out(), a.get(), idx.get(), axis, s_), "take_axis");
		return r;
	}
	Arr Concat(const Arr &a, const Arr &b, int axis) const {
		mlx_vector_array v = mlx_vector_array_new();
		mlx_vector_array_append_value(v, a.get());
		mlx_vector_array_append_value(v, b.get());
		Arr r;
		int st = mlx_concatenate_axis(r.out(), v, axis, s_);
		mlx_vector_array_free(v);
		Ok(st, "concatenate_axis");
		return r;
	}
	//! Insert a trailing axis: (..., n) -> (..., n, 1). Reshape rather than
	//! expand_dims so the shape is explicit at every call site.
	Arr Unsqueeze(const Arr &a, int axis) const {
		std::vector<int> shape = a.shape();
		shape.insert(shape.begin() + (axis < 0 ? shape.size() + axis + 1 : axis), 1);
		return Reshape(a, shape);
	}

	Arr LayerNorm(const Arr &x, const Arr &w, const Arr &b) const {
		Arr r;
		Ok(mlx_fast_layer_norm(r.out(), x.get(), w.get(), b.get(), kLayerNormEps, s_), "fast_layer_norm");
		return r;
	}
	//! F.gelu with approximate='none' (the torch default): the exact erf form.
	//! The tanh approximation differs by ~1e-3, above the parity bar.
	Arr Gelu(const Arr &x) const {
		Arr inv_sqrt2 = Scalar(0.70710678118654752440f), half = Scalar(0.5f), one = Scalar(1.0f);
		Arr e = Erf(Mul(x, inv_sqrt2));
		return Mul(Mul(x, half), Add(e, one));
	}
	Arr Sdpa(const Arr &q, const Arr &k, const Arr &v, const Arr &mask, float scale) const {
		Arr r;
		Ok(mlx_fast_scaled_dot_product_attention(r.out(), q.get(), k.get(), v.get(), scale,
		                                         /*mask_mode=*/"array", mask.get(),
		                                         /*sinks=*/mlx_array_new(), s_),
		   "fast_scaled_dot_product_attention");
		return r;
	}
	//! torch.nn.Linear: weight is (out, in), so the op is x @ W.T + b.
	Arr Linear(const Arr &x, const Arr &w, const Arr &b) const {
		std::vector<int> axes = {1, 0};
		Arr y = MatMul(x, Transpose(w, axes));
		return Add(y, b);
	}

private:
	mlx_stream s_;
};

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------
//! The released safetensors, keyed by the bare Tab2D state_dict names. The
//! tensor map is transform-free (resources/tensor_map_mitra_*.json is the
//! identity), so no renaming happens here -- which is exactly why this plugin
//! can read the same cached file every other backend reads.
class MitraWeights {
public:
	//! `load_stream` must be a CPU stream: MLX's Load op has no GPU
	//! implementation, so scheduling the read on the GPU stream produces
	//! arrays that fail at eval with "[Load::eval_gpu] Not implemented" --
	//! and it fails at the first forward, not at load, which is why the read
	//! is forced to completion here rather than left lazy. Unified memory
	//! means the materialized tensors are then usable by GPU ops with no copy.
	MitraWeights(const std::string &path, mlx_stream load_stream) : map_(mlx_map_string_to_array_new()) {
		mlx_map_string_to_string meta = mlx_map_string_to_string_new();
		int st = mlx_load_safetensors(&map_, &meta, path.c_str(), load_stream);
		mlx_map_string_to_string_free(meta);
		if (st != 0) {
			mlx_map_string_to_array_free(map_);
			map_ = mlx_map_string_to_array_new();
			throw MlxError("could not read safetensors at " + path);
		}
		auto it = mlx_map_string_to_array_iterator_new(map_);
		const char *key = nullptr;
		mlx_array value = mlx_array_new();
		int n = 0;
		while (mlx_map_string_to_array_iterator_next(&key, &value, it) == 0) {
			if (mlx_array_eval(value) != 0) {
				mlx_array_free(value);
				mlx_map_string_to_array_iterator_free(it);
				throw MlxError("failed to materialize tensor '" + std::string(key ? key : "?") + "' from " + path);
			}
			n++;
		}
		mlx_array_free(value);
		mlx_map_string_to_array_iterator_free(it);
		if (n == 0) {
			throw MlxError("no tensors in " + path);
		}
	}
	MitraWeights(const MitraWeights &) = delete;
	MitraWeights &operator=(const MitraWeights &) = delete;
	~MitraWeights() {
		mlx_map_string_to_array_free(map_);
	}

	Arr operator[](const std::string &name) const {
		Arr r;
		if (mlx_map_string_to_array_get(r.out(), map_, name.c_str()) != 0) {
			throw MlxError("weight '" + name + "' not found; the cached safetensors do not look like mitra");
		}
		return r;
	}
	bool Has(const std::string &name) const {
		mlx_array probe = mlx_array_new();
		bool found = mlx_map_string_to_array_get(&probe, map_, name.c_str()) == 0;
		mlx_array_free(probe);
		return found;
	}

	//! Derived rather than assumed: a checkpoint whose depth or width differs
	//! from what the caller asked for must not be run with the wrong loop count.
	int NLayers() const {
		int n = 0;
		while (Has("layers." + std::to_string(n) + ".attention1.q.weight")) {
			n++;
		}
		return n;
	}
	int Dim() const {
		return (*this)["final_layer_norm.weight"].dim(0);
	}
	int DimOutput() const {
		return (*this)["final_layer.weight"].dim(0);
	}

private:
	mlx_map_string_to_array map_;
};

// ---------------------------------------------------------------------------
// The forward — a transcription of mitra_model_patched.py + its ExportWrapper
// ---------------------------------------------------------------------------
class MitraForward {
public:
	MitraForward(const MitraWeights &w, const Ops &ops, int n_heads, bool classification)
	    : w_(w), o_(ops), n_heads_(n_heads), classification_(classification), n_layers_(w.NLayers()),
	      dim_(w.Dim()), dim_output_(w.DimOutput()) {
	}

	int n_layers() const {
		return n_layers_;
	}
	int dim() const {
		return dim_;
	}
	int dim_output() const {
		return dim_output_;
	}

	//! x [1,T,H] f32, y [1,T] f32 -> logits [1,T,C].
	Arr Run(const Arr &x, const Arr &y, int64_t train_size, int64_t d) const {
		const int t = x.dim(1);
		const int h = x.dim(2);

		// ExportWrapper: the whole table is fed as BOTH support and query, and
		// train_size / d become masks rather than slices -- so no shape here is
		// data-dependent, which is what let the ONNX export be dynamic.
		Arr ar_t = o_.Arange(0, t, MLX_FLOAT32);
		Arr ar_h = o_.Arange(0, h, MLX_FLOAT32);
		Arr ts = o_.Scalar(static_cast<float>(train_size));
		Arr dv = o_.Scalar(static_cast<float>(d));
		Arr pad_obs = o_.Reshape(o_.GreaterEqual(ar_t, ts), {1, t});  // (1,T) true == pad
		Arr pad_feat = o_.Reshape(o_.GreaterEqual(ar_h, dv), {1, h}); // (1,H)

		Arr xs, xq;
		QuantileEmbedding(x, pad_obs, xs, xq);

		Arr xw = w_["x_embedding.x_embedding.weight"];
		Arr xb = w_["x_embedding.x_embedding.bias"];
		Arr x_support = o_.Linear(o_.Unsqueeze(xs, -1), xw, xb); // (1,T,H,dim)
		Arr x_query = o_.Linear(o_.Unsqueeze(xq, -1), xw, xb);

		Arr y_support_e, y_query_e;
		EmbedY(y, pad_obs, t, y_support_e, y_query_e);

		// einops.pack((y, x), "b s * d") -- the label is feature column 0.
		Arr support = o_.Concat(y_support_e, x_support, 2); // (1,T,H+1,dim)
		Arr query = o_.Concat(y_query_e, x_query, 2);

		Arr pad_y = o_.Full({1, 1}, 0.0f, MLX_BOOL);
		Arr pad_feat_full = o_.Concat(pad_y, pad_feat, 1); // (1,H+1)
		Arr neg = o_.Scalar(kNegInf), zero = o_.Scalar(0.0f);
		Arr feat_key_mask = o_.Reshape(o_.Where(pad_feat_full, neg, zero), {1, 1, 1, h + 1});
		Arr row_key_mask = o_.Reshape(o_.Where(pad_obs, neg, zero), {1, 1, 1, t});

		for (int i = 0; i < n_layers_; i++) {
			Layer(i, support, query, row_key_mask, feat_key_mask);
		}

		query = o_.LayerNorm(query, w_["final_layer_norm.weight"], w_["final_layer_norm.bias"]);
		query = o_.Linear(query, w_["final_layer.weight"], w_["final_layer.bias"]);
		// query[:, :, 0, :] -- read the label column back out.
		Arr idx0 = o_.FromData(&kZero, {1}, MLX_INT32);
		Arr picked = o_.TakeAxis(query, idx0, 2); // (1,T,1,C)
		return o_.Reshape(picked, {1, t, dim_output_});
	}

private:
	static constexpr int kZero = 0;

	//! Tab2DQuantileEmbeddingX: rank normalization over VALID support rows.
	void QuantileEmbedding(const Arr &x, const Arr &pad_obs, Arr &xs_out, Arr &xq_out) const {
		const int t = x.dim(1);
		const int f = x.dim(2);

		Arr valid = o_.Not(pad_obs);                                     // (1,T)
		Arr seq_len = o_.SumAxis(o_.AsType(valid, MLX_FLOAT32), 1, false); // (1,)
		Arr one = o_.Scalar(1.0f);
		Arr seq_len_c = o_.Max(seq_len, one);

		// Padded rows sort to +inf, so they never enter the interpolation
		// window [0, seq_len-1] and contribute to no statistic.
		Arr valid_col = o_.Reshape(valid, {1, t, 1});
		Arr big = o_.Full({1, t, f}, std::numeric_limits<float>::infinity());
		Arr sorted_x = o_.SortAxis(o_.Where(valid_col, x, big), 1);

		Arr q_num = o_.Arange(1, kNQuantiles + 1, MLX_FLOAT32);
		Arr q_den = o_.Scalar(static_cast<float>(kNQuantiles + 1));
		Arr q = o_.Reshape(o_.Div(q_num, q_den), {1, kNQuantiles});
		Arr len_m1 = o_.Reshape(o_.Sub(seq_len_c, one), {1, 1});
		Arr pos = o_.Mul(q, len_m1); // (1,Q)
		Arr zero = o_.Scalar(0.0f);
		Arr lo = o_.Max(o_.Floor(pos), zero);
		Arr hi = o_.Min(o_.Add(lo, one), len_m1);
		Arr frac = o_.Sub(pos, lo);

		Arr lo_i = o_.BroadcastTo(o_.Reshape(o_.AsType(lo, MLX_INT32), {1, kNQuantiles, 1}), {1, kNQuantiles, f});
		Arr hi_i = o_.BroadcastTo(o_.Reshape(o_.AsType(hi, MLX_INT32), {1, kNQuantiles, 1}), {1, kNQuantiles, f});
		Arr q_lo = o_.TakeAlongAxis(sorted_x, lo_i, 1);
		Arr q_hi = o_.TakeAlongAxis(sorted_x, hi_i, 1);
		Arr frac_c = o_.Reshape(frac, {1, kNQuantiles, 1});
		Arr quantiles = o_.Add(q_lo, o_.Mul(frac_c, o_.Sub(q_hi, q_lo))); // (1,Q,f)

		// torch.bucketize(right=False) == count of quantiles strictly below.
		// The (1,T,Q,f) comparison is the peak allocation of the whole forward.
		Arr qs = o_.Reshape(quantiles, {1, 1, kNQuantiles, f});
		Arr len_c = o_.Reshape(seq_len_c, {1, 1, 1});
		auto bucketize = [&](const Arr &vals) {
			Arr v = o_.Reshape(vals, {1, t, 1, f});
			Arr cmp = o_.AsType(o_.Less(qs, v), MLX_FLOAT32);
			return o_.Div(o_.SumAxis(cmp, 2, false), len_c);
		};
		Arr xs = bucketize(x);
		Arr xq = bucketize(x); // support and query are the same table here

		Arr zeros = o_.Full({1, t, f}, 0.0f);
		xs = o_.Where(valid_col, xs, zeros);
		Arr mean = o_.Div(o_.SumAxis(xs, 1, true), len_c);
		xs = o_.Sub(xs, mean);
		xq = o_.Sub(xq, mean);

		xs = o_.Where(valid_col, xs, zeros);
		Arr var = o_.Div(o_.SumAxis(o_.Mul(xs, xs), 1, true), len_c);
		Arr std = o_.Sqrt(var);
		xs = o_.Div(xs, std);
		xq = o_.Div(xq, std);

		// A constant feature has zero variance; the division above produced
		// NaN, and upstream replaces it with 0 rather than propagating.
		Arr zero_var = o_.Equal(var, zero);
		xs_out = o_.Where(zero_var, zeros, xs);
		xq_out = o_.Where(zero_var, zeros, xq);
	}

	void EmbedY(const Arr &y, const Arr &pad_obs, int n_query, Arr &support_out, Arr &query_out) const {
		const int t = y.dim(1);
		Arr emb;
		if (classification_) {
			// The engine feeds float labels; padded/query rows carry arbitrary
			// values, so clamp to a valid class id before the gather and zero
			// the row afterwards.
			Arr ids = o_.AsType(o_.Clip(y, 0.0f, static_cast<float>(dim_output_ - 1)), MLX_INT32);
			Arr zeros_i = o_.Full({1, t}, 0.0f, MLX_INT32);
			ids = o_.Where(pad_obs, zeros_i, ids);
			Arr table = w_["y_embedding.y_embedding.weight"];
			emb = o_.Reshape(o_.TakeAxis(table, o_.Reshape(ids, {t}), 0), {1, t, dim_});
		} else {
			emb = o_.Linear(o_.Unsqueeze(y, -1), w_["y_embedding.y_embedding.weight"],
			                w_["y_embedding.y_embedding.bias"]);
		}
		Arr zeros_d = o_.Full({1, t, dim_}, 0.0f);
		emb = o_.Where(o_.Reshape(pad_obs, {1, t, 1}), zeros_d, emb);
		support_out = o_.Reshape(emb, {1, t, 1, dim_});

		// nn.Embedding(1, dim): every query row gets the same learned mask row.
		Arr mask_row = o_.Reshape(w_["y_embedding.y_mask.weight"], {1, 1, 1, dim_});
		query_out = o_.BroadcastTo(mask_row, {1, n_query, 1, dim_});
	}

	Arr Attention(const std::string &prefix, const Arr &query, const Arr &key, const Arr &mask) const {
		const int b = query.dim(0);
		const int tq = query.dim(1);
		const int tk = key.dim(1);
		const int hd = dim_ / n_heads_;
		std::vector<int> to_heads = {0, 2, 1, 3};
		Arr q = o_.Transpose(o_.Reshape(o_.Linear(query, w_[prefix + ".q.weight"], w_[prefix + ".q.bias"]),
		                                {b, tq, n_heads_, hd}),
		                     to_heads);
		Arr k = o_.Transpose(
		    o_.Reshape(o_.Linear(key, w_[prefix + ".k.weight"], w_[prefix + ".k.bias"]), {b, tk, n_heads_, hd}),
		    to_heads);
		Arr v = o_.Transpose(
		    o_.Reshape(o_.Linear(key, w_[prefix + ".v.weight"], w_[prefix + ".v.bias"]), {b, tk, n_heads_, hd}),
		    to_heads);
		Arr o = o_.Sdpa(q, k, v, mask, 1.0f / std::sqrt(static_cast<float>(hd)));
		o = o_.Reshape(o_.Transpose(o, to_heads), {b, tq, dim_});
		return o_.Linear(o, w_[prefix + ".o.weight"], w_[prefix + ".o.bias"]);
	}

	void Layer(int i, Arr &support, Arr &query, const Arr &row_key_mask, const Arr &feat_key_mask) const {
		const std::string p = "layers." + std::to_string(i) + ".";
		const int n_s = support.dim(1);
		const int f1 = support.dim(2);
		const int n_q = query.dim(1);
		std::vector<int> swap = {0, 2, 1, 3};

		auto ln = [&](const char *name, const Arr &t) {
			return o_.LayerNorm(t, w_[p + name + ".weight"], w_[p + name + ".bias"]);
		};
		auto mlp = [&](const Arr &t, const char *a, const char *c) {
			Arr hidden = o_.Gelu(o_.Linear(t, w_[p + a + ".weight"], w_[p + a + ".bias"]));
			return o_.Linear(hidden, w_[p + c + ".weight"], w_[p + c + ".bias"]);
		};

		// --- attention across observations (rows); keys are the support rows ---
		{
			Arr s_flat = o_.Reshape(o_.Transpose(ln("layer_norm1", support), swap), {f1, n_s, dim_});
			Arr q_flat = o_.Reshape(o_.Transpose(ln("layer_norm1", query), swap), {f1, n_q, dim_});
			Arr rm = o_.Reshape(o_.BroadcastTo(row_key_mask, {f1, 1, 1, n_s}), {f1, 1, 1, n_s});
			Arr s_att = Attention(p + "attention1", s_flat, s_flat, rm);
			Arr q_att = Attention(p + "attention1", q_flat, s_flat, rm);
			support = o_.Add(support, o_.Transpose(o_.Reshape(s_att, {1, f1, n_s, dim_}), swap));
			query = o_.Add(query, o_.Transpose(o_.Reshape(q_att, {1, f1, n_q, dim_}), swap));
		}

		// --- MLP block 1 -----------------------------------------------------
		support = o_.Add(support, mlp(ln("layer_norm2", support), "linear1", "linear2"));
		query = o_.Add(query, mlp(ln("layer_norm2", query), "linear1", "linear2"));

		// --- attention across features; keys are the feature columns ---------
		{
			Arr s_feat = o_.Reshape(ln("layer_norm3", support), {n_s, f1, dim_});
			Arr q_feat = o_.Reshape(ln("layer_norm3", query), {n_q, f1, dim_});
			Arr fm_s = o_.Reshape(o_.BroadcastTo(feat_key_mask, {n_s, 1, 1, f1}), {n_s, 1, 1, f1});
			Arr fm_q = o_.Reshape(o_.BroadcastTo(feat_key_mask, {n_q, 1, 1, f1}), {n_q, 1, 1, f1});
			Arr s_fa = Attention(p + "attention2", s_feat, s_feat, fm_s);
			Arr q_fa = Attention(p + "attention2", q_feat, q_feat, fm_q);
			support = o_.Add(support, o_.Reshape(s_fa, {1, n_s, f1, dim_}));
			query = o_.Add(query, o_.Reshape(q_fa, {1, n_q, f1, dim_}));
		}

		// --- MLP block 2 -----------------------------------------------------
		support = o_.Add(support, mlp(ln("layer_norm4", support), "linear3", "linear4"));
		query = o_.Add(query, mlp(ln("layer_norm4", query), "linear3", "linear4"));
	}

	const MitraWeights &w_;
	const Ops &o_;
	int n_heads_;
	bool classification_;
	int n_layers_;
	int dim_;
	int dim_output_;
};

// ---------------------------------------------------------------------------
// Plugin backend
// ---------------------------------------------------------------------------
std::string JoinPath(const std::string &dir, const std::string &leaf) {
	if (dir.empty()) {
		return leaf;
	}
	return dir.back() == '/' ? dir + leaf : dir + "/" + leaf;
}

class MlxPluginBackend {
public:
	MlxPluginBackend(const std::string &weights_path, const std::string &arch, bool classification, int n_heads)
	    : gpu_(/*gpu=*/true), cpu_(/*gpu=*/false), ops_(gpu_.s), weights_(weights_path, cpu_.s),
	      forward_(weights_, ops_, n_heads, classification), arch_(arch) {
	}

	const Ops &ops() const {
		return ops_;
	}
	const MitraForward &forward() const {
		return forward_;
	}

private:
	Stream gpu_; // every forward op
	Stream cpu_; // the safetensors read only -- see MitraWeights
	Ops ops_;
	MitraWeights weights_;
	MitraForward forward_;
	std::string arch_;
};

void SetError(char *err, size_t err_len, const std::string &msg) {
	if (!err || err_len == 0) {
		return;
	}
	std::snprintf(err, err_len, "%s", msg.c_str());
}

const char *PluginName(void) {
	return "mlx";
}

void *PluginCreate(const TabFMPluginCreateParams *params, char *err, size_t err_len) {
	if (!params) {
		SetError(err, err_len, "null params");
		return nullptr;
	}
	try {
		const std::string arch = params->arch ? params->arch : "";
		// Hand-ported forward, so the plugin knows one architecture. Anything
		// else must fail here rather than run mitra's math on other weights.
		if (arch != "mitra" && arch != "mitra-classification" && arch != "mitra-regression") {
			throw MlxError("the mlx backend implements the 'mitra' architecture only, not '" + arch +
			               "'. Use SET anofox_tabfm_model='mitra', or SET anofox_tabfm_device='cpu' to run " +
			               arch + " through ONNX Runtime.");
		}
		const std::string precision = params->precision ? params->precision : "fp32";
		if (precision != "fp32" && precision != "auto" && precision.empty() == false && precision != "default") {
			throw MlxError("the mlx backend currently runs fp32 only, not '" + precision +
			               "'. Use SET anofox_tabfm_precision='fp32'.");
		}
		const std::string weights_dir = params->weights_dir ? params->weights_dir : "";
		const std::string weights_path = JoinPath(weights_dir, "model.safetensors");

		// graph_path is deliberately unused: MLX has no ONNX importer, so the
		// forward is transcribed from mitra_model_patched.py rather than read
		// from the graph. Kept in the ABI because every other backend needs it.
		const bool classification = arch != "mitra-regression";
		auto *backend = new MlxPluginBackend(weights_path, arch, classification, /*n_heads=*/4);
		return backend;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("anofox_tabfm mlx plugin: ") + e.what());
		return nullptr;
	}
}

TabFMPluginStatus PluginRun(void *handle, const TabFMPluginRunInput *input, TabFMPluginRunOutput *output, char *err,
                            size_t err_len) {
	auto *backend = static_cast<MlxPluginBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null handle, input or output");
		return TABFM_PLUGIN_ERROR;
	}
	std::memset(output, 0, sizeof(*output));
	try {
		const Ops &o = backend->ops();
		const int t = static_cast<int>(input->t);
		const int h = static_cast<int>(input->h);
		Arr x = o.FromData(input->x, {1, t, h}, MLX_FLOAT32);
		Arr y = o.FromData(input->y, {1, t}, MLX_FLOAT32);

		Arr logits = backend->forward().Run(x, y, input->train_size, input->d);
		Ok(mlx_array_eval(logits.get()), "eval"); // MLX is lazy; nothing ran until here

		const float *data = mlx_array_data_float32(logits.get());
		if (!data) {
			throw MlxError("logits buffer is null after eval");
		}
		const size_t n = mlx_array_size(logits.get());
		auto *out = static_cast<float *>(std::malloc(n * sizeof(float)));
		if (!out) {
			throw MlxError("out of memory copying logits");
		}
		std::memcpy(out, data, n * sizeof(float));

		std::vector<int> shape = logits.shape();
		auto *shape_out = static_cast<int64_t *>(std::malloc(shape.size() * sizeof(int64_t)));
		if (!shape_out) {
			std::free(out);
			throw MlxError("out of memory copying shape");
		}
		for (size_t i = 0; i < shape.size(); i++) {
			shape_out[i] = shape[i];
		}

		output->logits = out;
		output->logits_len = static_cast<int64_t>(n);
		output->shape = shape_out;
		output->shape_len = static_cast<int64_t>(shape.size());
		return TABFM_PLUGIN_OK;
	} catch (const std::exception &e) {
		SetError(err, err_len, std::string("anofox_tabfm mlx plugin: ") + e.what());
		return TABFM_PLUGIN_ERROR;
	}
}

TabFMPluginStatus PluginPrecompile(void *handle, int64_t rows, int64_t features, char *err, size_t err_len) {
	// Deliberately a no-op that succeeds. MIGraphX compiles per shape bucket and
	// needs warming; MLX's per-shape cost was measured at 217 ms (S-M4, 8% of a
	// single call), so there is nothing here worth precompiling. Reporting
	// success rather than "unsupported" keeps the engine's warm path uniform.
	(void)handle;
	(void)rows;
	(void)features;
	(void)err;
	(void)err_len;
	return TABFM_PLUGIN_OK;
}

void PluginFreeOutput(TabFMPluginRunOutput *output) {
	if (!output) {
		return;
	}
	std::free(output->logits);
	std::free(output->shape);
	output->logits = nullptr;
	output->shape = nullptr;
	output->logits_len = 0;
	output->shape_len = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<MlxPluginBackend *>(handle);
}

const TabFMPluginApi kApi = {
    TABFM_PLUGIN_ABI_VERSION, PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" TABFM_PLUGIN_EXPORT const TabFMPluginApi *TabFMGetPluginApi(void) {
	return &kApi;
}
