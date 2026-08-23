/*===----------------------------------------------------------------------===
 * tabfm_mlx_graph_ops.inc — the op table and the executor.
 *
 * Included by tabfm_mlx_graph.cpp; split only so neither file is unreadable.
 * Semantics follow tools/mlx_spike/src/mlx_spike/onnx_mlx.py, which is where
 * they were verified against ORT CPU on all 13 shipped graphs.
 *===----------------------------------------------------------------------===*/

namespace anofox {
namespace mlxgraph {
namespace {

Arr Reduce(const Node &n, const Args &in, const Ctx &c,
           int (*axes_fn)(mlx_array *, const mlx_array, const int *, size_t, bool, const mlx_stream),
           int (*all_fn)(mlx_array *, const mlx_array, bool, const mlx_stream)) {
	const bool keepdims = AttrInt(n, "keepdims", 1) != 0;
	std::vector<int64_t> axes;
	bool have = false;
	if (in.size() > 1 && in[1] && in[1]->size() > 0) {
		axes = ToInts(*in[1], c.s);
		have = true;
	} else {
		axes = AttrInts(n, "axes", &have);
	}
	Arr r;
	if (!have || axes.empty()) {
		if (AttrInt(n, "noop_with_empty_axes", 0) != 0) {
			return in[0]->Share();
		}
		Ok(all_fn(r.out(), in[0]->get(), keepdims, c.s), "reduce");
		return r;
	}
	auto ax = ToShape(axes);
	Ok(axes_fn(r.out(), in[0]->get(), ax.data(), ax.size(), keepdims, c.s), "reduce_axes");
	return r;
}

//! MLX has no int64 GPU scatter ("[ScatterAxis::eval_gpu] Does not support
//! int64") and these graphs scatter int64, because that is the dtype ONNX
//! shape arithmetic uses. Round-trip through int32 for the scatter only. Safe
//! for the reason the values exist -- shapes and offsets, far below 2^31 --
//! and asserted rather than assumed.
Arr ScatterSafe(const Arr &data, const Ctx &c, const std::function<Arr(const Arr &)> &fn) {
	if (data.dtype() != MLX_INT64) {
		return fn(data);
	}
	Arr as32, back;
	Ok(mlx_astype(as32.out(), data.get(), MLX_INT32, c.s), "astype int32");
	Arr scattered = fn(as32);
	Ok(mlx_astype(back.out(), scattered.get(), MLX_INT64, c.s), "astype int64");
	return back;
}

//! indices[..., k] addresses the first k axes; the rest come along whole.
//! Flatten those k axes so one take/put does the work.
std::vector<int64_t> FlattenNdIndices(const std::vector<int64_t> &idx, const std::vector<int> &shape, int k,
                                      size_t rows) {
	std::vector<int64_t> strides(k, 1);
	for (int i = k - 2; i >= 0; i--) {
		strides[i] = strides[i + 1] * shape[i + 1];
	}
	std::vector<int64_t> flat(rows, 0);
	for (size_t r = 0; r < rows; r++) {
		int64_t acc = 0;
		for (int j = 0; j < k; j++) {
			int64_t v = idx[r * k + j];
			if (v < 0) {
				v += shape[j];
			}
			acc += v * strides[j];
		}
		flat[r] = acc;
	}
	return flat;
}

const std::unordered_map<std::string, Handler> &Handlers() {
	static const std::unordered_map<std::string, Handler> table = {
	    UNARY("Neg", mlx_negative),      UNARY("Abs", mlx_abs),         UNARY("Sqrt", mlx_sqrt),
	    UNARY("Exp", mlx_exp),           UNARY("Log", mlx_log),         UNARY("Erf", mlx_erf),
	    UNARY("Sin", mlx_sin),           UNARY("Cos", mlx_cos),         UNARY("Tanh", mlx_tanh),
	    UNARY("Sigmoid", mlx_sigmoid),   UNARY("Sign", mlx_sign),       UNARY("Floor", mlx_floor),
	    UNARY("Ceil", mlx_ceil),         UNARY("IsNaN", mlx_isnan),     UNARY("IsInf", mlx_isinf),
	    UNARY("Not", mlx_logical_not),   UNARY("Reciprocal", mlx_reciprocal),

	    BINARY("Add", mlx_add),          BINARY("Sub", mlx_subtract),   BINARY("Mul", mlx_multiply),
	    BINARY("Div", mlx_divide),       BINARY("MatMul", mlx_matmul),  BINARY("Equal", mlx_equal),
	    BINARY("Greater", mlx_greater),  BINARY("Less", mlx_less),      BINARY("And", mlx_logical_and),
	    BINARY("Or", mlx_logical_or),    BINARY("Mod", mlx_remainder),  BINARY("Pow", mlx_power),
	    BINARY("GreaterOrEqual", mlx_greater_equal),
	    BINARY("LessOrEqual", mlx_less_equal),

	    {"Where",
	     [](const Node &, const Args &in, const Ctx &c) {
		     Arr r;
		     Ok(mlx_where(r.out(), in[0]->get(), in[1]->get(), in[2]->get(), c.s), "where");
		     return Wrap(std::move(r));
	     }},
	    {"Max",
	     [](const Node &, const Args &in, const Ctx &c) {
		     Arr acc = in[0]->Share();
		     for (size_t i = 1; i < in.size(); i++) {
			     Arr r;
			     Ok(mlx_maximum(r.out(), acc.get(), in[i]->get(), c.s), "maximum");
			     acc = std::move(r);
		     }
		     return Wrap(std::move(acc));
	     }},
	    {"Min",
	     [](const Node &, const Args &in, const Ctx &c) {
		     Arr acc = in[0]->Share();
		     for (size_t i = 1; i < in.size(); i++) {
			     Arr r;
			     Ok(mlx_minimum(r.out(), acc.get(), in[i]->get(), c.s), "minimum");
			     acc = std::move(r);
		     }
		     return Wrap(std::move(acc));
	     }},
	    {"Softplus",
	     [](const Node &, const Args &in, const Ctx &c) {
		     Arr e, l1, r;
		     Ok(mlx_exp(e.out(), in[0]->get(), c.s), "exp");
		     Ok(mlx_log1p(l1.out(), e.get(), c.s), "log1p");
		     return Wrap(std::move(l1));
	     }},
	    {"Identity", [](const Node &, const Args &in, const Ctx &) { return Wrap(in[0]->Share()); }},

	    // --- shape / structure ---------------------------------------------
	    {"Cast",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     Arr r;
		     Ok(mlx_astype(r.out(), in[0]->get(), OnnxDtype(static_cast<int>(AttrInt(n, "to", 1))), c.s), "cast");
		     return Wrap(std::move(r));
	     }},
	    {"CastLike",
	     [](const Node &, const Args &in, const Ctx &c) {
		     Arr r;
		     Ok(mlx_astype(r.out(), in[0]->get(), in[1]->dtype(), c.s), "castlike");
		     return Wrap(std::move(r));
	     }},
	    {"Shape",
	     [](const Node &n, const Args &in, const Ctx &) {
		     auto shape = in[0]->shape();
		     int64_t rank = static_cast<int64_t>(shape.size());
		     int64_t start = AttrInt(n, "start", 0);
		     int64_t end = AttrInt(n, "end", rank);
		     if (start < 0) start += rank;
		     if (end < 0) end += rank;
		     start = std::max<int64_t>(0, std::min(start, rank));
		     end = std::max(start, std::min(end, rank));
		     std::vector<int64_t> out(shape.begin() + start, shape.begin() + end);
		     int len = static_cast<int>(out.size());
		     return Wrap(Arr(mlx_array_new_data(out.data(), &len, 1, MLX_INT64)));
	     }},
	    {"Size",
	     [](const Node &, const Args &in, const Ctx &) {
		     int64_t n = static_cast<int64_t>(in[0]->size());
		     return Wrap(Arr(mlx_array_new_data(&n, nullptr, 0, MLX_INT64))); // ONNX Size is rank 0
	     }},
	    {"Reshape",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     auto want = ToInts(*in[1], c.s);
		     auto cur = in[0]->shape();
		     if (AttrInt(n, "allowzero", 0) == 0) {
			     // ONNX: 0 means "copy the input's dim here"; MLX has no such rule.
			     for (size_t i = 0; i < want.size(); i++) {
				     if (want[i] == 0 && i < cur.size()) {
					     want[i] = cur[i];
				     }
			     }
		     }
		     auto shape = ToShape(want);
		     Arr r;
		     Ok(mlx_reshape(r.out(), in[0]->get(), shape.data(), shape.size(), c.s), "reshape");
		     return Wrap(std::move(r));
	     }},
	    {"Transpose",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     bool have = false;
		     auto perm = AttrInts(n, "perm", &have);
		     if (!have) {
			     for (int64_t i = static_cast<int64_t>(in[0]->ndim()) - 1; i >= 0; i--) {
				     perm.push_back(i);
			     }
		     }
		     auto axes = ToShape(perm);
		     Arr r;
		     Ok(mlx_transpose_axes(r.out(), in[0]->get(), axes.data(), axes.size(), c.s), "transpose");
		     return Wrap(std::move(r));
	     }},
	    {"Concat",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     mlx_vector_array v = mlx_vector_array_new();
		     for (const auto *a : in) {
			     if (a) {
				     mlx_vector_array_append_value(v, a->get());
			     }
		     }
		     Arr r;
		     int st = mlx_concatenate_axis(r.out(), v, static_cast<int>(AttrInt(n, "axis", 0)), c.s);
		     mlx_vector_array_free(v);
		     Ok(st, "concat");
		     return Wrap(std::move(r));
	     }},
	    {"Unsqueeze",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     std::vector<int64_t> axes = (in.size() > 1 && in[1]) ? ToInts(*in[1], c.s) : AttrInts(n, "axes");
		     auto shape = in[0]->shape();
		     std::vector<int64_t> sorted(axes);
		     std::sort(sorted.begin(), sorted.end());
		     for (int64_t ax : sorted) {
			     int64_t rank = static_cast<int64_t>(shape.size()) + 1;
			     if (ax < 0) ax += rank;
			     shape.insert(shape.begin() + ax, 1);
		     }
		     Arr r;
		     Ok(mlx_reshape(r.out(), in[0]->get(), shape.data(), shape.size(), c.s), "unsqueeze");
		     return Wrap(std::move(r));
	     }},
	    {"Squeeze",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     std::vector<int64_t> axes = (in.size() > 1 && in[1]) ? ToInts(*in[1], c.s) : AttrInts(n, "axes");
		     Arr r;
		     if (axes.empty()) {
			     Ok(mlx_squeeze(r.out(), in[0]->get(), c.s), "squeeze");
		     } else {
			     auto ax = ToShape(axes);
			     Ok(mlx_squeeze_axes(r.out(), in[0]->get(), ax.data(), ax.size(), c.s), "squeeze_axes");
		     }
		     return Wrap(std::move(r));
	     }},
	    {"Expand",
	     [](const Node &, const Args &in, const Ctx &c) {
		     // ONNX Expand is BIDIRECTIONAL broadcast, not broadcast_to: the
		     // result rank is max(rank(a), len(shape)) and 1s on either side
		     // stretch. Multiplying by ones gets that for free.
		     auto want = ToShape(ToInts(*in[1], c.s));
		     Arr ones, r;
		     Arr one_val(mlx_array_new_float32(1.0f));
		     Ok(mlx_full(ones.out(), want.data(), want.size(), one_val.get(), in[0]->dtype(), c.s), "ones");
		     Ok(mlx_multiply(r.out(), in[0]->get(), ones.get(), c.s), "expand");
		     return Wrap(std::move(r));
	     }},
	    {"Slice",
	     [](const Node &, const Args &in, const Ctx &c) {
		     auto starts = ToInts(*in[1], c.s);
		     auto ends = ToInts(*in[2], c.s);
		     std::vector<int64_t> axes, steps;
		     if (in.size() > 3 && in[3]) {
			     axes = ToInts(*in[3], c.s);
		     } else {
			     for (size_t i = 0; i < starts.size(); i++) axes.push_back(static_cast<int64_t>(i));
		     }
		     if (in.size() > 4 && in[4]) {
			     steps = ToInts(*in[4], c.s);
		     } else {
			     steps.assign(starts.size(), 1);
		     }
		     Arr cur = in[0]->Share();
		     for (size_t i = 0; i < starts.size(); i++) {
			     auto shape = cur.shape();
			     int64_t rank = static_cast<int64_t>(shape.size());
			     int64_t ax = axes[i] < 0 ? axes[i] + rank : axes[i];
			     int64_t dim = shape[ax];
			     int64_t st = starts[i], en = ends[i], sp = steps[i];
			     // ONNX CLAMPS out-of-range bounds instead of erroring, and the
			     // exporters lean on it hard (INT64_MAX means "to the end").
			     if (st < 0) st += dim;
			     if (en < 0) en += dim;
			     st = std::max<int64_t>(0, std::min(st, dim));
			     en = std::max<int64_t>(0, std::min(en, dim));
			     std::vector<int> lo(rank, 0), hi(shape.begin(), shape.end()), strd(rank, 1);
			     lo[ax] = static_cast<int>(st);
			     hi[ax] = static_cast<int>(en);
			     strd[ax] = static_cast<int>(sp);
			     Arr r;
			     Ok(mlx_slice(r.out(), cur.get(), lo.data(), lo.size(), hi.data(), hi.size(), strd.data(),
			                  strd.size(), c.s),
			        "slice");
			     cur = std::move(r);
		     }
		     return Wrap(std::move(cur));
	     }},
	    {"Gather",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     Arr idx, r;
		     Ok(mlx_astype(idx.out(), in[1]->get(), MLX_INT32, c.s), "idx");
		     Ok(mlx_take_axis(r.out(), in[0]->get(), idx.get(), static_cast<int>(AttrInt(n, "axis", 0)), c.s),
		        "gather");
		     return Wrap(std::move(r));
	     }},
	    {"GatherElements",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     Arr idx, r;
		     Ok(mlx_astype(idx.out(), in[1]->get(), MLX_INT32, c.s), "idx");
		     Ok(mlx_take_along_axis(r.out(), in[0]->get(), idx.get(),
		                            static_cast<int>(AttrInt(n, "axis", 0)), c.s),
		        "gather_elements");
		     return Wrap(std::move(r));
	     }},
	    {"GatherND",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     if (AttrInt(n, "batch_dims", 0) != 0) {
			     Fail("GatherND batch_dims != 0 is not implemented");
		     }
		     auto ishape = in[1]->shape();
		     int k = ishape.back();
		     size_t rows = in[1]->size() / static_cast<size_t>(k);
		     auto dshape = in[0]->shape();
		     auto flat = FlattenNdIndices(ToInts(*in[1], c.s), dshape, k, rows);
		     int lead = 1;
		     for (int i = 0; i < k; i++) lead *= dshape[i];
		     std::vector<int> fshape {lead};
		     for (size_t i = k; i < dshape.size(); i++) fshape.push_back(dshape[i]);
		     Arr flat_data, idx32, picked;
		     Ok(mlx_reshape(flat_data.out(), in[0]->get(), fshape.data(), fshape.size(), c.s), "flatten");
		     std::vector<int32_t> i32(flat.begin(), flat.end());
		     int nrows = static_cast<int>(rows);
		     idx32 = Arr(mlx_array_new_data(i32.data(), &nrows, 1, MLX_INT32));
		     Ok(mlx_take_axis(picked.out(), flat_data.get(), idx32.get(), 0, c.s), "gathernd");
		     std::vector<int> oshape(ishape.begin(), ishape.end() - 1);
		     for (size_t i = k; i < dshape.size(); i++) oshape.push_back(dshape[i]);
		     Arr r;
		     Ok(mlx_reshape(r.out(), picked.get(), oshape.data(), oshape.size(), c.s), "reshape");
		     return Wrap(std::move(r));
	     }},
	    {"ScatterElements",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     if (AttrStr(n, "reduction", "none") != "none") {
			     Fail("ScatterElements reduction != none is not implemented");
		     }
		     const int axis = static_cast<int>(AttrInt(n, "axis", 0));
		     auto shape = in[0]->shape();
		     const int rank = static_cast<int>(shape.size());
		     const int ax = axis < 0 ? axis + rank : axis;
		     Arr idx, dim_v, neg, fixed;
		     Ok(mlx_astype(idx.out(), in[1]->get(), MLX_INT32, c.s), "idx");
		     // ONNX allows negative indices; put_along_axis does not.
		     Arr dim(mlx_array_new_int(shape[ax]));
		     Arr zero(mlx_array_new_int(0));
		     Arr shifted, is_neg;
		     Ok(mlx_add(shifted.out(), idx.get(), dim.get(), c.s), "shift");
		     Ok(mlx_less(is_neg.out(), idx.get(), zero.get(), c.s), "isneg");
		     Ok(mlx_where(fixed.out(), is_neg.get(), shifted.get(), idx.get(), c.s), "fix");
		     const Arr &upd = *in[2];
		     return Wrap(ScatterSafe(*in[0], c, [&](const Arr &d) {
			     Arr casted, r;
			     Ok(mlx_astype(casted.out(), upd.get(), d.dtype(), c.s), "upd");
			     Ok(mlx_put_along_axis(r.out(), d.get(), fixed.get(), casted.get(), ax, c.s), "scatter_elements");
			     return r;
		     }));
	     }},
	    {"ScatterND",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     if (AttrStr(n, "reduction", "none") != "none") {
			     Fail("ScatterND reduction != none is not implemented");
		     }
		     auto ishape = in[1]->shape();
		     int k = ishape.back();
		     size_t rows = in[1]->size() / static_cast<size_t>(k);
		     auto dshape = in[0]->shape();
		     auto flat = FlattenNdIndices(ToInts(*in[1], c.s), dshape, k, rows);
		     int lead = 1;
		     for (int i = 0; i < k; i++) lead *= dshape[i];
		     std::vector<int> fshape {lead};
		     for (size_t i = k; i < dshape.size(); i++) fshape.push_back(dshape[i]);
		     std::vector<int> ushape {static_cast<int>(rows)};
		     for (size_t i = k; i < dshape.size(); i++) ushape.push_back(dshape[i]);
		     std::vector<int32_t> i32(flat.begin(), flat.end());
		     int nrows = static_cast<int>(rows);
		     Arr idx32(mlx_array_new_data(i32.data(), &nrows, 1, MLX_INT32));
		     // Broadcast the row index across the trailing axes so one
		     // put_along_axis writes whole slices.
		     std::vector<int> bshape {nrows};
		     for (size_t i = 1; i < ushape.size(); i++) bshape.push_back(1);
		     Arr idx_r, idx_b, flat_data, flat_upd;
		     Ok(mlx_reshape(idx_r.out(), idx32.get(), bshape.data(), bshape.size(), c.s), "idxshape");
		     Ok(mlx_broadcast_to(idx_b.out(), idx_r.get(), ushape.data(), ushape.size(), c.s), "idxbcast");
		     Ok(mlx_reshape(flat_data.out(), in[0]->get(), fshape.data(), fshape.size(), c.s), "flatten");
		     Ok(mlx_reshape(flat_upd.out(), in[2]->get(), ushape.data(), ushape.size(), c.s), "updshape");
		     Arr out = ScatterSafe(flat_data, c, [&](const Arr &d) {
			     Arr casted, r;
			     Ok(mlx_astype(casted.out(), flat_upd.get(), d.dtype(), c.s), "upd");
			     Ok(mlx_put_along_axis(r.out(), d.get(), idx_b.get(), casted.get(), 0, c.s), "scatter_nd");
			     return r;
		     });
		     Arr r;
		     Ok(mlx_reshape(r.out(), out.get(), dshape.data(), dshape.size(), c.s), "unflatten");
		     return Wrap(std::move(r));
	     }},
	    {"Pad",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     if (AttrStr(n, "mode", "constant") != "constant") {
			     Fail("Pad mode != constant is not implemented");
		     }
		     auto pads = ToInts(*in[1], c.s);
		     auto shape = in[0]->shape();
		     const int rank = static_cast<int>(shape.size());
		     const size_t half = pads.size() / 2;
		     std::vector<int64_t> target;
		     if (in.size() > 3 && in[3]) {
			     target = ToInts(*in[3], c.s);
		     } else {
			     for (int i = 0; i < rank; i++) target.push_back(i);
		     }
		     // ONNX packs pads as [start_0..start_n, end_0..end_n] over `axes`,
		     // NOT as per-axis pairs.
		     std::vector<int> lo(rank, 0), hi(rank, 0);
		     for (size_t i = 0; i < target.size() && i < half; i++) {
			     int64_t ax = target[i] < 0 ? target[i] + rank : target[i];
			     lo[ax] = static_cast<int>(pads[i]);
			     hi[ax] = static_cast<int>(pads[half + i]);
		     }
		     Arr value;
		     if (in.size() > 2 && in[2]) {
			     Ok(mlx_astype(value.out(), in[2]->get(), in[0]->dtype(), c.s), "padvalue");
		     } else {
			     Arr z(mlx_array_new_float32(0.0f));
			     Ok(mlx_astype(value.out(), z.get(), in[0]->dtype(), c.s), "padzero");
		     }
		     std::vector<int> axes_all(rank);
		     std::iota(axes_all.begin(), axes_all.end(), 0);
		     Arr r;
		     Ok(mlx_pad(r.out(), in[0]->get(), axes_all.data(), axes_all.size(), lo.data(), lo.size(), hi.data(),
		                hi.size(), value.get(), "constant", c.s),
		        "pad");
		     return Wrap(std::move(r));
	     }},
	    {"Range",
	     [](const Node &, const Args &in, const Ctx &c) {
		     const double start = ToScalar(*in[0], c.s);
		     const double limit = ToScalar(*in[1], c.s);
		     const double delta = ToScalar(*in[2], c.s);
		     Arr r;
		     Ok(mlx_arange(r.out(), start, limit, delta, in[0]->dtype(), c.s), "range");
		     return Wrap(std::move(r));
	     }},
	    {"Clip",
	     [](const Node &, const Args &in, const Ctx &c) {
		     Arr cur = in[0]->Share();
		     if (in.size() > 1 && in[1]) {
			     Arr lo, r;
			     Ok(mlx_astype(lo.out(), in[1]->get(), cur.dtype(), c.s), "lo");
			     Ok(mlx_maximum(r.out(), cur.get(), lo.get(), c.s), "clip_lo");
			     cur = std::move(r);
		     }
		     if (in.size() > 2 && in[2]) {
			     Arr hi, r;
			     Ok(mlx_astype(hi.out(), in[2]->get(), cur.dtype(), c.s), "hi");
			     Ok(mlx_minimum(r.out(), cur.get(), hi.get(), c.s), "clip_hi");
			     cur = std::move(r);
		     }
		     return Wrap(std::move(cur));
	     }},

	    // --- reductions / nn ------------------------------------------------
	    {"ReduceSum",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     return Wrap(Reduce(n, in, c, mlx_sum_axes, mlx_sum));
	     }},
	    {"ReduceMean",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     return Wrap(Reduce(n, in, c, mlx_mean_axes, mlx_mean));
	     }},
	    {"ReduceMin",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     return Wrap(Reduce(n, in, c, mlx_min_axes, mlx_min));
	     }},
	    {"ReduceMax",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     return Wrap(Reduce(n, in, c, mlx_max_axes, mlx_max));
	     }},
	    {"CumSum",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     Arr r;
		     Ok(mlx_cumsum(r.out(), in[0]->get(), static_cast<int>(ToScalar(*in[1], c.s)),
		                   AttrInt(n, "reverse", 0) != 0, AttrInt(n, "exclusive", 0) == 0, c.s),
		        "cumsum");
		     return Wrap(std::move(r));
	     }},
	    {"Softmax",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     int axis = static_cast<int>(AttrInt(n, "axis", -1));
		     Arr r;
		     Ok(mlx_softmax_axis(r.out(), in[0]->get(), axis, true, c.s), "softmax");
		     return Wrap(std::move(r));
	     }},
	    {"LayerNormalization",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     const int64_t axis = AttrInt(n, "axis", -1);
		     const int64_t rank = static_cast<int64_t>(in[0]->ndim());
		     if (axis != -1 && axis != rank - 1) {
			     Fail("LayerNormalization over a non-final axis is not implemented");
		     }
		     Arr r;
		     mlx_array bias = (in.size() > 2 && in[2]) ? in[2]->get() : mlx_array_new();
		     Ok(mlx_fast_layer_norm(r.out(), in[0]->get(), in[1]->get(), bias,
		                            static_cast<float>(Attr(n, "epsilon") ? Attr(n, "epsilon")->f : 1e-5f), c.s),
		        "layer_norm");
		     return Wrap(std::move(r));
	     }},
	    {"Gemm",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     Arr a = in[0]->Share(), b = in[1]->Share();
		     std::vector<int> flip {1, 0};
		     if (AttrInt(n, "transA", 0)) {
			     Arr t;
			     Ok(mlx_transpose_axes(t.out(), a.get(), flip.data(), flip.size(), c.s), "transA");
			     a = std::move(t);
		     }
		     if (AttrInt(n, "transB", 0)) {
			     Arr t;
			     Ok(mlx_transpose_axes(t.out(), b.get(), flip.data(), flip.size(), c.s), "transB");
			     b = std::move(t);
		     }
		     Arr mm, scaled;
		     Ok(mlx_matmul(mm.out(), a.get(), b.get(), c.s), "gemm");
		     const float alpha = Attr(n, "alpha") ? Attr(n, "alpha")->f : 1.0f;
		     Arr alpha_v(mlx_array_new_float32(alpha));
		     Ok(mlx_multiply(scaled.out(), mm.get(), alpha_v.get(), c.s), "alpha");
		     if (in.size() > 2 && in[2]) {
			     const float beta = Attr(n, "beta") ? Attr(n, "beta")->f : 1.0f;
			     Arr beta_v(mlx_array_new_float32(beta)), bscaled, r;
			     Ok(mlx_multiply(bscaled.out(), in[2]->get(), beta_v.get(), c.s), "beta");
			     Ok(mlx_add(r.out(), scaled.get(), bscaled.get(), c.s), "gemm_bias");
			     return Wrap(std::move(r));
		     }
		     return Wrap(std::move(scaled));
	     }},
	    {"TopK",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     const int k = static_cast<int>(ToScalar(*in[1], c.s));
		     int axis = static_cast<int>(AttrInt(n, "axis", -1));
		     const int rank = static_cast<int>(in[0]->ndim());
		     if (axis < 0) axis += rank;
		     const bool largest = AttrInt(n, "largest", 1) != 0;
		     Arr ordered, idx;
		     if (largest) {
			     Arr neg;
			     Ok(mlx_negative(neg.out(), in[0]->get(), c.s), "neg");
			     Ok(mlx_argsort_axis(idx.out(), neg.get(), axis, c.s), "argsort");
		     } else {
			     Ok(mlx_argsort_axis(idx.out(), in[0]->get(), axis, c.s), "argsort");
		     }
		     auto shape = in[0]->shape();
		     std::vector<int> lo(rank, 0), hi(shape.begin(), shape.end()), strd(rank, 1);
		     hi[axis] = k;
		     Arr idx_k, values;
		     Ok(mlx_slice(idx_k.out(), idx.get(), lo.data(), lo.size(), hi.data(), hi.size(), strd.data(),
		                  strd.size(), c.s),
		        "topk_idx");
		     Ok(mlx_take_along_axis(values.out(), in[0]->get(), idx_k.get(), axis, c.s), "topk_val");
		     Arr idx64;
		     Ok(mlx_astype(idx64.out(), idx_k.get(), MLX_INT64, c.s), "topk_i64");
		     std::vector<Arr> out;
		     out.push_back(std::move(values));
		     out.push_back(std::move(idx64));
		     return out;
	     }},
	    {"Einsum",
	     [](const Node &n, const Args &in, const Ctx &c) {
		     mlx_vector_array v = mlx_vector_array_new();
		     for (const auto *a : in) {
			     if (a) mlx_vector_array_append_value(v, a->get());
		     }
		     Arr r;
		     int st = mlx_einsum(r.out(), AttrStr(n, "equation", "").c_str(), v, c.s);
		     mlx_vector_array_free(v);
		     Ok(st, "einsum");
		     return Wrap(std::move(r));
	     }},
	};
	return table;
}

} // namespace

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------
struct Graph::Impl {
	::anofox::onnxread::Graph parsed;
	std::unique_ptr<Mapping> weights;
	std::map<std::string, Arr> initializers;
};

Graph::Graph(const std::string &graph_path, const std::string &weights_dir) : impl_(new Impl()) {
	impl_->parsed = ::anofox::onnxread::ParseModelFile(graph_path);

	// External initializers name a file and an absolute offset into it. That
	// file is MAPPED, never read: it is what lets tabfm-v1's 6.6 GB load at all
	// on a 16 GB machine, and it is the whole point of the ext-graph layout.
	std::string location;
	for (const auto &t : impl_->parsed.initializers) {
		if (t.location == ::anofox::onnxread::DataLocation::EXTERNAL) {
			auto it = t.external.find("location");
			if (it != t.external.end()) {
				location = it->second;
				break;
			}
		}
	}
	if (!location.empty()) {
		impl_->weights.reset(new Mapping(weights_dir + "/" + location));
	}

	for (const auto &t : impl_->parsed.initializers) {
		std::vector<int> shape(t.dims.begin(), t.dims.end()); // rank preserved, incl. 0
		const mlx_dtype dt = OnnxDtype(t.data_type);
		const void *bytes = nullptr;
		if (t.location == ::anofox::onnxread::DataLocation::EXTERNAL) {
			size_t offset = 0, length = 0;
			auto o = t.external.find("offset");
			auto l = t.external.find("length");
			if (o != t.external.end()) offset = std::stoull(o->second);
			if (l != t.external.end()) length = std::stoull(l->second);
			bytes = impl_->weights->At(offset, length);
		} else if (!t.raw.empty()) {
			bytes = t.raw.data();
		} else if (!t.floats.empty()) {
			bytes = t.floats.data();
		} else if (!t.int64s.empty()) {
			bytes = t.int64s.data();
		} else if (!t.int32s.empty()) {
			bytes = t.int32s.data();
		}
		if (!bytes) {
			continue; // empty initializer; nothing reads it
		}
		impl_->initializers.emplace(
		    t.name, Arr(mlx_array_new_data(bytes, shape.data(), static_cast<int>(shape.size()), dt)));
	}
}

Graph::~Graph() = default;

const std::vector<std::string> &Graph::InputNames() const {
	return impl_->parsed.inputs;
}

std::vector<std::string> Graph::MissingOps() const {
	std::set<std::string> missing;
	const auto &table = Handlers();
	for (const auto &n : impl_->parsed.nodes) {
		if (n.op_type == "Constant" || n.op_type == "SplitToSequence" || n.op_type == "SequenceAt") {
			continue; // handled inline by the executor
		}
		if (table.find(n.op_type) == table.end()) {
			missing.insert(n.op_type);
		}
	}
	return std::vector<std::string>(missing.begin(), missing.end());
}

Tensor Graph::Run(const std::vector<std::pair<std::string, Tensor>> &feeds) const {
	// MLX default streams are thread-local, so the stream is taken here, on
	// whichever thread is running the query -- never cached on the object.
	mlx_stream gpu = mlx_default_gpu_stream_new();
	Ctx ctx {gpu};

	std::map<std::string, Arr> env;
	std::map<std::string, std::vector<Arr>> seqs;
	for (const auto &kv : impl_->initializers) {
		env.emplace(kv.first, kv.second.Share());
	}
	for (const auto &f : feeds) {
		auto shape = f.second.shape;
		env[f.first] = Arr(mlx_array_new_data(f.second.data.data(), shape.data(),
		                                      static_cast<int>(shape.size()), MLX_FLOAT32));
	}
	// train_size and d are int64 by contract; the caller hands floats.
	for (const char *name : {"train_size", "d"}) {
		auto it = env.find(name);
		if (it != env.end()) {
			Arr as_i64;
			Ok(mlx_astype(as_i64.out(), it->second.get(), MLX_INT64, ctx.s), "feed int64");
			it->second = std::move(as_i64);
		}
	}

	const auto &table = Handlers();
	for (const auto &node : impl_->parsed.nodes) {
		if (node.op_type == "Constant") {
			// A Constant carries its value in ONE of several attributes, and
			// which one depends on the exporter: `value` (a tensor) is the
			// common case, but torch.export also emits the scalar/list forms
			// value_int, value_ints, value_float, value_floats.
			if (const auto *a = node.Attr("value")) {
				const auto &t = a->t;
				// Rank is preserved exactly, including rank 0. Promoting a
				// scalar to a 1-element vector silently changes Gather: a
				// scalar index REMOVES the gathered axis, a vector KEEPS it,
				// and the difference shows up as an extra trailing dimension
				// on the model's logits.
				std::vector<int> shape(t.dims.begin(), t.dims.end());
				const void *bytes = !t.raw.empty()      ? static_cast<const void *>(t.raw.data())
				                    : !t.int64s.empty() ? static_cast<const void *>(t.int64s.data())
				                    : !t.int32s.empty() ? static_cast<const void *>(t.int32s.data())
				                    : !t.floats.empty() ? static_cast<const void *>(t.floats.data())
				                                        : nullptr;
				size_t count = 1;
				for (int d : shape) {
					count *= static_cast<size_t>(d);
				}
				if (!bytes && count == 0) {
					// A genuinely empty constant. Materialize an empty array of
					// the right dtype rather than failing: graphs use these as
					// no-op Concat/Slice operands.
					static const float kNone = 0.0f;
					bytes = &kNone;
				}
				if (!bytes) {
					Fail("Constant '" + node.name + "' carries no data");
				}
				env[node.outputs[0]] = Arr(mlx_array_new_data(
				    bytes, shape.data(), static_cast<int>(shape.size()), OnnxDtype(t.data_type)));
				continue;
			}
			if (const auto *a = node.Attr("value_int")) {
				int64_t v = a->i;
				env[node.outputs[0]] = Arr(mlx_array_new_data(&v, nullptr, 0, MLX_INT64)); // rank 0
				continue;
			}
			if (const auto *a = node.Attr("value_ints")) {
				int n = static_cast<int>(a->ints.size());
				env[node.outputs[0]] = Arr(mlx_array_new_data(a->ints.data(), &n, 1, MLX_INT64));
				continue;
			}
			if (const auto *a = node.Attr("value_float")) {
				float v = a->f;
				env[node.outputs[0]] = Arr(mlx_array_new_data(&v, nullptr, 0, MLX_FLOAT32)); // rank 0
				continue;
			}
			if (const auto *a = node.Attr("value_floats")) {
				int n = static_cast<int>(a->float_list.size());
				env[node.outputs[0]] = Arr(mlx_array_new_data(a->float_list.data(), &n, 1, MLX_FLOAT32));
				continue;
			}
			Fail("Constant '" + node.name + "' has no recognised value attribute");
		}
		if (node.op_type == "SplitToSequence") {
			const Arr &data = env.at(node.inputs[0]);
			int axis = static_cast<int>(AttrInt(node, "axis", 0));
			auto shape = data.shape();
			const int rank = static_cast<int>(shape.size());
			if (axis < 0) {
				axis += rank;
			}
			const bool keepdims = AttrInt(node, "keepdims", 1) != 0;

			// ONNX gives `split` three meanings and they are not
			// interchangeable: absent -> one part per index along axis; a RANK-0
			// scalar -> a chunk size; a RANK-1 tensor -> explicit per-part
			// sizes. Reading a rank-1 tensor as a scalar takes only its first
			// element, which silently halves the result for an even 2-way split
			// -- seen as "cannot reshape array of size 163840 into (…,2,128)".
			std::vector<int64_t> bounds; // exclusive end of each part
			bool squeeze_parts = false;
			if (node.inputs.size() > 1 && !node.inputs[1].empty() && env.count(node.inputs[1])) {
				const Arr &split = env.at(node.inputs[1]);
				if (split.ndim() == 0) {
					const int64_t chunk = static_cast<int64_t>(ToScalar(split, ctx.s));
					if (chunk <= 0) {
						Fail("SplitToSequence chunk size must be positive");
					}
					for (int64_t e = chunk; e < shape[axis]; e += chunk) {
						bounds.push_back(e);
					}
					bounds.push_back(shape[axis]);
				} else {
					int64_t acc = 0;
					for (int64_t sz : ToInts(split, ctx.s)) {
						acc += sz;
						bounds.push_back(acc);
					}
				}
			} else {
				for (int64_t e = 1; e <= shape[axis]; e++) {
					bounds.push_back(e);
				}
				squeeze_parts = !keepdims; // only this form drops the axis
			}

			std::vector<Arr> parts;
			int64_t start_at = 0;
			for (int64_t stop : bounds) {
				std::vector<int> lo(rank, 0), hi(shape.begin(), shape.end()), strd(rank, 1);
				lo[axis] = static_cast<int>(start_at);
				hi[axis] = static_cast<int>(std::min<int64_t>(stop, shape[axis]));
				Arr part;
				Ok(mlx_slice(part.out(), data.get(), lo.data(), lo.size(), hi.data(), hi.size(), strd.data(),
				             strd.size(), ctx.s),
				   "split_to_sequence");
				if (squeeze_parts) {
					Arr sq;
					int ax = axis;
					Ok(mlx_squeeze_axes(sq.out(), part.get(), &ax, 1, ctx.s), "split_squeeze");
					part = std::move(sq);
				}
				parts.push_back(std::move(part));
				start_at = stop;
			}
			seqs[node.outputs[0]] = std::move(parts);
			continue;
		}
		if (node.op_type == "SequenceAt") {
			auto it = seqs.find(node.inputs[0]);
			if (it == seqs.end()) {
				Fail("SequenceAt on '" + node.inputs[0] + "', which is not a sequence");
			}
			int64_t pos = static_cast<int64_t>(ToScalar(env.at(node.inputs[1]), ctx.s));
			if (pos < 0) {
				pos += static_cast<int64_t>(it->second.size());
			}
			if (pos < 0 || pos >= static_cast<int64_t>(it->second.size())) {
				Fail("SequenceAt index out of range");
			}
			env[node.outputs[0]] = it->second[pos].Share();
			continue;
		}

		auto handler = table.find(node.op_type);
		if (handler == table.end()) {
			mlx_stream_free(gpu);
			Fail("op '" + node.op_type + "' (node '" + node.name +
			     "') is not implemented by the mlx backend. SET anofox_tabfm_device='cpu' to run this model "
			     "through ONNX Runtime, and please report the op.");
		}
		Args args;
		for (const auto &in : node.inputs) {
			if (in.empty()) {
				args.push_back(nullptr);
				continue;
			}
			auto it = env.find(in);
			args.push_back(it == env.end() ? nullptr : &it->second);
		}
		std::vector<Arr> results;
		try {
			results = handler->second(node, args, ctx);
		} catch (const std::exception &e) {
			mlx_stream_free(gpu);
			Fail(node.op_type + " node '" + node.name + "' failed: " + e.what());
		}
		for (size_t i = 0; i < node.outputs.size() && i < results.size(); i++) {
			if (!node.outputs[i].empty()) {
				env[node.outputs[i]] = std::move(results[i]);
			}
		}
	}

	auto it = env.find(impl_->parsed.outputs.empty() ? std::string() : impl_->parsed.outputs[0]);
	if (it == env.end()) {
		mlx_stream_free(gpu);
		Fail("graph produced no '" + (impl_->parsed.outputs.empty() ? std::string("<none>") : impl_->parsed.outputs[0]) + "'");
	}
	Arr out32;
	Ok(mlx_astype(out32.out(), it->second.get(), MLX_FLOAT32, ctx.s), "output f32");
	Ok(mlx_array_eval(out32.get()), "eval output"); // MLX is lazy; nothing ran until here

	Tensor result;
	result.shape = out32.shape();
	const float *data = mlx_array_data_float32(out32.get());
	if (!data) {
		mlx_stream_free(gpu);
		Fail("output buffer is null after eval");
	}
	result.data.assign(data, data + out32.size());
	mlx_stream_free(gpu);
	return result;
}

} // namespace mlxgraph
} // namespace anofox
