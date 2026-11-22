#include "tabfm_ggml_lower.hpp"

#include "tabfm_rank_reduce.hpp"
#include "tabfm_shape_eval.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <functional>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {
namespace anofox {

namespace {

using Shape = std::vector<int64_t>;
using ::anofox::onnxread::Node;
using Ne4 = std::array<int64_t, 4>;

[[noreturn]] void Fail(const Node &n, const std::string &why) {
	throw std::runtime_error("ggml lowering: node '" + n.name + "' (" + n.op_type + "): " + why);
}

//! ONNX shape -> ggml ne (reversed, padded with 1). Rank above four needs the group map (not built yet): refused.
Ne4 ToNe(const Node &n, const Shape &shape) {
	if (shape.size() > 4) {
		Fail(n, "a tensor of rank " + std::to_string(shape.size()) +
		            " needs the rank reduction of docs/GGML_PLAN.md (P3 group map), which is not implemented yet");
	}
	Ne4 ne = {1, 1, 1, 1};
	for (size_t i = 0; i < shape.size(); i++) {
		if (shape[shape.size() - 1 - i] <= 0) {
			Fail(n, "empty tensors are not supported yet");
		}
		ne[i] = shape[shape.size() - 1 - i];
	}
	return ne;
}

int64_t Numel(const Shape &s) {
	int64_t c = 1;
	for (auto d : s) {
		c *= d;
	}
	return c;
}

//! How a tensor of ANY rank is held in ggml: up to four dimensions as they are; above that the leading dimensions merge into ne3.
//! Tensors are row-major contiguous, so merging adjacent dimensions does not move a single element. Ops regroup the dimensions they
//! need with As() (a free reshape) and give back Unfold()ed results; only the element count and the memory order are the contract.
Ne4 ToStorage(const Node &n, const Shape &shape) {
	if (shape.size() <= 4) {
		return ToNe(n, shape);
	}
	Shape folded(shape.end() - 3, shape.end());
	folded.insert(folded.begin(), Numel(Shape(shape.begin(), shape.end() - 3)));
	return ToNe(n, folded);
}

Ne4 NeOf(const ggml_tensor *t) {
	return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
}

bool SameNe(const ggml_tensor *t, const Ne4 &ne) {
	for (int i = 0; i < 4; i++) {
		if (t->ne[i] != ne[i]) {
			return false;
		}
	}
	return true;
}


//! Elementwise host kernels for the ops ggml has no node for (docs/GGML_LOWERING.md, CUSTOM). Operands broadcast
//! numpy-style (every ne is equal or 1); masks are float 0/1, so a NaN can never hide in one. Single task: the
//! mask chain is memory-bound and tiny next to the matmuls.
enum CustomOp : intptr_t { C_EQ, C_LT, C_LE, C_GT, C_GE, C_WHERE, C_ISNAN, C_ISINF_BOTH, C_ISINF_POS, C_ISINF_NEG,
	               C_ERF, C_ROUND, C_RECIP, C_POW, C_MAX, C_MIN, C_MOD, C_CLIP_BOTH, C_CLIP_MIN, C_CLIP_MAX, C_SIGN, C_RELU, C_NEQ0, C_TRUNC, C_MODFLOOR, C_F16 };

void ElementwiseKernel(ggml_tensor *dst, int ith, int nth, void *userdata) {
	const auto op = static_cast<CustomOp>(reinterpret_cast<intptr_t>(userdata));
	const ggml_tensor *s0 = dst->src[0], *s1 = dst->src[1], *s2 = dst->src[2];
	auto at = [&](const ggml_tensor *t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
		const char *p = static_cast<const char *>(t->data) + (t->ne[0] == 1 ? 0 : i0 * t->nb[0]) +
		                (t->ne[1] == 1 ? 0 : i1 * t->nb[1]) + (t->ne[2] == 1 ? 0 : i2 * t->nb[2]) +
		                (t->ne[3] == 1 ? 0 : i3 * t->nb[3]);
		return *reinterpret_cast<const float *>(p);
	};
	// Rows are independent: thread `ith` of `nth` takes every nth row.
	const int64_t rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
	for (int64_t row_index = ith; row_index < rows; row_index += nth) {
		{
			const int64_t i1 = row_index % dst->ne[1], i2 = (row_index / dst->ne[1]) % dst->ne[2], i3 = row_index / (dst->ne[1] * dst->ne[2]);
			{
				float *row = reinterpret_cast<float *>(static_cast<char *>(dst->data) + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1]);
				for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
					const float a = at(s0, i0, i1, i2, i3);
					float r = 0.0f;
					switch (op) {
					case C_EQ: r = a == at(s1, i0, i1, i2, i3); break;
					case C_LT: r = a < at(s1, i0, i1, i2, i3); break;
					case C_LE: r = a <= at(s1, i0, i1, i2, i3); break;
					case C_GT: r = a > at(s1, i0, i1, i2, i3); break;
					case C_GE: r = a >= at(s1, i0, i1, i2, i3); break;
					case C_WHERE: r = a != 0.0f ? at(s1, i0, i1, i2, i3) : at(s2, i0, i1, i2, i3); break;
					case C_ISNAN: r = std::isnan(a); break;
					case C_ISINF_BOTH: r = std::isinf(a); break;
					case C_ISINF_POS: r = std::isinf(a) && a > 0; break;
					case C_ISINF_NEG: r = std::isinf(a) && a < 0; break;
					case C_SIGN: r = std::isnan(a) ? a : static_cast<float>((a > 0) - (a < 0)); break; // ggml_sgn turns NaN into 0
					case C_TRUNC: r = std::trunc(a) + 0.0f; break; /* +0.0f: an integer has no -0 */ // Cast float -> int: toward zero; the integer lives on as an exact float
					case C_F16: r = RoundToHalf(a); break; // Cast to FLOAT16: rounded to half precision, carried as float32
					case C_NEQ0: r = a != 0.0f; break; // Cast to bool: NaN is true, like ORT
					case C_RELU: r = a < 0.0f ? 0.0f : a; break; // NaN passes (ggml_relu maps it to 0)
					case C_ERF: r = std::erf(a); break;
					case C_ROUND: r = std::nearbyint(a); break; // ONNX Round is half to even
					case C_RECIP: r = 1.0f / a; break;
					case C_POW: r = std::pow(a, at(s1, i0, i1, i2, i3)); break;
					case C_MAX:
					case C_MIN: {
						const float b = at(s1, i0, i1, i2, i3);
						r = (std::isnan(a) || std::isnan(b)) ? std::nanf("") : (op == C_MAX ? std::max(a, b) : std::min(a, b));
						break;
					}
					case C_MODFLOOR: { // integer Mod with fmod=0: the result takes the sign of the divisor
						const float b = at(s1, i0, i1, i2, i3);
						float m = std::fmod(a, b);
						if (m != 0.0f && ((m < 0.0f) != (b < 0.0f))) {
							m += b;
						}
						r = m + 0.0f;
						break;
					}
					case C_MOD: r = std::fmod(a, at(s1, i0, i1, i2, i3)); break;
					case C_CLIP_BOTH:
					case C_CLIP_MIN:
					case C_CLIP_MAX: {
						// NaN passes through (both comparisons are false); a missing bound is +-FLT_MAX like ORT, so it also clamps +-inf
						const float lo = op == C_CLIP_MAX ? -FLT_MAX : at(s1, i0, i1, i2, i3);
						const float hi = op == C_CLIP_MIN ? FLT_MAX : at(op == C_CLIP_MAX ? s1 : s2, i0, i1, i2, i3);
						r = a;
						if (r < lo) {
							r = lo;
						}
						if (r > hi) {
							r = hi; // min(max(x, lo), hi): with lo > hi the result is hi, like ORT
						}
						break;
					}
					}
					row[i0] = r;
				}
			}
		}
	}
}

enum RowOp : intptr_t { R_MAX, R_MIN, R_PROD };

//! Reduce ne0 of a contiguous tensor to one element per row. Max/Min skip NaN unless the first element is one (ORT).
void RowReduceKernel(ggml_tensor *dst, int ith, int nth, void *userdata) {
	const auto op = static_cast<RowOp>(reinterpret_cast<intptr_t>(userdata));
	const ggml_tensor *src = dst->src[0];
	const int64_t rows = ggml_nrows(src), n = src->ne[0];
	const float *in = static_cast<const float *>(src->data);
	float *out = static_cast<float *>(dst->data);
	for (int64_t r = ith; r < rows; r += nth) {
		const float *x = in + r * n;
		float acc = x[0];
		for (int64_t i = 1; i < n; i++) {
			if (op == R_PROD) {
				acc *= x[i];
			} else if (op == R_MAX ? x[i] > acc : x[i] < acc) {
				acc = x[i]; // a NaN is never greater or smaller: only a NaN FIRST element survives (ORT's behaviour)
			}
		}
		out[r] = acc;
	}
}

//! GatherElements over a ne axis (userdata): dst[i] = data[i with its axis coordinate replaced by idx[i]]. Indices are exact floats
//! (data-dependent integers live as floats below 2^24); a negative index wraps, an out-of-range one yields NaN (ORT would throw).
void GatherElementsKernel(ggml_tensor *dst, int ith, int /*nth*/, void *userdata) {
	if (ith != 0) {
		return;
	}
	const int axis = static_cast<int>(reinterpret_cast<intptr_t>(userdata));
	const ggml_tensor *data = dst->src[0], *idx = dst->src[1];
	for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
		for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
			for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
				for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
					int64_t c[4] = {i0, i1, i2, i3};
					const float raw = *reinterpret_cast<const float *>(static_cast<const char *>(idx->data) + i0 * idx->nb[0] + i1 * idx->nb[1] +
					                                                  i2 * idx->nb[2] + i3 * idx->nb[3]);
					int64_t v = static_cast<int64_t>(raw);
					if (v < 0) {
						v += data->ne[axis];
					}
					float result;
					if (std::isnan(raw) || v < 0 || v >= data->ne[axis]) {
						result = std::nanf("");
					} else {
						c[axis] = v;
						result = *reinterpret_cast<const float *>(static_cast<const char *>(data->data) + c[0] * data->nb[0] + c[1] * data->nb[1] +
						                                         c[2] * data->nb[2] + c[3] * data->nb[3]);
					}
					*reinterpret_cast<float *>(static_cast<char *>(dst->data) + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]) = result;
				}
			}
		}
	}
}

//! Gather along a ne axis (userdata) with indices held as exact floats: dst has data's shape with the axis replaced by the (flattened)
//! index count. A negative index wraps; an out-of-range one yields NaN (ORT would throw).
void GatherDynamicKernel(ggml_tensor *dst, int ith, int /*nth*/, void *userdata) {
	if (ith != 0) {
		return;
	}
	const int axis = static_cast<int>(reinterpret_cast<intptr_t>(userdata));
	const ggml_tensor *data = dst->src[0], *idx = dst->src[1];
	const float *indices = static_cast<const float *>(idx->data); // contiguous (the lowering guarantees it)
	for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
		for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
			for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
				for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
					int64_t c[4] = {i0, i1, i2, i3};
					const float raw = indices[c[axis]];
					int64_t v = static_cast<int64_t>(raw);
					if (v < 0) {
						v += data->ne[axis];
					}
					float result;
					if (std::isnan(raw) || v < 0 || v >= data->ne[axis]) {
						result = std::nanf("");
					} else {
						c[axis] = v;
						result = *reinterpret_cast<const float *>(static_cast<const char *>(data->data) + c[0] * data->nb[0] + c[1] * data->nb[1] +
						                                         c[2] * data->nb[2] + c[3] * data->nb[3]);
					}
					*reinterpret_cast<float *>(static_cast<char *>(dst->data) + i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]) = result;
				}
			}
		}
	}
}

//! TopK along ne0 of a contiguous tensor, sorted, ties broken by the lower index first (ONNX). userdata = k << 2 | largest << 1 | want_indices.
//! The values are not NaN-aware: the graphs that use TopK replace NaN before sorting.
void TopKKernel(ggml_tensor *dst, int ith, int nth, void *userdata) {
	const intptr_t packed = reinterpret_cast<intptr_t>(userdata);
	const bool indices = (packed & 1) != 0, largest = (packed & 2) != 0;
	const int64_t k = packed >> 2;
	const ggml_tensor *src = dst->src[0];
	const int64_t n = src->ne[0], rows = ggml_nrows(src);
	std::vector<int64_t> order(n);
	for (int64_t r = ith; r < rows; r += nth) {
		const float *x = static_cast<const float *>(src->data) + r * n;
		for (int64_t i = 0; i < n; i++) {
			order[i] = i;
		}
		std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return largest ? x[a] > x[b] : x[a] < x[b]; });
		float *out = static_cast<float *>(dst->data) + r * k;
		for (int64_t i = 0; i < k; i++) {
			out[i] = indices ? static_cast<float>(order[i]) : x[order[i]];
		}
	}
}

//! A node the ONNX graph computes in double precision. ggml has no usable float64 kernels, and the results matter bit for bit (the graph
//! extracts exponent digits with floor(log(x) / log(10)), which float32 gets wrong at exact powers of ten), so each such node runs the
//! shape pass's element-wise interpreter (the same code, checked against ORT) on its operands at inference time. Intermediate double
//! tensors are real float64 ggml tensors; the node that casts back to float produces a float32 one.
//! The first failure of a custom kernel. A kernel runs on ggml's worker threads inside C frames: an exception thrown there would skip the
//! thread-pool cleanup and leave the other workers spinning, so kernels record the failure here and the runner throws after the compute call.
struct RunError {
	std::mutex mutex;
	bool failed = false;
	std::string message;
	void Set(const std::string &what) {
		std::lock_guard<std::mutex> lock(mutex);
		if (!failed) {
			failed = true;
			message = what;
		}
	}
};

struct DoublePlan {
	RunError *error = nullptr;
	Node node;
	std::vector<int> arg_of;          // per node input: index into dst->src, or -1 (a host constant or an omitted input)
	std::vector<ShapeValue> inputs;   // per node input: shape and dtype (and the values when it is a host constant)
	Shape out_shape;
};

void RunDoubleNode(ggml_tensor *dst, const DoublePlan &plan) {
	std::vector<ShapeValue> materialised(plan.inputs.size());
	std::vector<const ShapeValue *> ptrs(plan.inputs.size(), nullptr);
	for (size_t i = 0; i < plan.inputs.size(); i++) {
		if (plan.node.inputs[i].empty()) {
			continue;
		}
		if (plan.arg_of[i] < 0) {
			ptrs[i] = &plan.inputs[i]; // a host constant
			continue;
		}
		const ggml_tensor *src = dst->src[plan.arg_of[i]];
		ShapeValue &v = materialised[i];
		v.shape = plan.inputs[i].shape;
		v.dtype = plan.inputs[i].dtype;
		v.known = true;
		const int64_t count = ggml_nelements(src);
		const bool is_float = v.dtype == ShapeDtype::F32 || v.dtype == ShapeDtype::F64;
		for (int64_t k = 0; k < count; k++) {
			const double value = src->type == GGML_TYPE_F64 ? static_cast<const double *>(src->data)[k] : static_cast<const float *>(src->data)[k];
			if (is_float) {
				v.floats.push_back(value);
			} else {
				v.ints.push_back(static_cast<int64_t>(std::llround(value)));
			}
		}
		ptrs[i] = &v;
	}
	const auto outs = EvaluateNode(plan.node, ptrs);
	const ShapeValue &r = outs.at(0);
	const int64_t count = ggml_nelements(dst);
	const bool r_float = r.dtype == ShapeDtype::F32 || r.dtype == ShapeDtype::F64;
	for (int64_t k = 0; k < count; k++) {
		const double value = r_float ? r.floats.at(k) : static_cast<double>(r.ints.at(k));
		if (dst->type == GGML_TYPE_F64) {
			static_cast<double *>(dst->data)[k] = value;
		} else {
			static_cast<float *>(dst->data)[k] = static_cast<float>(value);
		}
	}
}

void DoubleNodeKernel(ggml_tensor *dst, int ith, int /*nth*/, void *userdata) {
	if (ith != 0) {
		return;
	}
	const auto &plan = *static_cast<const DoublePlan *>(userdata);
	try {
		RunDoubleNode(dst, plan);
	} catch (const std::exception &e) {
		plan.error->Set("double-precision node '" + plan.node.name + "' (" + plan.node.op_type + "): " + e.what());
		std::memset(dst->data, 0, ggml_nbytes(dst)); // defined contents for the rest of the graph; the runner throws once compute returns
	}
}

//! float64 -> float32 copy: a double-precision tensor that is a graph output (or a debugging probe) is returned as float32.
void F64ToF32Kernel(ggml_tensor *dst, int ith, int /*nth*/, void * /*userdata*/) {
	if (ith != 0) {
		return;
	}
	const ggml_tensor *src = dst->src[0];
	for (int64_t k = 0; k < ggml_nelements(dst); k++) {
		static_cast<float *>(dst->data)[k] = static_cast<float>(static_cast<const double *>(src->data)[k]);
	}
}

//! Structural ops with no ggml node, on flat row-major memory with the LOGICAL shapes in a plan (so the rank does not matter): Pad, CumSum,
//! GatherND, ScatterND, ScatterElements. Operands are contiguous floats; integer indices are exact floats.
enum class FlatKind { PAD, CUMSUM, GATHERND, SCATTERND, SCATTERELEMENTS, SLICE };

struct FlatPlan {
	FlatKind kind = FlatKind::PAD;
	Shape a, b, c, out;          // data/input, indices, updates, result (logical shapes)
	std::vector<int64_t> before; // Pad: elements added before each axis
	std::vector<int64_t> starts, steps; // Slice: first source index and (possibly negative) step per axis
	float value = 0.0f;          // Pad: the constant
	int64_t axis = 0;            // CumSum, ScatterElements (normalised)
	bool exclusive = false, reverse = false;
};

std::vector<int64_t> RowMajorStrides(const Shape &shape) {
	std::vector<int64_t> st(shape.size(), 1);
	for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
		st[i] = st[i + 1] * shape[i + 1];
	}
	return st;
}

void FlatOpKernel(ggml_tensor *dst, int ith, int /*nth*/, void *userdata) {
	if (ith != 0) {
		return;
	}
	const auto &p = *static_cast<const FlatPlan *>(userdata);
	float *out = static_cast<float *>(dst->data);
	const int64_t out_count = ggml_nelements(dst);
	auto arg = [&](int i) { return static_cast<const float *>(dst->src[i]->data); };
	auto wrap = [](float raw, int64_t dim, int64_t &v) { // false when out of range (ORT would throw)
		if (std::isnan(raw)) {
			return false;
		}
		v = static_cast<int64_t>(raw);
		if (v < 0) {
			v += dim;
		}
		return v >= 0 && v < dim;
	};
	switch (p.kind) {
	case FlatKind::PAD: {
		const float *x = arg(0);
		for (int64_t i = 0; i < out_count; i++) {
			out[i] = p.value;
		}
		const auto out_strides = RowMajorStrides(p.out);
		const int64_t rank = static_cast<int64_t>(p.a.size());
		std::vector<int64_t> coord(rank, 0);
		const int64_t count = ggml_nelements(dst->src[0]);
		for (int64_t i = 0; i < count; i++) {
			int64_t off = 0;
			for (int64_t d = 0; d < rank; d++) {
				off += (coord[d] + p.before[d]) * out_strides[d];
			}
			out[off] = x[i];
			for (int64_t d = rank - 1; d >= 0; d--) { // advance the row-major coordinate
				if (++coord[d] < p.a[d]) {
					break;
				}
				coord[d] = 0;
			}
		}
		break;
	}
	case FlatKind::SLICE: {
		const float *x = arg(0);
		const auto in_strides = RowMajorStrides(p.a);
		const int64_t rank = static_cast<int64_t>(p.a.size());
		std::vector<int64_t> coord(rank, 0);
		for (int64_t i = 0; i < out_count; i++) {
			int64_t off = 0;
			for (int64_t d = 0; d < rank; d++) {
				off += (p.starts[d] + coord[d] * p.steps[d]) * in_strides[d];
			}
			out[i] = x[off];
			for (int64_t d = rank - 1; d >= 0; d--) {
				if (++coord[d] < p.out[d]) {
					break;
				}
				coord[d] = 0;
			}
		}
		break;
	}
	case FlatKind::CUMSUM: {
		const float *x = arg(0);
		int64_t outer = 1, inner = 1;
		for (int64_t d = 0; d < p.axis; d++) {
			outer *= p.a[d];
		}
		for (size_t d = p.axis + 1; d < p.a.size(); d++) {
			inner *= p.a[d];
		}
		const int64_t n = p.a[p.axis];
		for (int64_t o = 0; o < outer; o++) {
			for (int64_t in = 0; in < inner; in++) {
				float acc = 0.0f;
				for (int64_t k = 0; k < n; k++) {
					const int64_t at = p.reverse ? n - 1 - k : k;
					const int64_t idx = (o * n + at) * inner + in;
					if (p.exclusive) {
						out[idx] = acc;
						acc += x[idx];
					} else {
						acc += x[idx];
						out[idx] = acc;
					}
				}
			}
		}
		break;
	}
	case FlatKind::GATHERND:
	case FlatKind::SCATTERND: {
		const bool scatter = p.kind == FlatKind::SCATTERND;
		const float *data = arg(0), *idx = arg(1);
		const int64_t m = p.b.back();
		int64_t block = 1, tuples = 1;
		for (size_t d = m; d < p.a.size(); d++) {
			block *= p.a[d];
		}
		for (size_t d = 0; d + 1 < p.b.size(); d++) {
			tuples *= p.b[d];
		}
		const auto strides = RowMajorStrides(p.a);
		if (scatter) {
			std::memcpy(out, data, sizeof(float) * static_cast<size_t>(out_count));
		}
		const float *updates = scatter ? arg(2) : nullptr;
		for (int64_t t = 0; t < tuples; t++) {
			int64_t off = 0;
			bool ok = true;
			for (int64_t j = 0; j < m && ok; j++) {
				int64_t v = 0;
				ok = wrap(idx[t * m + j], p.a[j], v);
				off += v * strides[j];
			}
			for (int64_t k = 0; k < block; k++) {
				if (scatter) {
					if (ok) {
						out[off + k] = updates[t * block + k];
					}
				} else {
					out[t * block + k] = ok ? data[off + k] : std::nanf("");
				}
			}
		}
		break;
	}
	case FlatKind::SCATTERELEMENTS: {
		const float *data = arg(0), *idx = arg(1), *updates = arg(2);
		std::memcpy(out, data, sizeof(float) * static_cast<size_t>(out_count));
		const auto strides = RowMajorStrides(p.a), idx_strides = RowMajorStrides(p.b);
		const int64_t rank = static_cast<int64_t>(p.b.size());
		const int64_t count = ggml_nelements(dst->src[1]);
		for (int64_t i = 0; i < count; i++) {
			int64_t off = 0, rem = i;
			for (int64_t d = 0; d < rank; d++) {
				const int64_t coord = rem / idx_strides[d];
				rem %= idx_strides[d];
				if (d == p.axis) {
					int64_t v = 0;
					if (!wrap(idx[i], p.a[d], v)) {
						off = -1;
						break;
					}
					off += v * strides[d];
				} else {
					off += coord * strides[d];
				}
			}
			if (off >= 0) {
				out[off] = updates[i];
			}
		}
		break;
	}
	}
}

struct Lowering {
	ggml_context *ctx;
	bool strict_fp32 = false; // see GgmlExecutorConfig::strict_fp32
	std::unordered_map<std::string, ShapeValue> shapes;
	std::unordered_map<std::string, ggml_tensor *> tensors;
	std::unordered_map<const ggml_tensor *, ggml_tensor *> transposed_from; // copy -> the tensor whose last two dimensions it swapped
	std::deque<FlatPlan> *flat_plans = nullptr; // set by the runner: storage for plans of strided slices
	//! SplitToSequence results are never materialised: a sequence is the source tensor plus how it is cut, and SequenceAt slices it.
	struct Sequence {
		ggml_tensor *x;
		Shape shape;
		int64_t axis;
		std::vector<int64_t> sizes;
		bool squeeze; // keepdims == 0 and no explicit split: every element drops the axis
	};
	std::unordered_map<std::string, Sequence> sequences;

	ggml_tensor *Contiguous(ggml_tensor *t) { return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t); }

	ggml_tensor *Reshape(const Node &n, ggml_tensor *x, const Shape &shape) {
		const Ne4 ne = ToStorage(n, shape);
		if (ggml_nelements(x) != ne[0] * ne[1] * ne[2] * ne[3]) {
			Fail(n, "internal error: reshaping " + std::to_string(ggml_nelements(x)) + " elements to " + RankShapeToString(shape));
		}
		return ggml_reshape_4d(ctx, Contiguous(x), ne[0], ne[1], ne[2], ne[3]);
	}

	//! `t` regrouped to `shape` (at most four dimensions): a free reshape of a contiguous tensor, a copy of a strided view.
	ggml_tensor *As(const Node &n, ggml_tensor *t, const Shape &shape) {
		if (!t) {
			return t;
		}
		const Ne4 ne = ToNe(n, shape);
		if (SameNe(t, ne)) {
			return t;
		}
		if (ggml_nelements(t) != ne[0] * ne[1] * ne[2] * ne[3]) {
			Fail(n, "internal error: regrouping a tensor of " + std::to_string(ggml_nelements(t)) + " elements to " + RankShapeToString(shape));
		}
		return ggml_reshape_4d(ctx, Contiguous(t), ne[0], ne[1], ne[2], ne[3]);
	}

	//! The result of an op computed on regrouped dimensions, back in the storage form of the ONNX output shape.
	ggml_tensor *Unfold(const Node &n, ggml_tensor *t, const Shape &out_shape) {
		const Ne4 ne = ToStorage(n, out_shape);
		if (SameNe(t, ne)) {
			return t;
		}
		if (ggml_nelements(t) != ne[0] * ne[1] * ne[2] * ne[3]) {
			Fail(n, "internal error: " + std::to_string(ggml_nelements(t)) + " elements for an output of shape " + RankShapeToString(out_shape));
		}
		return ggml_reshape_4d(ctx, Contiguous(t), ne[0], ne[1], ne[2], ne[3]);
	}

	//! Run an elementwise `build` with numpy broadcasting. When no tensor has more than four dimensions that is direct; otherwise the
	//! operands are coalesced (dimensions that every operand treats alike merge) and the result is unfolded. A null operand only
	//! contributes its shape (Expand's target). Refuses when more than four groups remain.
	ggml_tensor *Broadcasted(const Node &n, std::vector<ggml_tensor *> ops, const std::vector<Shape> &shapes, const Shape &out_shape,
	                         const std::function<ggml_tensor *(std::vector<ggml_tensor *> &, const Ne4 &)> &build) {
		size_t rank = out_shape.size();
		for (const auto &sh : shapes) {
			rank = std::max(rank, sh.size());
		}
		if (rank <= 4) {
			return build(ops, ToNe(n, out_shape));
		}
		std::vector<Shape> all = shapes;
		all.push_back(out_shape);
		CoalescedBroadcast co;
		try {
			co = CoalesceBroadcast(all);
		} catch (const std::exception &e) {
			Fail(n, e.what());
		}
		if (Numel(co.out) == 0) {
			Fail(n, "empty tensors are not supported yet");
		}
		if (co.out.size() > 4) {
			// A genuine broadcast over more than four interleaved groups: no single ggml op expresses it. Materialise each operand
			// at the output shape (one 3-group repeat per broadcast axis) and run the op on a flat [inner, outer] layout.
			const int64_t inner = out_shape.empty() ? 1 : out_shape.back();
			const Shape flat = {inner, Numel(out_shape) / std::max<int64_t>(inner, 1)};
			for (size_t k = 0; k < ops.size(); k++) {
				if (ops[k]) {
					ops[k] = As(n, ExpandTo(n, ops[k], shapes[k], out_shape), flat);
				}
			}
			return Unfold(n, build(ops, ToNe(n, flat)), out_shape);
		}
		for (size_t k = 0; k < ops.size(); k++) {
			ops[k] = As(n, ops[k], co.operands[k]);
		}
		return Unfold(n, build(ops, ToNe(n, co.out)), out_shape);
	}

	//! `x` (logical shape `from`, right-aligned numpy broadcasting) repeated to `to`, one axis at a time: each step is a repeat over
	//! at most three groups [before, 1, after], which ggml can do whatever the rank.
	ggml_tensor *ExpandTo(const Node &n, ggml_tensor *x, const Shape &from, const Shape &to) {
		Shape cur(to.size() - from.size(), 1);
		cur.insert(cur.end(), from.begin(), from.end());
		for (size_t i = 0; i < to.size(); i++) {
			if (cur[i] == to[i]) {
				continue;
			}
			if (cur[i] != 1) {
				Fail(n, "cannot broadcast " + RankShapeToString(from) + " to " + RankShapeToString(to));
			}
			const Shape before_shape(cur.begin(), cur.begin() + i), after_shape(cur.begin() + i + 1, cur.end());
			const Shape from3 = {Numel(before_shape), 1, Numel(after_shape)};
			const Shape to3 = {from3[0], to[i], from3[2]};
			x = RepeatTo(As(n, x, from3), ToNe(n, to3));
			cur[i] = to[i];
		}
		return x;
	}

	//! ONNX Transpose of a tensor with more than four dimensions: size-1 dimensions are dropped and the runs of the permutation that
	//! keep neighbouring dimensions together are merged, then the (at most four) groups are permuted.
	ggml_tensor *TransposeBig(const Node &n, ggml_tensor *x, const Shape &in_shape, const Shape &perm, const Shape &out_shape) {
		std::vector<int64_t> renumber(in_shape.size(), -1);
		Shape kept_shape;
		for (size_t i = 0; i < in_shape.size(); i++) {
			if (in_shape[i] != 1) {
				renumber[i] = static_cast<int64_t>(kept_shape.size());
				kept_shape.push_back(in_shape[i]);
			}
		}
		Shape perm2;
		for (auto p : perm) {
			if (renumber[p] >= 0) {
				perm2.push_back(renumber[p]);
			}
		}
		if (perm2.empty()) {
			return Unfold(n, x, out_shape);
		}
		// Runs of the permutation: positions i, i+1 stay together when perm2[i+1] == perm2[i] + 1.
		std::vector<std::pair<int64_t, int64_t>> runs; // [first input dim, last input dim] per output group, in output order
		for (size_t i = 0; i < perm2.size(); i++) {
			if (i > 0 && perm2[i] == perm2[i - 1] + 1) {
				runs.back().second = perm2[i];
			} else {
				runs.push_back({perm2[i], perm2[i]});
			}
		}
		if (runs.size() > 4) {
			Fail(n, "this permutation still has " + std::to_string(runs.size()) + " separate runs after dropping size-1 dimensions");
		}
		// Input grouping: the runs sorted by their first input dimension partition the input dimensions.
		std::vector<size_t> order(runs.size());
		for (size_t i = 0; i < order.size(); i++) {
			order[i] = i;
		}
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return runs[a].first < runs[b].first; });
		Shape group_shape, group_perm(runs.size());
		for (size_t g = 0; g < order.size(); g++) {
			int64_t prod = 1;
			for (int64_t d = runs[order[g]].first; d <= runs[order[g]].second; d++) {
				prod *= kept_shape[d];
			}
			group_shape.push_back(prod);
			group_perm[order[g]] = static_cast<int64_t>(g); // output position -> input group
		}
		// group_perm[j] is the input group of output group j
		return Unfold(n, Permute(As(n, x, group_shape), group_perm), out_shape);
	}

	ggml_tensor *RepeatTo(ggml_tensor *x, const Ne4 &ne) {
		if (SameNe(x, ne)) {
			return x;
		}
		ggml_tensor *like = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		return ggml_repeat(ctx, x, like);
	}

	ggml_tensor *Binary(const Node &n, ggml_tensor *a, ggml_tensor *b, const Ne4 &out) {
		const bool commutative = n.op_type == "Add" || n.op_type == "Mul";
		if (!SameNe(a, out) && commutative && SameNe(b, out)) {
			std::swap(a, b); // ggml broadcasts src1 into src0 only
		}
		a = RepeatTo(a, out); // a is now full size; b is 1 or equal along every axis, which ggml broadcasts
		if (n.op_type == "Add") {
			return ggml_add(ctx, a, b);
		}
		if (n.op_type == "Sub") {
			return ggml_sub(ctx, a, b);
		}
		if (n.op_type == "Mul") {
			return ggml_mul(ctx, a, b);
		}
		return ggml_div(ctx, a, b);
	}

	//! ONNX Transpose: out dim j is input dim perm[j]. In ne terms (reversed) the source axis r-1-perm[j] moves to r-1-j.
	ggml_tensor *Permute(ggml_tensor *x, const Shape &perm) {
		const int64_t r = static_cast<int64_t>(perm.size());
		int axis[4] = {0, 1, 2, 3};
		for (int64_t j = 0; j < r; j++) {
			axis[r - 1 - perm[j]] = static_cast<int>(r - 1 - j);
		}
		ggml_tensor *copy = ggml_cont(ctx, ggml_permute(ctx, x, axis[0], axis[1], axis[2], axis[3]));
		// Remember a swap of the last two dimensions (a weight's Transpose, attention's K^T): MatMul wants the operand in exactly the layout
		// the transpose started from, so it can read the original instead of transposing the copy back.
		bool swaps_last_two = r >= 2;
		for (int64_t j = 0; swaps_last_two && j < r - 2; j++) {
			swaps_last_two = perm[j] == j;
		}
		if (swaps_last_two && perm[r - 2] == r - 1 && perm[r - 1] == r - 2) {
			transposed_from[copy] = x;
		}
		return copy;
	}

	ggml_tensor *MatMul(const Node &n, ggml_tensor *a, ggml_tensor *b, Shape sa, Shape sb, const Shape &out_shape) {
		if (sa.empty() || sb.empty()) {
			Fail(n, "MatMul needs rank >= 1 operands");
		}
		if (sa.size() == 1) {
			sa.insert(sa.begin(), 1);
		}
		if (sb.size() == 1) {
			sb.push_back(1);
		}
		const int64_t k = sa.back();
		if (sb[sb.size() - 2] != k) {
			Fail(n, "inner dimensions differ");
		}
		const size_t rank = std::max(sa.size(), sb.size());
		if (rank > 4) {
			Fail(n, "MatMul batch rank above two needs the rank reduction (not implemented yet)");
		}
		const int64_t m = sa[sa.size() - 2], nn = sb.back();
		Shape batch_a(sa.begin(), sa.end() - 2), batch_b(sb.begin(), sb.end() - 2);
		Ne4 ne_a = {k, m, 1, 1}, ne_b = {nn, k, 1, 1}, ne_out = {nn, m, 1, 1};
		for (size_t i = 0; i < batch_a.size(); i++) {
			ne_a[2 + (batch_a.size() - 1 - i)] = batch_a[i];
		}
		for (size_t i = 0; i < batch_b.size(); i++) {
			ne_b[2 + (batch_b.size() - 1 - i)] = batch_b[i];
		}
		for (int d = 2; d < 4; d++) {
			ne_out[d] = std::max(ne_a[d], ne_b[d]);
			if ((ne_a[d] != ne_out[d] && ne_a[d] != 1) || (ne_b[d] != ne_out[d] && ne_b[d] != 1)) {
				Fail(n, "batch dimensions do not broadcast");
			}
		}
		ggml_tensor *A = ggml_reshape_4d(ctx, Contiguous(a), ne_a[0], ne_a[1], ne_a[2], ne_a[3]);
		ggml_tensor *B = ggml_reshape_4d(ctx, Contiguous(b), ne_b[0], ne_b[1], ne_b[2], ne_b[3]);
		A = RepeatTo(A, {k, m, ne_out[2], ne_out[3]}); // ggml_mul_mat broadcasts its FIRST operand over batches only
		// ggml computes dst = src1 * src0^T over ne0: src0 must hold the weight with ne [k, n]; ONNX B is [n, k].
		ggml_tensor *BT = nullptr;
		const auto origin = transposed_from.find(b);
		if (origin != transposed_from.end() && ggml_is_contiguous(origin->second) && SameNe(origin->second, {k, nn, ne_b[2], ne_b[3]})) {
			BT = origin->second; // B is a Transpose of this very layout: no copy of the transpose, no transpose back
		} else {
			BT = ggml_cont(ctx, ggml_permute(ctx, B, 1, 0, 2, 3));
		}
		ggml_tensor *r = MulMat(BT, A);
		return Unfold(n, r, out_shape);
	}


	//! `weight` x `activations` through ggml's matmul. Strict fp32 (GgmlExecutorConfig::strict_fp32) pads the activations to ggml's cuBLAS threshold of 17 rows
	//! (ggml's CUDA small-batch kernel, at 16 rows or fewer, runs TF32 tensor-core instructions) and slices the extra zero rows off the result.
	ggml_tensor *MulMat(ggml_tensor *weight, ggml_tensor *activations) {
		constexpr int64_t kSmallBatchRows = 16;
		const int64_t rows = activations->ne[1];
		if (!strict_fp32 || rows > kSmallBatchRows) {
			return ggml_mul_mat(ctx, weight, activations);
		}
		ggml_tensor *padded = ggml_pad(ctx, Contiguous(activations), 0, static_cast<int>(kSmallBatchRows + 1 - rows), 0, 0);
		ggml_tensor *r = ggml_mul_mat(ctx, weight, padded);
		return ggml_cont(ctx, ggml_view_4d(ctx, r, r->ne[0], rows, r->ne[2], r->ne[3], r->nb[1], r->nb[2], r->nb[3], 0));
	}

	//! The elementwise and reduction kernels read rows with stride one: a strided view (a Slice with a step) is copied first.
	ggml_tensor *Dim0Contiguous(ggml_tensor *t) { return t->nb[0] == ggml_type_size(t->type) ? t : ggml_cont(ctx, t); }

	//! Integer operand of a node (axes, starts, ...) the shape pass folded; refuses data-dependent ones.
	const std::vector<int64_t> &HostInts(const Node &n, const std::string &name) {
		const auto it = shapes.find(name);
		if (it == shapes.end() || !it->second.known || it->second.dtype != ShapeDtype::I64) {
			Fail(n, "'" + name + "' must be an integer constant the host can fold");
		}
		return it->second.ints;
	}

	//! Swap ne axis `ia` with axis 0 (a permutation that is its own inverse), copying to a contiguous tensor.
	ggml_tensor *SwapWithFirst(ggml_tensor *x, int64_t ia) {
		if (ia == 0) {
			return x;
		}
		int sw[4] = {0, 1, 2, 3};
		sw[0] = static_cast<int>(ia);
		sw[ia] = 0;
		return ggml_cont(ctx, ggml_permute(ctx, x, sw[0], sw[1], sw[2], sw[3]));
	}

	//! One reduction over ne0 of a contiguous [d, rows] tensor, by the op of node `n`.
	ggml_tensor *ReduceRows(const Node &n, ggml_tensor *x) {
		if (n.op_type == "ReduceMean") {
			return ggml_mean(ctx, x);
		}
		if (n.op_type == "ReduceSum") {
			return ggml_sum_rows(ctx, x);
		}
		ggml_tensor *args[1] = {x};
		const RowOp rop = n.op_type == "ReduceMax" ? R_MAX : n.op_type == "ReduceMin" ? R_MIN : R_PROD;
		return ggml_custom_4d(ctx, GGML_TYPE_F32, 1, x->ne[1], x->ne[2], x->ne[3], args, 1, RowReduceKernel, GGML_N_TASKS_MAX,
		                      reinterpret_cast<void *>(static_cast<intptr_t>(rop)));
	}

	ggml_tensor *Reduce(const Node &n, ggml_tensor *x, const Shape &in_shape_arg, const Shape &out_shape) {
		Shape in_shape = in_shape_arg;
		int64_t r = static_cast<int64_t>(in_shape.size());
		std::vector<int64_t> axes;
		if (n.inputs.size() > 1 && !n.inputs[1].empty()) {
			axes = HostInts(n, n.inputs[1]);
		} else if (const auto *a = n.Attr("axes")) {
			axes = a->ints;
		}
		const bool noop = n.Attr("noop_with_empty_axes") && n.Attr("noop_with_empty_axes")->i != 0;
		if (axes.empty() && noop) {
			return x;
		}
		if (axes.empty()) {
			for (int64_t i = 0; i < r; i++) {
				axes.push_back(i);
			}
		}
		// One pass: move the reduced axes (in their original order) behind the kept ones, merge them into ne0 and reduce
		// that single row. ORT visits the reduced elements in flat row-major order, which decides NaN handling of Max/Min.
		std::vector<bool> reduced_axis(r, false);
		for (auto a : axes) {
			if (a < -r || a >= r) {
				Fail(n, "axis out of range");
			}
			reduced_axis[a < 0 ? a + r : a] = true;
		}
		if (r > 4) {
			// Neighbouring dimensions that are both reduced (or both kept) merge: the reduced elements keep their row-major order.
			Shape grouped;
			std::vector<bool> grouped_flag;
			for (int64_t i = 0; i < r; i++) {
				if (!grouped.empty() && grouped_flag.back() == reduced_axis[i]) {
					grouped.back() *= in_shape[i];
				} else {
					grouped.push_back(in_shape[i]);
					grouped_flag.push_back(reduced_axis[i]);
				}
			}
			if (grouped.size() > 4) {
				// Reduced and kept axes interleave into more than four groups: reduce one axis at a time, each over
				// [before, axis, after]. Identical for every element, except that ORT's flat-order NaN rule for Max/Min no longer
				// applies across axes (a NaN-free input gives the same result).
				Shape cur = in_shape;
				for (int64_t i = 0; i < r; i++) {
					if (!reduced_axis[i]) {
						continue;
					}
					const Shape before_shape(cur.begin(), cur.begin() + i), after_shape(cur.begin() + i + 1, cur.end());
					const int64_t before = Numel(before_shape), after = Numel(after_shape), dim = cur[i];
					ggml_tensor *rows = As(n, Permute(As(n, x, {before, dim, after}), {0, 2, 1}), {before * after, dim});
					x = ReduceRows(n, rows);
					cur[i] = 1; // x now holds before * after elements: the same tensor with this axis of size one
				}
				return Unfold(n, x, out_shape);
			}
			x = As(n, x, grouped);
			in_shape = grouped;
			reduced_axis = grouped_flag;
			r = static_cast<int64_t>(grouped.size());
		}
		Shape perm, kept;
		int64_t row = 1;
		for (int64_t i = 0; i < r; i++) {
			if (!reduced_axis[i]) {
				perm.push_back(i);
				kept.push_back(in_shape[i]);
			}
		}
		for (int64_t i = 0; i < r; i++) {
			if (reduced_axis[i]) {
				perm.push_back(i);
				row *= in_shape[i];
			}
		}
		bool identity = true;
		for (int64_t i = 0; i < r; i++) {
			identity = identity && perm[i] == i;
		}
		x = identity ? Contiguous(x) : Permute(x, perm);
		Shape merged = kept;
		merged.push_back(row);
		const Ne4 mne = ToNe(n, merged);
		x = ggml_reshape_4d(ctx, x, mne[0], mne[1], mne[2], mne[3]);
		x = ReduceRows(n, x);
		return Unfold(n, x, out_shape);
	}

	ggml_tensor *LayerNorm(const Node &n, ggml_tensor *x, const Shape &shape, const Shape &out_shape,
	                       std::vector<ggml_tensor *> params) {
		if (n.outputs.size() > 1 && !n.outputs[1].empty()) {
			Fail(n, "the Mean output is not implemented");
		}
		if (n.outputs.size() > 2 && !n.outputs[2].empty()) {
			Fail(n, "the InvStdDev output is not implemented");
		}
		int64_t axis = n.Attr("axis") ? n.Attr("axis")->i : -1;
		const int64_t r = static_cast<int64_t>(shape.size());
		if (axis < 0) {
			axis += r;
		}
		if (axis < 0 || axis >= r) {
			Fail(n, "axis out of range");
		}
		int64_t trailing = 1, lead = 1;
		for (int64_t i = 0; i < r; i++) {
			(i >= axis ? trailing : lead) *= shape[i];
		}
		const float eps = n.Attr("epsilon") ? n.Attr("epsilon")->f : 1e-5f;
		ggml_tensor *flat = ggml_reshape_2d(ctx, Contiguous(x), trailing, lead);
		ggml_tensor *y = ggml_norm(ctx, flat, eps); // ne [trailing, lead]: one normalised row per leading index, any rank
		if (!params.empty() && params[0]) {
			y = ggml_mul(ctx, y, As(n, Dim0Contiguous(params[0]), {trailing}));
		}
		if (params.size() > 1 && params[1]) {
			y = ggml_add(ctx, y, As(n, Dim0Contiguous(params[1]), {trailing}));
		}
		y = Unfold(n, y, out_shape);
		return y;
	}

	//! `axis` is normalised and `r` is the rank of the (possibly regrouped) operands.
	ggml_tensor *Concat(const Node &n, const std::vector<ggml_tensor *> &parts, int64_t r, int64_t axis) {
		if (axis < 0 || axis >= r) {
			Fail(n, "axis out of range");
		}
		ggml_tensor *acc = Contiguous(parts.at(0));
		for (size_t i = 1; i < parts.size(); i++) {
			acc = ggml_concat(ctx, acc, Contiguous(parts[i]), static_cast<int>(r - 1 - axis));
		}
		return acc;
	}

	struct Cut {
		int64_t axis, start, count, step;
	};

	//! ONNX Slice with positive steps: a view (offset and stride only), no copy.
	ggml_tensor *Slice(const Node &n, ggml_tensor *x, const Shape &shape) {
		const int64_t r = static_cast<int64_t>(shape.size());
		const auto &starts = HostInts(n, n.inputs.at(1));
		const auto &ends = HostInts(n, n.inputs.at(2));
		std::vector<int64_t> axes, steps;
		if (n.inputs.size() > 3 && !n.inputs[3].empty()) {
			axes = HostInts(n, n.inputs[3]);
		} else {
			for (int64_t i = 0; i < static_cast<int64_t>(starts.size()); i++) {
				axes.push_back(i);
			}
		}
		if (n.inputs.size() > 4 && !n.inputs[4].empty()) {
			steps = HostInts(n, n.inputs[4]);
		} else {
			steps.assign(starts.size(), 1);
		}
		if (starts.size() != ends.size() || axes.size() != starts.size() || steps.size() != starts.size()) {
			Fail(n, "starts, ends, axes and steps differ in length");
		}
		std::vector<Cut> cuts;
		for (size_t k = 0; k < axes.size(); k++) {
			int64_t a = axes[k] < 0 ? axes[k] + r : axes[k];
			if (a < 0 || a >= r) {
				Fail(n, "axis out of range");
			}
			if (steps[k] == 0) {
				Fail(n, "Slice step 0");
			}
			const int64_t dim = shape[a];
			int64_t s, count;
			if (steps[k] > 0) {
				auto clampv = [&](int64_t v) { return std::min(std::max<int64_t>(v < 0 ? v + dim : v, 0), dim); };
				s = clampv(starts[k]);
				const int64_t e = clampv(ends[k]);
				count = e > s ? (e - s - 1) / steps[k] + 1 : 0; // no (e - s + step) overflow for a huge step
			} else { // negative step: start clamps to [0, dim-1], end to [-1, dim-1]; the walk goes down
				const int64_t st = starts[k] < 0 ? starts[k] + dim : starts[k];
				s = std::min(std::max<int64_t>(st, 0), dim - 1);
				const int64_t en = ends[k] < 0 ? ends[k] + dim : ends[k];
				const int64_t e = std::min(std::max<int64_t>(en, -1), dim - 1);
				const int64_t down = steps[k] == INT64_MIN ? INT64_MAX : -steps[k];
				count = s > e ? (s - e - 1) / down + 1 : 0;
			}
			if (count == 0) {
				Fail(n, "an empty Slice result is not supported yet");
			}
			cuts.push_back({a, s, count, steps[k]});
		}
		return SliceCuts(n, x, shape, cuts);
	}

	//! The cuts of a Slice (axis, start, count, step per cut axis) as one strided view; regroups above four dimensions.
	ggml_tensor *SliceCuts(const Node &n, ggml_tensor *x, const Shape &shape, std::vector<Cut> cuts) {
		const int64_t r = static_cast<int64_t>(shape.size());
		// A view cannot walk backwards or stride dimension 0 (nb0 is the element size): those slices copy through a strided kernel.
		bool needs_kernel = false;
		for (const auto &c : cuts) {
			needs_kernel = needs_kernel || c.step < 0 || (c.axis == r - 1 && c.step != 1);
		}
		if (needs_kernel) {
			if (!flat_plans) {
				Fail(n, "internal error: no plan storage for a strided Slice");
			}
			flat_plans->emplace_back();
			FlatPlan &plan = flat_plans->back();
			plan.kind = FlatKind::SLICE;
			plan.a = shape;
			plan.out = shape;
			plan.starts.assign(r, 0);
			plan.steps.assign(r, 1);
			for (const auto &c : cuts) {
				plan.out[c.axis] = c.count;
				plan.starts[c.axis] = c.start;
				plan.steps[c.axis] = c.step;
			}
			return FlatOp(n, plan, {x});
		}
		// More than four dimensions: neighbours that are not cut merge into one group, every cut axis stays its own.
		int64_t rank = r;
		if (r > 4) {
			std::vector<int64_t> group_of(r);
			std::vector<bool> is_cut(r, false);
			for (const auto &c : cuts) {
				is_cut[c.axis] = !(c.start == 0 && c.count == shape[c.axis] && c.step == 1);
			}
			Shape grouped;
			for (int64_t i = 0; i < r; i++) {
				if (!grouped.empty() && !is_cut[i] && !is_cut[i - 1]) {
					grouped.back() *= shape[i];
				} else {
					grouped.push_back(shape[i]);
				}
				group_of[i] = static_cast<int64_t>(grouped.size()) - 1;
			}
			if (grouped.size() > 4) {
				Fail(n, "the cut axes leave " + std::to_string(grouped.size()) + " groups of dimensions (ggml has four)");
			}
			std::vector<Cut> regrouped;
			for (const auto &c : cuts) {
				if (is_cut[c.axis]) {
					regrouped.push_back({group_of[c.axis], c.start, c.count, c.step});
				}
			}
			cuts = regrouped;
			x = As(n, x, grouped);
			rank = static_cast<int64_t>(grouped.size());
		}
		int64_t ne[4], nb[4];
		for (int i = 0; i < 4; i++) {
			ne[i] = x->ne[i];
			nb[i] = static_cast<int64_t>(x->nb[i]);
		}
		size_t offset = 0;
		for (size_t k = 0; k < cuts.size(); k++) {
			const int64_t s = cuts[k].start, count = cuts[k].count;
			const int64_t ia = rank - 1 - cuts[k].axis;
			offset += static_cast<size_t>(s) * static_cast<size_t>(nb[ia]);
			ne[ia] = count;
			nb[ia] *= cuts[k].step;
		}
		return ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
	}

	//! One custom elementwise node over up to three operands (see ElementwiseKernel); every operand is read in place,
	//! so each must be an argument or the allocator would recycle its buffer.
	ggml_tensor *Custom(CustomOp op, std::vector<ggml_tensor *> args, const Ne4 &out) {
		for (auto &a : args) {
			a = Dim0Contiguous(a);
		}
		ggml_tensor *t = ggml_custom_4d(ctx, GGML_TYPE_F32, out[0], out[1], out[2], out[3], args.data(), static_cast<int>(args.size()),
		                                ElementwiseKernel, GGML_N_TASKS_MAX, reinterpret_cast<void *>(static_cast<intptr_t>(op)));
		ggml_set_name(t, ("elementwise#" + std::to_string(static_cast<int>(op))).c_str());
		return t;
	}

	ggml_tensor *Logic(const Node &n, std::vector<ggml_tensor *> in, const Ne4 &out) {
		const std::string &op = n.op_type;
		if (op == "Not") {
			return ggml_scale_bias(ctx, Contiguous(in[0]), -1.0f, 1.0f);
		}
		ggml_tensor *a = RepeatTo(Dim0Contiguous(in[0]), out), *b = RepeatTo(Dim0Contiguous(in[1]), out);
		ggml_tensor *both = ggml_mul(ctx, a, b);
		return op == "And" ? both : ggml_sub(ctx, ggml_add(ctx, a, b), both);
	}

	ggml_tensor *Gemm(const Node &n, ggml_tensor *a, ggml_tensor *b, ggml_tensor *c, Shape sa, Shape sb, const Shape &out_shape) {
		const float alpha = n.Attr("alpha") ? n.Attr("alpha")->f : 1.0f, beta = n.Attr("beta") ? n.Attr("beta")->f : 1.0f;
		if (sa.size() != 2 || sb.size() != 2) {
			Fail(n, "Gemm operands must be rank 2");
		}
		if (n.Attr("transA") && n.Attr("transA")->i != 0) {
			a = Permute(a, {1, 0});
			std::swap(sa[0], sa[1]);
		}
		if (n.Attr("transB") && n.Attr("transB")->i != 0) {
			b = Permute(b, {1, 0});
			std::swap(sb[0], sb[1]);
		}
		ggml_tensor *y = MatMul(n, a, b, sa, sb, out_shape);
		if (alpha != 1.0f) {
			y = ggml_scale(ctx, y, alpha);
		}
		if (c && beta != 0.0f) { // beta == 0 ignores C entirely (a NaN or inf in C must not leak), like ORT
			ggml_tensor *cc = Dim0Contiguous(c);
			if (beta != 1.0f) {
				cc = ggml_scale(ctx, Contiguous(cc), beta);
			}
			y = ggml_add(ctx, y, cc); // C broadcasts into the [M, N] result (unidirectional)
		}
		return y;
	}

	//! Swap two ne axes (a self-inverse permutation) into a contiguous tensor.
	ggml_tensor *SwapAxes(ggml_tensor *x, int64_t i, int64_t j) {
		if (i == j) {
			return x;
		}
		int sw[4] = {0, 1, 2, 3};
		sw[i] = static_cast<int>(j);
		sw[j] = static_cast<int>(i);
		return ggml_cont(ctx, ggml_permute(ctx, x, sw[0], sw[1], sw[2], sw[3]));
	}

	//! ONNX Gather with constant indices through ggml_get_rows (which gathers along ne1 and wants batched indices).
	ggml_tensor *Gather(const Node &n, ggml_tensor *x, const Shape &shape, const Shape &out_shape, const std::vector<int64_t> &idx,
	                    int64_t axis, const std::function<ggml_tensor *(const std::vector<int32_t> &, int64_t, int64_t)> &index_leaf) {
		const int64_t r = static_cast<int64_t>(shape.size());
		if (axis < 0 || axis >= r) {
			Fail(n, "axis out of range");
		}
		const int64_t dim = shape[axis];
		std::vector<int32_t> flat;
		for (auto v : idx) {
			const int64_t w = v < 0 ? v + dim : v;
			if (w < 0 || w >= dim) {
				Fail(n, "index " + std::to_string(v) + " is out of range for an axis of size " + std::to_string(dim));
			}
			flat.push_back(static_cast<int32_t>(w));
		}
		if (flat.empty()) {
			Fail(n, "an empty index list is not supported yet");
		}
		const int64_t ia = r - 1 - axis;
		ggml_tensor *moved = SwapAxes(Contiguous(x), ia, 1);
		ggml_tensor *gathered = ggml_get_rows(ctx, moved, index_leaf(flat, moved->ne[2], moved->ne[3]));
		ggml_tensor *back = SwapAxes(gathered, ia, 1);
		return Unfold(n, back, out_shape);
	}




	//! One flat-memory structural op (see FlatPlan); `plan` must outlive the compute call.
	ggml_tensor *FlatOp(const Node &n, FlatPlan &plan, const std::vector<ggml_tensor *> &args) {
		const Ne4 ne = ToStorage(n, plan.out);
		std::vector<ggml_tensor *> contiguous;
		for (auto *a : args) {
			contiguous.push_back(Contiguous(a));
		}
		return ggml_custom_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3], contiguous.data(), static_cast<int>(contiguous.size()), FlatOpKernel, 1, &plan);
	}

	//! Lower one double-precision node (see DoublePlan). `plan` must outlive the compute call.
	ggml_tensor *DoubleNode(const Node &n, DoublePlan &plan, const std::vector<ggml_tensor *> &args, ShapeDtype out_dtype) {
		if (n.outputs.size() != 1) {
			Fail(n, "a double-precision node with several outputs is not implemented");
		}
		if (args.size() >= 9) {
			Fail(n, "a double-precision node with " + std::to_string(args.size()) + " tensor operands is not implemented");
		}
		const Ne4 ne = ToStorage(n, plan.out_shape);
		std::vector<ggml_tensor *> contiguous;
		for (auto *a : args) {
			contiguous.push_back(Contiguous(a));
		}
		return ggml_custom_4d(ctx, out_dtype == ShapeDtype::F64 ? GGML_TYPE_F64 : GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3], contiguous.data(),
		                      static_cast<int>(contiguous.size()), DoubleNodeKernel, 1, &plan);
	}

	//! GatherElements with float-held indices: data and indices have the same rank (at most four), the result has the indices' shape.
	ggml_tensor *GatherElements(const Node &n, ggml_tensor *data, ggml_tensor *idx, int64_t r, int64_t axis, const Shape &out_shape) {
		if (axis < 0 || axis >= r) {
			Fail(n, "axis out of range");
		}
		ggml_tensor *args[2] = {Dim0Contiguous(data), Dim0Contiguous(idx)};
		const Ne4 ne = ToNe(n, out_shape);
		return ggml_custom_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3], args, 2, GatherElementsKernel, 1,
		                      reinterpret_cast<void *>(static_cast<intptr_t>(r - 1 - axis)));
	}

	//! Gather with float-held, data-dependent indices (any rank, flattened): data at most four dimensions.
	ggml_tensor *GatherDynamic(const Node &n, ggml_tensor *data, const Shape &shape, ggml_tensor *idx, int64_t axis, const Shape &out_shape) {
		const int64_t r = static_cast<int64_t>(shape.size());
		if (r > 4 || axis < 0 || axis >= r) {
			Fail(n, "Gather with computed indices needs data of at most four dimensions");
		}
		const int64_t ia = r - 1 - axis;
		Ne4 ne = ToNe(n, shape);
		ne[ia] = ggml_nelements(idx);
		ggml_tensor *args[2] = {Dim0Contiguous(data), Contiguous(idx)};
		ggml_tensor *g = ggml_custom_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3], args, 2, GatherDynamicKernel, 1,
		                                reinterpret_cast<void *>(static_cast<intptr_t>(ia)));
		return Reshape(n, g, out_shape);
	}

	//! One TopK output (values or indices): the axis moves to ne0, one kernel sorts each row, the axis moves back.
	ggml_tensor *TopK(const Node &n, ggml_tensor *x, const Shape &shape, int64_t axis, int64_t k, bool largest, bool indices) {
		const int64_t r = static_cast<int64_t>(shape.size());
		if (r > 4) {
			Fail(n, "TopK above four dimensions is not implemented");
		}
		if (axis < 0 || axis >= r || k < 1 || k > shape[axis]) {
			Fail(n, "axis or k out of range");
		}
		const int64_t ia = r - 1 - axis;
		ggml_tensor *moved = SwapWithFirst(Contiguous(x), ia);
		ggml_tensor *args[1] = {moved};
		ggml_tensor *r_ = ggml_custom_4d(ctx, GGML_TYPE_F32, k, moved->ne[1], moved->ne[2], moved->ne[3], args, 1, TopKKernel, GGML_N_TASKS_MAX,
		                                 reinterpret_cast<void *>(static_cast<intptr_t>((k << 2) | (largest ? 2 : 0) | (indices ? 1 : 0))));
		return SwapWithFirst(r_, ia);
	}

	//! `axis` is normalised for `shape` (at most four dimensions).
	ggml_tensor *Softmax(const Node &n, ggml_tensor *x, const Shape &shape, int64_t axis) {
		const int64_t r = static_cast<int64_t>(shape.size());
		if (axis < 0 || axis >= r) {
			Fail(n, "axis out of range");
		}
		const int64_t ia = r - 1 - axis; // ne index of the softmax axis
		x = Contiguous(x);
		if (ia == 0) {
			return ggml_soft_max(ctx, x);
		}
		int swap[4] = {0, 1, 2, 3};
		swap[0] = static_cast<int>(ia);
		swap[ia] = 0;
		ggml_tensor *moved = ggml_cont(ctx, ggml_permute(ctx, x, swap[0], swap[1], swap[2], swap[3]));
		ggml_tensor *s = ggml_soft_max(ctx, moved);
		return ggml_cont(ctx, ggml_permute(ctx, s, swap[0], swap[1], swap[2], swap[3]));
	}
};

//! Folded by the shape pass: the node never reaches ggml, its value is a host constant (and becomes a leaf on demand).
bool IsHost(const ShapeValue &v) {
	return v.known;
}

//! Everything one lowered graph owns between runs. A leaf reads the caller's memory (weights, inputs) or storage in here (folded constants, decoded
//! inline initializers, Gather index tensors); input leaves are re-pointed at each run's buffers. Declaration order is the destruction contract:
//! the context outlives everything that points into it.
struct Compiled {
	std::string signature;
	ggml_context_ptr ctx;
	ggml_cgraph *g = nullptr;
	ggml_backend_buffer_ptr external_marker;
	std::deque<FlatPlan> flat_plans;
	std::deque<DoublePlan> double_plans;
	std::deque<std::vector<int32_t>> index_storage;
	std::vector<std::vector<float>> initializer_floats;
	std::deque<std::vector<float>> folded;
	std::vector<ggml_tensor *> node_outs;
	RunError run_error;
	std::map<std::string, std::pair<ggml_tensor *, size_t>> input_leaves; // graph input -> its leaf and expected element count
	//! Views (reshapes, slices) of an input leaf. ggml stores a view's absolute data pointer when the view is created, so re-pointing the leaf for a
	//! new call leaves these reading the previous call's buffer unless each is refreshed: data = view_src->data + view_offs.
	std::vector<ggml_tensor *> input_views;
	std::vector<std::pair<std::string, ggml_tensor *>> outputs;
	std::map<std::string, Shape> output_shapes;
	HostTensors host_outputs; // outputs the host folded (integers, bool masks, constants): identical on every run of this signature
	ggml_gallocr_ptr alloc;   // the activation arena, kept across runs (the plain CPU path)
	ggml_backend_sched_ptr sched; // the scheduler path (a device, or "cpu-sched"): owns the activation buffers instead
	bool sched_ready = false;     // split and allocated once (see ExecuteCompiled)
	//! Device mode: a leaf is an ordinary tensor the scheduler allocates in device memory (a GPU kernel cannot read a host pointer), uploaded at every run.
	//! `host` is null for a graph input: its data comes from that call's buffers.
	struct Upload {
		ggml_tensor *tensor;
		const void *host;
		std::string input_name;
	};
	bool device_mode = false;
	std::vector<Upload> uploads;
};

//! A persistent, executor-owned copy of an external weight (the device path): returns the tensor to read, creating and uploading it on first use.
using WeightProvider = std::function<ggml_tensor *(const std::string &name, const Ne4 &ne, const float *host, int onnx_type)>;

//! ONNX element types a weight kept outside the ONNX file can be stored as (the device path upcasts 10 and 16 to float32 on upload).
bool IsStoredWeightType(int onnx_type) {
	return onnx_type == 1 || onnx_type == 10 || onnx_type == 16;
}

float HalfToFloat(uint16_t h) {
	const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
	uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, bits;
	if (exp == 0) {
		if (mant == 0) {
			bits = sign; // +-0
		} else { // subnormal half -> normal float
			exp = 127 - 15 + 1;
			while (!(mant & 0x400u)) {
				mant <<= 1;
				exp--;
			}
			bits = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
		}
	} else if (exp == 0x1Fu) {
		bits = sign | 0x7F800000u | (mant << 13); // inf / NaN
	} else {
		bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
	}
	float f;
	std::memcpy(&f, &bits, 4);
	return f;
}

float BFloat16ToFloat(uint16_t b) {
	const uint32_t bits = static_cast<uint32_t>(b) << 16;
	float f;
	std::memcpy(&f, &bits, 4);
	return f;
}

//! Phase timings on stderr when TABFM_GGML_PROFILE is set (a debugging aid).
struct PhaseClock {
	bool enabled = std::getenv("TABFM_GGML_PROFILE") != nullptr;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
	void Lap(const char *what) {
		if (enabled) {
			const auto now = std::chrono::steady_clock::now();
			std::fprintf(stderr, "[ggml] %-22s %8.1f ms\n", what, std::chrono::duration<double, std::milli>(now - last).count());
			last = now;
		}
	}
};

//! What identifies a compiled graph: input shapes, the integer inputs' values (they fold into shapes and constants), which inputs are masks, and which
//! outputs were asked for. Never the float data: the lowering does not look at it.
std::string SignatureOf(const ::anofox::onnxread::Graph &graph, const HostTensors &inputs, const GgmlRunOptions &options) {
	std::string key;
	for (const auto &name : graph.inputs) {
		key += name + "=";
		const auto integer = options.int_inputs.find(name);
		if (integer != options.int_inputs.end()) {
			key += "i[";
			for (auto d : integer->second.shape) {
				key += std::to_string(d) + ",";
			}
			key += "]{";
			for (auto v : integer->second.values) {
				key += std::to_string(v) + ",";
			}
			key += "}";
		} else {
			const auto it = inputs.find(name);
			key += options.bool_inputs.count(name) ? "b[" : "f[";
			if (it != inputs.end()) {
				for (auto d : it->second.shape) {
					key += std::to_string(d) + ",";
				}
			}
			key += "]";
		}
		key += ";";
	}
	key += "out:";
	for (const auto &o : options.outputs) {
		key += o + ",";
	}
	return key;
}

std::unique_ptr<Compiled> CompileGraph(const ::anofox::onnxread::Graph &graph, const HostTensors &inputs, const GgmlRunOptions &options,
                                       const WeightLookup &weights, const WeightProvider &provider) {
	auto compiled = std::make_unique<Compiled>();
	compiled->device_mode = static_cast<bool>(provider);
	compiled->signature = SignatureOf(graph, inputs, options);
	PhaseClock clock;
	auto lap = [&](const char *what) { clock.Lap(what); };
	std::unordered_map<std::string, const ::anofox::onnxread::Tensor *> inits;
	for (const auto &t : graph.initializers) {
		inits[t.name] = &t;
	}
	ShapeEvalInput sin;
	for (const auto &name : graph.inputs) {
		if (inits.count(name)) {
			continue;
		}
		const auto integer = options.int_inputs.find(name);
		if (integer != options.int_inputs.end()) {
			sin.ints[name] = integer->second.values;
			sin.int_shapes[name] = integer->second.shape;
			continue;
		}
		if (options.bool_inputs.count(name)) {
			sin.dtypes[name] = ShapeDtype::BOOL;
		}
		const auto it = inputs.find(name);
		if (it == inputs.end()) {
			throw std::runtime_error("ggml lowering: graph input '" + name + "' was not provided");
		}
		int64_t count = 1;
		for (auto d : it->second.shape) {
			count *= d;
		}
		if (count < 0 || static_cast<size_t>(count) != it->second.data.size()) {
			throw std::runtime_error("ggml lowering: graph input '" + name + "' has " + std::to_string(it->second.data.size()) +
			                         " values for a shape of " + std::to_string(count) + " elements");
		}
		sin.shapes[name] = it->second.shape;
	}
	ShapeEvalResult evaluated = EvaluateShapes(graph, sin);
	lap("shape pass");

	// Size the context from the nodes that reach ggml (the shape pass folds a third of a real graph away). The heaviest lowering
	// (a multi-axis reduction, a transposed Gemm, a Gather) creates about 16 tensors; 24 leaves headroom, and the per-node
	// reserve check below turns any shortfall into an exception, because ggml aborts the process when a context overflows.
	size_t lowered_nodes = 0;
	for (const auto &n : graph.nodes) {
		bool folded = !n.outputs.empty();
		for (const auto &o : n.outputs) {
			const auto it = evaluated.values.find(o);
			folded = folded && it != evaluated.values.end() && it->second.known;
		}
		lowered_nodes += folded ? 0 : 1;
	}
	const size_t budget = lowered_nodes * 24 + (graph.initializers.size() + graph.inputs.size()) * 4 + evaluated.values.size() + 256;
	ggml_init_params params {ggml_tensor_overhead() * budget + ggml_graph_overhead_custom(budget, false), nullptr, true};
	compiled->ctx.reset(ggml_init(params));
	ggml_context_ptr &ctx = compiled->ctx;
	if (!ctx) {
		throw std::runtime_error("ggml lowering: ggml_init failed");
	}
	// Built incrementally while lowering: ggml's graph builder recurses once per dependency level, so expanding only from
	// the outputs at the end would need a stack as deep as the longest chain (a small thread stack would overflow).
	ggml_cgraph *g = ggml_new_graph_custom(ctx.get(), budget, false);
	compiled->g = g;
	lap("context+graph alloc");

	Lowering low;
	low.ctx = ctx.get();
	low.strict_fp32 = options.strict_fp32;
	low.shapes = std::move(evaluated.values);

	// Leaves: graph inputs and float initializers. Their data is copied in after the allocator placed them.
	std::vector<ggml_tensor *> &node_outs = compiled->node_outs; // result of every lowered node, in lowering order
	RunError &run_error = compiled->run_error;                    // first failure of a custom kernel (see RunError)
	std::deque<FlatPlan> &flat_plans = compiled->flat_plans;      // Pad, CumSum, GatherND, ScatterND, ScatterElements, strided Slice (their kernels read these during compute)
	low.flat_plans = &flat_plans;
	const std::vector<std::string> &wanted_outputs = options.outputs.empty() ? graph.outputs : options.outputs;
	std::set<std::string> needed(wanted_outputs.begin(), wanted_outputs.end()); // names the requested outputs depend on (transitively)
	for (size_t i = graph.nodes.size(); i-- > 0;) {
		const auto &nd = graph.nodes[i];
		bool used = false;
		for (const auto &o : nd.outputs) {
			used = used || (!o.empty() && needed.count(o) != 0);
		}
		if (used) {
			for (const auto &in_name : nd.inputs) {
				if (!in_name.empty()) {
					needed.insert(in_name);
				}
			}
		}
	}
	std::deque<DoublePlan> &double_plans = compiled->double_plans;            // the double-precision nodes (their kernels read these during compute)
	std::deque<std::vector<int32_t>> &index_storage = compiled->index_storage; // I32 index tensors (Gather) the graph reads
	std::vector<std::vector<float>> &initializer_floats = compiled->initializer_floats; // decoded copies (the reader keeps raw bytes)
	// A leaf's data is the caller's memory (weights, inputs, constants): ggml sees a host tensor whose `buffer` is one shared marker buffer.
	// ggml_backend_cpu_buffer_from_ptr would demand 32-byte-aligned pointers, which a safetensors offset is not; the CPU kernels do not need it.
	compiled->external_marker.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 64));
	ggml_backend_buffer_ptr &external_marker = compiled->external_marker;
	auto bind_external = [&](ggml_tensor *t, void *data, const std::string &input_name = std::string()) {
		if (compiled->device_mode) {
			ggml_set_input(t); // allocated by the scheduler on the device; uploaded before each run
			compiled->uploads.push_back({t, data, input_name});
			return;
		}
		t->data = data;
		t->buffer = external_marker.get();
	};
	auto leaf = [&](const std::string &name, const Shape &shape, const float *data, bool is_graph_input = false) {
		Node pseudo;
		pseudo.name = name;
		pseudo.op_type = "<input>";
		const Ne4 ne = ToStorage(pseudo, shape);
		ggml_tensor *t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		ggml_set_name(t, name.c_str());
		// The leaf reads the caller's memory in place (weights, inputs, folded constants all outlive the call): nothing is copied into the
		// arena. gallocr leaves a tensor that already has data alone and never reuses its memory for another result.
		bind_external(t, const_cast<float *>(data), is_graph_input ? name : std::string());
		low.tensors[name] = t;
	};
	for (const auto &name : graph.inputs) {
		if (!inits.count(name) && !options.int_inputs.count(name)) {
			const auto found_input = inputs.find(name);
			if (found_input == inputs.end()) {
				throw std::runtime_error("ggml lowering: graph input '" + name + "' was not supplied");
			}
			const auto &in = found_input->second;
			if (in.data.empty()) {
				continue; // an empty input holds nothing; only ScatterND/ScatterElements/Concat accept one (handled where they are lowered)
			}
			leaf(name, in.shape, in.data.data(), true);
			compiled->input_leaves[name] = {low.tensors.at(name), in.data.size()};
		}
	}
	// Float initializers become leaves on first use (an unused one, whatever its rank, costs nothing).
	auto materialize_initializer = [&](const ::anofox::onnxread::Tensor &t) {
		size_t count = 1;
		for (auto d : t.dims) {
			count *= static_cast<size_t>(d);
		}
		const bool inline_data = !t.raw.empty() || !t.floats.empty();
		if (!inline_data && count > 0) {
			// Weights kept outside the ONNX file: the leaf points straight at the caller's buffer (nothing is copied here).
			const ExternalWeight external = weights ? weights(t.name, t.dims) : ExternalWeight();
			if (!external.data) {
				throw std::runtime_error("ggml lowering: initializer '" + t.name + "' has external or missing data and no weight was supplied");
			}
			if (external.count != count) {
				throw std::runtime_error("ggml lowering: weight '" + t.name + "' has " + std::to_string(external.count) + " values for a shape of " +
				                         std::to_string(count));
			}
			if (external.onnx_type != t.data_type) {
				throw std::runtime_error("ggml lowering: weight '" + t.name + "' is declared as ONNX type " + std::to_string(t.data_type) + " but was supplied as type " +
				                         std::to_string(external.onnx_type));
			}
			if (provider) {
				Node pseudo;
				pseudo.name = t.name;
				pseudo.op_type = "<weight>";
				low.tensors[t.name] = provider(t.name, ToStorage(pseudo, Shape(t.dims.begin(), t.dims.end())), external.data, external.onnx_type);
				return;
			}
			if (external.onnx_type != 1) {
				throw std::runtime_error("ggml lowering: weight '" + t.name + "' is stored as ONNX type " + std::to_string(external.onnx_type) +
				                         " (bf16/fp16); only a device executor upcasts those on upload (the CPU session upcasts while injecting)");
			}
			leaf(t.name, Shape(t.dims.begin(), t.dims.end()), external.data);
			return;
		}
		std::vector<float> values(count);
		if (!t.raw.empty()) {
			if (t.raw.size() < count * 4) {
				throw std::runtime_error("ggml lowering: initializer '" + t.name + "' is shorter than its shape");
			}
			std::memcpy(values.data(), t.raw.data(), count * 4);
		} else if (t.floats.size() >= count) {
			std::copy(t.floats.begin(), t.floats.begin() + count, values.begin());
		} else {
			throw std::runtime_error("ggml lowering: initializer '" + t.name + "' is shorter than its shape");
		}
		initializer_floats.push_back(std::move(values));
		leaf(t.name, Shape(t.dims.begin(), t.dims.end()), initializer_floats.back().data());
	};

	auto shape_of = [&](const Node &n, const std::string &name) -> const Shape & {
		const auto it = low.shapes.find(name);
		if (it == low.shapes.end()) {
			Fail(n, "no shape for '" + name + "'");
		}
		return it->second.shape;
	};
	std::deque<std::vector<float>> &folded = compiled->folded; // values of folded constants that became leaves (stable addresses)
	auto tensor_of = [&](const Node &n, const std::string &name) -> ggml_tensor * {
		const auto it = low.tensors.find(name);
		if (it != low.tensors.end()) {
			return it->second;
		}
		const auto init = inits.find(name);
		if (init != inits.end() && IsStoredWeightType(init->second->data_type)) {
			materialize_initializer(*init->second);
			return low.tensors.at(name);
		}
		const auto sv = low.shapes.find(name);
		if (sv != low.shapes.end() && sv->second.known) {
			int64_t count = 1;
			for (auto d : sv->second.shape) {
				count *= d;
			}
			std::vector<float> values;
			if (sv->second.dtype == ShapeDtype::F32 && sv->second.floats.size() == static_cast<size_t>(count)) {
				values.assign(sv->second.floats.begin(), sv->second.floats.end());
			} else if (sv->second.dtype != ShapeDtype::F32 && sv->second.ints.size() == static_cast<size_t>(count)) {
				// a host-known bool mask or integer constant used by a float op (Where cond, Pow exponent): exact as a
				// float only up to 2^24, anything larger is refused rather than rounded
				for (auto v : sv->second.ints) {
					if (v > (1 << 24) || v < -(1 << 24)) {
						Fail(n, "integer operand '" + name + "' is not exactly representable as a float");
					}
					values.push_back(static_cast<float>(v));
				}
			} else {
				Fail(n, "operand '" + name + "' has no usable host value");
			}
			folded.emplace_back(std::move(values));
			leaf(name, sv->second.shape, folded.back().data());
			return low.tensors.at(name);
		}
		Fail(n, "operand '" + name + "' is not a float tensor ggml holds");
	};

	for (const auto &n : graph.nodes) {
		bool needed_here = false;
		for (const auto &o : n.outputs) {
			needed_here = needed_here || (!o.empty() && needed.count(o) != 0);
		}
		if (!needed_here) {
			continue; // nothing that was asked for depends on this node
		}
		bool all_host = !n.outputs.empty();
		for (const auto &o : n.outputs) {
			const auto it = low.shapes.find(o);
			all_host = all_host && it != low.shapes.end() && IsHost(it->second);
		}
		if (all_host) {
			continue; // folded on the host by the shape pass
		}
		// The heaviest lowerings (the >4-group broadcast fallback, a sequential reduction) create a few tensors per dimension per operand.
		size_t max_rank = 1, live_operands = 0; // operands that are ggml tensors (host constants create none)
		for (const auto &name : n.inputs) {
			const auto it = name.empty() ? low.shapes.end() : low.shapes.find(name);
			if (it != low.shapes.end()) {
				max_rank = std::max(max_rank, it->second.shape.size());
				live_operands += it->second.known ? 0 : 1;
			}
		}
		if (!n.outputs.empty()) {
			const auto out_it = low.shapes.find(n.outputs[0]);
			max_rank = std::max(max_rank, out_it != low.shapes.end() ? out_it->second.shape.size() : size_t(1));
		}
		const size_t reserve = std::max<size_t>(64, 12 * max_rank * (live_operands + 1));
		if (ggml_get_mem_size(ctx.get()) - ggml_used_mem(ctx.get()) < reserve * ggml_tensor_overhead() ||
		    static_cast<size_t>(ggml_graph_n_nodes(g)) + reserve >= static_cast<size_t>(ggml_graph_size(g))) {
			Fail(n, "the graph is larger than the ggml context budget (" + std::to_string(budget) + " tensors)");
		}
		const std::string &op = n.op_type;
		if (op == "SplitToSequence") {
			const Shape &xs = shape_of(n, n.inputs.at(0));
			Lowering::Sequence seq;
			seq.x = tensor_of(n, n.inputs.at(0));
			seq.shape = xs;
			int64_t axis = n.Attr("axis") ? n.Attr("axis")->i : 0;
			axis += axis < 0 ? static_cast<int64_t>(xs.size()) : 0;
			if (axis < 0 || axis >= static_cast<int64_t>(xs.size())) {
				Fail(n, "axis out of range");
			}
			seq.axis = axis;
			const int64_t dim = xs[axis];
			seq.squeeze = false;
			if (n.inputs.size() > 1 && !n.inputs[1].empty()) {
				const auto &split = low.HostInts(n, n.inputs[1]);
				if (split.size() == 1 && split[0] <= 0) {
					Fail(n, "the split size must be positive");
				}
				if (split.size() == 1 && low.shapes.at(n.inputs[1]).shape.empty()) { // scalar: chunks of that size, the last may be shorter
					for (int64_t at = 0; at < dim; at += split[0]) {
						seq.sizes.push_back(std::min(split[0], dim - at));
					}
				} else {
					seq.sizes = split;
				}
			} else {
				seq.sizes.assign(dim, 1);
				seq.squeeze = n.Attr("keepdims") && n.Attr("keepdims")->i == 0;
			}
			int64_t total = 0;
			for (auto sz : seq.sizes) {
				total += sz;
			}
			if (total != dim) {
				Fail(n, "the split sizes do not add up to the axis length");
			}
			low.sequences[n.outputs.at(0)] = seq;
			continue;
		}
		const Shape &out_shape = shape_of(n, n.outputs.at(0));
		if (Numel(out_shape) == 0) {
			continue; // an empty tensor holds nothing; the only consumer that accepts one is Concat (below), which drops it
		}
		ggml_tensor *out = nullptr;
		auto in = [&](size_t i) { return tensor_of(n, n.inputs.at(i)); };
		auto sh = [&](size_t i) -> const Shape & { return shape_of(n, n.inputs.at(i)); };
		// An elementwise op over the first `count` inputs, numpy-broadcast; regrouped to at most four dimensions when needed.
		auto elementwise = [&](size_t count, const std::function<ggml_tensor *(std::vector<ggml_tensor *> &, const Ne4 &)> &build) {
			std::vector<ggml_tensor *> ops;
			std::vector<Shape> shapes;
			for (size_t i = 0; i < count; i++) {
				ops.push_back(in(i));
				shapes.push_back(sh(i));
			}
			return low.Broadcasted(n, ops, shapes, out_shape, build);
		};
		auto normalised_axis = [&](int64_t axis, size_t rank) {
			if (axis < 0) {
				axis += static_cast<int64_t>(rank);
			}
			if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
				Fail(n, "axis out of range");
			}
			return axis;
		};
		//! [before, axis, after]: any rank becomes three dimensions around one axis.
		auto around_axis = [&](const Shape &shape, int64_t axis) {
			Shape before(shape.begin(), shape.begin() + axis), after(shape.begin() + axis + 1, shape.end());
			return Shape {Numel(before), shape[axis], Numel(after)};
		};
		bool involves_double = low.shapes.at(n.outputs.at(0)).dtype == ShapeDtype::F64;
		for (const auto &name : n.inputs) {
			const auto it = name.empty() ? low.shapes.end() : low.shapes.find(name);
			involves_double = involves_double || (it != low.shapes.end() && it->second.dtype == ShapeDtype::F64);
		}
		if (involves_double) {
			// Only ops the shape pass's interpreter evaluates with real values can run on a double tensor; anything else would hand the
			// kernel an unknown value at inference time, so it is refused here by name.
			static const std::set<std::string> kDoubleOps = {
			    "Cast", "CastLike", "Identity", "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Expand", "Concat", "Slice", "Gather",
			    "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Max", "Min", "Neg", "Abs", "Sqrt", "Reciprocal", "Exp", "Log", "Tanh", "Erf",
			    "Sigmoid", "Ceil", "Floor", "Sin", "Cos", "Round", "Relu", "Sign", "Softplus", "Where", "Clip", "Equal", "Less",
			    "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Not", "IsNaN", "IsInf"};
			if (kDoubleOps.count(op) == 0) {
				Fail(n, "op " + op + " on a double-precision tensor is not implemented");
			}
			double_plans.emplace_back();
			DoublePlan &plan = double_plans.back();
			plan.error = &run_error;
			plan.node = n;
			plan.out_shape = out_shape;
			std::vector<ggml_tensor *> args;
			for (const auto &name : n.inputs) {
				plan.arg_of.push_back(-1);
				plan.inputs.emplace_back();
				if (name.empty()) {
					continue;
				}
				const ShapeValue &sv = low.shapes.at(name);
				if (sv.known) {
					plan.inputs.back() = sv; // a host constant carries its values into the plan
				} else {
					plan.inputs.back().shape = sv.shape;
					plan.inputs.back().dtype = sv.dtype;
					plan.arg_of.back() = static_cast<int>(args.size());
					args.push_back(tensor_of(n, name));
				}
			}
			out = low.DoubleNode(n, plan, args, low.shapes.at(n.outputs.at(0)).dtype);
		} else if (op == "Identity") {
			out = in(0);
		} else if (op == "Reshape" || op == "Squeeze" || op == "Unsqueeze") {
			out = low.Reshape(n, in(0), out_shape);
		} else if (op == "Neg" || op == "Abs" || op == "Sqrt" || op == "Exp" || op == "Log" || op == "Tanh") {
			ggml_tensor *x = low.Contiguous(in(0));
			out = op == "Neg"    ? ggml_neg(ctx.get(), x)
			      : op == "Abs"  ? ggml_abs(ctx.get(), x)
			      : op == "Sqrt" ? ggml_sqrt(ctx.get(), x)
			      : op == "Exp"  ? ggml_exp(ctx.get(), x)
			      : op == "Log"  ? ggml_log(ctx.get(), x)
			                     : ggml_tanh(ctx.get(), x);
		} else if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div") {
			// Integers computed from float data live as floats; ONNX integer Div truncates toward zero.
			const bool integer_div = op == "Div" && low.shapes.at(n.outputs.at(0)).dtype == ShapeDtype::I64;
			out = elementwise(2, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) {
				ggml_tensor *r = low.Binary(n, low.Dim0Contiguous(o[0]), low.Dim0Contiguous(o[1]), ne);
				return integer_div ? low.Custom(C_TRUNC, {r}, NeOf(r)) : r;
			});
		} else if (op == "Transpose") {
			Shape perm;
			if (const auto *a = n.Attr("perm")) {
				perm = a->ints;
			} else {
				for (size_t i = sh(0).size(); i-- > 0;) {
					perm.push_back(static_cast<int64_t>(i));
				}
			}
			{
				std::vector<bool> seen(perm.size(), false);
				bool valid = perm.size() == sh(0).size();
				for (auto p : perm) {
					valid = valid && p >= 0 && p < static_cast<int64_t>(perm.size()) && !seen[p];
					if (valid) {
						seen[p] = true;
					}
				}
				if (!valid) {
					Fail(n, "perm is not a permutation of the input axes");
				}
			}
			out = sh(0).size() > 4 ? low.TransposeBig(n, in(0), sh(0), perm, out_shape) : low.Permute(in(0), perm);
		} else if (op == "MatMul") {
			const Shape &sa = sh(0), &sb = sh(1);
			if (sa.size() > 4 || sb.size() > 4 || out_shape.size() > 4) {
				if (sa.size() < 2 || sb.size() < 2) {
					Fail(n, "MatMul with a vector operand and rank above four is not implemented");
				}
				// The batch dimensions coalesce like a broadcast; they must fit ggml's two batch dimensions.
				Shape ba(sa.begin(), sa.end() - 2), bb(sb.begin(), sb.end() - 2);
				CoalescedBroadcast co;
				try {
					co = CoalesceBroadcast({ba, bb});
				} catch (const std::exception &e) {
					Fail(n, e.what());
				}
				if (co.out.size() > 2) {
					Fail(n, "the batch dimensions still need " + std::to_string(co.out.size()) + " groups after merging (ggml has two)");
				}
				Shape sa2 = co.operands[0], sb2 = co.operands[1], out2 = co.out;
				sa2.push_back(sa[sa.size() - 2]);
				sa2.push_back(sa.back());
				sb2.push_back(sb[sb.size() - 2]);
				sb2.push_back(sb.back());
				out2.push_back(sa[sa.size() - 2]);
				out2.push_back(sb.back());
				out = low.Unfold(n, low.MatMul(n, low.As(n, in(0), sa2), low.As(n, in(1), sb2), sa2, sb2, out2), out_shape);
			} else {
				out = low.MatMul(n, in(0), in(1), sa, sb, out_shape);
			}
		} else if (op == "Softmax") {
			const int64_t axis = normalised_axis(n.Attr("axis") ? n.Attr("axis")->i : -1, sh(0).size());
			if (sh(0).size() > 4) {
				const Shape g3 = around_axis(sh(0), axis);
				out = low.Unfold(n, low.Softmax(n, low.As(n, in(0), g3), g3, 1), out_shape);
			} else {
				out = low.Softmax(n, in(0), sh(0), axis);
			}
		} else if (op == "ReduceMean" || op == "ReduceSum" || op == "ReduceMax" || op == "ReduceMin" || op == "ReduceProd") {
			out = low.Reduce(n, in(0), sh(0), out_shape);
			if (op == "ReduceMean" && low.shapes.at(n.outputs.at(0)).dtype == ShapeDtype::I64) {
				out = low.Custom(C_TRUNC, {out}, NeOf(out)); // an integer mean is a truncating integer division
			}
		} else if (op == "LayerNormalization") {
			std::vector<ggml_tensor *> params;
			for (size_t i = 1; i < n.inputs.size(); i++) {
				params.push_back(n.inputs[i].empty() ? nullptr : in(i));
			}
			out = low.LayerNorm(n, in(0), sh(0), out_shape, params);
		} else if (op == "Expand") {
			out = low.Broadcasted(n, {in(0), nullptr}, {sh(0), out_shape}, out_shape, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) {
				return low.RepeatTo(low.Dim0Contiguous(o[0]), ne);
			});
		} else if (op == "Concat") {
			const size_t rank = out_shape.size();
			const int64_t axis = normalised_axis(n.Attr("axis") ? n.Attr("axis")->i : 0, rank);
			std::vector<ggml_tensor *> parts;
			std::vector<size_t> live; // inputs with elements: an empty operand adds nothing to a Concat
			for (size_t i = 0; i < n.inputs.size(); i++) {
				if (Numel(sh(i)) != 0) {
					live.push_back(i);
				}
			}
			if (live.empty()) {
				Fail(n, "Concat of only empty tensors");
			}
			if (rank > 4) {
				for (auto i : live) {
					parts.push_back(low.As(n, in(i), around_axis(sh(i), axis)));
				}
				out = low.Unfold(n, low.Concat(n, parts, 3, 1), out_shape);
			} else if (live.size() == 1) {
				out = in(live[0]);
			} else {
				for (auto i : live) {
					parts.push_back(in(i));
				}
				out = low.Concat(n, parts, static_cast<int64_t>(rank), axis);
			}
		} else if (op == "Slice") {
			out = low.Slice(n, in(0), sh(0));
		} else if (op == "And" || op == "Or") {
			out = elementwise(2, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) { return low.Logic(n, o, ne); });
		} else if (op == "Not") {
			out = low.Logic(n, {in(0)}, NeOf(in(0)));
		} else if (op == "Equal" || op == "Less" || op == "LessOrEqual" || op == "Greater" || op == "GreaterOrEqual") {
			const CustomOp code = op == "Equal" ? C_EQ : op == "Less" ? C_LT : op == "LessOrEqual" ? C_LE : op == "Greater" ? C_GT : C_GE;
			out = elementwise(2, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) { return low.Custom(code, o, ne); });
		} else if (op == "IsNaN" || op == "IsInf") {
			CustomOp code = C_ISNAN;
			if (op == "IsInf") {
				const bool pos = !n.Attr("detect_positive") || n.Attr("detect_positive")->i != 0;
				const bool neg = !n.Attr("detect_negative") || n.Attr("detect_negative")->i != 0;
				code = pos && neg ? C_ISINF_BOTH : pos ? C_ISINF_POS : neg ? C_ISINF_NEG : C_ISNAN;
				if (!pos && !neg) {
					Fail(n, "IsInf with both detect flags off is not implemented");
				}
			}
			out = low.Custom(code, {in(0)}, NeOf(in(0)));
		} else if (op == "Ceil" || op == "Floor" || op == "Sigmoid" || op == "Sin" || op == "Cos" || op == "Softplus") {
			ggml_tensor *x = low.Contiguous(in(0));
			out = op == "Ceil"       ? ggml_ceil(ctx.get(), x)
			      : op == "Floor"    ? ggml_floor(ctx.get(), x)
			      : op == "Sigmoid"  ? ggml_sigmoid(ctx.get(), x)
			      : op == "Sin"      ? ggml_sin(ctx.get(), x)
			      : op == "Cos"      ? ggml_cos(ctx.get(), x)
			                         : ggml_softplus(ctx.get(), x);
		} else if (op == "Round" || op == "Reciprocal" || op == "Erf" || op == "Sign" || op == "Relu") {
			out = low.Custom(op == "Relu" ? C_RELU : op == "Sign" ? C_SIGN : op == "Round" ? C_ROUND : op == "Reciprocal" ? C_RECIP : C_ERF,
			                 {in(0)}, NeOf(in(0)));
		} else if (op == "Pow" || op == "Mod") {
			const bool fmod = n.Attr("fmod") && n.Attr("fmod")->i != 0;
			bool integer_operands = true; // integers computed from float data live as exact floats; the shape pass knows their type
			for (size_t i = 0; i < 2; i++) {
				integer_operands = integer_operands && low.shapes.at(n.inputs.at(i)).dtype == ShapeDtype::I64;
			}
			if (op == "Mod" && !fmod && !integer_operands) {
				Fail(n, "Mod on floats needs fmod=1; fmod=0 is only defined for integers");
			}
			const CustomOp code = op == "Pow" ? C_POW : (fmod ? C_MOD : C_MODFLOOR);
			// Pow(x, 2) is the squared deviation of every normalisation in the graphs: a multiply, not a libm call per element.
			double constant_exponent = std::nan("");
			if (op == "Pow") {
				const auto ev = low.shapes.find(n.inputs.at(1));
				if (ev != low.shapes.end() && ev->second.known && ev->second.dtype == ShapeDtype::F32 && ev->second.floats.size() == 1) {
					constant_exponent = ev->second.floats[0];
				}
			}
			const auto base_dtype = low.shapes.at(n.inputs.at(0)).dtype;
			if (base_dtype == ShapeDtype::F32 && (constant_exponent == 2.0 || constant_exponent == 3.0) && sh(0) == out_shape) {
				ggml_tensor *x = low.Contiguous(in(0));
				out = constant_exponent == 2.0 ? ggml_sqr(ctx.get(), x) : ggml_mul(ctx.get(), ggml_sqr(ctx.get(), x), x);
			} else {
				out = elementwise(2, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) { return low.Custom(code, o, ne); });
			}
		} else if (op == "Max" || op == "Min") {
			// variadic: fold pairwise; each pairwise result has the broadcast shape of the operands so far
			ggml_tensor *acc = in(0);
			Shape acc_shape = sh(0);
			const CustomOp code = op == "Max" ? C_MAX : C_MIN;
			for (size_t i = 1; i < n.inputs.size(); i++) {
				const Shape &next = sh(i);
				Shape merged(std::max(acc_shape.size(), next.size()), 1);
				for (size_t d = 0; d < merged.size(); d++) {
					const int64_t x = d < acc_shape.size() ? acc_shape[acc_shape.size() - 1 - d] : 1;
					const int64_t y = d < next.size() ? next[next.size() - 1 - d] : 1;
					merged[merged.size() - 1 - d] = std::max(x, y);
				}
				acc = low.Broadcasted(n, {acc, in(i)}, {acc_shape, next}, merged,
				                      [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) { return low.Custom(code, o, ne); });
				acc_shape = merged;
			}
			out = acc;
		} else if (op == "Clip") {
			const bool has_lo = n.inputs.size() > 1 && !n.inputs[1].empty(), has_hi = n.inputs.size() > 2 && !n.inputs[2].empty();
			std::vector<ggml_tensor *> args {in(0)};
			if (has_lo) {
				args.push_back(in(1));
			}
			if (has_hi) {
				args.push_back(in(2));
			}
			if (!has_lo && !has_hi) {
				out = args[0];
			} else {
				out = low.Custom(has_lo && has_hi ? C_CLIP_BOTH : has_lo ? C_CLIP_MIN : C_CLIP_MAX, args, NeOf(args[0]));
			}
		} else if (op == "Gather") {
			const auto make_index_leaf = [&](const std::vector<int32_t> &flat, int64_t ne2, int64_t ne3) {
				ggml_tensor *t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_I32, static_cast<int64_t>(flat.size()), ne2, ne3, 1);
				std::vector<int32_t> batched; // the same rows for every batch slice
				for (int64_t b = 0; b < ne2 * ne3; b++) {
					batched.insert(batched.end(), flat.begin(), flat.end());
				}
				index_storage.push_back(std::move(batched));
				bind_external(t, index_storage.back().data()); // read in place, like every other leaf
				return t;
			};
			const int64_t axis = normalised_axis(n.Attr("axis") ? n.Attr("axis")->i : 0, sh(0).size());
			const auto idx_value = low.shapes.find(n.inputs.at(1));
			const bool idx_known = idx_value != low.shapes.end() && idx_value->second.known && idx_value->second.dtype == ShapeDtype::I64;
			if (!idx_known) {
				// indices computed from float data: held as exact floats, gathered by a host kernel
				out = low.GatherDynamic(n, in(0), sh(0), in(1), axis, out_shape);
			} else if (sh(0).size() > 4) {
				const auto &idx = idx_value->second.ints;
				const Shape g3 = around_axis(sh(0), axis);
				const Shape out3 = {g3[0], static_cast<int64_t>(idx.size()), g3[2]};
				out = low.Unfold(n, low.Gather(n, low.As(n, in(0), g3), g3, out3, idx, 1, make_index_leaf), out_shape);
			} else {
				out = low.Gather(n, in(0), sh(0), out_shape, idx_value->second.ints, axis, make_index_leaf);
			}
		} else if (op == "CastLike") {
			const auto like = low.shapes.find(n.inputs.at(1));
			if (like == low.shapes.end()) {
				Fail(n, "no shape for the like operand");
			}
			if (like->second.dtype == ShapeDtype::F32) {
				out = in(0);
			} else if (like->second.dtype == ShapeDtype::BOOL) {
				out = low.Custom(C_NEQ0, {in(0)}, NeOf(in(0)));
			} else if (like->second.dtype == ShapeDtype::I64) {
				out = low.Custom(C_TRUNC, {in(0)}, NeOf(in(0)));
			} else {
				Fail(n, "CastLike to this type of data the host cannot fold is not implemented");
			}
		} else if (op == "Gemm") {
			out = low.Gemm(n, in(0), in(1), n.inputs.size() > 2 && !n.inputs[2].empty() ? in(2) : nullptr, sh(0), sh(1), out_shape);
		} else if (op == "Cast") {
			const int64_t to = n.Attr("to") ? n.Attr("to")->i : 0;
			if (to == 1) {
				out = in(0); // float32 -> float32, or a 0/1 mask -> float (casts from/to float64 take the double-precision path above)
			} else if (to == 9) {
				out = low.Custom(C_NEQ0, {in(0)}, NeOf(in(0)));
			} else if (to == 10) {
				out = low.Custom(C_F16, {in(0)}, NeOf(in(0)));
			} else if (to == 6 || to == 7) {
				out = low.Custom(C_TRUNC, {in(0)}, NeOf(in(0))); // float -> int32/int64: toward zero, kept as an exact float
			} else {
				Fail(n, "Cast to ONNX type " + std::to_string(to) + " of data the host cannot fold is not implemented");
			}
		} else if (op == "SequenceAt") {
			const auto seq = low.sequences.find(n.inputs.at(0));
			if (seq == low.sequences.end()) {
				Fail(n, "'" + n.inputs.at(0) + "' is not a sequence made by SplitToSequence");
			}
			const auto &sq = seq->second;
			int64_t at = low.HostInts(n, n.inputs.at(1)).at(0);
			if (at < 0) {
				at += static_cast<int64_t>(sq.sizes.size());
			}
			if (at < 0 || at >= static_cast<int64_t>(sq.sizes.size())) {
				Fail(n, "sequence index out of range");
			}
			int64_t start = 0;
			for (int64_t i = 0; i < at; i++) {
				start += sq.sizes[i];
			}
			ggml_tensor *view = low.SliceCuts(n, sq.x, sq.shape, {{sq.axis, start, sq.sizes[at], 1}});
			out = low.Unfold(n, view, out_shape); // keepdims=0 drops the axis; above four dimensions the view is regrouped
		} else if (op == "TopK") {
			const int64_t k = low.HostInts(n, n.inputs.at(1)).at(0);
			const int64_t axis = normalised_axis(n.Attr("axis") ? n.Attr("axis")->i : -1, sh(0).size());
			const bool largest = !n.Attr("largest") || n.Attr("largest")->i != 0;
			out = low.TopK(n, in(0), sh(0), axis, k, largest, false);
			if (n.outputs.size() > 1 && !n.outputs[1].empty()) {
				ggml_tensor *indices = low.TopK(n, in(0), sh(0), axis, k, largest, true);
				low.tensors[n.outputs[1]] = indices;
				node_outs.push_back(indices);
			}
		} else if (op == "GatherElements") {
			const int64_t axis = normalised_axis(n.Attr("axis") ? n.Attr("axis")->i : 0, sh(0).size());
			if (sh(0).size() != sh(1).size() || sh(0).size() > 4) {
				Fail(n, "GatherElements needs data and indices of the same rank, at most four");
			}
			out = low.GatherElements(n, in(0), in(1), static_cast<int64_t>(sh(0).size()), axis, out_shape);
		} else if (op == "Pad") {
			if (n.Attr("mode") && n.Attr("mode")->s != "constant") {
				Fail(n, "Pad mode '" + n.Attr("mode")->s + "' is not implemented (only constant)");
			}
			if (n.inputs.size() > 3 && !n.inputs[3].empty()) {
				Fail(n, "Pad with an axes input is not implemented");
			}
			const auto &pads = low.HostInts(n, n.inputs.at(1));
			const size_t r = sh(0).size();
			if (pads.size() != 2 * r) {
				Fail(n, "pads has " + std::to_string(pads.size()) + " values for rank " + std::to_string(r));
			}
			flat_plans.emplace_back();
			FlatPlan &plan = flat_plans.back();
			plan.kind = FlatKind::PAD;
			plan.a = sh(0);
			plan.out = out_shape;
			for (size_t i = 0; i < 2 * r; i++) {
				if (pads[i] < 0) {
					Fail(n, "negative pads (cropping) are not implemented");
				}
			}
			plan.before.assign(pads.begin(), pads.begin() + r);
			for (size_t i = 0; i < r; i++) {
				if (out_shape.size() != r || out_shape[i] != plan.a[i] + pads[i] + pads[r + i]) {
					Fail(n, "the output shape does not equal the input plus its pads");
				}
			}
			if (n.inputs.size() > 2 && !n.inputs[2].empty()) {
				const auto &value = low.shapes.at(n.inputs[2]);
				if (!value.known || value.floats.size() != 1) {
					Fail(n, "the Pad constant must be a host-known scalar float");
				}
				plan.value = static_cast<float>(value.floats[0]);
			}
			out = low.FlatOp(n, plan, {in(0)});
		} else if (op == "CumSum") {
			flat_plans.emplace_back();
			FlatPlan &plan = flat_plans.back();
			plan.kind = FlatKind::CUMSUM;
			plan.a = sh(0);
			plan.out = out_shape;
			plan.axis = normalised_axis(low.HostInts(n, n.inputs.at(1)).at(0), sh(0).size());
			plan.exclusive = n.Attr("exclusive") && n.Attr("exclusive")->i != 0;
			plan.reverse = n.Attr("reverse") && n.Attr("reverse")->i != 0;
			out = low.FlatOp(n, plan, {in(0)});
		} else if ((op == "ScatterND" || op == "ScatterElements") && Numel(sh(1)) == 0) {
			out = in(0); // nothing to scatter: the data passes through unchanged
		} else if (op == "GatherND" || op == "ScatterND" || op == "ScatterElements") {
			if (op == "GatherND" && n.Attr("batch_dims") && n.Attr("batch_dims")->i != 0) {
				Fail(n, "GatherND with batch_dims != 0 is not implemented");
			}
			if (op != "GatherND" && n.Attr("reduction") && n.Attr("reduction")->s != "none") {
				Fail(n, op + " with reduction '" + n.Attr("reduction")->s + "' is not implemented");
			}
			flat_plans.emplace_back();
			FlatPlan &plan = flat_plans.back();
			plan.kind = op == "GatherND" ? FlatKind::GATHERND : op == "ScatterND" ? FlatKind::SCATTERND : FlatKind::SCATTERELEMENTS;
			plan.a = sh(0);
			plan.b = sh(1);
			plan.out = out_shape;
			std::vector<ggml_tensor *> args {in(0), in(1)};
			if (op != "GatherND") {
				plan.c = sh(2);
				args.push_back(in(2));
			}
			if (op == "ScatterElements") {
				plan.axis = normalised_axis(n.Attr("axis") ? n.Attr("axis")->i : 0, sh(0).size());
				if (plan.a.size() != plan.b.size() || plan.b != plan.c) {
					Fail(n, "ScatterElements needs data, indices and updates of the same rank, with indices and updates of equal shape");
				}
			} else if (plan.b.empty() || plan.b.back() > static_cast<int64_t>(plan.a.size())) {
				Fail(n, "the last indices dimension is out of range");
			}
			if (op == "ScatterND") {
				int64_t tuples = 1, block = 1;
				for (size_t d = 0; d + 1 < plan.b.size(); d++) {
					tuples *= plan.b[d];
				}
				for (size_t d = plan.b.back(); d < plan.a.size(); d++) {
					block *= plan.a[d];
				}
				if (Numel(plan.c) != tuples * block) {
					Fail(n, "updates has " + std::to_string(Numel(plan.c)) + " elements, indices and data need " + std::to_string(tuples * block));
				}
			}
			if (op == "ScatterElements") {
				for (size_t d = 0; d < plan.b.size(); d++) {
					if (static_cast<int64_t>(d) != plan.axis && plan.b[d] > plan.a[d]) {
						Fail(n, "indices are larger than the data outside the scatter axis");
					}
				}
			}
			out = low.FlatOp(n, plan, args);
		} else if (op == "Einsum") {
			std::string eq = n.Attr("equation") ? n.Attr("equation")->s : "";
			eq.erase(std::remove(eq.begin(), eq.end(), ' '), eq.end());
			const Shape &sa = sh(0), &sb = sh(1);
			if (eq == "...s,jhds->...jhd" || eq == "...s,hds->...hd") {
				// out[..., (j h d)] = sum_s a[..., s] * b[(j h d), s]: one matmul whose weight is b as it lies in memory
				if (sa.empty() || sb.empty() || sa.back() != sb.back()) {
					Fail(n, "Einsum '" + eq + "': the contracted dimension s differs");
				}
				const int64_t s_dim = sa.back(), rows = Numel(sa) / s_dim, cols = Numel(sb) / s_dim;
				ggml_tensor *a2 = low.As(n, in(0), {rows, s_dim}), *b2 = low.As(n, in(1), {cols, s_dim});
				out = low.Unfold(n, low.MulMat(b2, a2), out_shape);
			} else if (eq == "...hd,hds->...s") {
				if (sa.size() < 2 || sb.size() != 3 || sa[sa.size() - 2] != sb[0] || sa.back() != sb[1]) {
					Fail(n, "Einsum '" + eq + "': h and d do not match");
				}
				const int64_t hd = sb[0] * sb[1], rows = Numel(sa) / hd;
				out = low.Unfold(n, low.MatMul(n, low.As(n, in(0), {rows, hd}), low.As(n, in(1), {hd, sb[2]}), {rows, hd}, {hd, sb[2]}, {rows, sb[2]}), out_shape);
			} else if (eq == "...,f->...f") {
				Shape a_col = sa;
				a_col.push_back(1);
				Node as_mul = n; // Binary dispatches on the node's op type: an outer product is a Mul
				as_mul.op_type = "Mul";
				out = low.Broadcasted(n, {a_col.size() <= 4 ? low.As(n, in(0), a_col) : in(0), in(1)}, {a_col, sb}, out_shape, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) {
					return low.Binary(as_mul, low.Dim0Contiguous(o[0]), low.Dim0Contiguous(o[1]), ne);
				});
			} else {
				Fail(n, "Einsum equation '" + eq + "' is not one of the four the bundled graphs use");
			}
		} else if (op == "Where") {
			out = elementwise(3, [&](std::vector<ggml_tensor *> &o, const Ne4 &ne) { return low.Custom(C_WHERE, o, ne); });
		} else {
			Fail(n, "op is not implemented in the ggml lowering");
		}
		// Every lowering must produce the shape the shape pass (checked against ORT) says: a mismatch is a bug here. Above four
		// dimensions only the element count is the contract (the storage form is a regrouping of the same memory).
		const bool shape_ok = out_shape.size() > 4 ? ggml_nelements(out) == Numel(out_shape) : SameNe(out, ToNe(n, out_shape));
		if (!shape_ok) {
			Fail(n, "lowered to ne [" + std::to_string(out->ne[0]) + "," + std::to_string(out->ne[1]) + "," +
			            std::to_string(out->ne[2]) + "," + std::to_string(out->ne[3]) + "] but the shape pass says a different shape");
		}
		low.tensors[n.outputs.at(0)] = out;
		node_outs.push_back(out); // added to the graph later, in this order, if the requested outputs depend on it
	}

	std::vector<std::pair<std::string, ggml_tensor *>> &outputs = compiled->outputs;
	HostTensors &host_outputs = compiled->host_outputs;
	for (const auto &name : wanted_outputs) {
		Node pseudo;
		pseudo.name = name;
		pseudo.op_type = "<output>";
		auto it = low.tensors.find(name);
		if (it != low.tensors.end() && it->second->type == GGML_TYPE_F64) {
			ggml_tensor *src[1] = {it->second}; // returned rounded to float32
			ggml_tensor *as_float = ggml_custom_4d(ctx.get(), GGML_TYPE_F32, it->second->ne[0], it->second->ne[1], it->second->ne[2], it->second->ne[3], src, 1,
			                                       F64ToF32Kernel, 1, nullptr);
			low.tensors[name] = as_float;
			it = low.tensors.find(name);
		}
		if (it == low.tensors.end()) {
			const auto sv = low.shapes.find(name);
			if (sv != low.shapes.end() && sv->second.known && sv->second.dtype != ShapeDtype::F32) {
				HostTensor h; // an integer, bool or double output folded on the host (returned as floats)
				h.shape = sv->second.shape;
				for (auto v : sv->second.ints) {
					h.data.push_back(static_cast<float>(v));
				}
				for (auto v : sv->second.floats) {
					h.data.push_back(static_cast<float>(v));
				}
				host_outputs[name] = std::move(h);
				continue;
			}
			if (sv == low.shapes.end() || !sv->second.known) {
				throw std::runtime_error("ggml lowering: graph output '" + name + "' is not a tensor the engine computed");
			}
			tensor_of(pseudo, name); // a folded float constant
			it = low.tensors.find(name);
		}
		// An output must be a fresh, contiguous node: a graph input or a view cannot be read back in place.
		ggml_tensor *t = ggml_cont(ctx.get(), it->second);
		ggml_set_output(t);
		outputs.emplace_back(name, t);
		compiled->output_shapes[name] = low.shapes.at(name).shape;
	}

	// Build the graph from what the outputs actually depend on. Lowering can leave a node dead (a transposed copy that a MatMul replaced by
	// the original layout, a weight's Transpose), and a dead node is never computed. Expanding in lowering order keeps ggml's recursive
	// parent walk shallow: every operand of a node is already in the graph when its turn comes.
	{
		std::unordered_set<const ggml_tensor *> live;
		std::vector<const ggml_tensor *> stack;
		for (const auto &o : outputs) {
			stack.push_back(o.second);
		}
		while (!stack.empty()) {
			const ggml_tensor *t = stack.back();
			stack.pop_back();
			if (!live.insert(t).second) {
				continue;
			}
			for (int i = 0; i < GGML_MAX_SRC; i++) {
				if (t->src[i]) {
					stack.push_back(t->src[i]);
				}
			}
			if (t->view_src) {
				stack.push_back(t->view_src);
			}
		}
		for (ggml_tensor *out : node_outs) {
			if (live.count(out)) {
				ggml_build_forward_expand(g, out);
			}
		}
		for (const auto &o : outputs) {
			ggml_build_forward_expand(g, o.second);
		}
	}

	{
		std::unordered_set<const ggml_tensor *> roots;
		for (const auto &leaf_entry : compiled->input_leaves) {
			roots.insert(leaf_entry.second.first);
		}
		for (int i = 0; i < ggml_graph_n_nodes(g) && !compiled->device_mode; i++) {
			ggml_tensor *t = ggml_graph_node(g, i);
			if (t->view_src && roots.count(t->view_src)) {
				compiled->input_views.push_back(t);
			}
		}
	}
	lap("lowering");
	return compiled;
}

//! Run a compiled graph: point the input leaves at this call's buffers, allocate the activation arena (kept for the next run), compute, read the
//! outputs. The weights are read where the caller keeps them; nothing is copied in.
//! The backends one executor computes on.
struct ExecutorRuntime {
	ggml_backend_t cpu = nullptr;                    // always present; runs the custom kernels and, for device "cpu", everything
	ggml_threadpool_t pool = nullptr;
	int threads = 1;
	std::vector<ggml_backend_t> sched_backends;      // scheduler path: the device first, the CPU last (empty = plain CPU path)
	std::vector<ggml_backend_buffer_type_t> sched_bufts;
};

//! Wakes the thread pool for one compute and puts it back to sleep on every way out of the scope.
struct PoolRun {
	ggml_threadpool_t pool;
	explicit PoolRun(ggml_threadpool_t p) : pool(p) {
		if (pool) {
			ggml_threadpool_resume(pool);
		}
	}
	~PoolRun() {
		if (pool) {
			ggml_threadpool_pause(pool);
		}
	}
	PoolRun(const PoolRun &) = delete;
	PoolRun &operator=(const PoolRun &) = delete;
};

HostTensors ReadOutputs(Compiled &c) {
	HostTensors result = c.host_outputs; // the host-folded outputs are the same on every run of this signature
	for (const auto &o : c.outputs) {
		HostTensor h;
		h.shape = c.output_shapes.at(o.first);
		h.data.resize(ggml_nelements(o.second));
		ggml_backend_tensor_get(o.second, h.data.data(), 0, ggml_nbytes(o.second));
		result[o.first] = std::move(h);
	}
	return result;
}

HostTensors ExecuteCompiled(Compiled &c, const ExecutorRuntime &rt, const HostTensors &inputs) {
	ggml_backend_t backend = rt.cpu;
	ggml_threadpool_t pool = rt.pool;
	const int threads = rt.threads;
	PhaseClock clock;
	for (auto &leaf : c.input_leaves) {
		const auto it = inputs.find(leaf.first);
		if (it == inputs.end() || it->second.data.size() != leaf.second.second) {
			throw std::runtime_error("ggml lowering: graph input '" + leaf.first + "' changed size since the graph was compiled");
		}
		if (!c.device_mode) {
			leaf.second.first->data = const_cast<float *>(it->second.data.data());
		}
	}
	for (ggml_tensor *view : c.input_views) {
		view->data = static_cast<char *>(view->view_src->data) + view->view_offs; // follow the re-pointed leaf
	}
	c.run_error.failed = false;
	c.run_error.message.clear();
	ggml_backend_cpu_set_n_threads(backend, std::max(1, threads));
	ggml_backend_cpu_set_threadpool(backend, pool);

	const char *profile_env = std::getenv("TABFM_GGML_PROFILE");
	if (profile_env && std::string(profile_env) == "2" && !c.device_mode) { // the per-op profile builds a CPU-only scheduler: not for a device
		// Per-op-kind timing: the scheduler calls back around every node. Slower than the normal path (a graph view per node), and it
		// allocates through the scheduler, so a graph run this way must not be reused (the executor discards it).
		struct Acc {
			std::map<std::string, std::pair<double, int>> by_kind;
			std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
		} acc;
		ggml_backend_t backends[1] = {backend};
		ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, ggml_graph_size(c.g), false, false);
		ggml_backend_sched_set_eval_callback(
		    sched,
		    [](ggml_tensor *t, bool ask, void *user) -> bool {
			    auto &a = *static_cast<Acc *>(user);
			    if (ask) {
				    return true; // call back after the node has been computed
			    }
			    const auto now = std::chrono::steady_clock::now();
			    std::string kind = ggml_op_name(t->op);
			    if (t->op == GGML_OP_CUSTOM) {
				    kind += std::string(":") + t->name;
			    } else if (t->op == GGML_OP_MUL_MAT && std::getenv("TABFM_GGML_PROFILE_SHAPES")) {
				    char buf[160];
				    std::snprintf(buf, sizeof(buf), ":[%lld,%lld,%lld,%lld]x[%lld,%lld,%lld,%lld]", static_cast<long long>(t->src[0]->ne[0]),
				                  static_cast<long long>(t->src[0]->ne[1]), static_cast<long long>(t->src[0]->ne[2]),
				                  static_cast<long long>(t->src[0]->ne[3]), static_cast<long long>(t->src[1]->ne[0]),
				                  static_cast<long long>(t->src[1]->ne[1]), static_cast<long long>(t->src[1]->ne[2]),
				                  static_cast<long long>(t->src[1]->ne[3]));
				    kind += buf;
			    }
			    auto &slot = a.by_kind[kind];
			    slot.first += std::chrono::duration<double, std::milli>(now - a.last).count();
			    slot.second++;
			    a.last = std::chrono::steady_clock::now();
			    return true;
		    },
		    &acc);
		if (!ggml_backend_sched_alloc_graph(sched, c.g)) {
			ggml_backend_sched_free(sched);
			throw std::runtime_error("ggml lowering: could not allocate the compute buffers (profile)");
		}
		acc.last = std::chrono::steady_clock::now();
		{
			PoolRun running(pool); // the pool is created paused: without waking it the compute barrier never opens
			ggml_backend_sched_graph_compute(sched, c.g);
		}
		std::vector<std::pair<std::string, std::pair<double, int>>> rows(acc.by_kind.begin(), acc.by_kind.end());
		std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.second.first > b.second.first; });
		double total = 0;
		for (auto &r : rows) {
			total += r.second.first;
		}
		std::fprintf(stderr, "[ggml] compute by op kind (total %.1f ms)\n", total);
		for (size_t i = 0; i < rows.size() && i < (std::getenv("TABFM_GGML_PROFILE_SHAPES") ? 24u : 14u); i++) {
			std::fprintf(stderr, "[ggml]   %-34s %9.1f ms %5.1f%%  x%d\n", rows[i].first.c_str(), rows[i].second.first,
			             100.0 * rows[i].second.first / std::max(total, 1e-9), rows[i].second.second);
		}
		HostTensors result = c.host_outputs;
		for (const auto &o : c.outputs) {
			HostTensor h;
			h.shape = c.output_shapes.at(o.first);
			h.data.resize(ggml_nelements(o.second));
			ggml_backend_tensor_get(o.second, h.data.data(), 0, ggml_nbytes(o.second));
			result[o.first] = std::move(h);
		}
		ggml_backend_sched_free(sched);
		return result;
	}

	if (!rt.sched_backends.empty()) {
		// A device (or the test path): the scheduler splits the graph between the device and the CPU, copies inputs across the split and
		// owns the activation buffers. The weights are already in device memory (the executor uploaded them once).
		// The scheduler rewrites the graph's sources in place (an input that must cross to another backend is replaced by its copy tensor), so a
		// graph is split and allocated ONCE; later runs only upload the inputs and compute (the scheduler copies the inputs across each time).
		// Splitting the same graph again would find the stale copy tensors and never refresh them.
		if (!c.sched_ready) {
			c.sched.reset(ggml_backend_sched_new(const_cast<ggml_backend_t *>(rt.sched_backends.data()),
			                                     const_cast<ggml_backend_buffer_type_t *>(rt.sched_bufts.data()),
			                                     static_cast<int>(rt.sched_backends.size()), ggml_graph_size(c.g), false, true));
			if (!c.sched) {
				throw std::runtime_error("ggml lowering: could not create the device scheduler");
			}
			// Inputs and constants live on the device: left to the scheduler, a leaf feeding a device op can land in a host buffer.
			for (const auto &u : c.uploads) {
				ggml_backend_sched_set_tensor_backend(c.sched.get(), u.tensor, rt.sched_backends[0]);
			}
			if (!ggml_backend_sched_alloc_graph(c.sched.get(), c.g)) {
				c.sched.reset();
				throw std::runtime_error("ggml lowering: could not allocate the device compute buffers");
			}
			c.sched_ready = true;
			clock.Lap("sched alloc");
		}
		for (const auto &u : c.uploads) { // inputs and constants into their (device) tensors
			if (!u.tensor->buffer) {
				continue; // nothing in the graph reads it (an unused input, or a constant only a dead node needed): the scheduler never allocated it
			}
			const void *host = u.input_name.empty() ? u.host : inputs.at(u.input_name).data.data();
			ggml_backend_tensor_set(u.tensor, host, 0, ggml_nbytes(u.tensor));
		}
		clock.Lap("upload");
		ggml_status status;
		{
			PoolRun running(pool);
			status = ggml_backend_sched_graph_compute(c.sched.get(), c.g);
		}
		if (status != GGML_STATUS_SUCCESS) {
			throw std::runtime_error("ggml lowering: graph compute failed on the device");
		}
		clock.Lap("device compute");
		if (c.run_error.failed) {
			throw std::runtime_error("ggml lowering: " + c.run_error.message);
		}
		return ReadOutputs(c);
	}
	if (!c.alloc) {
		c.alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
	}
	if (!c.alloc || !ggml_gallocr_alloc_graph(c.alloc.get(), c.g)) {
		throw std::runtime_error("ggml lowering: could not allocate the compute buffers");
	}
	clock.Lap("gallocr alloc");
	ggml_status status;
	{
		PoolRun running(pool); // idle workers must not spin between calls: paused again on every exit, an exception included
		status = ggml_backend_graph_compute(backend, c.g);
	}
	if (status != GGML_STATUS_SUCCESS) {
		throw std::runtime_error("ggml lowering: graph compute failed");
	}
	clock.Lap("compute");
	if (c.run_error.failed) {
		throw std::runtime_error("ggml lowering: " + c.run_error.message);
	}
	return ReadOutputs(c);
}

} // namespace

std::vector<std::string> GgmlDeviceNames() {
	std::vector<std::string> names;
	for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
		names.push_back(ggml_backend_dev_name(ggml_backend_dev_get(i)));
	}
	return names;
}

struct GgmlExecutor::Impl {
	const ::anofox::onnxread::Graph &graph;
	GgmlExecutorConfig config;
	std::mutex mutex;
	ggml_backend_ptr cpu;
	ggml_backend_ptr device;                // the GPU backend, when config.device asks for one
	ggml_threadpool_t pool = nullptr;
	ExecutorRuntime runtime;
	// The device path: external weights live here, uploaded once, shared by every compiled graph.
	ggml_context_ptr weight_ctx;
	std::vector<ggml_backend_buffer_ptr> weight_buffers;
	std::map<std::string, ggml_tensor *> weights_on_device;
	std::list<std::unique_ptr<Compiled>> cache; // most recently used first
	size_t compiles = 0;

	Impl(const ::anofox::onnxread::Graph &graph_p, GgmlExecutorConfig config_p) : graph(graph_p), config(std::move(config_p)) {
		try {
			Init();
		} catch (...) {
			// A throwing constructor never runs the destructor: the worker threads of the pool would leak (a retry loop on a bad device leaks without bound).
			if (pool) {
				ggml_threadpool_free(pool);
				pool = nullptr;
			}
			throw;
		}
	}

	void Init() {
		config.threads = std::min(std::max(config.threads, 1), 256);
		cpu.reset(ggml_backend_cpu_init());
		if (!cpu) {
			throw std::runtime_error("ggml lowering: no CPU backend");
		}
		if (config.threads > 1 && !std::getenv("TABFM_GGML_NO_POOL")) {
			// One pool for the executor's life: a pool per call spawns and joins its threads every time, and its workers spin-wait.
			ggml_threadpool_params params = ggml_threadpool_params_default(config.threads);
			params.paused = true; // workers sleep until a run resumes them
			pool = ggml_threadpool_new(&params);
		}
		runtime.cpu = cpu.get();
		runtime.pool = pool;
		runtime.threads = config.threads;
		if (config.device == "cpu") {
			return;
		}
		if (config.device == "cpu-sched") {
			runtime.sched_backends = {cpu.get()};
			runtime.sched_bufts = {ggml_backend_get_default_buffer_type(cpu.get())};
		} else if (config.device.rfind("gpu", 0) == 0) {
			size_t want = 0; // the N of "gpu:N"
			if (config.device.size() > 3) {
				if (config.device[3] != ':' || config.device.size() == 4) {
					throw std::runtime_error("ggml lowering: device '" + config.device + "' is not 'cpu', 'gpu' or 'gpu:N'");
				}
				char *end = nullptr;
				const unsigned long parsed = std::strtoul(config.device.c_str() + 4, &end, 10);
				if (!end || *end != '\0' || !std::isdigit(static_cast<unsigned char>(config.device[4]))) {
					throw std::runtime_error("ggml lowering: device '" + config.device + "' is not 'cpu', 'gpu' or 'gpu:N' (N is a device number)");
				}
				want = static_cast<size_t>(parsed);
			}
			size_t seen = 0;
			ggml_backend_dev_t chosen = nullptr;
			for (size_t i = 0; i < ggml_backend_dev_count() && !chosen; i++) {
				ggml_backend_dev_t dev = ggml_backend_dev_get(i);
				const auto type = ggml_backend_dev_type(dev);
				if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
					chosen = seen++ == want ? dev : nullptr;
				}
			}
			if (!chosen) {
				std::string have;
				for (const auto &n : GgmlDeviceNames()) {
					have += (have.empty() ? "" : ", ") + n;
				}
				throw std::runtime_error("ggml lowering: device '" + config.device + "' was requested but this build's ggml has no such GPU (it contains: " + have + ")");
			}
			device.reset(ggml_backend_dev_init(chosen, nullptr));
			if (!device) {
				throw std::runtime_error(std::string("ggml lowering: could not initialise the GPU device ") + ggml_backend_dev_name(chosen));
			}
			runtime.sched_backends = {device.get(), cpu.get()};
			runtime.sched_bufts = {ggml_backend_get_default_buffer_type(device.get()), ggml_backend_get_default_buffer_type(cpu.get())};
		} else {
			throw std::runtime_error("ggml lowering: device '" + config.device + "' is not 'cpu', 'cpu-sched', 'gpu' or 'gpu:N'");
		}
		// Tensor objects of the persistent weights (no data: each weight gets its own device buffer when first used).
		// One tensor object per external initializer (a context that fills up aborts the process inside ggml): size it from the graph.
		const size_t weight_slots = std::max<size_t>(16384, graph.initializers.size() + 64);
		ggml_init_params params {ggml_tensor_overhead() * weight_slots, nullptr, true};
		weight_ctx.reset(ggml_init(params));
		if (!weight_ctx) {
			throw std::runtime_error("ggml lowering: ggml_init failed for the device weight table");
		}
	}

	//! The persistent device copy of an external weight: allocated and uploaded the first time a graph needs it.
	ggml_tensor *DeviceWeight(const std::string &name, const Ne4 &ne, const float *host, int onnx_type = 1) {
		const auto found = weights_on_device.find(name);
		if (found != weights_on_device.end()) {
			const ggml_tensor *have = found->second;
			if (have->ne[0] != ne[0] || have->ne[1] != ne[1] || have->ne[2] != ne[2] || have->ne[3] != ne[3]) {
				throw std::runtime_error("ggml lowering: weight '" + name + "' is requested with two different shapes");
			}
			return found->second;
		}
		const size_t weight_bytes = static_cast<size_t>(ne[0] * ne[1] * ne[2] * ne[3]) * sizeof(float);
		// Allocate before creating the tensor object: a failed allocation must not burn one of the table's fixed slots (a full ggml context aborts).
		ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(runtime.sched_bufts.front(), std::max<size_t>(weight_bytes, 1));
		if (!buffer) {
			throw std::runtime_error("ggml lowering: out of device memory uploading weight '" + name + "' (" + std::to_string(weight_bytes) +
			                         " bytes). Free the device (CALL tabfm_unload() for a model you no longer use, or SET anofox_tabfm_max_sessions = 1), use a smaller model, "
			                         "or SET anofox_tabfm_device='cpu'.");
		}
		ggml_tensor *t = ggml_new_tensor_4d(weight_ctx.get(), GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
		ggml_set_name(t, name.c_str());
		weight_buffers.emplace_back(buffer);
		ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); // the scheduler places ops where their weights live
		ggml_backend_tensor_alloc(buffer, t, ggml_backend_buffer_get_base(buffer));
		if (onnx_type == 1) {
			ggml_backend_tensor_set(t, host, 0, ggml_nbytes(t));
		} else {
			// bf16 / fp16 weights are upcast to float32 here, in chunks from the caller's (memory-mapped) bytes, so no second full host copy exists.
			const uint16_t *raw = reinterpret_cast<const uint16_t *>(host);
			const size_t total = static_cast<size_t>(ne[0] * ne[1] * ne[2] * ne[3]);
			std::vector<float> chunk(std::min<size_t>(total, size_t(1) << 20));
			for (size_t at = 0; at < total; at += chunk.size()) {
				const size_t n = std::min(chunk.size(), total - at);
				for (size_t i = 0; i < n; i++) {
					chunk[i] = onnx_type == 16 ? BFloat16ToFloat(raw[at + i]) : HalfToFloat(raw[at + i]);
				}
				ggml_backend_tensor_set(t, chunk.data(), at * sizeof(float), n * sizeof(float));
			}
		}
		weights_on_device[name] = t;
		return t;
	}

	Compiled *Acquire(const HostTensors &inputs, const GgmlRunOptions &options, std::unique_ptr<Compiled> &transient);

	~Impl() {
		cache.clear(); // compiled graphs (and their schedulers) before the backends and buffers they use
		weights_on_device.clear();
		weight_buffers.clear();
		weight_ctx.reset();
		device.reset();
		if (pool) { // before the CPU backend that was pointed at it
			ggml_threadpool_free(pool);
			pool = nullptr;
		}
		cpu.reset();
	}
};

GgmlExecutor::GgmlExecutor(const ::anofox::onnxread::Graph &graph, GgmlExecutorConfig config)
    : impl_(std::make_unique<Impl>(graph, std::move(config))) {
}

GgmlExecutor::~GgmlExecutor() = default;

//! The compiled graph for these inputs: the cached one (most recently used first) or a fresh compile. `transient` holds a graph that must not be cached.
Compiled *GgmlExecutor::Impl::Acquire(const HostTensors &inputs, const GgmlRunOptions &options, std::unique_ptr<Compiled> &transient) {
	Impl &impl = *this;
	const std::string signature = SignatureOf(impl.graph, inputs, options);
	for (auto it = impl.cache.begin(); it != impl.cache.end(); ++it) {
		if ((*it)->signature == signature) {
			impl.cache.splice(impl.cache.begin(), impl.cache, it); // most recently used goes first
			return impl.cache.front().get();
		}
	}
	const bool discard_after = [] {
		const char *p = std::getenv("TABFM_GGML_PROFILE");
		return p && std::string(p) == "2"; // the scheduler-allocated profile run cannot be reused
	}();
	GgmlRunOptions compile_options = options;
	compile_options.weights = impl.config.weights;
	compile_options.strict_fp32 = impl.config.strict_fp32;
	WeightProvider provider;
	if (!impl.runtime.sched_backends.empty()) {
		provider = [&impl](const std::string &name, const Ne4 &ne, const float *host, int onnx_type) { return impl.DeviceWeight(name, ne, host, onnx_type); };
	}
	auto fresh = CompileGraph(impl.graph, inputs, compile_options, impl.config.weights, provider);
	impl.compiles++;
	if (discard_after) {
		transient = std::move(fresh);
		return transient.get();
	}
	impl.cache.push_front(std::move(fresh));
	while (impl.cache.size() > std::max<size_t>(1, impl.config.cache_entries)) {
		impl.cache.pop_back();
	}
	return impl.cache.front().get();
}

std::vector<int64_t> GgmlExecutor::MulMatActivationRows(const HostTensors &inputs, const GgmlRunOptions &options) {
	std::lock_guard<std::mutex> guard(impl_->mutex);
	std::unique_ptr<Compiled> transient;
	Compiled *compiled = impl_->Acquire(inputs, options, transient);
	std::vector<int64_t> rows;
	for (int i = 0; i < ggml_graph_n_nodes(compiled->g); i++) {
		const ggml_tensor *t = ggml_graph_node(compiled->g, i);
		if (t->op == GGML_OP_MUL_MAT) {
			rows.push_back(t->src[1]->ne[1]);
		}
	}
	return rows;
}

HostTensors GgmlExecutor::Run(const HostTensors &inputs, const GgmlRunOptions &options) {
	std::lock_guard<std::mutex> guard(impl_->mutex);
	std::unique_ptr<Compiled> transient;
	Compiled *compiled = impl_->Acquire(inputs, options, transient);
	try {
		return ExecuteCompiled(*compiled, impl_->runtime, inputs);
	} catch (...) {
		// The scheduler rewrites a graph's sources when it splits it, so a Compiled whose allocation or run failed part way must not be reused:
		// the retry (after the user frees device memory, say) compiles afresh.
		if (!transient) {
			for (auto it = impl_->cache.begin(); it != impl_->cache.end(); ++it) {
				if (it->get() == compiled) {
					impl_->cache.erase(it);
					break;
				}
			}
		}
		throw;
	}
}

size_t GgmlExecutor::PreloadWeights() {
	std::lock_guard<std::mutex> guard(impl_->mutex);
	if (impl_->runtime.sched_backends.empty() || !impl_->config.weights) {
		return 0;
	}
	// The model has to fit: the weights stay resident in float32 (bf16/fp16 are upcast on upload). Fail here, at load, with the numbers and the way out,
	// instead of part way through the upload (or the first query).
	size_t needed = 0;
	for (const auto &t : impl_->graph.initializers) {
		const bool inline_data = !t.raw.empty() || !t.floats.empty() || !t.int64s.empty() || !t.int32s.empty();
		size_t count = 1;
		for (auto d : t.dims) {
			count *= static_cast<size_t>(d);
		}
		if (IsStoredWeightType(t.data_type) && !inline_data && count > 0) {
			const ExternalWeight external = impl_->config.weights(t.name, t.dims);
			if (external.data && external.count == count && external.onnx_type == t.data_type && !impl_->weights_on_device.count(t.name)) {
				needed += count * sizeof(float);
			}
		}
	}
	size_t free_bytes = 0;
	if (impl_->config.device_memory_probe) {
		free_bytes = impl_->config.device_memory_probe();
	} else if (impl_->device) {
		size_t total = 0;
		ggml_backend_dev_memory(ggml_backend_get_device(impl_->device.get()), &free_bytes, &total);
		if (total == 0) {
			free_bytes = 0; // the backend cannot say: no precheck
		}
	}
	if (free_bytes > 0 && needed > free_bytes) {
		throw std::runtime_error("ggml lowering: out of device memory: this model's weights need " + std::to_string(needed) + " bytes on the device but only " +
		                         std::to_string(free_bytes) + " bytes are free. Free the device (CALL tabfm_unload() for a model you no longer use, or SET anofox_tabfm_max_sessions = 1 "
		                         "so fewer models stay loaded), use a smaller model, or SET anofox_tabfm_device='cpu'.");
	}
	size_t uploaded = 0;
	for (const auto &t : impl_->graph.initializers) {
		const bool inline_data = !t.raw.empty() || !t.floats.empty() || !t.int64s.empty() || !t.int32s.empty();
		size_t count = 1;
		for (auto d : t.dims) {
			count *= static_cast<size_t>(d);
		}
		if (!IsStoredWeightType(t.data_type) || inline_data || count == 0) {
			continue; // only the float weights kept outside the ONNX file are persistent device tensors
		}
		const ExternalWeight external = impl_->config.weights(t.name, t.dims);
		if (!external.data || external.count != count || external.onnx_type != t.data_type) {
			continue;
		}
		try {
			Node pseudo; // the same call, with the same layout, materialize_initializer makes
			pseudo.name = t.name;
			pseudo.op_type = "<weight>";
			impl_->DeviceWeight(t.name, ToStorage(pseudo, Shape(t.dims.begin(), t.dims.end())), external.data, external.onnx_type);
			uploaded++;
		} catch (const std::bad_alloc &) {
			throw;
		} catch (const std::exception &e) {
			if (std::string(e.what()).find("out of device memory") != std::string::npos) {
				throw; // a real failure to report at load
			}
			// an unusable layout (rank above what a leaf holds): left to the run that needs it
		}
	}
	return uploaded;
}

void GgmlExecutor::Trim() {
	std::lock_guard<std::mutex> guard(impl_->mutex);
	impl_->cache.clear();
}

size_t GgmlExecutor::CompileCount() const {
	std::lock_guard<std::mutex> guard(impl_->mutex);
	return impl_->compiles;
}

HostTensors GgmlRunGraph(const ::anofox::onnxread::Graph &graph, const HostTensors &inputs, const GgmlRunOptions &options) {
	GgmlExecutorConfig config;
	config.threads = options.threads;
	config.weights = options.weights;
	config.cache_entries = 1;
	config.device = options.device;
	config.strict_fp32 = options.strict_fp32;
	GgmlExecutor executor(graph, std::move(config));
	return executor.Run(inputs, options);
}

HostTensors GgmlRunGraph(const ::anofox::onnxread::Graph &graph, const HostTensors &inputs, int threads, const WeightLookup &weights) {
	GgmlRunOptions options;
	options.threads = threads;
	options.weights = weights;
	return GgmlRunGraph(graph, inputs, options);
}

} // namespace anofox
} // namespace duckdb
