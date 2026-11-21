#include "tabfm_shape_eval.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <functional>
#include <cstring>
#include <numeric>

namespace duckdb {
namespace anofox {

using ::anofox::onnxread::Node;
using Shape = std::vector<int64_t>;

namespace {

[[noreturn]] void Fail(const Node &n, const std::string &what) {
	throw std::runtime_error("shape pass: " + n.op_type + " '" + n.name + "': " + what);
}

int64_t Numel(const Shape &s) {
	int64_t n = 1;
	for (auto d : s) {
		n *= d;
	}
	return n;
}

Shape StridesOf(const Shape &s) {
	Shape st(s.size(), 1);
	for (int i = static_cast<int>(s.size()) - 2; i >= 0; i--) {
		st[i] = st[i + 1] * s[i + 1];
	}
	return st;
}

//! ORT computes float32 tensors in float32; a double would give int(float(10) * 0.7f) = 6 where ORT gives 7.
double R32(double v) {
	return static_cast<double>(static_cast<float>(v));
}

//! One float operation's result in the precision of its tensor: float32 unless the ONNX graph made it double.
double RoundTo(ShapeDtype dt, double v) {
	return dt == ShapeDtype::F64 ? v : R32(v);
}

//! Materialising more than this many known elements is refused (the value becomes unknown): the pass folds shape
//! plumbing, not data, and ConstantOfShape([1e6, 1e6]) must not allocate terabytes.
constexpr int64_t kDefaultMaxKnownElements = 1 << 20;
thread_local int64_t g_max_known_elements = kDefaultMaxKnownElements;
#define kMaxKnownElements g_max_known_elements

int64_t WrapAdd(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}
int64_t WrapSub(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
}
int64_t WrapMul(int64_t a, int64_t b) {
	return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}
int64_t WrapNeg(int64_t a) {
	return static_cast<int64_t>(0 - static_cast<uint64_t>(a));
}

//! Exact integer power; a negative exponent truncates toward zero, as ORT's integer Pow does.
int64_t PowInt(int64_t base, int64_t exp) {
	if (exp < 0) {
		if (base == 1) {
			return 1;
		}
		if (base == -1) {
			return (exp % 2 == 0) ? 1 : -1;
		}
		return 0;
	}
	int64_t result = 1;
	for (int64_t i = 0; i < exp; i++) {
		result = WrapMul(result, base);
		if (result == 0 || (base == 1 || base == 0)) {
			break;
		}
		if (base == -1) {
			result = (exp % 2 == 0) ? 1 : -1;
			break;
		}
	}
	return base == 1 ? 1 : result;
}

//! Narrow integer casts wrap modulo the target width (the values stay in int64 storage).
int64_t WrapToOnnxInt(int64_t v, int32_t to) {
	switch (to) {
	case 2:
		return static_cast<uint8_t>(v);
	case 3:
		return static_cast<int8_t>(v);
	case 4:
		return static_cast<uint16_t>(v);
	case 5:
		return static_cast<int16_t>(v);
	case 6:
		return static_cast<int32_t>(v);
	case 12:
		return static_cast<uint32_t>(v);
	default:
		return v;
	}
}

ShapeDtype DtypeOfOnnx(int32_t t) {
	if (t == 11) {
		return ShapeDtype::F64;
	}
	if (t == 1 || t == 10 || t == 16) {
		return ShapeDtype::F32;
	}
	return t == 9 ? ShapeDtype::BOOL : ShapeDtype::I64;
}

bool IsFloat(const ShapeValue &v) {
	return v.dtype == ShapeDtype::F32 || v.dtype == ShapeDtype::F64;
}

double AsDouble(const ShapeValue &v, size_t i) {
	return IsFloat(v) ? v.floats[i] : static_cast<double>(v.ints[i]);
}

//! Saturating float -> int64, the way a sane cast behaves (a plain C cast of 9.2233720368547758e18 is undefined).
int64_t SaturateToInt64(double d) {
	if (std::isnan(d)) {
		return 0;
	}
	d = std::trunc(d);
	if (d >= 9.2233720368547758e18) {
		return INT64_MAX;
	}
	if (d <= -9.2233720368547758e18) {
		return INT64_MIN;
	}
	return static_cast<int64_t>(d);
}

ShapeValue Unknown(Shape shape, ShapeDtype dt) {
	ShapeValue v;
	v.shape = std::move(shape);
	v.dtype = dt;
	return v;
}

ShapeValue IntValue(Shape shape, std::vector<int64_t> ints, ShapeDtype dt = ShapeDtype::I64) {
	ShapeValue v;
	v.shape = std::move(shape);
	v.dtype = dt;
	v.known = true;
	v.ints = std::move(ints);
	return v;
}

ShapeValue FloatValue(Shape shape, std::vector<double> floats) {
	ShapeValue v;
	v.shape = std::move(shape);
	v.dtype = ShapeDtype::F32;
	v.known = true;
	v.floats = std::move(floats);
	return v;
}

//! A value with data from `source` rearranged: out element i comes from source element index[i].
ShapeValue Gathered(const ShapeValue &source, const std::vector<int64_t> &index, Shape shape) {
	ShapeValue out;
	out.shape = std::move(shape);
	out.dtype = source.dtype;
	out.known = true;
	if (IsFloat(source)) {
		for (auto i : index) {
			out.floats.push_back(source.floats[i]);
		}
	} else {
		for (auto i : index) {
			out.ints.push_back(source.ints[i]);
		}
	}
	return out;
}

//! Source offsets for an output of shape `os` read with `strides` (0 broadcasts) from `base`.
std::vector<int64_t> StridedIndex(const Shape &os, const Shape &strides, int64_t base = 0) {
	std::vector<int64_t> out;
	const int64_t n = Numel(os);
	out.reserve(static_cast<size_t>(n));
	if (n == 0) {
		return out;
	}
	std::vector<int64_t> idx(os.size(), 0);
	for (int64_t k = 0; k < n; k++) {
		int64_t off = base;
		for (size_t d = 0; d < os.size(); d++) {
			off += idx[d] * strides[d];
		}
		out.push_back(off);
		for (int d = static_cast<int>(os.size()) - 1; d >= 0; d--) {
			if (++idx[d] < os[d]) {
				break;
			}
			idx[d] = 0;
		}
	}
	return out;
}

Shape BroadcastShape(const Node &n, const Shape &a, const Shape &b) {
	const size_t rank = std::max(a.size(), b.size());
	Shape out(rank);
	for (size_t i = 0; i < rank; i++) {
		const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
		const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
		if (da != db && da != 1 && db != 1) {
			Fail(n, "operands do not broadcast");
		}
		out[i] = da == 1 ? db : da;
	}
	return out;
}

Shape BroadcastStrides(const Shape &s, const Shape &os) {
	const Shape st = StridesOf(s);
	Shape out(os.size(), 0);
	const size_t off = os.size() - s.size();
	for (size_t i = 0; i < s.size(); i++) {
		out[off + i] = s[i] == 1 ? 0 : st[i];
	}
	return out;
}

std::vector<int64_t> BroadcastIndex(const ShapeValue &v, const Shape &os) {
	return StridedIndex(os, BroadcastStrides(v.shape, os));
}

//! The integer values a shape-carrying operand must have (a Reshape target, a Slice bound, a Squeeze axis).
const std::vector<int64_t> &RequireInts(const Node &n, const ShapeValue &v, const char *what) {
	if (!v.known || IsFloat(v)) {
		Fail(n, std::string(what) + " depends on tensor data, not on shapes and constants");
	}
	return v.ints;
}

std::vector<int64_t> NormAxes(const std::vector<int64_t> &axes, size_t rank) {
	std::vector<int64_t> a;
	for (auto x : axes) {
		a.push_back(x < 0 ? x + static_cast<int64_t>(rank) : x);
	}
	return a;
}

int AxisAttr(const Node &n, const char *name, int64_t fallback, size_t rank) {
	const auto *a = n.Attr(name);
	int64_t v = a ? a->i : fallback;
	if (v < 0) {
		v += static_cast<int64_t>(rank);
	}
	return static_cast<int>(v);
}

//! Element-wise binary op with broadcasting. `fi` for two integer operands, `ff` otherwise; `out_dtype` overrides
//! the result type (comparisons give BOOL).
template <class FI, class FF>
ShapeValue Binary(const Node &n, const ShapeValue &a, const ShapeValue &b, FI fi, FF ff, bool boolean_out = false) {
	const Shape os = BroadcastShape(n, a.shape, b.shape);
	const bool floating = IsFloat(a) || IsFloat(b);
	const ShapeDtype dt = boolean_out ? ShapeDtype::BOOL
	                                  : (floating ? (a.dtype == ShapeDtype::F64 || b.dtype == ShapeDtype::F64 ? ShapeDtype::F64 : ShapeDtype::F32) : a.dtype);
	if (!a.known || !b.known || Numel(os) > kMaxKnownElements) {
		return Unknown(os, dt);
	}
	const auto ia = BroadcastIndex(a, os), ib = BroadcastIndex(b, os);
	ShapeValue out;
	out.shape = os;
	out.dtype = dt;
	out.known = true;
	for (size_t i = 0; i < ia.size(); i++) {
		if (floating) {
			const double r = ff(AsDouble(a, ia[i]), AsDouble(b, ib[i]));
			if (dt == ShapeDtype::F32 || dt == ShapeDtype::F64) {
				out.floats.push_back(RoundTo(dt, r));
			} else {
				out.ints.push_back(static_cast<int64_t>(r));
			}
		} else {
			out.ints.push_back(fi(a.ints[ia[i]], b.ints[ib[i]]));
		}
	}
	return out;
}

ShapeValue CastValue(const ShapeValue &x, ShapeDtype to, int32_t onnx_to = 7) {
	if (!x.known) {
		return Unknown(x.shape, to);
	}
	ShapeValue out;
	out.shape = x.shape;
	out.dtype = to;
	out.known = true;
	const size_t count = static_cast<size_t>(Numel(x.shape));
	for (size_t i = 0; i < count; i++) {
		if (to == ShapeDtype::F32 && onnx_to == 10) { // FLOAT16: stored as float32 but carrying half precision
			out.floats.push_back(static_cast<double>(RoundToHalf(static_cast<float>(AsDouble(x, i)))));
		} else if (to == ShapeDtype::F32 || to == ShapeDtype::F64) {
			out.floats.push_back(RoundTo(to, AsDouble(x, i)));
		} else if (to == ShapeDtype::BOOL) {
			out.ints.push_back(AsDouble(x, i) != 0.0 ? 1 : 0);
		} else {
			out.ints.push_back(WrapToOnnxInt(IsFloat(x) ? SaturateToInt64(x.floats[i]) : x.ints[i], onnx_to));
		}
	}
	return out;
}

ShapeValue TensorConstant(const ::anofox::onnxread::Tensor &t, const Node *node) {
	Shape shape(t.dims.begin(), t.dims.end());
	const ShapeDtype dt = DtypeOfOnnx(t.data_type);
	const size_t count = static_cast<size_t>(Numel(shape));
	const bool has_data = !t.raw.empty() || !t.floats.empty() || !t.int64s.empty() || !t.int32s.empty();
	if (t.location == ::anofox::onnxread::DataLocation::EXTERNAL || (!has_data && count > 0)) {
		return Unknown(shape, dt); // a weight: its values are not needed here
	}
	if (count == 0) {
		ShapeValue v = Unknown(shape, dt);
		v.known = true;
		return v;
	}
	auto width_of = [](int32_t type) -> size_t {
		switch (type) {
		case 1:
		case 6:
			return 4;
		case 7:
		case 11:
			return 8;
		case 2:
		case 3:
		case 9:
			return 1;
		default:
			return 0;
		}
	};
	if (!t.raw.empty() && width_of(t.data_type) != 0 && t.raw.size() < count * width_of(t.data_type)) {
		if (node) {
			Fail(*node, "constant has " + std::to_string(t.raw.size()) + " bytes for " + std::to_string(count) + " elements");
		}
		return Unknown(shape, dt);
	}
	if (t.raw.empty() && count > t.floats.size() + t.int64s.size() + t.int32s.size()) {
		return Unknown(shape, dt); // data fields shorter than the shape: not something we can read
	}
	auto raw_at = [&](size_t i, size_t width) { return t.raw.data() + i * width; };
	ShapeValue v;
	v.shape = shape;
	v.dtype = dt;
	v.known = true;
	for (size_t i = 0; i < count; i++) {
		switch (t.data_type) {
		case 1: {
			float f = t.raw.empty() ? t.floats.at(i) : 0.0f;
			if (!t.raw.empty()) {
				std::memcpy(&f, raw_at(i, 4), 4);
			}
			v.floats.push_back(f);
			break;
		}
		case 11: {
			if (t.raw.empty()) {
				return Unknown(shape, dt); // double_data is not parsed by the reader
			}
			double d = 0;
			std::memcpy(&d, raw_at(i, 8), 8);
			v.floats.push_back(d); // float64 constant: exact (its dtype is F64)
			break;
		}
		case 7: {
			int64_t x = t.raw.empty() ? t.int64s.at(i) : 0;
			if (!t.raw.empty()) {
				std::memcpy(&x, raw_at(i, 8), 8);
			}
			v.ints.push_back(x);
			break;
		}
		case 6: {
			int32_t x = t.raw.empty() ? t.int32s.at(i) : 0;
			if (!t.raw.empty()) {
				std::memcpy(&x, raw_at(i, 4), 4);
			}
			v.ints.push_back(x);
			break;
		}
		case 9:
		case 2:
			v.ints.push_back(t.raw.empty() ? t.int32s.at(i) : static_cast<uint8_t>(t.raw[i]));
			break;
		case 3:
			v.ints.push_back(t.raw.empty() ? t.int32s.at(i) : static_cast<int8_t>(t.raw[i]));
			break;
		default:
			return Unknown(shape, dt); // an element type the pass does not read: unknown, and refused only if a shape needs it
		}
	}
	return v;
}

struct Evaluator {
	std::unordered_map<std::string, ShapeValue> values;
	std::unordered_map<std::string, std::vector<ShapeValue>> sequences;

	const ShapeValue &Get(const Node &n, const std::string &name) const {
		auto it = values.find(name);
		if (it == values.end()) {
			Fail(n, "input '" + name + "' has no value (graph not topologically ordered?)");
		}
		return it->second;
	}

	std::vector<ShapeValue> Run(const Node &n, const std::vector<const ShapeValue *> &in);
};

#define UNARY_FLOAT(NAME, EXPR)                                                                                        \
	if (op == NAME) {                                                                                                  \
		const auto &x = *in[0];                                                                                        \
		if (!x.known) {                                                                                                \
			return {Unknown(x.shape, x.dtype)};                                                                            \
		}                                                                                                              \
		ShapeValue out = x;                                                                                            \
		for (auto &v : out.floats) {                                                                                   \
			const double a = v;                                                                                        \
			v = RoundTo(x.dtype, EXPR);                                                                                \
		}                                                                                                              \
		return {out};                                                                                                  \
	}

std::vector<ShapeValue> Evaluator::Run(const Node &n, const std::vector<const ShapeValue *> &in) {
	const std::string &op = n.op_type;

	// ---- constants and identity-like ----
	if (op == "Constant") {
		if (const auto *a = n.Attr("value")) {
			return {TensorConstant(a->t, &n)};
		}
		if (const auto *a = n.Attr("value_float")) {
			return {FloatValue({}, {static_cast<double>(a->f)})};
		}
		if (const auto *a = n.Attr("value_int")) {
			return {IntValue({}, {a->i})};
		}
		if (const auto *a = n.Attr("value_ints")) {
			return {IntValue({static_cast<int64_t>(a->ints.size())}, a->ints)};
		}
		if (const auto *a = n.Attr("value_floats")) {
			return {FloatValue({static_cast<int64_t>(a->float_list.size())},
			                   std::vector<double>(a->float_list.begin(), a->float_list.end()))};
		}
		Fail(n, "unsupported Constant form");
	}
	if (op == "Identity") {
		return {*in[0]};
	}

	// ---- shape-carrying ops ----
	if (op == "Shape" || op == "Size") {
		const auto &x = *in[0];
		const int64_t rank = static_cast<int64_t>(x.shape.size());
		if (op == "Size") {
			return {IntValue({}, {Numel(x.shape)})};
		}
		int64_t s = n.Attr("start") ? n.Attr("start")->i : 0;
		int64_t e = n.Attr("end") ? n.Attr("end")->i : rank;
		if (s < 0) {
			s += rank;
		}
		if (e < 0) {
			e += rank;
		}
		s = std::min(std::max<int64_t>(s, 0), rank);
		e = std::min(std::max<int64_t>(e, 0), rank);
		std::vector<int64_t> dims;
		for (int64_t i = s; i < e; i++) {
			dims.push_back(x.shape[i]);
		}
		return {IntValue({static_cast<int64_t>(dims.size())}, dims)};
	}
	if (op == "Cast") {
		if (!n.Attr("to")) {
			Fail(n, "the 'to' attribute is missing");
		}
		const int32_t to = static_cast<int32_t>(n.Attr("to")->i);
		return {CastValue(*in[0], DtypeOfOnnx(to), to)};
	}
	if (op == "CastLike") {
		return {CastValue(*in[0], in[1]->dtype)};
	}
	if (op == "Reshape") {
		const auto &x = *in[0];
		auto target = RequireInts(n, *in[1], "the target shape");
		const bool allow_zero = n.Attr("allowzero") && n.Attr("allowzero")->i;
		Shape os;
		int64_t neg = -1, known_product = 1;
		for (size_t i = 0; i < target.size(); i++) {
			int64_t v = target[i];
			if (v == 0 && !allow_zero) {
				v = x.shape.at(i);
			}
			if (v == -1) {
				if (neg >= 0) {
					Fail(n, "the target shape has more than one -1");
				}
				neg = static_cast<int64_t>(i);
				os.push_back(1);
			} else {
				os.push_back(v);
				known_product *= v;
			}
		}
		if (neg >= 0) {
			os[neg] = known_product ? Numel(x.shape) / known_product : 0;
		}
		if (Numel(os) != Numel(x.shape)) {
			Fail(n, "cannot reshape " + std::to_string(Numel(x.shape)) + " elements into the requested shape");
		}
		ShapeValue out = x;
		out.shape = os;
		return {out};
	}
	if (op == "Squeeze" || op == "Unsqueeze") {
		const auto &x = *in[0];
		std::vector<int64_t> axes;
		if (in.size() > 1 && in[1]) {
			axes = RequireInts(n, *in[1], "the axes");
		} else if (n.Attr("axes")) {
			axes = n.Attr("axes")->ints;
		}
		ShapeValue out = x;
		if (op == "Unsqueeze") {
			const size_t rank = x.shape.size() + axes.size();
			auto a = NormAxes(axes, rank);
			std::vector<bool> is_axis(rank, false);
			for (auto v : a) {
				if (v < 0 || v >= static_cast<int64_t>(rank) || is_axis[v]) {
					Fail(n, "the axes are out of range or repeated");
				}
				is_axis[v] = true;
			}
			Shape os(rank);
			size_t j = 0;
			for (size_t i = 0; i < rank; i++) {
				os[i] = is_axis[i] ? 1 : x.shape[j++];
			}
			out.shape = os;
		} else {
			auto a = NormAxes(axes, x.shape.size());
			for (auto v : a) {
				if (v < 0 || v >= static_cast<int64_t>(x.shape.size())) {
					Fail(n, "the axes are out of range");
				}
			}
			Shape os;
			for (size_t i = 0; i < x.shape.size(); i++) {
				const bool drop = axes.empty() ? x.shape[i] == 1
				                               : std::find(a.begin(), a.end(), static_cast<int64_t>(i)) != a.end();
				if (!drop) {
					os.push_back(x.shape[i]);
				}
			}
			out.shape = os;
		}
		return {out};
	}
	if (op == "Transpose") {
		const auto &x = *in[0];
		const size_t rank = x.shape.size();
		std::vector<int64_t> perm;
		if (n.Attr("perm")) {
			perm = n.Attr("perm")->ints;
		} else {
			for (size_t i = 0; i < rank; i++) {
				perm.push_back(static_cast<int64_t>(rank - 1 - i));
			}
		}
		Shape os, pst;
		const Shape st = StridesOf(x.shape);
		for (auto p : perm) {
			os.push_back(x.shape.at(p));
			pst.push_back(st[p]);
		}
		if (!x.known) {
			return {Unknown(os, x.dtype)};
		}
		return {Gathered(x, StridedIndex(os, pst), os)};
	}
	if (op == "Expand") {
		const auto &x = *in[0];
		const Shape target = RequireInts(n, *in[1], "the target shape");
		const Shape os = BroadcastShape(n, x.shape, target);
		if (!x.known || Numel(os) > kMaxKnownElements) {
			return {Unknown(os, x.dtype)};
		}
		return {Gathered(x, BroadcastIndex(x, os), os)};
	}
	if (op == "Concat") {
		const int axis = AxisAttr(n, "axis", 0, in[0]->shape.size());
		Shape os = in[0]->shape;
		int64_t total = 0;
		bool all_known = true;
		for (const auto *t : in) {
			if (t->shape.size() != os.size()) {
				Fail(n, "inputs have different ranks");
			}
			for (size_t d = 0; d < os.size(); d++) {
				if (static_cast<int>(d) != axis && t->shape[d] != os[d]) {
					Fail(n, "inputs differ in a dimension other than the concatenation axis");
				}
			}
			total += t->shape.at(axis);
			all_known = all_known && t->known;
		}
		os[axis] = total;
		if (!all_known) {
			return {Unknown(os, in[0]->dtype)};
		}
		size_t outer = 1, inner = 1;
		for (int i = 0; i < axis; i++) {
			outer *= static_cast<size_t>(os[i]);
		}
		for (size_t i = axis + 1; i < os.size(); i++) {
			inner *= static_cast<size_t>(os[i]);
		}
		ShapeValue out;
		out.shape = os;
		out.dtype = in[0]->dtype;
		out.known = true;
		for (size_t b = 0; b < outer; b++) {
			for (const auto *t : in) {
				const size_t len = static_cast<size_t>(t->shape[axis]) * inner;
				for (size_t k = 0; k < len; k++) {
					if (IsFloat(*t)) {
						out.floats.push_back(t->floats[b * len + k]);
					} else {
						out.ints.push_back(t->ints[b * len + k]);
					}
				}
			}
		}
		return {out};
	}
	if (op == "Slice") {
		const auto &x = *in[0];
		const size_t rank = x.shape.size();
		const auto starts = RequireInts(n, *in[1], "the slice starts");
		const auto ends = RequireInts(n, *in[2], "the slice ends");
		std::vector<int64_t> axes, steps;
		if (in.size() > 3 && in[3]) {
			axes = RequireInts(n, *in[3], "the slice axes");
		} else {
			for (size_t i = 0; i < starts.size(); i++) {
				axes.push_back(static_cast<int64_t>(i));
			}
		}
		if (in.size() > 4 && in[4]) {
			steps = RequireInts(n, *in[4], "the slice steps");
		} else {
			steps.assign(starts.size(), 1);
		}
		Shape os = x.shape;
		const Shape st = StridesOf(x.shape);
		Shape ss = st;
		int64_t base = 0;
		for (size_t k = 0; k < axes.size(); k++) {
			const int64_t a = axes[k] < 0 ? axes[k] + static_cast<int64_t>(rank) : axes[k];
			if (steps[k] == 0) {
				Fail(n, "the step must not be 0");
			}
			const int64_t dim = x.shape.at(a), step = steps[k];
			int64_t s = starts[k], e = ends[k], len;
			if (s < 0) {
				s += dim;
			}
			if (e < 0) {
				e += dim;
			}
			if (step > 0) {
				s = std::min(std::max<int64_t>(s, 0), dim);
				e = std::min(std::max<int64_t>(e, 0), dim);
				len = e > s ? static_cast<int64_t>((static_cast<__int128>(e) - s + step - 1) / step) : 0;
			} else {
				s = std::min(std::max<int64_t>(s, -1), dim - 1);
				e = std::min(std::max<int64_t>(e, -1), dim - 1);
				len = s > e ? static_cast<int64_t>((static_cast<__int128>(s) - e + (-static_cast<__int128>(step)) - 1) / (-static_cast<__int128>(step))) : 0;
			}
			os[a] = len;
			base += (len > 0 ? s : 0) * st[a];
			ss[a] = st[a] * step;
		}
		if (!x.known) {
			return {Unknown(os, x.dtype)};
		}
		return {Gathered(x, StridedIndex(os, ss, base), os)};
	}
	if (op == "Gather") {
		const auto &d = *in[0];
		const auto &idx = *in[1];
		const int axis = AxisAttr(n, "axis", 0, d.shape.size());
		Shape os(d.shape.begin(), d.shape.begin() + axis);
		os.insert(os.end(), idx.shape.begin(), idx.shape.end());
		os.insert(os.end(), d.shape.begin() + axis + 1, d.shape.end());
		if (!d.known || !idx.known) {
			return {Unknown(os, d.dtype)};
		}
		size_t outer = 1, inner = 1;
		for (int i = 0; i < axis; i++) {
			outer *= static_cast<size_t>(d.shape[i]);
		}
		for (size_t i = axis + 1; i < d.shape.size(); i++) {
			inner *= static_cast<size_t>(d.shape[i]);
		}
		const int64_t dim = d.shape[axis];
		std::vector<int64_t> index;
		for (size_t b = 0; b < outer; b++) {
			for (auto raw : idx.ints) {
				const int64_t id = raw < 0 ? raw + dim : raw;
				if (id < 0 || id >= dim) {
					Fail(n, "index " + std::to_string(raw) + " is out of range for an axis of " + std::to_string(dim));
				}
				for (size_t k = 0; k < inner; k++) {
					index.push_back((static_cast<int64_t>(b) * dim + id) * static_cast<int64_t>(inner) + k);
				}
			}
		}
		return {Gathered(d, index, os)};
	}
	if (op == "GatherElements") {
		const auto &d = *in[0];
		const auto &idx = *in[1];
		if (!d.known || !idx.known) {
			return {Unknown(idx.shape, d.dtype)};
		}
		const int axis = AxisAttr(n, "axis", 0, d.shape.size());
		const Shape ds = StridesOf(d.shape), is = StridesOf(idx.shape);
		std::vector<int64_t> index;
		const int64_t count = Numel(idx.shape);
		for (int64_t i = 0; i < count; i++) {
			int64_t rem = i, src = 0;
			for (size_t k = 0; k < idx.shape.size(); k++) {
				int64_t c = rem / is[k];
				rem %= is[k];
				if (static_cast<int>(k) == axis) {
					c = idx.ints[i] < 0 ? idx.ints[i] + d.shape[axis] : idx.ints[i];
					if (c < 0 || c >= d.shape[axis]) {
						Fail(n, "index " + std::to_string(idx.ints[i]) + " is out of range for an axis of " + std::to_string(d.shape[axis]));
					}
				}
				src += c * ds[k];
			}
			index.push_back(src);
		}
		return {Gathered(d, index, idx.shape)};
	}
	if (op == "Range") {
		if (!in[0]->known || !in[1]->known || !in[2]->known) {
			Fail(n, "Range bounds depend on tensor data");
		}
		if (!IsFloat(*in[0])) {
			const int64_t s = in[0]->ints[0], l = in[1]->ints[0], d = in[2]->ints[0];
			if (d == 0) {
				Fail(n, "the delta must not be 0");
			}
			const __int128 span = static_cast<__int128>(l) - s;
			__int128 count = d > 0 ? (span + d - 1) / d : (span + d + 1) / d;
			if (count < 0) {
				count = 0;
			}
			if (count > kMaxKnownElements) {
				return {Unknown({static_cast<int64_t>(count)}, ShapeDtype::I64)};
			}
			std::vector<int64_t> v;
			for (int64_t i = 0; i < static_cast<int64_t>(count); i++) {
				v.push_back(WrapAdd(s, WrapMul(i, d)));
			}
			return {IntValue({static_cast<int64_t>(v.size())}, v)};
		}
		const float s = static_cast<float>(in[0]->floats[0]), l = static_cast<float>(in[1]->floats[0]), d = static_cast<float>(in[2]->floats[0]);
		if (d == 0.0f) {
			Fail(n, "the delta must not be 0");
		}
		const float quotient = (l - s) / d; // float32 arithmetic, as ORT does it
		const int64_t count = std::isfinite(quotient) ? std::max<int64_t>(0, static_cast<int64_t>(std::ceil(quotient))) : 0;
		if (count > kMaxKnownElements) {
			return {Unknown({count}, ShapeDtype::F32)};
		}
		std::vector<double> v;
		for (int64_t i = 0; i < count; i++) {
			v.push_back(static_cast<double>(s + static_cast<float>(i) * d));
		}
		return {FloatValue({count}, v)};
	}
	if (op == "ConstantOfShape") {
		const Shape os = RequireInts(n, *in[0], "the output shape");
		ShapeValue fill = FloatValue({}, {0.0});
		if (const auto *a = n.Attr("value")) {
			fill = TensorConstant(a->t, &n);
		}
		const int64_t count = Numel(os);
		if (count > kMaxKnownElements || !fill.known) {
			return {Unknown(os, fill.dtype)};
		}
		ShapeValue out;
		out.shape = os;
		out.dtype = fill.dtype;
		out.known = true;
		for (int64_t i = 0; i < count; i++) {
			if (IsFloat(fill)) {
				out.floats.push_back(fill.floats.at(0));
			} else {
				out.ints.push_back(fill.ints.at(0));
			}
		}
		return {out};
	}

	// ---- element-wise ----
	if (op == "Add") {
		return {Binary(n, *in[0], *in[1], WrapAdd, [](double a, double b) { return a + b; })};
	}
	if (op == "Sub") {
		return {Binary(n, *in[0], *in[1], WrapSub, [](double a, double b) { return a - b; })};
	}
	if (op == "Mul") {
		return {Binary(n, *in[0], *in[1], WrapMul, [](double a, double b) { return a * b; })};
	}
	if (op == "Div") {
		return {Binary(n, *in[0], *in[1], [&n](int64_t a, int64_t b) {
			               if (b == 0) {
				               Fail(n, "integer division by zero");
			               }
			               return b == -1 ? WrapNeg(a) : a / b;
		               },
		               [](double a, double b) { return a / b; })};
	}
	if (op == "Pow") {
		return {Binary(n, *in[0], *in[1], PowInt, [](double a, double b) { return std::pow(a, b); })};
	}
	if (op == "Max") {
		ShapeValue acc = *in[0];
		for (size_t i = 1; i < in.size(); i++) {
			acc = Binary(n, acc, *in[i], [](int64_t a, int64_t b) { return std::max(a, b); }, [](double a, double b) { return (std::isnan(a) || std::isnan(b)) ? std::nan("") : std::max(a, b); });
		}
		return {acc};
	}
	if (op == "Min") {
		ShapeValue acc = *in[0];
		for (size_t i = 1; i < in.size(); i++) {
			acc = Binary(n, acc, *in[i], [](int64_t a, int64_t b) { return std::min(a, b); }, [](double a, double b) { return (std::isnan(a) || std::isnan(b)) ? std::nan("") : std::min(a, b); });
		}
		return {acc};
	}
	if (op == "Mod") {
		const bool fmod_mode = n.Attr("fmod") && n.Attr("fmod")->i;
		return {Binary(n, *in[0], *in[1],
		               [fmod_mode, &n](int64_t a, int64_t b) {
			               if (!b) {
				               Fail(n, "integer modulo by zero");
			               }
			               if (b == -1) {
				               return int64_t(0);
			               }
			               int64_t r = a % b;
			               if (!fmod_mode && r != 0 && ((r < 0) != (b < 0))) {
				               r += b;
			               }
			               return r;
		               },
		               [](double a, double b) { return std::fmod(a, b); })};
	}
	if (op == "Equal") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a == b); }, [](double a, double b) { return double(a == b); }, true)};
	}
	if (op == "Less") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a < b); }, [](double a, double b) { return double(a < b); }, true)};
	}
	if (op == "LessOrEqual") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a <= b); }, [](double a, double b) { return double(a <= b); }, true)};
	}
	if (op == "Greater") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a > b); }, [](double a, double b) { return double(a > b); }, true)};
	}
	if (op == "GreaterOrEqual") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a >= b); }, [](double a, double b) { return double(a >= b); }, true)};
	}
	if (op == "And") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a && b); }, [](double a, double b) { return double(a && b); }, true)};
	}
	if (op == "Or") {
		return {Binary(n, *in[0], *in[1], [](int64_t a, int64_t b) { return int64_t(a || b); }, [](double a, double b) { return double(a || b); }, true)};
	}
	if (op == "Not") {
		const auto &x = *in[0];
		if (!x.known) {
			return {Unknown(x.shape, ShapeDtype::BOOL)};
		}
		ShapeValue out = x;
		for (auto &v : out.ints) {
			v = !v;
		}
		return {out};
	}
	if (op == "IsNaN" || op == "IsInf") {
		const auto &x = *in[0];
		if (!x.known || !IsFloat(x)) {
			ShapeValue out = Unknown(x.shape, ShapeDtype::BOOL);
			if (x.known && !IsFloat(x)) {
				out = IntValue(x.shape, std::vector<int64_t>(x.ints.size(), 0), ShapeDtype::BOOL);
			}
			return {out};
		}
		const bool detect_pos = !n.Attr("detect_positive") || n.Attr("detect_positive")->i != 0;
		const bool detect_neg = !n.Attr("detect_negative") || n.Attr("detect_negative")->i != 0;
		std::vector<int64_t> bits;
		for (auto v : x.floats) {
			bits.push_back(op == "IsNaN" ? std::isnan(v) : (std::isinf(v) && (v > 0 ? detect_pos : detect_neg)));
		}
		return {IntValue(x.shape, bits, ShapeDtype::BOOL)};
	}
	if (op == "Neg" || op == "Abs") {
		const auto &x = *in[0];
		if (!x.known) {
			return {x};
		}
		ShapeValue out = x;
		for (auto &v : out.ints) {
			v = op == "Neg" ? WrapNeg(v) : (v < 0 ? WrapNeg(v) : v);
		}
		for (auto &v : out.floats) {
			v = op == "Neg" ? -v : std::fabs(v);
		}
		return {out};
	}
	if (op == "Where") {
		const auto &c = *in[0], &x = *in[1], &y = *in[2];
		const Shape os = BroadcastShape(n, BroadcastShape(n, c.shape, x.shape), y.shape);
		if (!c.known || !x.known || !y.known || Numel(os) > kMaxKnownElements) {
			return {Unknown(os, x.dtype)};
		}
		const auto ic = BroadcastIndex(c, os), ix = BroadcastIndex(x, os), iy = BroadcastIndex(y, os);
		ShapeValue out;
		out.shape = os;
		out.dtype = x.dtype;
		out.known = true;
		for (size_t i = 0; i < ic.size(); i++) {
			const bool take_x = AsDouble(c, ic[i]) != 0.0;
			if (IsFloat(x)) {
				out.floats.push_back(take_x ? x.floats[ix[i]] : y.floats[iy[i]]);
			} else {
				out.ints.push_back(take_x ? x.ints[ix[i]] : y.ints[iy[i]]);
			}
		}
		return {out};
	}
	if (op == "Clip") {
		const auto &x = *in[0];
		bool bounds_known = true;
		for (size_t i = 1; i < in.size(); i++) {
			bounds_known = bounds_known && (!in[i] || in[i]->known);
		}
		if (!x.known || !bounds_known) {
			return {Unknown(x.shape, x.dtype)};
		}
		ShapeValue out = x;
		// A missing bound is the largest finite value of the type (ORT), so the other bound alone still clamps +-inf.
		const double biggest = x.dtype == ShapeDtype::F64 ? std::numeric_limits<double>::max() : static_cast<double>(std::numeric_limits<float>::max());
		const bool has_lo = in.size() > 1 && in[1], has_hi = in.size() > 2 && in[2];
		for (auto &v : out.floats) {
			if (has_lo || has_hi) {
				v = std::max(v, has_lo ? AsDouble(*in[1], 0) : -biggest);
				v = std::min(v, has_hi ? AsDouble(*in[2], 0) : biggest);
			}
		}
		for (auto &v : out.ints) {
			if (in.size() > 1 && in[1]) {
				v = std::max(v, in[1]->ints[0]);
			}
			if (in.size() > 2 && in[2]) {
				v = std::min(v, in[2]->ints[0]);
			}
		}
		return {out};
	}
	if ((op == "Relu" || op == "Sign") && !IsFloat(*in[0])) { // integer tensors: the macro below only touches floats
		const auto &x = *in[0];
		if (!x.known) {
			return {Unknown(x.shape, x.dtype)};
		}
		ShapeValue out = x;
		for (auto &v : out.ints) {
			v = op == "Relu" ? std::max<int64_t>(v, 0) : static_cast<int64_t>((v > 0) - (v < 0));
		}
		return {out};
	}
	UNARY_FLOAT("Sqrt", std::sqrt(a))
	UNARY_FLOAT("Reciprocal", 1.0 / a)
	UNARY_FLOAT("Exp", std::exp(a))
	UNARY_FLOAT("Log", std::log(a))
	UNARY_FLOAT("Tanh", std::tanh(a))
	UNARY_FLOAT("Erf", std::erf(a))
	UNARY_FLOAT("Sigmoid", 1.0 / (1.0 + std::exp(-a)))
	UNARY_FLOAT("Ceil", std::ceil(a))
	UNARY_FLOAT("Floor", std::floor(a))
	UNARY_FLOAT("Sin", std::sin(a))
	UNARY_FLOAT("Cos", std::cos(a))
	UNARY_FLOAT("Round", std::nearbyint(a))
	UNARY_FLOAT("Relu", a < 0 ? 0.0 : a)
	UNARY_FLOAT("Sign", std::isnan(a) ? a : (a > 0) - (a < 0))
	UNARY_FLOAT("Softplus", std::log1p(std::exp(a)))

	// ---- compute ops (shape only: their values never feed shapes) ----
	if (op == "MatMul") {
		Shape a = in[0]->shape, b = in[1]->shape;
		const bool a_vec = a.size() == 1, b_vec = b.size() == 1;
		if (a_vec) {
			a.insert(a.begin(), 1);
		}
		if (b_vec) {
			b.push_back(1);
		}
		if (a.back() != b[b.size() - 2]) {
			Fail(n, "inner dimensions differ");
		}
		Shape ba(a.begin(), a.end() - 2), bb(b.begin(), b.end() - 2);
		Shape os = BroadcastShape(n, ba, bb);
		if (!a_vec) {
			os.push_back(a[a.size() - 2]);
		}
		if (!b_vec) {
			os.push_back(b.back());
		}
		return {Unknown(os, ShapeDtype::F32)};
	}
	if (op == "Gemm") {
		const Shape &a = in[0]->shape, &b = in[1]->shape;
		if (a.size() != 2 || b.size() != 2) {
			Fail(n, "Gemm operands must be rank 2");
		}
		const bool ta = n.Attr("transA") && n.Attr("transA")->i != 0, tb = n.Attr("transB") && n.Attr("transB")->i != 0;
		const int64_t m = ta ? a[1] : a[0], k = ta ? a[0] : a[1], k2 = tb ? b[1] : b[0], nn = tb ? b[0] : b[1];
		if (k != k2) {
			Fail(n, "inner dimensions differ");
		}
		return {Unknown({m, nn}, ShapeDtype::F32)};
	}
	if (op == "Softmax" || op == "LayerNormalization") {
		std::vector<ShapeValue> out = {Unknown(in[0]->shape, ShapeDtype::F32)};
		Shape stats = in[0]->shape; // Mean and InvStdDev: the normalised axes collapse to 1
		const int norm_axis = op == "LayerNormalization" ? AxisAttr(n, "axis", -1, stats.size()) : 0;
		for (size_t d = static_cast<size_t>(norm_axis); d < stats.size(); d++) {
			stats[d] = 1;
		}
		for (size_t k = 1; k < n.outputs.size(); k++) {
			out.push_back(Unknown(stats, ShapeDtype::F32));
		}
		return out;
	}
	if (op.rfind("Reduce", 0) == 0) {
		const auto &x = *in[0];
		std::vector<int64_t> axes;
		if (in.size() > 1 && in[1]) {
			axes = RequireInts(n, *in[1], "the reduction axes");
		} else if (n.Attr("axes")) {
			axes = n.Attr("axes")->ints;
		}
		const bool keep = n.Attr("keepdims") ? n.Attr("keepdims")->i != 0 : true;
		const bool noop = n.Attr("noop_with_empty_axes") && n.Attr("noop_with_empty_axes")->i;
		if (axes.empty() && noop) {
			return {x};
		}
		std::vector<bool> red(x.shape.size(), axes.empty());
		for (auto a : NormAxes(axes, x.shape.size())) {
			if (a < 0 || a >= static_cast<int64_t>(x.shape.size())) {
				Fail(n, "the reduction axes are out of range");
			}
			red[a] = true;
		}
		Shape os;
		for (size_t i = 0; i < x.shape.size(); i++) {
			if (!red[i]) {
				os.push_back(x.shape[i]);
			} else if (keep) {
				os.push_back(1);
			}
		}
		const bool evaluable = op == "ReduceSum" || op == "ReduceProd" || op == "ReduceMax" || op == "ReduceMin" || op == "ReduceMean";
		if (x.known && !IsFloat(x) && evaluable) {
			// integer reductions feed shapes (ReduceProd of a Shape is a size): evaluate them
			Shape ps, pst;
			const Shape st = StridesOf(x.shape);
			int64_t outer = 1, group = 1;
			for (size_t i = 0; i < x.shape.size(); i++) {
				if (!red[i]) {
					ps.push_back(x.shape[i]);
					pst.push_back(st[i]);
					outer *= x.shape[i];
				}
			}
			for (size_t i = 0; i < x.shape.size(); i++) {
				if (red[i]) {
					ps.push_back(x.shape[i]);
					pst.push_back(st[i]);
					group *= x.shape[i];
				}
			}
			const auto ix = StridedIndex(ps, pst);
			std::vector<int64_t> result;
			for (int64_t o = 0; o < outer; o++) {
				int64_t acc = op == "ReduceMax" ? INT64_MIN : op == "ReduceMin" ? INT64_MAX : op == "ReduceProd" ? 1 : 0;
				for (int64_t j = 0; j < group; j++) {
					const int64_t v = x.shape.empty() ? x.ints[0] : x.ints[ix[o * group + j]];
					acc = op == "ReduceMax" ? std::max(acc, v) : op == "ReduceMin" ? std::min(acc, v) : op == "ReduceProd" ? WrapMul(acc, v) : WrapAdd(acc, v);
				}
				result.push_back(op == "ReduceMean" && group ? acc / group : acc);
			}
			return {IntValue(os, result)};
		}
		return {Unknown(os, x.dtype)};
	}
	if (op == "TopK") {
		const auto &x = *in[0];
		const int64_t k = RequireInts(n, *in[1], "K").at(0);
		const int axis = AxisAttr(n, "axis", -1, x.shape.size());
		Shape os = x.shape;
		os[axis] = k;
		return {Unknown(os, x.dtype), Unknown(os, ShapeDtype::I64)};
	}
	if (op == "SplitToSequence") {
		const auto &x = *in[0];
		const int axis = AxisAttr(n, "axis", 0, x.shape.size());
		const bool keep = n.Attr("keepdims") ? n.Attr("keepdims")->i != 0 : true;
		std::vector<int64_t> sizes;
		const int64_t dim = x.shape[axis];
		const bool has_split = in.size() > 1 && in[1];
		if (has_split) {
			const auto &s = RequireInts(n, *in[1], "the split sizes");
			if (in[1]->shape.empty()) {
				if (s.at(0) <= 0) {
					Fail(n, "the split size must be positive");
				}
				for (int64_t o = 0; o < dim; o += s[0]) {
					sizes.push_back(std::min(s[0], dim - o));
				}
			} else {
				sizes = s;
			}
		} else {
			sizes.assign(static_cast<size_t>(dim), 1);
		}
		std::vector<ShapeValue> seq;
		for (auto sz : sizes) {
			Shape os = x.shape;
			os[axis] = sz;
			if (!has_split && !keep) {
				os.erase(os.begin() + axis);
			}
			seq.push_back(Unknown(os, x.dtype));
		}
		sequences[n.outputs[0]] = seq;
		return {};
	}
	if (op == "Pad") {
		const auto &x = *in[0];
		const Shape pads = RequireInts(n, *in[1], "the pads");
		const size_t r = x.shape.size();
		if (pads.size() != 2 * r || (in.size() > 3 && in[3])) {
			Fail(n, "pads must have two values per axis (an axes input is not supported)");
		}
		Shape os = x.shape;
		for (size_t i = 0; i < r; i++) {
			os[i] = WrapAdd(WrapAdd(os[i], pads[i]), pads[r + i]);
			if (os[i] < 0) {
				Fail(n, "negative pads crop an axis below zero");
			}
		}
		return {Unknown(os, x.dtype)};
	}
	if (op == "CumSum" || op == "ScatterND" || op == "ScatterElements") {
		return {Unknown(in[0]->shape, in[0]->dtype)};
	}
	if (op == "GatherND") {
		const auto &d = *in[0], &idx = *in[1];
		if (n.Attr("batch_dims") && n.Attr("batch_dims")->i != 0) {
			Fail(n, "batch_dims != 0 is not supported");
		}
		if (idx.shape.empty() || idx.shape.back() > static_cast<int64_t>(d.shape.size())) {
			Fail(n, "the last indices dimension exceeds the data rank");
		}
		Shape os(idx.shape.begin(), idx.shape.end() - 1);
		os.insert(os.end(), d.shape.begin() + idx.shape.back(), d.shape.end());
		return {Unknown(os, d.dtype)};
	}
	if (op == "Einsum") {
		std::string eq = n.Attr("equation") ? n.Attr("equation")->s : "";
		eq.erase(std::remove(eq.begin(), eq.end(), ' '), eq.end());
		const Shape &a = in[0]->shape, &b = in[1]->shape;
		Shape os;
		if (eq == "...s,jhds->...jhd" || eq == "...s,hds->...hd") {
			if (a.empty() || b.empty() || a.back() != b.back()) {
				Fail(n, "the contracted dimension differs");
			}
			os.assign(a.begin(), a.end() - 1);
			os.insert(os.end(), b.begin(), b.end() - 1);
		} else if (eq == "...hd,hds->...s") {
			if (a.size() < 2 || b.size() != 3 || a[a.size() - 2] != b[0] || a.back() != b[1]) {
				Fail(n, "h and d do not match");
			}
			os.assign(a.begin(), a.end() - 2);
			os.push_back(b[2]);
		} else if (eq == "...,f->...f") {
			os = a;
			os.insert(os.end(), b.begin(), b.end());
		} else {
			Fail(n, "equation '" + eq + "' is not supported by the shape pass");
		}
		return {Unknown(os, in[0]->dtype)};
	}
	Fail(n, "op is not supported by the shape pass yet");
}

} // namespace

float RoundToHalf(float f) {
	if (!std::isfinite(f) || f == 0.0f) {
		return f;
	}
	int exp2 = 0;
	std::frexp(f, &exp2);                  // f = m * 2^exp2 with m in [0.5, 1)
	const int e = std::max(exp2 - 1, -14); // half has 10 fraction bits; below 2^-14 it is subnormal with a fixed quantum of 2^-24
	const double quantum = std::ldexp(1.0, e - 10);
	const double rounded = std::nearbyint(static_cast<double>(f) / quantum) * quantum; // nearbyint: ties to even
	if (std::fabs(rounded) > 65504.0) {
		return f > 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
	}
	return static_cast<float>(rounded);
}

std::vector<ShapeValue> EvaluateNode(const ::anofox::onnxread::Node &node, const std::vector<const ShapeValue *> &inputs) {
	struct Unlimited { // the cap protects the host-side shape pass from huge constants, not inference from big tensors
		int64_t saved = g_max_known_elements;
		Unlimited() {
			g_max_known_elements = INT64_MAX;
		}
		~Unlimited() {
			g_max_known_elements = saved;
		}
	} unlimited;
	Evaluator ev;
	return ev.Run(node, inputs);
}

ShapeEvalResult EvaluateShapes(const ::anofox::onnxread::Graph &graph, const ShapeEvalInput &input) {
	Evaluator ev;
	std::unordered_map<std::string, bool> is_input;
	for (const auto &name : graph.inputs) {
		is_input[name] = true;
	}
	for (const auto &t : graph.initializers) {
		if (!is_input.count(t.name) || true) {
			ev.values[t.name] = TensorConstant(t, nullptr);
		}
	}
	for (const auto &kv : input.shapes) {
		const auto dt = input.dtypes.find(kv.first);
		ev.values[kv.first] = Unknown(kv.second, dt == input.dtypes.end() ? ShapeDtype::F32 : dt->second);
	}
	for (const auto &kv : input.ints) {
		Shape shape = kv.second.size() == 1 ? Shape() : Shape {static_cast<int64_t>(kv.second.size())};
		const auto given = input.int_shapes.find(kv.first);
		if (given != input.int_shapes.end()) {
			shape = given->second;
			int64_t count = 1;
			for (auto d : shape) {
				count *= d;
			}
			if (count != static_cast<int64_t>(kv.second.size())) {
				throw std::runtime_error("shape pass: integer input '" + kv.first + "' has " + std::to_string(kv.second.size()) +
				                         " values for a shape of " + std::to_string(count) + " elements");
			}
		}
		ev.values[kv.first] = IntValue(shape, kv.second);
	}
	for (const auto &node : graph.nodes) {
		if (node.op_type == "SequenceAt") {
			const auto &seq = ev.sequences.at(node.inputs.at(0));
			const auto &pos = ev.Get(node, node.inputs.at(1));
			int64_t p = RequireInts(node, pos, "the sequence position").at(0);
			if (p < 0) {
				p += static_cast<int64_t>(seq.size());
			}
			ev.values[node.outputs[0]] = seq.at(static_cast<size_t>(p));
			continue;
		}
		std::vector<const ShapeValue *> in;
		for (const auto &name : node.inputs) {
			in.push_back(name.empty() ? nullptr : &ev.Get(node, name));
		}
		auto outs = ev.Run(node, in);
		for (size_t k = 0; k < node.outputs.size() && k < outs.size(); k++) {
			if (!node.outputs[k].empty()) {
				ev.values[node.outputs[k]] = std::move(outs[k]);
			}
		}
	}
	ShapeEvalResult result;
	result.values = std::move(ev.values);
	return result;
}

} // namespace anofox
} // namespace duckdb
