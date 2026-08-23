/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_mlx_graph.cpp — the ONNX-over-MLX interpreter. See the header for why
 * this exists rather than seven hand-ported forwards.
 *
 * The op table below is a transcription of
 * tools/mlx_spike/src/mlx_spike/onnx_mlx.py, which is where each op's semantics
 * were settled against ORT CPU on real weights for all 13 shipped graphs. Where
 * a handler looks over-careful — Slice clamping its bounds, Expand multiplying
 * by ones instead of broadcasting, scatter round-tripping through int32 — it is
 * because the straightforward version was measured to be wrong on one of them.
 *===----------------------------------------------------------------------===*/

#include "tabfm_mlx_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include "tabfm_onnx_reader.hpp"

#include "mlx/c/array.h"
#include "mlx/c/fast.h"
#include "mlx/c/ops.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"

namespace anofox {
namespace mlxgraph {
namespace {

[[noreturn]] void Fail(const std::string &what) {
	throw std::runtime_error("anofox_tabfm mlx graph: " + what);
}

void Ok(int status, const char *what) {
	if (status != 0) {
		Fail(std::string("mlx call failed: ") + what);
	}
}

//! Move-only owner of an mlx_array (mlx-c hands back owned handles).
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
	Arr Share() const {
		Arr copy;
		Ok(mlx_array_set(copy.out(), a_), "array_set");
		return copy;
	}
	mlx_array get() const {
		return a_;
	}
	mlx_array *out() {
		return &a_;
	}
	bool valid() const {
		return mlx_array_ndim(a_) > 0 || mlx_array_size(a_) > 0;
	}
	size_t ndim() const {
		return mlx_array_ndim(a_);
	}
	size_t size() const {
		return mlx_array_size(a_);
	}
	mlx_dtype dtype() const {
		return mlx_array_dtype(a_);
	}
	std::vector<int> shape() const {
		const int *s = mlx_array_shape(a_);
		return std::vector<int>(s, s + mlx_array_ndim(a_));
	}

private:
	mlx_array a_;
};

//! Read a small integer tensor to host. Used for shapes, axes and indices --
//! never for weights, which stay on device.
//!
//! The cast runs on the CALLER'S stream, not a freshly minted one. Two reasons,
//! both learned the hard way: mlx_default_*_stream_new() returns an owned
//! handle that has to be freed (calling it inline leaked one per op), and
//! evaluating a GPU-produced array through an unrelated stream is how
//! mlx_array_data_int64 comes back null -- which surfaces as the misleading
//! "expected an integer tensor" rather than as a stream error.
std::vector<int64_t> ToInts(const Arr &a, mlx_stream s) {
	Arr i64;
	Ok(mlx_astype(i64.out(), a.get(), MLX_INT64, s), "astype int64");
	Ok(mlx_array_eval(i64.get()), "eval ints");
	if (i64.size() == 0) {
		return {}; // a legitimately empty shape/axes operand
	}
	const int64_t *p = mlx_array_data_int64(i64.get());
	if (!p) {
		// Say what was actually seen: "expected an integer tensor" on its own
		// sent me looking at dtypes when the cause was elsewhere.
		std::string shape;
		for (int d : i64.shape()) {
			shape += (shape.empty() ? "" : "x") + std::to_string(d);
		}
		Fail("could not read an integer operand (dtype=" + std::to_string(static_cast<int>(a.dtype())) +
		     " shape=[" + shape + "] size=" + std::to_string(i64.size()) + ")");
	}
	return std::vector<int64_t>(p, p + i64.size());
}

double ToScalar(const Arr &a, mlx_stream s) {
	Arr f;
	Ok(mlx_astype(f.out(), a.get(), MLX_FLOAT32, s), "astype f32");
	Ok(mlx_array_eval(f.get()), "eval scalar");
	const float *p = mlx_array_data_float32(f.get());
	if (!p || f.size() == 0) {
		Fail("expected a scalar");
	}
	return static_cast<double>(p[0]); // ONNX scalars arrive rank-0 OR rank-1
}

//! ONNX TensorProto.DataType -> mlx. Integers stay integers: these graphs
//! compute shapes at runtime, and a Shape result promoted to fp32 breaks any
//! dimension past 2^24.
mlx_dtype OnnxDtype(int elem_type) {
	switch (elem_type) {
	case 1:
		return MLX_FLOAT32;
	case 11:
		return MLX_FLOAT32; // DOUBLE: MLX has no float64; used only for constants
	case 10:
		return MLX_FLOAT16;
	case 16:
		return MLX_BFLOAT16;
	case 7:
		return MLX_INT64;
	case 6:
		return MLX_INT32;
	case 3:
		return MLX_INT8;
	case 2:
		return MLX_UINT8;
	case 9:
		return MLX_BOOL;
	default:
		Fail("unhandled ONNX dtype " + std::to_string(elem_type));
	}
}

//! A read-only mapping of the safetensors blob the graph's initializers index.
//! Mapped rather than read: tabfm-v1 is 6.6 GB and the point of the ext graph
//! layout is that nobody copies it.
class Mapping {
public:
	explicit Mapping(const std::string &path) {
		fd_ = ::open(path.c_str(), O_RDONLY);
		if (fd_ < 0) {
			Fail("cannot open weights file '" + path + "'");
		}
		struct stat st {};
		if (::fstat(fd_, &st) != 0) {
			::close(fd_);
			Fail("cannot stat '" + path + "'");
		}
		size_ = static_cast<size_t>(st.st_size);
		base_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
		if (base_ == MAP_FAILED) {
			::close(fd_);
			Fail("cannot map '" + path + "'");
		}
	}
	~Mapping() {
		if (base_ && base_ != MAP_FAILED) {
			::munmap(base_, size_);
		}
		if (fd_ >= 0) {
			::close(fd_);
		}
	}
	Mapping(const Mapping &) = delete;
	Mapping &operator=(const Mapping &) = delete;

	const void *At(size_t offset, size_t length) const {
		if (offset + length > size_) {
			Fail("initializer runs past the end of the weights file");
		}
		return static_cast<const uint8_t *>(base_) + offset;
	}

private:
	int fd_ = -1;
	void *base_ = nullptr;
	size_t size_ = 0;
};

using Node = ::anofox::onnxread::Node;

// --- attribute access -------------------------------------------------------
const ::anofox::onnxread::Attribute *Attr(const Node &n, const char *name) {
	return n.Attr(name);
}
int64_t AttrInt(const Node &n, const char *name, int64_t fallback) {
	const auto *a = n.Attr(name);
	return a ? a->i : fallback;
}
std::string AttrStr(const Node &n, const char *name, const std::string &fallback) {
	const auto *a = n.Attr(name);
	return a ? a->s : fallback;
}
std::vector<int64_t> AttrInts(const Node &n, const char *name, bool *found = nullptr) {
	const auto *a = n.Attr(name);
	if (found) {
		*found = a != nullptr;
	}
	return a ? a->ints : std::vector<int64_t>();
}

} // namespace

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------
namespace {

struct Ctx {
	mlx_stream s;
};

using Args = std::vector<const Arr *>;
using Handler = std::function<std::vector<Arr>(const Node &, const Args &, const Ctx &)>;

std::vector<int> ToShape(const std::vector<int64_t> &v) {
	return std::vector<int>(v.begin(), v.end());
}

Arr One(std::function<int(mlx_array *)> fn, const char *what) {
	Arr r;
	Ok(fn(r.out()), what);
	return r;
}

#define UNARY(NAME, CALL)                                                                                              \
	{                                                                                                                  \
		NAME, [](const Node &, const Args &in, const Ctx &c) -> std::vector<Arr> {                                      \
			Arr r;                                                                                                     \
			Ok(CALL(r.out(), in[0]->get(), c.s), NAME);                                                                 \
			std::vector<Arr> out;                                                                                      \
			out.push_back(std::move(r));                                                                               \
			return out;                                                                                                \
		}                                                                                                              \
	}

#define BINARY(NAME, CALL)                                                                                             \
	{                                                                                                                  \
		NAME, [](const Node &, const Args &in, const Ctx &c) -> std::vector<Arr> {                                      \
			Arr r;                                                                                                     \
			Ok(CALL(r.out(), in[0]->get(), in[1]->get(), c.s), NAME);                                                   \
			std::vector<Arr> out;                                                                                      \
			out.push_back(std::move(r));                                                                               \
			return out;                                                                                                \
		}                                                                                                              \
	}

std::vector<Arr> Wrap(Arr a) {
	std::vector<Arr> out;
	out.push_back(std::move(a));
	return out;
}

const std::unordered_map<std::string, Handler> &Handlers();

} // namespace
} // namespace mlxgraph
} // namespace anofox

#include "tabfm_mlx_graph_ops.inc"
