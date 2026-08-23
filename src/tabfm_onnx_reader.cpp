/*===----------------------------------------------------------------------===
 * tabfm_onnx_reader.cpp — protobuf wire-format decoding for the ONNX subset
 * the MLX interpreter needs. See the header for why this exists instead of a
 * dependency on libonnx + libprotobuf.
 *===----------------------------------------------------------------------===*/

#include "tabfm_onnx_reader.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace anofox {
namespace onnxread {
namespace {

[[noreturn]] void Fail(const std::string &what) {
	throw std::runtime_error("anofox_tabfm onnx reader: " + what);
}

//! A bounded cursor over one message's bytes. Every read is range-checked, so
//! a truncated or hostile file produces an exception rather than a walk off
//! the end of the buffer.
class Reader {
public:
	Reader(const uint8_t *begin, const uint8_t *end) : p_(begin), end_(end) {
	}

	bool Done() const {
		return p_ >= end_;
	}

	uint64_t Varint() {
		uint64_t value = 0;
		int shift = 0;
		while (true) {
			if (p_ >= end_) {
				Fail("truncated varint");
			}
			const uint8_t byte = *p_++;
			value |= static_cast<uint64_t>(byte & 0x7f) << shift;
			if (!(byte & 0x80)) {
				return value;
			}
			shift += 7;
			if (shift > 63) {
				Fail("varint too long");
			}
		}
	}

	Reader Sub() {
		const uint64_t len = Varint();
		if (static_cast<uint64_t>(end_ - p_) < len) {
			Fail("length-delimited field runs past the end");
		}
		Reader sub(p_, p_ + len);
		p_ += len;
		return sub;
	}

	std::string Bytes() {
		const uint64_t len = Varint();
		if (static_cast<uint64_t>(end_ - p_) < len) {
			Fail("string runs past the end");
		}
		std::string out(reinterpret_cast<const char *>(p_), len);
		p_ += len;
		return out;
	}

	uint32_t Fixed32() {
		if (end_ - p_ < 4) {
			Fail("truncated fixed32");
		}
		uint32_t v;
		std::memcpy(&v, p_, 4);
		p_ += 4;
		return v;
	}

	uint64_t Fixed64() {
		if (end_ - p_ < 8) {
			Fail("truncated fixed64");
		}
		uint64_t v;
		std::memcpy(&v, p_, 8);
		p_ += 8;
		return v;
	}

	//! Skip a field we do not model. Wire type alone determines the length,
	//! which is what makes ignoring unknown fields safe.
	void Skip(uint32_t wire) {
		switch (wire) {
		case 0:
			Varint();
			break;
		case 1:
			Fixed64();
			break;
		case 2:
			Sub();
			break;
		case 5:
			Fixed32();
			break;
		default:
			Fail("unsupported wire type " + std::to_string(wire));
		}
	}

private:
	const uint8_t *p_;
	const uint8_t *end_;
};

//! Packed repeated numerics may arrive packed (wire type 2) or one-per-field.
template <typename T>
void ReadPacked(Reader &r, uint32_t wire, std::vector<T> &out, bool fixed) {
	if (wire == 2) {
		Reader sub = r.Sub();
		while (!sub.Done()) {
			out.push_back(fixed ? static_cast<T>(sub.Fixed32()) : static_cast<T>(sub.Varint()));
		}
	} else if (wire == 5) {
		out.push_back(static_cast<T>(r.Fixed32()));
	} else {
		out.push_back(static_cast<T>(r.Varint()));
	}
}

void ReadPackedFloat(Reader &r, uint32_t wire, std::vector<float> &out) {
	auto bits_to_float = [](uint32_t bits) {
		float f;
		std::memcpy(&f, &bits, 4);
		return f;
	};
	if (wire == 2) {
		Reader sub = r.Sub();
		while (!sub.Done()) {
			out.push_back(bits_to_float(sub.Fixed32()));
		}
	} else {
		out.push_back(bits_to_float(r.Fixed32()));
	}
}

Tensor ParseTensor(Reader r);

std::map<std::string, std::string> ParseStringEntry(Reader r) {
	std::string key, value;
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		if (field == 1 && wire == 2) {
			key = r.Bytes();
		} else if (field == 2 && wire == 2) {
			value = r.Bytes();
		} else {
			r.Skip(wire);
		}
	}
	return {{key, value}};
}

Tensor ParseTensor(Reader r) {
	Tensor t;
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		switch (field) {
		case 1:
			ReadPacked<int64_t>(r, wire, t.dims, false);
			break;
		case 2:
			t.data_type = static_cast<int32_t>(r.Varint());
			break;
		case 4:
			ReadPackedFloat(r, wire, t.floats);
			break;
		case 5:
			ReadPacked<int32_t>(r, wire, t.int32s, false);
			break;
		case 7:
			ReadPacked<int64_t>(r, wire, t.int64s, false);
			break;
		case 8:
			t.name = r.Bytes();
			break;
		case 9:
			t.raw = r.Bytes();
			break;
		case 13: {
			auto kv = ParseStringEntry(r.Sub());
			for (auto &e : kv) {
				t.external[e.first] = e.second;
			}
			break;
		}
		case 14:
			t.location = static_cast<DataLocation>(r.Varint());
			break;
		default:
			r.Skip(wire);
		}
	}
	return t;
}

Attribute ParseAttribute(Reader r) {
	Attribute a;
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		switch (field) {
		case 1:
			a.name = r.Bytes();
			break;
		case 2: {
			uint32_t bits = r.Fixed32();
			std::memcpy(&a.f, &bits, 4);
			break;
		}
		case 3:
			a.i = static_cast<int64_t>(r.Varint());
			break;
		case 4:
			a.s = r.Bytes();
			break;
		case 5:
			a.t = ParseTensor(r.Sub());
			a.has_tensor = true;
			break;
		case 7:
			ReadPackedFloat(r, wire, a.float_list);
			break;
		case 8:
			ReadPacked<int64_t>(r, wire, a.ints, false);
			break;
		case 20:
			a.type = static_cast<int32_t>(r.Varint());
			break;
		default:
			r.Skip(wire);
		}
	}
	return a;
}

Node ParseNode(Reader r) {
	Node n;
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		switch (field) {
		case 1:
			n.inputs.push_back(r.Bytes());
			break;
		case 2:
			n.outputs.push_back(r.Bytes());
			break;
		case 3:
			n.name = r.Bytes();
			break;
		case 4:
			n.op_type = r.Bytes();
			break;
		case 5:
			n.attributes.push_back(ParseAttribute(r.Sub()));
			break;
		default:
			r.Skip(wire);
		}
	}
	return n;
}

std::string ParseValueInfoName(Reader r) {
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		if (field == 1 && wire == 2) {
			return r.Bytes();
		}
		r.Skip(wire);
	}
	return {};
}

Graph ParseGraph(Reader r) {
	Graph g;
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		switch (field) {
		case 1:
			g.nodes.push_back(ParseNode(r.Sub()));
			break;
		case 5:
			g.initializers.push_back(ParseTensor(r.Sub()));
			break;
		case 11:
			g.inputs.push_back(ParseValueInfoName(r.Sub()));
			break;
		case 12:
			g.outputs.push_back(ParseValueInfoName(r.Sub()));
			break;
		default:
			r.Skip(wire);
		}
	}
	return g;
}

} // namespace

Graph ParseModelFile(const std::string &path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		Fail("cannot open '" + path + "'");
	}
	std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (blob.empty()) {
		Fail("'" + path + "' is empty");
	}
	const auto *begin = reinterpret_cast<const uint8_t *>(blob.data());
	Reader r(begin, begin + blob.size());
	while (!r.Done()) {
		const uint64_t tag = r.Varint();
		const uint32_t field = static_cast<uint32_t>(tag >> 3);
		const uint32_t wire = static_cast<uint32_t>(tag & 7);
		if (field == 7 && wire == 2) { // ModelProto.graph
			return ParseGraph(r.Sub());
		}
		r.Skip(wire);
	}
	Fail("'" + path + "' contains no graph");
}

} // namespace onnxread
} // namespace anofox
