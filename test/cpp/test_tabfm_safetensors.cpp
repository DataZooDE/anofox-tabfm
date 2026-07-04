// Catch2 tests for tabfm_safetensors — WS-B.
//
// All positive/negative cases hand-craft safetensors buffers in memory
// (8-byte LE header length + JSON header + data section); one optional test
// reads the WS-A fixture test/fixtures/model.safetensors when it exists.

#include "catch.hpp"

#include "tabfm_safetensors.hpp"

#include "duckdb/common/exception.hpp"

#include <cstring>
#include <fstream>

using namespace duckdb;
using namespace duckdb::anofox;
using Catch::Matchers::Contains;

namespace {

//! Append a little-endian scalar to a byte string.
template <class T>
void Append(string &out, T value) {
	char bytes[sizeof(T)];
	std::memcpy(bytes, &value, sizeof(T));
	out.append(bytes, sizeof(T));
}

//! Assemble a safetensors buffer: [u64 LE header_len][header JSON][data].
string MakeSafetensors(const string &header_json, const string &data) {
	string out;
	Append<uint64_t>(out, header_json.size());
	out += header_json;
	out += data;
	return out;
}

SafetensorsView Parse(const string &buffer, const string &source = "test buffer") {
	return ParseSafetensors(const_data_ptr_cast(buffer.data()), buffer.size(), source);
}

} // anonymous namespace

TEST_CASE("safetensors: valid F32 tensor parses", "[tabfm][safetensors]") {
	string data;
	Append<float>(data, 1.0f);
	Append<float>(data, -2.5f);
	Append<float>(data, 0.0f);
	Append<float>(data, 42.0f);
	auto buffer = MakeSafetensors(R"({"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})", data);

	auto view = Parse(buffer);
	REQUIRE(view.tensors.size() == 1);
	REQUIRE(view.Has("w"));
	auto &info = view.Get("w");
	REQUIRE(info.dtype == SafetensorsDtype::F32);
	REQUIRE(info.shape == vector<int64_t> {2, 2});
	REQUIRE(info.ElementCount() == 4);
	REQUIRE(info.nbytes == 16);
	REQUIRE(info.data_offset == 0);

	float values[4];
	std::memcpy(values, info.data, sizeof(values));
	REQUIRE(values[0] == 1.0f);
	REQUIRE(values[1] == -2.5f);
	REQUIRE(values[2] == 0.0f);
	REQUIRE(values[3] == 42.0f);
	REQUIRE(view.metadata.empty());
}

TEST_CASE("safetensors: multiple tensors with I64 and BF16", "[tabfm][safetensors]") {
	string data;
	Append<int64_t>(data, -7);
	Append<int64_t>(data, 1LL << 40);
	Append<uint16_t>(data, 0x3F80); // bf16 1.0
	Append<uint16_t>(data, 0x4000); // bf16 2.0
	auto buffer = MakeSafetensors(R"({"idx":{"dtype":"I64","shape":[2],"data_offsets":[0,16]},)"
	                              R"("w":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}})",
	                              data);

	auto view = Parse(buffer);
	REQUIRE(view.tensors.size() == 2);
	auto &idx = view.Get("idx");
	REQUIRE(idx.dtype == SafetensorsDtype::I64);
	REQUIRE(idx.nbytes == 16);
	int64_t ints[2];
	std::memcpy(ints, idx.data, sizeof(ints));
	REQUIRE(ints[0] == -7);
	REQUIRE(ints[1] == (1LL << 40));

	auto &w = view.Get("w");
	REQUIRE(w.dtype == SafetensorsDtype::BF16);
	REQUIRE(w.ElementCount() == 2);
	REQUIRE(w.nbytes == 4);
	REQUIRE(w.data_offset == 16);
}

TEST_CASE("safetensors: __metadata__ is exposed", "[tabfm][safetensors]") {
	auto buffer = MakeSafetensors(R"({"__metadata__":{"format":"pt","source":"tabfm"},)"
	                              R"("t":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})",
	                              "");
	auto view = Parse(buffer);
	REQUIRE(view.metadata.size() == 2);
	REQUIRE(view.metadata.at("format") == "pt");
	REQUIRE(view.metadata.at("source") == "tabfm");
	REQUIRE(view.tensors.size() == 1);
}

TEST_CASE("safetensors: empty tensor (zero-sized dim) is valid", "[tabfm][safetensors]") {
	auto buffer = MakeSafetensors(R"({"e":{"dtype":"F32","shape":[2,0,3],"data_offsets":[0,0]}})", "");
	auto view = Parse(buffer);
	auto &info = view.Get("e");
	REQUIRE(info.ElementCount() == 0);
	REQUIRE(info.nbytes == 0);
	REQUIRE(info.shape == vector<int64_t> {2, 0, 3});
}

TEST_CASE("safetensors: scalar tensor (rank 0) is valid", "[tabfm][safetensors]") {
	string data;
	Append<int64_t>(data, 99);
	auto buffer = MakeSafetensors(R"({"s":{"dtype":"I64","shape":[],"data_offsets":[0,8]}})", data);
	auto view = Parse(buffer);
	auto &info = view.Get("s");
	REQUIRE(info.shape.empty());
	REQUIRE(info.ElementCount() == 1);
	REQUIRE(info.nbytes == 8);
}

TEST_CASE("safetensors: overflowing shape product is rejected, not silently wrapped", "[tabfm][safetensors]") {
	// A crafted header could declare dimensions whose idx_t product wraps to a
	// small value matching a tiny data_offsets range, smuggling a huge shape past
	// validation over a too-small backing buffer (OOB reads downstream). The
	// checked product must reject it instead.
	auto buffer =
	    MakeSafetensors(R"({"w":{"dtype":"F32","shape":[9223372036854775807,4],"data_offsets":[0,16]}})", string(16, '\0'));
	REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
	REQUIRE_THROWS_WITH(Parse(buffer), Contains("overflow"));
}

TEST_CASE("safetensors: error paths", "[tabfm][safetensors]") {
	SECTION("truncated header length") {
		string tiny = "\x01\x02\x03";
		REQUIRE_THROWS_AS(Parse(tiny), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(tiny), Contains("truncated"));
	}
	SECTION("header length beyond file") {
		string buffer;
		Append<uint64_t>(buffer, 1000); // claims 1000-byte header, file has 4 more bytes
		buffer += "{}xx";
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("header length"));
	}
	SECTION("malformed JSON header") {
		auto buffer = MakeSafetensors(R"({"w": {"dtype": )", "");
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("JSON"));
	}
	SECTION("header is not a JSON object") {
		auto buffer = MakeSafetensors(R"([1,2,3])", "");
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("object"));
	}
	SECTION("unsupported dtype") {
		string data(2, '\0');
		auto buffer = MakeSafetensors(R"({"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}})", data);
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("F16") && Contains("F32, BF16, I64"));
	}
	SECTION("byte size does not match shape * dtype size") {
		string data(8, '\0');
		auto buffer = MakeSafetensors(R"({"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", data);
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("12") && Contains("8"));
	}
	SECTION("data_offsets out of bounds") {
		string data(4, '\0');
		auto buffer = MakeSafetensors(R"({"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})", data);
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("out of bounds") || Contains("beyond"));
	}
	SECTION("data_offsets begin > end") {
		string data(8, '\0');
		auto buffer = MakeSafetensors(R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}})", data);
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
	}
	SECTION("overlapping tensor ranges") {
		string data(12, '\0');
		auto buffer = MakeSafetensors(R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
		                              R"("b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})",
		                              data);
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
		REQUIRE_THROWS_WITH(Parse(buffer), Contains("overlap"));
	}
	SECTION("negative shape dimension") {
		string data(8, '\0');
		auto buffer = MakeSafetensors(R"({"w":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})", data);
		REQUIRE_THROWS_AS(Parse(buffer), InvalidInputException);
	}
	SECTION("missing dtype / shape / data_offsets fields") {
		REQUIRE_THROWS_AS(Parse(MakeSafetensors(R"({"w":{"shape":[1],"data_offsets":[0,4]}})", string(4, '\0'))),
		                  InvalidInputException);
		REQUIRE_THROWS_AS(Parse(MakeSafetensors(R"({"w":{"dtype":"F32","data_offsets":[0,4]}})", string(4, '\0'))),
		                  InvalidInputException);
		REQUIRE_THROWS_AS(Parse(MakeSafetensors(R"({"w":{"dtype":"F32","shape":[1]}})", string(4, '\0'))),
		                  InvalidInputException);
	}
	SECTION("error message names the source") {
		string tiny = "abc";
		REQUIRE_THROWS_WITH(Parse(tiny, "weights.safetensors"), Contains("weights.safetensors"));
	}
	SECTION("Get on unknown tensor") {
		auto buffer = MakeSafetensors(R"({"w":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})", "");
		auto view = Parse(buffer);
		REQUIRE(!view.Has("nope"));
		REQUIRE_THROWS_AS(view.Get("nope"), InvalidInputException);
		REQUIRE_THROWS_WITH(view.Get("nope"), Contains("nope"));
	}
}

TEST_CASE("safetensors: bf16 -> f32 upcast is a 16-bit shift into the high half", "[tabfm][safetensors]") {
	REQUIRE(Bf16ToF32(0x0000) == 0.0f);
	REQUIRE(Bf16ToF32(0x3F80) == 1.0f);
	REQUIRE(Bf16ToF32(0xBF80) == -1.0f);
	REQUIRE(Bf16ToF32(0x4000) == 2.0f);
	REQUIRE(Bf16ToF32(0x4049) == 3.140625f); // bf16-rounded pi
	REQUIRE(Bf16ToF32(0xC0A0) == -5.0f);
}

TEST_CASE("safetensors: f32 arena materialization", "[tabfm][safetensors]") {
	string data;
	// f32 tensor
	Append<float>(data, 1.5f);
	Append<float>(data, -3.0f);
	// bf16 tensor
	Append<uint16_t>(data, 0x3F80); // 1.0
	Append<uint16_t>(data, 0x4049); // 3.140625
	Append<uint16_t>(data, 0xC0A0); // -5.0
	Append<uint16_t>(data, 0x0000); // 0.0
	// i64 tensor
	Append<int64_t>(data, 123);
	auto buffer = MakeSafetensors(R"({"f":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
	                              R"("b":{"dtype":"BF16","shape":[4],"data_offsets":[8,16]},)"
	                              R"("i":{"dtype":"I64","shape":[1],"data_offsets":[16,24]}})",
	                              data);
	auto view = Parse(buffer);
	auto arena = MaterializeF32Arena(view);
	REQUIRE(arena.tensors.size() == 3);

	SECTION("F32 passes through without a copy") {
		auto &f = arena.Get("f");
		REQUIRE(f.dtype == SafetensorsDtype::F32);
		REQUIRE(f.source_dtype == SafetensorsDtype::F32);
		REQUIRE(f.data == view.Get("f").data); // same pointer: no copy
		REQUIRE(f.nbytes == 8);
	}
	SECTION("BF16 is upcast into the arena with exact values") {
		auto &b = arena.Get("b");
		REQUIRE(b.dtype == SafetensorsDtype::F32);
		REQUIRE(b.source_dtype == SafetensorsDtype::BF16);
		REQUIRE(b.nbytes == 16); // 4 elements * 4 bytes after upcast
		REQUIRE(b.data != view.Get("b").data);
		float values[4];
		std::memcpy(values, b.data, sizeof(values));
		REQUIRE(values[0] == 1.0f);
		REQUIRE(values[1] == 3.140625f);
		REQUIRE(values[2] == -5.0f);
		REQUIRE(values[3] == 0.0f);
	}
	SECTION("I64 is exposed as-is") {
		auto &i = arena.Get("i");
		REQUIRE(i.dtype == SafetensorsDtype::I64);
		REQUIRE(i.source_dtype == SafetensorsDtype::I64);
		REQUIRE(i.data == view.Get("i").data);
		int64_t value;
		std::memcpy(&value, i.data, sizeof(value));
		REQUIRE(value == 123);
	}
	SECTION("arena owns exactly the upcast bytes and is movable") {
		REQUIRE(arena.ArenaBytes() == 16); // only the bf16 tensor is copied
		auto moved = std::move(arena);
		REQUIRE(moved.tensors.size() == 3);
		float first;
		std::memcpy(&first, moved.Get("b").data, sizeof(first));
		REQUIRE(first == 1.0f); // pointers stable across the move
		REQUIRE_THROWS_AS(moved.Get("nope"), InvalidInputException);
	}
}

TEST_CASE("safetensors: WS-A fixture file parses when present", "[tabfm][safetensors]") {
	const char *candidates[] = {"test/fixtures/model.safetensors", "../test/fixtures/model.safetensors"};
	string path;
	for (auto candidate : candidates) {
		std::ifstream probe(candidate, std::ios::binary);
		if (probe.good()) {
			path = candidate;
			break;
		}
	}
	if (path.empty()) {
		WARN("test/fixtures/model.safetensors not present (WS-A not landed yet) - skipping");
		SUCCEED();
		return;
	}
	std::ifstream in(path, std::ios::binary);
	string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	REQUIRE(contents.size() > 8);

	auto view = ParseSafetensors(const_data_ptr_cast(contents.data()), contents.size(), path);
	REQUIRE(!view.tensors.empty());
	// Every tensor must materialize; fixture is random-init but structurally real.
	auto arena = MaterializeF32Arena(view);
	REQUIRE(arena.tensors.size() == view.tensors.size());
}
