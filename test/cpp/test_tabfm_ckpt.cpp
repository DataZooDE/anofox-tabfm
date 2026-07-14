#include "catch.hpp"

#include "tabfm_ckpt.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace duckdb;
using namespace duckdb::anofox;

namespace {
std::vector<uint8_t> ReadFile(const char *path) {
	FILE *f = fopen(path, "rb");
	REQUIRE(f != nullptr);
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<uint8_t> buf(n);
	REQUIRE(fread(buf.data(), 1, n, f) == (size_t)n);
	fclose(f);
	return buf;
}
} // namespace

TEST_CASE("tabfm_ckpt: native torch .ckpt reader recovers the state_dict", "[tabfm][ckpt]") {
	auto buf = ReadFile("test/fixtures/ckpt/tiny.ckpt");
	REQUIRE(IsTorchCkpt(buf.data(), buf.size()));

	auto sd = ReadTorchCkpt(buf.data(), buf.size());
	REQUIRE(sd.size() == 3);
	REQUIRE(sd.count("layer.weight"));
	REQUIRE(sd.count("layer.bias"));
	REQUIRE(sd.count("idx"));

	// f32 [2,3] = 0..5
	auto &w = sd.at("layer.weight");
	REQUIRE(w.dtype == "f32");
	REQUIRE(w.shape == vector<int64_t> {2, 3});
	REQUIRE(w.nbytes == 6 * sizeof(float));
	const float *wf = reinterpret_cast<const float *>(w.data);
	for (int i = 0; i < 6; i++) {
		REQUIRE(wf[i] == Approx((float)i));
	}

	// f32 [2] = {1.5, -2.5}
	auto &b = sd.at("layer.bias");
	REQUIRE(b.dtype == "f32");
	REQUIRE(b.shape == vector<int64_t> {2});
	const float *bf = reinterpret_cast<const float *>(b.data);
	REQUIRE(bf[0] == Approx(1.5f));
	REQUIRE(bf[1] == Approx(-2.5f));

	// i64 [1,3] = {7,8,9}
	auto &idx = sd.at("idx");
	REQUIRE(idx.dtype == "i64");
	REQUIRE(idx.shape == vector<int64_t> {1, 3});
	const int64_t *ii = reinterpret_cast<const int64_t *>(idx.data);
	REQUIRE(ii[0] == 7);
	REQUIRE(ii[1] == 8);
	REQUIRE(ii[2] == 9);
}

TEST_CASE("tabfm_ckpt: non-zip buffer is rejected", "[tabfm][ckpt]") {
	const char *garbage = "not a zip file at all";
	REQUIRE_FALSE(IsTorchCkpt(reinterpret_cast<const uint8_t *>(garbage), strlen(garbage)));
}
