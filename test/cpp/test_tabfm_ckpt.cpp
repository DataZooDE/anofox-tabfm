#include "catch.hpp"

#include "tabfm_ckpt.hpp"
#include "tabfm_predict.hpp"

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

// Synthefy Nori (nori.pt) is a training-style checkpoint: {"model_config": {...nested config...},
// "ema_state_dict": {...}}. Upstream's load_model reads the EMA weights (falling back to
// "model_state_dict"), so the reader must descend into "ema_state_dict" rather than treat the top-level
// dict -- which holds no tensors at all -- as the state_dict. The fixture also carries 0-dim tensors (the
// checkpoint's `column_y_aware_alpha` / `deepnorm_alpha` scalars) and a config with nested dicts,
// floats, bools and None (the released file has no lists). Built with torch.save from tiny random values (no Synthefy bytes).
TEST_CASE("tabfm_ckpt: an ema_state_dict checkpoint (Nori) yields the EMA weights", "[tabfm][ckpt]") {
	auto buf = ReadFile("test/fixtures/ckpt/tiny_ema.ckpt");
	REQUIRE(IsTorchCkpt(buf.data(), buf.size()));

	auto sd = ReadTorchCkpt(buf.data(), buf.size());
	REQUIRE(sd.size() == 3);
	REQUIRE(sd.count("layer.weight"));
	REQUIRE(sd.count("alpha"));
	REQUIRE(sd.count("layers.0.deepnorm_alpha"));

	auto &w = sd.at("layer.weight");
	REQUIRE(w.dtype == "f32");
	REQUIRE(w.shape == vector<int64_t> {2, 3});
	const float *wf = reinterpret_cast<const float *>(w.data);
	for (int i = 0; i < 6; i++) {
		REQUIRE(wf[i] == Approx((float)i));
	}

	auto &a = sd.at("alpha");
	REQUIRE(a.dtype == "f32");
	REQUIRE(a.shape.empty());
	REQUIRE(a.nbytes == sizeof(float));
	REQUIRE(*reinterpret_cast<const float *>(a.data) == Approx(1.25f));
	REQUIRE(*reinterpret_cast<const float *>(sd.at("layers.0.deepnorm_alpha").data) == Approx(0.5f));
}

// The GPU graphs of a checkpoint-shipped model (Nori, Kumo) read their weights as ONNX external data at
// offsets baked against ONE file: what tools/export_*/convert_*_weights writes (every tensor, keys sorted,
// contiguous float32, no __metadata__). tabfm_download now writes that file itself, so it must be the same
// file byte for byte -- the references are the Python converter's output for these fixtures.
TEST_CASE("tabfm_ckpt: a checkpoint converts to the converter's safetensors, byte for byte", "[tabfm][ckpt]") {
	struct Case {
		const char *ckpt;
		const char *file_sha256;
	};
	for (auto &c : {Case {"test/fixtures/ckpt/tiny.ckpt",
	                      "0bc1ab53774e0b709a8e8a31e0f402d9167d38dd8935712bc2585b9785a593ac"}, // i64 tensor, padded
	                Case {"test/fixtures/ckpt/tiny_ema.ckpt",
	                      "51966197e7a5767686dd3841ed412dd9b5038d56018970c4dd541e6ce715f933"}, // 0-dim tensors
	                Case {"test/fixtures/ckpt/tiny_strided.ckpt",
	                      "69af242a835eaa7aba93932367de1570a2d6822c542620aab06d60f810fb695c"}, // transposed view
	                Case {"test/fixtures/nori/model.ckpt",
	                      "34753212675d8fce6bb998e8e4751453d61492b6dc6f7f5f9c51f3e9907bc91d"}}) {
		INFO(c.ckpt);
		auto buf = ReadFile(c.ckpt);
		auto sd = ReadTorchCkpt(buf.data(), buf.size());
		std::string out;
		WriteCkptAsSafetensors(sd, [&](const void *data, idx_t n) { out.append(static_cast<const char *>(data), n); });
		REQUIRE(Sha256Hex(const_data_ptr_cast(out.data()), out.size()) == c.file_sha256);
	}
}

// What WeightsHeaderMatches hashes: the JSON header between the length prefix and the data.
TEST_CASE("tabfm_ckpt: the converted header is computable without writing the file", "[tabfm][ckpt]") {
	auto buf = ReadFile("test/fixtures/nori/model.ckpt");
	auto sd = ReadTorchCkpt(buf.data(), buf.size());
	auto header = CkptSafetensorsHeader(sd);
	REQUIRE(header.size() == 11040);
	REQUIRE(Sha256Hex(const_data_ptr_cast(header.data()), header.size()) ==
	        "40c9c1c626e161b980fea82c53f1dd77e49c15dc095e4335a8286a5e68538fcc");
}

// torch.save keeps a view's strides: NVIDIA's Kumo-Tabular .pt stores both y_emb.weight tensors transposed
// (shape [10, 128], stride (1, 10)). Handing out the raw storage fed every consumer -- the CPU injection
// path included -- those weights in the wrong element order; on a cache holding only the .pt, Kumo-small
// classified at 28% instead of 94%. The reader must return a view's elements in index order, as torch does.
TEST_CASE("tabfm_ckpt: a strided view is read in index order, as torch reads it", "[tabfm][ckpt]") {
	auto buf = ReadFile("test/fixtures/ckpt/tiny_strided.ckpt");
	auto sd = ReadTorchCkpt(buf.data(), buf.size());
	auto &w = sd.at("w"); // torch.arange(6).reshape(2, 3).t()
	REQUIRE(w.shape == vector<int64_t> {3, 2});
	REQUIRE(w.IsContiguous());
	REQUIRE(w.nbytes == 6 * sizeof(float));
	const float *wf = reinterpret_cast<const float *>(w.data);
	const float expected[] = {0, 3, 1, 4, 2, 5};
	for (int i = 0; i < 6; i++) {
		REQUIRE(wf[i] == expected[i]);
	}
	REQUIRE(sd.at("b").IsContiguous());
}
