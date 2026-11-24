/*===----------------------------------------------------------------------===
 * A backend plugin that computes nothing real, so tabfm's side of the plugin boundary can be tested without a GPU.
 *
 * It speaks anofox-ggml-adapter's plugin ABI v3 like the real plugins, and returns a deterministic function of the named inputs, so a
 * test asserts the data made the round trip (including the weights tabfm mapped and handed over) rather than that nothing threw:
 *
 *   logits[0][row][c] = x[row][0] + 100 * cat_mask[0] + c + train_size + (number of weights supplied > 0 ? 1000 : 0)
 *
 * arch "refuse" makes create fail. Built twice; TABFM_FAKE_PLUGIN_BAD_ABI builds it against another ABI version.
 *===----------------------------------------------------------------------===*/

#include "anofox_ggml_adapter/plugin_abi.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct FakeBackend {
	size_t weights = 0;
};

void SetError(char *err, size_t err_len, const char *message) {
	if (err && err_len) {
		std::strncpy(err, message, err_len - 1);
		err[err_len - 1] = '\0';
	}
}

const char *PluginName(void) {
	return "fake";
}

void *PluginCreate(const AnofoxGgmlCreateParams *params, char *err, size_t err_len) {
	if (!params) {
		SetError(err, err_len, "no params");
		return nullptr;
	}
	if (params->arch && std::strcmp(params->arch, "refuse") == 0) {
		SetError(err, err_len, "no device matching arch 'refuse'");
		return nullptr;
	}
	auto *b = new FakeBackend();
	b->weights = params->weight_count;
	return b;
}

const char *const *PluginGraphInputs(void *) {
	static const char *const inputs[] = {"x", "y", "cat_mask", "train_size", "d", nullptr};
	return inputs;
}

const AnofoxGgmlTensorIn *Find(const AnofoxGgmlTensorIn *inputs, size_t n, const char *name) {
	for (size_t i = 0; i < n; i++) {
		if (std::strcmp(inputs[i].name, name) == 0) {
			return &inputs[i];
		}
	}
	return nullptr;
}

AnofoxGgmlPluginStatus PluginRun(void *handle, const AnofoxGgmlTensorIn *inputs, size_t input_count, const char *const *outputs,
                                 size_t output_count, AnofoxGgmlRunOutput *result, char *err, size_t err_len) {
	auto *b = static_cast<FakeBackend *>(handle);
	result->tensors = nullptr;
	result->count = 0;
	const auto *x = Find(inputs, input_count, "x");
	const auto *mask = Find(inputs, input_count, "cat_mask");
	const auto *ts = Find(inputs, input_count, "train_size");
	if (!x || x->rank != 3 || !mask || mask->dtype != ANOFOX_GGML_DTYPE_BOOL || !ts || !ts->fold || output_count != 1) {
		SetError(err, err_len, "expected x [1,t,h], a bool cat_mask, a folded train_size and one output");
		return ANOFOX_GGML_PLUGIN_ERROR;
	}
	const int64_t t = x->shape[1], h = x->shape[2];
	const auto *xv = static_cast<const float *>(x->data);
	const float cat = static_cast<const uint8_t *>(mask->data)[0] ? 100.0f : 0.0f;
	const float train_size = static_cast<float>(*static_cast<const int64_t *>(ts->data));
	const float weights = b->weights > 0 ? 1000.0f : 0.0f;
	result->tensors = static_cast<AnofoxGgmlTensorOut *>(std::calloc(1, sizeof(AnofoxGgmlTensorOut)));
	auto &out = result->tensors[0];
	out.name = static_cast<char *>(std::malloc(std::strlen(outputs[0]) + 1));
	std::strcpy(out.name, outputs[0]);
	out.rank = 3;
	out.shape = static_cast<int64_t *>(std::malloc(3 * sizeof(int64_t)));
	out.shape[0] = 1;
	out.shape[1] = t;
	out.shape[2] = 3;
	out.count = t * 3;
	out.data = static_cast<float *>(std::malloc(static_cast<size_t>(out.count) * sizeof(float)));
	for (int64_t row = 0; row < t; row++) {
		for (int64_t c = 0; c < 3; c++) {
			out.data[row * 3 + c] = xv[row * h] + cat + static_cast<float>(c) + train_size + weights;
		}
	}
	result->count = 1;
	return ANOFOX_GGML_PLUGIN_OK;
}

void PluginFreeOutput(AnofoxGgmlRunOutput *result) {
	if (!result || !result->tensors) {
		return;
	}
	for (size_t i = 0; i < result->count; i++) {
		std::free(result->tensors[i].name);
		std::free(result->tensors[i].shape);
		std::free(result->tensors[i].data);
	}
	std::free(result->tensors);
	result->tensors = nullptr;
	result->count = 0;
}

void PluginDestroy(void *handle) {
	delete static_cast<FakeBackend *>(handle);
}

#ifdef TABFM_FAKE_PLUGIN_BAD_ABI
constexpr int kAbi = ANOFOX_GGML_PLUGIN_ABI_VERSION + 1;
#else
constexpr int kAbi = ANOFOX_GGML_PLUGIN_ABI_VERSION;
#endif

const AnofoxGgmlPluginApi kApi = {kAbi, ANOFOX_GGML_PLUGIN_LAYOUT, PluginName, PluginCreate, PluginGraphInputs, PluginRun, PluginFreeOutput, PluginDestroy};

} // namespace

extern "C" ANOFOX_GGML_PLUGIN_EXPORT const AnofoxGgmlPluginApi *AnofoxGgmlGetPluginApi(void) {
	return &kApi;
}
