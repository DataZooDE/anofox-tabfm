/*===----------------------------------------------------------------------===
 * A backend plugin that computes nothing, so the LOADER can be tested.
 *
 * The real backend behind this ABI needs a GPU, which CI does not have. That
 * would leave the loader — the trust boundary, where an ABI mismatch is
 * undefined behaviour rather than a wrong answer — exercised only by hand on
 * one developer's machine. This stands in for it: a deterministic function of
 * the inputs, so the test can assert the result actually made the round trip
 * rather than merely that nothing threw.
 *
 * Built twice, with TABFM_FAKE_PLUGIN_BAD_ABI toggling the version, so the
 * refusal path is tested with a genuinely mismatched library rather than a
 * mocked one.
 *===----------------------------------------------------------------------===*/

#include "tabfm_plugin_abi.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

struct FakeBackend {
	std::string precision;
	int64_t precompiled_rows = 0;
	int64_t precompiled_features = 0;
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

void *PluginCreate(const TabFMPluginCreateParams *params, char *err, size_t err_len) {
	if (!params || !params->graph_path) {
		SetError(err, err_len, "no graph path supplied");
		return nullptr;
	}
	// A create failure has to be testable too: this arch is the trigger.
	if (params->arch && std::strcmp(params->arch, "refuse") == 0) {
		SetError(err, err_len, "no device matching arch 'refuse'");
		return nullptr;
	}
	auto *backend = new FakeBackend();
	backend->precision = params->precision ? params->precision : "";
	return backend;
}

TabFMPluginStatus PluginRun(void *handle, const TabFMPluginRunInput *input, TabFMPluginRunOutput *output, char *err,
                            size_t err_len) {
	auto *backend = static_cast<FakeBackend *>(handle);
	if (!backend || !input || !output) {
		SetError(err, err_len, "null argument");
		return TABFM_PLUGIN_ERROR;
	}
	if (input->t <= 0 || input->h <= 0) {
		SetError(err, err_len, "non-positive t or h");
		return TABFM_PLUGIN_ERROR;
	}

	// logits[1, t, 3], a deterministic function of the inputs so the test can
	// tell a real round trip from a zero-filled buffer.
	const int64_t classes = 3;
	const int64_t count = input->t * classes;
	output->logits = static_cast<float *>(std::malloc(sizeof(float) * (size_t)count));
	output->logits_len = count;
	output->shape = static_cast<int64_t *>(std::malloc(sizeof(int64_t) * 3));
	output->shape_len = 3;
	if (!output->logits || !output->shape) {
		SetError(err, err_len, "allocation failed");
		return TABFM_PLUGIN_ERROR;
	}
	output->shape[0] = 1;
	output->shape[1] = input->t;
	output->shape[2] = classes;

	for (int64_t row = 0; row < input->t; row++) {
		// Reads x and the cat_mask so a mis-marshalled pointer shows up as a
		// wrong number rather than passing unnoticed.
		const float first_feature = input->x ? input->x[row * input->h] : 0.0f;
		const float mask_bias = (input->cat_mask && input->cat_mask[0]) ? 100.0f : 0.0f;
		for (int64_t c = 0; c < classes; c++) {
			output->logits[row * classes + c] = first_feature + mask_bias + (float)c + (float)input->train_size;
		}
	}
	return TABFM_PLUGIN_OK;
}

TabFMPluginStatus PluginPrecompile(void *handle, int64_t rows, int64_t features, char *err, size_t err_len) {
	auto *backend = static_cast<FakeBackend *>(handle);
	if (!backend) {
		SetError(err, err_len, "null handle");
		return TABFM_PLUGIN_ERROR;
	}
	backend->precompiled_rows = rows;
	backend->precompiled_features = features;
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
	delete static_cast<FakeBackend *>(handle);
}

const TabFMPluginApi kApi = {
#ifdef TABFM_FAKE_PLUGIN_BAD_ABI
    TABFM_PLUGIN_ABI_VERSION + 1000,
#else
    TABFM_PLUGIN_ABI_VERSION,
#endif
    PluginName, PluginCreate, PluginRun, PluginPrecompile, PluginFreeOutput, PluginDestroy,
};

} // namespace

extern "C" TABFM_PLUGIN_EXPORT const TabFMPluginApi *TabFMGetPluginApi(void) {
	return &kApi;
}
