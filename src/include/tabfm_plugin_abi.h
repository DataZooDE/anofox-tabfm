/*===----------------------------------------------------------------------===
 *                         anofox-tabfm
 *
 * tabfm_plugin_abi.h — the C ABI between the extension and a backend plugin.
 *
 * Phase 1 of docs/DYNAMIC_BACKENDS.md. A GPU backend is a shared library the
 * extension dlopens when the user asks for that device, not code compiled into
 * a separate build flavor. The shape is deliberately the one ONNX Runtime uses
 * for its own providers: one exported symbol returning a function table.
 *
 * Deliberately C, and deliberately narrow:
 *
 *   - no C++ types cross the boundary. The plugin and the extension are built
 *     as separate objects, potentially by different compilers; passing a
 *     std::vector across that means agreeing on an allocator and a layout, and
 *     the day that disagreement bites it does so as a corrupted heap rather
 *     than a link error.
 *   - errors come back as a status plus a caller-owned buffer, because a C++
 *     exception cannot cross a dlopen boundary safely either.
 *   - output buffers are owned by the plugin and released through the table, so
 *     whichever side allocated also frees.
 *
 * ABI stability: TABFM_PLUGIN_ABI_VERSION is checked on load and a mismatch is
 * refused with an actionable message. Bump it for ANY change to the structs or
 * the function signatures below — a plugin built against an older layout must
 * fail to load rather than read the wrong offsets.
 *===----------------------------------------------------------------------===*/

#ifndef ANOFOX_TABFM_PLUGIN_ABI_H
#define ANOFOX_TABFM_PLUGIN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TABFM_PLUGIN_ABI_VERSION 1

/*! The symbol every plugin must export. */
#define TABFM_PLUGIN_ENTRY_SYMBOL "TabFMGetPluginApi"

/*! Marks the entry point as exported from the plugin's shared library. On
 *  Windows a DLL exports nothing by default, so extern "C" alone leaves
 *  TabFMGetPluginApi invisible to GetProcAddress; ELF exports by default but
 *  the attribute keeps that true under -fvisibility=hidden. */
#if defined(_WIN32)
#define TABFM_PLUGIN_EXPORT __declspec(dllexport)
#else
#define TABFM_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/*! Status codes. Anything non-zero leaves *err populated. */
typedef enum {
	TABFM_PLUGIN_OK = 0,
	TABFM_PLUGIN_ERROR = 1,
} TabFMPluginStatus;

/*! One forward pass. Mirrors TabFMRunInput; buffers are borrowed and must
 *  outlive the call, not the handle. */
typedef struct {
	const float *x;        /* [1, t, h] */
	const float *y;        /* [1, t]    */
	const uint8_t *cat_mask; /* [1, h]; uint8 rather than bool — _Bool's size is
	                          * implementation-defined and this crosses a
	                          * compiler boundary */
	int64_t t;
	int64_t h;
	int64_t train_size;
	int64_t d;
} TabFMPluginRunInput;

/*! Result of a forward pass. `logits` and `shape` are owned by the PLUGIN and
 *  released with free_output; the caller must not free them itself. */
typedef struct {
	float *logits;
	int64_t logits_len;
	int64_t *shape;
	int64_t shape_len;
} TabFMPluginRunOutput;

/*! Everything a backend needs to construct itself. Strings are borrowed for the
 *  duration of the create call only. */
typedef struct {
	const char *graph_path;
	const char *weights_dir;
	const char *cache_dir;
	const char *arch;
	const char *precision;
	const char *mxr_source;
	int device_ordinal;
} TabFMPluginCreateParams;

/*! The function table. `abi_version` is first so a mismatched plugin can be
 *  rejected before anything else in the struct is read. */
typedef struct {
	int abi_version;

	/*! Human-readable backend name for diagnostics ("migraphx"). */
	const char *(*name)(void);

	/*! Construct a backend. Returns NULL on failure with *err populated. */
	void *(*create)(const TabFMPluginCreateParams *params, char *err, size_t err_len);

	/*! One forward pass. */
	TabFMPluginStatus (*run)(void *handle, const TabFMPluginRunInput *input, TabFMPluginRunOutput *output, char *err,
	                         size_t err_len);

	/*! Warm a shape bucket without predicting (MIGraphX compiles per bucket). */
	TabFMPluginStatus (*precompile)(void *handle, int64_t rows, int64_t features, char *err, size_t err_len);

	/*! Release a run output. Safe on a zeroed struct. */
	void (*free_output)(TabFMPluginRunOutput *output);

	/*! Destroy a handle from create. Safe on NULL. */
	void (*destroy)(void *handle);
} TabFMPluginApi;

/*! Signature of TABFM_PLUGIN_ENTRY_SYMBOL. */
typedef const TabFMPluginApi *(*TabFMGetPluginApiFn)(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ANOFOX_TABFM_PLUGIN_ABI_H */
