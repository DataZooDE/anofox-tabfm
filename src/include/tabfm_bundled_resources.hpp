//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_bundled_resources.hpp — accessor for the weight-free artifacts compiled
// into the extension binary (the .onnx graphs and their tensor maps from
// resources/). Bundling lets the built-in flow work with no companion files:
// after `tabfm_download` fetches the weights, the graph + tensor map come from
// the binary itself. The bytes are Google-weight-free (license wall: S01/S06).
//
// The generator is cmake/embed_resources.cmake (runs at configure time and
// emits tabfm_bundled_resources.cpp into the build tree). Resources are
// registered under the exact ids the built-in manifest uses
// ("graph_classification", "tensor_map_classification.json", ...); graphs are
// additionally registered under their ".onnx" filename.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {
namespace anofox {

//! A read-only view of a resource compiled into the binary. `data` points at
//! static storage that lives for the whole process; do not free it.
struct BundledResource {
	const char *data = nullptr;
	idx_t size = 0;
};

//! The bundled resource registered under `id`, or {nullptr, 0} when unknown.
//! `id` is matched exactly against the registered ids (bundled graph ids such
//! as "graph_classification", their ".onnx" filenames, and tensor-map filenames
//! like "tensor_map_classification.json"). A filesystem path never matches, so
//! manifests that point `graph`/`tensor_map` at an on-disk file fall through to
//! the existing disk resolution.
BundledResource GetBundledResource(const string &id);

} // namespace anofox
} // namespace duckdb
