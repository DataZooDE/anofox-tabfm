//===----------------------------------------------------------------------===//
//                         anofox-tabfm
//
// tabfm_registry.hpp — the model registry (multi-model, FR-5.1 / M4). Holds N
// ModelSpecs keyed by id: the built-ins compiled into the binary (today the
// Google TabFM v1) plus models registered in SQL (CALL tabfm_register_model)
// (a file or a directory). One registry is the single source of truth for both
// the predict path and the weights functions.
//===----------------------------------------------------------------------===//

#pragma once

#include "tabfm_model_spec.hpp"

#include "duckdb/common/map.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace anofox {

//! The built-in models compiled into the binary (Google TabFM v1 today).
vector<ModelSpec> BuiltinModelSpecs();

class ModelRegistry {
public:
	//! Build the registry: the built-ins, plus `registered` models added in SQL
	//! (CALL tabfm_register_model, held in TabFMState). A registered id shadows a
	//! built-in of the same id; a single registration is the implicit default.
	static ModelRegistry Build(const vector<ModelSpec> &registered = {});

	const map<string, ModelSpec> &Models() const {
		return models_;
	}
	bool Has(const string &id) const {
		return models_.find(id) != models_.end();
	}
	//! Throws an actionable "unknown model" error (lists the registered ids).
	const ModelSpec &Get(const string &id) const;

	//! Resolve the active model: `requested` (per-call `model :=`, may be "") →
	//! `default_setting` (`anofox_tabfm_default_model`, may be "") → the implicit
	//! default (a single-file manifest source) → the sole registered model. An
	//! unknown id or an unresolvable ambiguity throws.
	const ModelSpec &Resolve(const string &requested, const string &default_setting) const;

	//! The id selected by a single-file `model_manifest` source ("" if none).
	const string &ImplicitDefault() const {
		return implicit_default_;
	}

	//! Comma-separated registered ids (sorted) — for error messages / discovery.
	string RegisteredIds() const;

private:
	map<string, ModelSpec> models_;
	string implicit_default_;
};

} // namespace anofox
} // namespace duckdb
