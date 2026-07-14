#pragma once

//===----------------------------------------------------------------------===//
// tabfm_state.hpp — DB-instance-level model state (WS-C, HLD §6)
//
// TabFMState lives in DuckDB's ObjectCache (one per DatabaseInstance, shared
// by every connection) under the key "anofox_tabfm_state". It maps a model
// key (task or task@revision) to a LoadedModel holding an opaque engine
// session handle.
//
// Concurrency contract (HLD §6):
//   - predicts take a shared_ptr snapshot of the LoadedModel and run against
//     it without holding the state lock;
//   - unload marks the model evicted and drops it from the map — the actual
//     free happens when the last snapshot holder releases (no torn sessions);
//   - finalize-time forward passes are serialized per device via
//     DeviceMutex() (parallel groups accumulate concurrently, the expensive
//     ORT Run is one-at-a-time per device).
//===----------------------------------------------------------------------===//

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/map.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/storage/object_cache.hpp"

#include "tabfm_model_spec.hpp"

namespace duckdb {

class ClientContext;
class DatabaseInstance;

namespace anofox {

//! One loaded model. `session` is deliberately an opaque shared_ptr<void>
//! (holds a SessionHolder with the TabFMSessionHandle) so the state layer never
//! depends on ORT headers; the safetensors arena behind it is kept alive by
//! that holder for the whole session lifetime — ORT reads the injected weight
//! buffers lazily during inference, so they must outlive the session.
struct LoadedModel {
	//! Model key, e.g. "classification" or "classification@main"
	string model_key;
	//! Opaque engine session (TabFMSessionHandle stored type-erased)
	shared_ptr<void> session;
	//! Resolved device the session runs on ("cpu", "cuda:0", ...)
	string device_id;
	//! Weight dtype loaded into the session ("f32", "f16", "bf16")
	string dtype;
	//! Resident weight bytes (for tabfm_models() reporting)
	idx_t bytes = 0;
	//! Set by Unload; snapshot holders may finish their forward, new
	//! snapshots will not see this model anymore.
	atomic<bool> evicted {false};
};

//! Canonical loaded-model key for a (model, task, revision). The engine (which
//! registers sessions during predict) and the lifecycle SQL (tabfm_models /
//! tabfm_unload, which report and free them) MUST agree on this format, so both
//! build the key here. Format: "<model>:<task>@<revision>".
string TabFMModelCacheKey(const string &model, const string &task_name, const string &revision);

class TabFMState : public ObjectCacheEntry {
public:
	static constexpr const char *OBJECT_CACHE_KEY = "anofox_tabfm_state";

	static string ObjectType() {
		return OBJECT_CACHE_KEY;
	}
	string GetObjectType() override {
		return ObjectType();
	}
	//! Not evictable by the LRU: weights lifecycle is user-controlled via
	//! tabfm_load/tabfm_unload, never dropped behind the user's back.
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}

	//! The per-database-instance state (ObjectCache GetOrCreate).
	static shared_ptr<TabFMState> Get(DatabaseInstance &db);
	static shared_ptr<TabFMState> Get(ClientContext &context);

	//! Register a freshly created model under `key` (tabfm_load). Replacing
	//! an existing entry marks the old one evicted first.
	void Register(const string &key, shared_ptr<LoadedModel> model);
	//! Snapshot for a predict: shared ownership, nullptr if not loaded.
	shared_ptr<LoadedModel> Snapshot(const string &key) const;
	//! Unload one model: marks evicted + drops from the map. Returns false if
	//! the key was not loaded. The session is freed when the last snapshot
	//! holder releases.
	bool Unload(const string &key);
	//! Unload everything; returns the number of models dropped.
	idx_t UnloadAll();
	//! Keys currently loaded (sorted, for tabfm_models()).
	vector<string> LoadedKeys() const;
	//! Serialize finalize-time forward passes per device (HLD §6). The
	//! returned mutex lives as long as this state object.
	mutex &DeviceMutex(const string &device_id);

	//! Models registered in SQL (CALL tabfm_register_model). They are merged into
	//! the registry alongside the built-ins; a registered id shadows a built-in of
	//! the same id. Lives for the database-instance lifetime.
	void RegisterModelSpec(const ModelSpec &spec);
	//! Drop a registered model; false if the id was not registered.
	bool UnregisterModelSpec(const string &id);
	//! Snapshot of all SQL-registered specs (sorted by id).
	vector<ModelSpec> RegisteredSpecs() const;

private:
	mutable mutex lock;
	map<string, shared_ptr<LoadedModel>> models;
	map<string, unique_ptr<mutex>> device_mutexes;
	map<string, ModelSpec> registered_specs;
};

} // namespace anofox
} // namespace duckdb
