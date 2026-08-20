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
	//! Effective GPU numeric mode this session was built with ("fp32"/"tf32"/
	//! "bf16"/"fp16"), and "" for CPU sessions — anofox_tabfm_gpu_precision
	//! does not shape those, so flipping it must not rebuild them.
	string precision;
	//! Weight dtype loaded into the session ("f32", "f16", "bf16")
	string dtype;
	//! Resident weight bytes (for tabfm_models() reporting)
	idx_t bytes = 0;
	//! Loaded as a split (prepare/query) pair with a cached labelled context,
	//! rather than as the single combined graph. Recorded so that flipping
	//! anofox_tabfm_context_cache rebuilds the session instead of silently
	//! answering from whichever backend happened to be cached first. The same
	//! argument applies to device_id above, which is why CanReuseSession below
	//! checks both — it used to check only this one.
	bool split_context = false;
	//! Set by Unload; snapshot holders may finish their forward, new
	//! snapshots will not see this model anymore.
	atomic<bool> evicted {false};
};

//! Whether a cached session can serve a request, or must be rebuilt.
//!
//! Both conditions exist for the same reason: a session is built FOR a
//! particular device and a particular split/combined shape, so reusing it
//! whenever the cache key matches silently answers from whichever
//! configuration happened to be loaded first. The device half of that was a
//! real bug — a connection that ran anything on the cpu and then
//! SET anofox_tabfm_device='cuda' kept getting the cpu session, with no error
//! and no GPU, which is the "requested device quietly becomes CPU" outcome the
//! tier-4 contract in docs/DYNAMIC_BACKENDS.md exists to rule out.
//!
//! Pulled out of LoadOrGetSession as a pure function so it is testable without
//! a GPU: CI cannot run the cpu->cuda switch that exposed the bug, but it can
//! assert this predicate, which is where the mistake actually lived.
//!
//! `wanted_precision` joined in Track A (docs/GPU_HARDENING_PLAN.md P1/P2),
//! because the device story repeats one setting over: once gpu_precision
//! shapes the session (fp32 vs tf32 vs bf16 build different programs),
//! ignoring it here silently serves the old mode after a SET — the exact
//! failure shape 70a6800 fixed for the device. Callers pass "" for CPU
//! sessions, which the setting does not shape.
inline bool CanReuseSession(const LoadedModel &cached, bool want_split, const string &wanted_device,
                            const string &wanted_precision) {
	return cached.split_context == want_split && cached.device_id == wanted_device &&
	       cached.precision == wanted_precision;
}

//! Canonical loaded-model key for a (model, task, revision). The engine (which
//! registers sessions during predict) and the lifecycle SQL (tabfm_models /
//! tabfm_unload, which report and free them) MUST agree on this format, so both
//! build the key here. Format: "<model>:<task>@<revision>".
string TabFMModelCacheKey(const string &model, const string &task_name, const string &revision);

//! This process's resident memory, in bytes; 0 when it cannot be read (platform
//! not covered, or the read failed). 0 also disables every check built on it,
//! since there is nothing trustworthy to compare against.
//!
//! Lives here rather than beside one caller because two now need it: the
//! bind-time watchdog in tabfm_predict_agg.cpp and the per-forward accounting in
//! tabfm_engine.cpp.
idx_t CurrentProcessResidentBytes();

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
	//!
	//! **Pass a RESOLVED device id, never the raw setting string.** Two connections
	//! whose `anofox_tabfm_device` settings differ textually but resolve to the same
	//! physical device — 'auto' and 'cpu' being the everyday pair — share one backend
	//! under `resolved.cache_key`, so keying this on the setting hands them different
	//! mutexes and they stop excluding each other. `ResolvedDeviceCached()` in
	//! tabfm_engine.cpp is what callers should pass; it is there rather than here
	//! because resolving needs the ORT device probes and this layer deliberately does
	//! not depend on ORT headers.
	mutex &DeviceMutex(const string &device_id);

	//! Models registered in SQL (CALL tabfm_register_model). They are merged into
	//! the registry alongside the built-ins; a registered id shadows a built-in of
	//! the same id. Lives for the database-instance lifetime.
	void RegisterModelSpec(const ModelSpec &spec);
	//! Drop a registered model; false if the id was not registered.
	bool UnregisterModelSpec(const string &id);
	//! Snapshot of all SQL-registered specs (sorted by id).
	vector<ModelSpec> RegisteredSpecs() const;

	//! What one forward pass of this shape was last observed to add to resident
	//! memory, in bytes. `anofox_tabfm_max_memory` on its own is a watchdog on
	//! memory ALREADY held, so it cannot see a single call that is small at entry
	//! and enormous at exit; that is the case that gets OOM-killed. Recording what
	//! a shape actually cost lets the next call of the same shape be checked
	//! before it runs, with no model internals and no hardcoded constant.
	//!
	//! Deliberately keyed on the exact shape rather than interpolated: a measured
	//! cost for (model, device, T, H) is evidence, and anything else is a guess
	//! that could refuse work that would have succeeded.
	void RecordForwardCost(const string &model_key, const string &device_id, int64_t t, int64_t h,
	                       idx_t bytes);
	//! Bytes the last forward of this shape added, or 0 if none has been seen.
	idx_t ForwardCost(const string &model_key, const string &device_id, int64_t t, int64_t h) const;

private:
	mutable mutex lock;
	map<string, shared_ptr<LoadedModel>> models;
	map<string, unique_ptr<mutex>> device_mutexes;
	map<string, ModelSpec> registered_specs;
	//! "<model_key>|<device>|<T>x<H>" -> observed resident-memory growth.
	map<string, idx_t> forward_costs;
};

} // namespace anofox
} // namespace duckdb
