//===----------------------------------------------------------------------===//
// tabfm_state.cpp — DB-instance-level model state (WS-C, HLD §6)
//===----------------------------------------------------------------------===//

#include "tabfm_state.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {
namespace anofox {

shared_ptr<TabFMState> TabFMState::Get(DatabaseInstance &db) {
	return db.GetObjectCache().GetOrCreate<TabFMState>(OBJECT_CACHE_KEY);
}

shared_ptr<TabFMState> TabFMState::Get(ClientContext &context) {
	return ObjectCache::GetObjectCache(context).GetOrCreate<TabFMState>(OBJECT_CACHE_KEY);
}

void TabFMState::Register(const string &key, shared_ptr<LoadedModel> model) {
	lock_guard<mutex> guard(lock);
	auto entry = models.find(key);
	if (entry != models.end()) {
		// Replacement: the old session dies when its last snapshot releases.
		entry->second->evicted = true;
	}
	model->model_key = key;
	models[key] = std::move(model);
}

shared_ptr<LoadedModel> TabFMState::Snapshot(const string &key) const {
	lock_guard<mutex> guard(lock);
	auto entry = models.find(key);
	if (entry == models.end()) {
		return nullptr;
	}
	return entry->second;
}

bool TabFMState::Unload(const string &key) {
	lock_guard<mutex> guard(lock);
	auto entry = models.find(key);
	if (entry == models.end()) {
		return false;
	}
	// Mark evicted, then drop our reference. In-flight predicts holding a
	// snapshot keep the session alive until they release (HLD §6: no torn
	// sessions); the ORT session is destroyed with the last shared_ptr.
	entry->second->evicted = true;
	models.erase(entry);
	return true;
}

idx_t TabFMState::UnloadAll() {
	lock_guard<mutex> guard(lock);
	const idx_t count = models.size();
	for (auto &entry : models) {
		entry.second->evicted = true;
	}
	models.clear();
	return count;
}

vector<string> TabFMState::LoadedKeys() const {
	lock_guard<mutex> guard(lock);
	vector<string> keys;
	keys.reserve(models.size());
	for (auto &entry : models) {
		keys.push_back(entry.first); // std::map iterates sorted
	}
	return keys;
}

mutex &TabFMState::DeviceMutex(const string &device_id) {
	lock_guard<mutex> guard(lock);
	auto entry = device_mutexes.find(device_id);
	if (entry == device_mutexes.end()) {
		entry = device_mutexes.emplace(device_id, make_uniq<mutex>()).first;
	}
	return *entry->second;
}

} // namespace anofox
} // namespace duckdb
