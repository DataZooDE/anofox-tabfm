//===----------------------------------------------------------------------===//
// tabfm_state.cpp — DB-instance-level model state (WS-C, HLD §6)
//===----------------------------------------------------------------------===//

#include "tabfm_state.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

#if defined(__linux__)
#include <cinttypes>
#include <cstdio>
#elif defined(_WIN32)
#include <windows.h>
// psapi.h must follow windows.h
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace duckdb {
namespace anofox {

string TabFMModelCacheKey(const string &model, const string &task_name, const string &revision) {
	return model + ":" + task_name + "@" + revision;
}

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

void TabFMState::RegisterModelSpec(const ModelSpec &spec) {
	lock_guard<mutex> guard(lock);
	registered_specs[spec.id] = spec;
}

bool TabFMState::UnregisterModelSpec(const string &id) {
	lock_guard<mutex> guard(lock);
	return registered_specs.erase(id) > 0;
}

vector<ModelSpec> TabFMState::RegisteredSpecs() const {
	lock_guard<mutex> guard(lock);
	vector<ModelSpec> out;
	out.reserve(registered_specs.size());
	for (auto &entry : registered_specs) {
		out.push_back(entry.second); // std::map iterates sorted by id
	}
	return out;
}

idx_t CurrentProcessResidentBytes() {
#if defined(__linux__)
	FILE *f = fopen("/proc/self/status", "r");
	if (!f) {
		return 0;
	}
	uint64_t kb = 0;
	bool found = false;
	char line[256];
	while (fgets(line, sizeof(line), f)) {
		if (std::sscanf(line, "VmRSS: %" SCNu64 " kB", &kb) == 1) {
			found = true;
			break;
		}
	}
	fclose(f);
	return found ? static_cast<idx_t>(kb) * 1024 : 0;
#elif defined(_WIN32)
	PROCESS_MEMORY_COUNTERS counters;
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
		return static_cast<idx_t>(counters.WorkingSetSize);
	}
	return 0;
#elif defined(__APPLE__)
	mach_task_basic_info_data_t info;
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
	if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
	    KERN_SUCCESS) {
		return static_cast<idx_t>(info.resident_size);
	}
	return 0;
#else
	return 0;
#endif
}

namespace {
string ForwardCostKey(const string &model_key, const string &device_id, int64_t t, int64_t h) {
	return model_key + "|" + device_id + "|" + std::to_string(t) + "x" + std::to_string(h);
}
} // namespace

void TabFMState::RecordForwardCost(const string &model_key, const string &device_id, int64_t t, int64_t h,
                                   idx_t bytes) {
	lock_guard<mutex> guard(lock);
	auto key = ForwardCostKey(model_key, device_id, t, h);
	auto entry = forward_costs.find(key);
	// The maximum, not the latest. Resident memory is sampled around the forward
	// and an allocator that reuses freed pages can make a later call of the same
	// shape look free; taking the max keeps the estimate on the safe side of a
	// check whose whole purpose is to refuse before the kernel does.
	//
	// It also means the first observation of a shape can carry one-off arena
	// warm-up and read high for the rest of the session. That biases towards
	// refusing, which is the direction to be wrong in here: a refusal is an
	// actionable error, and the alternative it exists to prevent is an
	// unattributable SIGKILL.
	if (entry == forward_costs.end() || entry->second < bytes) {
		forward_costs[key] = bytes;
	}
}

idx_t TabFMState::ForwardCost(const string &model_key, const string &device_id, int64_t t, int64_t h) const {
	lock_guard<mutex> guard(lock);
	auto entry = forward_costs.find(ForwardCostKey(model_key, device_id, t, h));
	return entry == forward_costs.end() ? 0 : entry->second;
}

} // namespace anofox
} // namespace duckdb
