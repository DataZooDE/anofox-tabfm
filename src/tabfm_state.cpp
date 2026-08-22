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

namespace {
//! Inner-map key: a session is identified by what shaped it. '\x1f' cannot
//! appear in a device id or a precision name, so the join is unambiguous.
string SessionKey(const string &device_id, const string &precision) {
	return device_id + '\x1f' + precision;
}
} // anonymous namespace

void TabFMState::Register(const string &key, shared_ptr<LoadedModel> model, idx_t max_sessions) {
	lock_guard<mutex> guard(lock);
	auto session_key = SessionKey(model->device_id, model->precision);
	auto &sessions = models[key];
	auto entry = sessions.find(session_key);
	if (entry != sessions.end()) {
		// Same-configuration replacement: the old session dies when its last
		// snapshot releases.
		entry->second->evicted = true;
	}
	model->model_key = key;
	model->sequence = ++next_sequence;
	sessions[session_key] = std::move(model);

	if (max_sessions == 0) {
		return;
	}
	// Cap the TOTAL loaded sessions (all models, all configurations): evict
	// oldest-first until under the cap, never the entry just registered.
	// Multi-GB sessions must not accumulate behind the user's back now that
	// one model can hold several.
	while (true) {
		idx_t total = 0;
		for (auto &m : models) {
			total += m.second.size();
		}
		if (total <= max_sessions) {
			return;
		}
		map<string, map<string, shared_ptr<LoadedModel>>>::iterator oldest_model = models.end();
		map<string, shared_ptr<LoadedModel>>::iterator oldest_session;
		idx_t oldest_sequence = next_sequence + 1;
		for (auto m = models.begin(); m != models.end(); ++m) {
			for (auto sess = m->second.begin(); sess != m->second.end(); ++sess) {
				if (sess->second->sequence < oldest_sequence) {
					oldest_sequence = sess->second->sequence;
					oldest_model = m;
					oldest_session = sess;
				}
			}
		}
		if (oldest_model == models.end() || oldest_sequence == next_sequence) {
			return; // only the fresh entry remains; the cap cannot evict it
		}
		oldest_session->second->evicted = true;
		oldest_model->second.erase(oldest_session);
		if (oldest_model->second.empty()) {
			models.erase(oldest_model);
		}
	}
}

shared_ptr<LoadedModel> TabFMState::Snapshot(const string &key, const string &device_id,
                                             const string &precision) const {
	lock_guard<mutex> guard(lock);
	auto entry = models.find(key);
	if (entry == models.end()) {
		return nullptr;
	}
	auto session = entry->second.find(SessionKey(device_id, precision));
	if (session == entry->second.end()) {
		return nullptr;
	}
	return session->second;
}

vector<shared_ptr<LoadedModel>> TabFMState::SnapshotsFor(const string &key) const {
	lock_guard<mutex> guard(lock);
	vector<shared_ptr<LoadedModel>> result;
	auto entry = models.find(key);
	if (entry == models.end()) {
		return result;
	}
	for (auto &session : entry->second) { // std::map iterates sorted
		result.push_back(session.second);
	}
	return result;
}

bool TabFMState::Unload(const string &key) {
	lock_guard<mutex> guard(lock);
	auto entry = models.find(key);
	if (entry == models.end()) {
		return false;
	}
	// Mark every configuration evicted, then drop our references. In-flight
	// predicts holding a snapshot keep their session alive until they release
	// (HLD §6: no torn sessions); each ORT session is destroyed with its last
	// shared_ptr. All-or-nothing so one name reaches everything.
	for (auto &session : entry->second) {
		session.second->evicted = true;
	}
	models.erase(entry);
	return true;
}

idx_t TabFMState::UnloadAll() {
	lock_guard<mutex> guard(lock);
	const idx_t count = models.size();
	for (auto &entry : models) {
		for (auto &session : entry.second) {
			session.second->evicted = true;
		}
	}
	models.clear();
	return count;
}

vector<string> TabFMState::LoadedKeys() const {
	lock_guard<mutex> guard(lock);
	vector<string> keys;
	keys.reserve(models.size());
	for (auto &entry : models) {
		keys.push_back(entry.first); // one per model key; std::map iterates sorted
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
