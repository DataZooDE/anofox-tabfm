#include "tabfm_cpu_budget.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

namespace duckdb {
namespace anofox {

namespace {

//! Read a whole number, rejecting anything that is not exactly one. Returns false on trailing
//! junk, so a malformed cgroup file reports "unknown" instead of a plausible-looking number.
bool ParseWholeNumber(const std::string &text, int64_t &result) {
	if (text.empty()) {
		return false;
	}
	errno = 0;
	char *end = nullptr;
	const auto value = std::strtoll(text.c_str(), &end, 10);
	if (errno != 0 || end == text.c_str()) {
		return false;
	}
	// Trailing whitespace is expected (these files end in a newline); anything else is not.
	while (*end != '\0') {
		if (*end != ' ' && *end != '\t' && *end != '\n' && *end != '\r') {
			return false;
		}
		end++;
	}
	result = value;
	return true;
}

//! quota/period, rounded up: a fractional allocation still permits one runnable thread.
idx_t CoresFromQuota(int64_t quota_us, int64_t period_us) {
	if (quota_us <= 0 || period_us <= 0) {
		return 0;
	}
	return static_cast<idx_t>((quota_us + period_us - 1) / period_us);
}

std::string ReadFirstLine(const char *path) {
	std::ifstream stream(path);
	if (!stream.is_open()) {
		return std::string();
	}
	std::string line;
	std::getline(stream, line);
	return line;
}

std::string ReadWholeFile(const char *path) {
	std::ifstream stream(path);
	if (!stream.is_open()) {
		return std::string();
	}
	std::ostringstream buffer;
	buffer << stream.rdbuf();
	return buffer.str();
}

//! Strip one path component, stopping at `root`. Returns false once `dir` is `root`.
bool AscendTo(std::string &dir, const std::string &root) {
	if (dir.size() <= root.size()) {
		return false;
	}
	const auto slash = dir.rfind('/');
	if (slash == std::string::npos || slash < root.size()) {
		return false;
	}
	dir.resize(slash);
	return true;
}

//! Tightest cgroup v2 quota from our own cgroup up to the mount root.
//!
//! Walking up matters because the limits are hierarchical: a parent slice's cap binds us even
//! when our own cgroup is uncapped, and under `cgroupns=host` the pod's cap lives on an ancestor.
idx_t UnifiedQuotaCoreCount(const std::string &relative_path) {
	const std::string root = "/sys/fs/cgroup";
	std::string dir = root + (relative_path == "/" ? std::string() : relative_path);
	idx_t limit = 0;
	do {
		const auto line = ReadFirstLine((dir + "/cpu.max").c_str());
		if (!line.empty()) {
			limit = CombineCoreLimits(limit, ParseCpuMaxV2(line));
		}
	} while (AscendTo(dir, root));
	return limit;
}

//! Tightest cgroup v1 quota from our own cgroup up to the controller mount root.
idx_t LegacyQuotaCoreCount(const std::string &relative_path) {
	const std::string root = "/sys/fs/cgroup/cpu";
	std::string dir = root + (relative_path == "/" ? std::string() : relative_path);
	idx_t limit = 0;
	do {
		const auto quota = ReadFirstLine((dir + "/cpu.cfs_quota_us").c_str());
		const auto period = ReadFirstLine((dir + "/cpu.cfs_period_us").c_str());
		if (!quota.empty() && !period.empty()) {
			limit = CombineCoreLimits(limit, ParseCpuQuotaV1(quota, period));
		}
	} while (AscendTo(dir, root));
	return limit;
}

//! Cores permitted by the CFS bandwidth quota, or 0 if uncapped or undiscoverable.
//!
//! Assumes the conventional mount points; a host that remaps them via /proc/self/mountinfo falls
//! back to the affinity mask alone, which is the pre-existing behaviour.
idx_t QuotaCoreCount() {
	const auto self = ReadWholeFile("/proc/self/cgroup");
	if (self.empty()) {
		return 0;
	}
	const auto unified = ParseUnifiedCgroupPath(self);
	if (!unified.empty()) {
		return UnifiedQuotaCoreCount(unified);
	}
	const auto legacy = ParseLegacyCpuCgroupPath(self);
	if (!legacy.empty()) {
		return LegacyQuotaCoreCount(legacy);
	}
	return 0;
}

//! Cores in the affinity mask, or 0 if undiscoverable.
//!
//! `cpu_set_t` covers CPU_SETSIZE (1024) CPUs; on a host with more, `sched_getaffinity` fails
//! with EINVAL and we report unknown rather than a truncated count.
idx_t AffineCoreCount() {
#ifdef __linux__
	cpu_set_t set;
	CPU_ZERO(&set);
	if (sched_getaffinity(0, sizeof(set), &set) == 0) {
		return static_cast<idx_t>(CPU_COUNT(&set));
	}
#endif
	return 0;
}

} // anonymous namespace

idx_t ParseCpuMaxV2(const std::string &contents) {
	std::istringstream stream(contents);
	std::string quota_token;
	std::string period_token;
	if (!(stream >> quota_token) || !(stream >> period_token)) {
		return 0;
	}
	if (quota_token == "max") {
		return 0;
	}
	int64_t quota_us = 0;
	int64_t period_us = 0;
	if (!ParseWholeNumber(quota_token, quota_us) || !ParseWholeNumber(period_token, period_us)) {
		return 0;
	}
	return CoresFromQuota(quota_us, period_us);
}

idx_t ParseCpuQuotaV1(const std::string &quota_us, const std::string &period_us) {
	int64_t quota = 0;
	int64_t period = 0;
	if (!ParseWholeNumber(quota_us, quota) || !ParseWholeNumber(period_us, period)) {
		return 0;
	}
	// v1 spells "uncapped" as -1; CoresFromQuota rejects it along with any other non-positive.
	return CoresFromQuota(quota, period);
}

std::string ParseUnifiedCgroupPath(const std::string &contents) {
	std::istringstream stream(contents);
	std::string line;
	while (std::getline(stream, line)) {
		// hierarchy-id:controllers:path -- the unified hierarchy has an empty controller field.
		const auto first = line.find(':');
		if (first == std::string::npos) {
			continue;
		}
		const auto second = line.find(':', first + 1);
		if (second == std::string::npos || second != first + 1) {
			continue;
		}
		auto path = line.substr(second + 1);
		if (!path.empty() && path.back() == '\r') {
			path.pop_back();
		}
		if (!path.empty() && path.front() == '/') {
			return path;
		}
	}
	return std::string();
}

std::string ParseLegacyCpuCgroupPath(const std::string &contents) {
	std::istringstream stream(contents);
	std::string line;
	while (std::getline(stream, line)) {
		const auto first = line.find(':');
		if (first == std::string::npos) {
			continue;
		}
		const auto second = line.find(':', first + 1);
		if (second == std::string::npos || second == first + 1) {
			continue;
		}
		// The controller field is a comma-separated list, e.g. "cpu,cpuacct".
		const auto controllers = line.substr(first + 1, second - first - 1);
		bool has_cpu = false;
		size_t begin = 0;
		while (begin <= controllers.size()) {
			auto end = controllers.find(',', begin);
			if (end == std::string::npos) {
				end = controllers.size();
			}
			if (controllers.compare(begin, end - begin, "cpu") == 0) {
				has_cpu = true;
				break;
			}
			begin = end + 1;
		}
		if (!has_cpu) {
			continue;
		}
		auto path = line.substr(second + 1);
		if (!path.empty() && path.back() == '\r') {
			path.pop_back();
		}
		if (!path.empty() && path.front() == '/') {
			return path;
		}
	}
	return std::string();
}

idx_t CombineCoreLimits(idx_t left, idx_t right) {
	if (left == 0) {
		return right;
	}
	if (right == 0) {
		return left;
	}
	return left < right ? left : right;
}

idx_t UsableCoreCount() {
	const auto limit = CombineCoreLimits(AffineCoreCount(), QuotaCoreCount());
	if (limit > 0) {
		return limit;
	}
	const auto visible = static_cast<idx_t>(std::thread::hardware_concurrency());
	return visible > 0 ? visible : 1;
}

} // namespace anofox
} // namespace duckdb
